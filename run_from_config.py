#!/usr/bin/env python3
"""
run_from_config.py

Helper script to run training or inference from YAML config files.

Usage:
  # Train (calls train.py with CLI args built from train.yml)
  python run_from_config.py --mode train --config train.yml

  # Inference / evaluation (runs inference using test.yml)
  python run_from_config.py --mode test --config test.yml

Requirements:
  - pyyaml (`pip install pyyaml`)
  - torch, torchvision, PIL, numpy, tqdm (same as project requirements)
  - This script expects train.py and psinet_denoiser.py to be present in the same project.

Notes:
  - For training mode this script builds a subprocess call to train.py, mapping common fields.
  - For test mode it loads the checkpoint and runs inference (supports optional tiling and TTA).
  - If test.yml does not contain a 'model' section, the default model constructor parameters will be used.
"""

import os
import sys
import argparse
import subprocess
import math
from pathlib import Path
from glob import glob

try:
    import yaml
except Exception:
    print("Missing dependency 'pyyaml'. Install with: pip install pyyaml")
    sys.exit(1)

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms as T

# Import model class
from psinet_denoiser import PSINetRDBUNet

# ---------------------------
# Utilities: CLI -> train.py
# ---------------------------
def build_train_cmd(cfg):
    """Build a command list to call train.py with CLI args from train.yml"""
    cmd = [sys.executable, "train.py"]
    ds = cfg.get("dataset", {})
    model = cfg.get("model", {})
    tr = cfg.get("training", {})
    losses = cfg.get("losses", {})
    scheduler = cfg.get("scheduler", {})

    # dataset
    if "clean_dir" in ds:
        cmd += ["--clean_dir", str(ds["clean_dir"])]
    if ds.get("noisy_dir"):
        cmd += ["--noisy_dir", str(ds["noisy_dir"])]
    if "patch" in ds:
        cmd += ["--patch", str(ds["patch"])]
    if "num_workers" in ds:
        cmd += ["--num_workers", str(ds["num_workers"])]
    if "synth_prob" in ds:
        cmd += ["--synth_prob", str(ds["synth_prob"])]

    # model params
    if "base_channels" in model:
        cmd += ["--base_channels", str(model["base_channels"])]
    if "growth_rate" in model:
        cmd += ["--growth_rate", str(model["growth_rate"])]
    if "rdb_layers" in model:
        cmd += ["--rdb_layers", str(model["rdb_layers"])]
    if "n_rdb_per_scale" in model:
        cmd += ["--n_rdb_per_scale", str(model["n_rdb_per_scale"])]
    if "n_scales" in model:
        cmd += ["--n_scales", str(model["n_scales"])]

    # training
    if "batch_size" in tr:
        cmd += ["--batch_size", str(tr["batch_size"])]
    if "epochs" in tr:
        cmd += ["--epochs", str(tr["epochs"])]
    if "lr" in tr:
        cmd += ["--lr", str(tr["lr"])]
    if "out_dir" in tr:
        cmd += ["--out_dir", str(tr["out_dir"])]
    if "val_split" in ds:
        cmd += ["--val_split", str(ds["val_split"])]
    if "seed" in tr:
        cmd += ["--seed", str(tr["seed"])]

    # losses
    if "lambda_fft" in losses:
        cmd += ["--lambda_fft", str(losses["lambda_fft"])]
    if "lambda_perc" in losses:
        cmd += ["--lambda_perc", str(losses["lambda_perc"])]
    if "lambda_ssim" in losses:
        cmd += ["--lambda_ssim", str(losses["lambda_ssim"])]

    # amp
    amp = tr.get("amp", True)
    if not amp:
        cmd += ["--no_amp"]

    # device/gpu (optional)
    if "gpu" in tr:
        cmd += ["--gpu", str(tr["gpu"])]

    return cmd

# ---------------------------
# Utilities: inference
# ---------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return arr, img

def save_image(img_arr, path):
    arr = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def img_to_tensor(img_arr, device):
    t = torch.from_numpy(img_arr.transpose(2,0,1)).unsqueeze(0).to(device)
    return t.float()

def tensor_to_img(tensor):
    # tensor: 1,C,H,W, on cpu, values in [0,1]
    arr = tensor.squeeze(0).clamp(0,1).cpu().numpy().transpose(1,2,0)
    return arr

def tiles_from_image(img, tile_size, overlap):
    H, W = img.shape[:2]
    stride = tile_size - overlap
    ys = list(range(0, max(H - tile_size + 1, 1), stride))
    xs = list(range(0, max(W - tile_size + 1, 1), stride))
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)
    tiles = []
    coords = []
    for y in ys:
        for x in xs:
            tiles.append(img[y:y+tile_size, x:x+tile_size])
            coords.append((y,x))
    return tiles, coords, (H, W)

def reconstruct_from_tiles(tiles_out, coords, out_shape, tile_size, overlap):
    H, W = out_shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    weight = np.zeros((H, W, 3), dtype=np.float32)
    stride = tile_size - overlap
    for t, (y,x) in zip(tiles_out, coords):
        h, w = t.shape[:2]
        canvas[y:y+h, x:x+w] += t
        weight[y:y+h, x:x+w] += 1.0
    weight[weight == 0] = 1.0
    return canvas / weight

def do_tta_forward(model, inp_tensor, device, use_amp, tta_transforms):
    # inp_tensor: 1,C,H,W
    resolutions = []
    with torch.no_grad():
        for t in tta_transforms:
            if t == "none":
                img = inp_tensor
            elif t == "hflip":
                img = torch.flip(inp_tensor, dims=[3])
            elif t == "vflip":
                img = torch.flip(inp_tensor, dims=[2])
            else:
                img = inp_tensor
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(img.to(device))
            # unflip
            if t == "hflip":
                out = torch.flip(out, dims=[3])
            elif t == "vflip":
                out = torch.flip(out, dims=[2])
            resolutions.append(out)
    # average
    out_avg = torch.stack(resolutions, dim=0).mean(dim=0)
    return out_avg

# ---------------------------
# Inference runner
# ---------------------------
def run_inference(cfg):
    inf = cfg.get("inference", {})
    chk = inf.get("checkpoint")
    if not chk:
        print("test.yml must set inference.checkpoint to a model file.")
        return 1
    checkpoint = Path(chk)
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return 1

    device = inf.get("device", "cuda:0")
    use_amp = inf.get("use_amp", True)
    batch_size = int(inf.get("batch_size", 1))
    patch = int(inf.get("patch", 0))
    tile_cfg = inf.get("tile", {}) or {}
    tile_enabled = tile_cfg.get("enabled", False)
    tile_size = int(tile_cfg.get("tile_size", 512))
    overlap = int(tile_cfg.get("overlap", 32))

    input_path = Path(inf.get("input_path"))
    clean_root = inf.get("clean_path", None)
    output_dir = Path(inf.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    save_visuals = bool(inf.get("save_visuals", True))
    save_individual = bool(inf.get("save_individual", True))
    save_residual = bool(inf.get("save_residual", False))
    clamp_output = bool(inf.get("clamp_output", True))
    convert_to_uint8 = bool(inf.get("convert_to_uint8", True))
    compute_psnr = bool(cfg.get("metrics", {}).get("compute_psnr", False))
    compute_ssim = bool(cfg.get("metrics", {}).get("compute_ssim", False))
    tta_cfg = cfg.get("tta", {}) or {}
    tta_enabled = bool(tta_cfg.get("enabled", False))
    tta_transforms = tta_cfg.get("transforms", ["none"]) if tta_enabled else ["none"]

    # model construction params (optional)
    model_cfg = cfg.get("model", {})
    model_kwargs = {
        "in_channels": model_cfg.get("in_channels", 3),
        "base_channels": model_cfg.get("base_channels", 32),
        "growth_rate": model_cfg.get("growth_rate", 16),
        "rdb_layers": model_cfg.get("rdb_layers", 3),
        "n_rdb_per_scale": model_cfg.get("n_rdb_per_scale", 2),
        "n_scales": model_cfg.get("n_scales", 3),
    }

    # build model and load checkpoint
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = PSINetRDBUNet(**model_kwargs).to(device)
    ck = torch.load(str(checkpoint), map_location=device)
    # support ck being either state dict or dict with 'model_state'
    if isinstance(ck, dict) and 'model_state' in ck:
        state = ck['model_state']
    else:
        state = ck
    try:
        model.load_state_dict(state)
    except Exception as e:
        # try flexible loading
        model.load_state_dict(state, strict=False)
        print("Warning: loaded checkpoint with strict=False:", e)

    model.eval()

    # prepare input list
    inputs = []
    if input_path.is_dir():
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
            inputs.extend(sorted(glob(str(input_path / ext))))
    elif input_path.is_file():
        inputs = [str(input_path)]
    else:
        print("input_path not found:", input_path)
        return 1

    # transforms
    to_tensor = lambda img_arr: img_to_tensor(img_arr, device)
    to_pil = lambda arr: Image.fromarray((arr*255.0).astype(np.uint8))

    for inp_fp in tqdm(inputs, desc="Inference"):
        img_arr, pil = load_image(inp_fp)  # H,W,3
        H, W = img_arr.shape[:2]

        if tile_enabled or (patch > 0 and (H > patch or W > patch)):
            # tile-based inference
            tiles, coords, out_shape = tiles_from_image(img_arr, tile_size, overlap)
            tiles_out = []
            # process tiles in batches
            for i in range(0, len(tiles), batch_size):
                batch_tiles = tiles[i:i+batch_size]
                batch_tensors = torch.cat([img_to_tensor(t, device) for t in batch_tiles], dim=0)
                with torch.no_grad():
                    if tta_enabled:
                        # perform per-sample TTA: here simplified to process each sample
                        outs = []
                        for b in range(batch_tensors.shape[0]):
                            out = do_tta_forward(model, batch_tensors[b:b+1], device, use_amp, tta_transforms)
                            outs.append(out)
                        out_batch = torch.cat(outs, dim=0)
                    else:
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            out_batch = model(batch_tensors)
                out_batch = out_batch.clamp(0,1).cpu().numpy().transpose(0,2,3,1)
                tiles_out.extend([o for o in out_batch])
            recon = reconstruct_from_tiles(tiles_out, coords, out_shape, tile_size, overlap)
            denoised = recon
        else:
            # full-image inference (single pass)
            inp_t = img_to_tensor(img_arr, device)
            with torch.no_grad():
                if tta_enabled:
                    out_t = do_tta_forward(model, inp_t, device, use_amp, tta_transforms)
                else:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out_t = model(inp_t)
            denoised = tensor_to_img(out_t)

        if clamp_output:
            denoised = np.clip(denoised, 0.0, 1.0)

        # save outputs
        fname = Path(inp_fp).stem
        if save_individual:
            out_path = output_dir / f"{fname}_denoised.png"
            save_image(denoised, out_path)

        if save_visuals:
            # save composite noisy | denoised | clean (if available)
            comps = []
            max_vis_h = min(512, H)
            # Using full images; if very large you may want to resize for visuals
            noisy_vis = img_arr
            denoised_vis = denoised
            if clean_root:
                clean_fp = Path(clean_root) / Path(inp_fp).name
                if clean_fp.exists():
                    clean_arr, _ = load_image(clean_fp)
                    comp = np.concatenate([noisy_vis, denoised_vis, clean_arr], axis=1)
                else:
                    comp = np.concatenate([noisy_vis, denoised_vis], axis=1)
            else:
                comp = np.concatenate([noisy_vis, denoised_vis], axis=1)
            viz_path = output_dir / f"{fname}_viz.png"
            save_image(comp, viz_path)

        # optional residual
        if save_residual:
            if save_individual:
                # infer residual = denoised - noisy
                noisy = img_arr
                residual = denoised - noisy
                res_path = output_dir / f"{fname}_residual.png"
                # normalize residual for visualization (centered)
                vmax = np.percentile(np.abs(residual), 99)
                if vmax <= 0: vmax = 1e-6
                vis = (residual / (2*vmax) + 0.5)
                save_image(vis, res_path)

        # metrics if clean exists
        if compute_psnr or compute_ssim:
            if clean_root:
                clean_fp = Path(clean_root) / Path(inp_fp).name
                if clean_fp.exists():
                    clean_arr, _ = load_image(clean_fp)
                    # compute PSNR
                    mse = np.mean((denoised - clean_arr) ** 2)
                    psnr_val = 10 * math.log10(1.0 / (mse + 1e-12))
                    print(f"{fname} PSNR: {psnr_val:.3f} dB")
                    # SSIM compute is not included here; for quickness user can use skimage.metrics.structural_similarity
                else:
                    print(f"clean not found for {inp_fp}")
            else:
                print("No clean_root supplied; skipping PSNR/SSIM")

    print("Inference finished. Outputs saved to:", output_dir)
    return 0

# ---------------------------
# Entry point
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","test"], default="train", help="Mode: train or test")
    parser.add_argument("--config", type=str, default="train.yml", help="Path to YAML config (train.yml or test.yml)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print("Config file not found:", cfg_path)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "train":
        cmd = build_train_cmd(cfg)
        print("Running training command:")
        print(" ".join(cmd))
        # spawn subprocess and stream output
        proc = subprocess.Popen(cmd)
        ret = proc.wait()
        sys.exit(ret)
    else:
        ret = run_inference(cfg)
        sys.exit(ret)

if __name__ == "__main__":
    main()
