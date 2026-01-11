# run_from_config.py (updated build_train_cmd to forward new args)
# Adds mapping for new losses and extras so YAML -> CLI works for new flags.

#!/usr/bin/env python3
"""
run_from_config.py
Updated to forward new train.yml fields (lambda_highfreq, lambda_edge, lambda_gate, lambda_lowfreq, lambda_tv, region_weight, binary_output, clip_grad)
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
from glob import glob

try:
    import yaml
except Exception:
    print("Missing dependency 'pyyaml'. Install with: pip install pyyaml")
    sys.exit(1)

def build_train_cmd(cfg):
    cmd = [sys.executable, "train.py"]
    ds = cfg.get("dataset", {})
    model = cfg.get("model", {})
    tr = cfg.get("training", {})
    losses = cfg.get("losses", {})
    extra = cfg.get("extra", {})

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
    if "seed" in tr and tr["seed"] is not None:
        cmd += ["--seed", str(tr["seed"])]

    # losses
    if "lambda_fft" in losses:
        cmd += ["--lambda_fft", str(losses["lambda_fft"])]
    if "lambda_perc" in losses:
        cmd += ["--lambda_perc", str(losses["lambda_perc"])]
    if "lambda_ssim" in losses:
        cmd += ["--lambda_ssim", str(losses["lambda_ssim"])]
    if "lambda_lowfreq" in losses:
        cmd += ["--lambda_lowfreq", str(losses["lambda_lowfreq"])]
    if "lambda_tv" in losses:
        cmd += ["--lambda_tv", str(losses["lambda_tv"])]
    if "lambda_highfreq" in losses:
        cmd += ["--lambda_highfreq", str(losses["lambda_highfreq"])]
    if "lambda_edge" in losses:
        cmd += ["--lambda_edge", str(losses["lambda_edge"])]
    if "lambda_gate" in losses:
        cmd += ["--lambda_gate", str(losses["lambda_gate"])]

    # extra
    if extra.get("binary_output", False):
        cmd += ["--binary_output"]
    if "region_weight" in extra:
        cmd += ["--region_weight", str(extra["region_weight"])]
    if "lowfreq_blur_sigma" in extra:
        cmd += ["--lowfreq_blur_sigma", str(extra["lowfreq_blur_sigma"])]
    if "clip_grad" in extra:
        cmd += ["--clip_grad", str(extra["clip_grad"])]

    # amp
    amp = tr.get("amp", True)
    if not amp:
        cmd += ["--no_amp"]

    # device/gpu (optional)
    if "gpu" in tr:
        cmd += ["--gpu", str(tr["gpu"])]

    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","test"], default="train")
    parser.add_argument("--config", type=str, default="train.yml")
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
        proc = subprocess.Popen(cmd)
        ret = proc.wait()
        sys.exit(ret)
    else:
        print("Test mode not changed by this wrapper")
        sys.exit(0)

if __name__ == "__main__":
    main()
