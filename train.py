#!/usr/bin/env python3
# train.py (modified)
# - supports res_high / edge / gate supervision
# - gradient clipping, NaN detection, and debug save on NaN
# - saves gate/res_low/res_high visuals during validation
# - args added: lambda_highfreq, lambda_edge, lambda_gate, clip_grad

import os
import argparse
import time
import math
import gc
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as vutils
import torch.nn.functional as F

from psinet_denoiser import PSINetRDBUNet as Model
from stripe_dataset import StripeDataset
import losses

# Helpers
def psnr(pred, target, data_range=1.0, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=[1,2,3])
    psnr_vals = 10.0 * torch.log10((data_range ** 2) / (mse + eps))
    return psnr_vals.mean().item()

def tv_loss(x):
    dx = x[:,:,1:,:] - x[:,:,:-1,:]
    dy = x[:,:,:,1:] - x[:,:,:, :-1]
    return (dx.abs().mean() + dy.abs().mean())

def gaussian_kernel2d(sigma, device):
    radius = int(math.ceil(3 * sigma))
    size = radius * 2 + 1
    coords = torch.arange(-radius, radius+1, dtype=torch.float32, device=device)
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel = g[:,None] * g[None,:]
    return kernel, size

def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    device = x.device
    kernel2d, size = gaussian_kernel2d(sigma, device)
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)
    C = x.shape[1]
    kernel = kernel2d.repeat(C,1,1,1)
    padding = size // 2
    return F.conv2d(x, kernel, groups=C, padding=padding)

# Checkpoint utils (unchanged)
def _find_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "state_dicts", "net", "model"):
            if key in ckpt:
                return ckpt[key]
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return None

def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def load_checkpoint_if_requested(path, model, optimizer=None, scaler=None, scheduler=None, device='cpu'):
    if not path:
        return 1, None, None
    ck = torch.load(str(path), map_location=device)
    state = _find_state_dict(ck) or ck
    sd = _strip_module_prefix(state) if isinstance(state, dict) else state
    try:
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print("Warning loading checkpoint state_dict with strict=False:", e)
        model.load_state_dict(sd, strict=False)
    start_epoch = 1
    best_val = None
    best_psnr = None
    if isinstance(ck, dict):
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("best_val", None)
        best_psnr = ck.get("best_psnr", None)
        if optimizer is not None and "opt_state" in ck:
            try:
                optimizer.load_state_dict(ck["opt_state"])
                _move_optimizer_state_to_device(optimizer, device)
            except Exception as e:
                print("Warning: failed to load optimizer state:", e)
        if scaler is not None and "scaler_state" in ck:
            try:
                scaler.load_state_dict(ck["scaler_state"])
            except Exception:
                print("Warning: failed to load scaler state.")
        if scheduler is not None and "scheduler_state" in ck:
            try:
                scheduler.load_state_dict(ck["scheduler_state"])
            except Exception:
                print("Warning: failed to load scheduler state.")
    print(f"Loaded checkpoint '{path}' (resuming from epoch {start_epoch})")
    return start_epoch, best_val, best_psnr

# Training / validation loops with highfreq/edge/gate supervision
def train_epoch(model, loader, optimizer, device, loss_params, scaler, use_amp=True, vgg_extractor=None, args=None):
    model.train()
    running_loss = 0.0
    bce_loss = nn.BCEWithLogitsLoss()
    for i, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            out = model(noisy)
            if isinstance(out, dict):
                denoised = out.get('denoised')
                res_low = out.get('res_low', None)
                res_high = out.get('res_high', None)
                gate = out.get('gate', None)
            else:
                denoised = out
                res_low = res_high = gate = None

            # prepare clean gray
            if clean.shape[1] == 3:
                clean_gray = 0.2989*clean[:,0:1,:,:] + 0.5870*clean[:,1:2,:,:] + 0.1140*clean[:,2:3,:,:]
            else:
                clean_gray = clean

            # region mask
            mask = None
            if args is not None and args.region_weight != 1.0:
                H = clean_gray.shape[2]
                mask = torch.ones((1,1,H,1), device=clean_gray.device)
                mask[:,:,H//2:,:] = args.region_weight

            # base loss
            if args is not None and args.binary_output:
                base_loss = bce_loss(denoised, clean_gray)
            else:
                pred_sig = torch.sigmoid(denoised)
                if mask is not None:
                    base_loss = (torch.abs(pred_sig - clean_gray) * mask).mean()
                else:
                    base_loss = F.l1_loss(pred_sig, clean_gray)
            loss = base_loss

            # low-frequency loss (if enabled)
            if args is not None and args.lambda_lowfreq > 0:
                pred_lp = gaussian_blur(torch.sigmoid(denoised), args.lowfreq_blur_sigma)
                gt_lp = gaussian_blur(clean_gray, args.lowfreq_blur_sigma)
                if mask is not None:
                    lowfreq_loss = (torch.abs(pred_lp - gt_lp) * mask).mean()
                else:
                    lowfreq_loss = F.l1_loss(pred_lp, gt_lp)
                loss = loss + args.lambda_lowfreq * lowfreq_loss

            # TV on res_low
            if args is not None and args.lambda_tv > 0 and res_low is not None:
                tv = tv_loss(res_low)
                loss = loss + args.lambda_tv * tv

            # HIGH-FREQUENCY residual supervision
            if args is not None and getattr(args, 'lambda_highfreq', 0.0) > 0 and res_high is not None:
                gt_lp = gaussian_blur(clean_gray, args.lowfreq_blur_sigma)
                gt_hf = (clean_gray - gt_lp).detach()
                hf_loss = F.l1_loss(res_high, gt_hf)
                loss = loss + args.lambda_highfreq * hf_loss

            # Sobel / edge loss
            if args is not None and getattr(args, 'lambda_edge', 0.0) > 0:
                pred_sig = torch.sigmoid(denoised)
                sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=pred_sig.device).view(1,1,3,3)
                sobel_y = sobel_x.transpose(2,3)
                gx = F.conv2d(pred_sig, sobel_x, padding=1)
                gy = F.conv2d(pred_sig, sobel_y, padding=1)
                gxt = F.conv2d(clean_gray, sobel_x, padding=1)
                gyt = F.conv2d(clean_gray, sobel_y, padding=1)
                grad_pred = torch.sqrt(gx*gx + gy*gy + 1e-6)
                grad_gt = torch.sqrt(gxt*gxt + gyt*gyt + 1e-6)
                edge_loss = F.l1_loss(grad_pred, grad_gt)
                loss = loss + args.lambda_edge * edge_loss

            # optional gate supervision (weak)
            if args is not None and getattr(args, 'lambda_gate', 0.0) > 0 and gate is not None:
                with torch.no_grad():
                    gt_lp = gaussian_blur(clean_gray, args.lowfreq_blur_sigma)
                    gt_hf = (clean_gray - gt_lp).abs()
                    edge_mask = (gt_hf > (gt_hf.mean() * 0.25)).float()  # heuristic mask
                gate_loss = F.mse_loss(gate, edge_mask)
                loss = loss + args.lambda_gate * gate_loss

            # existing spectral/perceptual/ssim
            if loss_params['lambda_fft'] > 0:
                fft_l = losses.fft_mag_loss(torch.sigmoid(denoised), clean_gray)
                loss = loss + loss_params['lambda_fft'] * fft_l
            if loss_params['lambda_perc'] > 0 and vgg_extractor is not None:
                perc = losses.vgg_perceptual_loss(vgg_extractor, torch.sigmoid(denoised), clean_gray)
                loss = loss + loss_params['lambda_perc'] * perc
            if loss_params['lambda_ssim'] > 0:
                ssim_val = losses.ssim_torch(torch.sigmoid(denoised).clamp(0,1), clean_gray.clamp(0,1))
                ssim_loss = (1.0 - ssim_val)
                loss = loss + loss_params['lambda_ssim'] * ssim_loss

        # NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            debug_dir = os.path.join(getattr(args, 'out_dir', 'checkpoints'), "debug_nan")
            os.makedirs(debug_dir, exist_ok=True)
            try:
                pred = torch.sigmoid(denoised).detach().cpu()
            except Exception:
                pred = None
            noisy_cpu = noisy.detach().cpu()
            clean_cpu = clean.detach().cpu()
            if pred is not None:
                vutils.save_image(pred[:4].repeat(1,3,1,1), os.path.join(debug_dir, f"pred_nan_iter{i}.png"), nrow=4, normalize=True)
            vutils.save_image(noisy_cpu[:4], os.path.join(debug_dir, f"noisy_nan_iter{i}.png"), nrow=4, normalize=True)
            vutils.save_image(clean_cpu[:4].repeat(1,3,1,1), os.path.join(debug_dir, f"clean_nan_iter{i}.png"), nrow=4, normalize=True)
            raise RuntimeError("NaN encountered during training; debug images saved to " + debug_dir)

        # backward + unscale + grad clip + step
        scaler.scale(loss).backward()
        # unscale before clip
        try:
            scaler.unscale_(optimizer)
        except Exception:
            pass
        max_norm = getattr(args, 'clip_grad', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))

def val_epoch(model, loader, device, loss_params, vgg_extractor=None, vis_dir=None, epoch=0, max_vis=4, args=None):
    model.eval()
    running = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            out = model(noisy)
            if isinstance(out, dict):
                denoised = out.get('denoised')
                res_low = out.get('res_low', None)
                res_high = out.get('res_high', None)
                gate = out.get('gate', None)
            else:
                denoised = out
                res_low = res_high = gate = None

            # ensure clean_gray
            if clean.shape[1] == 3:
                clean_gray = 0.2989*clean[:,0:1,:,:] + 0.5870*clean[:,1:2,:,:] + 0.1140*clean[:,2:3,:,:]
            else:
                clean_gray = clean

            pred = torch.sigmoid(denoised)
            l1 = F.l1_loss(pred, clean_gray)
            loss = l1
            if loss_params['lambda_fft'] > 0:
                loss = loss + loss_params['lambda_fft'] * losses.fft_mag_loss(pred, clean_gray)
            if loss_params['lambda_perc'] > 0 and vgg_extractor is not None:
                loss = loss + loss_params['lambda_perc'] * losses.vgg_perceptual_loss(vgg_extractor, pred, clean_gray)
            if loss_params['lambda_ssim'] > 0:
                ssim_val = losses.ssim_torch(pred.clamp(0,1), clean_gray.clamp(0,1))
                loss = loss + loss_params['lambda_ssim'] * (1.0 - ssim_val)
                total_ssim += ssim_val.item()
            running += loss.item()
            total_psnr += psnr(pred.clamp(0,1), clean_gray.clamp(0,1))
            n += 1

            if vis_dir and i == 0:
                os.makedirs(vis_dir, exist_ok=True)

                # prepare noisy_vis, pred_vis, clean_vis all as 3-channel tensors on CPU
                def to_3ch(t):
                    # t: tensor (B,C,H,W) on device or cpu
                    t_cpu = t[:max_vis].cpu()
                    if t_cpu.ndim == 4 and t_cpu.shape[1] == 1:
                        return t_cpu.repeat(1,3,1,1)
                    elif t_cpu.ndim == 4 and t_cpu.shape[1] == 3:
                        return t_cpu
                    else:
                        # fallback: try to expand last dim if needed
                        return t_cpu

                noisy_vis = to_3ch(noisy)
                pred_vis = to_3ch(pred)
                clean_vis = to_3ch(clean)

                comp = torch.cat([noisy_vis, pred_vis, clean_vis], dim=0)
                vutils.save_image(comp, os.path.join(vis_dir, f"viz_epoch{epoch}.png"), nrow=max_vis, normalize=False)

                # save intermediate maps if available (also ensure 3 channels)
                if res_low is not None:
                    rl = to_3ch(res_low)
                    vutils.save_image(rl, os.path.join(vis_dir, f"res_low_epoch{epoch}.png"), nrow=max_vis, normalize=False)
                if res_high is not None:
                    rh = to_3ch(res_high)
                    vutils.save_image(rh, os.path.join(vis_dir, f"res_high_epoch{epoch}.png"), nrow=max_vis, normalize=False)
                if gate is not None:
                    g = to_3ch(gate)
                    vutils.save_image(g, os.path.join(vis_dir, f"gate_epoch{epoch}.png"), nrow=max_vis, normalize=False)

    avg_loss = running / max(1, n)
    mean_psnr = total_psnr / max(1, n)
    mean_ssim = total_ssim / max(1, n) if loss_params['lambda_ssim'] > 0 else 0.0
    return avg_loss, mean_psnr, mean_ssim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--noisy_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--growth_rate", type=int, default=16)
    parser.add_argument("--rdb_layers", type=int, default=3)
    parser.add_argument("--n_rdb_per_scale", type=int, default=2)
    parser.add_argument("--n_scales", type=int, default=3)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--test_batch", action="store_true")
    parser.add_argument("--synth_prob", type=float, default=0.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    # loss weights (existing)
    parser.add_argument("--lambda_fft", type=float, default=0.0)
    parser.add_argument("--lambda_perc", type=float, default=0.0)
    parser.add_argument("--lambda_ssim", type=float, default=0.0)
    # new losses / options
    parser.add_argument("--binary_output", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lambda_tv", type=float, default=0.0)
    parser.add_argument("--lambda_lowfreq", type=float, default=0.0)
    parser.add_argument("--lowfreq_blur_sigma", type=float, default=2.0)
    parser.add_argument("--region_weight", type=float, default=1.0)
    parser.add_argument("--lambda_highfreq", type=float, default=0.0)
    parser.add_argument("--lambda_edge", type=float, default=0.0)
    parser.add_argument("--lambda_gate", type=float, default=0.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}; cuDNN deterministic=True")
    else:
        torch.backends.cudnn.benchmark = True

    # Use StripeDataset with force_gray / binarize if desired. If you want to pass those flags
    # through YAML/run_from_config, ensure run_from_config forwards them.
    train_ds = StripeDataset(args.clean_dir, noisy_dir=args.noisy_dir, patch_size=args.patch, augment=True, synth_prob=args.synth_prob, synth_params_generator=None, force_gray=True, binarize_threshold=0.5)
    val_ds = StripeDataset(args.clean_dir, noisy_dir=args.noisy_dir, patch_size=args.patch, augment=False, synth_prob=args.synth_prob, synth_params_generator=None, force_gray=True, binarize_threshold=0.5)

    n_total = len(train_ds.clean_files)
    val_n = max(1, int(n_total * args.val_split)) if n_total > 1 else 0
    train_files = train_ds.clean_files[: n_total - val_n]
    val_files = train_ds.clean_files[n_total - val_n : ]
    train_ds.clean_files = train_files
    val_ds.clean_files = val_files

    if args.noisy_dir:
        noisy_dir = Path(args.noisy_dir)
        noisy_files = []
        exts = ("png","jpg","jpeg","bmp","tif","tiff")
        for ext in exts:
            noisy_files += list(noisy_dir.rglob(f"*.{ext}"))
        noisy_by_name = {Path(p).name: p for p in noisy_files}
        train_ds.noisy_map = {c: noisy_by_name.get(Path(c).name, None) for c in train_ds.clean_files}
        val_ds.noisy_map = {c: noisy_by_name.get(Path(c).name, None) for c in val_ds.clean_files}

    print(f"Dataset split: total={n_total}, train={len(train_ds.clean_files)}, val={len(val_ds.clean_files)}")

    pin_memory = use_cuda
    persistent = True if args.num_workers > 0 else False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(1,args.num_workers//2), pin_memory=pin_memory, persistent_workers=max(1,persistent))

    # model
    model = Model(in_channels=3, base_channels=args.base_channels, growth_rate=args.growth_rate,
                  rdb_layers=args.rdb_layers, n_rdb_per_scale=args.n_rdb_per_scale, n_scales=args.n_scales).to(device)

    if args.test_batch:
        model.eval()
        for b in [32,16,8,4,2,1]:
            try:
                torch.cuda.empty_cache(); gc.collect()
                x = torch.randn(b,3,args.patch,args.patch, device=device)
                with torch.no_grad(): _ = model(x)
                print("batch", b, "ok")
            except Exception as e:
                print("batch", b, "failed:", e)
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

    scaler = GradScaler(enabled=(not args.no_amp) and use_cuda)

    vgg_extractor = None
    if args.lambda_perc > 0:
        vgg_extractor = losses.VGGFeatureExtractor(device=device).to(device)

    loss_params = {'lambda_fft': args.lambda_fft, 'lambda_perc': args.lambda_perc, 'lambda_ssim': args.lambda_ssim}

    start_epoch = 1
    best_val = 1e9
    best_psnr = -1.0
    if args.resume:
        try:
            se, bv, bp = load_checkpoint_if_requested(args.resume, model, optimizer=optimizer, scaler=scaler, scheduler=scheduler, device=device)
            start_epoch = se if se is not None else 1
            if bv is not None: best_val = bv
            if bp is not None: best_psnr = bp
        except Exception as e:
            print("Warning: failed to load resume checkpoint:", e)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs+1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_params, scaler, use_amp=(use_cuda and not args.no_amp), vgg_extractor=vgg_extractor, args=args)
        val_loss, val_psnr, val_ssim = val_epoch(model, val_loader, device, loss_params, vgg_extractor=vgg_extractor, vis_dir=os.path.join(args.out_dir,"vis"), epoch=epoch, max_vis=min(4, args.batch_size), args=args)
        scheduler.step(val_loss)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} - train_loss {train_loss:.6f} val_loss {val_loss:.6f} PSNR {val_psnr:.3f} SSIM {val_ssim:.4f} time {(t1-t0):.1f}s")
        ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': optimizer.state_dict(), 'scaler_state': scaler.state_dict() if hasattr(scaler,'state_dict') else {}, 'best_val': best_val, 'best_psnr': best_psnr}
        torch.save(ckpt, os.path.join(args.out_dir, f"psinet_epoch{epoch}.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.out_dir, "psinet_best_by_loss.pth"))
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(ckpt, os.path.join(args.out_dir, "psinet_best_by_psnr.pth"))

if __name__ == "__main__":
    main()
