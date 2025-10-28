# train.py (updated to use FFT + VGG perceptual + SSIM losses with flags)
import os
import argparse
import time
import math
import gc
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as vutils

from psinet_denoiser import PSINetRDBUNet as Model
from stripe_dataset import StripeDataset
import losses

# PSNR helper
def psnr(pred, target, data_range=1.0, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=[1,2,3])
    psnr_vals = 10.0 * torch.log10((data_range ** 2) / (mse + eps))
    return psnr_vals.mean().item()

def test_batch_sizes(model, input_size=(3,128,128), device='cuda'):
    model = model.to(device)
    model.eval()
    print("Running batch size memory test (increasing -> decreasing attempt)...")
    for b in [32,24,16,12,8,6,4,2,1]:
        try:
            torch.cuda.empty_cache(); gc.collect()
            x = torch.randn(b, *input_size, device=device)
            with torch.no_grad():
                torch.cuda.reset_peak_memory_stats(device)
                _ = model(x)
            peak = torch.cuda.max_memory_allocated(device)/1024**3
            print(f"batch {b}: peak {peak:.2f} GB")
            del x; torch.cuda.empty_cache()
        except Exception as e:
            print(f"batch {b}: OOM or error ({e})")

def train_epoch(model, loader, optimizer, device, loss_params, scaler, use_amp=True, vgg_extractor=None):
    model.train()
    running_loss = 0.0
    for i, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device)
        clean = clean.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            out = model(noisy)
            l1 = nn.functional.l1_loss(out, clean)
            loss = l1
            if loss_params['lambda_fft'] > 0:
                fft_l = losses.fft_mag_loss(out, clean)
                loss = loss + loss_params['lambda_fft'] * fft_l
            if loss_params['lambda_perc'] > 0 and vgg_extractor is not None:
                perc = losses.vgg_perceptual_loss(vgg_extractor, out, clean)
                loss = loss + loss_params['lambda_perc'] * perc
            if loss_params['lambda_ssim'] > 0:
                ssim_val = losses.ssim_torch(out.clamp(0,1), clean.clamp(0,1))
                ssim_loss = (1.0 - ssim_val)
                loss = loss + loss_params['lambda_ssim'] * ssim_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)

def val_epoch(model, loader, device, loss_params, vgg_extractor=None, vis_dir=None, epoch=0, max_vis=4):
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
            l1 = nn.functional.l1_loss(out, clean)
            loss = l1
            if loss_params['lambda_fft'] > 0:
                loss = loss + loss_params['lambda_fft'] * losses.fft_mag_loss(out, clean)
            if loss_params['lambda_perc'] > 0 and vgg_extractor is not None:
                loss = loss + loss_params['lambda_perc'] * losses.vgg_perceptual_loss(vgg_extractor, out, clean)
            if loss_params['lambda_ssim'] > 0:
                ssim_val = losses.ssim_torch(out.clamp(0,1), clean.clamp(0,1))
                loss = loss + loss_params['lambda_ssim'] * (1.0 - ssim_val)
                total_ssim += ssim_val.item()
            running += loss.item()
            total_psnr += psnr(out.clamp(0,1), clean.clamp(0,1))
            n += 1
            if vis_dir and i == 0:
                os.makedirs(vis_dir, exist_ok=True)
                comp = torch.cat([noisy[:max_vis], out.clamp(0,1)[:max_vis], clean[:max_vis]], dim=0)
                vutils.save_image(comp, os.path.join(vis_dir, f"viz_epoch{epoch}.png"), nrow=max_vis, normalize=False)
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
    # loss weights
    parser.add_argument("--lambda_fft", type=float, default=0.2, help="weight for FFT magnitude loss")
    parser.add_argument("--lambda_perc", type=float, default=0.01, help="weight for VGG perceptual loss")
    parser.add_argument("--lambda_ssim", type=float, default=0.1, help="weight for SSIM-based loss (1-SSIM)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # prepare datasets
    train_ds = StripeDataset(args.clean_dir, noisy_dir=args.noisy_dir, patch_size=args.patch, augment=True, synth_prob=args.synth_prob, synth_params_generator=None)
    val_ds = StripeDataset(args.clean_dir, noisy_dir=args.noisy_dir, patch_size=args.patch, augment=False, synth_prob=args.synth_prob, synth_params_generator=None)

    # simple deterministic split
    n_total = len(train_ds.clean_files)
    val_n = max(1, int(n_total * args.val_split)) if n_total > 1 else 0
    train_files = train_ds.clean_files[: n_total - val_n]
    val_files = train_ds.clean_files[n_total - val_n : ]
    train_ds.clean_files = train_files
    val_ds.clean_files = val_files
    # rebuild noisy_map if noisy_dir provided
    if args.noisy_dir:
        noisy_dir = Path(args.noisy_dir)
        noisy_files = []
        exts = ("png","jpg","jpeg","bmp","tif","tiff")
        for ext in exts:
            noisy_files += list(noisy_dir.rglob(f"*.{ext}"))
        noisy_by_name = {Path(p).name: p for p in noisy_files}
        train_ds.noisy_map = {c: noisy_by_name.get(Path(c).name, None) for c in train_ds.clean_files}
        val_ds.noisy_map = {c: noisy_by_name.get(Path(c).name, None) for c in val_ds.clean_files}

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.num_workers//2), pin_memory=True)

    # model
    model = Model(in_channels=3, base_channels=args.base_channels, growth_rate=args.growth_rate,
                  rdb_layers=args.rdb_layers, n_rdb_per_scale=args.n_rdb_per_scale, n_scales=args.n_scales).to(device)

    # test batch option
    if args.test_batch:
        test_batch_sizes(model, input_size=(3,args.patch,args.patch), device=device)
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
    scaler = GradScaler(enabled=(not args.no_amp))

    # prepare VGG feature extractor if perceptual enabled
    vgg_extractor = None
    if args.lambda_perc > 0:
        vgg_extractor = losses.VGGFeatureExtractor(device=device)
        vgg_extractor = vgg_extractor.to(device)

    loss_params = {
        'lambda_fft': args.lambda_fft,
        'lambda_perc': args.lambda_perc,
        'lambda_ssim': args.lambda_ssim
    }

    best_val = 1e9
    best_psnr = -1.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_params, scaler, use_amp=not args.no_amp, vgg_extractor=vgg_extractor)
        val_loss, val_psnr, val_ssim = val_epoch(model, val_loader, device, loss_params, vgg_extractor=vgg_extractor,
                                                 vis_dir=os.path.join(args.out_dir, "vis"), epoch=epoch)
        scheduler.step(val_loss)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} - train_loss {train_loss:.6f} val_loss {val_loss:.6f} PSNR {val_psnr:.3f} SSIM {val_ssim:.4f} time {(t1-t0):.1f}s")
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict()
        }
        torch.save(ckpt, os.path.join(args.out_dir, f"psinet_epoch{epoch}.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.out_dir, "psinet_best_by_loss.pth"))
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(ckpt, os.path.join(args.out_dir, "psinet_best_by_psnr.pth"))

if __name__ == "__main__":
    main()