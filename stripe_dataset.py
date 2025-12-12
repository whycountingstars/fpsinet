# stripe_dataset.py
# Stripe dataset with paired transforms (fixes misalignment between clean and noisy)
# Supports:
#  - paired random crop / flips so clean and noisy remain aligned
#  - optional force_gray and binarize_threshold for clean targets
#  - synthetic stripe generation when noisy not provided
#
# Usage remains the same, but now when noisy_dir is provided, transforms are applied identically
# to both clean and noisy images to preserve alignment.

from pathlib import Path
import random
import math
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ----------------------------
# Realistic stripe synthesis (same as original)
# ----------------------------
def _smooth_noise(H, W, scale=64, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    n = rng.randn(H, W).astype(np.float32)
    k = max(1, int(scale))
    out = cv2.GaussianBlur(n, (0, 0), sigmaX=k, sigmaY=k, borderType=cv2.BORDER_REFLECT)
    out = out - out.mean()
    std = out.std()
    if std > 0:
        out = out / (std + 1e-12)
    return out

def add_realistic_stripe_noise(img,
                               layers=None,
                               amplitude_map_scale=160,
                               jitter_scale=8.0,
                               multiplicative_prob=0.5,
                               add_gauss=0.005,
                               jpeg_quality_range=(70, 95),
                               seed=None):
    H, W = img.shape[:2]
    rng = np.random.RandomState(seed) if seed is not None else np.random

    if layers is None:
        layers = [
            {"orientation": -30 + rng.uniform(-5, 5), "wavelength": rng.uniform(100, 160), "amplitude": rng.uniform(0.12, 0.35), "thickness": rng.uniform(2, 5), "mode": "add"},
            {"orientation": 60 + rng.uniform(-8, 8), "wavelength": rng.uniform(8, 24), "amplitude": rng.uniform(0.03, 0.12), "thickness": rng.uniform(1, 2), "mode": "mult"},
            {"orientation": rng.uniform(-90, 90), "wavelength": rng.uniform(6, 12), "amplitude": rng.uniform(0.015, 0.06), "thickness": 1, "mode": "add"}
        ]

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    amp_map = _smooth_noise(H, W, scale=amplitude_map_scale, seed=(seed + 1) if seed else None)
    amp_map = (amp_map - amp_map.min()) / (amp_map.max() - amp_map.min() + 1e-12)
    amp_map = 0.6 + amp_map * 1.4

    dx = cv2.GaussianBlur((rng.randn(H, W) * jitter_scale).astype(np.float32), (0, 0), sigmaX=max(1, jitter_scale/2), sigmaY=max(1, jitter_scale/2))
    dy = cv2.GaussianBlur((rng.randn(H, W) * jitter_scale).astype(np.float32), (0, 0), sigmaX=max(1, jitter_scale/2), sigmaY=max(1, jitter_scale/2))

    total_noise = np.zeros((H, W), dtype=np.float32)

    for layer in layers:
        orientation = float(layer.get("orientation", 0.0))
        wavelength = float(layer.get("wavelength", 20.0))
        base_amp = float(layer.get("amplitude", 0.08))
        thickness = float(layer.get("thickness", 1.0))
        mode = layer.get("mode", None)
        if mode is None:
            mode = "mult" if rng.rand() < multiplicative_prob else "add"

        theta = np.deg2rad(orientation)
        cos_t = math.cos(theta); sin_t = math.sin(theta)
        coord = (xs + dx) * cos_t + (ys + dy) * sin_t

        freq = 2.0 * math.pi / max(1.0, wavelength)
        stripe = np.sin(coord * freq + (rng.rand() * 2 * math.pi))

        k = max(0.5, 8.0 / max(1.0, thickness))
        pulse = 0.5 * (1.0 + np.tanh(k * stripe))

        blur_k = int(max(1, round(thickness * 2)))
        if blur_k % 2 == 0:
            blur_k += 1
        if blur_k > 1:
            pulse = cv2.GaussianBlur(pulse, (blur_k, blur_k), sigmaX=blur_k/2, sigmaY=blur_k/2)

        local_amp = base_amp * amp_map * (0.6 + 0.8 * rng.rand())
        layer_pattern = pulse * local_amp

        if rng.rand() < 0.45:
            fine = _smooth_noise(H, W, scale=max(2, wavelength / 8.0), seed=None)
            layer_pattern = layer_pattern * (1.0 + 0.12 * fine)

        total_noise += layer_pattern

    mult_weight_map = np.clip(total_noise * 3.0, 0.0, 1.0)

    noisy = img.copy().astype(np.float32)
    for c in range(noisy.shape[2]):
        noisy[..., c] = noisy[..., c] * (1.0 + total_noise * mult_weight_map)
        noisy[..., c] = noisy[..., c] + total_noise * (1.0 - mult_weight_map)

    if rng.rand() < 0.85:
        col_bias = _smooth_noise(1, W, scale=max(1, W / 8.0), seed=None).reshape(W)
        col_bias = (col_bias - col_bias.min()) / (col_bias.max() - col_bias.min() + 1e-12)
        col_bias = (col_bias - 0.5) * 0.03 * (0.5 + rng.rand())
        noisy = noisy + col_bias[None, :, None]

    if rng.rand() < 0.5:
        shift = (rng.randn(3) * 0.01).astype(np.float32)
        noisy = noisy + shift[None, None, :]

    if add_gauss and add_gauss > 0:
        noisy = noisy + np.random.randn(*noisy.shape).astype(np.float32) * add_gauss

    noisy = np.clip(noisy, 0.0, 1.0)

    if jpeg_quality_range is not None:
        q = int(rng.uniform(jpeg_quality_range[0], jpeg_quality_range[1]))
        enc = (noisy * 255.0).astype(np.uint8)
        ok, encimg = cv2.imencode('.jpg', enc, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
            if dec is not None:
                noisy = dec.astype(np.float32) / 255.0

    return noisy.astype(np.float32)

# ----------------------------
# Fallback simple stripe (kept for compatibility)
# ----------------------------
def _fallback_add_stripe_noise(img, orientation='horizontal', amplitude=0.12, frequency=20, thickness=1, phase=0.0):
    H, W, C = img.shape
    noise = np.zeros((H, W), dtype=np.float32)
    if orientation == 'horizontal' or orientation == 'both':
        ys = np.arange(H)
        pattern_h = 0.5 * (1 + np.sign(np.sin(2 * math.pi * (ys / max(1, H / frequency)) + phase)))
        if thickness > 1:
            kernel = np.ones(thickness)
            pattern_h = np.convolve(pattern_h, kernel, mode='same')
        noise += pattern_h[:, None]
    if orientation == 'vertical' or orientation == 'both':
        xs = np.arange(W)
        pattern_w = 0.5 * (1 + np.sign(np.sin(2 * math.pi * (xs / max(1, W / frequency)) + phase)))
        if thickness > 1:
            kernel = np.ones(thickness)
            pattern_w = np.convolve(pattern_w, kernel, mode='same')
        noise += pattern_w[None, :]
    noise = noise - noise.mean()
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise)) * amplitude
    noisy = img + noise[..., None]
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

# ----------------------------
# Dataset class (paired transforms)
# ----------------------------
class StripeDataset(Dataset):
    """
    clean_dir: required
    noisy_dir: optional (paired noisy)
    patch_size: crop size
    augment: whether to apply random paired augmentations
    synth_prob: when paired noisy exists, prob to use synthetic noisy instead
    force_gray: convert clean to single channel
    binarize_threshold: if set, threshold clean to 0/1
    """
    def __init__(self, clean_dir, noisy_dir=None, patch_size=128, augment=True, synth_prob=0.0, synth_params_generator=None, force_gray=False, binarize_threshold=None):
        super().__init__()
        self.clean_dir = Path(clean_dir)
        if not self.clean_dir.exists():
            raise RuntimeError(f"clean_dir not found: {clean_dir}")

        exts = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")
        self.clean_files = []
        for ext in exts:
            self.clean_files += list(self.clean_dir.rglob(f"*.{ext}"))
        self.clean_files = sorted([str(p) for p in self.clean_files])
        if len(self.clean_files) == 0:
            raise RuntimeError(f"No images found in clean_dir: {clean_dir}")

        self.noisy_map = {}
        if noisy_dir is not None:
            noisy_dir = Path(noisy_dir)
            noisy_files = []
            if noisy_dir.exists():
                for ext in exts:
                    noisy_files += list(noisy_dir.rglob(f"*.{ext}"))
                noisy_files = sorted([str(p) for p in noisy_files])
                noisy_by_name = {Path(p).name: p for p in noisy_files}
                for c in self.clean_files:
                    self.noisy_map[c] = noisy_by_name.get(Path(c).name, None)
            else:
                self.noisy_map = {c: None for c in self.clean_files}
        else:
            self.noisy_map = {c: None for c in self.clean_files}

        self.patch_size = patch_size
        self.augment = augment
        self.synth_prob = float(synth_prob)
        self.synth_params_generator = synth_params_generator
        self.force_gray = bool(force_gray)
        self.binarize_threshold = float(binarize_threshold) if binarize_threshold is not None else None

    def __len__(self):
        return len(self.clean_files)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _image_to_np(self, pil_img, force_gray=False, binarize_threshold=None):
        arr = np.array(pil_img).astype(np.float32) / 255.0
        if force_gray:
            if arr.ndim == 3:
                arr = 0.2989 * arr[...,0] + 0.5870 * arr[...,1] + 0.1140 * arr[...,2]
            arr = arr[..., None]
        else:
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
        if binarize_threshold is not None:
            gray = arr[...,0] if arr.ndim==3 else arr[...,0]
            mask = (gray >= binarize_threshold).astype(np.float32)
            arr = mask[..., None]
        return arr

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_map.get(clean_path, None)

        pil_clean = self._load_image(clean_path)
        pil_noisy = self._load_image(noisy_path) if noisy_path is not None else None

        # Paired transforms (ensure both clean and noisy share same random crop/flips)
        if self.augment:
            # RandomCrop params from torchvision
            i, j, h, w = T.RandomCrop.get_params(pil_clean, output_size=(self.patch_size, self.patch_size))
            pil_clean = TF.crop(pil_clean, i, j, h, w)
            if pil_noisy is not None:
                pil_noisy = TF.crop(pil_noisy, i, j, h, w)

            # Random horizontal flip
            if random.random() < 0.5:
                pil_clean = TF.hflip(pil_clean)
                if pil_noisy is not None:
                    pil_noisy = TF.hflip(pil_noisy)

            # Random vertical flip
            if random.random() < 0.5:
                pil_clean = TF.vflip(pil_clean)
                if pil_noisy is not None:
                    pil_noisy = TF.vflip(pil_noisy)
        else:
            pil_clean = TF.center_crop(pil_clean, (self.patch_size, self.patch_size))
            if pil_noisy is not None:
                pil_noisy = TF.center_crop(pil_noisy, (self.patch_size, self.patch_size))

        clean_np = self._image_to_np(pil_clean, force_gray=self.force_gray, binarize_threshold=self.binarize_threshold)

        use_synthetic = False
        if noisy_path is None:
            use_synthetic = True
        else:
            if random.random() < self.synth_prob:
                use_synthetic = True
            else:
                use_synthetic = False

        if use_synthetic:
            params = {}
            if callable(self.synth_params_generator):
                try:
                    params = self.synth_params_generator()
                except Exception:
                    params = {}
            img_for_synth = clean_np
            if img_for_synth.ndim == 3 and img_for_synth.shape[-1] == 1:
                img_for_synth = np.repeat(img_for_synth, 3, axis=-1)
            noisy_np = add_realistic_stripe_noise(img_for_synth, seed=None, **params)
        else:
            noisy_np = np.array(pil_noisy).astype(np.float32) / 255.0
            if noisy_np.ndim == 2:
                noisy_np = np.stack([noisy_np]*3, axis=-1)

        clean_t = torch.from_numpy(clean_np.transpose(2,0,1)).float()
        noisy_t = torch.from_numpy(noisy_np.transpose(2,0,1)).float()
        return noisy_t, clean_t

# ----------------------------
# Example synth_params_generator helper (unchanged)
# ----------------------------
def default_synth_params_generator():
    rng = np.random.RandomState(None)
    layers = []
    layers.append({
        "orientation": float(-30 + rng.uniform(-6, 6)),
        "wavelength": float(rng.uniform(100, 160)),
        "amplitude": float(rng.uniform(0.12, 0.35)),
        "thickness": float(rng.uniform(2, 5)),
        "mode": "add"
    })
    layers.append({
        "orientation": float(60 + rng.uniform(-8, 8)),
        "wavelength": float(rng.uniform(8, 24)),
        "amplitude": float(rng.uniform(0.03, 0.12)),
        "thickness": float(rng.uniform(1, 2)),
        "mode": "mult"
    })
    layers.append({
        "orientation": float(rng.uniform(-90, 90)),
        "wavelength": float(rng.uniform(6, 12)),
        "amplitude": float(rng.uniform(0.015, 0.06)),
        "thickness": 1.0,
        "mode": "add"
    })
    params = {
        "layers": layers,
        "amplitude_map_scale": int(rng.uniform(80, 220)),
        "jitter_scale": float(rng.uniform(3, 12)),
        "multiplicative_prob": 0.5,
        "add_gauss": float(rng.uniform(0.002, 0.01)),
        "jpeg_quality_range": (70, 95),
        "seed": None
    }
    return params
