# stripe_dataset.py
# Stripe dataset with integrated realistic stripe/noise synthesis (no external realistic_stripe.py required)
#
# Supports three data modes:
# 1) clean-only: synthesize noisy from clean images
# 2) clean + noisy paired: load real noisy (matching filenames)
# 3) mixed: when paired noisy exists, with synth_prob probability use synthetic noisy instead of real noisy
#
# Returns (noisy_tensor, clean_tensor) as float32 in [0,1], shape (C,H,W)
#
# Usage:
#   ds = StripeDataset(clean_dir, noisy_dir=None, patch_size=128, augment=True, synth_prob=0.0)
#   ds = StripeDataset(clean_dir, noisy_dir="/path/to/noisy", synth_prob=0.2)
#
# Requirements: numpy, opencv-python (cv2), Pillow, torch, torchvision

from pathlib import Path
import random
import math
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# ----------------------------
# Realistic stripe synthesis
# ----------------------------
def _smooth_noise(H, W, scale=64, seed=None):
    """Generate a smooth random field by blurring white noise."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    n = rng.randn(H, W).astype(np.float32)
    k = max(1, int(scale))
    # Gaussian blur: use OpenCV; kernel size 0 lets us specify sigma directly
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
    """
    img: HxW x C, float in [0,1]
    layers: list of dicts, each with keys:
        - orientation (deg), wavelength (px), amplitude (base), thickness, mode ('add'/'mult' optional)
      If None, a default set of layers is sampled (suitable for many stripe-like noises).
    amplitude_map_scale: scale for smooth amplitude modulation
    jitter_scale: coordinate jitter magnitude in pixels (warp)
    multiplicative_prob: probability to set layer.mode to 'mult' if not specified
    add_gauss: std of additive gaussian after synthesis
    jpeg_quality_range: tuple for optional JPEG compression to add artifacts
    seed: random seed for reproducibility
    """
    H, W = img.shape[:2]
    rng = np.random.RandomState(seed) if seed is not None else np.random

    # default layers if not provided: mix coarse strong slanted stripes + finer crossing stripes + background ripple
    if layers is None:
        layers = [
            {"orientation": -30 + rng.uniform(-5, 5), "wavelength": rng.uniform(100, 160), "amplitude": rng.uniform(0.12, 0.35), "thickness": rng.uniform(2, 5), "mode": "add"},
            {"orientation": 60 + rng.uniform(-8, 8), "wavelength": rng.uniform(8, 24), "amplitude": rng.uniform(0.03, 0.12), "thickness": rng.uniform(1, 2), "mode": "mult"},
            {"orientation": rng.uniform(-90, 90), "wavelength": rng.uniform(6, 12), "amplitude": rng.uniform(0.015, 0.06), "thickness": 1, "mode": "add"}
        ]

    # coordinate grid
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    # amplitude modulation map (smooth)
    amp_map = _smooth_noise(H, W, scale=amplitude_map_scale, seed=(seed + 1) if seed else None)
    amp_map = (amp_map - amp_map.min()) / (amp_map.max() - amp_map.min() + 1e-12)  # 0..1
    amp_map = 0.6 + amp_map * 1.4  # maps to ~[0.6, 2.0]

    # coordinate jitter fields for warp (makes lines not perfectly straight)
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

        # convert sine to pulse-like stripes
        k = max(0.5, 8.0 / max(1.0, thickness))
        pulse = 0.5 * (1.0 + np.tanh(k * stripe))  # values in [0,1]

        # blur to control thickness
        blur_k = int(max(1, round(thickness * 2)))
        if blur_k % 2 == 0:
            blur_k += 1
        if blur_k > 1:
            pulse = cv2.GaussianBlur(pulse, (blur_k, blur_k), sigmaX=blur_k/2, sigmaY=blur_k/2)

        # local amplitude map
        local_amp = base_amp * amp_map * (0.6 + 0.8 * rng.rand())
        layer_pattern = pulse * local_amp

        # add fine-scale texture sometimes
        if rng.rand() < 0.45:
            fine = _smooth_noise(H, W, scale=max(2, wavelength / 8.0), seed=None)
            layer_pattern = layer_pattern * (1.0 + 0.12 * fine)

        # accumulate
        total_noise += layer_pattern

    # create multiplicative weight map to mix multiplicative/additive behavior locally
    mult_weight_map = np.clip(total_noise * 3.0, 0.0, 1.0)

    noisy = img.copy().astype(np.float32)
    # apply combined multiplicative/additive effect per-channel
    for c in range(noisy.shape[2]):
        noisy[..., c] = noisy[..., c] * (1.0 + total_noise * mult_weight_map)
        noisy[..., c] = noisy[..., c] + total_noise * (1.0 - mult_weight_map)

    # per-column bias (simulate readout / column artifacts)
    if rng.rand() < 0.85:
        col_bias = _smooth_noise(1, W, scale=max(1, W / 8.0), seed=None).reshape(W)
        col_bias = (col_bias - col_bias.min()) / (col_bias.max() - col_bias.min() + 1e-12)
        col_bias = (col_bias - 0.5) * 0.03 * (0.5 + rng.rand())  # small ± bias
        noisy = noisy + col_bias[None, :, None]

    # small color shift sometimes
    if rng.rand() < 0.5:
        shift = (rng.randn(3) * 0.01).astype(np.float32)
        noisy = noisy + shift[None, None, :]

    # additive gaussian
    if add_gauss and add_gauss > 0:
        noisy = noisy + np.random.randn(*noisy.shape).astype(np.float32) * add_gauss

    noisy = np.clip(noisy, 0.0, 1.0)

    # optional JPEG compression to simulate compression artifacts
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
# Dataset class
# ----------------------------
class StripeDataset(Dataset):
    """
    clean_dir: required (folder with clean images)
    noisy_dir: optional (folder with noisy images, same filenames expected)
    patch_size: random crop size
    augment: whether to use random crop + flips
    synth_prob: when paired noisy exists, probability to use synthetic noisy instead of real noisy
    synth_params_generator: optional callable returning a dict of parameters for add_realistic_stripe_noise per sample
    """
    def __init__(self, clean_dir, noisy_dir=None, patch_size=128, augment=True, synth_prob=0.0, synth_params_generator=None):
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

        # build noisy map by filename if noisy_dir provided
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
                    name = Path(c).name
                    self.noisy_map[c] = noisy_by_name.get(name, None)
            else:
                # noisy dir specified but not found: treat as no noisy files
                self.noisy_map = {c: None for c in self.clean_files}
        else:
            # no noisy dir: synthesize for all
            self.noisy_map = {c: None for c in self.clean_files}

        self.patch_size = patch_size
        self.augment = augment
        self.synth_prob = float(synth_prob)
        self.synth_params_generator = synth_params_generator

        if self.augment:
            self.transforms = T.Compose([
                T.RandomCrop(patch_size),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ])
        else:
            self.transforms = T.CenterCrop(patch_size)

    def __len__(self):
        return len(self.clean_files)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _image_to_np(self, pil_img):
        arr = np.array(pil_img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_map.get(clean_path, None)

        # load clean and apply same transform (we apply transforms separately; if strict alignment needed,
        # ensure files are same size and transforms are deterministic or implement paired transforms)
        pil_clean = self._load_image(clean_path)
        pil_clean = self.transforms(pil_clean)
        clean_np = self._image_to_np(pil_clean)

        use_synthetic = False
        if noisy_path is None:
            use_synthetic = True
        else:
            if random.random() < self.synth_prob:
                use_synthetic = True
            else:
                use_synthetic = False

        if use_synthetic:
            # generate synthetic noisy; pass params from generator if provided
            params = {}
            if callable(self.synth_params_generator):
                try:
                    params = self.synth_params_generator()
                except Exception:
                    params = {}
            # ensure we pass an image-sized seed for reproducibility if desired
            noisy_np = add_realistic_stripe_noise(clean_np, seed=None, **params)
        else:
            # load real noisy and apply same transforms
            pil_noisy = self._load_image(noisy_path)
            pil_noisy = self.transforms(pil_noisy)
            noisy_np = self._image_to_np(pil_noisy)

        clean_t = torch.from_numpy(clean_np.transpose(2, 0, 1)).float()
        noisy_t = torch.from_numpy(noisy_np.transpose(2, 0, 1)).float()
        return noisy_t, clean_t


# ----------------------------
# Example synth_params_generator helper
# ----------------------------
def default_synth_params_generator():
    """Return a params dict with randomized layers appropriate for many stripe types."""
    rng = np.random.RandomState(None)
    layers = []
    # main coarse slanted line
    layers.append({
        "orientation": float(-30 + rng.uniform(-6, 6)),
        "wavelength": float(rng.uniform(100, 160)),
        "amplitude": float(rng.uniform(0.12, 0.35)),
        "thickness": float(rng.uniform(2, 5)),
        "mode": "add"
    })
    # crossing fine lines
    layers.append({
        "orientation": float(60 + rng.uniform(-8, 8)),
        "wavelength": float(rng.uniform(8, 24)),
        "amplitude": float(rng.uniform(0.03, 0.12)),
        "thickness": float(rng.uniform(1, 2)),
        "mode": "mult"
    })
    # background ripple/noise
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

# End of file