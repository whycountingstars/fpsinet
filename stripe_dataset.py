# stripe_dataset.py (updated: optional force_gray + binarize support)
# - if force_gray=True, clean images are converted to single-channel gray
# - if binarize_threshold is not None, clean images are thresholded to 0/1 (useful for binary GT)
# - noisy images remain RGB but will be converted to tensors; model will convert to gray internally if needed

from pathlib import Path
import random
import math
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# ... (keep realistic stripe synthesis functions unchanged: _smooth_noise, add_realistic_stripe_noise, etc.)
# For brevity this file keeps the existing add_realistic_stripe_noise implementation (unchanged).
# Only the dataset class below is adapted to support force_gray/binarize.

class StripeDataset(Dataset):
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

        # build noisy map if provided (same logic as before)
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

        if self.augment:
            self.transforms = T.Compose([
                T.RandomCrop(patch_size),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ])
        else:
            self.transforms = T.CenterCrop(patch_size)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _image_to_np(self, pil_img, force_gray=False, binarize_threshold=None):
        arr = np.array(pil_img).astype(np.float32) / 255.0
        # if force_gray requested, convert
        if force_gray:
            if arr.ndim == 3:
                # convert RGB -> gray
                arr = 0.2989 * arr[...,0] + 0.5870 * arr[...,1] + 0.1140 * arr[...,2]
            arr = arr[..., None]  # H,W,1
        else:
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
        if binarize_threshold is not None:
            # threshold the single channel (if multi-channel, threshold first channel)
            gray = arr[...,0] if arr.ndim==3 else arr[...,0]
            mask = (gray >= binarize_threshold).astype(np.float32)
            arr = mask[..., None]  # return single channel
        return arr

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_map.get(clean_path, None)

        pil_clean = self._load_image(clean_path)
        pil_clean = self.transforms(pil_clean)

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
            noisy_np = add_realistic_stripe_noise(np.repeat(clean_np if clean_np.shape[-1]==1 else clean_np, 3, axis=-1), seed=None, **params)
        else:
            pil_noisy = self._load_image(noisy_path)
            pil_noisy = self.transforms(pil_noisy)
            noisy_np = np.array(pil_noisy).astype(np.float32) / 255.0
            if noisy_np.ndim == 2:
                noisy_np = np.stack([noisy_np]*3, axis=-1)

        # convert to tensors
        clean_t = torch.from_numpy(clean_np.transpose(2,0,1)).float()
        noisy_t = torch.from_numpy(noisy_np.transpose(2,0,1)).float()
        return noisy_t, clean_t
