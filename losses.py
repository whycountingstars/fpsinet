# losses.py
# Combined losses for stripe denoising: L1, FFT loss, VGG perceptual, SSIM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# ------------------------
# FFT loss (magnitude / log-magnitude)
# ------------------------
def fft_mag_loss(pred, target, eps=1e-8, use_log=True):
    """
    pred, target: tensors [N,C,H,W] in [0,1]
    Compute L1 loss between magnitude (or log magnitude) of FFT2 per channel and average.
    """
    # compute rfft2 for speed: returns complex tensor
    p_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    t_fft = torch.fft.rfft2(target, dim=(-2, -1))
    p_mag = torch.abs(p_fft)
    t_mag = torch.abs(t_fft)
    if use_log:
        p_mag = torch.log(p_mag + eps)
        t_mag = torch.log(t_mag + eps)
    loss = F.l1_loss(p_mag, t_mag)
    return loss

# ------------------------
# VGG perceptual loss
# ------------------------
class VGGFeatureExtractor(nn.Module):
    """
    Extract features from pretrained VGG16 layers.
    Returns list of feature maps to compute perceptual loss.
    """
    def __init__(self, layers=(3,8,15,22), device='cpu'):
        super().__init__()
        # load pretrained VGG16 features
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
        except Exception:
            vgg = models.vgg16(pretrained=True).features.to(device)
        vgg.eval()
        # freeze parameters
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layers = [int(l) for l in layers]
        self.device = device

    def forward(self, x):
        # x in [0,1], convert to VGG expected normalized input
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        # normalize
        x_in = (x - mean) / std
        features = []
        out = x_in
        max_layer = max(self.layers) if len(self.layers) > 0 else -1
        for idx, layer in enumerate(self.vgg):
            out = layer(out)
            if idx in self.layers:
                features.append(out)
            if idx >= max_layer:
                break
        return features

def vgg_perceptual_loss(feat_extractor, pred, target, loss_fn=F.l1_loss):
    """
    Compute perceptual loss as sum of L1 between VGG features
    feat_extractor: instance of VGGFeatureExtractor
    pred, target: [N,C,H,W] in [0,1]
    """
    pred_feats = feat_extractor(pred)
    tgt_feats = feat_extractor(target)
    loss = 0.0
    for pf, tf in zip(pred_feats, tgt_feats):
        loss = loss + loss_fn(pf, tf)
    return loss

# ------------------------
# SSIM (differentiable, returns tensor scalar)
# ------------------------
def gaussian_window(window_size=11, sigma=1.5, channel=1, device='cpu'):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    g2d = g[:, None] * g[None, :]
    window = g2d.expand(channel, 1, window_size, window_size).to(device)
    return window

def ssim_torch(img1, img2, window_size=11, data_range=1.0, K=(0.01,0.03)):
    """
    Differentiable SSIM implementation that returns a tensor scalar (mean across batch).
    img1, img2: (N,C,H,W), values in [0,1]
    """
    device = img1.device
    N, C, H, W = img1.shape
    window = gaussian_window(window_size=window_size, channel=C, device=device)
    pad = window_size//2
    mu1 = F.conv2d(img1, window, groups=C, padding=pad)
    mu2 = F.conv2d(img2, window, groups=C, padding=pad)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, groups=C, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, groups=C, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, groups=C, padding=pad) - mu1_mu2

    C1 = (K[0]*data_range)**2
    C2 = (K[1]*data_range)**2

    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    # average over spatial and channel dims, then mean over batch -> returns tensor scalar
    return ssim_map.mean(dim=[1,2,3]).mean()