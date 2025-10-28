# psinet_denoiser.py
# U-Net-like encoder-decoder with Residual Dense Blocks (RDB) at each scale.
# Predicts residual map which is added to input to obtain denoised output.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class RDB_Layer(nn.Module):
    """Single layer inside Residual Dense Block (conv -> concat)."""
    def __init__(self, in_ch, growth_rate, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, growth_rate, kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], dim=1)

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB):
    - n_layers conv layers with growth_rate
    - concatenation across layers
    - local feature fusion via 1x1 conv to reduce channels back to in_channels
    - final residual scaling
    """
    def __init__(self, in_channels, growth_rate=32, n_layers=4, res_scale=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.res_scale = res_scale

        layers = []
        ch = in_channels
        for i in range(n_layers):
            layers.append(RDB_Layer(ch, growth_rate))
            ch += growth_rate
        self.layers = nn.Sequential(*layers)
        # local feature fusion: reduce channels back to in_channels
        self.lff = nn.Conv2d(ch, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.layers(x)
        out = self.lff(out)
        return x + out * self.res_scale

class EncoderBlock(nn.Module):
    """Encoder block: ResidualDenseBlock(s) then downsample (stride-2 conv)."""
    def __init__(self, in_ch, out_ch, growth_rate, rdb_layers, n_rdb=2):
        super().__init__()
        self.rdbs = nn.Sequential(*[
            ResidualDenseBlock(in_ch if i == 0 else out_ch, growth_rate, n_layers=rdb_layers)
            for i in range(n_rdb)
        ])
        # after stacking, if out_ch differs, use 1x1 to match channels
        self.match = None
        if in_ch != out_ch:
            self.match = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        # downsample conv
        self.down = nn.Conv2d(out_ch if self.match else in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        out = x
        out = self.rdbs(out)
        if self.match is not None:
            out = self.match(out)
        down = self.down(out)
        return out, down  # return features for skip, and downsampled

class DecoderBlock(nn.Module):
    """Decoder block: upsample (interpolate+conv), concat skip, RDBs."""
    def __init__(self, in_ch, skip_ch, out_ch, growth_rate, rdb_layers, n_rdb=2):
        super().__init__()
        # upsample: interpolate + conv to avoid checkerboard artifacts
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.rdbs = nn.Sequential(*[
            ResidualDenseBlock(out_ch + skip_ch if i == 0 else out_ch, growth_rate, n_layers=rdb_layers)
            for i in range(n_rdb)
        ])

    def forward(self, x, skip):
        x = self.up_conv(x)
        # pad if necessary due to odd sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        cat = torch.cat([x, skip], dim=1)
        out = self.rdbs(cat)
        return out

class PSINetRDBUNet(nn.Module):
    """
    PSINet-like U-Net with Residual Dense Blocks.
    Predicts residual map; denoised = input + residual.
    """
    def __init__(self, in_channels=3, base_channels=32, growth_rate=16, rdb_layers=3,
                 n_rdb_per_scale=2, n_scales=3):
        """
        n_scales: number of downsampling steps (encoder depth). Typical 2-4.
        base_channels: channels at first scale; channels double each downsample.
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.rdb_layers = rdb_layers
        self.n_rdb_per_scale = n_rdb_per_scale
        self.n_scales = n_scales

        # initial conv
        self.initial = ConvBNReLU(in_channels, base_channels)

        # build encoder
        self.enc_blocks = nn.ModuleList()
        ch = base_channels
        for s in range(n_scales):
            out_ch = ch if s == 0 else ch * 2
            enc = EncoderBlock(ch, out_ch, growth_rate, rdb_layers, n_rdb=n_rdb_per_scale)
            self.enc_blocks.append(enc)
            ch = out_ch

        # bottleneck RDBs
        self.bottleneck = nn.Sequential(*[
            ResidualDenseBlock(ch, growth_rate, n_layers=rdb_layers) for _ in range(n_rdb_per_scale)
        ])

        # build decoder (reverse order)
        self.dec_blocks = nn.ModuleList()
        for s in reversed(range(n_scales)):
            skip_ch = base_channels if s == 0 else (base_channels * (2 ** s))
            in_ch = ch
            out_ch = skip_ch
            dec = DecoderBlock(in_ch, skip_ch, out_ch, growth_rate, rdb_layers, n_rdb=n_rdb_per_scale)
            self.dec_blocks.append(dec)
            ch = out_ch

        # final refine convs to residual
        mid = max(32, ch // 2)
        self.refine = nn.Sequential(
            ConvBNReLU(ch, mid),
            ConvBNReLU(mid, mid//2),
            nn.Conv2d(mid//2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        inp = x
        f = self.initial(x)
        skips = []
        out = f
        # encoder pass
        for enc in self.enc_blocks:
            skip_feat, out = enc(out)
            skips.append(skip_feat)
        # bottleneck
        out = self.bottleneck(out)
        # decoder pass: dec_blocks aligned with reversed skips
        for dec, skip in zip(self.dec_blocks, reversed(skips)):
            out = dec(out, skip)
        # refine
        res = self.refine(out)
        denoised = inp + res
        return denoised

if __name__ == "__main__":
    # quick unit test
    model = PSINetRDBUNet(in_channels=3, base_channels=32, growth_rate=16, rdb_layers=3, n_rdb_per_scale=2, n_scales=3)
    x = torch.randn(2,3,128,128)
    y = model(x)
    print("in:", x.shape, "out:", y.shape)