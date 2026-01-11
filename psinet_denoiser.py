# PSINetRDBUNet (modified)
# - single-channel logits output (suitable for binary/single-value GT)
# - CBAM attention (channel + spatial)
# - LowFreq branch with reflection padding to reduce border artifacts
# - Gate & lowfreq final conv initialized small to avoid lowfreq dominating early training
# - Returns dict: {'denoised','res_low','res_high','gate'}

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rgb_to_gray(x):
    r, g, b = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

# Helper conv using reflection padding to reduce border artifacts
def conv3x3_reflect(in_ch, out_ch, stride=1, bias=True):
    layers = []
    # For stride>1 we keep reflection pad 1 and use stride in conv
    layers.append(nn.ReflectionPad2d(1))
    layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=0, bias=bias))
    return nn.Sequential(*layers)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class RDB_Layer(nn.Module):
    def __init__(self, in_ch, growth_rate, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, growth_rate, kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, n_layers=4, res_scale=0.2):
        super().__init__()
        self.in_channels = in_channels
        layers = []
        ch = in_channels
        for _ in range(n_layers):
            layers.append(RDB_Layer(ch, growth_rate))
            ch += growth_rate
        self.layers = nn.Sequential(*layers)
        self.lff = nn.Conv2d(ch, in_channels, kernel_size=1, bias=True)
        self.res_scale = res_scale
    def forward(self, x):
        out = self.layers(x)
        out = self.lff(out)
        return x + out * self.res_scale

# CBAM (channel + spatial)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        a = self.fc(self.avg_pool(x))
        m = self.fc(self.max_pool(x))
        out = self.sig(a + m)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        maxc,_ = torch.max(x, dim=1, keepdim=True)
        avgc = torch.mean(x, dim=1, keepdim=True)
        cat = torch.cat([maxc, avgc], dim=1)
        return x * self.sig(self.conv(cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, growth_rate, rdb_layers, n_rdb=2):
        super().__init__()
        self.match = None
        if in_ch != out_ch:
            self.match = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.rdbs = nn.Sequential(*[ResidualDenseBlock(out_ch, growth_rate, n_layers=rdb_layers) for _ in range(n_rdb)])
        # use reflection pad conv for downsample to reduce artifacts
        self.down = conv3x3_reflect(out_ch, out_ch, stride=2, bias=True)
    def forward(self, x):
        if self.match is not None:
            x = self.match(x)
        out = self.rdbs(x)
        down = self.down(out)
        return out, down

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, growth_rate, rdb_layers, n_rdb=2):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.reduce = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.rdbs = nn.Sequential(*[ResidualDenseBlock(out_ch, growth_rate, n_layers=rdb_layers) for _ in range(n_rdb)])
    def forward(self, x, skip):
        x = self.up_conv(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        cat = torch.cat([x, skip], dim=1)
        reduced = self.reduce(cat)
        out = self.rdbs(reduced)
        return out

class LowFreqBranch(nn.Module):
    def __init__(self, in_ch, mid_ch=256, out_ch=1, num_down=5):
        super().__init__()
        self.num_down = max(1, num_down)
        layers = []
        ch = in_ch
        # Use standard conv with padding=1 (safe on small spatial sizes)
        for _ in range(self.num_down):
            layers.append(nn.Conv2d(ch, mid_ch, kernel_size=3, stride=2, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            ch = mid_ch
        layers.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        self.down_stack = nn.Sequential(*layers)

        up_layers = []
        for _ in range(self.num_down):
            up_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            up_layers.append(nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=True))
            up_layers.append(nn.ReLU(inplace=True))
        self.up_stack = nn.Sequential(*up_layers)

        self.to_res = nn.Sequential(
            nn.Conv2d(mid_ch, max(mid_ch//2,1), kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(mid_ch//2,1), out_ch, kernel_size=3, padding=1, bias=True)
        )

        # smoothing conv (keep as before)
        self.smooth = nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2, bias=False, groups=1)
        self._init_smooth()

    def _init_smooth(self):
        k = torch.tensor([[1,4,6,4,1],
                          [4,16,24,16,4],
                          [6,24,36,24,6],
                          [4,16,24,16,4],
                          [1,4,6,4,1]], dtype=torch.float32)
        k = k / k.sum()
        k = k.unsqueeze(0).unsqueeze(0)
        out_ch = self.smooth.weight.shape[0]
        k = k.repeat(out_ch,1,1,1)
        with torch.no_grad():
            self.smooth.weight.copy_(k)

    def forward(self, x):
        # x: bottleneck feature (B, Cb, Hb, Wb)
        # If spatial size is extremely small, conv with padding still works.
        d = self.down_stack(x)
        u = self.up_stack(d)
        # restore to bottleneck spatial size before final projection
        if u.shape[-2:] != x.shape[-2:]:
            u = F.interpolate(u, size=x.shape[-2:], mode='bilinear', align_corners=False)
        res = self.to_res(u)
        res = self.smooth(res)
        return res

class PSINetRDBUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=48, growth_rate=24, rdb_layers=4,
                 n_rdb_per_scale=2, n_scales=4, lowfreq_mid=256, lowfreq_down=5, out_channels=1, use_cbam=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_cbam = use_cbam

        self.initial = ConvBNReLU(in_channels, base_channels)

        # encoder
        self.enc_blocks = nn.ModuleList()
        ch = base_channels
        for s in range(n_scales):
            out_ch = ch if s == 0 else ch * 2
            enc = EncoderBlock(ch, out_ch, growth_rate, rdb_layers, n_rdb=n_rdb_per_scale)
            self.enc_blocks.append(enc)
            ch = out_ch

        bottleneck_ch = ch
        self.bottleneck = nn.Sequential(*[ResidualDenseBlock(bottleneck_ch, growth_rate, n_layers=rdb_layers) for _ in range(n_rdb_per_scale)])
        if self.use_cbam:
            self.bottleneck_attn = CBAM(bottleneck_ch, reduction=8)
        else:
            self.bottleneck_attn = nn.Identity()

        # decoder
        self.dec_blocks = nn.ModuleList()
        curr_ch = bottleneck_ch
        for s in reversed(range(n_scales)):
            skip_ch = base_channels if s == 0 else (base_channels * (2 ** s))
            in_ch = curr_ch
            out_ch = skip_ch
            dec = DecoderBlock(in_ch, skip_ch, out_ch, growth_rate, rdb_layers, n_rdb=n_rdb_per_scale)
            self.dec_blocks.append(dec)
            curr_ch = out_ch

        mid = max(32, curr_ch // 2)
        self.refine_high = nn.Sequential(
            ConvBNReLU(curr_ch, mid),
            ConvBNReLU(mid, mid//2),
            nn.Conv2d(mid//2, out_channels, kernel_size=3, padding=1)
        )

        # optional attention on decoder output (identity if channel mismatch)
        if self.use_cbam:
            self.out_attn = CBAM(max(mid//2, out_channels), reduction=8)
        else:
            self.out_attn = nn.Identity()

        # low-frequency branch
        self.lowfreq = LowFreqBranch(in_ch=bottleneck_ch, mid_ch=lowfreq_mid, out_ch=out_channels, num_down=lowfreq_down)

        gate_mid = max(bottleneck_ch//2, 1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(bottleneck_ch, gate_mid, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # careful initialization: make gate bias zero (sigmoid ~0.5) and small gate weights
        with torch.no_grad():
            try:
                final_gate_conv = self.gate_conv[2]
                if isinstance(final_gate_conv, nn.Conv2d):
                    nn.init.constant_(final_gate_conv.bias, 0.0)
                    nn.init.normal_(final_gate_conv.weight, mean=0.0, std=1e-2)
            except Exception:
                pass
            # set lowfreq final conv weights small so it doesn't dominate initially
            try:
                last_conv = self.lowfreq.to_res[-1]
                if isinstance(last_conv, nn.Conv2d):
                    nn.init.constant_(last_conv.weight, 0.0)
                    if last_conv.bias is not None:
                        nn.init.constant_(last_conv.bias, 0.0)
            except Exception:
                pass

    def forward(self, x):
        inp_gray = rgb_to_gray(x)  # B,1,H,W
        f = self.initial(x)
        skips = []
        out = f
        for enc in self.enc_blocks:
            skip_feat, out = enc(out)
            skips.append(skip_feat)
        out = self.bottleneck(out)
        out = self.bottleneck_attn(out)
        bottleneck_feat = out
        for dec, skip in zip(self.dec_blocks, reversed(skips)):
            out = dec(out, skip)
        res_high = self.refine_high(out)  # B,1,H,W

        res_low = self.lowfreq(bottleneck_feat)
        gate = self.gate_conv(bottleneck_feat)
        if gate.shape[-2:] != res_high.shape[-2:]:
            gate = F.interpolate(gate, size=res_high.shape[-2:], mode='bilinear', align_corners=False)
        if res_low.shape[-2:] != res_high.shape[-2:]:
            res_low = F.interpolate(res_low, size=res_high.shape[-2:], mode='bilinear', align_corners=False)
        res = gate * res_high + (1.0 - gate) * res_low
        denoised = inp_gray + res
        return {'denoised': denoised, 'res_low': res_low, 'res_high': res_high, 'gate': gate}
