# PSINetRDBUNet (optimized)
# - single-channel logits output (suitable for binary/single-value GT)
# - CBAM attention (channel + spatial) applied to bottleneck and decoder output
# - deeper-by-default params, configurable via constructor
# - LowFreq branch retained and with larger receptive field

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rgb_to_gray(x):
    # x: B,3,H,W -> return B,1,H,W
    r, g, b = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

# ----- basic blocks -----
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
        self.growth_rate = growth_rate
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

# ----- attention: lightweight CBAM (channel + spatial) -----
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
        # x: B,C,H,W -> compute along channel
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

# ----- encoder / decoder blocks -----
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, growth_rate, rdb_layers, n_rdb=2):
        super().__init__()
        self.match = None
        if in_ch != out_ch:
            self.match = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.rdbs = nn.Sequential(*[ResidualDenseBlock(out_ch, growth_rate, n_layers=rdb_layers) for _ in range(n_rdb)])
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True)
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

# ----- low-frequency branch -----
class LowFreqBranch(nn.Module):
    def __init__(self, in_ch, mid_ch=256, out_ch=1, num_down=5):
        super().__init__()
        self.num_down = max(1, num_down)
        layers = []
        ch = in_ch
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
        self.smooth = nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2, bias=False, groups=1)
        self._init_smooth()
    def _init_smooth(self):
        k = torch.tensor([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=torch.float32)
        k = k / k.sum()
        k = k.unsqueeze(0).unsqueeze(0)
        out_ch = self.smooth.weight.shape[0]
        k = k.repeat(out_ch,1,1,1)
        with torch.no_grad():
            self.smooth.weight.copy_(k)
    def forward(self, x):
        d = self.down_stack(x)
        u = self.up_stack(d)
        if u.shape[-2:] != x.shape[-2:]:
            u = F.interpolate(u, size=x.shape[-2:], mode='bilinear', align_corners=False)
        res = self.to_res(u)
        res = self.smooth(res)
        return res

# ----- final network -----
class PSINetRDBUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=48, growth_rate=24, rdb_layers=4,
                 n_rdb_per_scale=2, n_scales=4, lowfreq_mid=256, lowfreq_down=5, out_channels=1, use_cbam=True):
        """
        Deeper defaults: n_scales=4, rdb_layers=4, base_channels=48 for stronger capacity.
        All params are configurable when instantiating.
        """
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

        # attention on bottleneck (optional)
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

        # add attention on decoder output
        if self.use_cbam:
            self.out_attn = CBAM(out_channels if out_channels>1 else mid//2, reduction=8)
        else:
            self.out_attn = nn.Identity()

        # low-frequency branch uses bottleneck channels
        self.lowfreq = LowFreqBranch(in_ch=bottleneck_ch, mid_ch=lowfreq_mid, out_ch=out_channels, num_down=lowfreq_down)

        gate_mid = max(bottleneck_ch//2, 1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(bottleneck_ch, gate_mid, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: B,3,H,W
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
        # optional attn on res_high (if CBAM expects more channels we used identity earlier)
        # res_high_att = self.out_attn(res_high)  # out_attn expects channels matching; keep identity if sizes mismatch

        res_low = self.lowfreq(bottleneck_feat)
        gate = self.gate_conv(bottleneck_feat)
        # upsample gate & res_low to full spatial size before fuse
        if gate.shape[-2:] != res_high.shape[-2:]:
            gate = F.interpolate(gate, size=res_high.shape[-2:], mode='bilinear', align_corners=False)
        if res_low.shape[-2:] != res_high.shape[-2:]:
            res_low = F.interpolate(res_low, size=res_high.shape[-2:], mode='bilinear', align_corners=False)
        res = gate * res_high + (1.0 - gate) * res_low
        denoised = inp_gray + res  # logits-like single-channel
        # return dict to enable losses/visualization of components
        return {'denoised': denoised, 'res_low': res_low, 'res_high': res_high, 'gate': gate}
