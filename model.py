"""
Comprehensive model implementations for Bladder Volume Estimation and ablation variants.

Includes:
- CoordConv2d
- SpatialAttention
- ChannelAttention
- Backbone (configurable to enable/disable CoordConv and SpatialAttention)
- Five full models:
    - BladderVolumeEstimator (base: CoordConv + SpatialAtt + ChannelAtt)
    - BladderVolumeEstimator_NoCoord (no CoordConv)
    - BladderVolumeEstimator_NoSpatial (no SpatialAttention)
    - BladderVolumeEstimator_NoChannel (no ChannelAttention)
    - BladderVolumeEstimator_NoSpatial_NoChannel (no SpatialAttention & no ChannelAttention)
- create_model(name, in_channels=1) factory
- utilities: count_parameters(model), try_compute_flops(model, input_size)
Notes:
- FLOPs computation uses `thop` if available; otherwise it will skip with a message.
- All models expose the same forward signature: (img_t, img_s) -> tensor (B,)
"""

from typing import Tuple
import time
import torch
import torch.nn as nn

try:
    from thop import profile
    _THOP_AVAILABLE = True
except Exception as e:
    print(e)
    _THOP_AVAILABLE = False


class CoordConv2d(nn.Module):
    """
    CoordConv: concatenate normalized (x,y) channels to input and apply Conv2d.
    Produces same spatial resolution as a normal Conv2d with padding.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, H, W = x.shape
        _device = x.device
        ys = torch.linspace(-1.0, 1.0, steps=H, device=_device).view(1, 1, H, 1).expand(b, 1, H, W)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=_device).view(1, 1, 1, W).expand(b, 1, H, W)
        x = torch.cat([x, xs, ys], dim=1)
        return self.conv(x)


class SpatialAttention(nn.Module):
    """
    Spatial attention as described: 7x7 conv -> sigmoid -> elementwise multiply
    Produces single-channel attention map (B,1,H,W) which is broadcast to input channels.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv7 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.conv7(x)  # (B,1,H,W)
        m = self.sig(m)
        return x * m


class ChannelAttention(nn.Module):
    """
    Lightweight channel attention (bottleneck FC -> ReLU -> FC -> Sigmoid).
    Input: (B, C)
    Produces channel weights (B, C) and returns weighted features.
    """
    def __init__(self, channels: int = 256, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        w = self.fc1(x)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sig(w)
        return x * w


class CoordConvBlock(nn.Module):
    """
    Single block that starts with CoordConv -> SpatialAttention (optional) -> BN -> ReLU -> MaxPool
    """
    def __init__(self, in_ch: int, out_ch: int, use_spatial: bool = True):
        super().__init__()
        self.coordconv = CoordConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial = SpatialAttention(out_ch)
        else:
            self.spatial = None
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coordconv(x)
        if self.use_spatial:
            x = self.spatial(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ConvBlock(nn.Module):
    """
    Single standard Conv2d block (no CoordConv). Keeps SpatialAttention optional.
    """
    def __init__(self, in_ch: int, out_ch: int, use_spatial: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial = SpatialAttention(out_ch)
        else:
            self.spatial = None
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_spatial:
            x = self.spatial(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class Backbone(nn.Module):
    """
    Generic backbone that can be configured:
    - use_coord: if True, use CoordConvBlock; else use ConvBlock
    - use_spatial: if True, include SpatialAttention in each block; else omit
    Produces a final global-pooled vector of size channels_list[-1] (128).
    """
    def __init__(self, in_channels: int = 1, channels_list=(16, 32, 64, 128),
                 use_coord: bool = True, use_spatial: bool = True):
        super().__init__()
        self.use_coord = use_coord
        self.use_spatial = use_spatial
        self.blocks = nn.ModuleList()
        prev = in_channels
        for out_ch in channels_list:
            if use_coord:
                block = CoordConvBlock(prev, out_ch, use_spatial=use_spatial)
            else:
                block = ConvBlock(prev, out_ch, use_spatial=use_spatial)
            self.blocks.append(block)
            prev = out_ch
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        return x


class BaseRegressor(nn.Module):
    """
    MLP regressor shared across all variants:
    Input dim = 256 (concat of two 128-d branch outputs)
    MLP: 256 -> 64 -> 32 -> 1
    """
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class BladderVolumeEstimatorBase(nn.Module):
    """
    Generic model that accepts a backbone module and a flag whether to use channel attention.
    The backbone passed is used for both transverse and sagittal inputs (shared weights).
    """
    def __init__(self, backbone: Backbone, use_channel_attn: bool = True):
        super().__init__()
        self.backbone = backbone
        self.use_channel_attn = use_channel_attn
        if use_channel_attn:
            self.channel_att = ChannelAttention(channels=256, reduction=4)
        else:
            self.channel_att = None
        self.regressor = BaseRegressor(in_dim=256)

    def forward(self, img_t: torch.Tensor, img_s: torch.Tensor) -> torch.Tensor:
        f_t = self.backbone(img_t)  # (B, 128)
        f_s = self.backbone(img_s)  # (B, 128)
        f = torch.cat([f_t, f_s], dim=1)  # (B, 256)
        if self.use_channel_attn:
            f = self.channel_att(f)
        out = self.regressor(f)  # (B,)
        return out


def create_model(name: str, in_channels: int = 1) -> nn.Module:
    """
    Create one of:
      - 'base' : CoordConv + SpatialAtt + ChannelAtt
      - 'no_coord' : standard Conv2d (no CoordConv) + SpatialAtt + ChannelAtt
      - 'no_spatial' : CoordConv + no SpatialAttention + ChannelAtt
      - 'no_channel' : CoordConv + SpatialAtt + no ChannelAtt
      - 'no_spatial_no_channel' : CoordConv + no Spatial + no Channel
    Returns nn.Module ready to .to(device).
    """
    name = name.lower()
    if name == "base":
        backbone = Backbone(in_channels=in_channels, use_coord=True, use_spatial=True)
        model = BladderVolumeEstimatorBase(backbone=backbone, use_channel_attn=True)
    elif name == "no_coord":
        backbone = Backbone(in_channels=in_channels, use_coord=False, use_spatial=True)
        model = BladderVolumeEstimatorBase(backbone=backbone, use_channel_attn=True)
    elif name == "no_spatial":
        backbone = Backbone(in_channels=in_channels, use_coord=True, use_spatial=False)
        model = BladderVolumeEstimatorBase(backbone=backbone, use_channel_attn=True)
    elif name == "no_channel":
        backbone = Backbone(in_channels=in_channels, use_coord=True, use_spatial=True)
        model = BladderVolumeEstimatorBase(backbone=backbone, use_channel_attn=False)
    elif name == "no_spatial_no_channel":
        backbone = Backbone(in_channels=in_channels, use_coord=True, use_spatial=False)
        model = BladderVolumeEstimatorBase(backbone=backbone, use_channel_attn=False)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Return (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_compute_flops(model: nn.Module, input_size=(1, 1, 224, 224), device=None):
    """
    Try to compute FLOPs using thop.profile.

    Returns (macs, params) as reported by thop (or None, None if thop unavailable).
    Note: pass a single-tensor input representing the concatenated two-branch input as required.

    The backbone expects two inputs (img_t, img_s). thop expects a single tuple of inputs -> we provide (img_t, img_s)
    """
    if not _THOP_AVAILABLE:
        print("thop not available: install via `pip install thop` to compute FLOPs.")
        return None, None

    # Prepare dummy inputs (batch=1)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    model.eval()
    b, c, h, w = input_size
    dummy_t = torch.randn((b, c, h, w), device=device)
    dummy_s = torch.randn((b, c, h, w), device=device)

    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_t, dummy_s), verbose=False)
    return macs, params


def benchmark_runtime(model: nn.Module, input_size=(1, 1, 224, 224), device=None, n_iters=50, warmup=10):
    """
    Rough runtime benchmark (ms per forward pass). Uses CUDA sync if available.
    Runs warmup iterations then times `n_iters` forward passes and returns median and mean ms.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    model.eval()

    b, c, h, w = input_size
    dummy_t = torch.randn((b, c, h, w), device=device)
    dummy_s = torch.randn((b, c, h, w), device=device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_t, dummy_s)
        torch.cuda.synchronize() if device.type == 'cuda' else None

    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.time()
            _ = model(dummy_t, dummy_s)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)  # ms

    import numpy as np
    return float(np.mean(times)), float(np.median(times))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--flops', action='store_true', help='attempt to compute FLOPs (requires thop)')
    parser.add_argument('--time', action='store_true', help='benchmark runtime')
    args = parser.parse_args()

    test_device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device chosen:", test_device)
    models_to_test = [
        "base",
        "no_coord",
        "no_spatial",
        "no_channel",
        "no_spatial_no_channel"
    ]

    summary = []
    for test_name in models_to_test:
        print("--------------------------------------------------")
        print("Model:", test_name)
        test_model = create_model(test_name, in_channels=args.in_channels)
        test_total, test_trainable = count_parameters(test_model)
        print(f"Parameters: total={test_total:,} | trainable={test_trainable:,}")
        flops = None
        if args.flops:
            try:
                test_macs, params_thop = try_compute_flops(test_model, input_size=(1, args.in_channels, args.input_size, args.input_size), device=test_device)
                if test_macs is not None:
                    print(f"FLOPs (MACs): {test_macs:,} | params (thop): {params_thop:,}")
                    flops = test_macs
            except Exception as e:
                print("Error computing FLOPs:", e)
        if args.time:
            mean_ms, median_ms = benchmark_runtime(test_model, input_size=(1, args.in_channels, args.input_size, args.input_size), device=test_device)
            print(f"Runtime (ms): mean={mean_ms:.2f} | median={median_ms:.2f}")
        summary.append({
            'name': test_name,
            'params_total': test_total,
            'params_trainable': test_trainable,
            'flops_macs': flops
        })

    # Print summary table
    print("\n================ Summary ================\n")
    for s in summary:
        print(f"{s['name']:25s} params={s['params_total']:,} trainable={s['params_trainable']:,} flops_macs={s['flops_macs']}")
    print("\n=========================================\n")
