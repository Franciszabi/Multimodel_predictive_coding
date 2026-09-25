"""Independent James-style visual baseline with strict frame causality.

CNN topology and frame-level attention follow Gornet & Thomson's public code:
https://github.com/jgornet/predictive-coding-recovers-maps
This is a causal adaptation, not a bit-for-bit reproduction: GroupNorm replaces
BatchNorm and temporal-block normalization excludes the sequence dimension.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class DownResidual(nn.Module):
    def __init__(self, in_channels: int, channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.skip = nn.Conv2d(in_channels, channels, 1, stride=2) if downsample else nn.Identity()

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(h)) + self.skip(x))


class UpResidual(nn.Module):
    def __init__(self, in_channels: int, channels: int, upsample: bool = False):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels, channels, 3, stride=2 if upsample else 1,
            padding=1, output_padding=1 if upsample else 0,
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.ConvTranspose2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.skip = (
            nn.ConvTranspose2d(in_channels, channels, 4, stride=2, padding=1)
            if upsample else nn.Identity()
        )

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(h)) + self.skip(x))


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.norm = nn.GroupNorm(8, 64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            DownResidual(64, 64), DownResidual(64, 64),
            DownResidual(64, 128, True), DownResidual(128, 128),
        )

    def forward(self, x):
        return self.blocks(self.pool(F.relu(self.norm(self.conv(x)))))


class VisualDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            UpResidual(128, 64, True), UpResidual(64, 64),
            UpResidual(64, 64, True), UpResidual(64, 64),
            UpResidual(64, 64, True),
        )
        self.output = nn.ConvTranspose2d(64, 3, 1)

    def forward(self, x):
        return self.output(self.blocks(x))


class FrameAttention(nn.Module):
    """Attention over frames, not over individual spatial patches."""

    def __init__(self, channels: int = 128, num_heads: int = 8):
        super().__init__()
        if num_heads < 1 or channels % num_heads:
            raise ValueError("num_heads must be a positive divisor of 128")
        self.num_heads = num_heads
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, length, c, h, w = x.shape
        folded = x.reshape(b * length, c, h, w)
        q, k, v = [
            projection(folded).reshape(b, length, self.num_heads, -1).transpose(1, 2)
            for projection in (self.query, self.key, self.value)
        ]
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return result.transpose(1, 2).reshape(b, length, c, h, w)


class TemporalBlock(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.attention = FrameAttention(num_heads=num_heads)
        self.ffn = nn.Sequential(nn.Conv2d(128, 128, 1), nn.ReLU(), nn.Conv2d(128, 128, 1))

    def forward(self, x):
        x = F.layer_norm(x + self.attention(x), x.shape[2:])
        b, length, c, h, w = x.shape
        update = self.ffn(x.reshape(b * length, c, h, w)).reshape_as(x)
        return F.layer_norm(x + update, x.shape[2:])


class VisualPredictiveModel(nn.Module):
    def __init__(
        self, num_heads: int = 8, num_layers: int = 2,
        position_encoding: str = "none", model_type: str = "predictive",
    ):
        super().__init__()
        if model_type not in {"predictive", "autoencoder"}:
            raise ValueError("Unknown model_type")
        if position_encoding not in {"none", "sinusoidal"}:
            raise ValueError("Unknown position_encoding")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if model_type == "autoencoder" and position_encoding != "none":
            raise ValueError("The same-frame AE must not use time position encoding")
        self.model_type = model_type
        self.position_encoding = position_encoding
        self.encoder = VisualEncoder()
        self.decoder = VisualDecoder()
        self.blocks = nn.ModuleList(
            TemporalBlock(num_heads) for _ in range(num_layers if model_type == "predictive" else 0)
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward_features(self, images, return_layers: bool = False):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("Expected images [B,L,3,H,W]")
        b, length, c, h, w = images.shape
        if min(h, w) < 16 or h % 8 or w % 8:
            raise ValueError("Image dimensions must be >=16 and divisible by 8")
        f = self.encoder(images.reshape(b * length, c, h, w))
        f = f.reshape(b, length, *f.shape[1:])
        layers = {"encoder": f} if return_layers else {}
        if self.position_encoding == "sinusoidal":
            time = torch.arange(length, device=f.device, dtype=torch.float32)[:, None]
            scale = torch.exp(torch.arange(0, 128, 2, device=f.device) * (-math.log(10000.0) / 128))
            pe = torch.zeros(length, 128, device=f.device)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(time * scale), torch.cos(time * scale)
            f = f + pe.to(f.dtype)[None, :, :, None, None]
        for index, block in enumerate(self.blocks, 1):
            f = block(f)
            if return_layers:
                layers[f"temporal_{index}"] = f
        if return_layers:
            layers["final"] = f
            return layers
        return f

    def forward(self, images, return_latents: bool = False):
        layers = self.forward_features(images, return_layers=return_latents)
        features = layers["final"] if return_latents else layers
        b, length, c, h, w = features.shape
        prediction = self.decoder(features.reshape(b * length, c, h, w))
        prediction = prediction.reshape(b, length, 3, *prediction.shape[-2:])
        return (prediction, layers) if return_latents else prediction
