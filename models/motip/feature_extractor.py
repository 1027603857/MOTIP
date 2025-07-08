"""
Custom U‑Net‑like network with unconventional skip connections.
--------------------------------------------------------------
Encoder
  • Input  :  N × 64 × 64 × 3
  • Conv   :  3  → 16 channels (k=3, s=1)
  • Down‑sampling ×4 (stride‑2 conv) so that channels: 16→32→64→128→256
    Spatial size halves each step (64→32→16→8→4).

Decoder ("right side")
  • Starts from an all‑zero latent vector  z₀ ∈ ℝ^{N×256}.
  • For each encoder feature map (coarsest → finest):
      1. Global average pool over spatial dims →  N×C.
      2. Linear (FC) layer  C → 256  (acts like a 1×1 "deconv").
      3. Accumulate into latent vector:  z ← z + f(GAP).
  • Returns the final latent vector  z  and (optionally) an image‑sized map.

This demo shows the core idea; you can extend the decoder to a full image
reconstruction head if needed.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------  Encoder blocks  ------------------------- #
class ConvBNReLU(nn.Sequential):
    """3×3 convolution + BN + ReLU"""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class DownBlock(nn.Module):
    """A single encoder stage: conv → conv → (optional) downsample."""

    def __init__(self, in_c: int, out_c: int, downsample: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_c, out_c),
            ConvBNReLU(out_c, out_c),
        )
        self.pool = nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1, bias=False) if downsample else None

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            return self.pool(x), x  # return down‑sampled, and pre‑pooled for skip
        return x, x


# -------------------------  Decoder blocks  ------------------------- #
class nmODEBlocks(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.g = nn.Sequential(
            nn.ReLU()
        )

        self.sigma = nn.Sequential(
            nn.LayerNorm(in_c),
        )

    def forward(self, x, y, delta=0.2):
        y_next = (1 - delta) * y + delta * self.sigma(y + self.g(x))
        return y_next


# -------------------------  Model  ------------------------- #
class nmUNet(nn.Module):
    def __init__(self, chs):
        super().__init__()
        # Channel progression 3→16→32→64→128→256
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(1, len(chs)):
            # last block has no further down‑sampling
            self.encoders.append(DownBlock(chs[i - 1], chs[i], downsample=(i < len(chs) - 1)))
            self.decoders.append(nmODEBlocks(chs[-1]))
        # For each skip connection produce a FC layer  C→256
        self.fc_skips = nn.ModuleList([
            nn.Linear(c, 256) for c in chs[1:]  # skip the 1st element (input channels)
        ])

    def forward(self, x):
        skips: List[torch.Tensor] = []
        for enc in self.encoders:
            x, feat = enc(x)
            skips.append(feat)
        # x is the final 4×4×256 feature map

        # Latent vector initialised to zeros
        z = torch.zeros(x.size(0), 256, device=x.device, dtype=x.dtype)

        # traverse skip maps from coarse→fine (reverse list)
        for feat, fc, dec in zip(reversed(skips), self.fc_skips[::-1], self.decoders):
            # Global average pool: N×C
            gap = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # N×C
            z = dec(fc(gap), z)

        return z  # (optionally, add a decoder head here)

class FeatureExtractor(nn.Module):
    def __init__(
            self,
            channels,
    ):
        super().__init__()
        self.nmUNet = nmUNet(channels)

    def forward(self, annotations):
        _B, _T = len(annotations), len(annotations[0])
        _N = annotations[0][0]["trajectory_id_labels"].shape[-1]
        _, _C, _H, _W = annotations[0][0]["feature"].shape
        _device = annotations[0][0]["feature"].device
        objects = torch.zeros((_B, _T, _N, _C, _H, _W), dtype=torch.float32, device=_device)
        for b in range(_B):
            for t in range(_T):
                feat = annotations[b][t]["feature"]
                n_t = feat.size(0)
                objects[b, t, :n_t] = feat

        feature_embed = objects.flatten(0, -4)
        feature_embed = self.nmUNet(feature_embed)
        feature_embed = feature_embed.reshape(_B, _T, _N, -1)

        return feature_embed