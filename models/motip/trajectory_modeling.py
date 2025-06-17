# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch.nn as nn
import torch
import math

from models.ffn import FFN


class XYWHPositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, exchange_xy=True):
        super(XYWHPositionEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi
        self.exchange_xy = exchange_xy

    def forward(self, pos_tensor) -> torch.Tensor:
        dim_i = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
        dim_i = self.temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / self.num_pos_feats)

        orig_shape = pos_tensor.shape[:-1]  # e.g., [1, 6, 30, 12]
        pos_tensor = pos_tensor.view(-1, 4)  # flatten all but last dim

        pos_tensor = (pos_tensor * self.scale)[:, :, None] / dim_i[None, None, :]
        pos_tensor = pos_tensor.split([1] * pos_tensor.shape[1], dim=1)
        pos_res = [torch.stack((x[:, 0, 0::2].sin(), x[:, 0, 1::2].cos()), dim=2).flatten(1) for x in pos_tensor]

        if self.exchange_xy:
            pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
        pos_embed = torch.cat(pos_res, dim=1)

        pos_embed = pos_embed.view(*orig_shape, -1)  # reshape back
        return pos_embed

class SimpleFeat(nn.Module):
    """Conv → Conv → GAP → FC  (≈ 30 k 参数)"""

    def __init__(self, in_ch: int = 3, input_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),  # (S/2)
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),     # (S/4)
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)                                    # (1×1)
        )
        self.head = nn.Linear(64, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, 3, self.dim, self.dim)
        x = self.backbone(x)
        x = torch.flatten(x, 1)          # (N, 64)
        x = self.head(x)                 # (N, 256)
        return self.norm(x.view(*shape[:-1], -1))

class Fusion(nn.Module):
    def __init__(self, input_dim: int, output_dim):
        super().__init__()
        self.img_proj = SimpleFeat(input_dim=input_dim, output_dim=output_dim)
        self.bbox_proj = nn.Sequential(
            XYWHPositionEmbedding(output_dim // 4),
            nn.Linear(output_dim, output_dim),
            nn.GELU()
        )
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, f, b):
        feature = self.img_proj(f)
        bbox = self.bbox_proj(b)
        h = feature + bbox
        return self.feature_fusion(h)

class TrajectoryModeling(nn.Module):
    def __init__(
            self,
            detr_dim: int,
            ffn_dim_ratio: int,
            feature_dim: int,
            n_grid: int,
    ):
        super().__init__()

        self.detr_dim = detr_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.feature_dim = feature_dim

        self.trajectory_features_fusion = Fusion(n_grid, feature_dim)
        self.unknown_feature_fusion = Fusion(n_grid, feature_dim)

        self.adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        pass

    def forward(self, seq_info):
        trajectory_features, trajectory_boxes = seq_info["trajectory_features"], seq_info["trajectory_boxes"]
        trajectory_features = self.trajectory_features_fusion(trajectory_features, trajectory_boxes)
        trajectory_features = trajectory_features + self.adapter(trajectory_features)
        trajectory_features = self.norm(trajectory_features)
        trajectory_features = trajectory_features + self.ffn(trajectory_features)
        trajectory_features = self.ffn_norm(trajectory_features)
        seq_info["trajectory_features"] = trajectory_features

        unknown_features, unknown_boxes = seq_info["unknown_features"], seq_info["unknown_boxes"]
        unknown_features = self.unknown_feature_fusion(unknown_features, unknown_boxes)
        seq_info["unknown_features"] = unknown_features

        return seq_info
