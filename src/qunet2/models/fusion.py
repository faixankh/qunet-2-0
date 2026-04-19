from __future__ import annotations
from torch import nn
import torch
import torch.nn.functional as F
from .encoders import CrossAttentionFusion

class FeaturePyramidFusion(nn.Module):
    def __init__(self, channels: list[int], bottleneck_dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, bottleneck_dim, 1) for c in channels])
        self.fuse = CrossAttentionFusion(bottleneck_dim, heads=heads, dropout=dropout)

    def forward(self, feats: list[torch.Tensor]):
        projected = []
        token_maps = []
        base_size = feats[0].shape[-2:]
        for i, f in enumerate(feats):
            p = self.proj[i](f)
            if p.shape[-2:] != base_size:
                p = F.interpolate(p, size=base_size, mode="bilinear", align_corners=False)
            projected.append(p)
            token_maps.append(p.flatten(2).transpose(1, 2))
        context = torch.cat(token_maps, dim=1)
        query = token_maps[-1]
        fused_tokens = self.fuse(query, context)
        fused_map = fused_tokens.transpose(1, 2).reshape(feats[0].shape[0], -1, base_size[0], base_size[1])
        return fused_map, projected
