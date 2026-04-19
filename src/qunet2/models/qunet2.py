from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .encoders import MultiScaleEncoder, OCTEncoder, TransformerBottleneck, Tokenizer
from .fusion import FeaturePyramidFusion
from .heads import SegmentationHead, ClassificationHead, UncertaintyHead, DeepSupervisionHead

class QUNet2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        oct_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        bottleneck_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.15,
        deep_supervision: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.rgb_encoder = MultiScaleEncoder(in_channels, base_channels, dropout=dropout)
        self.oct_encoder = OCTEncoder(oct_channels, max(base_channels // 2, 16), dropout=dropout)
        self.rgb_token = Tokenizer(self.rgb_encoder.out_channels[-1], bottleneck_dim)
        self.oct_token = Tokenizer(self.oct_encoder.encoder.out_channels[-1], bottleneck_dim)
        self.transformer = TransformerBottleneck(bottleneck_dim, transformer_heads, transformer_layers, dropout)
        self.use_cross_attention = use_cross_attention
        self.fusion = FeaturePyramidFusion(
            channels=self.rgb_encoder.out_channels,
            bottleneck_dim=bottleneck_dim,
            heads=transformer_heads,
            dropout=dropout,
        )
        self.seg_head = SegmentationHead(bottleneck_dim, 1)
        self.cls_head = ClassificationHead(bottleneck_dim, num_classes, dropout=dropout)
        self.uncertainty = UncertaintyHead(bottleneck_dim, num_classes)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                DeepSupervisionHead(base_channels),
                DeepSupervisionHead(base_channels * 2),
                DeepSupervisionHead(base_channels * 4),
                DeepSupervisionHead(base_channels * 8),
            ])
        else:
            self.ds_heads = nn.ModuleList([])

    def forward(self, batch: dict):
        image = batch["image"]
        oct_map = batch.get("oct", None)
        rgb_feats = self.rgb_encoder(image)
        oct_feats = self.oct_encoder(oct_map if oct_map is not None else torch.zeros(image.shape[0], 1, image.shape[2]//2, image.shape[3]//2, device=image.device))
        fused_map, projected = self.fusion(rgb_feats)
        tokens = self.rgb_token(rgb_feats[-1]) + self.oct_token(oct_feats[-1])
        tokens = self.transformer(tokens)
        token_summary = tokens.mean(dim=1)
        pooled = F.adaptive_avg_pool2d(fused_map, 1).flatten(1) + token_summary
        seg_logits = self.seg_head(fused_map)
        seg_logits = F.interpolate(seg_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cls_logits = self.cls_head(pooled)
        mean_logits, log_var_logits = self.uncertainty(pooled)
        outputs = {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
            "mean_logits": mean_logits,
            "log_var_logits": log_var_logits,
            "fused_map": fused_map,
        }
        if self.deep_supervision:
            ds = []
            for head, feat in zip(self.ds_heads, rgb_feats[:-1]):
                z = head(feat)
                z = F.interpolate(z, size=image.shape[-2:], mode="bilinear", align_corners=False)
                ds.append(z)
            outputs["deep_supervision"] = ds
        return outputs
