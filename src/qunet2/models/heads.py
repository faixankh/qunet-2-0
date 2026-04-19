from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch // 2, out_ch, 1),
        )
    def forward(self, x):
        return self.head(x)

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_classes),
        )
    def forward(self, x):
        return self.head(x)

class UncertaintyHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.mean = nn.Linear(in_dim, num_classes)
        self.log_var = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.mean(x), torch.clamp(self.log_var(x), -5.0, 3.0)

class DeepSupervisionHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 1, 1),
        )
    def forward(self, x):
        return self.proj(x)
