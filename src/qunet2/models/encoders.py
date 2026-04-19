from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvNormAct(ch, ch, dropout=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + x)

class MultiScaleEncoder(nn.Module):
    """
    Returns 4 spatial stages and one global token map.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, c),
            ResidualBlock(c, dropout=dropout),
        )
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvNormAct(c, c*2, dropout=dropout), ResidualBlock(c*2, dropout=dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvNormAct(c*2, c*4, dropout=dropout), ResidualBlock(c*4, dropout=dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvNormAct(c*4, c*8, dropout=dropout), ResidualBlock(c*8, dropout=dropout))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvNormAct(c*8, c*8, dropout=dropout), ResidualBlock(c*8, dropout=dropout))
        self.out_channels = [c, c*2, c*4, c*8, c*8]

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        return [s1, s2, s3, s4, s5]

class OCTEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16, dropout: float = 0.1):
        super().__init__()
        self.encoder = MultiScaleEncoder(in_channels, base_channels, dropout=dropout)
        self.proj = nn.Conv2d(self.encoder.out_channels[-1], self.encoder.out_channels[-1], 1)

    def forward(self, x):
        feats = self.encoder(x)
        feats[-1] = self.proj(feats[-1])
        return feats

class Tokenizer(nn.Module):
    def __init__(self, in_ch: int, token_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.proj = nn.Linear(in_ch, token_dim)

    def forward(self, x):
        z = self.pool(x)
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        return self.proj(tokens)

class TransformerBottleneck(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        return self.norm(self.encoder(tokens))

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, q, kv):
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn)
        z = q + attn_out
        z = z + self.ff(self.norm_out(z))
        return z
