"""
Minimal Vision Transformer encoder for TransUNet (Phase 5).

This is an original, self-contained implementation (standard pre-norm ViT
blocks) — NOT copied from the official TransUNet repository — so there is no
third-party license to carry. It operates on tokens projected from a CNN
feature map (hybrid R50-ViT design).
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, mlp_dim, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_dim, dropout_rate) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class PatchEmbedding(nn.Module):
    """Project a CNN feature map (B×C×h×w) into transformer tokens + positional embedding.

    The hybrid design uses a 1×1 projection over the already-downsampled CNN grid,
    so the number of tokens equals the CNN feature grid (h*w).
    """

    def __init__(self, in_channels: int, hidden_size: int, n_patches: int, dropout_rate: float = 0.0):
        super().__init__()
        self.n_patches = n_patches
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, feat):
        x = self.proj(feat)                 # B, hidden, h, w
        x = x.flatten(2).transpose(1, 2)    # B, h*w, hidden
        if x.shape[1] != self.n_patches:
            raise ValueError(
                f"TransUNet token grid mismatch: got {x.shape[1]} patches but the "
                f"positional embedding expects {self.n_patches}. Ensure the input "
                f"spatial size matches img_size and that img_size/patch_size equals "
                f"the CNN stride-16 grid."
            )
        return self.drop(x + self.pos_embed)
