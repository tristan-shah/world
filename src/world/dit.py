from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding for continuous t ∈ [0, 1] followed by an MLP."""

    def __init__(self, embed_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        x = t[:, None] * 1000 * freqs[None]
        return torch.cat([x.cos(), x.sin()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class DiTBlock(nn.Module):
    """DiT block with AdaLN-Zero conditioning on timestep."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = Attention(embed_dim, num_heads)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        # produces (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.adaLN(t_emb).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(self.norm1(x) * (1 + s1.unsqueeze(1)) + b1.unsqueeze(1))
        x = x + g2.unsqueeze(1) * self.mlp(self.norm2(x) * (1 + s2.unsqueeze(1)) + b2.unsqueeze(1))
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for flow matching.

    Takes a noisy image x_t and continuous timestep t ∈ [0, 1] and predicts
    the velocity field v_θ(x_t, t) = x_1 - x_0.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.t_embed = TimestepEmbedding(embed_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        # final adaLN + linear to predict velocity per patch
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.final_proj = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        p = self.patch_size
        h, w = H // p, W // p
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(x.shape[0], self.in_channels, H, W)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        t_emb = self.t_embed(t)
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x, t_emb)
        scale, shift = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_proj(x)
        return self.unpatchify(x, H, W)
