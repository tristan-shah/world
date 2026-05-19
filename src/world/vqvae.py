from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb

from world.vit import PatchEmbed, TransformerBlock


class VectorQuantizer(nn.Module):
    """Straight-through vector quantizer with codebook and commitment losses."""

    def __init__(self, num_codes: int, codebook_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
        self.commitment_weight = commitment_weight

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, N, D)
        B, N, D = z.shape
        flat = z.reshape(-1, D)  # (B*N, D)

        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * (flat @ self.codebook.weight.t())
            + self.codebook.weight.pow(2).sum(1)
        )  # (B*N, num_codes)

        indices = dist.argmin(dim=1)  # (B*N,)
        quantized = self.codebook(indices).reshape(B, N, D)

        codebook_loss = nn.functional.mse_loss(quantized, z.detach())
        commitment_loss = nn.functional.mse_loss(z, quantized.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # straight-through estimator: copy gradients from quantized to z
        quantized_st = z + (quantized - z).detach()

        return quantized_st, indices.reshape(B, N), vq_loss


class VQVAEEncoder(nn.Module):
    """ViT-based encoder: patches an image and processes with TransformerBlocks."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        codebook_dim: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, codebook_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed  # (B, N, embed_dim)
        x = self.blocks(x)
        x = self.norm(x)
        return self.proj(x)  # (B, N, codebook_dim)


class VQVAEDecoder(nn.Module):
    """ViT-based decoder: processes quantized tokens and folds back to pixel space."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        codebook_dim: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.grid_size = img_size // patch_size

        self.proj_in = nn.Linear(codebook_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # each token reconstructs one patch
        self.proj_out = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: (B, N, codebook_dim)
        B = z_q.shape[0]
        x = self.proj_in(z_q) + self.pos_embed  # (B, N, embed_dim)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.proj_out(x)  # (B, N, patch_size^2 * C)

        # fold patches back to image: (B, H, W, C) -> (B, C, H, W)
        g = self.grid_size
        p = self.patch_size
        C = self.in_channels
        x = x.reshape(B, g, g, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, g, p, g, p)
        x = x.reshape(B, C, g * p, g * p)
        return x


class VQVAE(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        codebook_dim: int = 256,
        num_codes: int = 1024,
        commitment_weight: float = 0.25,
        lr: float = 1e-4,
        warmup_steps: int = 1000,
        log_images_every_n_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = VQVAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            codebook_dim=codebook_dim,
        )
        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            codebook_dim=codebook_dim,
            commitment_weight=commitment_weight,
        )
        self.decoder = VQVAEDecoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            codebook_dim=codebook_dim,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, indices, vq_loss

    def _step(self, batch) -> dict[str, torch.Tensor]:
        x, _ = batch
        x_recon, _, vq_loss = self(x)
        recon_loss = nn.functional.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss
        return {"loss": loss, "recon_loss": recon_loss, "vq_loss": vq_loss}

    def training_step(self, batch, batch_idx):
        metrics = self._step(batch)
        self.log_dict(
            {"train/" + k: v for k, v in metrics.items()},
            on_step=True, on_epoch=True, prog_bar=True,
        )
        if self.global_step % self.hparams.log_images_every_n_steps == 0:
            self._log_images(batch)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._step(batch)
        self.log_dict(
            {"val/" + k: v for k, v in metrics.items()},
            on_epoch=True, prog_bar=True,
        )

    def on_validation_epoch_end(self):
        # log a fixed reconstruction at end of each val epoch
        pass

    @torch.no_grad()
    def _log_images(self, batch, n: int = 8):
        x, _ = batch
        x = x[:n]
        x_recon, _, _ = self(x)
        imgs = torch.cat([x, x_recon], dim=0)
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        grid = [
            wandb.Image(img.permute(1, 2, 0).cpu().float().numpy())
            for img in imgs
        ]
        self.logger.experiment.log(
            {"reconstructions": grid, "global_step": self.global_step}
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps - warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
