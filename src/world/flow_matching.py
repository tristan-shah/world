from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb

from world.dit import DiT

NUM_SAMPLE_IMAGES = 8
EULER_STEPS = 50


class FlowMatching(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DiT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    @torch.no_grad()
    def sample(self, n: int, steps: int = EULER_STEPS) -> torch.Tensor:
        """Euler integration from x0 ~ N(0, I) to x1 ~ data."""
        h = self.hparams
        x = torch.randn(n, h.in_channels, h.img_size, h.img_size, device=self.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=self.device)
            x = x + self.model(x, t) * dt
        return x

    def _flow_loss(self, batch) -> torch.Tensor:
        x1, _ = batch
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=self.device)
        t_bc = t.reshape(B, 1, 1, 1)
        x_t = (1 - t_bc) * x0 + t_bc * x1
        u_t = x1 - x0
        v = self.model(x_t, t)
        return nn.functional.mse_loss(v, u_t)

    def training_step(self, batch, batch_idx):
        loss = self._flow_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._flow_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        images = self.sample(NUM_SAMPLE_IMAGES)
        # denormalize from [-1, 1] to [0, 1] and clamp
        images = (images * 0.5 + 0.5).clamp(0, 1)
        grid = [
            wandb.Image(img.permute(1, 2, 0).cpu().float().numpy())
            for img in images
        ]
        self.logger.experiment.log(
            {"generated_images": grid, "epoch": self.current_epoch}
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
