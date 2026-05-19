from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from diffusers import AutoencoderKL
from torch_ema import ExponentialMovingAverage

from world.dit import DiT

NUM_SAMPLE_IMAGES = 8
EULER_STEPS = 50
VAE_SCALE = 0.18215  # standard SD VAE latent scaling factor


class FlowMatching(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        lr: float = 1e-4,
        warmup_steps: int = 1000,
        ema_decay: float = 0.9999,
        sample_every_n_steps: int = 500,
        use_vae: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        if use_vae:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            self.vae.requires_grad_(False)
            dit_channels = 4
            dit_img_size = img_size // 8  # VAE compresses 8x spatially
        else:
            dit_channels = in_channels
            dit_img_size = img_size

        self.model = DiT(
            img_size=dit_img_size,
            patch_size=patch_size,
            in_channels=dit_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)

    def on_fit_start(self):
        self.ema.to(self.device)
        self._latents_precomputed = self.trainer.datamodule.latents_precomputed

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.sample() * VAE_SCALE

    @torch.no_grad()
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / VAE_SCALE).sample

    @torch.no_grad()
    def sample(self, n: int, steps: int = EULER_STEPS) -> torch.Tensor:
        """Euler integration from x0 ~ N(0, I) to x1 ~ data using EMA weights."""
        h = self.hparams
        if h.use_vae:
            latent_size = h.img_size // 8
            x = torch.randn(n, 4, latent_size, latent_size, device=self.device)
        else:
            x = torch.randn(n, h.in_channels, h.img_size, h.img_size, device=self.device)

        dt = 1.0 / steps
        with self.ema.average_parameters():
            for i in range(steps):
                t = torch.full((n,), i * dt, device=self.device)
                x = x + self.model(x, t) * dt

        if h.use_vae:
            x = self._decode(x)
        return x

    def _flow_loss(self, batch) -> torch.Tensor:
        x1, _ = batch
        if self.hparams.use_vae and not getattr(self, "_latents_precomputed", False):
            x1 = self._encode(x1)
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
        if self.global_step % self.hparams.sample_every_n_steps == 0:
            self._log_images()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._flow_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self._log_images()

    def _log_images(self):
        images = self.sample(NUM_SAMPLE_IMAGES)
        images = (images * 0.5 + 0.5).clamp(0, 1)
        grid = [
            wandb.Image(img.permute(1, 2, 0).cpu().float().numpy())
            for img in images
        ]
        self.logger.experiment.log(
            {"generated_images": grid, "global_step": self.global_step}
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

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
