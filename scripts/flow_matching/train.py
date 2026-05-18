import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("high")

from world.data import MushroomDataModule
from world.flow_matching import FlowMatching

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mushroom"


def parse_args():
    parser = argparse.ArgumentParser(description="Flow matching DiT training")

    # data
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    # model
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)

    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--sample_every_n_steps", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # wandb
    parser.add_argument("--wandb_project", type=str, default="mushroom-flow-matching")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dm = MushroomDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    model = FlowMatching(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        sample_every_n_steps=args.sample_every_n_steps,
    )

    logger = WandbLogger(project=args.wandb_project, log_model=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
    )

    trainer.fit(model, dm)
