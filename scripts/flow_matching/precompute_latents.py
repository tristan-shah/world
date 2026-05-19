"""Pre-encode dataset images into VAE latents and save to disk."""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from world.data import MushroomDataset

VAE_SCALE = 0.18215
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mushroom"


def precompute_split(split, data_dir, img_size, batch_size, num_workers, vae, device):
    latents_path = data_dir / f"{split}_latents.npy"
    labels_path = data_dir / f"{split}_labels.npy"

    if latents_path.exists():
        print(f"{split}: latents already exist at {latents_path}, skipping.")
        return

    dataset = MushroomDataset(
        csv_path=data_dir / f"{split}.csv",
        data_root=data_dir,
        img_size=img_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    N = len(dataset)
    latent_size = img_size // 8
    latents = np.memmap(latents_path, dtype=np.float16, mode="w+", shape=(N, 4, latent_size, latent_size))
    labels = np.zeros(N, dtype=np.int64)

    idx = 0
    for images, batch_labels in loader:
        images = images.to(device)
        with torch.no_grad():
            encoded = vae.encode(images).latent_dist.sample() * VAE_SCALE
        latents[idx : idx + len(images)] = encoded.cpu().float().numpy().astype(np.float16)
        labels[idx : idx + len(images)] = batch_labels.numpy()
        idx += len(images)
        print(f"  {split}: {idx}/{N}", end="\r")

    latents.flush()
    np.save(labels_path, labels)
    print(f"\n{split}: saved to {latents_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-encode dataset images into VAE latents")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    for split in args.splits:
        print(f"\nEncoding {split} split...")
        precompute_split(
            split=split,
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            vae=vae,
            device=device,
        )

    print("\nDone. Run training with --use_vae to use precomputed latents.")
