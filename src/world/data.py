from __future__ import annotations

import zipfile
from pathlib import Path

import kaggle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

KAGGLE_DATASET = "zlatan599/mushroom1"


def make_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class MushroomDataset(Dataset):
    """Mushroom image classification dataset."""

    def __init__(self, csv_path: str | Path, data_root: str | Path, transform=None, img_size: int = 64):
        self.data_root = Path(data_root)
        self.transform = transform if transform is not None else make_transforms(img_size)
        df = pd.read_csv(csv_path)
        # CSV paths use the Kaggle prefix; strip it to get the relative image path.
        df["local_path"] = df["image_path"].str.replace(
            r"^/kaggle/working/", "", regex=True
        )
        self.paths = df["local_path"].tolist()
        classes = sorted(df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.class_to_idx[label] for label in df["label"]]
        self.classes = classes

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.data_root / self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


class MushroomLatentDataset(Dataset):
    """Loads pre-encoded VAE latents from disk."""

    def __init__(self, data_dir: Path, split: str, img_size: int):
        latent_size = img_size // 8
        csv_path = data_dir / f"{split}.csv"
        N = len(pd.read_csv(csv_path))
        self.latents = np.memmap(
            data_dir / f"{split}_latents.npy",
            dtype=np.float16, mode="r",
            shape=(N, 4, latent_size, latent_size),
        )
        self.labels = np.load(data_dir / f"{split}_labels.npy")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.latents[idx].astype(np.float32)), int(self.labels[idx])


class MushroomDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the mushroom dataset."""

    def __init__(
        self,
        data_dir: str | Path = "data/mushroom",
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 64,
        use_vae: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.use_vae = use_vae
        self.num_classes: int | None = None

    @property
    def latents_precomputed(self) -> bool:
        return self.use_vae and (self.data_dir / "train_latents.npy").exists()

    def _make_dataset(self, split: str) -> Dataset:
        if self.latents_precomputed:
            return MushroomLatentDataset(self.data_dir, split, self.img_size)
        return MushroomDataset(self.data_dir / f"{split}.csv", self.data_dir, img_size=self.img_size)

    def prepare_data(self):
        if not (self.data_dir / "train.csv").exists():
            print(f"Dataset not found at {self.data_dir}, downloading from Kaggle...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                KAGGLE_DATASET, path=self.data_dir, unzip=False
            )
            zip_path = self.data_dir / "mushroom1.zip"
            print(f"Unzipping {zip_path} — this may take several minutes...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.namelist()
                for i, member in enumerate(members, 1):
                    zf.extract(member, self.data_dir)
                    if i % 10000 == 0:
                        print(f"  {i}/{len(members)} files extracted...")
            zip_path.unlink()
            print("Download and extraction complete.")

    def setup(self, stage: str | None = None):
        if self.latents_precomputed:
            print("Using precomputed VAE latents.")
        if stage in ("fit", None):
            self.train = self._make_dataset("train")
            self.val = self._make_dataset("val")
            if hasattr(self.train, "classes"):
                self.num_classes = len(self.train.classes)
        if stage in ("test", None):
            self.test = self._make_dataset("test")

    def _dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test)
