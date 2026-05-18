from __future__ import annotations

import zipfile
from pathlib import Path

import kaggle
import pandas as pd
import pytorch_lightning as pl
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


class MushroomDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the mushroom dataset."""

    def __init__(
        self,
        data_dir: str | Path = "data/mushroom",
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 64,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.num_classes: int | None = None

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
        if stage in ("fit", None):
            self.train = MushroomDataset(
                self.data_dir / "train.csv",
                self.data_dir,
                img_size=self.img_size,
            )
            self.val = MushroomDataset(
                self.data_dir / "val.csv",
                self.data_dir,
                img_size=self.img_size,
            )
            self.num_classes = len(self.train.classes)
        if stage in ("test", None):
            self.test = MushroomDataset(
                self.data_dir / "test.csv",
                self.data_dir,
                img_size=self.img_size,
            )
            if self.num_classes is None:
                self.num_classes = len(self.test.classes)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
