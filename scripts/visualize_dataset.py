"""Plot a random sample of images from the mushroom dataset."""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from world.data import MushroomDataset  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "mushroom"
SAMPLE_SIZE = 16
COLS = 4


def main():
    dataset = MushroomDataset(csv_path=DATA_DIR / "train.csv", data_root=DATA_DIR)

    indices = random.sample(range(len(dataset.paths)), SAMPLE_SIZE)
    rows = SAMPLE_SIZE // COLS

    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3, rows * 3))
    fig.suptitle("Random mushroom samples", fontsize=14)

    for ax, idx in zip(axes.flat, indices):
        img = Image.open(DATA_DIR / dataset.paths[idx]).convert("RGB")
        label = dataset.classes[dataset.labels[idx]]
        ax.imshow(img)
        ax.set_title(label, fontsize=7, wrap=True)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
