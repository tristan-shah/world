import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from world.dit import TimestepEmbedding

t = torch.linspace(0, 1, 100)

embed = TimestepEmbedding(embed_dim=256, freq_dim=256)
with torch.no_grad():
    embs = embed._sinusoidal(t)  # (100, 256) — raw sinusoidals before MLP

plt.figure(figsize=(10, 4))
plt.imshow(embs.T, aspect="auto", origin="lower", extent=[0, 1, 0, 256])
plt.colorbar(label="value")
plt.xlabel("t")
plt.ylabel("dimension")
plt.title("Sinusoidal timestep embeddings")
plt.tight_layout()
plt.show()
