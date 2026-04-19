from __future__ import annotations
from pathlib import Path
from PIL import Image
import numpy as np
import random

def make_dataset(root: str = "data/synthetic", n: int = 64, image_size: int = 224):
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (image_size, image_size), (20, 20, 20))
        mask = Image.new("L", (image_size, image_size), 0)
        arr = np.asarray(img)
        # simple synthetic lesion generation can be added here
        img.save(root / "images" / f"{i:04d}.png")
        mask.save(root / "masks" / f"{i:04d}.png")

if __name__ == "__main__":
    make_dataset()
