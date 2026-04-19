from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from .augmentations import MedicalAugment
from .utils import ensure_dir

@dataclass
class Sample:
    image: torch.Tensor
    mask: torch.Tensor
    cls: torch.Tensor
    oct_map: torch.Tensor
    meta: dict[str, Any]

def _to_tensor_rgb(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size))
    arr = np.asarray(image).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def _to_tensor_mask(mask: Image.Image, size: int) -> torch.Tensor:
    mask = mask.resize((size, size))
    arr = np.asarray(mask).astype("float32")
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 127).astype("float32")
    return torch.from_numpy(arr[None, ...])

class SyntheticRetinaDataset(Dataset):
    def __init__(self, length: int = 128, image_size: int = 224, num_classes: int = 3, use_aug: bool = True):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = MedicalAugment() if use_aug else None

    def __len__(self):
        return self.length

    def _make_image(self, idx: int):
        size = self.image_size
        image = Image.new("RGB", (size, size), color=(20, 20, 20))
        draw = ImageDraw.Draw(image)
        cls = idx % self.num_classes
        # synthetic optic disc and vessel-like geometry
        draw.ellipse((size*0.56, size*0.18, size*0.86, size*0.48), fill=(230, 190, 90))
        for y in range(24, size, 18):
            draw.line((size*0.48, y, size*0.95, y + (y % 7) - 3), fill=(120, 90, 40), width=1)
        # lesions
        lesion_count = 5 + (idx % 7)
        for j in range(lesion_count):
            x = int((idx * 37 + j * 53) % (size - 20)) + 10
            y = int((idx * 19 + j * 71) % (size - 20)) + 10
            r = 2 + (j % 4)
            color = (255, 40, 40) if cls == 0 else (255, 230, 120)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
        return image, cls

    def _make_mask(self, image: Image.Image, idx: int):
        size = self.image_size
        mask = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(mask)
        lesion_count = 5 + (idx % 7)
        for j in range(lesion_count):
            x = int((idx * 37 + j * 53) % (size - 20)) + 10
            y = int((idx * 19 + j * 71) % (size - 20)) + 10
            r = 2 + (j % 4)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
        return mask

    def _make_oct(self, idx: int):
        size = self.image_size // 2
        grid = np.zeros((1, size, size), dtype="float32")
        yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
        band = np.exp(-((yy + 0.1*np.sin(idx))**2 * 18 + xx**2 * 2.0))
        grid[0] = band
        return torch.from_numpy(grid)

    def __getitem__(self, idx: int):
        image, cls = self._make_image(idx)
        mask = self._make_mask(image, idx)
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        return {
            "image": _to_tensor_rgb(image, self.image_size),
            "mask": _to_tensor_mask(mask, self.image_size),
            "cls": torch.tensor(cls, dtype=torch.long),
            "oct": self._make_oct(idx),
            "meta": {"index": idx, "source": "synthetic"}
        }

class FolderRetinaDataset(Dataset):
    """
    Placeholder for real datasets.
    Expected layout:
        images/*.png or *.jpg
        masks/*.png or *.jpg
        optional oct/*.npy
    """
    def __init__(self, root: str | Path, image_size: int = 224):
        self.root = Path(root)
        self.image_size = image_size
        self.images = sorted((self.root / "images").glob("*"))
        self.masks = sorted((self.root / "masks").glob("*"))
        self.oct = sorted((self.root / "oct").glob("*"))
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.root / 'images'}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L") if idx < len(self.masks) else Image.new("L", image.size)
        image_t = _to_tensor_rgb(image, self.image_size)
        mask_t = _to_tensor_mask(mask, self.image_size)
        cls = torch.tensor(int(mask_t.sum() > 0), dtype=torch.long)
        oct_t = torch.zeros((1, self.image_size//2, self.image_size//2), dtype=torch.float32)
        return {"image": image_t, "mask": mask_t, "cls": cls, "oct": oct_t, "meta": {"index": idx, "source": str(self.root)}}

def build_dataloaders(
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    train_length: int = 96,
    valid_length: int = 24,
    num_classes: int = 3,
):
    train_ds = SyntheticRetinaDataset(length=train_length, image_size=image_size, num_classes=num_classes, use_aug=True)
    valid_ds = SyntheticRetinaDataset(length=valid_length, image_size=image_size, num_classes=num_classes, use_aug=False)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "valid": DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
