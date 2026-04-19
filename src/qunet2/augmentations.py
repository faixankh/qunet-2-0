from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

@dataclass
class AugmentationConfig:
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.15
    brightness: float = 0.12
    contrast: float = 0.12
    saturation: float = 0.10

class MedicalAugment:
    def __init__(self, cfg: AugmentationConfig | None = None):
        self.cfg = cfg or AugmentationConfig()

    def __call__(self, image: Image.Image, mask: Image.Image | None = None):
        if random.random() < self.cfg.horizontal_flip_prob:
            image = ImageOps.mirror(image)
            if mask is not None:
                mask = ImageOps.mirror(mask)
        if random.random() < self.cfg.vertical_flip_prob:
            image = ImageOps.flip(image)
            if mask is not None:
                mask = ImageOps.flip(mask)
        if random.random() < self.cfg.brightness:
            image = ImageEnhance.Brightness(image).enhance(0.85 + random.random() * 0.3)
        if random.random() < self.cfg.contrast:
            image = ImageEnhance.Contrast(image).enhance(0.85 + random.random() * 0.3)
        if random.random() < self.cfg.saturation:
            image = ImageEnhance.Color(image).enhance(0.85 + random.random() * 0.3)
        return image, mask
