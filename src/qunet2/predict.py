from __future__ import annotations
import torch
from PIL import Image
import numpy as np

def predict_image(model, image_path: str, device: str = "cpu"):
    model = model.to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.asarray(image).astype("float32") / 255.0
    image_t = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).to(device)
    oct_t = torch.zeros(1, 1, 112, 112, device=device)
    with torch.no_grad():
        out = model({"image": image_t, "oct": oct_t})
    return out
