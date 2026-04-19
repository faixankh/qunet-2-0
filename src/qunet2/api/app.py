from __future__ import annotations
from typing import Any
import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
from ..models.qunet2 import QUNet2

class PredictResponse(BaseModel):
    seg_shape: tuple[int, ...]
    cls_logits: list[float]

def create_app(model: QUNet2 | None = None) -> FastAPI:
    app = FastAPI(title="QUNet 2.0 API")
    model = model or QUNet2()
    model.eval()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
        arr = np.asarray(image).astype("float32") / 255.0
        arr = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
        batch = {"image": arr, "oct": torch.zeros(1, 1, 112, 112)}
        with torch.no_grad():
            out = model(batch)
        return {
            "seg_shape": list(out["seg_logits"].shape),
            "cls_logits": out["cls_logits"].squeeze(0).tolist(),
            "mean_logits": out["mean_logits"].squeeze(0).tolist(),
            "log_var_logits": out["log_var_logits"].squeeze(0).tolist(),
        }

    return app
