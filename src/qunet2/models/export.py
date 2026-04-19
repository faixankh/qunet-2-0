from __future__ import annotations
from pathlib import Path
import torch

def export_onnx(model, path: str | Path, image_size: int = 224):
    path = Path(path)
    model.eval()
    dummy = {
        "image": torch.randn(1, 3, image_size, image_size),
        "oct": torch.randn(1, 1, image_size // 2, image_size // 2),
    }
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["image", "oct"],
        output_names=["seg_logits", "cls_logits", "mean_logits", "log_var_logits"],
        dynamic_axes={"image": {0: "batch"}, "oct": {0: "batch"}},
        opset_version=17,
    )
