from __future__ import annotations
import torch
import torch.nn.functional as F

def saliency_map(model, image: torch.Tensor, target_index: int | None = None):
    model.eval()
    image = image.requires_grad_(True)
    outputs = model({"image": image, "oct": torch.zeros(image.shape[0], 1, image.shape[2]//2, image.shape[3]//2, device=image.device)})
    logits = outputs["cls_logits"]
    if target_index is None:
        target_index = logits.argmax(dim=1).item()
    score = logits[:, target_index].sum()
    score.backward()
    saliency = image.grad.abs().amax(dim=1, keepdim=True)
    saliency = saliency / (saliency.max().clamp_min(1e-6))
    return saliency.detach()
