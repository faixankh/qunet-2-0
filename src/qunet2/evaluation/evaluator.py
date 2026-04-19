from __future__ import annotations
import torch
from ..models.metrics import binary_dice, binary_iou, classification_metrics

def evaluate_model(model, loader, device):
    model.eval()
    seg_dices, seg_ious = [], []
    cls_logits_all, cls_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(batch)
            seg_dices.append(binary_dice(out["seg_logits"], batch["mask"]))
            seg_ious.append(binary_iou(out["seg_logits"], batch["mask"]))
            cls_logits_all.append(out["cls_logits"].cpu())
            cls_all.append(batch["cls"].cpu())
    cls_metrics = classification_metrics(torch.cat(cls_logits_all), torch.cat(cls_all))
    return {
        "dice": float(sum(seg_dices) / max(1, len(seg_dices))),
        "iou": float(sum(seg_ious) / max(1, len(seg_ious))),
        **cls_metrics,
    }
