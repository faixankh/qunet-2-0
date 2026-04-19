from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.no_grad()
def predict_with_tta(model, batch: dict, flips: bool = True):
    model.eval()
    variants = [batch["image"]]
    if flips:
        variants += [torch.flip(batch["image"], dims=[3]), torch.flip(batch["image"], dims=[2])]
    seg_logits_list = []
    cls_logits_list = []
    for image in variants:
        b = dict(batch)
        b["image"] = image
        out = model(b)
        seg_logits_list.append(out["seg_logits"])
        cls_logits_list.append(out["cls_logits"])
    seg_logits = torch.stack(seg_logits_list).mean(dim=0)
    cls_logits = torch.stack(cls_logits_list).mean(dim=0)
    return {"seg_logits": seg_logits, "cls_logits": cls_logits, "seg_prob": torch.sigmoid(seg_logits), "cls_prob": torch.softmax(cls_logits, dim=1)}
