from __future__ import annotations
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def binary_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    return float((inter + eps) / (union + eps))

def binary_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return float((2 * inter + eps) / (denom + eps))

def classification_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
    pred = prob.argmax(axis=1)
    y = target.detach().cpu().numpy()
    out = {
        "acc": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
    }
    try:
        if prob.shape[1] > 2:
            out["auc_ovr"] = float(roc_auc_score(y, prob, multi_class="ovr"))
        else:
            out["auc"] = float(roc_auc_score(y, prob[:, 1]))
    except Exception:
        out["auc"] = float("nan")
    return out
