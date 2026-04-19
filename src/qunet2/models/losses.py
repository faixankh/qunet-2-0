from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred) if pred.dtype.is_floating_point else pred.float()
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1).float()
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return ((2 * inter + eps) / (denom + eps)).mean()

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1).float()
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()

def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    prob = torch.sigmoid(logits)
    p_t = target * prob + (1 - target) * (1 - prob)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    return loss.mean()

def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    target = target.float()
    tp = (pred * target).sum(dim=(1,2,3))
    fp = (pred * (1 - target)).sum(dim=(1,2,3))
    fn = ((1 - pred) * target).sum(dim=(1,2,3))
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score.mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 0.75, cls_weight: float = 0.65, consistency_weight: float = 0.1):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.cls_weight = cls_weight
        self.consistency_weight = consistency_weight

    def forward(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        seg_logits = outputs["seg_logits"]
        cls_logits = outputs["cls_logits"]
        mask = batch["mask"]
        cls = batch["cls"]
        seg_dice = dice_loss(seg_logits, mask)
        seg_focal = focal_loss(seg_logits, mask)
        cls_loss = F.cross_entropy(cls_logits, cls)
        pred_area = torch.sigmoid(seg_logits).mean(dim=(1,2,3))
        cls_prob = torch.softmax(cls_logits, dim=1).max(dim=1).values
        consistency = F.mse_loss(pred_area, cls_prob)
        total = self.dice_weight * seg_dice + self.focal_weight * seg_focal + self.cls_weight * cls_loss + self.consistency_weight * consistency
        return {
            "loss": total,
            "seg_dice": seg_dice.detach(),
            "seg_focal": seg_focal.detach(),
            "cls_loss": cls_loss.detach(),
            "consistency": consistency.detach(),
        }
