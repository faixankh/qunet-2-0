from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class EarlyStopping:
    patience: int = 5
    best_score: float = float("inf")
    bad_epochs: int = 0
    should_stop: bool = False

    def step(self, score: float):
        if score < self.best_score:
            self.best_score = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True

@dataclass
class CheckpointManager:
    directory: str
    best_score: float = float("inf")

    def save(self, model, optimizer, epoch: int, metrics: dict, name: str = "best.pt"):
        path = Path(self.directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        }, path / name)

def update_ema(ema_model, model, decay: float = 0.999):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
