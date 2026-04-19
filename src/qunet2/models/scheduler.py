from __future__ import annotations
import torch

def build_scheduler(optimizer, total_steps: int, warmup_steps: int = 50):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-3, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
