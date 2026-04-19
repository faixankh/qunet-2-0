from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from .models.qunet2 import QUNet2
from .models.losses import MultiTaskLoss
from .models.optim import build_optimizer
from .models.callbacks import EarlyStopping, CheckpointManager
from .models.scheduler import build_scheduler
from .evaluation.evaluator import evaluate_model
from .utils import ensure_dir, get_device, seed_everything
from .data import build_dataloaders

def train(config):
    seed_everything(config.seed)
    device = get_device(config.device)
    loaders = build_dataloaders(
        image_size=config.task.image_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        num_classes=config.task.num_classes,
    )
    model = QUNet2(
        num_classes=config.task.num_classes,
        base_channels=config.model.base_channels,
        bottleneck_dim=config.model.bottleneck_dim,
        transformer_heads=config.model.transformer_heads,
        transformer_layers=config.model.transformer_layers,
        dropout=config.model.dropout,
        deep_supervision=config.model.deep_supervision,
        use_cross_attention=config.model.use_cross_attention,
    ).to(device)

    criterion = MultiTaskLoss(
        dice_weight=config.loss.dice_weight,
        focal_weight=config.loss.focal_weight,
        cls_weight=config.loss.cls_weight,
        consistency_weight=config.loss.consistency_weight,
    )
    optimizer = build_optimizer(model, lr=config.training.lr, weight_decay=config.training.weight_decay)
    total_steps = max(1, len(loaders["train"]) * config.training.epochs)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=max(1, len(loaders["train"])))
    scaler = GradScaler(enabled=config.training.amp)
    early = EarlyStopping(patience=config.training.early_stop_patience)
    ckpt = CheckpointManager(config.training.checkpoint_dir)

    history = {"train_loss": [], "valid_dice": [], "valid_iou": [], "valid_acc": []}

    step = 0
    for epoch in range(config.training.epochs):
        model.train()
        total = 0.0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch+1}/{config.training.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config.training.amp):
                out = model(batch)
                loss_dict = criterion(out, batch)
                loss = loss_dict["loss"]
                if "deep_supervision" in out:
                    ds_loss = 0.0
                    for ds in out["deep_supervision"]:
                        ds_loss = ds_loss + torch.nn.functional.binary_cross_entropy_with_logits(ds, batch["mask"].float())
                    loss = loss + 0.15 * ds_loss / max(1, len(out["deep_supervision"]))
            scaler.scale(loss).backward()
            if config.training.grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach().cpu())
            step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()))
        history["train_loss"].append(total / max(1, len(loaders["train"])))
        metrics = evaluate_model(model, loaders["valid"], device)
        history["valid_dice"].append(metrics["dice"])
        history["valid_iou"].append(metrics["iou"])
        history["valid_acc"].append(metrics.get("acc", float("nan")))
        ckpt.save(model, optimizer, epoch + 1, metrics)
        early.step(1.0 - metrics["dice"])
        if early.should_stop:
            break
    return {"history": history, "final_metrics": metrics}
