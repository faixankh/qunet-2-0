from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class TaskConfig:
    image_size: int = 224
    num_classes: int = 3
    lesion_channels: int = 4
    use_oct: bool = True

@dataclass
class ModelConfig:
    base_channels: int = 32
    bottleneck_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 2
    dropout: float = 0.15
    deep_supervision: bool = True
    use_cross_attention: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    amp: bool = False
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    checkpoint_dir: str = "outputs/checkpoints"

@dataclass
class LossConfig:
    dice_weight: float = 1.0
    focal_weight: float = 0.75
    cls_weight: float = 0.65
    consistency_weight: float = 0.1

@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls(
            seed=data.get("seed", 42),
            device=data.get("device", "auto"),
            task=TaskConfig(**data.get("task", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            loss=LossConfig(**data.get("loss", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "device": self.device,
            "task": self.task.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "loss": self.loss.__dict__,
        }
