from qunet2.models.metrics import classification_metrics
import torch

def test_classification_metrics_runs():
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 1, 2, 1])
    metrics = classification_metrics(logits, target)
    assert "acc" in metrics
