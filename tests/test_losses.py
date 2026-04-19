from qunet2.models.losses import dice_loss
import torch

def test_dice_loss_runs():
    logits = torch.randn(2, 1, 16, 16)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    loss = dice_loss(logits, target)
    assert loss >= 0
