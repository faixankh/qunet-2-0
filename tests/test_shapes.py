from qunet2.models.qunet2 import QUNet2
import torch

def test_forward_shapes():
    model = QUNet2()
    batch = {"image": torch.randn(2, 3, 224, 224), "oct": torch.randn(2, 1, 112, 112)}
    out = model(batch)
    assert out["seg_logits"].shape == (2, 1, 224, 224)
    assert out["cls_logits"].shape == (2, 3)
    assert out["mean_logits"].shape == (2, 3)
    assert out["log_var_logits"].shape == (2, 3)
