import numpy as np
import torch

from dti_fusion.model import BranchMLP, metrics


def test_branch_shapes():
    for mode in ("fusion", "drug", "protein"):
        m = BranchMLP(d_dim=64, p_dim=32, hidden=16, dropout=0.0, mode=mode)
        out = m(torch.randn(5, 64), torch.randn(5, 32))
        assert out.shape == (5,)


def test_metrics_on_toy():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = metrics(y, y.copy())
    assert perfect["mse"] == 0.0 and perfect["pearson"] == 1.0
    flat = metrics(y, np.full(4, 2.5))
    assert flat["mse"] > 0


def test_fusion_uses_both_modalities():
    # a model in "fusion" mode should change output when protein input
    # changes even with identical drug input
    torch.manual_seed(0)
    m = BranchMLP(8, 8, 16, 0.0, "fusion").eval()
    xd = torch.ones(2, 8)
    xp1 = torch.ones(2, 8)
    xp2 = -torch.ones(2, 8)
    assert not torch.allclose(m(xd, xp1), m(xd, xp2))
