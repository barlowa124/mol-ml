import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("jax")
pytest.importorskip("flax")

from comp_tox.features.graphs import smiles_to_graph
from comp_tox.models.gnn import GINNet
from comp_tox.models.gnn_jax import parity_check, predict_flax, torch_to_flax_params


def _graphs():
    return [
        smiles_to_graph(s)
        for s in ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "O", "C"]
    ]


def test_flax_parity_random_init():
    torch.manual_seed(0)
    net = GINNet()
    graphs = _graphs()
    diff = parity_check(net, graphs)
    assert diff < 1e-4, f"flax/torch logits diverge by {diff}"


def test_predict_flax_shape():
    torch.manual_seed(0)
    net = GINNet()
    params = torch_to_flax_params(net.state_dict())
    logits = predict_flax(params, _graphs())
    assert logits.shape == (5,)
    assert np.isfinite(logits).all()
