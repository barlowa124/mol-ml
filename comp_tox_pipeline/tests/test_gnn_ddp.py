import numpy as np
import pytest

torch = pytest.importorskip("torch")

from comp_tox.features.graphs import smiles_to_graph
from comp_tox.models.gnn_ddp import train_ddp


def _graphs():
    smiles = [
        "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "O", "C",
        "CCN", "CCC", "COC", "CCCl", "CCBr",
    ] * 4
    return [smiles_to_graph(s) for s in smiles]


@pytest.mark.slow
def test_ddp_two_processes_train_and_predict():
    graphs = _graphs()
    y = np.array([0, 1] * 20, dtype=np.float32)
    result = train_ddp(graphs, y, world=2, epochs=2)
    logits = np.array(result["logits"])
    assert logits.shape == (len(graphs),)
    assert np.isfinite(logits).all()
    assert result["final_shard_loss"] >= 0
