"""JAX/Flax port of the PyTorch GIN baseline, verified to numerical parity.

The Flax module replicates torch_geometric's GINConv math exactly:
    h_i' = MLP(x_i + sum_{j in N(i)} x_j)
then global mean pool + linear head, identical to models.gnn.GINNet.

`torch_to_flax_params` maps a trained PyTorch state_dict into the Flax
parameter tree; `parity_check` runs both models on the same graphs and
returns the max absolute logit difference. Cross-framework agreement is
the evidence that the port is correct.
"""

from __future__ import annotations

import numpy as np


def _flax_modules():
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    class GINLayer(nn.Module):
        hidden: int

        @nn.compact
        def __call__(self, x, senders, receivers):
            agg = x + jax.ops.segment_sum(
                x[senders], receivers, x.shape[0]
            )
            h = nn.Dense(self.hidden, name="fc1")(agg)
            h = nn.relu(h)
            h = nn.Dense(self.hidden, name="fc2")(h)
            return nn.relu(h)

    class GINNetJax(nn.Module):
        hidden: int = 64

        @nn.compact
        def __call__(self, x, senders, receivers, batch, n_graphs):
            for i in range(3):
                x = GINLayer(self.hidden, name=f"gin{i}")(
                    x, senders, receivers
                )
            counts = jax.ops.segment_sum(
                jnp.ones(x.shape[0]), batch, n_graphs
            )
            pooled = jax.ops.segment_sum(x, batch, n_graphs) / jnp.maximum(
                counts[:, None], 1.0
            )
            return nn.Dense(1, name="head")(pooled).squeeze(-1)

    return GINNetJax


def torch_to_flax_params(state_dict: dict, hidden: int = 64) -> dict:
    """Map GINNet's PyTorch state_dict to the Flax param tree."""
    import jax.numpy as jnp

    params = {}
    for i in range(3):
        params[f"gin{i}"] = {
            "fc1": {
                "kernel": jnp.array(
                    state_dict[f"convs.{i}.nn.0.weight"].numpy().T
                ),
                "bias": jnp.array(state_dict[f"convs.{i}.nn.0.bias"].numpy()),
            },
            "fc2": {
                "kernel": jnp.array(
                    state_dict[f"convs.{i}.nn.2.weight"].numpy().T
                ),
                "bias": jnp.array(state_dict[f"convs.{i}.nn.2.bias"].numpy()),
            },
        }
    params["head"] = {
        "kernel": jnp.array(state_dict["head.weight"].numpy().T),
        "bias": jnp.array(state_dict["head.bias"].numpy()),
    }
    return {"params": params}


def predict_flax(params: dict, graphs, hidden: int = 64) -> np.ndarray:
    """Run the Flax port on a list of (x, edge_index, edge_attr) graphs."""
    import jax.numpy as jnp

    # Batch the graph list like torch_geometric's DataLoader: concatenate
    # nodes, offset edge indices, build a batch vector.
    xs, senders, receivers, batch_vec = [], [], [], []
    node_offset = 0
    for gi, (x, edge_index, _attr) in enumerate(graphs):
        xs.append(jnp.asarray(x.numpy()))
        senders.append(jnp.asarray(edge_index[0].numpy()) + node_offset)
        receivers.append(jnp.asarray(edge_index[1].numpy()) + node_offset)
        batch_vec.append(jnp.full(x.shape[0], gi))
        node_offset += x.shape[0]
    net = _flax_modules()(hidden)
    logits = net.apply(
        params,
        jnp.concatenate(xs),
        jnp.concatenate(senders) if senders else jnp.array([], dtype=jnp.int32),
        jnp.concatenate(receivers)
        if receivers
        else jnp.array([], dtype=jnp.int32),
        jnp.concatenate(batch_vec),
        len(graphs),
    )
    return np.asarray(logits)


def parity_check(torch_net, graphs, hidden: int = 64) -> float:
    """Max |logit diff| between the PyTorch and Flax models on `graphs`."""
    import torch

    params = torch_to_flax_params(torch_net.state_dict(), hidden)
    flax_logits = predict_flax(params, graphs, hidden)
    torch_net.eval()
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    data = [Data(x=g[0], edge_index=g[1], edge_attr=g[2]) for g in graphs]
    with torch.no_grad():
        torch_logits = np.concatenate(
            [
                torch_net(b.x, b.edge_index, b.batch).numpy()
                for b in DataLoader(data, batch_size=256)
            ]
        )
    return float(np.abs(flax_logits - torch_logits).max())
