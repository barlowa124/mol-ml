"""DDP worker, launched either by torchrun or by torch.multiprocessing.spawn.

Reads graphs + labels from a torch.save file, trains the GIN under
DistributedDataParallel on the gloo backend, and has rank 0 write
predictions + final loss to a JSON file.

Launch (torchrun path):
    DDP_DATA=... DDP_OUT=... \\
    python -m torch.distributed.run --nproc_per_node=2 \\
        -m comp_tox.models.ddp_main

The mp.spawn path (`spawn_worker`) is the one used on macOS: the
torchrun elastic agent does not spawn workers reliably there (Python
3.13), while spawn + a tcp:// init method works identically.
"""

from __future__ import annotations

import json
import os

import numpy as np


def _dbg(msg: str) -> None:
    if os.environ.get("DDP_DEBUG"):
        import sys

        print(
            f"[rank {os.environ.get('RANK', '?')} pid {os.getpid()}] {msg}",
            file=sys.stderr,
            flush=True,
        )


def _autograd_warmup() -> None:
    """Warm up the autograd engine while the process is still
    single-threaded. On macOS the first backward() probes Metal via
    MTLDeviceArrayInitialize, which segfaults if gloo/libuv threads
    are already running."""
    import torch

    _w = torch.nn.Linear(1, 1)
    _w(torch.ones(1, 1)).sum().backward()


def _run(data_path: str, out_path: str, epochs: int, seed: int) -> None:
    """Shared body: expects the process group to be initialized already."""
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from comp_tox.models.gnn import GINNet

    rank = dist.get_rank()
    world = dist.get_world_size()

    payload = torch.load(data_path, weights_only=False)
    graphs, y = payload["graphs"], np.asarray(payload["y"], np.float32)
    torch.manual_seed(seed)

    ddp = DistributedDataParallel(GINNet())
    opt = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    y_t = torch.tensor(y)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)])
    )

    my_idx = list(range(rank, len(graphs), world))
    data = [
        Data(
            x=graphs[i][0],
            edge_index=graphs[i][1],
            edge_attr=graphs[i][2],
            y=y_t[i : i + 1],
        )
        for i in my_idx
    ]
    last = 0.0
    for ep in range(epochs):
        _dbg(f"epoch {ep} start")
        total, n = 0.0, 0
        for batch in DataLoader(data, batch_size=32, shuffle=True):
            opt.zero_grad()
            loss = loss_fn(ddp(batch.x, batch.edge_index, batch.batch), batch.y)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        last = total / max(n, 1)
        _dbg(f"epoch {ep} done")

    if rank == 0:
        net = ddp.module
        net.eval()
        full = [Data(x=g[0], edge_index=g[1], edge_attr=g[2]) for g in graphs]
        with torch.no_grad():
            logits = np.concatenate(
                [
                    net(b.x, b.edge_index, b.batch).numpy()
                    for b in DataLoader(full, batch_size=256)
                ]
            )
        with open(out_path, "w") as f:
            json.dump(
                {
                    "world_size": world,
                    "final_shard_loss": last,
                    "logits": logits.tolist(),
                },
                f,
            )
    _dbg("before destroy barrier")
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """torchrun/env:// entry point."""
    import faulthandler

    faulthandler.enable()
    import torch.distributed as dist

    _autograd_warmup()
    _dbg("before init")
    dist.init_process_group("gloo")
    _dbg("after init")
    _run(
        os.environ["DDP_DATA"],
        os.environ["DDP_OUT"],
        int(os.environ.get("DDP_EPOCHS", "5")),
        int(os.environ.get("DDP_SEED", "0")),
    )


def spawn_worker(
    rank: int,
    world: int,
    data_path: str,
    out_path: str,
    epochs: int,
    seed: int,
    port: int,
) -> None:
    """torch.multiprocessing.spawn entry, tcp:// rendezvous on loopback."""
    import torch.distributed as dist

    os.environ["RANK"] = str(rank)
    _autograd_warmup()
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    _run(data_path, out_path, epochs, seed)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    _dbg("main returned")
