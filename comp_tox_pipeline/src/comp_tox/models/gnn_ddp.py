"""Distributed training mechanics: the GIN under torch.distributed DDP.

Launches `world` real worker processes via `torch.multiprocessing.spawn`,
each running `comp_tox.models.ddp_main.spawn_worker` with a tcp://
loopback rendezvous on the gloo backend, with separate processes, partitioned
data, gradient all-reduce via DistributedDataParallel. The code path is
identical to multi-GPU (nccl); only the backend and scale differ.

The spawn launcher is used instead of `torch.distributed.run` because the
elastic agent does not spawn workers reliably on macOS (observed: agent
listens on the store socket but no worker processes ever start; Python
3.13). `comp_tox.models.ddp_main.main` remains usable under torchrun on
platforms where the agent works.

Verified on CPU. DDP mechanics demonstrated, multi-GPU
scaling untested.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile

import numpy as np


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def train_ddp(graphs, y: np.ndarray, world: int = 2, epochs: int = 5,
              seed: int = 0) -> dict:
    """Spawn `world` DDP workers; return rank-0 results."""
    import torch
    import torch.multiprocessing as mp

    from comp_tox.models.ddp_main import spawn_worker

    with tempfile.TemporaryDirectory() as td:
        data_path = os.path.join(td, "ddp_in.pt")
        out_path = os.path.join(td, "ddp_out.json")
        torch.save(
            {"graphs": graphs, "y": np.asarray(y, dtype=np.float32)},
            data_path,
        )
        mp.spawn(
            spawn_worker,
            args=(world, data_path, out_path, epochs, seed, _free_port()),
            nprocs=world,
            join=True,
        )
        with open(out_path) as f:
            return json.load(f)
