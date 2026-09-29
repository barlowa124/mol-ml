"""Applicability domain via Tanimoto nearest-neighbor distance.

A compound is in-domain iff its distance to the nearest training-set
neighbor (on Morgan fingerprints) is at or below the configured threshold.
The threshold is calibrated on validation scaffolds, see
config.evaluation.ad_threshold.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def nn_tanimoto_distances(
    test_X: sparse.spmatrix, train_X: sparse.spmatrix, block: int = 512
) -> np.ndarray:
    """Distance from each test row to its nearest train row (boolean Tanimoto)."""
    t = test_X.astype(bool).tocsr()
    r = train_X.astype(bool).tocsr()
    t_nnz = np.asarray(t.sum(axis=1)).ravel()
    r_nnz = np.asarray(r.sum(axis=1)).ravel()

    best = np.empty(t.shape[0])
    for lo in range(0, t.shape[0], block):
        hi = min(lo + block, t.shape[0])
        inter = np.asarray(t[lo:hi].astype(np.int32) @ r.T.astype(np.int32).todense())
        union = t_nnz[lo:hi, None] + r_nnz[None, :] - inter
        sim = np.divide(inter, union, out=np.ones_like(inter, dtype=float), where=union > 0)
        best[lo:hi] = 1.0 - sim.max(axis=1)
    return best


def in_domain(nn_distances: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask: True where a compound is inside the applicability domain."""
    return nn_distances <= threshold
