"""Splitters. Primary eval is cold-target: held-out proteins never appear
in training: the prospective estimate for "predict affinity for a protein we
haven't screened." Random split is reported as secondary (it leaks target
identity and is expected to look better).
"""

import numpy as np


def cold_target_split(target_ids: np.ndarray, frac: float, seed: int):
    """Hold out all pairs whose protein is in a held-out target set."""
    rng = np.random.default_rng(seed)
    targets = np.unique(target_ids)
    n_test = max(1, int(round(len(targets) * frac)))
    test_targets = set(rng.choice(targets, size=n_test, replace=False))
    te = np.array([t in test_targets for t in target_ids])
    return np.where(~te)[0], np.where(te)[0]


def random_split(n: int, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    te = rng.random(n) < frac
    return np.where(~te)[0], np.where(te)[0]
