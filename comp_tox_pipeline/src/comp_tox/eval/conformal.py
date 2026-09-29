"""Split-conformal prediction sets for the tox endpoint.

Nonconformity score: 1 - p(true class). The threshold is calibrated on
validation compounds and evaluated on held-out test scaffolds. Coverage is
marginal; it is additionally reported conditioned on the applicability-domain
flag, where degradation is expected and quantified instead of hidden.
"""

from __future__ import annotations

import numpy as np


def nonconformity(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 - predicted probability of the true class. probs is (n, 2)."""
    return 1.0 - probs[np.arange(len(y)), y]


def conformal_threshold(scores: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample (1-alpha) quantile of calibration scores."""
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    if n == 0:
        raise ValueError("empty calibration set, check split.frac_valid")
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean (n, 2) set membership: classes whose score clears qhat."""
    return probs >= (1.0 - qhat)
