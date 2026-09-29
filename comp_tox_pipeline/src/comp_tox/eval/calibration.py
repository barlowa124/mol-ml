"""Probability calibration measurement and plotting."""

from __future__ import annotations

import numpy as np


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin centers, observed accuracy, bin counts) for equal-width bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    centers, accs, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            centers.append((edges[b] + edges[b + 1]) / 2)
            accs.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))
    return np.asarray(centers), np.asarray(accs), np.asarray(counts)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """ECE: count-weighted mean |accuracy - confidence| over equal-width bins."""
    centers, accs, counts = reliability_curve(y_true, y_prob, n_bins)
    if len(counts) == 0:
        return float("nan")
    return float(np.average(np.abs(accs - centers), weights=counts))
