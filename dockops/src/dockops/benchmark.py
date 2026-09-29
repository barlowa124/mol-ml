"""Benchmark evaluation: ROC-AUC and enrichment factor over actives/decoys.

Docking scores are more-negative-is-better, so scores are negated before
ranking (higher rank value = predicted active). Enrichment factor at
fraction f: (actives in top f / f·N) / overall active rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dockops.util import load_config


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """scores: docking energies (lower = better). labels: 1=active, 0=decoy."""
    return float(roc_auc_score(labels, -np.asarray(scores, dtype=float)))


def enrichment_factor(
    scores: np.ndarray, labels: np.ndarray, fraction: float = 0.01
) -> float | None:
    """EF at top `fraction` of ranked list (lower score = ranked earlier)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n = len(scores)
    k = max(1, int(np.ceil(fraction * n)))
    top = labels[np.argsort(scores)[:k]]
    prevalence = labels.mean()
    if prevalence == 0:
        return None
    return float((top.sum() / k) / prevalence)


def evaluate(scores_csv: str, ligands_csv: str, metrics_out: str) -> dict:
    cfg = load_config()
    label_col = cfg["benchmark"]["label_col"]

    scores_df = pd.read_csv(scores_csv)
    ligands = pd.read_csv(ligands_csv)
    df = scores_df.merge(
        ligands[["compound_id", label_col]], on="compound_id", how="left"
    )
    ok = df[(df["status"] == "ok") & df[label_col].notna()].copy()
    y = ok[label_col].astype(int).to_numpy()
    s = ok["score"].astype(float).to_numpy()

    engines = ok["engine"].unique()
    engine = engines[0] if len(engines) == 1 else "mixed"
    metrics = {
        "engine": engine,
        "mock_mode": engine == "mock",
        "n_scored": int(len(ok)),
        "n_actives": int(y.sum()),
        "roc_auc": roc_auc(s, y) if len(np.unique(y)) > 1 else None,
        "enrichment": {
            f"ef@{int(f * 100)}%": enrichment_factor(s, y, f)
            for f in cfg["benchmark"]["ef_fractions"]
        },
        "note": "mock engine, mechanics demo, not a docking result"
        if engine == "mock"
        else "real backend",
    }
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    scores_csv, ligands_csv, metrics_out = sys.argv[1:4]
    evaluate(scores_csv, ligands_csv, metrics_out)


if __name__ == "__main__":
    main()
