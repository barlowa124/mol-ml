"""Evaluate on held-out scaffolds and produce the report artifacts.

Orchestrates the eval submodules (test split only):
    - eval.calibration: ECE + reliability curve -> results/calibration.png
    - eval.conformal: split-conformal coverage at 90%, calibrated on the
      validation split
    - eval.applicability: Tanimoto NN distance to train set; metrics reported
      conditioned on in/out of domain
    - AUROC, AUPRC (report both; imbalance makes AUPRC the informative one)

All metrics get bootstrap CIs over scaffold groups (not over rows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from comp_tox.eval.applicability import in_domain, nn_tanimoto_distances
from comp_tox.eval.calibration import expected_calibration_error, reliability_curve
from comp_tox.eval.conformal import (
    conformal_threshold,
    nonconformity,
    prediction_sets,
)
from comp_tox.util import load_config


def _metrics_at(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    return out


def _scaffold_bootstrap(
    df: pd.DataFrame, p: np.ndarray, n_boot: int, seed: int
) -> dict:
    """CIs by resampling scaffold groups, preserving group structure."""
    groups = df["scaffold_id"].to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    y = df["label"].to_numpy()
    rng = np.random.default_rng(seed)
    aurocs, auprcs = [], []
    group_rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    for _ in range(n_boot):
        if not group_rows:
            break
        draw = rng.choice(len(uniq), size=len(uniq), replace=True)
        rows = np.concatenate([group_rows[i] for i in draw])
        if len(np.unique(y[rows])) < 2:
            continue
        aurocs.append(roc_auc_score(y[rows], p[rows]))
        auprcs.append(average_precision_score(y[rows], p[rows]))
    if not aurocs:
        return {"auroc_ci95": [None, None], "auprc_ci95": [None, None], "n_boot_used": 0}
    return {
        "auroc_ci95": [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))],
        "auprc_ci95": [float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))],
        "n_boot_used": len(aurocs),
    }


def evaluate(
    model_path: str,
    splits_path: str,
    features_path: str,
    graphs_path: str,
    metrics_out: str,
    cal_out: str,
) -> None:
    cfg = load_config()
    ad_threshold = cfg["evaluation"]["ad_threshold"]
    n_boot = cfg["evaluation"]["n_bootstrap"]
    alpha = cfg["evaluation"].get("conformal_alpha", 0.1)
    seed = cfg["split"]["seed"]

    bundle = joblib.load(model_path)
    models = bundle["models"]
    inputs = bundle["inputs"]
    primary = bundle["primary"]
    df = pd.read_parquet(splits_path)
    X = sparse.load_npz(features_path)

    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "valid").to_numpy()
    te = (df["split"] == "test").to_numpy()
    y = df["label"].to_numpy()

    graphs = None
    if "graphs" in inputs.values():
        import torch

        graphs = torch.load(graphs_path, weights_only=False)

    def X_sel(kind: str, mask: np.ndarray):
        if kind == "graphs":
            return [graphs[i] for i in np.where(mask)[0]]
        return X[mask]

    # Applicability domain is measured in fingerprint space; the flag is
    # data-side. Each model is then scored conditioned on it.
    nn_dist = nn_tanimoto_distances(X[te], X[tr])
    ad = in_domain(nn_dist, ad_threshold)

    def eval_one(model, kind: str) -> dict:
        p_va = model.predict_proba(X_sel(kind, va))[:, 1]
        p_te = model.predict_proba(X_sel(kind, te))[:, 1]

        m = _metrics_at(y[te], p_te)
        m.update(
            _scaffold_bootstrap(
                df[te].reset_index(drop=True), p_te, n_boot, seed
            )
        )
        m["ece"] = expected_calibration_error(y[te], p_te)

        probs_va = np.column_stack([1 - p_va, p_va])
        probs_te = np.column_stack([1 - p_te, p_te])
        qhat = conformal_threshold(nonconformity(probs_va, y[va]), alpha=alpha)
        sets = prediction_sets(probs_te, qhat)
        covered = sets[np.arange(te.sum()), y[te]]
        m["conformal"] = {
            "alpha": alpha,
            "qhat": qhat,
            "coverage": float(covered.mean()),
            "mean_set_size": float(sets.sum(axis=1).mean()),
            "singleton_fraction": float((sets.sum(axis=1) == 1).mean()),
        }
        m["applicability_domain"] = {
            "threshold": ad_threshold,
            "in_domain_fraction": float(ad.mean()),
            "median_nn_distance": float(np.median(nn_dist)),
            "in_domain": _metrics_at(y[te][ad], p_te[ad]),
            "out_domain": _metrics_at(y[te][~ad], p_te[~ad])
            if (~ad).any()
            else {},
            "conformal_coverage_in_domain": float(covered[ad].mean())
            if ad.any()
            else None,
            "conformal_coverage_out_domain": float(covered[~ad].mean())
            if (~ad).any()
            else None,
        }
        return m, p_te

    model_metrics, test_probs = {}, {}
    for name, model in models.items():
        model_metrics[name], test_probs[name] = eval_one(model, inputs[name])
    metrics = {
        "endpoint": cfg["endpoint"]["assay_id"],
        "primary_model": primary,
        "counts": {
            "train": int(tr.sum()),
            "valid": int(va.sum()),
            "test": int(te.sum()),
            "train_actives": int(y[tr].sum()),
            "train_prevalence": float(y[tr].mean()),
            "valid_actives": int(y[va].sum()),
            "test_actives": int(y[te].sum()),
            "test_prevalence": float(y[te].mean()),
        },
        "models": model_metrics,
        # headline block duplicates the primary model's metrics for
        # convenience; "primary_model" records which entry it is
        **model_metrics[primary],
    }
    p_te = test_probs[primary]

    # Reliability curve figure
    centers, accs, counts = reliability_curve(y[te], p_te)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(centers, accs, "o-", label="model")
    ax.set_xlabel("predicted probability (bin center)")
    ax.set_ylabel("observed active fraction")
    ax.set_title(f"Tox21 {cfg['endpoint']['assay_id']} test scaffolds")
    ax.legend()
    fig.tight_layout()
    Path(cal_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cal_out, dpi=150)
    plt.close(fig)

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(metrics_out).with_name(
        f"{Path(metrics_out).stem}_predictions.parquet")
    predictions = df.loc[te, ["compound_id", "scaffold_id", "label"]].reset_index(drop=True)
    for name, probabilities in test_probs.items():
        predictions[name] = probabilities
    predictions.to_parquet(predictions_path, index=False)
    metrics["bootstrap"] = {
        "status": "computed",
        "method": "scaffold_cluster_with_replacement",
        "seed": seed,
        "n_bootstrap": n_boot,
        "predictions": predictions_path.name,
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    }
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("splits")
    parser.add_argument("features")
    parser.add_argument("graphs")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--calibration", required=True)
    args = parser.parse_args()
    evaluate(
        args.model,
        args.splits,
        args.features,
        args.graphs,
        args.metrics,
        args.calibration,
    )


if __name__ == "__main__":
    main()
