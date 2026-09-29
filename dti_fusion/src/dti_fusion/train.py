"""Train fusion + modality-ablation models; evaluate on the cold-target
split (primary) and a random split (secondary, labeled as leaky).

Writes results/summary.json with per-model metrics and bootstrap CIs,
results/scatter.png, and results/provenance.json.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dti_fusion.config import config_path, load_config
from dti_fusion.model import metrics, mse, train_model
from dti_fusion.provenance import write_manifest
from dti_fusion.splits import cold_target_split, random_split


def bootstrap_ci(y_true, y_pred, fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        j = rng.choice(len(y_true), size=len(y_true), replace=True)
        vals.append(fn(y_true[j], y_pred[j]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run_split(Xd, xp, y, tr, te, cfg, n_boot, seed, label):
    out, preds = {}, {}
    for mode in ("fusion", "drug", "protein"):
        preds[mode] = train_model(mode, Xd, xp, y, tr, te, cfg["model"], seed)
        m = metrics(y[te], preds[mode])
        m["mse_ci"] = bootstrap_ci(y[te], preds[mode], mse, n_boot, seed)
        out[mode] = m
        print(f"  {label}/{mode}: mse {m['mse']:.3f} spearman {m['spearman']:.3f}")
    return out, preds["fusion"]


def main(feat_npz: str, out_json: str, out_png: str):
    cfg = load_config()
    d = np.load(feat_npz, allow_pickle=True)
    Xd, xp, y = d["X_drug"], d["X_prot"], d["pkd"].astype(np.float32)
    targets = d["target_id"]

    tr_c, te_c = cold_target_split(
        targets, cfg["split"]["cold_target_frac"], cfg["split"]["seed"]
    )
    tr_r, te_r = random_split(len(y), cfg["split"]["cold_target_frac"],
                              cfg["split"]["seed"])

    print(f"cold-target split: {len(tr_c)} train / {len(te_c)} test "
          f"({len(np.unique(targets[te_c]))} held-out proteins)")
    cold_metrics, cold_pred = run_split(
        Xd, xp, y, tr_c, te_c, cfg,
        cfg["evaluation"]["n_boot"], cfg["model"]["seed"], "cold",
    )
    result = {
        "primary_split": "cold_target",
        "n_pairs": int(len(y)),
        "n_targets_held_out": int(len(np.unique(targets[te_c]))),
        "frac_at_cap": float(d["frac_at_cap"]) if "frac_at_cap" in d else None,
        "frac_truncated": (
            float(d["frac_truncated"]) if "frac_truncated" in d else None
        ),
        "cold_target": cold_metrics,
        "random_split_fusion": None,
    }
    # secondary: random split, fusion only, reported as the leaky variant
    pred_r = train_model("fusion", Xd, xp, y, tr_r, te_r, cfg["model"],
                         cfg["model"]["seed"])
    result["random_split_fusion"] = metrics(y[te_r], pred_r)

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(y[te_c], cold_pred, s=6, alpha=0.4)
    ax.set_xlabel("measured pKd")
    ax.set_ylabel("predicted pKd")
    ax.set_title("DAVIS cold-target holdout (fusion)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)

    write_manifest("results/provenance.json", inputs=[feat_npz], config_path=str(config_path()))
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
