"""Batch drift check: compare an incoming batch of compounds against the
frozen training/eval reference before trusting predictions on it.

Signals (all cheap, all measured):
- nearest-neighbor Tanimoto distance of the batch vs. train reference,
  compared against the eval set's median
- in-domain fraction at the configured AD threshold vs. the eval set's
- Jensen-Shannon divergence between batch and train bit frequencies
- predicted-positive rate vs. the eval set's

A batch reports ``status: "drift_warning"`` when any check breaches its
configured margin. Thresholds live in config.serve.drift; this is a
screening signal for when measured metrics stop applying, not a
deployment gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import jensenshannon

from comp_tox.eval.applicability import in_domain, nn_tanimoto_distances
from comp_tox.serve.predictor import Predictor
from comp_tox.serve.reference import load_reference
from comp_tox.util import load_config


def _js(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric JS divergence between two bit-frequency vectors."""
    return float(jensenshannon(p, q, base=2.0) ** 2)


def check_drift(
    batch_X: sparse.spmatrix,
    probs: np.ndarray,
    ref: dict,
    margins: dict | None = None,
) -> dict:
    margins = margins or {}
    nn = nn_tanimoto_distances(batch_X, ref["train_X"])
    dom = in_domain(nn, ref["ad_threshold"])
    batch_freq = np.asarray(batch_X.astype(bool).mean(axis=0)).ravel()

    report = {
        "n_batch": int(batch_X.shape[0]),
        "nn_distance_median": float(np.median(nn)),
        "nn_distance_p90": float(np.quantile(nn, 0.9)),
        "in_domain_fraction": float(dom.mean()),
        "bitfreq_js_divergence": _js(ref["bit_freq"], batch_freq),
        "predicted_positive_rate": float(probs.mean()),
        "baseline": {
            "nn_distance_median": ref["baseline_nn_median"],
            "in_domain_fraction": ref["baseline_in_domain_fraction"],
            "positive_rate": ref["baseline_pos_rate"],
        },
        "warnings": [],
    }

    nn_margin = margins.get("nn_shift_margin", 0.10)
    if report["nn_distance_median"] > ref["baseline_nn_median"] + nn_margin:
        report["warnings"].append(
            f"median nn-distance {report['nn_distance_median']:.3f} exceeds "
            f"eval baseline {ref['baseline_nn_median']:.3f} by >{nn_margin}"
        )
    min_dom = margins.get("in_domain_min_fraction", 0.05)
    if report["in_domain_fraction"] < min_dom:
        report["warnings"].append(
            f"in-domain fraction {report['in_domain_fraction']:.3f} below "
            f"minimum {min_dom}"
        )
    js_margin = margins.get("js_margin", 0.15)
    report["baseline"]["js_divergence"] = ref["baseline_js"]
    if report["bitfreq_js_divergence"] > ref["baseline_js"] + js_margin:
        report["warnings"].append(
            f"bit-frequency JS divergence {report['bitfreq_js_divergence']:.3f} "
            f"exceeds eval baseline {ref['baseline_js']:.3f} by >{js_margin}"
        )
    if ref["baseline_pos_rate"] is not None:
        pos_margin = margins.get("pos_rate_shift", 0.05)
        if abs(report["predicted_positive_rate"] - ref["baseline_pos_rate"]) > pos_margin:
            report["warnings"].append(
                f"predicted positive rate {report['predicted_positive_rate']:.3f} "
                f"shifted from eval baseline {ref['baseline_pos_rate']:.3f} "
                f"by >{pos_margin}"
            )
    report["status"] = "drift_warning" if report["warnings"] else "ok"
    return report


def main() -> None:
    """python -m comp_tox.serve.drift batch.parquet model.joblib reference.npz out.json

    batch.parquet needs a ``smiles`` or ``canonical_smiles`` column.
    """
    batch_path, model_path, ref_path, out_path = sys.argv[1:5]
    cfg = load_config()
    df = pd.read_parquet(batch_path)
    col = "smiles" if "smiles" in df.columns else "canonical_smiles"
    pred = Predictor(model_path, ref_path)
    X, _, keep = pred.fingerprints(df[col].tolist())
    probs = np.asarray(pred.model.predict_proba(X))[:, 1]
    report = check_drift(
        X, probs, pred.ref, margins=cfg["serve"].get("drift", {})
    )
    report["batch"] = str(Path(batch_path).name)
    report["n_unparseable"] = int(len(df) - len(keep))
    Path(out_path).write_text(json.dumps(report, indent=1))
    print(f"drift: {report['status']} ({len(report['warnings'])} warnings) -> {out_path}")


if __name__ == "__main__":
    main()
