"""Build the serving reference profile: training-set fingerprints plus
the scalar baselines drift checks compare against.

The reference is a derived artifact (single npz) so the service never
re-reads pipeline intermediates. Baselines come from the committed
evaluation: the eval set's AD statistics define "looks like the eval
distribution", and the validation-calibrated qhat drives conformal sets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


def build_reference(
    splits_path: str,
    features_path: str,
    metrics_path: str,
    model_path: str,
    out_path: str,
) -> dict:
    df = pd.read_parquet(splits_path)
    X = sparse.load_npz(features_path).tocsr()
    metrics = json.load(open(metrics_path))
    bundle = joblib.load(model_path)

    train_X = X[df["split"].to_numpy() == "train"]
    primary = metrics["primary_model"]
    m = metrics["models"][primary]
    ad = m["applicability_domain"]

    scalars = {
        "primary_model": primary,
        "fingerprint": bundle.get("fingerprint"),
        "endpoint": metrics["endpoint"],
        "qhat": m["conformal"]["qhat"],
        "conformal_alpha": m["conformal"]["alpha"],
        "ad_threshold": ad["threshold"],
        "train_prevalence": float(df.loc[df["split"] == "train", "label"].mean()),
        # eval-set baselines: what an on-distribution batch looked like
        "baseline_nn_median": ad["median_nn_distance"],
        "baseline_in_domain_fraction": ad["in_domain_fraction"],
        "baseline_pos_rate": None,  # filled below
        "train_rows": int(train_X.shape[0]),
        "features_sha256": bundle.get("features_sha256"),
    }
    Xb = train_X.astype(bool)
    train_freq = np.asarray(Xb.mean(axis=0)).ravel()
    te = X[df["split"].to_numpy() == "test"].astype(bool)
    scalars["baseline_pos_rate"] = float(
        np.asarray(bundle["models"][primary].predict_proba(te))[:, 1].mean()
    )
    # eval set's own bit-frequency divergence from train — the floor a
    # same-distribution batch scores above only by sampling noise
    from scipy.spatial.distance import jensenshannon

    scalars["baseline_js"] = float(
        jensenshannon(train_freq, np.asarray(te.mean(axis=0)).ravel(), base=2.0) ** 2
    )

    np.savez_compressed(
        out_path,
        X_data=Xb.data,
        X_indices=Xb.indices,
        X_indptr=Xb.indptr,
        X_shape=np.asarray(Xb.shape),
        bit_freq=train_freq,
        scalars=np.asarray([json.dumps(scalars)]),
    )
    print(f"serve_reference: {train_X.shape[0]} train fps -> {out_path}")
    return scalars


def load_reference(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    ref = {
        "train_X": sparse.csr_matrix(
            (z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"])
        ),
        "bit_freq": z["bit_freq"],
    }
    ref.update(json.loads(str(z["scalars"][0])))
    return ref


def main() -> None:
    build_reference(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


if __name__ == "__main__":
    main()
