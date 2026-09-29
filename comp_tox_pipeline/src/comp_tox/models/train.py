"""Train the endpoint models on the train split; calibrate on validation.

Baselines: logistic regression and random forest on Morgan fingerprints,
each Platt-calibrated on the validation split via FrozenEstimator. The
calibrator never sees training or test data. All configured models are
trained on identical scaffold splits so comparison is apples-to-apples.

Persisted bundle: {name: calibrated model} dict + metadata.
"""

from __future__ import annotations

import sys

import hashlib
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from comp_tox.util import load_config

MODEL_REGISTRY = {
    # lbfgs is deterministic; random_state recorded anyway for reproducibility
    "logistic_regression": lambda seed: LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs",
        random_state=seed,
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=seed
    ),
}


def train(
    splits_path: str, features_path: str, graphs_path: str, out_path: str
) -> None:
    cfg = load_config()
    seed = cfg["split"]["seed"]
    names = [cfg["model"]["type"]] + cfg["model"].get("compare", [])

    df = pd.read_parquet(splits_path)
    X = sparse.load_npz(features_path)
    y = df["label"].to_numpy()

    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "valid").to_numpy()

    models, inputs = {}, {}
    for name in dict.fromkeys(names):
        if name == "gnn":
            import torch

            from comp_tox.models.gnn import train_gnn

            graphs = torch.load(graphs_path, weights_only=False)
            models[name] = train_gnn(graphs, df, tr, va, seed)
            inputs[name] = "graphs"
        elif name == "mlp_tf":
            from comp_tox.models.mlp_tf import train_mlp_tf

            base = train_mlp_tf(X[tr], y[tr], seed)
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(base), method="sigmoid"
            )
            calibrated.fit(X[va], y[va])
            models[name] = calibrated
            inputs[name] = "npz"
        else:
            if name not in MODEL_REGISTRY:
                raise ValueError(
                    f"unknown model {name!r} (registry: {sorted(MODEL_REGISTRY)},"
                    " 'gnn', 'mlp_tf')"
                )
            base = MODEL_REGISTRY[name](seed)
            base.fit(X[tr], y[tr])
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(base), method="sigmoid"
            )
            calibrated.fit(X[va], y[va])
            models[name] = calibrated
            inputs[name] = "npz"
        print(f"trained {name}")

    bundle = {
        "models": models,
        "inputs": inputs,
        "primary": cfg["model"]["type"],
        "fingerprint": cfg["model"].get("fingerprint"),
        "features_sha256": hashlib.sha256(
            Path(features_path).read_bytes()
        ).hexdigest(),
        "splits_sha256": hashlib.sha256(
            Path(splits_path).read_bytes()
        ).hexdigest(),
        "train_rows": int(tr.sum()),
        "valid_rows": int(va.sum()),
        "train_actives": int(y[tr].sum()),
        "valid_actives": int(y[va].sum()),
    }
    joblib.dump(bundle, out_path)
    print(
        f"train: {tr.sum()} train / {va.sum()} valid rows "
        f"({y[tr].sum()} train actives), models={list(models)} -> {out_path}"
    )


def main() -> None:
    splits_path, features_path, graphs_path, out_path = sys.argv[1:5]
    train(splits_path, features_path, graphs_path, out_path)


if __name__ == "__main__":
    main()
