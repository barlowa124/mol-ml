"""Serving-layer tests on a tiny synthetic bundle — no real model needed."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from comp_tox.serve.drift import check_drift
from comp_tox.serve.predictor import Predictor
from comp_tox.serve.reference import build_reference


def _tiny_bundle_and_reference(tmp_path, n_train=80, fp_size=512, seed=0):
    rng = np.random.default_rng(seed)
    # sparse-ish train fingerprints; actives share a few high-weight bits
    X = sparse.csr_matrix(rng.binomial(1, 0.08, (n_train + 40, fp_size)))
    y = (X[:, :5].sum(axis=1).A.ravel() > 0).astype(int)
    tr, va, te = np.arange(n_train), np.arange(n_train, n_train + 20), np.arange(
        n_train + 20, n_train + 40
    )

    clf = LogisticRegression(max_iter=500).fit(X[tr], y[tr])
    bundle = {
        "models": {"logistic_regression": clf},
        "inputs": {"logistic_regression": "npz"},
        "primary": "logistic_regression",
        "fingerprint": "morgan-r2-512",
        "features_sha256": "f" * 64,
        "splits_sha256": "s" * 64,
        "train_rows": int(len(tr)),
        "valid_rows": int(len(va)),
        "train_actives": int(y[tr].sum()),
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(bundle, model_path)

    import pandas as pd

    df = pd.DataFrame(
        {
            "compound_id": [f"C{i}" for i in range(X.shape[0])],
            "canonical_smiles": ["CC"] * X.shape[0],
            "scaffold_id": ["s"] * X.shape[0],
            "label": y,
            "split": ["train"] * n_train + ["valid"] * 20 + ["test"] * 20,
        }
    )
    splits_path = tmp_path / "splits.parquet"
    df.to_parquet(splits_path)
    features_path = tmp_path / "features.npz"
    sparse.save_npz(features_path, X)

    import comp_tox.eval.conformal as conf

    probs_va = np.asarray(clf.predict_proba(X[va]))[:, 1]
    p2 = np.column_stack([1 - probs_va, probs_va])
    qhat = conf.conformal_threshold(conf.nonconformity(p2, y[va]), alpha=0.1)
    nn_med, in_frac = 0.5, 0.5
    metrics = {
        "endpoint": "TEST",
        "primary_model": "logistic_regression",
        "counts": {"train": n_train, "valid": 20, "test": 20},
        "models": {
            "logistic_regression": {
                "conformal": {"alpha": 0.1, "qhat": qhat},
                "applicability_domain": {
                    "threshold": 0.9,
                    "median_nn_distance": nn_med,
                    "in_domain_fraction": in_frac,
                },
            }
        },
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    ref_path = tmp_path / "ref.npz"
    build_reference(
        str(splits_path), str(features_path), str(metrics_path),
        str(model_path), str(ref_path),
    )
    return model_path, ref_path, bundle, X, y, tr


def test_reference_freezes_train_profile(tmp_path):
    _, ref_path, bundle, X, y, tr = _tiny_bundle_and_reference(tmp_path)
    from comp_tox.serve.reference import load_reference

    ref = load_reference(ref_path)
    assert ref["train_X"].shape == (len(tr), 512)
    assert ref["qhat"] > 0 and ref["ad_threshold"] == 0.9
    assert ref["primary_model"] == "logistic_regression"


def test_predict_emits_probability_conformal_and_ad(tmp_path):
    model_path, ref_path, *_ = _tiny_bundle_and_reference(tmp_path)
    pred = Predictor(str(model_path), str(ref_path))
    out = pred.predict(["CCO", "c1ccccc1", "not-a-smiles"])
    good = [r for r in out if "prob_active" in r]
    assert len(good) == 2 and out[2]["error"] == "unparseable_smiles"
    for r in good:
        assert 0.0 <= r["prob_active"] <= 1.0
        assert set(r["prediction_set"]) <= {0, 1}
        assert isinstance(r["in_domain"], bool)


def test_drift_flags_shifted_batch(tmp_path):
    model_path, ref_path, bundle, X, y, tr = _tiny_bundle_and_reference(tmp_path)
    pred = Predictor(str(model_path), str(ref_path))
    rng = np.random.default_rng(1)

    probs_tr = np.asarray(pred.model.predict_proba(X[tr]))[:, 1]
    same = check_drift(
        X[tr], probs_tr, pred.ref, margins={"pos_rate_shift": 0.5}
    )
    assert same["status"] == "ok", same["warnings"]

    # dense random bits = clearly off-distribution
    shifted = sparse.csr_matrix(rng.binomial(1, 0.5, (50, 512)))
    probs = np.asarray(pred.model.predict_proba(shifted))[:, 1]
    bad = check_drift(shifted, probs, pred.ref)
    assert bad["status"] == "drift_warning" and bad["warnings"]


def test_app_endpoints(tmp_path, monkeypatch):
    model_path, ref_path, *_ = _tiny_bundle_and_reference(tmp_path)
    monkeypatch.setenv("COMP_TOX_MODEL", str(model_path))
    monkeypatch.setenv("COMP_TOX_REFERENCE", str(ref_path))
    from fastapi.testclient import TestClient

    from comp_tox.serve.app import create_app

    client = TestClient(create_app())
    r = client.post("/predict", json={"smiles": ["CCO", "zz"]})
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"][0]["prob_active"] >= 0
    assert body["predictions"][1]["error"] == "unparseable_smiles"
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "logistic_regression"
    d = client.post("/drift", json={"smiles": ["CCO", "CCN", "CCC"]})
    assert d.status_code == 200 and "status" in d.json()


def test_model_card_renders_metrics_not_hardcode():
    from comp_tox.serve.modelcard import render_card

    metrics = {
        "endpoint": "NR-AR",
        "primary_model": "logistic_regression",
        "counts": {"train": 10, "valid": 2, "test": 2, "test_actives": 1,
                   "train_actives": 3, "train_prevalence": 0.3,
                   "test_prevalence": 0.5},
        "models": {
            "logistic_regression": {
                "auroc": 0.7, "auprc": 0.4, "ece": 0.03,
                "auroc_ci95": [0.6, 0.8], "auprc_ci95": [None, None],
                "conformal": {"alpha": 0.1, "qhat": 0.2, "coverage": 0.9,
                              "mean_set_size": 1.0},
                "applicability_domain": {
                    "threshold": 0.3, "in_domain_fraction": 0.5,
                    "median_nn_distance": 0.5,
                    "in_domain": {"auroc": 0.8, "auprc": 0.5},
                    "out_domain": {"auroc": 0.6, "auprc": 0.3},
                    "conformal_coverage_in_domain": 0.9,
                    "conformal_coverage_out_domain": 0.9,
                },
            }
        },
    }
    card = render_card(metrics, {"fingerprint": "morgan-r2-2048"},
                       {"split": {"seed": 0},
                        "evaluation": {"conformal_alpha": 0.1}})
    assert "NR-AR" in card and "NR-ER" not in card
    assert "0.700" in card and "unavailable" in card  # None CI -> withdrawn
