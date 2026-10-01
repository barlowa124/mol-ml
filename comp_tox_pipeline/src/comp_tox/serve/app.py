"""FastAPI serving app for the trained Tox21 endpoint.

    COMP_TOX_MODEL=results/model.joblib \
    COMP_TOX_REFERENCE=results/serve_reference.npz \
    uvicorn comp_tox.serve.app:app

Paths default to ``config.serve`` entries, so the configured run is the
served run. Endpoints:

- ``POST /predict``  — {"smiles": [...]} -> per-compound calibrated
  probability, 90% conformal set, nn-distance, in-domain flag
- ``POST /drift``    — {"smiles": [...]} -> batch drift report against
  the frozen train/eval reference
- ``GET /health``    — liveness + model provenance
- ``GET /model_card`` — the generated model card (docs/model-card.md)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from comp_tox.serve.drift import check_drift
from comp_tox.serve.predictor import Predictor
from comp_tox.util import load_config


class Batch(BaseModel):
    smiles: list[str]


def create_app(
    model_path: str | None = None, reference_path: str | None = None
) -> FastAPI:
    cfg = load_config()
    model_path = model_path or os.environ.get(
        "COMP_TOX_MODEL", cfg["serve"]["model"]
    )
    reference_path = reference_path or os.environ.get(
        "COMP_TOX_REFERENCE", cfg["serve"]["reference"]
    )
    pred = Predictor(model_path, reference_path)
    margins = cfg["serve"].get("drift", {})

    app = FastAPI(title="comp-tox serving", version="0.1.0")

    @app.post("/predict")
    def predict(batch: Batch) -> dict:
        return {
            "endpoint": pred.ref["endpoint"],
            "model": pred.primary,
            "conformal_alpha": pred.ref["conformal_alpha"],
            "predictions": pred.predict(batch.smiles),
        }

    @app.post("/drift")
    def drift(batch: Batch) -> dict:
        X, _, keep = pred.fingerprints(batch.smiles)
        if not keep:
            return {"status": "drift_warning", "warnings": ["no parseable SMILES"]}
        probs = np.asarray(pred.model.predict_proba(X))[:, 1]
        report = check_drift(X, probs, pred.ref, margins=margins)
        report["n_unparseable"] = len(batch.smiles) - len(keep)
        return report

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": pred.primary,
            "fingerprint": pred.provenance.get("fingerprint"),
            "train_rows": pred.provenance.get("train_rows"),
            "features_sha256": pred.provenance.get("features_sha256"),
        }

    @app.get("/model_card")
    def model_card() -> dict:
        card = Path(cfg["serve"].get("model_card", "docs/model-card.md"))
        return {
            "path": str(card),
            "content": card.read_text() if card.exists() else None,
        }

    return app


app = create_app()
