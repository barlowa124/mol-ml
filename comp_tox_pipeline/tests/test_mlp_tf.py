import io

import joblib
import numpy as np
import pytest
from scipy import sparse

tf = pytest.importorskip("tensorflow")

from comp_tox.models.mlp_tf import _KerasMlp, train_mlp_tf


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    X = sparse.csr_matrix((rng.random((n, 2048)) > 0.97).astype(np.float32))
    # signal: compounds with bit 7 set are mostly active
    y = ((X[:, 7].toarray().ravel() > 0).astype(int) | (rng.random(n) < 0.05)).astype(
        np.float32
    )
    return X, y


def test_predict_proba_shape_and_range():
    X, y = _toy()
    m = train_mlp_tf(X, y, seed=0, epochs=2, batch_size=64)
    p = m.predict_proba(X[:10])
    assert p.shape == (10, 2)
    assert np.all(p >= 0) and np.all(p <= 1)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_adapter_survives_joblib_roundtrip():
    X, y = _toy()
    m = train_mlp_tf(X, y, seed=0, epochs=2, batch_size=64)
    buf = io.BytesIO()
    joblib.dump(m, buf)
    buf.seek(0)
    m2 = joblib.load(buf)
    assert np.allclose(m.predict_proba(X[:8]), m2.predict_proba(X[:8]))


def test_seed_determinism():
    X, y = _toy()
    a = train_mlp_tf(X, y, seed=3, epochs=2, batch_size=64)
    b = train_mlp_tf(X, y, seed=3, epochs=2, batch_size=64)
    assert np.allclose(a.predict_proba(X[:16]), b.predict_proba(X[:16]))


def test_model_learns_toy_signal():
    X, y = _toy()
    m = train_mlp_tf(X, y, seed=0, epochs=10, batch_size=64)
    p = m.predict_proba(X)[:, 1]
    pos = y == 1
    assert p[pos].mean() > p[~pos].mean()
