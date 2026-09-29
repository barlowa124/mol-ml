import numpy as np
import pytest

from comp_tox.eval.conformal import (
    conformal_threshold,
    nonconformity,
    prediction_sets,
)


def test_alpha_bounds_rejected():
    s = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="alpha"):
        conformal_threshold(s, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        conformal_threshold(s, alpha=1.0)


def test_empty_calibration_set_rejected():
    with pytest.raises(ValueError, match="empty"):
        conformal_threshold(np.array([]))


def test_coverage_guarantee_at_threshold():
    # calibration scores -> qhat; all calibration points are covered
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1], size=200)
    y = rng.integers(0, 2, size=200)
    scores = nonconformity(probs, y)
    qhat = conformal_threshold(scores, alpha=0.1)
    sets = prediction_sets(probs, qhat)
    covered = sets[np.arange(len(y)), y]
    assert covered.mean() >= 0.9  # finite-sample guarantee


def test_prediction_sets_nonempty_when_uncertain():
    probs = np.array([[0.5, 0.5]])
    sets = prediction_sets(probs, qhat=0.6)
    assert sets.sum() == 2  # both classes retained under uncertainty
