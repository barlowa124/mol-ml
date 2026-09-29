"""Parity between the repo's conformal helpers and the vendored
canonical shared module (src/comp_tox/eval/conformal_shared.py).

The repo implementation computes the same finite-sample level through
np.quantile(method='higher') at a ceiled level, which lands on the same
or the next order statistic as the canonical index — never lower.
"""
import numpy as np

from comp_tox.eval import conformal, conformal_shared


def test_threshold_is_canonical_or_one_step_conservative():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 400))
        s = rng.exponential(0.3, n)
        for alpha in (0.05, 0.1, 0.2):
            q_repo = conformal.conformal_threshold(s, alpha=alpha)
            q_shared = conformal_shared.qhat(s, alpha)
            # never below canonical; at most one sorted position above
            assert q_repo >= q_shared - 1e-12
            sorted_s = np.sort(s)
            rank_repo = int((sorted_s < q_repo - 1e-12).sum())
            rank_shared = int((sorted_s < q_shared - 1e-12).sum())
            assert rank_repo - rank_shared <= 1


def test_classification_helpers_identical():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1], 50)
    y = rng.integers(0, 2, 50)
    np.testing.assert_allclose(
        conformal.nonconformity(probs, y),
        conformal_shared.class_scores(probs, y))
    q = 0.35
    np.testing.assert_array_equal(
        conformal.prediction_sets(probs, q),
        conformal_shared.class_sets(probs, q))


def test_shared_small_n_does_not_crash():
    # ceil((n+1)(1-a))/n exceeds 1 for tiny n at alpha=0.05; the
    # canonical formulation floors at the largest order statistic.
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert conformal_shared.qhat(s, 0.05) == 0.5
