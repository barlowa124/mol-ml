import numpy as np

from dockops.benchmark import enrichment_factor, roc_auc


def test_roc_auc_perfect_ranking():
    # lower score = better docking; actives get the best (lowest) scores
    scores = np.array([0.9, 0.8, 0.1, 0.2])  # actives at idx 2,3
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == 1.0


def test_roc_auc_reversed_ranking():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == 0.0


def test_enrichment_factor():
    scores = np.arange(10, dtype=float)
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # top 10% = 1 compound, an active; prevalence 0.2 -> EF = 5.0
    assert enrichment_factor(scores, labels, 0.1) == 5.0


def test_enrichment_factor_no_actives():
    scores = np.arange(5, dtype=float)
    labels = np.zeros(5, dtype=int)
    # zero prevalence is undefined, reported as null so metrics.json
    # stays valid strict JSON (NaN literals are not)
    assert enrichment_factor(scores, labels, 0.2) is None
