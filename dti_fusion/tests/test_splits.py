import numpy as np

from dti_fusion.splits import cold_target_split, random_split


def test_cold_target_disjoint_proteins():
    targets = np.array(["A", "A", "B", "C", "C", "D"] * 50)
    tr, te = cold_target_split(targets, frac=0.25, seed=0)
    assert len(set(targets[tr]) & set(targets[te])) == 0
    assert len(tr) + len(te) == len(targets)
    assert len(np.unique(targets[te])) == 1  # 25% of 4 targets


def test_cold_target_deterministic():
    targets = np.array(list("ABCDEFGH") * 20)
    a = cold_target_split(targets, 0.25, seed=3)
    b = cold_target_split(targets, 0.25, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_random_split_sizes():
    tr, te = random_split(1000, 0.2, seed=1)
    assert len(tr) + len(te) == 1000
    assert abs(len(te) - 200) < 60  # binomial slack
