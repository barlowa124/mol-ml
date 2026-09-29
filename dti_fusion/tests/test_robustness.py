"""Robustness battery: fingerprint rejects, split integrity, model modes."""

import numpy as np
import pytest
import torch

from dti_fusion.features import drug_fingerprints, morgan_fingerprint
from dti_fusion.model import BranchMLP
from dti_fusion.splits import cold_target_split, random_split


class TestFingerprintEdges:
    def test_invalid_smiles_zero_fp_not_crash(self):
        X = drug_fingerprints(["CCO", "not_a_smiles", "C(C("],
                              "morgan-r2-1024")
        assert X.shape == (3, 1024)
        assert (X[1] == 0).all() and (X[2] == 0).all()
        assert X[0].sum() > 0

    def test_none_and_empty_smiles(self):
        assert morgan_fingerprint(None, 2, 64) is None
        assert morgan_fingerprint("", 2, 64) is None

    def test_bad_spec_rejected(self):
        for bad in ["morgan", "morgan-r2", "ecfp-r2-1024",
                    "morgan-rx-1024", ""]:
            with pytest.raises(ValueError):
                drug_fingerprints(["CCO"], bad)

    def test_spec_variants(self):
        assert drug_fingerprints(["CCO"], "morgan-r3-2048").shape == (1, 2048)

    def test_nan_smiles_zero_fp(self):
        X = drug_fingerprints([float("nan")], "morgan-r2-64")
        assert (X[0] == 0).all()


class TestSplitIntegrity:
    def test_cold_target_no_leak(self):
        tids = np.array([f"p{i}" for i in range(50) for _ in range(4)])
        tr, te = cold_target_split(tids, 0.2, 0)
        assert not set(tids[tr]) & set(tids[te])

    def test_cold_target_min_one(self):
        tids = np.array(["a", "b", "c"])
        for frac in (0.01, 0.1):
            tr, te = cold_target_split(tids, frac, 0)
            assert len(te) >= 1

    def test_cold_target_deterministic(self):
        tids = np.array([f"p{i}" for i in range(30)])
        a = cold_target_split(tids, 0.2, 5)
        b = cold_target_split(tids, 0.2, 5)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_cold_target_single_target(self):
        tids = np.array(["only"] * 10)
        tr, te = cold_target_split(tids, 0.5, 0)
        # the single target lands entirely in one partition
        assert len(tr) == 0 or len(te) == 0

    def test_random_split_shapes(self):
        tr, te = random_split(100, 0.2, 0)
        assert len(tr) + len(te) == 100
        assert not set(tr) & set(te)


class TestModelEdges:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            BranchMLP(64, 32, 16, 0.1, mode="concat")

    def test_all_modes_forward(self):
        mlp = {m: BranchMLP(64, 32, 16, 0.0, mode=m)
               for m in ("fusion", "drug", "protein")}
        xd, xp = torch.randn(4, 64), torch.randn(4, 32)
        for mode, m in mlp.items():
            out = m(xd, xp)
            assert out.shape[0] == 4

    def test_dropout_zero_deterministic(self):
        m = BranchMLP(8, 4, 8, 0.0).eval()
        xd, xp = torch.randn(2, 8), torch.randn(2, 4)
        assert torch.allclose(m(xd, xp), m(xd, xp))
