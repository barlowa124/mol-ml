"""Robustness battery: split integrity invariants, Tanimoto distance edges,
calibration degenerate inputs, and canonicalization traps."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from comp_tox.data.standardize import canonicalize
from comp_tox.eval.applicability import in_domain, nn_tanimoto_distances
from comp_tox.eval.calibration import (expected_calibration_error,
                                       reliability_curve)
from comp_tox.eval.splits import scaffold_split
from comp_tox.eval.evaluate import _scaffold_bootstrap


class TestScaffoldBootstrap:
    def test_repeated_draws_preserve_all_group_rows(self, monkeypatch):
        df = pd.DataFrame({
            "scaffold_id": ["A", "A", "B", "C"],
            "label": [0, 1, 1, 0],
        })
        probabilities = np.array([0.7, 0.2, 0.9, 0.1])
        observed = []

        class FixedDraw:
            def choice(self, n, size, replace):
                assert (n, size, replace) == (3, 3, True)
                return np.array([0, 0, 1])

        def metric(labels, scores):
            observed.append((labels.tolist(), scores.tolist()))
            return 0.5

        monkeypatch.setattr("comp_tox.eval.evaluate.np.random.default_rng",
                            lambda seed: FixedDraw())
        monkeypatch.setattr("comp_tox.eval.evaluate.roc_auc_score", metric)
        monkeypatch.setattr("comp_tox.eval.evaluate.average_precision_score", metric)
        result = _scaffold_bootstrap(df, probabilities, n_boot=1, seed=0)
        expected = ([0, 1, 0, 1, 1], [0.7, 0.2, 0.7, 0.2, 0.9])
        assert observed == [expected, expected]
        assert result["n_boot_used"] == 1

    def test_single_class_has_no_valid_replicates(self):
        df = pd.DataFrame({"scaffold_id": ["A", "B"], "label": [0, 0]})
        result = _scaffold_bootstrap(df, np.array([0.1, 0.2]), 5, 0)
        assert result == {"auroc_ci95": [None, None],
                          "auprc_ci95": [None, None], "n_boot_used": 0}

    def test_seed_replays_intervals(self):
        df = pd.DataFrame({"scaffold_id": ["A", "A", "B", "B"],
                           "label": [0, 1, 0, 1]})
        probabilities = np.array([0.7, 0.2, 0.1, 0.9])
        assert _scaffold_bootstrap(df, probabilities, 20, 7) == \
            _scaffold_bootstrap(df, probabilities, 20, 7)

    def test_corrected_nr_ar_intervals_replay_from_frozen_predictions(self):
        results = Path(__file__).resolve().parents[1] / "results"
        metrics = json.loads((results / "metrics_NR-AR.json").read_text())
        protocol = metrics["bootstrap"]
        path = results / protocol["predictions"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol["predictions_sha256"]
        frame = pd.read_parquet(path)
        assert len(frame) == metrics["counts"]["test"]
        assert int(frame.label.sum()) == metrics["counts"]["test_actives"]
        for name, expected in metrics["models"].items():
            actual = _scaffold_bootstrap(frame, frame[name].to_numpy(),
                                         protocol["n_bootstrap"], protocol["seed"])
            assert actual["n_boot_used"] == expected["n_boot_used"]
            for metric in ("auroc_ci95", "auprc_ci95"):
                assert np.allclose(actual[metric], expected[metric], rtol=0, atol=1e-12)
                assert metrics[metric] == metrics["models"][metrics["primary_model"]][metric]


def _df(n=200, n_scaff=50, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "compound_id": [f"c{i}" for i in range(n)],
        "scaffold_id": [f"s{rng.randint(n_scaff)}" for _ in range(n)],
        "label": rng.randint(0, 2, n),
    })


class TestScaffoldSplit:
    def test_scaffold_never_straddles_partitions(self):
        df = _df()
        out = scaffold_split(df, seed=0)
        for _, g in out.groupby("scaffold_id"):
            assert g["split"].nunique() == 1

    def test_every_row_assigned(self):
        out = scaffold_split(_df(), seed=0)
        assert out["split"].isin(["train", "valid", "test"]).all()
        assert out["split"].notna().all()

    def test_deterministic_same_seed(self):
        df = _df()
        a = scaffold_split(df, seed=3)["split"].tolist()
        b = scaffold_split(df, seed=3)["split"].tolist()
        assert a == b
        c = scaffold_split(df, seed=4)["split"].tolist()
        assert a != c

    def test_single_scaffold_group_goes_to_train(self):
        df = _df(20, n_scaff=1)
        out = scaffold_split(df)
        assert (out["split"] == "train").all()

    def test_invalid_fractions_raise(self):
        df = _df()
        for ft, fv in [(0.0, 0.1), (1.0, 0.0), (0.5, 0.6), (-0.1, 0.1)]:
            with pytest.raises(ValueError):
                scaffold_split(df, frac_train=ft, frac_valid=fv)

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            scaffold_split(pd.DataFrame({"label": [1]}))

    def test_zero_valid_fraction(self):
        out = scaffold_split(_df(), frac_train=0.8, frac_valid=0.0)
        assert not (out["split"] == "valid").any()


class TestTanimotoAD:
    def _fp(self, rows, nbits=64, seed=0):
        rng = np.random.RandomState(seed)
        return sparse.csr_matrix(
            rng.rand(rows, nbits) > 0.7, dtype=np.float32)

    def test_identical_row_distance_zero(self):
        X = self._fp(20)
        d = nn_tanimoto_distances(X[:5], X[:5])
        assert np.allclose(d, 0.0)

    def test_disjoint_bits_distance_one(self):
        a = sparse.csr_matrix(np.eye(2, 8, dtype=np.float32))
        b = sparse.csr_matrix(np.fliplr(np.eye(2, 8)).copy())
        d = nn_tanimoto_distances(a, b)
        assert np.allclose(d, 1.0)

    def test_block_boundary_consistency(self):
        X = self._fp(600, seed=1)
        full = nn_tanimoto_distances(X, X[:50], block=512)
        small = nn_tanimoto_distances(X, X[:50], block=100)
        assert np.allclose(full, small)

    def test_empty_fingerprints_match_empty(self):
        # union=0 pairs default to sim=1 -> distance 0: two "empty"
        # compounds count as identical, not maximally distant
        a = sparse.csr_matrix(np.zeros((3, 16), dtype=np.float32))
        d = nn_tanimoto_distances(a, a)
        assert np.allclose(d, 0.0)

    def test_in_domain_threshold_edges(self):
        d = np.array([0.0, 0.3, 0.7])
        assert in_domain(d, 0.3).tolist() == [True, True, False]
        assert in_domain(d, 0.0).tolist() == [True, False, False]


class TestCalibrationEdges:
    def test_ece_perfect_prediction_edge_bias(self):
        # p=0/1 land at bin edges, not centers: perfect predictions still
        # contribute |edge-center| = half-bin-width bias. Pin that contract.
        y = np.array([0, 0, 1, 1])
        ece = expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0]),
                                         n_bins=4)
        assert ece == pytest.approx(0.125)
        # observed accuracy equal to bin-center confidence -> ECE 0
        # (0.6 lands strictly inside the [0.5,0.75) bin, center 0.625;
        # digitize assigns edge values to the upper bin, so 0.5 itself
        # would sit in the 0.625 bin too)
        ece0 = expected_calibration_error(
            np.array([1] * 5 + [0] * 3), np.full(8, 0.6), n_bins=4)
        assert ece0 == pytest.approx(0.0)

    def test_ece_bounded(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 500)
        p = rng.rand(500)
        ece = expected_calibration_error(y, p, 15)
        assert 0.0 <= ece <= 1.0

    def test_probs_outside_unit_interval_clipped(self):
        # digitize clips into the outer bins; must not crash or drop
        c, a, n = reliability_curve(
            np.array([0, 1]), np.array([-0.5, 1.5]), n_bins=10)
        assert n.sum() == 2

    def test_empty_input_nan(self):
        ece = expected_calibration_error(np.array([]), np.array([]), 10)
        assert np.isnan(ece)

    def test_single_bin(self):
        c, a, n = reliability_curve(np.array([0, 1, 1]),
                                    np.array([0.4, 0.5, 0.6]), n_bins=1)
        assert n.tolist() == [3] and a.tolist() == [pytest.approx(2 / 3)]


class TestCanonicalizeEdges:
    def test_garbage_smiles_returns_none(self):
        for bad in ["", "not_a_smiles", "C(C(", "12345", None]:
            assert canonicalize(bad) == (None, None)

    def test_salt_stripped_to_parent(self):
        canon, _ = canonicalize("CC(=O)[O-].[Na+]")
        assert canon is not None and "Na" not in canon

    def test_acyclic_compound_scaffold_self(self):
        canon, scaff = canonicalize("CCO")
        assert canon is not None and scaff.startswith("acyclic:")

    def test_duplicate_mol_consistent_canonical(self):
        a, _ = canonicalize("OCC")
        b, _ = canonicalize("CCO")
        assert a == b
