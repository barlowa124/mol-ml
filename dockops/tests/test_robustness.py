"""Robustness battery: benchmark metric edges, ligand-prep rejects,
PDB fixed-width parsing traps, and API error paths."""

import numpy as np
import pytest

from dockops.benchmark import enrichment_factor, roc_auc
from dockops.engine import MockEngine, get_engine
from dockops.ligands import embed_smiles
from dockops.pdb import records
from dockops.targets import get_target


class TestBenchmarkEdges:
    def test_auc_lower_score_ranks_active(self):
        # docking convention: more negative = better; negated for AUC
        scores = np.array([-10.0, -5.0, -1.0])
        labels = np.array([1, 0, 0])
        assert roc_auc(scores, labels) == pytest.approx(1.0)
        assert roc_auc(scores, labels[::-1]) == pytest.approx(0.0)

    def test_ef_perfect_and_chance(self):
        scores = np.array([-9.0, -8.0, -1.0, -0.5])
        labels = np.array([1, 1, 0, 0])
        assert enrichment_factor(scores, labels, 0.5) == pytest.approx(2.0)
        # random ordering -> EF ~ 1
        rng = np.random.RandomState(0)
        s, y = rng.randn(1000), rng.randint(0, 2, 1000)
        ef = enrichment_factor(s, y, 0.1)
        assert 0.0 <= ef <= 10.0

    def test_ef_no_actives_returns_none(self):
        s = np.array([-1.0, -2.0])
        y = np.array([0, 0])
        assert enrichment_factor(s, y, 0.5) is None

    def test_ef_fraction_edges(self):
        s = np.array([-5.0, -4.0, -3.0])
        y = np.array([1, 0, 0])
        assert enrichment_factor(s, y, 0.0) is not None  # ceil -> k=1
        ef = enrichment_factor(s, y, 1.0)
        assert ef == pytest.approx(1.0)  # top-100% rate == prevalence

    def test_auc_ties(self):
        # all-identical scores -> AUC 0.5 (no ranking information)
        assert roc_auc(np.full(4, -5.0), np.array([1, 0, 1, 0])) == 0.5


class TestEngineEdges:
    def test_mock_deterministic_and_bounded(self):
        from dockops.targets import TargetSpec
        t = TargetSpec("t", "", (0, 0, 0), (10, 10, 10))
        e = MockEngine()
        a = e.dock("CCO", t)
        b = e.dock("CCO", t)
        assert a.score == b.score and -12.0 <= a.score <= -3.0
        assert a.engine == "mock" and a.status == "ok"

    def test_mock_score_depends_on_target(self):
        from dockops.targets import TargetSpec
        t1 = TargetSpec("t1", "", (0, 0, 0), (10, 10, 10))
        t2 = TargetSpec("t2", "", (0, 0, 0), (10, 10, 10))
        e = MockEngine()
        # same ligand on different targets should differ (usually)
        s1 = [e.dock(f"C{'C' * i}O", t1).score for i in range(5)]
        s2 = [e.dock(f"C{'C' * i}O", t2).score for i in range(5)]
        assert s1 != s2

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError):
            get_engine("glide")

    def test_empty_smiles_still_scores(self):
        from dockops.targets import TargetSpec
        t = TargetSpec("t", "", (0, 0, 0), (10, 10, 10))
        # mock engine doesn't parse chemistry; pin that behavior
        assert MockEngine().dock("", t).status == "ok"


class TestLigandPrepEdges:
    def test_invalid_smiles_none_not_crash(self):
        for bad in ["", "not_a_mol", "C(C(", "xyz"]:
            assert embed_smiles(bad) is None

    def test_none_smiles_rejected_not_raised(self):
        # MolFromSmiles(None) throws a C++ TypeError; prep must reject it
        assert embed_smiles(None) is None

    def test_valid_smiles_embeds_3d(self):
        mol = embed_smiles("CCO", seed=0)
        assert mol is not None and mol.GetNumConformers() == 1
        conf = mol.GetConformer()
        zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
        assert not all(z == 0 for z in zs)  # actually 3D, not planar zeros

    def test_seed_determinism(self):
        a = embed_smiles("c1ccccc1O", seed=42).GetConformer().GetAtomPosition(0)
        b = embed_smiles("c1ccccc1O", seed=42).GetConformer().GetAtomPosition(0)
        assert (a.x, a.y, a.z) == (b.x, b.y, b.z)


class TestPdbParsingEdges:
    def _atom_line(self, resseq=1, bfactor=" 20.00", hetatm=False):
        # PDB fixed-width layout per the column spec in dockops.pdb
        rec = "HETATM" if hetatm else "ATOM  "
        return (f"{rec}{1:>5} {'CA':>4} {'ALA':>3} A{resseq:>4}    "
                f"{11.111:8.3f}{22.222:8.3f}{33.333:8.3f}{1.00:6.2f}"
                f"{bfactor:>6}           C  ")

    def test_truncated_line_skipped(self, tmp_path):
        p = tmp_path / "x.pdb"
        p.write_text(self._atom_line()[:40] + "\n" + self._atom_line() + "\n")
        recs = list(records(str(p)))
        assert len(recs) == 1  # <54-char line skipped, valid one parsed

    def test_endmdl_stops_parsing(self, tmp_path):
        p = tmp_path / "x.pdb"
        p.write_text(self._atom_line(1) + "\nENDMDL\n"
                     + self._atom_line(2) + "\n")
        recs = list(records(str(p)))
        assert len(recs) == 1 and recs[0]["resseq"] == 1

    def test_missing_bfactor_is_nan_not_crash(self, tmp_path):
        p = tmp_path / "x.pdb"
        p.write_text(self._atom_line(bfactor="     ") + "\n")
        recs = list(records(str(p)))
        assert len(recs) == 1 and np.isnan(recs[0]["bfactor"])

    def test_hetatm_excluded_by_default(self, tmp_path):
        p = tmp_path / "x.pdb"
        p.write_text(self._atom_line(hetatm=True) + "\n")
        assert list(records(str(p))) == []
        assert len(list(records(str(p), ("HETATM",)))) == 1


class TestTargetEdges:
    def test_unknown_target_keyerror(self):
        with pytest.raises(KeyError):
            get_target("nonexistent_target_xyz")

    def test_target_names_listed_in_error(self):
        try:
            get_target("nope")
        except KeyError as e:
            assert "known:" in str(e)


class TestApiEdges:
    def test_unknown_target_and_job_404(self):
        from fastapi.testclient import TestClient
        from dockops.api import app
        c = TestClient(app)
        r = c.post("/jobs", json={"smiles": "CCO",
                                  "target": "no_such_target"})
        assert r.status_code == 404
        assert c.get("/jobs/deadbeef").status_code == 404

    def test_submit_returns_provenance_and_mock_label(self):
        from fastapi.testclient import TestClient
        from dockops.api import app
        c = TestClient(app)
        targets = c.get("/targets").json()
        r = c.post("/jobs", json={"smiles": "CCO", "target": targets[0]})
        assert r.status_code == 201
        body = r.json()
        assert body["provenance"] and body["engine"] == "mock"
        jid = body["job_id"]
        assert c.get(f"/jobs/{jid}").json()["job_id"] == jid
