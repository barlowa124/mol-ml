"""NAM dossier: generated text binds every number to the metrics artifact."""
import json
from pathlib import Path

import pytest
import yaml

from comp_tox.dossier import build_dossier, verify_dossier

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text())
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())


def _build(tmp_path, metrics=None):
    mp = tmp_path / "metrics.json"
    cp = tmp_path / "config.yaml"
    mp.write_text(json.dumps(metrics or METRICS))
    cp.write_text(yaml.safe_dump(CONFIG))
    return build_dossier(metrics or METRICS, CONFIG, str(mp), str(cp))


def test_dossier_claims_bind(tmp_path):
    text = _build(tmp_path)
    (tmp_path / "dossier.md").write_text(text)
    rep = verify_dossier(str(tmp_path / "dossier.md"), METRICS, CONFIG)
    assert rep["passed"], rep["unbound_claims"]
    assert rep["n_claims"] > 20


def test_withdrawn_bootstrap_surfaces(tmp_path):
    assert METRICS["bootstrap"]["status"] == "withdrawn"
    text = _build(tmp_path)
    assert "Bootstrap intervals are withdrawn" in text
    assert "not available" in text


def test_low_domain_fraction_surfaces(tmp_path):
    text = _build(tmp_path)
    frac = METRICS["applicability_domain"]["in_domain_fraction"]
    assert frac < 0.5
    assert "applicability domain" in text
    assert "out-of-domain by default" in text


def test_tampered_number_fails_verify(tmp_path):
    text = _build(tmp_path)
    # 0.9777 is no recorded leaf; an edited metric must not bind
    bad = text.replace("0.6437", "0.9777")
    assert "0.6437" in text  # fixture sanity: the primary AUROC is present
    (tmp_path / "d.md").write_text(bad)
    rep = verify_dossier(str(tmp_path / "d.md"), METRICS, CONFIG)
    assert not rep["passed"]
    assert any(u["token"].startswith("0.9777") for u in rep["unbound_claims"])


def test_scaffold_split_is_headline(tmp_path):
    text = _build(tmp_path)
    assert "scaffold split" in text
    assert "random split" not in text.lower().replace("-", " ")
