import pytest

from dockops.engine import MockEngine, get_engine
from dockops.targets import TargetSpec

TARGET = TargetSpec(
    name="demo_target",
    receptor_pdbqt="missing.pdbqt",
    box_center=(0.0, 0.0, 0.0),
    box_size=(22.5, 22.5, 22.5),
)


def test_mock_engine_is_deterministic():
    a = MockEngine().dock("CCO", TARGET)
    b = MockEngine().dock("CCO", TARGET)
    assert a.score == b.score
    assert a.status == "ok"
    assert a.engine == "mock"


def test_mock_engine_score_in_plausible_range():
    res = MockEngine().dock("c1ccccc1", TARGET)
    assert -12.0 <= res.score <= -3.0


def test_mock_score_depends_on_target():
    other = TargetSpec("other", "x", (0, 0, 0), (20, 20, 20))
    assert MockEngine().dock("CCO", TARGET).score != MockEngine().dock("CCO", other).score


def test_unknown_engine_rejected():
    with pytest.raises(ValueError, match="unknown engine"):
        get_engine("nonsense")


def test_vina_engine_guards_missing_backend():
    try:
        engine = get_engine("vina")
    except RuntimeError:
        return  # vina not installed: guard fired as designed
    res = engine.dock("CCO", TARGET)  # receptor file absent -> explicit error
    assert res.status == "error"
    assert res.score is None
