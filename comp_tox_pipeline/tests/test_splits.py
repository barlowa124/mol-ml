import pandas as pd
import pytest

from comp_tox.eval.splits import scaffold_split


def _df(n_scaffolds: int = 10, per_scaffold: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_scaffolds):
        for j in range(per_scaffold):
            rows.append(
                {"smiles": f"C{i}{j}", "scaffold_id": f"s{i}", "label": j % 2}
            )
    return pd.DataFrame(rows)


def test_scaffolds_do_not_leak_across_partitions():
    out = scaffold_split(_df())
    partitions_per_scaffold = out.groupby("scaffold_id")["split"].nunique()
    assert (partitions_per_scaffold == 1).all()


def test_all_partitions_populated():
    out = scaffold_split(_df())
    assert {"train", "valid", "test"} <= set(out["split"].unique())


def test_train_dominates_rows():
    out = scaffold_split(_df())
    frac = out["split"].value_counts(normalize=True)
    assert frac["train"] >= 0.6


def test_deterministic_given_seed():
    a = scaffold_split(_df(), seed=7)
    b = scaffold_split(_df(), seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_rejects_missing_scaffold_column():
    with pytest.raises(ValueError, match="scaffold"):
        scaffold_split(pd.DataFrame({"smiles": ["CC"]}))


def test_zero_frac_valid_yields_no_valid_partition():
    out = scaffold_split(_df(), frac_valid=0.0)
    assert "valid" not in set(out["split"].unique())
