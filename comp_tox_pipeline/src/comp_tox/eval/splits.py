"""Scaffold-aware train/validation/test splitting.

Structural (scaffold) splitting is the prospective evaluation for molecular ML:
random splits overestimate prospective performance because near-identical
analogs land in both train and test. Rows sharing a Bemis-Murcko scaffold
(here, a precomputed ``scaffold_id``) are always kept in the same partition.

Groups are sorted by descending size, the standard DeepChem/MoleculeNet
convention, so the test set preferentially contains rarer, more structurally
distinct chemotypes. Ties in group size are broken deterministically with a
seeded shuffle.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from comp_tox.util import load_config


def scaffold_split(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold_id",
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``split`` column (train/valid/test).

    Args:
        df: one row per compound, with a scaffold group column.
        scaffold_col: column identifying shared-scaffold groups.
        frac_train: target fraction of rows for train.
        frac_valid: target fraction for validation; remainder goes to test.
        seed: RNG seed for tie-breaking equal-size groups.
    """
    if scaffold_col not in df.columns:
        raise ValueError(f"missing scaffold column: {scaffold_col!r}")
    if not (0 < frac_train < 1 and 0 <= frac_valid < 1 - frac_train):
        raise ValueError("invalid split fractions")

    rng = np.random.default_rng(seed)
    n = len(df)

    sizes = df.groupby(scaffold_col).size()
    groups = sizes.index.to_numpy()
    rng.shuffle(groups)
    groups = groups[np.argsort(-sizes.loc[groups].to_numpy(), kind="stable")]

    assignment: dict[str, str] = {}
    counts = {"train": 0, "valid": 0, "test": 0}
    for g in groups:
        size = int(sizes.loc[g])
        if counts["train"] + size <= frac_train * n or counts["train"] == 0:
            part = "train"
        elif frac_valid > 0 and (
            counts["valid"] + size <= frac_valid * n or counts["valid"] == 0
        ):
            part = "valid"
        else:
            part = "test"
        assignment[g] = part
        counts[part] += size

    out = df.copy()
    out["split"] = out[scaffold_col].map(assignment)
    return out


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    df = pd.read_parquet(in_path)
    s = load_config()["split"]
    if s.get("method", "scaffold") != "scaffold":
        raise ValueError(f"unsupported split method: {s['method']!r}")
    scaffold_split(
        df,
        frac_train=s["frac_train"],
        frac_valid=s["frac_valid"],
        seed=s["seed"],
    ).to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()
