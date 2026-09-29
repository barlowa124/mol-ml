"""Standardize chemical structures and assign scaffold IDs.

Steps:
    - Parse SMILES; unparseable rows are dropped and counted (never silently)
    - Strip salts/solvents and neutralize via rdMolStandardize.FragmentParent
    - Emit canonical isomeric SMILES + Bemis-Murcko scaffold SMILES
    - Dedup by canonical SMILES; groups with conflicting labels are dropped
      and counted (Tox21 has one call per compound, so this is a guard)

Output schema (data/processed/compounds.parquet):
    compound_id, canonical_smiles, scaffold_id, label
"""

from __future__ import annotations

import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles: str) -> tuple[str | None, str | None]:
    if not smiles or not isinstance(smiles, str):
        return None, None  # NaN/None input is unparseable, not a crash
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    parent = rdMolStandardize.FragmentParent(mol)
    canonical = Chem.MolToSmiles(parent, isomericSmiles=True) if parent else ""
    if not canonical:
        return None, None
    scaff_mol = Chem.MolFromSmiles(canonical)
    if scaff_mol is None:
        # RDKit can emit a canonical SMILES that fails to re-parse
        # (kekulization edge cases); downstream stages would drop it
        # later anyway. Drop here so it counts as unparseable.
        return None, None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=scaff_mol)
    if not scaffold:  # acyclic compounds are their own singleton scaffold
        scaffold = f"acyclic:{canonical}"
    return canonical, scaffold


def standardize(in_path: str, out_path: str) -> None:
    df = pd.read_parquet(in_path)

    if "assay_id" in df.columns:
        from comp_tox.util import load_config

        expected = load_config()["endpoint"]["assay_id"]
        assays = set(df["assay_id"].unique())
        if assays != {expected}:
            raise ValueError(
                f"input assays {sorted(assays)} do not match config endpoint "
                f"{expected!r}, stale intermediate; regenerate raw data"
            )

    canon = df["smiles"].map(canonicalize)
    df["canonical_smiles"] = [c for c, _ in canon]
    df["scaffold_id"] = [s for _, s in canon]
    n_unparseable = int(df["canonical_smiles"].isna().sum())
    df = df.dropna(subset=["canonical_smiles"])

    label_sets = df.groupby("canonical_smiles")["label"].nunique()
    conflicting = set(label_sets[label_sets > 1].index)
    n_conflicting = len(conflicting)
    df = df[~df["canonical_smiles"].isin(conflicting)]
    df = df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

    df[["compound_id", "canonical_smiles", "scaffold_id", "label"]].to_parquet(
        out_path, index=False
    )
    print(
        f"standardize: {len(df)} compounds kept; "
        f"{n_unparseable} unparseable dropped; "
        f"{n_conflicting} conflicting-label structures dropped"
    )


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    standardize(in_path, out_path)


if __name__ == "__main__":
    main()
