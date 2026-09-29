"""Compute Morgan fingerprints and persist them as a sparse matrix.

Primary representation: Morgan (ECFP-style) fingerprints, radius and bit
width come from `config.model.fingerprint` ("morgan-r<radius>-<bits>"), a
strong, interpretable baseline for toxicity endpoints.

Outputs:
    features.npz              csr_matrix, row-aligned with the meta parquet
    features_meta.parquet     compound_id, canonical_smiles, scaffold_id, label
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

from comp_tox.util import load_config
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

DEFAULT_FINGERPRINT = "morgan-r2-2048"
_FINGERPRINT_RE = re.compile(r"morgan-r(\d+)-(\d+)")


def _parse_fingerprint(spec: str) -> tuple[int, int]:
    """Parse a 'morgan-r<radius>-<bits>' spec from config.model.fingerprint."""
    m = _FINGERPRINT_RE.fullmatch(spec.strip())
    if m is None:
        raise ValueError(
            f"unsupported fingerprint spec {spec!r}, expected "
            "'morgan-r<radius>-<bits>' (e.g. morgan-r2-2048)"
        )
    return int(m.group(1)), int(m.group(2))


def build_features(
    in_path: str, npz_out: str, meta_out: str,
    fingerprint: str = DEFAULT_FINGERPRINT,
) -> None:
    radius, fp_size = _parse_fingerprint(fingerprint)
    df = pd.read_parquet(in_path)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    rows = []
    keep = []
    for i, smi in enumerate(df["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append(gen.GetFingerprintAsNumPy(mol))
        keep.append(i)

    X = sparse.csr_matrix(np.asarray(rows, dtype=np.uint8))
    sparse.save_npz(npz_out, X)
    df.iloc[keep].reset_index(drop=True)[
        ["compound_id", "canonical_smiles", "scaffold_id", "label"]
    ].to_parquet(meta_out, index=False)
    print(f"features: {X.shape[0]} x {X.shape[1]} matrix -> {npz_out}")


def main() -> None:
    in_path, npz_out, meta_out = sys.argv[1], sys.argv[2], sys.argv[3]
    fingerprint = load_config()["model"]["fingerprint"]
    build_features(in_path, npz_out, meta_out, fingerprint)


if __name__ == "__main__":
    main()
