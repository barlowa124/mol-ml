"""AlphaFold DB structure QC: is this predicted model dock-worthy?

AlphaFold DB stores per-residue confidence (pLDDT) in the PDB B-factor
column. This module:
  - fetches AF-{uniprot}-F1-model_*.pdb from alphafold.ebi.ac.uk
  - reports pLDDT statistics (mean, fraction confident >=70)
  - computes C-alpha RMSD against an experimental structure on
    shared residue numbering (Kabsch, numpy)

Rigid docking against a predicted structure is only defensible in
high-confidence regions, which is the QC question this answers.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

from dockops.pdb import ca_records, records

AFDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_{ver}.pdb"
VERSIONS = ["v6", "v4"]


def fetch_afdb(uniprot: str, out_path: str) -> str:
    if Path(out_path).exists():
        # a staged model is pinned evidence; do not silently re-fetch a
        # possibly different AFDB version over it
        return f"staged:{out_path}"
    for ver in VERSIONS:
        url = AFDB_URL.format(uniprot=uniprot, ver=ver)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "dockops/0.1"}
            )
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as r, open(
                out_path, "wb"
            ) as f:
                f.write(r.read())
            return url
        except urllib.error.HTTPError:
            continue
    raise RuntimeError(f"no AFDB model for {uniprot} (tried {VERSIONS})")


def plddt_by_residue(pdb_path: str) -> dict[int, float]:
    """pLDDT per residue, read from the B-factor column of CA atoms."""
    return {
        r["resseq"]: r["bfactor"]
        for r in records(pdb_path)
        if r["atom"] == "CA"
    }


# AlphaFold confidence bands (EBI convention): >=70 confident, >=90 very high
DOWNLOAD_TIMEOUT_S = 120

PLDDT_CONFIDENT = 70.0
PLDDT_VERY_HIGH = 90.0
# Cα agreement cutoff: <2 Å after superposition counts as fold-level match
CA_CLOSE_A = 2.0


def plddt_stats(pdb_path: str) -> dict:
    p = plddt_by_residue(pdb_path)
    if not p:
        raise ValueError(f"no CA atoms in {pdb_path}, not a protein PDB?")
    vals = np.array(list(p.values()))
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        raise ValueError(f"no B-factor (pLDDT) values in {pdb_path}")
    return {
        "n_residues": len(p),
        "mean_plddt": round(float(vals.mean()), 2),
        "frac_confident_70": round(float((vals >= PLDDT_CONFIDENT).mean()), 3),
        "frac_very_high_90": round(float((vals >= PLDDT_VERY_HIGH).mean()), 3),
        "min_plddt": round(float(vals.min()), 1),
    }


def _ca_records(pdb_path: str) -> dict[int, tuple[str, np.ndarray]]:
    """resseq -> (resname, coords) for CA atoms."""
    return ca_records(pdb_path)


def _best_offset(a: dict, b: dict) -> tuple[int, int]:
    """Constant resseq offset maximizing residue-identity agreement.

    Crystal numbering is often shifted vs UniProt (e.g. EGFR's 24-residue
    signal peptide). Scan offsets and pick the one where the most
    paired residues have identical 3-letter names.
    """
    best_d, best_n = 0, 0
    for d in range(-120, 121):
        n = sum(
            1 for r in a if r + d in b and a[r][0] == b[r + d][0]
        )
        if n > best_n:
            best_n, best_d = n, d
    return best_d, best_n


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimal rotation matrix aligning a onto b (both centered)."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD between two (n,3) coordinate sets after optimal rotation."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return float(
        np.sqrt(((a @ _kabsch(a, b).T - b) ** 2).sum(axis=1).mean())
    )


def aligned_deviations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point distances between a and b after optimal superposition."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return np.sqrt(((a @ _kabsch(a, b).T - b) ** 2).sum(axis=1))


def ca_rmsd(pdb_a: str, pdb_b: str) -> dict:
    """C-alpha RMSD between two structures.

    Residues are paired by best constant numbering offset (handles
    UniProt-vs-crystal shifts like EGFR's 24-residue signal peptide);
    only identity-matched pairs are used. Reports the offset so the
    mapping is auditable.
    """
    ca_a, ca_b = _ca_records(pdb_a), _ca_records(pdb_b)
    d, n_match = _best_offset(ca_a, ca_b)
    pairs = [
        (r, r + d)
        for r in ca_a
        if r + d in ca_b and ca_a[r][0] == ca_b[r + d][0]
    ]
    if len(pairs) < 10:
        raise ValueError(
            f"only {len(pairs)} residue-identity pairs (offset {d}), "
            "structures may not be the same protein"
        )
    a = np.stack([ca_a[r][1] for r, _ in pairs])
    b = np.stack([ca_b[s][1] for _, s in pairs])
    dev = aligned_deviations(a, b)
    return {
        "n_paired_residues": len(pairs),
        "numbering_offset": d,
        "ca_rmsd": round(float(np.sqrt((dev**2).mean())), 3),
        "frac_ca_within_2a": round(float((dev < CA_CLOSE_A).mean()), 3),
        "median_ca_dev": round(float(np.median(dev)), 3),
    }


def main() -> None:
    uniprot = sys.argv[1] if len(sys.argv) > 1 else "P00533"  # human EGFR
    out = sys.argv[2] if len(sys.argv) > 2 else "data/raw/afdb_model.pdb"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    url = fetch_afdb(uniprot, out)
    print(f"fetched {url}")
    print(plddt_stats(out))


if __name__ == "__main__":
    main()
