"""Shared PDB fixed-width column parsing.

PDB ATOM/HETATM records are column-format; the slice offsets are
constants of the format spec, not tunable values, named once here so
callers don't each hardcode the same slices.
"""

from __future__ import annotations

import numpy as np

# PDB format column slices (1-based spec columns -> python slice)
ATOM_NAME = slice(12, 16)
RES_NAME = slice(17, 20)
CHAIN_ID = 21
RES_SEQ = slice(22, 26)
X = slice(30, 38)
Y = slice(38, 46)
Z = slice(46, 54)
B_FACTOR = slice(60, 66)


def records(pdb_path: str, kinds: tuple[str, ...] = ("ATOM",)):
    """Yield parsed field tuples for ATOM/HETATM lines.

    Stops at the first ENDMDL. For NMR ensembles only MODEL 1 is used,
    matching the fixture convention instead of silently mixing models.
    """
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                return
            # need through the Z-coordinate column (54) to be usable
            if not line.startswith(kinds) or len(line) < 54:
                continue
            bf = line[B_FACTOR]
            yield {
                "atom": line[ATOM_NAME].strip(),
                "resname": line[RES_NAME].strip(),
                "chain": line[CHAIN_ID],
                "resseq": int(line[RES_SEQ]),
                "coords": np.array(
                    [
                        float(line[X]),
                        float(line[Y]),
                        float(line[Z]),
                    ]
                ),
                # truncated records may lack a B-factor column
                "bfactor": float(bf) if bf.strip() else float("nan"),
            }


def ca_records(pdb_path: str) -> dict[int, tuple[str, np.ndarray]]:
    """resseq -> (resname, CA coords) for one structure."""
    return {
        r["resseq"]: (r["resname"], r["coords"])
        for r in records(pdb_path)
        if r["atom"] == "CA"
    }
