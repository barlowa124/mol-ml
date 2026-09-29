"""Receptor-side helpers: derive the search-space box from a co-crystallized
ligand instead of hand-typed coordinates.

Docking boxes in published benchmarks are almost always defined around the
bound ligand; encoding that derivation in code removes an unverifiable
constant from config.
"""

from __future__ import annotations

from dockops.pdb import records


def ligand_atoms(pdb_path: str, resname: str) -> list[tuple[float, float, float]]:
    """3D coordinates of all HETATM atoms belonging to resname.

    A resname present at multiple sites (multiple chains or residue
    numbers) is ambiguous for box derivation, so refuse instead of spanning
    a box across separate binding sites.
    """
    hits = [
        r for r in records(pdb_path, kinds=("HETATM",)) if r["resname"] == resname
    ]
    if not hits:
        raise ValueError(f"no HETATM atoms for resname {resname!r} in {pdb_path}")
    sites = {(r["chain"], r["resseq"]) for r in hits}
    if len(sites) > 1:
        raise ValueError(
            f"{resname!r} found at {len(sites)} sites {sorted(sites)} in "
            f"{pdb_path}, a docking box would span multiple sites; "
            "trim the receptor or pick a site explicitly"
        )
    return [tuple(r["coords"]) for r in hits]


def box_from_ligand(
    pdb_path: str, resname: str, padding: float = 5.0
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(center, size) of a box around the ligand, padded by `padding` Å."""
    coords = ligand_atoms(pdb_path, resname)
    mins = [min(c[i] for c in coords) for i in range(3)]
    maxs = [max(c[i] for c in coords) for i in range(3)]
    center = tuple(round((lo + hi) / 2, 3) for lo, hi in zip(mins, maxs))
    size = tuple(round(hi - lo + 2 * padding, 3) for lo, hi in zip(mins, maxs))
    return center, size
