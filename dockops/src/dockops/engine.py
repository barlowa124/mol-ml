"""Docking engine interface and backends.

DockingEngine is the seam every backend implements. MockEngine is real code
with fake physics: deterministic pseudo-scores seeded by (smiles, target),
used so the pipeline, API, and tests run without a docking binary. Its scores
are labeled engine="mock" everywhere and must never be presented as docking
results. VinaEngine is the real backend (meeko ligand prep + Vina scoring);
it needs the `vina`/`meeko` packages and a prepared receptor PDBQT.
See docs/engine-setup.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from dockops.targets import TargetSpec


@dataclass(frozen=True)
class DockResult:
    score: float | None
    status: str  # "ok" | "error" | "unprepared"
    engine: str
    detail: dict = field(default_factory=dict)


class DockingEngine(Protocol):
    name: str

    def dock(self, smiles: str, target: TargetSpec) -> DockResult: ...


class MockEngine:
    """Deterministic pseudo-scores in a plausible Vina range (-12..-3).

    Score is derived from sha256(smiles|target|engine), stable across runs,
    meaningless as physics.
    """

    name = "mock"

    def dock(self, smiles: str, target: TargetSpec) -> DockResult:
        digest = hashlib.sha256(
            f"{smiles}|{target.name}|{self.name}".encode()
        ).digest()
        score = -3.0 - 9.0 * (int.from_bytes(digest[:8]) / 2**64)
        return DockResult(
            score=round(score, 3), status="ok", engine=self.name, detail={}
        )


class VinaEngine:
    """AutoDock Vina backend.

    Requires the `vina` and `meeko` packages (pip install .[vina]; vina ships
    no macOS arm64 wheel, so use the Dockerfile or conda-forge on Macs). The
    receptor PDBQT must exist at target.receptor_pdbqt; ligand prep is
    SMILES -> RDKit 3D -> meeko PDBQT per call.
    """

    name = "vina"

    def __init__(self, exhaustiveness: int = 8, seed: int = 0) -> None:
        try:
            import vina  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "vina package not installed, see docs/engine-setup.md"
            ) from e
        self.exhaustiveness = exhaustiveness
        self.seed = seed

    def dock(self, smiles: str, target: TargetSpec) -> DockResult:
        from pathlib import Path

        from vina import Vina

        from dockops.ligands import to_pdbqt

        if not Path(target.receptor_pdbqt).exists():
            return DockResult(
                None,
                "error",
                self.name,
                {"error": f"receptor PDBQT missing: {target.receptor_pdbqt}"},
            )
        pdbqt = to_pdbqt(smiles)
        if pdbqt is None:
            return DockResult(None, "unprepared", self.name, {})

        v = Vina(sf_name="vina", seed=self.seed)
        v.set_receptor(target.receptor_pdbqt)
        v.set_ligand_from_string(pdbqt)
        v.compute_vina_maps(
            center=list(target.box_center), box_size=list(target.box_size)
        )
        v.dock(exhaustiveness=self.exhaustiveness, n_poses=1)
        score = float(v.energies(n_poses=1)[0][0])
        return DockResult(
            score=score,
            status="ok",
            engine=self.name,
            detail={
                "exhaustiveness": self.exhaustiveness,
                "seed": self.seed,
                "box_center": list(target.box_center),
                "box_size": list(target.box_size),
            },
        )


def get_engine(name: str, **engine_kwargs) -> DockingEngine:
    if name == "mock":
        return MockEngine()
    if name == "vina":
        return VinaEngine(**engine_kwargs)
    raise ValueError(f"unknown engine: {name!r}")
