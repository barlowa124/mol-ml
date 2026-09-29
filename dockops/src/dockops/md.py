"""Minimal OpenMM mechanics: implicit-solvent minimization + short NVT.

A mechanics check, not a production simulation. It uses amber14 + GBn2 implicit
solvent, picoseconds not microseconds. What it demonstrates is real: force
field setup, integrator, energy minimization, trajectory-energy drift, and
RMSD vs the starting geometry.
"""

from __future__ import annotations

import sys

import numpy as np

# GBn2 implicit-solvent protocol constants (standard OBC2 defaults)
SOLUTE_DIELECTRIC = 1.0
SOLVENT_DIELECTRIC = 78.5  # water
TEMPERATURE_K = 300
FRICTION_PS_INV = 1.0      # Langevin collision rate, per-picosecond
TIMESTEP_FS = 1.0
MINIMIZE_TIMESTEP_FS = 2.0
PROTONATION_PH = 7.0


def _positions(context) -> np.ndarray:
    from openmm import unit

    return (
        context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer)
    )


def _potential(context) -> float:
    from openmm import unit

    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )



def _prepare(pdb_path: str):
    """PDB -> protonated (topology, positions) via PDBFixer.

    Crystal structures routinely have missing loops/atoms; PDBFixer fills
    them so force-field templating succeeds. Falls back to raw Modeller
    protonation if pdbfixer is absent.
    """
    from openmm.app import Modeller, PDBFile, ForceField

    ff_file = ("amber14-all.xml", "implicit/gbn2.xml")
    try:
        from pdbfixer import PDBFixer

        fixer = PDBFixer(pdb_path)
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        ff = ForceField(*ff_file)
        fixer.addMissingHydrogens(PROTONATION_PH)
        return ff, fixer.topology, fixer.positions
    except ImportError:
        ff = ForceField(*ff_file)
        pdb = PDBFile(pdb_path)
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        return ff, modeller.topology, modeller.positions

def minimize(pdb_path: str, max_iterations: int = 500) -> dict:
    from openmm import LangevinMiddleIntegrator, LocalEnergyMinimizer, unit
    from openmm.app import Simulation

    ff, topology, positions = _prepare(pdb_path)
    system = ff.createSystem(
        topology,
        soluteDielectric=SOLUTE_DIELECTRIC,
        solventDielectric=SOLVENT_DIELECTRIC,
    )
    sim = Simulation(
        topology,
        system,
        LangevinMiddleIntegrator(
            TEMPERATURE_K * unit.kelvin,
            FRICTION_PS_INV / unit.picosecond,
            MINIMIZE_TIMESTEP_FS * unit.femtoseconds,
        ),
    )
    sim.context.setPositions(positions)
    pe_before = _potential(sim.context)
    pos_before = _positions(sim.context)
    LocalEnergyMinimizer.minimize(sim.context, maxIterations=max_iterations)
    pe_after = _potential(sim.context)
    from dockops.alphafold import kabsch_rmsd

    return {
        "n_atoms": system.getNumParticles(),
        "pe_before_kj_mol": round(pe_before, 1),
        "pe_after_kj_mol": round(pe_after, 1),
        "delta_pe_kj_mol": round(pe_after - pe_before, 1),
        "rmsd_to_start_nm": round(
            kabsch_rmsd(pos_before, _positions(sim.context)), 4
        ),
    }


def short_nvt(pdb_path: str, steps: int = 2000, report_every: int = 500) -> dict:
    """Picoseconds of Langevin dynamics at 300K; reports energy drift."""
    from openmm import LangevinMiddleIntegrator, unit
    from openmm.app import Simulation

    ff, topology, positions = _prepare(pdb_path)
    system = ff.createSystem(
        topology,
        soluteDielectric=SOLUTE_DIELECTRIC,
        solventDielectric=SOLVENT_DIELECTRIC,
    )
    integrator = LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        FRICTION_PS_INV / unit.picosecond,
        TIMESTEP_FS * unit.femtoseconds,
    )
    sim = Simulation(topology, system, integrator)
    sim.context.setPositions(positions)
    from openmm import LocalEnergyMinimizer

    LocalEnergyMinimizer.minimize(sim.context, maxIterations=200)
    pos_start = _positions(sim.context)
    energies = []
    for _ in range(steps // report_every):
        integrator.step(report_every)
        energies.append(round(_potential(sim.context), 1))
    from dockops.alphafold import kabsch_rmsd

    return {
        "steps": steps,
        "timestep_fs": TIMESTEP_FS,
        "temperature_k": TEMPERATURE_K,
        "energies_kj_mol": energies,
        "energy_drift_kj_mol": (
            round(energies[-1] - energies[0], 1) if energies else None
        ),
        "rmsd_to_start_nm": round(
            kabsch_rmsd(pos_start, _positions(sim.context)), 4
        ),
    }


def main() -> None:
    pdb_path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/1L2Y.pdb"
    print("minimize:", minimize(pdb_path))
    print("nvt:", short_nvt(pdb_path))


if __name__ == "__main__":
    main()
