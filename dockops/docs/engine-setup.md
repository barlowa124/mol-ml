# Docking backend setup

`VinaEngine` and `ligands.to_pdbqt` (meeko) are implemented. What remains
environment-specific is installing `vina` and preparing a receptor.

## Install

```bash
pip install -e .[vina]    # vina (AutoDock Vina python bindings) + meeko + gemmi
```

**Platform note:** `vina` ships manylinux and macOS x86_64 wheels only, so on
Apple Silicon the pip build needs Boost headers and fails. Use the Dockerfile
(Linux), conda-forge (`conda install -c conda-forge vina meeko`), or an
x86_64 environment. `meeko` alone is pure Python and installs everywhere and
ligand prep works on macOS.

## Receptor prep (one-time per target)

- Convert a cleaned receptor PDB to PDBQT with `mk_prepare_receptor.py`
  (Meeko) or ADFR's `prepare_receptor`; keep protonation/tautomers explicit
  and record the prep tool + version in provenance.
- Box: derived in code, not config, so set `box_from_ligand: <resname>` on the
  target and `dockops.receptor.box_from_ligand` computes the center/size
  around the co-crystallized ligand (plus `box_padding`, default 5 Å).
  `egfr_1m17` in config is a working example (PDB 1M17, ligand AQ4).
- Add the receptor path to `config/config.yaml` under `targets:`, then set
  `engine: vina` (with `engine_params` for exhaustiveness/seed).

## Real benchmark data

`.venv/bin/python -m dockops.fetch` stages a real DUD-E benchmark target:
actives/decoys `.ism` files → `data/raw/ligands_real.csv` (same schema as
the demo fixture) and the co-crystallized receptor PDB →
`data/raw/receptor.pdb`. Point `benchmark.ligands_csv` at the CSV to run the
pipeline on it. Scores under `engine: mock` remain labeled demos. The data
is real, the affinities are not.

## What VinaEngine does

Per ligand: SMILES → RDKit ETKDG+MMFF 3D → meeko PDBQT →
`Vina(sf_name="vina", seed)` → `set_receptor` → `compute_vina_maps(box)` →
`dock(exhaustiveness, n_poses=1)` → `energies()[0][0]` affinity. Missing
receptor files and unembeddable ligands return error/unprepared statuses.
They do not raise.
