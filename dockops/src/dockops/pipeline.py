"""Batch docking pipeline: ligands x target -> scores.csv + provenance.json.

Each ligand is 3D-embedded (rdkit) before docking so failures stay
visible: unprepared ligands are marked "unprepared", engine failures "error",
successes "ok". Nothing is silently dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from dockops.engine import get_engine
from dockops.ligands import embed_smiles
from dockops.provenance import manifest, write_manifest
from dockops.targets import get_target
from dockops.util import load_config


def run_batch(
    ligands_csv: str,
    target_name: str,
    out_csv: str,
    provenance_out: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    cfg = load_config(config_path)
    engine = get_engine(cfg.get("engine", "mock"), **cfg.get("engine_params", {}))
    target = get_target(target_name, config_path)

    ligands = pd.read_csv(ligands_csv)
    rows = []
    for _, row in ligands.iterrows():
        smiles = row["smiles"]
        if embed_smiles(smiles) is None:
            rows.append(
                {
                    "compound_id": row["compound_id"],
                    "smiles": smiles,
                    "score": None,
                    "status": "unprepared",
                    "engine": engine.name,
                }
            )
            continue
        res = engine.dock(smiles, target)
        rows.append(
            {
                "compound_id": row["compound_id"],
                "smiles": smiles,
                "score": res.score,
                "status": res.status,
                "engine": res.engine,
            }
        )

    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    inputs = [ligands_csv]
    if target.receptor_pdbqt and Path(target.receptor_pdbqt).exists():
        inputs.append(target.receptor_pdbqt)
    write_manifest(
        manifest(inputs, config_path, engine.name)
        | {"target": target_name, "n_ligands": len(out)},
        provenance_out,
    )
    print(
        f"dock[{engine.name}]: {(out['status'] == 'ok').sum()}/{len(out)} ligands "
        f"docked vs {target_name} -> {out_csv}"
    )
    return out


def main() -> None:
    ligands_csv, target, out_csv, prov_out = sys.argv[1:5]
    run_batch(ligands_csv, target, out_csv, prov_out)


if __name__ == "__main__":
    main()
