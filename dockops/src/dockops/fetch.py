"""Fetch a real benchmark target: DUD-E actives/decoys + RCSB receptor.

DUD-E (dude.docking.org) ships per-target `actives_final.ism` /
`decoys_final.ism` files with "SMILES CHEMBLID" per line. The receptor is a
co-crystallized PDB from RCSB; the docking box is derived from its ligand
at target-resolution time (see receptor.box_from_ligand).

Not part of the default DAG. Run once to stage real inputs:

    .venv/bin/python -m dockops.fetch

then point benchmark.ligands_csv at the produced CSV. Scores on the mock
engine are still engine="mock" demos; the data is real, the scoring is not.
"""

from __future__ import annotations

import csv
import sys
import urllib.request

from dockops.util import load_config

DUDE_URL = "https://dude.docking.org/targets/{target}/{fname}"
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def parse_ism(text: str, label: int) -> list[tuple[str, str, int]]:
    """Parse a DUD-E .ism file into (compound_id, smiles, label) rows."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        smiles, compound_id = parts[0], parts[1] if len(parts) > 1 else parts[0]
        rows.append((compound_id, smiles, label))
    return rows


DOWNLOAD_TIMEOUT_S = 120


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dockops/0.1"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as r:
        return r.read()


def fetch(cfg: dict, out_csv: str, pdb_out: str) -> None:
    f = cfg["fetch"]
    target = f["dude_target"]
    rows = parse_ism(
        _download(DUDE_URL.format(target=target, fname="actives_final.ism")).decode(),
        label=1,
    )
    decoys = parse_ism(
        _download(DUDE_URL.format(target=target, fname="decoys_final.ism")).decode(),
        label=0,
    )
    n = f["max_decoys"]
    if n and len(decoys) > n:
        import random

        random.Random(f["seed"]).shuffle(decoys)
        decoys = decoys[:n]
    rows += decoys
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["compound_id", "smiles", "label"])
        w.writerows(rows)

    with open(pdb_out, "wb") as fh:
        fh.write(_download(RCSB_URL.format(pdb_id=f["receptor_pdb_id"])))
    print(
        f"fetch: {sum(1 for r in rows if r[2] == 1)} actives / "
        f"{sum(1 for r in rows if r[2] == 0)} decoys -> {out_csv}; "
        f"receptor {f['receptor_pdb_id']} -> {pdb_out}"
    )


def main() -> None:
    out_csv = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ligands_real.csv"
    pdb_out = sys.argv[2] if len(sys.argv) > 2 else "data/raw/receptor.pdb"
    fetch(load_config(), out_csv, pdb_out)


if __name__ == "__main__":
    main()
