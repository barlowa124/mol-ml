"""Fetch DAVIS (GraphDTA mirror) and assemble the pairs table.

Files: proteins.txt (target_id -> sequence JSON), ligands_can.txt
(drug_id -> canonical SMILES JSON), Y (pickled 68x442 Kd matrix in nM).
Affinity is stored as pKd = -log10(Kd/1e9), the DeepDTA convention: the
10000 nM cap becomes pKd 5.0, which is reported as the weak-binder floor.
"""

import io
import json
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from dti_fusion.config import load_config

TIMEOUT = 300


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return r.read()


def fetch_raw(base_url: str, raw_dir: str):
    out = Path(raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("proteins.txt", "ligands_can.txt", "Y"):
        p = out / name
        if not p.exists():
            tmp = p.with_suffix(p.suffix + ".part")
            tmp.write_bytes(_get(f"{base_url}/{name}"))
            tmp.rename(p)  # atomic: an interrupted fetch can't leave a
            # truncated file that later runs silently reuse
    return out


def load_pairs(raw_dir: str, kd_cap_nm: float) -> pd.DataFrame:
    raw = Path(raw_dir)
    proteins = json.loads((raw / "proteins.txt").read_text())
    ligands = json.loads((raw / "ligands_can.txt").read_text())
    Y = pickle.loads((raw / "Y").read_bytes(), encoding="latin1")
    drug_ids = list(ligands)
    target_ids = list(proteins)
    if Y.shape != (len(drug_ids), len(target_ids)):
        raise ValueError(
            f"affinity matrix {Y.shape} != {len(drug_ids)}x{len(target_ids)}"
        )
    rows = []
    for di, d in enumerate(drug_ids):
        for ti, t in enumerate(target_ids):
            kd = float(Y[di, ti])
            rows.append(
                {
                    "drug_id": d,
                    "smiles": ligands[d],
                    "target_id": t,
                    "sequence": proteins[t],
                    "kd_nm": kd,
                    "pkd": float(-np.log10(kd / 1e9)),
                    "at_cap": bool(kd >= kd_cap_nm),
                }
            )
    return pd.DataFrame(rows)


def main(raw_dir: str, out_parquet: str):
    cfg = load_config()
    fetch_raw(cfg["dataset"]["base_url"], raw_dir)
    df = load_pairs(raw_dir, cfg["dataset"]["kd_cap_nm"])
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(
        f"davis pairs: {len(df)} | drugs {df.drug_id.nunique()} "
        f"targets {df.target_id.nunique()} | at-cap {df.at_cap.mean():.1%}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
