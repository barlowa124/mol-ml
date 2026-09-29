"""Download the configured Tox21 endpoint from the public MoleculeNet mirror.

Tox21 (2014 challenge): ~12k compounds screened across 12 nuclear-receptor
and stress-response assays by NIH/EPA/FDA. The consolidated CSV is mirrored
publicly by DeepChem/MoleculeNet; we extract the configured endpoint column.

Raw archive is cached under data/raw/ (gitignored); the stage output is a
compact parquet of just the endpoint's labels.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request

DOWNLOAD_TIMEOUT_S = 300
from pathlib import Path

import pandas as pd

from comp_tox.util import load_config

RAW_ARCHIVE = Path("data/raw/tox21.csv.gz")


def download(url: str, dest: Path = RAW_ARCHIVE) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_name(dest.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
    return dest


def extract_endpoint(archive: Path, assay_id: str) -> pd.DataFrame:
    df = pd.read_csv(archive, usecols=["mol_id", "smiles", assay_id])
    df = df.rename(columns={assay_id: "label", "mol_id": "compound_id"})
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["assay_id"] = assay_id
    df["source"] = "tox21"
    return df[["compound_id", "smiles", "label", "assay_id", "source"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config()["endpoint"]
    archive = download(cfg["url"])
    df = extract_endpoint(archive, cfg["assay_id"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(
        f"{cfg['assay_id']}: {len(df)} labeled compounds "
        f"({int(df['label'].sum())} actives, {df['label'].mean():.1%})"
    )


if __name__ == "__main__":
    main()
