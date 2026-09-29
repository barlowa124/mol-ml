"""Structure readiness QC: AFDB confidence + agreement with experiment + MD sanity.

Answers "is this receptor model defensible to dock into?" for the EGFR
target: fetches the AFDB model for the configured UniProt ID, reports
pLDDT statistics, C-alpha RMSD vs the fetched experimental structure, and
an implicit-solvent minimization energy delta on that structure.

Writes one JSON artifact. Requires data/raw/receptor.pdb (run
`python -m dockops.fetch` first).
"""

from __future__ import annotations

import json
import sys

from dockops.util import load_config


def structure_qc(
    receptor_pdb: str,
    afdb_path: str,
    uniprot: str,
    out_path: str,
    config_path: str = "config/config.yaml",
) -> dict:
    from dockops.alphafold import ca_rmsd, fetch_afdb, plddt_stats
    from dockops.md import minimize
    from dockops.provenance import manifest

    url = fetch_afdb(uniprot, afdb_path)
    result = {
        "uniprot": uniprot,
        "afdb_source": url,
        "afdb_plddt": plddt_stats(afdb_path),
        "af_vs_experimental": ca_rmsd(afdb_path, receptor_pdb),
        "experimental_minimization": minimize(receptor_pdb),
        "provenance": manifest(
            [receptor_pdb, afdb_path], config_path, "structure_qc"
        ),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    receptor = sys.argv[1] if len(sys.argv) > 1 else "data/raw/receptor.pdb"
    afdb_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/afdb_model.pdb"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "results/structure_qc.json"
    uniprot = load_config()["fetch"]["uniprot"]
    print(json.dumps(structure_qc(receptor, afdb_path, uniprot, out_path), indent=2))


if __name__ == "__main__":
    main()
