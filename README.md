# mol-ml

Small-molecule machine learning over public datasets: toxicity, affinity,
docking. Three related projects merged into one repository, each a
self-contained package with its own tests and commit history (imported via
subtree merge).

## Packages

| Directory | What it does |
|---|---|
| `comp_tox_pipeline/` | Computational toxicology on EPA ToxCast/Tox21 + PubChem. Bemis-Murcko scaffold splits are the headline metric; conformal coverage and ECE reported on held-out scaffolds. |
| `dti_fusion/` | Drug-target interaction on DAVIS: fingerprint + ESM-2 fusion. Cold-target split is the headline result; random split labeled as target-identity leakage. |
| `dockops/` | Reproducible docking pipeline: batch ligand prep, Vina backend, mock engine for pipeline mechanics (labeled `engine="mock"` on every row). |

## Running tests

Each package is independent. From its directory:

```bash
cd comp_tox_pipeline && PYTHONPATH=src python -m pytest tests/ -q
```

Each subdirectory retains its own `AGENTS.md` with project-specific rules
(split policy, provenance requirements, mock-vs-real labeling), which still
apply.

## Why one repo

Same domain, same conventions. Public data only, structural splits as the
prospective estimate, provenance manifests on every result. One repo
keeps them consistent across three projects.
