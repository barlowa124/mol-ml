# Project Guidance

- The mock engine exists to exercise pipeline mechanics. Never describe its
  scores as docking results; every output must carry `engine` provenance and
  mock-mode must be visible in metrics.
- Do not claim enrichment or virtual-screening performance until a real
  backend (Vina or other) is wired and run against a real benchmark set
  (DUD-E / LIT-PCBA), with the dataset name and version recorded.
- Fixture labels in `data/raw/ligands.csv` are illustrative; they exist to
  exercise the benchmark code path.
- Never commit receptor PDBQTs or benchmark archives unless they are the
  small committed fixtures; keep bulk data under `data/` gitignored.
- Provenance (git sha, package versions, input hashes, config hash) is
  mandatory on every pipeline and API result — do not bypass it.
- Run `python -m pytest tests/` and `snakemake -n` after changes.
- Research/education only; no clinical, regulatory, or safety claims.
