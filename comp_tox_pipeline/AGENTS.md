# Project Guidance

- Use only public datasets (EPA ToxCast/Tox21, PubChem, ChEMBL). Never commit
  raw downloaded archives; commit only compact derived artifacts, metrics, and
  figures under `results/`.
- Do not claim regulatory validity, GxP compliance, or clinical/patient-safety
  assurance. All outputs are research-grade predictions with explicit
  applicability-domain limits.
- Never report random-split metrics as headline results. Scaffold-split numbers
  are the honest estimate of prospective performance on new chemotypes.
- Uncertainty claims must be backed by measured calibration (ECE) and conformal
  coverage on held-out scaffolds, not assumed.
- Run `python -m pytest tests/` and `snakemake -n` after changes.
- Keep `config/config.yaml` the single source of truth for endpoint, split, and
  evaluation settings; no hardcoded endpoints in `src/`.
- `src/comp_tox/claims.py` is vendored byte-identical with
  `bio-qc/statgen`, `bio-qc/scrna_qc`, and `llm-posttraining/evals`. The
  mol-ml root parity test pins the sha256. Edit all four copies together.
- `comp_tox.dossier` generates `docs/nam_dossier*.md` from a metrics
  artifact. Every number in the dossier must bind via `--verify`, and the
  limitations section is generated from artifact status fields, not
  hand-written.
