# Project Guidance

- Use only public datasets (DAVIS via the GraphDTA GitHub mirror). Never
  commit raw downloads or feature matrices; commit only compact metrics
  and figures under `results/`.
- Cold-target split is the headline result; random-split numbers must be
  labeled as leaking target identity.
- Fusion claims require the modality ablations — never report the fused
  model alone.
- Report honest nuances (e.g. drug-only beating fusion on a secondary
  metric) rather than selecting the most flattering number.
- Keep `config/config.yaml` the single source of truth for model, split,
  and evaluation settings.
- Run `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` and
  `.venv/bin/snakemake -n` after changes. Snakemake does not track source
  edits — use `-F` to force reruns.
