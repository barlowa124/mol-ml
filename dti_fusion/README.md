# dti-fusion

Multimodal drug-target interaction regression: **drug fingerprint + protein
language-model embedding -> kinase affinity** on DAVIS (30,056 drug-protein
pairs, 68 drugs x 442 kinases), evaluated on held-out proteins.

**Status: working demonstration.** Snakemake DAG runs fetch -> featurize ->
train -> evaluate. The demonstration is the fusion ablation: does combining the two
modalities beat either alone when the test proteins were never seen in
training?

## Design

- **Drug modality**: Morgan fingerprint (radius 2, 2048 bits, RDKit).
- **Protein modality**: mean-pooled ESM-2 (`esm2_t6_8M_UR50D`, 320-dim)
  embedding, a protein language model, not a hand-engineered descriptor.
  24.7% of pairs involve proteins longer than the 1024-token context. They
  are truncated and the truncation is logged (`frac_truncated` in
  `results/summary.json`).
- **Fusion**: each modality gets its own linear branch. Concatenated
  branches feed a shared MLP head regressing pKd (`-log10(Kd/1e9)`).
  DAVIS's 10 uM cap becomes the pKd 5.0 floor. 69.6% of pairs are at cap,
  `frac_at_cap` in `results/summary.json`).
- **Primary split: cold-target** (88 held-out proteins, zero overlap with
  training). Random split is reported secondarily and labeled leaky.
- **Ablations**: identical head trained drug-only and protein-only.

## Result (seed 11, committed in `results/summary.json`)

Cold-target holdout (the prospective number for "new protein, known drugs"):

| Model | MSE | MAE | Pearson | Spearman |
|---|---|---|---|---|
| **fusion** | **0.515** (0.48–0.55) | 0.460 | **0.601** | **0.538** |
| drug only | 0.664 (0.61–0.72) | **0.454** | 0.455 | 0.437 |
| protein only | 0.828 (0.79–0.87) | 0.755 | 0.255 | 0.203 |
| fusion, random split | 0.514 | 0.534 | 0.638 | 0.592 |


- **Fusion wins on ranking and MSE** on unseen proteins, since the two
  modalities carry complementary signal. That is the claim.
- Drug-only is a strong ablation (drug identity alone
  explains much of DAVIS) and edges fusion on MAE.
- Protein-only is weakest but non-trivial: the ESM-2 embedding alone ranks
  affinities on never-seen proteins at Spearman 0.20.
- Random vs cold-target Spearman (0.59 vs 0.54) shows the leakage cost is
  modest but real here. Cold-target is the headline number.

## Caveats

- DAVIS is 69.6% at-cap pairs (pKd 5.0 floor), so the task is partly
  "rank the binders vs the cap," not full-affinity regression.
- ESM-2 truncation means 25% of pairs use a partial protein sequence.
- One seed, one split, no hyperparameter search. Kept as a baseline.
- Cold-target holds out proteins only. All 68 drugs appear in training,
  so fusion and drug-only heads can still lean on memorized drug identity.

## Run

```bash
.venv/bin/snakemake -j1          # fetch -> features -> train -> report
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

`DTI_CONFIG` env var selects an alternate config. Outputs:
`results/summary.json`, `results/scatter.png`, `results/provenance.json`.
Raw downloads and feature matrices are regenerable and gitignored.

## Data

DAVIS via the GraphDTA mirror (`thinng/GraphDTA`, `data/davis/`):
`proteins.txt` (442 kinase sequences), `ligands_can.txt` (68 canonical
SMILES), `Y` (68x442 pickled Kd matrix, nM). Original study: Davis et al.,
Nat Biotechnol 2011.

## Related work

- [protein-design-ops](https://github.com/barlowa124/protein-design-ops) and [active-learning-loop](https://github.com/barlowa124/active-learning-loop) share the ESM-2 encoder used for target embeddings here.
