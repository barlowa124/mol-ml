# Evidence dossier for NR-ER

## Provenance

This dossier covers endpoint NR-ER under a scaffold split, where held-out chemotypes are the headline estimate. The run used 0 bootstrap resamples. Metrics artifact sha256 is e48a46665e6e7badff56fd135fd2a17583bca645e4124109bfe9d11a496d58f4 and config artifact sha256 is 4ad9f6dfab775f6c1d217767d8c74469f8b8756dc400213c9c911e407c71b346.

## Data

The split produced 4872 training rows against 609 validation and 610 test rows. The held-out set contains 55 actives at 9.02% prevalence.

## Performance

| model | AUROC | AUROC CI95 | AUPRC | AUPRC CI95 | ECE |
|---|---|---|---|---|---|
| logistic_regression (primary) | 0.6437 | not available | 0.2445 | not available | 0.0328 |
| random_forest | 0.7261 | not available | 0.2740 | not available | 0.0445 |
| gnn | 0.6657 | not available | 0.2487 | not available | 0.0403 |
| mlp_tf | 0.6788 | not available | 0.3136 | not available | 0.0528 |

## Uncertainty

The conformal miscoverage target is alpha = 0.1 and measured coverage on held-out scaffolds is 92.95%. The conformal quantile qhat is 0.8322. The mean prediction-set size is 1.0443 with a 95.57% singleton fraction.

## Applicability domain

The nearest-neighbor Tanimoto threshold is 0.3 and 17.87% of held-out compounds land in-domain, at a median nearest-neighbor distance of 0.5026. The primary model scores 0.7510 AUROC and 0.3249 AUPRC in-domain against 0.6186 AUROC and 0.2384 AUPRC out-of-domain. Conformal coverage holds at 93.58% in-domain and 92.81% out-of-domain.

## Limitations

- Bootstrap intervals are withdrawn for this run. The original sampler discarded repeated scaffold draws. Original NR-ER checkpoint and test predictions are unavailable, so corrected intervals cannot be recovered for this run. Point estimates are retained from the original artifact. The interval columns are empty accordingly.
- Only 17.87% of held-out compounds fall inside the applicability domain. Predictions on novel chemotypes should be treated as out-of-domain by default.
- These outputs are research predictions with explicit applicability-domain limits. They carry no regulatory or GxP standing and no patient-safety standing. This dossier does not establish fitness for any regulatory submission.
