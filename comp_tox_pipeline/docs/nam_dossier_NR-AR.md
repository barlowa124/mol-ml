# Evidence dossier for NR-AR

## Provenance

This dossier covers endpoint NR-AR under a scaffold split, where held-out chemotypes are the headline estimate. The run used 0 bootstrap resamples. Metrics artifact sha256 is 36f368afc5d9d4d6a260f66a2ece685eb8a20f99d223857bb2e079fa9eb5d929 and config artifact sha256 is 4ad9f6dfab775f6c1d217767d8c74469f8b8756dc400213c9c911e407c71b346.

## Data

The split produced 5694 training rows against 711 validation and 713 test rows. The held-out set contains 17 actives at 2.38% prevalence.

## Performance

| model | AUROC | AUROC CI95 | AUPRC | AUPRC CI95 | ECE |
|---|---|---|---|---|---|
| logistic_regression (primary) | 0.7284 | [0.5789, 0.8611] | 0.3568 | [0.1322, 0.5968] | 0.0095 |
| random_forest | 0.8300 | [0.7094, 0.9276] | 0.3205 | [0.1245, 0.5908] | 0.0095 |
| gnn | 0.8309 | [0.7120, 0.9343] | 0.2983 | [0.0995, 0.5427] | 0.0192 |
| mlp_tf | 0.3445 | [0.1674, 0.5393] | 0.0200 | [0.0112, 0.0376] | 0.0095 |

## Uncertainty

The conformal miscoverage target is alpha = 0.1 and measured coverage on held-out scaffolds is 90.18%. The conformal quantile qhat is 0.0227. The mean prediction-set size is 0.9144 with a 91.44% singleton fraction.

## Applicability domain

The nearest-neighbor Tanimoto threshold is 0.3 and 19.78% of held-out compounds land in-domain, at a median nearest-neighbor distance of 0.5000. The primary model scores 0.9203 AUROC and 0.5841 AUPRC in-domain against 0.7016 AUROC and 0.3232 AUPRC out-of-domain. Conformal coverage holds at 90.78% in-domain and 90.03% out-of-domain.

## Limitations

- Bootstrap status: corrected (scaffold_cluster_with_replacement).
- Only 19.78% of held-out compounds fall inside the applicability domain. Predictions on novel chemotypes should be treated as out-of-domain by default.
- Test prevalence is 2.38%. AUPRC on a low-prevalence held-out set is noisy and should not be compared across endpoints.
- These outputs are research predictions with explicit applicability-domain limits. They carry no regulatory or GxP standing and no patient-safety standing. This dossier does not establish fitness for any regulatory submission.
