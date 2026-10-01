# Model card: NR-AR activity classifier

Generated from the committed metrics and the trained model bundle by
`comp_tox.serve.modelcard`. Every number below is a measured artifact.

## Model

- Primary: `logistic_regression` on Morgan fingerprints
  (`morgan-r2-2048`), Platt-calibrated on validation scaffolds
- Compared on identical scaffold splits: `random_forest`, `gnn`, `mlp_tf`
- Trained on 5694 compounds (272 actives, prevalence 0.048). Evaluated on 713 held-out scaffolds (17 actives)

## Intended use

Research-grade prioritization signal for Tox21 NR-AR.
Not a regulatory determination, not a safety
assessment, not validated for clinical or GxP use. Compounds
flagged out-of-domain carry weaker measured performance (see below).

## Scaffold-split test performance

- AUROC 0.728 (95% CI [0.579, 0.861])
- AUPRC 0.357 (95% CI [0.132, 0.597], prevalence 0.024)
- ECE 0.009 (Platt-calibrated on validation scaffolds)

## Uncertainty and applicability domain

- Split-conformal at alpha=0.1: coverage 0.902,
  mean set size 0.91, qhat 0.023
- AD (Tanimoto nn <= 0.3): 0.198 of the eval set is in-domain.
  In-domain AUROC 0.920 vs. out-domain 0.702. Conformal coverage holds (0.908 in / 0.900 out).

## Serving and monitoring

`comp_tox.serve.app` exposes /predict (probability + conformal set +
AD flag) and /drift (batch shift vs. the frozen train reference
across nn-distance, bit-frequency JS divergence, and predicted-
positive rate). Drift warnings mean the measured metrics above
stop applying to that batch.

## Provenance

- features sha256 `390069d797f5...`, splits sha256 `447406c39109...`
- fingerprint `morgan-r2-2048`, split seed 0, conformal alpha 0.1
- Upstream data: EPA Tox21 (deepchem mirror); models: scikit-learn
  logistic regression / random forest, PyG GIN, TF/Keras MLP.

## Limitations

- Single endpoint, single public dataset. No prospective validation.
- In-domain fraction is low on held-out scaffolds; treat
  out-of-domain predictions as triage signals only.
- Conformal coverage is marginal, not conditional on chemotype.
