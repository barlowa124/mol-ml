"""NAM evidence dossier: turn a metrics.json run into a reviewable document.

Every number printed in the dossier is bound to a recorded value — the
claims verifier (`comp_tox.claims`, vendored with statgen/scrna_qc/
llm-posttraining) checks each numeric token in the generated markdown
against the flattened metric and config leaves. A dossier containing a
number no artifact produced fails `--verify`.

The document is a New Approach Methodology style evidence package:
provenance, split protocol, per-model performance with intervals,
calibration and conformal coverage, applicability-domain limits, and an
explicit limitations section generated from the artifact's own status
fields (withdrawn intervals, low in-domain fraction, low prevalence).

    python -m comp_tox.dossier --metrics results/metrics.json \
        --config config/config.yaml --out docs/nam_dossier.md
    python -m comp_tox.dossier --metrics results/metrics.json \
        --config config/config.yaml --verify docs/nam_dossier.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from .claims import flatten_results, verify_markdown


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _f(v: float) -> str:
    """Metric value at the precision the claims binder tolerates."""
    return f"{v:.4f}"


def _pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _ci(bounds) -> str:
    lo, hi = bounds
    if lo is None or hi is None:
        return "not available"
    return f"[{_f(lo)}, {_f(hi)}]"


def build_dossier(metrics: dict, config: dict,
                  metrics_path: str, config_path: str) -> str:
    """Render the dossier. All displayed numbers come from `metrics` or
    `config`; `_f`/`_pct` keep every token inside binding tolerance."""
    ep = metrics["endpoint"]
    counts = metrics["counts"]
    split = config.get("split", {})
    method = split.get("method", "unknown")
    n_boot = metrics.get("n_bootstrap", 0)
    boot = metrics.get("bootstrap", {})
    conformal = metrics.get("conformal", {})
    ad = metrics.get("applicability_domain", {})
    primary = metrics.get("primary_model")
    models = metrics.get("models", {})

    L = []
    L.append(f"# Evidence dossier for {ep}")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(f"This dossier covers endpoint {ep} under a {method} split, "
             "where held-out chemotypes are the headline estimate. The "
             f"run used {n_boot} bootstrap resamples. Metrics artifact "
             f"sha256 is {_sha256(metrics_path)} and config artifact "
             f"sha256 is {_sha256(config_path)}.")
    L.append("")
    L.append("## Data")
    L.append("")
    L.append(f"The split produced {counts['train']} training rows "
             f"against {counts['valid']} validation and {counts['test']} "
             f"test rows. The held-out set contains "
             f"{counts['test_actives']} actives at "
             f"{_pct(counts['test_prevalence'])} prevalence.")
    L.append("")

    L.append("## Performance")
    L.append("")
    L.append("| model | AUROC | AUROC CI95 | AUPRC | AUPRC CI95 | ECE |")
    L.append("|---|---|---|---|---|---|")
    for name, m in models.items():
        marker = " (primary)" if name == primary else ""
        L.append(f"| {name}{marker} | {_f(m['auroc'])} | "
                 f"{_ci(m['auroc_ci95'])} | {_f(m['auprc'])} | "
                 f"{_ci(m['auprc_ci95'])} | {_f(m['ece'])} |")
    L.append("")

    L.append("## Uncertainty")
    L.append("")
    L.append(f"The conformal miscoverage target is alpha = "
             f"{conformal['alpha']} and measured coverage on held-out "
             f"scaffolds is {_pct(conformal['coverage'])}. The conformal "
             f"quantile qhat is {_f(conformal['qhat'])}. The mean "
             f"prediction-set size is {_f(conformal['mean_set_size'])} "
             f"with a {_pct(conformal['singleton_fraction'])} singleton "
             "fraction.")
    L.append("")

    L.append("## Applicability domain")
    L.append("")
    L.append(f"The nearest-neighbor Tanimoto threshold is "
             f"{ad['threshold']} and {_pct(ad['in_domain_fraction'])} of "
             "held-out compounds land in-domain, at a median "
             f"nearest-neighbor distance of {_f(ad['median_nn_distance'])}. "
             f"The primary model scores {_f(ad['in_domain']['auroc'])} "
             f"AUROC and {_f(ad['in_domain']['auprc'])} AUPRC in-domain "
             f"against {_f(ad['out_domain']['auroc'])} AUROC and "
             f"{_f(ad['out_domain']['auprc'])} AUPRC out-of-domain. "
             "Conformal coverage holds at "
             f"{_pct(ad['conformal_coverage_in_domain'])} in-domain and "
             f"{_pct(ad['conformal_coverage_out_domain'])} out-of-domain.")
    L.append("")

    L.append("## Limitations")
    L.append("")
    if boot.get("status") == "withdrawn":
        L.append(f"- Bootstrap intervals are withdrawn for this run. "
                 f"{boot['reason']} The interval columns are empty "
                 "accordingly.")
    elif boot.get("status"):
        L.append(f"- Bootstrap status: {boot['status']} "
                 f"({boot.get('method', 'n/a')}).")
    if ad.get("in_domain_fraction", 1.0) < 0.5:
        L.append(f"- Only {_pct(ad['in_domain_fraction'])} of held-out "
                 "compounds fall inside the applicability domain. "
                 "Predictions on novel chemotypes should be treated as "
                 "out-of-domain by default.")
    if counts.get("test_prevalence", 1.0) < 0.05:
        L.append(f"- Test prevalence is {_pct(counts['test_prevalence'])}. "
                 "AUPRC on a low-prevalence held-out set is noisy and "
                 "should not be compared across endpoints.")
    L.append("- These outputs are research predictions with explicit "
             "applicability-domain limits. They carry no regulatory or "
             "GxP standing and no patient-safety standing. This dossier "
             "does not establish fitness for any regulatory submission.")
    L.append("")
    return "\n".join(L)


def verify_dossier(path: str, metrics: dict, config: dict) -> dict:
    pool = flatten_results(metrics)
    for k, v in flatten_results(config).items():
        pool.setdefault(k, v)
    return verify_markdown(Path(path).read_text(encoding="utf-8"), pool)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="comp_tox.dossier")
    p.add_argument("--metrics", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", help="write the dossier here")
    p.add_argument("--verify", help="verify an existing dossier binds")
    p.add_argument("--claims-out",
                   help="write the claims binding report JSON")
    a = p.parse_args(argv)

    metrics = json.loads(Path(a.metrics).read_text())
    config = yaml.safe_load(Path(a.config).read_text())

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(build_dossier(metrics, config,
                                             a.metrics, a.config),
                               encoding="utf-8")
        print(f"wrote {a.out}")
    if a.verify:
        rep = verify_dossier(a.verify, metrics, config)
        if a.claims_out:
            Path(a.claims_out).write_text(json.dumps(rep, indent=2))
        print(json.dumps({k: rep[k] for k in
                          ("passed", "n_claims")}, indent=2))
        for u in rep["unbound_claims"]:
            print("unbound:", u["token"], "|", u["context"][:70])
        return 0 if rep["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
