PY = ".venv/bin/python"
PP = "PYTHONPATH=src"


rule all:
    input:
        "results/summary.json",
        "results/scatter.png",


rule fetch:
    output:
        directory("data/raw"),
    shell:
        "{PP} {PY} -c \"from dti_fusion.data import fetch_raw; "
        "from dti_fusion.config import load_config; "
        "fetch_raw(load_config()['dataset']['base_url'], '{output}')\""


rule prepare:
    input:
        rules.fetch.output,
    output:
        "data/processed/pairs.parquet",
    shell:
        "{PP} {PY} -m dti_fusion.data {input} {output}"


rule features:
    input:
        rules.prepare.output,
    output:
        "data/processed/features.npz",
    shell:
        "{PP} {PY} -m dti_fusion.features {input} {output}"


rule report:
    input:
        rules.features.output,
    output:
        "results/summary.json",
        "results/scatter.png",
    shell:
        "{PP} {PY} -m dti_fusion.train {input} {output[0]} {output[1]}"
