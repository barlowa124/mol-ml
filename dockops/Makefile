PY ?= .venv/bin/python

.PHONY: install test dag run serve clean

install:
	uv pip install --python $(PY) -e .[dev]

test:
	$(PY) -m pytest tests/ -q

dag:
	$(PY) -m snakemake -n

run:
	$(PY) -m snakemake --cores 2

serve:
	$(PY) -m uvicorn dockops.api:app --port 8000

clean:
	rm -rf data/processed/* results/*
	touch data/processed/.gitkeep results/.gitkeep
