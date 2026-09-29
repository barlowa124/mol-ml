PY ?= .venv/bin/python

.PHONY: install test dag run clean

install:
	uv pip install --python $(PY) -e .[dev]

test:
	$(PY) -m pytest tests/ -q

dag:
	$(PY) -m snakemake -n

run:
	$(PY) -m snakemake --cores 4

clean:
	rm -rf data/raw/* data/processed/* results/*
	touch data/raw/.gitkeep data/processed/.gitkeep results/.gitkeep
