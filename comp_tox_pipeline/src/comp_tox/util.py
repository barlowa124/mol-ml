"""Shared config loading: config/config.yaml is the single source of truth.

COMPTOX_CONFIG env var overrides the path so alternate configs can drive
the same DAG entry points.
"""

from __future__ import annotations

import os

import yaml

DEFAULT_CONFIG = "config/config.yaml"
ENV_VAR = "COMPTOX_CONFIG"


def config_path() -> str:
    return os.environ.get(ENV_VAR, DEFAULT_CONFIG)


def load_config(path: str | None = None) -> dict:
    with open(path or config_path()) as f:
        return yaml.safe_load(f)
