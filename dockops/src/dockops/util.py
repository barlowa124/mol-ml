"""Shared config loading."""

from __future__ import annotations

import yaml

DEFAULT_CONFIG = "config/config.yaml"


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
