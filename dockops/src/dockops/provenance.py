"""Run provenance: every pipeline/API result carries who/what/how.

Records git commit + dirty flag, package versions, SHA-256 of inputs and
config, engine name, and UTC timestamp. The manifest is written next to the
outputs and embedded in API responses. It is part of the result, not a log.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import dockops

_TRACKED = ["numpy", "pandas", "scipy", "scikit-learn", "rdkit", "fastapi"]


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True
            ).stdout.strip()
        )
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def manifest(inputs: list[str], config_path: str, engine: str) -> dict:
    versions = {"dockops": dockops.__version__}
    for pkg in _TRACKED:
        try:
            versions[pkg] = version(pkg)
        except Exception:
            pass
    return {
        "git": _git_sha(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "versions": versions,
        "input_sha256": {p: _sha256(p) for p in inputs if Path(p).exists()},
        "config_sha256": _sha256(config_path) if Path(config_path).exists() else None,
    }


def write_manifest(m: dict, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(m, indent=2))
