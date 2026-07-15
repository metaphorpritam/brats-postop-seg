"""Config loading — base + per-track deep merge (CLAUDE.md §5, §4.1/§4.2).

The experiment is a controlled A/B: the shared architecture and hyper-parameters
live in ``configs/base.yaml`` and are IDENTICAL across tracks (§4.2, guardrail #2),
while ``configs/track_<track>.yaml`` overrides only the data pipeline, loss, and
model-selection criterion that encode / fix defects D1–D9 (§3, §4.1).

``load_config`` reads base, reads the track file, drops its ``defaults`` provenance
key, then DEEP-merges the track over base (nested dicts merged recursively; scalars
and lists overridden wholesale) so a track need only state what it changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import REPO_ROOT

# Where base.yaml / track_<track>.yaml live (repo-relative; §5 layout).
CONFIG_DIR = REPO_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``; empty files yield an empty dict."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a top-level mapping, got {type(data).__name__}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``.

    Nested dicts are merged key-by-key; every non-dict value (scalars, lists) is
    overridden wholesale — e.g. a track's ``channels`` list replaces base's rather
    than concatenating. Base is not mutated.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_config(track: str) -> dict[str, Any]:
    """Return the deep-merged config for ``track`` ('a' or 'b').

    Reads ``configs/base.yaml`` and ``configs/track_<track>.yaml``, drops the track
    file's ``defaults`` key (it only records provenance of the base it extends), and
    deep-merges the track over base (CLAUDE.md §5).
    """
    base = _read_yaml(CONFIG_DIR / "base.yaml")
    override = _read_yaml(CONFIG_DIR / f"track_{track}.yaml")
    override.pop("defaults", None)
    return _deep_merge(base, override)
