"""YAML config loading: `base:` inheritance, `--set a.b=value` overrides, paths relative to repo root."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if cfg.get("base"):
        cfg = _merge(load_config(path.parent / cfg.pop("base")), cfg)
    for o in overrides:
        key, val = o.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def resolve(p) -> Path:
    """Relative paths in configs are relative to the repo root, not the CWD."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
