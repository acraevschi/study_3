"""Configuration loading. Paths in the YAML are relative to the repository root."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    parent = cfg.pop("_inherit", None)
    if parent:  # shallow config inheritance: values here override the parent's
        base = load_config(path.parent / parent)
        base = {k: v for k, v in base.items() if not k.startswith("_")}
        cfg = deep_update(base, cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(cfg: Dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def resolve(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a cfg['paths'][key] entry against the pipeline root."""
    return (PIPELINE_ROOT / cfg["paths"][key]).resolve()


def output_dir(cfg: Dict[str, Any]) -> Path:
    d = resolve(cfg, "outputs") / cfg["experiment"]["id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def unit_cells(unit: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return {'source': cell_norm, 'panel': [(slot, cell_norm), ...]} for a unit."""
    cells = unit["cells"]
    task = cfg["task"]
    return {
        "source": cells[task["source_slot"]],
        "panel": [(slot, cells[slot]) for slot in task["panel_slots"]],
    }
