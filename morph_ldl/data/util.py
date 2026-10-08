"""Small shared helpers for the data stage: hashing, NFC, guarded writing, project imports."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from morph_ldl.config import PIPELINE_ROOT, resolve

MISSING_MARKERS = {"", "NA"}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def sha256_file(path: Path, prefix: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    return d[:prefix] if prefix else d


def resource_version(path: Path) -> str:
    """Contract §1: sha256:<first 16 hex> of the source file."""
    return "sha256:" + sha256_file(path, 16)


def repo_root(cfg: Dict[str, Any]) -> Path:
    return resolve(cfg, "repo_root")


# --------------------------------------------------------------------------
# Guarded writing: the data stage may only write below outputs/ (contract §0).
# --------------------------------------------------------------------------

class RawWriteError(RuntimeError):
    pass


def allowed_output_roots(cfg: Dict[str, Any]) -> list[Path]:
    out = resolve(cfg, "outputs")
    return [out / cfg["experiment"]["id"], out / "scratch_data"]


def guard_path(path: Path, cfg: Dict[str, Any]) -> Path:
    path = Path(path).resolve()
    for root in allowed_output_roots(cfg):
        root = root.resolve()
        if path == root or root in path.parents:
            return path
    raise RawWriteError(f"refusing to write outside the experiment outputs: {path}")


def write_csv(df: pd.DataFrame, path: Path, cfg: Dict[str, Any]) -> Path:
    path = guard_path(path, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def write_json(obj: Any, path: Path, cfg: Dict[str, Any]) -> Path:
    path = guard_path(path, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Read-only imports of project modules in <repo>/src (never modified).
# --------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=None)
def project_module(stem: str, root: str | None = None):
    """Load a vendored study_3 module (``morph_ldl/data/vendor/<stem>.py``) by path."""
    base = Path(root) if root else (Path(__file__).resolve().parent / "vendor")
    return _load_module(f"_study3_src_{stem}", base / f"{stem}.py")


@lru_cache(maxsize=None)
def project_cell_map(root: str | None = None) -> Dict[str, str]:
    base = Path(root) if root else (Path(__file__).resolve().parent / "resources")
    with open(base / "cells_to_unimorph.json", encoding="utf-8") as fh:
        return json.load(fh)


def git_state(path: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"commit": None, "dirty": None, "error": str(exc)}


def read_text_csv(path: Path, **kw) -> pd.DataFrame:
    """Read a CSV keeping every value as a literal string (NA stays 'NA')."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False,
                       encoding="utf-8", **kw)


def chunks(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
