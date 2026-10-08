"""Stage manifests: config, inputs and their hashes, code and environment revisions."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, Optional

from morph_ldl.config import PIPELINE_ROOT

PACKAGES = ("numpy", "pandas", "torch", "pyyaml", "languages-of-the-world")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(args: list, cwd: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def code_state() -> Dict[str, object]:
    repo = PIPELINE_ROOT
    status = _git(["status", "--porcelain", "--", "morph_ldl", "julia", "configs", "scripts", "tests"], repo)
    h = hashlib.sha256()
    for sub in ("morph_ldl", "julia/src", "julia/bin", "configs"):
        for p in sorted((PIPELINE_ROOT / sub).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".jl", ".yaml", ".toml"}:
                h.update(str(p.relative_to(PIPELINE_ROOT)).encode()); h.update(p.read_bytes())
    return {"git_commit": _git(["rev-parse", "HEAD"], repo),
            "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
            "pipeline_dirty": bool(status),
            "pipeline_source_sha256": h.hexdigest()}


def external_revisions() -> Dict[str, Optional[str]]:
    ext = PIPELINE_ROOT / "external"
    out = {}
    if ext.exists():
        for d in sorted(p for p in ext.iterdir() if (p / ".git").exists()):
            out[d.name] = _git(["rev-parse", "HEAD"], d)
    return out


def environment() -> Dict[str, object]:
    pk = {}
    for name in PACKAGES:
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    jl_manifest = PIPELINE_ROOT / "julia" / "Manifest.toml"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "packages": pk,
            "julia_manifest_sha256": sha256_file(jl_manifest) if jl_manifest.exists() else None}


class StageRecorder:
    """Context manager writing ``stage_manifest.json`` for one stage run."""

    def __init__(self, stage: str, cfg: dict, out_dir: Path, inputs: Iterable[Path] = (),
                 seeds: Optional[Dict[str, int]] = None):
        self.stage, self.cfg, self.out_dir = stage, cfg, Path(out_dir)
        self.inputs = [Path(p) for p in inputs]
        self.seeds = seeds or {}
        self.extra: Dict[str, object] = {}

    def __enter__(self) -> "StageRecorder":
        self.t0 = time.time()
        self.started = datetime.now(timezone.utc).isoformat()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "stage": self.stage,
            "status": "ok" if exc is None else f"error: {exc_type.__name__}: {exc}",
            "started_utc": self.started,
            "runtime_s": round(time.time() - self.t0, 3),
            "config_path": self.cfg.get("_config_path"),
            "config_hash": self.cfg.get("_config_hash"),
            "experiment_id": self.cfg["experiment"]["id"],
            "inputs": {str(p): sha256_file(p) for p in self.inputs if p.exists()},
            "seeds": self.seeds,
            "code": code_state(),
            "external": external_revisions(),
            "environment": environment(),
            **self.extra,
        }
        with open(self.out_dir / "stage_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False, default=str)
