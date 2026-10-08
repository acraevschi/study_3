"""Python bridge to the Julia LDL runner (julia/bin/run_jobs.jl). CONTRACT §6.

A job is a dict::

    {"train_csv": path, "queries_csv": path, "out_dir": path,
     "unit_id": str, "repetition": int, "fold": int, "overrides": {...}}

PCFP (known lexemes): ``train_csv`` holds the shown forms of the training verbs and
``queries_csv`` (lemma_id, target_cell[, item_set]) names hidden cells of verbs that are in
the training sample. No form of a queried cell is passed.

``run_ldl_jobs`` resolves each job's LDL config (``cfg["ldl"]`` + overrides + the
contract semantic seed), skips jobs whose outputs exist with a matching job-config hash,
shards the rest over ``n_procs`` Julia processes and returns the jobs' out_dirs.
Each out_dir receives ``predictions.csv``, ``diagnostics.json`` and, last, as the
completion marker, ``job_config.json``.

``score_mapping_jobs`` is the separate, gold-reading step (mapping_quality.csv). It is
never called by prediction code and refuses to run before predictions exist.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

from morph_ldl import seeds
from morph_ldl.config import PIPELINE_ROOT

JULIA_PROJECT = PIPELINE_ROOT / "julia"
JULIA_ENTRY = JULIA_PROJECT / "bin" / "run_jobs.jl"
RUNNER_VERSION = "ldl-runner-3-pcfp"     # keep in sync with LDLRunner.RUNNER_VERSION
VARIANT_SEP = " || "                     # = morph_ldl.cv.evaluate.VARIANT_SEP

# keys of cfg["ldl"] that are orchestration settings, not model settings
_NON_MODEL_KEYS = {"tune", "n_procs"}
JOB_KEYS = ("train_csv", "queries_csv", "out_dir", "unit_id", "repetition", "fold")


class LDLJobError(RuntimeError):
    """One or more LDL jobs failed; see <out_dir>/error.json."""


# ----------------------------------------------------------------------------- config

def resolve_ldl_config(cfg: Mapping, unit_id: str, repetition: int, fold: int,
                       overrides: Optional[Mapping] = None) -> Dict:
    """cfg['ldl'] (model keys only) + overrides + semantic_seed (CONTRACT §6/§8)."""
    base = {k: copy.deepcopy(v) for k, v in cfg["ldl"].items() if k not in _NON_MODEL_KEYS}
    for k, v in (overrides or {}).items():
        if k in ("semantic_seed",):
            raise ValueError("semantic_seed is derived from the master seed, not overridable")
        base[k] = copy.deepcopy(v)
    base["semantic_seed"] = seeds.derive(cfg["experiment"]["master_seed"], "semantic",
                                         unit_id, int(repetition), int(fold))
    return base


def _sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _julia_manifest_sha() -> str:
    p = JULIA_PROJECT / "Manifest.toml"
    return _sha256_file(p)[:16] if p.exists() else "none"


def job_config(job: Mapping, cfg: Mapping) -> Dict:
    """The full, hashable description of a job (written as job_config.json)."""
    missing = [k for k in JOB_KEYS if k not in job]
    if missing:
        raise KeyError(f"job lacks {missing}")
    resolved = resolve_ldl_config(cfg, job["unit_id"], job["repetition"], job["fold"],
                                  job.get("overrides"))
    body = {
        "runner_version": RUNNER_VERSION,
        "julia_manifest_sha": _julia_manifest_sha(),
        "unit_id": job["unit_id"], "repetition": int(job["repetition"]), "fold": int(job["fold"]),
        "overrides": dict(job.get("overrides") or {}),
        "ldl_config": resolved,
        "inputs": {"train_csv": str(Path(job["train_csv"]).resolve()),
                   "train_sha256": _sha256_file(job["train_csv"]),
                   "queries_csv": str(Path(job["queries_csv"]).resolve()),
                   "queries_sha256": _sha256_file(job["queries_csv"])},
    }
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    body["job_hash"] = hashlib.sha256(blob).hexdigest()[:16]
    return body


def job_is_done(out_dir: Union[str, Path], job_hash: str) -> bool:
    out = Path(out_dir)
    marker = out / "job_config.json"
    if not (marker.exists() and (out / "predictions.csv").exists() and (out / "diagnostics.json").exists()):
        return False
    try:
        return json.loads(marker.read_text())["job_hash"] == job_hash
    except Exception:
        return False


# ----------------------------------------------------------------------------- processes

def _resolve_parallelism(cfg: Mapping, n_procs: Optional[int], n_jobs: int) -> tuple[int, int]:
    limits = cfg.get("limits", {})
    configured = max(1, int(n_procs or cfg["ldl"].get("n_procs", 1)))
    procs = max(1, min(configured, n_jobs))
    max_threads = int(limits.get("max_threads", 8))
    threads = int(limits.get("julia_threads", 4))
    # Threads depend on the *configured* process count, not on how many jobs are still
    # pending, so resumed runs use the same thread count (main-agent fix after review).
    threads = max(1, min(threads, max_threads // configured))
    return procs, threads


def julia_executable() -> str:
    exe = os.environ.get("MORPH_LDL_JULIA") or shutil.which("julia")
    if not exe:
        raise FileNotFoundError("julia not found; set MORPH_LDL_JULIA")
    return exe


def _launch_shards(mode: str, shards: List[List[Dict]], threads: int, run_dir: Path) -> List[int]:
    run_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, shard in enumerate(shards):
        man = run_dir / f"{mode}_shard{i}.json"
        man.write_text(json.dumps({"mode": mode, "blas_threads": threads, "jobs": shard},
                                  ensure_ascii=False, indent=1))
        log = open(run_dir / f"{mode}_shard{i}.log", "w")
        cmd = [julia_executable(), f"--project={JULIA_PROJECT}", "--startup-file=no",
               "-t", str(threads), str(JULIA_ENTRY), str(man)]
        env = dict(os.environ, JULIA_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    codes = []
    for p, log in procs:
        codes.append(p.wait())
        log.close()
    return codes


def _shard(items: List[Dict], n: int) -> List[List[Dict]]:
    return [s for s in (items[i::n] for i in range(n)) if s]


def _default_run_dir(out_dirs: Sequence[Path]) -> Path:
    common = Path(os.path.commonpath([str(p.resolve()) for p in out_dirs]))
    return common / "_ldl_runs" / time.strftime("%Y%m%dT%H%M%S")


# ----------------------------------------------------------------------------- API

def run_ldl_jobs(jobs: Iterable[Mapping], cfg: Mapping, n_procs: Optional[int] = None,
                 force: bool = False, run_dir: Optional[Path] = None,
                 raise_on_error: bool = True) -> List[Path]:
    """Run LDL prediction jobs; return their out_dirs in input order (CONTRACT §6)."""
    jobs = list(jobs)
    outs = [Path(j["out_dir"]) for j in jobs]
    if len(set(map(str, outs))) != len(outs):
        raise ValueError("two jobs share an out_dir")
    todo = []
    for job, out in zip(jobs, outs):
        jc = job_config(job, cfg)
        if not force and job_is_done(out, jc["job_hash"]):
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "job_config.json").unlink(missing_ok=True)     # completion marker written last
        todo.append({"train_csv": str(Path(job["train_csv"]).resolve()),
                     "queries_csv": str(Path(job["queries_csv"]).resolve()),
                     "out_dir": str(out.resolve()), "config": jc["ldl_config"], "job_config": jc})
    if todo:
        procs, threads = _resolve_parallelism(cfg, n_procs, len(todo))
        _launch_shards("predict", _shard(todo, procs), threads, run_dir or _default_run_dir(outs))
        failed = [t["out_dir"] for t in todo if not job_is_done(t["out_dir"], t["job_config"]["job_hash"])]
        if failed and raise_on_error:
            raise LDLJobError(f"{len(failed)} LDL job(s) failed, e.g. {failed[0]} (see error.json / shard logs)")
    return outs


def score_mapping_jobs(jobs: Iterable[Mapping], gold_csv_by_job: Union[Sequence, Mapping],
                       cfg: Mapping, n_procs: Optional[int] = None,
                       run_dir: Optional[Path] = None) -> List[Path]:
    """Gold-side mapping diagnostics after prediction; writes <out_dir>/mapping_quality.csv.

    ``gold_csv_by_job``: list aligned with ``jobs`` or {out_dir: path}. Gold CSV columns:
    lemma_id, target_cell, gold_variants (segment strings joined by " || "); see
    ``write_gold_csv``. Returns the mapping_quality.csv paths."""
    jobs = list(jobs)
    todo, paths = [], []
    for i, job in enumerate(jobs):
        out = Path(job["out_dir"])
        gold = (gold_csv_by_job[str(job["out_dir"])] if isinstance(gold_csv_by_job, Mapping)
                else gold_csv_by_job[i])
        jc = job_config(job, cfg)
        if not job_is_done(out, jc["job_hash"]):
            raise LDLJobError(f"no finished predictions with matching config in {out}; "
                              "score_mapping_jobs runs only after run_ldl_jobs")
        sc = {"job_hash": jc["job_hash"], "gold_csv": str(Path(gold).resolve()),
              "gold_sha256": _sha256_file(gold)}
        target = out / "mapping_quality.csv"
        marker = out / "mapping_quality.json"
        paths.append(target)
        if target.exists() and marker.exists() and json.loads(marker.read_text()) == sc:
            continue
        marker.unlink(missing_ok=True)
        todo.append({"train_csv": jc["inputs"]["train_csv"], "queries_csv": jc["inputs"]["queries_csv"],
                     "out_dir": str(out.resolve()), "config": jc["ldl_config"],
                     "gold_csv": sc["gold_csv"], "score_config": sc})
    if todo:
        procs, threads = _resolve_parallelism(cfg, n_procs, len(todo))
        _launch_shards("score", _shard(todo, procs), threads,
                       run_dir or _default_run_dir([Path(t["out_dir"]) for t in todo]))
        failed = [t["out_dir"] for t in todo if not (Path(t["out_dir"]) / "mapping_quality.json").exists()]
        if failed:
            raise LDLJobError(f"{len(failed)} scoring job(s) failed, e.g. {failed[0]}")
    return paths


def write_gold_csv(forms: pd.DataFrame, queries: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Evaluation-side helper: gold variants (segments, variant order) for each query item.
    Must not be called by prediction code."""
    ok = forms[~forms["is_missing"].astype(bool)].sort_values(["lemma_id", "cell_norm", "variant_idx"])
    g = (ok.groupby(["lemma_id", "cell_norm"], sort=False)["segments"]
         .agg(lambda s: VARIANT_SEP.join(map(str, s))).rename("gold_variants").reset_index()
         .rename(columns={"cell_norm": "target_cell"}))
    out = queries[["lemma_id", "target_cell"]].merge(g, on=["lemma_id", "target_cell"], how="left")
    if out["gold_variants"].isna().any():
        raise KeyError("some query items have no gold form")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def read_predictions(out_dir: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(Path(out_dir) / "predictions.csv", dtype=str, keep_default_na=False)
