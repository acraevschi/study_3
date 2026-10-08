"""LDL selector: a persistent Julia process (julia/bin/selector_server.jl) and the
uncertainty scores computed from its candidate supports (docs/SELECTION.md §PCFP).

Scores (declared in configs/pcfp_v1.yaml before any run; higher = more uncertain):

* per (semantic seed, candidate, shown cell), from the top ``max_can`` decoded candidates
  with synthesis-by-analysis supports s_1 >= s_2 >= ...:
    low_confidence  u = 1 - s_1
    high_entropy    H = -sum_i w_i log w_i,  w = softmax(s / T)  (T = entropy_temperature)
  A cell with no candidate (or a decoder error) gets s_1 = ``no_candidate_support`` (-1,
  so u = 2) and H = log(max_can), i.e. it counts as maximally uncertain; such cells are
  counted in ``n_nonfinite``.
* lemma score = mean over its pre-drawn shown cells, then mean over semantic seeds.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from morph_ldl import seeds as seedlib
from morph_ldl.ldl.runner import JULIA_PROJECT, julia_executable, resolve_ldl_config

SERVER = JULIA_PROJECT / "bin" / "selector_server.jl"
SCORE_COLUMNS = {"low_confidence": "u_low_confidence", "high_entropy": "u_entropy"}


# ----------------------------------------------------------------------------- configs

def selector_configs(cfg: Mapping, unit_id: str, repetition: int, fold: int, frozen: Mapping,
                     n_seeds: int) -> List[Dict]:
    """Resolved LDL configs for the selector: index 0 uses the evaluated LDL's semantic
    seed (CONTRACT §8); indices 1.. use ``selector_semantic`` seeds."""
    base = resolve_ldl_config(cfg, unit_id, repetition, fold, frozen)
    out = [base]
    master = int(cfg["experiment"]["master_seed"])
    for j in range(1, int(n_seeds)):
        c = dict(base)
        c["semantic_seed"] = seedlib.derive(master, "selector_semantic", unit_id, int(repetition), int(fold), j)
        out.append(c)
    return out


# ----------------------------------------------------------------------------- scores

def _softmax_entropy(supports: Sequence[float], temperature: float) -> float:
    s = np.asarray(supports, dtype=float) / float(temperature)
    s = s - s.max()
    w = np.exp(s)
    w = w / w.sum()
    w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def add_cell_scores(cells: pd.DataFrame, temperature: float, no_candidate_support: float,
                    max_can: int) -> pd.DataFrame:
    """Add u_low_confidence, u_entropy and a `scored` flag to the server's cell table."""
    out = cells.copy()
    sup_lists = [json.loads(s) if isinstance(s, str) and s else [] for s in out["supports"]]
    ok = (out["status"].astype(str) == "ok") & np.array([len(x) > 0 for x in sup_lists])
    top = np.array([x[0] if len(x) else np.nan for x in sup_lists], dtype=float)
    out["u_low_confidence"] = np.where(ok, 1.0 - top, 1.0 - float(no_candidate_support))
    out["u_entropy"] = [(_softmax_entropy(x, temperature) if good else math.log(max_can))
                        for x, good in zip(sup_lists, ok)]
    out["scored"] = ok
    return out


def lemma_scores(cells: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Mean over shown cells per seed, then over seeds. Columns follow ACQ_LOG_COLUMNS."""
    col = SCORE_COLUMNS[policy]
    per_seed = (cells.groupby(["lemma_id", "semantic_seed_idx"])
                .agg(score=(col, "mean"), n_cells=(col, "size"), n_bad=("scored", lambda s: int((~s).sum())))
                .reset_index())
    agg = (per_seed.groupby("lemma_id")
           .agg(lemma_score=("score", "mean"), n_cells_scored=("n_cells", "first"),
                n_nonfinite=("n_bad", "sum"), score_seed_sd=("score", "std"), n_seeds=("score", "size"))
           .reset_index())
    return agg.sort_values("lemma_id").reset_index(drop=True)


# ----------------------------------------------------------------------------- server

class SelectorServerError(RuntimeError):
    pass


class SelectorServer:
    """One Julia selector process; use as a context manager."""

    def __init__(self, log_path: Path, threads: int = 1):
        self.log_path = Path(log_path)
        self.threads = int(threads)
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "SelectorServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "a")
        env = dict(os.environ, JULIA_NUM_THREADS=str(self.threads), OPENBLAS_NUM_THREADS=str(self.threads))
        cmd = [julia_executable(), f"--project={JULIA_PROJECT}", "--startup-file=no", "-t", str(self.threads),
               str(SERVER)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                     text=True, bufsize=1, env=env)
        self.info = self.request({"cmd": "ping"})
        return self

    def request(self, msg: Mapping) -> Dict:
        if self.proc is None or self.proc.poll() is not None:
            raise SelectorServerError(f"selector process not running (see {self.log_path})")
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise SelectorServerError(f"selector process exited (see {self.log_path})")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise SelectorServerError(reply.get("error", "unknown error"))
        return reply

    def score(self, train_csv: Path, candidates_csv: Path, out_cells: Path, out_comp: Path,
              configs: Sequence[Mapping]) -> Dict:
        return self.request({"cmd": "score", "train_csv": str(Path(train_csv).resolve()),
                             "candidates_csv": str(Path(candidates_csv).resolve()),
                             "out_cells_csv": str(Path(out_cells).resolve()),
                             "out_comp_csv": str(Path(out_comp).resolve()),
                             "configs": list(configs), "blas_threads": self.threads})

    def __exit__(self, *exc) -> None:
        try:
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.request({"cmd": "quit"})
                except Exception:
                    pass
                self.proc.wait(timeout=60)
        finally:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.kill()
            self._log.close()
