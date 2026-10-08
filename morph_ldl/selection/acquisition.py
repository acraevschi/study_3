"""Pool-based lemma acquisition with gold isolation (docs/CONTRACT.md §5).

Information flow
----------------
* ``CandidatePoolView`` is built from the variant-0 *source-cell* rows of the pool lemmas
  only. It exposes ``CandidateQuery`` objects (lemma id, source form/segments, source
  cell, target cells) and never holds a target form.
* ``Oracle`` holds the gold rows of the seed and pool lemmas. ``select_and_reveal``
  records the lemmas as selected (with round and rank) and only then returns their rows;
  ``reveal`` of an unselected lemma raises ``GoldAccessError``. Every reveal is logged
  (``oracle_reveals.csv``). Oracle *policies* (``oracle_*``) read candidate gold through
  ``peek_for_oracle_policy``, which is logged with reason ``oracle_policy_scoring`` and
  is unavailable to the substantive policies.
* The selector is trained only on revealed rows (seed + selected). Dev gold is read once
  for early stopping (logged as ``dev_early_stopping``); dev lemmas never become
  training data. Outer-test lemmas are never passed in (``test_ids`` is only checked).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from morph_ldl import seeds as seedlib
from morph_ldl.schemas import (ACQ_LOG_COLUMNS, CELL_SCORE_COLUMNS, FORMS_COLUMNS, ORDER_COLUMNS,
                               TEST_QUERY_COLUMNS)
from morph_ldl.selection.model import (Example, Hypothesis, SelectorConfig, TrainedSelector, encode_input,
                                       split_segments, train_selector)
from morph_ldl.selection.scoring import (SCORE_NAMES, CellScore, aggregate_lemmas, cell_summary_frame,
                                         rank_lemmas, score_cell)

UPSTREAM_REVISION = "3caf0d059846569ed0aff4b833492c104786c8e1"  # smuradoglu/ALmorphinfl (logic rewritten)
POLICIES = ("random", "low_confidence", "high_entropy", "oracle_incorrect")
ORACLE_POLICIES = ("oracle_incorrect",)
REVEAL_COLUMNS = ["round", "lemma_id", "reason", "acquisition_rank", "n_rows"]


class GoldAccessError(RuntimeError):
    """Gold forms were requested for a lemma that has not been selected."""


# ----------------------------------------------------------------------------- task / seeds

@dataclass(frozen=True)
class SelectionTask:
    unit_id: str
    source_cell: str
    panel_cells: Tuple[str, ...]
    panel_slots: Tuple[str, ...] = ()
    training_mode: str = "panel"  # panel | all_cells

    def __post_init__(self):
        if self.source_cell in self.panel_cells:
            raise ValueError("source cell cannot be a panel target")
        if len(set(self.panel_cells)) != len(self.panel_cells):
            raise ValueError("panel cells repeat")
        if self.training_mode not in ("panel", "all_cells"):
            raise ValueError(f"unknown training_mode {self.training_mode!r}")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], unit_id: str) -> "SelectionTask":
        unit = next(u for u in cfg["units"] if u["unit_id"] == unit_id)
        t = cfg["task"]
        slots = tuple(t["panel_slots"])
        return cls(unit_id=unit_id, source_cell=unit["cells"][t["source_slot"]],
                   panel_cells=tuple(unit["cells"][s] for s in slots), panel_slots=slots,
                   training_mode=t.get("training_mode", "panel"))


@dataclass(frozen=True)
class SelectionSeeds:
    selector_init: int
    random_policy: int
    tie: int

    @classmethod
    def derive(cls, master: int, unit_id: str, repetition: int, outer_fold: int) -> "SelectionSeeds":
        k = (unit_id, repetition, outer_fold)
        return cls(selector_init=seedlib.derive(master, "selector_init", *k),
                   random_policy=seedlib.derive(master, "random_policy", *k),
                   tie=seedlib.derive(master, "tie", *k))


def policy_dir(outputs_root: Path, unit_id: str, repetition: int, outer_fold: int, policy: str) -> Path:
    """``<outputs_root>/selection/<unit_id>/rep{r}/fold{k}/<policy>/`` (CONTRACT §5)."""
    return Path(outputs_root) / "selection" / unit_id / f"rep{repetition}" / f"fold{outer_fold}" / policy


# ----------------------------------------------------------------------------- pool / oracle

@dataclass(frozen=True)
class CandidateQuery:
    lemma_id: str
    source_form: str
    source_segments: Tuple[str, ...]
    source_cell: str
    target_cells: Tuple[str, ...]


def _source_rows(forms: pd.DataFrame, lemma_ids: Iterable[str], source_cell: str) -> pd.DataFrame:
    ids = set(lemma_ids)
    src = forms[forms["lemma_id"].isin(ids) & (forms["cell_norm"] == source_cell)
                & (forms["variant_idx"] == 0) & (~forms["is_missing"].astype(bool))]
    src = src.drop_duplicates("lemma_id")
    missing = sorted(ids - set(src["lemma_id"]))
    if missing:
        raise ValueError(f"{len(missing)} lemmas lack a source anchor, e.g. {missing[:3]}")
    return src


class CandidatePoolView:
    """Read-only view of the candidate pool: source anchors only, no target forms."""

    def __init__(self, queries: Sequence[CandidateQuery]):
        self._q: Dict[str, CandidateQuery] = {q.lemma_id: q for q in queries}

    @classmethod
    def from_forms(cls, forms: pd.DataFrame, pool_ids: Iterable[str], task: SelectionTask) -> "CandidatePoolView":
        src = _source_rows(forms, pool_ids, task.source_cell)
        qs = [CandidateQuery(r.lemma_id, str(r.form), split_segments(str(r.segments)), task.source_cell,
                             tuple(task.panel_cells)) for r in src.itertuples(index=False)]
        return cls(qs)

    @property
    def lemma_ids(self) -> List[str]:
        return sorted(self._q)

    def __len__(self) -> int:
        return len(self._q)

    def queries(self, exclude: Iterable[str] = ()) -> List[CandidateQuery]:
        ex = set(exclude)
        return [self._q[k] for k in sorted(self._q) if k not in ex]


class Oracle:
    """Holds gold rows for seed + pool lemmas; reveals a lemma only once selected."""

    def __init__(self, forms: pd.DataFrame, lemma_ids: Iterable[str]):
        ids = set(lemma_ids)
        self._rows = forms[forms["lemma_id"].isin(ids)].copy()
        self._ids = ids
        self._selected: Dict[str, Tuple[int, int]] = {}
        self.log: List[dict] = []

    @property
    def selected(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._selected)

    def _rows_for(self, lemma_ids: Sequence[str]) -> pd.DataFrame:
        return self._rows[self._rows["lemma_id"].isin(set(lemma_ids))].copy()

    def select_and_reveal(self, lemma_ids: Sequence[str], round_: int, first_rank: int,
                          reason: str = "selected") -> pd.DataFrame:
        for j, lid in enumerate(lemma_ids):
            if lid not in self._ids:
                raise GoldAccessError(f"{lid!r} is not a seed/pool lemma of this oracle")
            if lid in self._selected:
                raise ValueError(f"{lid!r} selected twice")
            self._selected[lid] = (round_, first_rank + j)
        rows = self._rows_for(lemma_ids)
        counts = rows.groupby("lemma_id").size().to_dict()
        for lid in lemma_ids:
            self.log.append({"round": round_, "lemma_id": lid, "reason": reason,
                             "acquisition_rank": self._selected[lid][1], "n_rows": int(counts.get(lid, 0))})
        return rows

    def reveal(self, lemma_ids: Sequence[str]) -> pd.DataFrame:
        bad = [l for l in lemma_ids if l not in self._selected]
        if bad:
            raise GoldAccessError(f"gold requested for {len(bad)} unselected lemma(s), e.g. {bad[:3]}")
        return self._rows_for(lemma_ids)

    def peek_for_oracle_policy(self, lemma_ids: Sequence[str], round_: int) -> pd.DataFrame:
        """ORACLE POLICIES ONLY: candidate gold before selection (logged)."""
        rows = self._rows_for(lemma_ids)
        counts = rows.groupby("lemma_id").size().to_dict()
        for lid in lemma_ids:
            self.log.append({"round": round_, "lemma_id": lid, "reason": "oracle_policy_scoring",
                             "acquisition_rank": -1, "n_rows": int(counts.get(lid, 0))})
        return rows

    def reveal_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=REVEAL_COLUMNS)


# ----------------------------------------------------------------------------- examples

def _v0(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows[(rows["variant_idx"] == 0) & (~rows["is_missing"].astype(bool))
              & (rows["cell_norm"].fillna("") != "") & (rows["segments"].fillna("") != "")]
    return ok.drop_duplicates(["lemma_id", "cell_norm"])


def build_examples(rows: pd.DataFrame, task: SelectionTask, mode: Optional[str] = None,
                   with_gold_variants: bool = False) -> Tuple[List[Example], Dict[str, int]]:
    """Training (or dev) examples: source variant 0 -> each target cell, variant 0.

    ``mode='panel'``: the fixed panel cells. ``mode='all_cells'``: every non-missing cell
    of the lemma except the source cell. Returns (examples, stats).
    """
    mode = mode or task.training_mode
    v0 = _v0(rows)
    src = v0[v0["cell_norm"] == task.source_cell].set_index("lemma_id")
    variants: Dict[Tuple[str, str], List[Tuple[str, ...]]] = {}
    if with_gold_variants:
        ok = rows[~rows["is_missing"].astype(bool)].sort_values(["lemma_id", "cell_norm", "variant_idx"])
        for (lid, cell), sub in ok.groupby(["lemma_id", "cell_norm"], sort=False):
            variants[(lid, cell)] = [split_segments(str(s)) for s in sub["segments"]]
    exs: List[Example] = []
    n_missing_target = 0
    panel = set(task.panel_cells)
    for r in v0.itertuples(index=False):
        cell = r.cell_norm
        if cell == task.source_cell or (mode == "panel" and cell not in panel):
            continue
        if r.lemma_id not in src.index:
            continue
        s_segs = split_segments(str(src.at[r.lemma_id, "segments"]))
        exs.append(Example(src=encode_input(task.source_cell, s_segs, cell), tgt=split_segments(str(r.segments)),
                           lemma_id=r.lemma_id, target_cell=cell, n_source_segments=len(s_segs),
                           gold_variants=tuple(variants.get((r.lemma_id, cell), ()))))
    lemmas = set(rows["lemma_id"])
    if mode == "panel":
        have = {(e.lemma_id, e.target_cell) for e in exs}
        n_missing_target = sum((l, c) not in have for l in lemmas for c in task.panel_cells)
    exs.sort(key=lambda e: (e.lemma_id, e.target_cell))
    stats = {"n_lemmas": len(lemmas), "n_examples": len(exs),
             "n_target_cells_distinct": len({e.target_cell for e in exs}),
             "n_missing_panel_targets": int(n_missing_target), "mode": mode}
    return exs, stats


def copy_examples(queries: Iterable["CandidateQuery"]) -> List[Example]:
    """Auxiliary autoencoding items (source cell -> source cell) from source anchors only."""
    out = [Example(src=encode_input(q.source_cell, q.source_segments, q.source_cell), tgt=tuple(q.source_segments),
                   lemma_id=q.lemma_id, target_cell=q.source_cell, n_source_segments=len(q.source_segments))
           for q in queries]
    return sorted(out, key=lambda e: e.lemma_id)


def aux_copy_examples(mode: str, train_ids: Sequence[str], seed_pool_view: Optional["CandidatePoolView"]) -> List[Example]:
    if mode == "none" or seed_pool_view is None:
        return []
    ids = set(train_ids) if mode == "train" else None
    return copy_examples(q for q in seed_pool_view.queries() if ids is None or q.lemma_id in ids)


# ----------------------------------------------------------------------------- scoring

def beam_candidates(selector: TrainedSelector, queries: Sequence[CandidateQuery],
                    beam_size: Optional[int] = None) -> Dict[Tuple[str, str], List[Hypothesis]]:
    """Beam hypotheses for every (candidate, target cell). Sees only ``CandidateQuery``."""
    keys, srcs, lens = [], [], []
    for q in queries:
        for cell in q.target_cells:
            keys.append((q.lemma_id, cell))
            srcs.append(encode_input(q.source_cell, q.source_segments, cell))
            lens.append(len(q.source_segments))
    beams = selector.beam_search(srcs, lens, beam_size=beam_size)
    return dict(zip(keys, beams))


def score_candidates(selector: TrainedSelector, queries: Sequence[CandidateQuery],
                     min_prob: float = 0.05, beam_size: Optional[int] = None) -> List[CellScore]:
    beams = beam_candidates(selector, queries, beam_size)
    return [score_cell(lid, cell, hyps, min_prob) for (lid, cell), hyps in beams.items()]


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def oracle_incorrect_scores(cells: Sequence[CellScore], gold_rows: pd.DataFrame) -> Dict[str, float]:
    """ORACLE: mean over panel cells of the normalised edit distance between the top
    hypothesis and the closest gold variant (0 = correct)."""
    ok = gold_rows[~gold_rows["is_missing"].astype(bool)]
    gold: Dict[Tuple[str, str], List[Tuple[str, ...]]] = {}
    for r in ok.itertuples(index=False):
        gold.setdefault((r.lemma_id, r.cell_norm), []).append(split_segments(str(r.segments)))
    per: Dict[str, List[float]] = {}
    for c in cells:
        golds = gold.get((c.lemma_id, c.target_cell))
        if not golds:
            continue
        hyp = split_segments(c.top_hyp)
        d = min(_levenshtein(hyp, g) / max(1, len(g)) for g in golds)
        per.setdefault(c.lemma_id, []).append(d)
    return {lid: float(np.mean(v)) for lid, v in per.items()}


# ----------------------------------------------------------------------------- loop

@dataclass
class AcquisitionResult:
    policy: str
    out_dir: Path
    order: pd.DataFrame
    rounds: List[dict]
    sample_paths: Dict[int, Dict[str, Path]]
    summary: Dict[str, Any]
    selectors: Dict[int, TrainedSelector] = field(default_factory=dict)


def round_plan(seed_size: int, budgets: Sequence[int], batch_size: int, pool_size: int) -> Dict[str, Any]:
    budgets = sorted({int(b) for b in budgets})
    if budgets[0] < seed_size:
        raise ValueError(f"budget {budgets[0]} < seed size {seed_size}")
    target = budgets[-1]
    attainable = seed_size + pool_size
    shortfall = max(0, target - attainable)
    to_acquire = min(target, attainable) - seed_size
    sizes = [batch_size] * (to_acquire // batch_size)
    if to_acquire % batch_size:
        sizes.append(to_acquire % batch_size)
    return {"budgets": budgets, "batch_sizes": sizes, "remainder_last_round": bool(to_acquire % batch_size),
            "shortfall": shortfall, "attainable_max": min(target, attainable)}


def _git_state() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
                                    timeout=20).stdout.strip())
    except Exception:  # pragma: no cover
        commit, dirty = "", None
    return {"git_commit": commit, "git_dirty": dirty}


def _frame_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes())
    return "sha256:" + h.hexdigest()[:16]


def run_acquisition(policy: str, seed_ids: Sequence[str], pool_ids: Sequence[str], dev_ids: Sequence[str],
                    forms_df: pd.DataFrame, task: SelectionTask, cfg: Mapping[str, Any], seeds: SelectionSeeds,
                    out_dir: Path, *, test_ids: Sequence[str] = (), train_random_rounds: Optional[bool] = None,
                    final_fit: bool = False, keep_selectors: bool = False,
                    log: Optional[Callable[[str], None]] = None,
                    anchor_ids: Optional[Sequence[str]] = None) -> AcquisitionResult:
    """Run one policy on one (unit, repetition, outer fold) and write CONTRACT §5 outputs
    to ``out_dir`` (the policy directory, see ``policy_dir``).

    ``cfg`` needs ``selection`` (batch_size, budgets, entropy_min_prob) and ``selector``
    (+ optional ``limits.max_threads``). ``train_random_rounds`` (default
    ``cfg['selection'].get('random_trains_selector', False)``) fits the selector in each
    random round too, only to report matched dev-accuracy curves; it never affects the
    random choice. ``final_fit`` trains once more on the max-budget sample (dev accuracy
    only). ``anchor_ids`` (main-agent integration): lemmas whose *source anchors* feed the
    ``aux_copy: pool`` copy items; default seed + pool. The pipeline passes a fixed set
    (seed + first pool lemmas up to the smallest pool cap) so that copy items are identical
    across policies and pool caps.
    """
    t_start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    say = log or (lambda s: None)
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    seed_ids, pool_ids, dev_ids = list(seed_ids), list(pool_ids), list(dev_ids)
    sets = {"seed": set(seed_ids), "pool": set(pool_ids), "dev": set(dev_ids), "test": set(test_ids)}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise ValueError(f"{a} and {b} lemma sets overlap ({len(sets[a] & sets[b])} lemmas)")
    for name, ids in (("seed", seed_ids), ("pool", pool_ids), ("dev", dev_ids)):
        if len(sets[name]) != len(ids):
            raise ValueError(f"duplicate lemma ids in {name}")
    sel_cfg = cfg["selection"]
    scfg = SelectorConfig.from_cfg(cfg)
    batch_size = int(sel_cfg["batch_size"])
    min_prob = float(sel_cfg.get("entropy_min_prob", 0.05))
    if sel_cfg.get("aggregation", "mean_cell") != "mean_cell":
        raise ValueError("only aggregation=mean_cell is implemented")
    if train_random_rounds is None:
        train_random_rounds = bool(sel_cfg.get("random_trains_selector", False))
    plan = round_plan(len(seed_ids), sel_cfg["budgets"], batch_size, len(pool_ids))
    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Only the rows of this fold's seed/pool/dev lemmas are kept; test rows are dropped here.
    forms = forms_df[forms_df["lemma_id"].isin(sets["seed"] | sets["pool"] | sets["dev"])]
    pool = CandidatePoolView.from_forms(forms, pool_ids, task)
    anchor_list = list(anchor_ids) if anchor_ids is not None else seed_ids + pool_ids
    if set(anchor_list) & (sets["dev"] | sets["test"]):
        raise ValueError("anchor_ids must not include dev or test lemmas")
    # Source anchors only (aux copy). Anchors may lie outside this fold's seed/pool (the
    # pipeline uses non-inventory lemmas), so they are looked up in the full table; only
    # their source cell is read.
    anchor_src = forms_df[forms_df["lemma_id"].isin(set(anchor_list))
                          & (forms_df["cell_norm"] == task.source_cell)]
    anchors = CandidatePoolView.from_forms(anchor_src, anchor_list, task)
    _source_rows(forms, seed_ids, task.source_cell)
    oracle = Oracle(forms, seed_ids + pool_ids)
    dev_rows = forms[forms["lemma_id"].isin(sets["dev"])]
    dev_examples, dev_stats = build_examples(dev_rows, task, mode="panel", with_gold_variants=True)
    dev_log = [{"round": 0, "lemma_id": l, "reason": "dev_early_stopping", "acquisition_rank": -1,
                "n_rows": int(n)} for l, n in dev_rows.groupby("lemma_id").size().items()]

    seed_sorted = sorted(seed_ids, key=lambda l: seedlib.tie_key(seeds.tie, l))
    oracle.select_and_reveal(seed_sorted, 0, 1, reason="seed")
    order_rows = [{"lemma_id": l, "acquisition_rank": i + 1, "round": 0, "lemma_score": float("nan"),
                   "score_name": "seed"} for i, l in enumerate(seed_sorted)]
    selected: List[str] = list(seed_sorted)

    rng_scores: Dict[str, float] = {}
    if policy == "random":
        rng = np.random.default_rng(seeds.random_policy)
        ids = sorted(pool_ids)
        rng_scores = dict(zip(ids, rng.random(len(ids)).tolist()))

    score_name = {"random": "random_uniform", "oracle_incorrect": "oracle_norm_edit_distance"}.get(
        policy, SCORE_NAMES.get(policy, ""))
    acq_frames, cell_frames, summ_frames, rounds = [], [], [], []
    selectors: Dict[int, TrainedSelector] = {}

    def fit(round_: int) -> Tuple[Optional[TrainedSelector], Dict[str, Any]]:
        rows = oracle.reveal(selected)
        exs, st = build_examples(rows, task)
        aux = aux_copy_examples(scfg.aux_copy, selected, anchors)
        st["n_aux_copy_examples"] = len(aux)
        say(f"[{policy}] round {round_}: training on {len(selected)} lemmas / {len(exs)} examples"
            + (f" + {len(aux)} aux copy items" if aux else ""))
        sel = train_selector(exs + aux, dev_examples, scfg, seeds.selector_init)
        if keep_selectors:
            selectors[round_] = sel
        info = {k: v for k, v in sel.info.items()}
        return sel, {**st, **info}

    for r, size in enumerate(plan["batch_sizes"], start=1):
        rec: Dict[str, Any] = {"round": r, "n_train_lemmas": len(selected), "batch_size_planned": batch_size,
                               "n_to_select": size, "is_remainder_round": size != batch_size}
        candidates = pool.queries(exclude=selected)
        rec["n_candidates"] = len(candidates)
        need_model = policy != "random" or train_random_rounds
        selector = None
        if need_model:
            selector, info = fit(r)
            rec.update({"n_train_examples": info["n_examples"], "training_mode": info["mode"],
                        "n_aux_copy_examples": info["n_aux_copy_examples"],
                        "n_target_cells_distinct": info["n_target_cells_distinct"],
                        "dev_acc": info["dev_acc"], "dev_loss": info["dev_loss"], "best_step": info["best_step"],
                        "steps_run": info["steps_run"], "stopped_early": info["stopped_early"],
                        "train_runtime_s": info["train_runtime_s"], "model_hash": info["model_hash"],
                        "device": info["device"], "n_params": info["n_params"], "dev_curve": info["dev_curve"]})
        else:
            _, st = build_examples(oracle.reveal(selected), task)
            rec.update({"n_train_examples": st["n_examples"], "training_mode": st["mode"], "dev_acc": None,
                        "train_runtime_s": 0.0, "model_hash": None})
        t0 = time.perf_counter()
        if policy == "random":
            lem = pd.DataFrame({"lemma_id": [q.lemma_id for q in candidates],
                                "lemma_score": [rng_scores[q.lemma_id] for q in candidates],
                                "n_cells_scored": 0, "n_nonfinite": 0})
        else:
            cells = score_candidates(selector, candidates, min_prob=min_prob)
            crow = [dict(round=r, **row) for c in cells for row in c.rows]
            cell_frames.append(pd.DataFrame(crow, columns=CELL_SCORE_COLUMNS))
            cs = cell_summary_frame(cells)
            cs.insert(0, "round", r)
            summ_frames.append(cs)
            if policy in ORACLE_POLICIES:
                gold = oracle.peek_for_oracle_policy([q.lemma_id for q in candidates], r)
                sc = oracle_incorrect_scores(cells, gold)
                agg = aggregate_lemmas(cells, "surprisal_norm", task.panel_cells)
                agg["lemma_score"] = agg["lemma_id"].map(sc).astype(float).fillna(float("inf"))
                lem = agg
            else:
                lem = aggregate_lemmas(cells, SCORE_NAMES[policy], task.panel_cells)
            rec["n_lemmas_all_nonfinite"] = int(lem.get("all_nonfinite", pd.Series(dtype=bool)).sum())
            rec["n_cells_nonfinite"] = int(lem["n_nonfinite"].sum())
        scores = dict(zip(lem["lemma_id"], lem["lemma_score"].astype(float)))
        ranked = rank_lemmas(scores, seeds.tie)
        chosen = ranked[:size]
        rank_in_round = {l: i + 1 for i, l in enumerate(ranked)}
        lem = lem.assign(round=r, rank_in_round=lem["lemma_id"].map(rank_in_round),
                         selected=lem["lemma_id"].isin(set(chosen)))
        acq_frames.append(lem.sort_values("rank_in_round")[ACQ_LOG_COLUMNS])
        rec["score_runtime_s"] = round(time.perf_counter() - t0, 3)
        first = len(selected) + 1
        oracle.select_and_reveal(chosen, r, first)
        for i, l in enumerate(chosen):
            order_rows.append({"lemma_id": l, "acquisition_rank": first + i, "round": r,
                               "lemma_score": scores[l], "score_name": score_name})
        selected.extend(chosen)
        rec["n_selected_total"] = len(selected)
        rounds.append(rec)
        say(f"[{policy}] round {r}: selected {len(chosen)} (total {len(selected)}), "
            f"train {rec['train_runtime_s']}s, score {rec['score_runtime_s']}s, dev_acc={rec.get('dev_acc')}")

    if final_fit:
        _, info = fit(len(rounds) + 1)
        rounds.append({"round": len(rounds) + 1, "final_fit": True, "n_train_lemmas": len(selected),
                       "n_train_examples": info["n_examples"], "dev_acc": info["dev_acc"],
                       "dev_loss": info["dev_loss"], "best_step": info["best_step"],
                       "train_runtime_s": info["train_runtime_s"], "model_hash": info["model_hash"],
                       "dev_curve": info["dev_curve"]})

    # ------------------------------------------------------------------ outputs
    order = pd.DataFrame(order_rows, columns=ORDER_COLUMNS)
    order.to_csv(out_dir / "order.csv", index=False)
    acq = pd.concat(acq_frames, ignore_index=True) if acq_frames else pd.DataFrame(columns=ACQ_LOG_COLUMNS)
    acq.to_csv(out_dir / "acquisition_log.csv", index=False)
    cells_df = (pd.concat(cell_frames, ignore_index=True) if cell_frames
                else pd.DataFrame(columns=CELL_SCORE_COLUMNS))
    cells_df.to_csv(out_dir / "cell_scores.csv", index=False)
    if summ_frames:
        pd.concat(summ_frames, ignore_index=True).to_csv(out_dir / "cell_summary.csv", index=False)
    reveals = pd.concat([pd.DataFrame(dev_log, columns=REVEAL_COLUMNS), oracle.reveal_log()], ignore_index=True)
    reveals.to_csv(out_dir / "oracle_reveals.csv", index=False)

    sample_paths: Dict[int, Dict[str, Path]] = {}
    budget_summary = {}
    rank_of = dict(zip(order["lemma_id"], order["acquisition_rank"]))
    round_of = dict(zip(order["lemma_id"], order["round"]))
    keep_cells = {task.source_cell, *task.panel_cells}
    for b in plan["budgets"]:
        b_eff = min(b, len(order))
        ids = order["lemma_id"].iloc[:b_eff].tolist()
        rows = oracle.reveal(ids)
        rows = rows.assign(_rank=rows["lemma_id"].map(rank_of)).sort_values(
            ["_rank", "cell_norm", "variant_idx"], kind="stable")
        panel_rows = rows[rows["cell_norm"].isin(keep_cells)]
        all_rows = rows[(~rows["is_missing"].astype(bool)) & (rows["cell_norm"].fillna("") != "")]
        cols = [c for c in FORMS_COLUMNS if c in rows.columns]
        p1, p2, p3 = (out_dir / "samples" / f"budget_{b}.csv", out_dir / "samples" / f"budget_{b}_allforms.csv",
                      out_dir / "samples" / f"budget_{b}_lemmas.csv")
        panel_rows[cols].to_csv(p1, index=False)
        all_rows[cols].to_csv(p2, index=False)
        lem_tab = pd.DataFrame({"lemma_id": ids, "acquisition_rank": [rank_of[i] for i in ids],
                                "round": [round_of[i] for i in ids], "weight": 1.0})
        lem_tab.to_csv(p3, index=False)
        sample_paths[b] = {"panel": p1, "allforms": p2, "lemmas": p3}
        v0 = _v0(panel_rows)
        last_round = int(lem_tab["round"].max()) if len(lem_tab) else 0
        budget_summary[str(b)] = {
            "n_lemmas": len(ids), "shortfall": b - b_eff,
            "n_rows_panel": int(len(panel_rows)), "n_forms_panel_v0": int(len(v0)),
            "n_training_examples_panel": int((v0["cell_norm"] != task.source_cell).sum()),
            "n_rows_allforms": int(len(all_rows)),
            "n_training_examples_all_cells": int((_v0(all_rows)["cell_norm"] != task.source_cell).sum()),
            "weights": "all 1.0 (frequency-free)", "rounds_used": last_round,
            "partial_last_round": bool(len(lem_tab) and
                                       (lem_tab["round"] == last_round).sum() <
                                       (order["round"] == last_round).sum()),
        }
    summary = {
        "policy": policy, "is_oracle_policy": policy in ORACLE_POLICIES, "unit_id": task.unit_id,
        "score_name": score_name, "plan": plan, "budgets": budget_summary,
        "n_seed": len(seed_ids), "n_pool": len(pool_ids), "n_dev": len(dev_ids), "dev_examples": dev_stats,
        "training_mode": task.training_mode, "seeds": vars(seeds),
        "selector_config": scfg.to_dict(), "train_random_rounds": bool(train_random_rounds),
        "runtime_s": round(time.perf_counter() - t_start, 3),
        "train_runtime_total_s": round(sum(r.get("train_runtime_s") or 0 for r in rounds), 3),
        "score_runtime_total_s": round(sum(r.get("score_runtime_s") or 0 for r in rounds), 3),
    }
    with open(out_dir / "rounds.json", "w", encoding="utf-8") as fh:
        json.dump({"policy": policy, "plan": plan, "rounds": rounds}, fh, indent=1, default=_json_default)
    with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)
    _write_manifest(out_dir, cfg, forms, seeds, started, policy)
    return AcquisitionResult(policy, out_dir, order, rounds, sample_paths, summary, selectors)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def _write_manifest(out_dir: Path, cfg: Mapping[str, Any], forms: pd.DataFrame, seeds: SelectionSeeds,
                    started: str, policy: str) -> None:
    import torch
    man = {
        "stage": "selection", "policy": policy, "config_hash": cfg.get("_config_hash"),
        "inputs": {"forms_rows_for_fold": _frame_hash(forms)},
        "external_revisions": {"ALmorphinfl": UPSTREAM_REVISION + " (reference only; logic rewritten)"},
        "package_versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                             "python": platform.python_version()},
        "seeds": vars(seeds), "start_time": started, "end_time": datetime.now(timezone.utc).isoformat(),
        **_git_state(),
    }
    with open(out_dir / "stage_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)


# ----------------------------------------------------------------------------- selector evaluation hook

def train_selector_on_sample(sample_rows: pd.DataFrame, dev_rows: pd.DataFrame, task: SelectionTask,
                             cfg: Mapping[str, Any], seed: int,
                             anchor_rows: Optional[pd.DataFrame] = None) -> TrainedSelector:
    """Fit the selector on a final sample (e.g. ``samples/budget_{B}.csv``) with the same
    protocol as the acquisition rounds (dev gold for early stopping only).

    With ``selector.aux_copy == 'pool'`` pass ``anchor_rows`` = forms rows (only the
    source cell is read) of the fold's seed + pool lemmas, to reproduce the round setup.
    """
    scfg = SelectorConfig.from_cfg(cfg)
    exs, _ = build_examples(sample_rows, task)
    dev, _ = build_examples(dev_rows, task, mode="panel", with_gold_variants=True)
    ids = sorted(set(sample_rows["lemma_id"]))
    if scfg.aux_copy == "pool" and anchor_rows is None:
        raise ValueError("aux_copy='pool' needs anchor_rows (seed + pool source rows)")
    src = anchor_rows if anchor_rows is not None else sample_rows
    view = CandidatePoolView.from_forms(src, sorted(set(src["lemma_id"])), task)
    aux = aux_copy_examples(scfg.aux_copy, ids, view)
    return train_selector(exs + aux, dev, scfg, seed)


PREDICTION_COLUMNS_SELECTOR = ["lemma_id", "source_cell", "target_cell", "prediction", "prediction_segments",
                               "logprob_sum", "hyp_len", "surprisal_norm", "n_hyps", "status"]


def predict_queries(model_or_sample, queries, *, task: Optional[SelectionTask] = None,
                    dev_rows: Optional[pd.DataFrame] = None, cfg: Optional[Mapping[str, Any]] = None,
                    seed: Optional[int] = None, beam_size: Optional[int] = None,
                    anchor_rows: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Predict held-out queries with the selector (top beam hypothesis).

    ``model_or_sample``: a ``TrainedSelector``, or a sample forms table (then ``task``,
    ``dev_rows``, ``cfg`` and ``seed`` are required and a selector is trained first).
    ``queries``: a DataFrame with ``TEST_QUERY_COLUMNS`` (only those columns are read) or
    a sequence of ``CandidateQuery``. Returns ``PREDICTION_COLUMNS_SELECTOR``;
    ``prediction`` = segments concatenated (``_`` -> space), status ok | no_hypothesis.
    """
    if isinstance(model_or_sample, TrainedSelector):
        sel = model_or_sample
    else:
        if task is None or dev_rows is None or cfg is None or seed is None:
            raise ValueError("training from a sample needs task, dev_rows, cfg and seed")
        sel = train_selector_on_sample(model_or_sample, dev_rows, task, cfg, seed, anchor_rows=anchor_rows)
    if isinstance(queries, pd.DataFrame):
        q = queries[TEST_QUERY_COLUMNS]
        items = [(r.lemma_id, r.source_cell, split_segments(str(r.source_segments)), r.target_cell)
                 for r in q.itertuples(index=False)]
    else:
        items = [(c.lemma_id, c.source_cell, c.source_segments, t) for c in queries for t in c.target_cells]
    beams = sel.beam_search([encode_input(sc, ss, tc) for _, sc, ss, tc in items],
                            [len(ss) for _, _, ss, _ in items], beam_size=beam_size)
    rows = []
    for (lid, sc, ss, tc), hyps in zip(items, beams):
        fin = [h for h in hyps if math.isfinite(h.logprob_sum)]
        if fin:
            h = fin[0]
            rows.append({"lemma_id": lid, "source_cell": sc, "target_cell": tc,
                         "prediction": "".join(h.segments).replace("_", " "), "prediction_segments": h.hyp,
                         "logprob_sum": h.logprob_sum, "hyp_len": h.hyp_len,
                         "surprisal_norm": -h.logprob_sum / (h.hyp_len + 1), "n_hyps": len(hyps), "status": "ok"})
        else:
            rows.append({"lemma_id": lid, "source_cell": sc, "target_cell": tc, "prediction": "",
                         "prediction_segments": "", "logprob_sum": float("nan"), "hyp_len": 0,
                         "surprisal_norm": float("nan"), "n_hyps": len(hyps), "status": "no_hypothesis"})
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS_SELECTOR)
