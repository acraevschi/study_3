"""PCFP acquisition with the LDL selector (docs/SELECTION.md, PCFP section; CONTRACT §5).

Information flow
----------------
* ``ShownOracle`` is built from the *shown* cells only (exposure manifest) of the core,
  seed and pool verbs. Hidden-cell forms never enter this module. A verb's shown rows are
  returned only once it is core (round 0), seed (round 0) or selected; every reveal is
  logged (``oracle_reveals.csv``).
* ``CitationPoolView`` exposes, per pool candidate, only (lemma_id, citation cell,
  segments of the citation *label*, names of its pre-drawn shown cells). It never holds a
  form from the forms table.
* Each round the LDL selector is fitted on the revealed shown forms (core + seed +
  acquired so far) and scores every remaining candidate from its citation row, which is
  added and removed per candidate (Julia ``score_candidate``). Candidate forms never enter
  a fit, a cue inventory or a candidate list before the candidate is acquired.
* The exposure draw (k and shown cells) is fixed per verb before selection; the policy
  only chooses verbs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from morph_ldl import seeds as seedlib
from morph_ldl.cv import pcfp
from morph_ldl.schemas import ACQ_LOG_COLUMNS, FORMS_COLUMNS, ORDER_COLUMNS
from morph_ldl.selection.acquisition import GoldAccessError, SelectionSeeds, _json_default, round_plan
from morph_ldl.selection.scoring import rank_lemmas

POLICIES_LDL = ("random", "low_confidence", "high_entropy")
REVEAL_COLUMNS = ["round", "lemma_id", "reason", "acquisition_rank", "n_rows"]
CANDIDATE_COLUMNS = ["lemma_id", "citation_cell", "citation_segments", "shown_cells"]


# ----------------------------------------------------------------------------- pool / oracle

@dataclass(frozen=True)
class CitationCandidate:
    lemma_id: str
    citation_cell: str
    citation_segments: str
    shown_cells: tuple


class CitationPoolView:
    """Pool candidates as the selector may see them: citation label + shown-cell names."""

    def __init__(self, cands: Sequence[CitationCandidate]):
        self._c = {c.lemma_id: c for c in cands}

    @classmethod
    def build(cls, pool_ids: Iterable[str], labels: Mapping[str, str], exposure: Mapping[str, pcfp.Exposure],
              citation_cell: str, representation: str) -> "CitationPoolView":
        out = []
        for lid in pool_ids:
            out.append(CitationCandidate(lid, citation_cell, pcfp.citation_segments(labels[lid], representation),
                                         tuple(exposure[lid].shown)))
        return cls(out)

    @property
    def lemma_ids(self) -> List[str]:
        return sorted(self._c)

    def frame(self, exclude: Iterable[str] = ()) -> pd.DataFrame:
        ex = set(exclude)
        rows = [{"lemma_id": c.lemma_id, "citation_cell": c.citation_cell,
                 "citation_segments": c.citation_segments, "shown_cells": pcfp.SEP.join(c.shown_cells)}
                for k, c in sorted(self._c.items()) if k not in ex]
        return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


class ShownOracle:
    """Shown-cell rows of core, seed and pool verbs; reveals a verb only once it is in
    the training sample. Hidden cells are dropped at construction."""

    def __init__(self, forms: pd.DataFrame, exposure: Mapping[str, pcfp.Exposure], lemma_ids: Iterable[str]):
        ids = list(dict.fromkeys(lemma_ids))
        self._rows = pcfp.shown_rows(forms, exposure, ids).copy()
        self._ids = set(ids)
        self._revealed: Dict[str, tuple] = {}
        self.log: List[dict] = []

    def select_and_reveal(self, lemma_ids: Sequence[str], round_: int, first_rank: int, reason: str) -> pd.DataFrame:
        for j, lid in enumerate(lemma_ids):
            if lid not in self._ids:
                raise GoldAccessError(f"{lid!r} is not a core/seed/pool verb of this oracle")
            if lid in self._revealed:
                raise ValueError(f"{lid!r} revealed twice")
            self._revealed[lid] = (round_, first_rank + j if first_rank >= 0 else -1)
        rows = self._rows[self._rows["lemma_id"].isin(set(lemma_ids))]
        counts = rows.groupby("lemma_id").size().to_dict()
        for lid in lemma_ids:
            self.log.append({"round": round_, "lemma_id": lid, "reason": reason,
                             "acquisition_rank": self._revealed[lid][1], "n_rows": int(counts.get(lid, 0))})
        return rows.copy()

    def reveal(self, lemma_ids: Sequence[str]) -> pd.DataFrame:
        bad = [l for l in lemma_ids if l not in self._revealed]
        if bad:
            raise GoldAccessError(f"shown forms requested for {len(bad)} unacquired verb(s), e.g. {bad[:3]}")
        return self._rows[self._rows["lemma_id"].isin(set(lemma_ids))].copy()

    def reveal_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=REVEAL_COLUMNS)


# ----------------------------------------------------------------------------- loop

@dataclass
class LDLAcquisitionResult:
    policy: str
    out_dir: Path
    order: pd.DataFrame
    rounds: List[dict]
    sample_paths: Dict[int, Dict[str, Path]]
    summary: Dict[str, Any]


def _ordered_rows(rows: pd.DataFrame, rank_of: Mapping[str, int]) -> pd.DataFrame:
    rows = rows.assign(_rank=rows["lemma_id"].map(rank_of))
    rows = rows.sort_values(["_rank", "lemma_id", "cell_norm", "variant_idx"], kind="stable")
    return rows[[c for c in FORMS_COLUMNS if c in rows.columns]]


def run_ldl_acquisition(policy: str, core_ids: Sequence[str], seed_ids: Sequence[str], pool_ids: Sequence[str],
                        forms_df: pd.DataFrame, exposure: Mapping[str, pcfp.Exposure], labels: Mapping[str, str],
                        citation_cell: str, representation: str, cfg: Mapping[str, Any], seeds: SelectionSeeds,
                        out_dir: Path, *, selector_configs: Sequence[Mapping] = (),
                        server_factory: Optional[Callable[[], Any]] = None,
                        excluded_ids: Sequence[str] = (), log: Optional[Callable[[str], None]] = None
                        ) -> LDLAcquisitionResult:
    """Run one policy on one (unit, repetition, outer fold, pool cap) and write the
    selection outputs to ``out_dir``.

    ``server_factory`` returns a context manager with ``score(train, cands, out_cells,
    out_comp, configs)`` (``morph_ldl.ldl.selector.SelectorServer``); it is required for
    the active policies. ``excluded_ids`` (other roles of the fold) are only checked for
    disjointness."""
    t_start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    say = log or (lambda s: None)
    if policy not in POLICIES_LDL:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES_LDL}")
    core_ids, seed_ids, pool_ids = list(core_ids), list(seed_ids), list(pool_ids)
    sets = {"core": set(core_ids), "seed": set(seed_ids), "pool": set(pool_ids), "excluded": set(excluded_ids)}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise ValueError(f"{a} and {b} verb sets overlap ({len(sets[a] & sets[b])} verbs)")
    for name in ("core", "seed", "pool"):
        if len(sets[name]) != len({"core": core_ids, "seed": seed_ids, "pool": pool_ids}[name]):
            raise ValueError(f"duplicate verb ids in {name}")
    sel = cfg["selection"]
    batch_size = int(sel["batch_size"])
    active = policy != "random"
    if active and (server_factory is None or not selector_configs):
        raise ValueError("active policies need a selector server and selector configs")
    temperature = float(sel["entropy_temperature"])
    no_cand = float(sel["no_candidate_support"])
    max_can = int(cfg["ldl"]["max_can"])
    plan = round_plan(len(seed_ids), sel["budgets"], batch_size, len(pool_ids))
    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    forms = forms_df[forms_df["lemma_id"].isin(sets["core"] | sets["seed"] | sets["pool"])]
    oracle = ShownOracle(forms, exposure, core_ids + seed_ids + pool_ids)
    pool = CitationPoolView.build(pool_ids, labels, exposure, citation_cell, representation)
    core_sorted = sorted(core_ids)
    oracle.select_and_reveal(core_sorted, 0, -1, reason="core")
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
    score_name = {"random": "random_uniform", "low_confidence": "mean_one_minus_top_support",
                  "high_entropy": "mean_support_softmax_entropy"}[policy]

    acq_frames, cell_frames, comp_frames, rounds = [], [], [], []
    ctx = server_factory() if active else None
    server = ctx.__enter__() if ctx is not None else None
    try:
        for r, size in enumerate(plan["batch_sizes"], start=1):
            rec: Dict[str, Any] = {"round": r, "n_selected_verbs": len(selected), "n_core_verbs": len(core_ids),
                                   "n_to_select": size, "is_remainder_round": size != batch_size}
            rdir = out_dir / "rounds" / f"r{r}"
            train_ids = core_sorted + selected
            train = oracle.reveal(train_ids)
            train_v0 = train[(train["variant_idx"] == 0) & (~train["is_missing"].astype(bool))]
            rec.update({"n_train_verbs": len(train_ids), "n_train_forms": int(len(train_v0))})
            cands = pool.frame(exclude=selected)
            rec["n_candidates"] = len(cands)
            t0 = time.perf_counter()
            if active:
                rdir.mkdir(parents=True, exist_ok=True)
                _ordered_rows(train, {l: i for i, l in enumerate(train_ids)}).to_csv(rdir / "train.csv", index=False)
                cands.to_csv(rdir / "candidates.csv", index=False)
                reply = server.score(rdir / "train.csv", rdir / "candidates.csv", rdir / "cells.csv",
                                     rdir / "comprehension.csv", selector_configs)
                cells = pd.read_csv(rdir / "cells.csv", keep_default_na=False,
                                    dtype={"supports": str, "top_prediction_segments": str})
                from morph_ldl.ldl.selector import add_cell_scores, lemma_scores
                cells = add_cell_scores(cells, temperature, no_cand, max_can)
                lem = lemma_scores(cells, policy)
                if set(lem["lemma_id"]) != set(cands["lemma_id"]):
                    raise RuntimeError(f"round {r}: selector did not score every candidate")
                comp = pd.read_csv(rdir / "comprehension.csv")
                cell_frames.append(cells.assign(round=r))
                comp_frames.append(comp.assign(round=r))
                rec.update({"fit_seconds": reply["fit_seconds"], "score_seconds": reply["score_seconds"],
                            "n_cues": reply["n_cues"], "n_train_rows_selector": reply["n_train_rows"],
                            "n_cells_no_candidate": int((cells["status"].astype(str) == "no_candidate").sum()),
                            "n_cells_error": int((cells["status"].astype(str) == "error").sum()),
                            "top_equals_citation_rate": float(cells["top_equals_citation"].astype(str)
                                                              .str.lower().eq("true").mean()),
                            "lemma_score_sd": float(lem["lemma_score"].std()),
                            "mean_seed_sd": float(lem["score_seed_sd"].mean()) if len(lem) else float("nan")})
            else:
                lem = pd.DataFrame({"lemma_id": cands["lemma_id"],
                                    "lemma_score": [rng_scores[l] for l in cands["lemma_id"]],
                                    "n_cells_scored": 0, "n_nonfinite": 0})
            scores = dict(zip(lem["lemma_id"], lem["lemma_score"].astype(float)))
            ranked = rank_lemmas(scores, seeds.tie)
            chosen = ranked[:size]
            rank_in_round = {l: i + 1 for i, l in enumerate(ranked)}
            lem = lem.assign(round=r, rank_in_round=lem["lemma_id"].map(rank_in_round),
                             selected=lem["lemma_id"].isin(set(chosen)))
            acq_frames.append(lem.sort_values("rank_in_round"))
            rec["score_runtime_s"] = round(time.perf_counter() - t0, 3)
            first = len(selected) + 1
            oracle.select_and_reveal(chosen, r, first, reason="selected")
            for i, l in enumerate(chosen):
                order_rows.append({"lemma_id": l, "acquisition_rank": first + i, "round": r,
                                   "lemma_score": scores[l], "score_name": score_name})
            selected.extend(chosen)
            rec["n_selected_total"] = len(selected)
            rounds.append(rec)
            say(f"[{policy}] round {r}: selected {len(chosen)} (total {len(selected)}), "
                f"scoring {rec['score_runtime_s']}s")
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------ outputs
    order = pd.DataFrame(order_rows, columns=ORDER_COLUMNS)
    order.to_csv(out_dir / "order.csv", index=False)
    acq = pd.concat(acq_frames, ignore_index=True) if acq_frames else pd.DataFrame(columns=ACQ_LOG_COLUMNS)
    extra = [c for c in acq.columns if c not in ACQ_LOG_COLUMNS]
    acq[ACQ_LOG_COLUMNS + extra].to_csv(out_dir / "acquisition_log.csv", index=False)
    if cell_frames:
        pd.concat(cell_frames, ignore_index=True).to_csv(out_dir / "cell_scores.csv", index=False)
    if comp_frames:
        comp = pd.concat(comp_frames, ignore_index=True)
        comp = comp.merge(acq[["round", "lemma_id", "lemma_score"]], on=["round", "lemma_id"], how="left")
        comp.to_csv(out_dir / "comprehension_check.csv", index=False)
    oracle.reveal_log().to_csv(out_dir / "oracle_reveals.csv", index=False)

    sample_paths: Dict[int, Dict[str, Path]] = {}
    budget_summary = {}
    rank_of = dict(zip(order["lemma_id"], order["acquisition_rank"]))
    round_of = dict(zip(order["lemma_id"], order["round"]))
    for b in plan["budgets"]:
        b_eff = min(b, len(order))
        ids = order["lemma_id"].iloc[:b_eff].tolist()
        rows = oracle.reveal(core_sorted + ids)
        rk = {**{l: -len(core_sorted) + i for i, l in enumerate(core_sorted)}, **rank_of}
        rows = _ordered_rows(rows, rk)
        p1 = out_dir / "samples" / f"budget_{b}.csv"
        p2 = out_dir / "samples" / f"budget_{b}_lemmas.csv"
        rows.to_csv(p1, index=False)
        lem_tab = pd.DataFrame(
            [{"lemma_id": l, "role": "core", "acquisition_rank": -1, "round": -1, "k": exposure[l].k,
              "weight": 1.0} for l in core_sorted]
            + [{"lemma_id": l, "role": "seed" if round_of[l] == 0 else "acquired", "acquisition_rank": rank_of[l],
                "round": round_of[l], "k": exposure[l].k, "weight": 1.0} for l in ids])
        lem_tab.to_csv(p2, index=False)
        sample_paths[b] = {"sample": p1, "lemmas": p2}
        v0 = rows[(rows["variant_idx"] == 0) & (~rows["is_missing"].astype(bool))]
        n_core_forms = int(v0["lemma_id"].isin(sets["core"]).sum())
        budget_summary[str(b)] = {
            "n_selected_verbs": len(ids), "n_core_verbs": len(core_sorted), "shortfall": b - b_eff,
            "n_forms_total": int(len(v0)), "n_forms_core": n_core_forms,
            "n_forms_selected": int(len(v0)) - n_core_forms, "n_rows": int(len(rows)),
            "k_mean_selected": float(np.mean([exposure[l].k for l in ids])) if ids else float("nan"),
            "weights": "all 1.0 (frequency-free)",
            "rounds_used": int(max([round_of[l] for l in ids], default=0)),
        }
    summary = {
        "policy": policy, "selector": "ldl" if active else "none (random)", "score_name": score_name,
        "plan": plan, "budgets": budget_summary, "n_core": len(core_ids), "n_seed": len(seed_ids),
        "n_pool": len(pool_ids), "seeds": vars(seeds),
        "n_selector_semantic_seeds": len(selector_configs) if active else 0,
        "selector_semantic_seeds": [c["semantic_seed"] for c in selector_configs] if active else [],
        "entropy_temperature": temperature, "no_candidate_support": no_cand,
        "core_forms_in_every_training_sample": True,
        "ldl_settings": sel.get("_ldl_settings") if active else None,
        "runtime_s": round(time.perf_counter() - t_start, 3), "started_utc": started,
    }
    with open(out_dir / "rounds.json", "w", encoding="utf-8") as fh:
        json.dump({"policy": policy, "plan": plan, "rounds": rounds}, fh, indent=1, default=_json_default)
    with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)
    return LDLAcquisitionResult(policy, out_dir, order, rounds, sample_paths, summary)
