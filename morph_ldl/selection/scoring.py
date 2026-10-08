"""Uncertainty scores from selector beams, lemma aggregation and deterministic ranking.

Definitions (natural logarithms throughout; docs/SELECTION.md §3):

* ``logprob_sum``  log P(hyp, EOS | input) under the selector, summed over output steps
  including EOS; ``hyp_len`` = number of output symbols excluding EOS.
* ``surprisal_norm = -logprob_sum / (hyp_len + 1)``: per-step surprisal of a hypothesis,
  normalised by its *predicted* length (+1 for EOS). Gold length is never used.
  Cell confidence score = ``surprisal_norm`` of the top hypothesis (highest
  ``logprob_sum``). Higher = less confident = selected first.
* ``prob_renorm_i = exp(lp_i - logsumexp(lp))`` over the finite beam hypotheses of a cell.
* ``entropy = -sum_{i: p_i >= min_prob} p_i log p_i`` (paper: p_i >= 0.05). The
  probabilities are *not* renormalised again after the threshold is applied, so mass
  below the threshold simply does not contribute. Higher = selected first.
* Lemma score = mean over the panel target cells with a finite cell score. A cell with no
  finite hypothesis is counted in ``n_nonfinite``; a lemma with no finite cell gets
  ``+inf`` (most uncertain; flagged ``all_nonfinite``).
* Ranking: lemma score descending, ties by ``seeds.tie_key(tie_seed, lemma_id)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from morph_ldl.seeds import tie_key

SCORE_NAMES = {"low_confidence": "surprisal_norm", "high_entropy": "entropy"}


def surprisal_norm(logprob_sum: float, hyp_len: int) -> float:
    return -float(logprob_sum) / (int(hyp_len) + 1)


def renormalise(logprobs: Sequence[float]) -> np.ndarray:
    lp = np.asarray(logprobs, dtype=np.float64)
    if lp.size == 0:
        return lp
    m = lp.max()
    lse = m + math.log(float(np.exp(lp - m).sum()))
    return np.exp(lp - lse)


def thresholded_entropy(probs: Sequence[float], min_prob: float = 0.05) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p >= min_prob]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


@dataclass
class CellScore:
    lemma_id: str
    target_cell: str
    surprisal_norm: float  # of the top finite hypothesis; nan if none
    entropy: float  # nan if no finite hypothesis
    finite: bool
    n_hyps: int
    n_hyps_finite: int
    n_hyps_entropy: int  # hypotheses with p >= min_prob
    top_hyp: str
    rows: List[dict] = field(default_factory=list)  # one per hypothesis (cell_scores.csv)

    def value(self, score_name: str) -> float:
        return getattr(self, score_name)


def score_cell(lemma_id: str, target_cell: str, hyps: Sequence, min_prob: float = 0.05) -> CellScore:
    """``hyps``: objects with ``segments``/``hyp``, ``logprob_sum``, ``hyp_len`` (any order).

    Non-finite hypotheses (nan / +-inf ``logprob_sum``) are dropped before ranking and
    renormalisation; they are still written to ``rows`` with ``prob_renorm`` = nan.
    """
    def _hyp_str(h) -> str:
        return h.hyp if hasattr(h, "hyp") else " ".join(h.segments)

    fin = [h for h in hyps if math.isfinite(float(h.logprob_sum))]
    nonfin = [h for h in hyps if not math.isfinite(float(h.logprob_sum))]
    fin = sorted(fin, key=lambda h: (-float(h.logprob_sum), _hyp_str(h)))
    probs = renormalise([h.logprob_sum for h in fin])
    rows = []
    for r, (h, p) in enumerate(zip(fin, probs)):
        rows.append({"lemma_id": lemma_id, "target_cell": target_cell, "hyp_rank": r, "hyp": _hyp_str(h),
                     "logprob_sum": float(h.logprob_sum), "hyp_len": int(h.hyp_len),
                     "surprisal_norm": surprisal_norm(h.logprob_sum, h.hyp_len), "prob_renorm": float(p)})
    for j, h in enumerate(nonfin):
        rows.append({"lemma_id": lemma_id, "target_cell": target_cell, "hyp_rank": len(fin) + j,
                     "hyp": _hyp_str(h), "logprob_sum": float(h.logprob_sum), "hyp_len": int(h.hyp_len),
                     "surprisal_norm": float("nan"), "prob_renorm": float("nan")})
    if not fin:
        return CellScore(lemma_id, target_cell, float("nan"), float("nan"), False, len(hyps), 0, 0, "", rows)
    return CellScore(
        lemma_id, target_cell,
        surprisal_norm=surprisal_norm(fin[0].logprob_sum, fin[0].hyp_len),
        entropy=thresholded_entropy(probs, min_prob),
        finite=True, n_hyps=len(hyps), n_hyps_finite=len(fin),
        n_hyps_entropy=int((probs >= min_prob).sum()), top_hyp=_hyp_str(fin[0]), rows=rows,
    )


def aggregate_lemmas(cells: Iterable[CellScore], score_name: str,
                     expected_cells: Sequence[str] | None = None) -> pd.DataFrame:
    """Mean of the finite cell scores per lemma (``aggregation: mean_cell``).

    Returns lemma_id, lemma_score, n_cells_scored, n_nonfinite, n_cells_expected,
    all_nonfinite. Expected cells that were not scored at all count as non-finite.
    """
    by: Dict[str, List[CellScore]] = {}
    for c in cells:
        by.setdefault(c.lemma_id, []).append(c)
    out = []
    for lid in sorted(by):
        cs = by[lid]
        seen = {c.target_cell for c in cs}
        n_missing = len(set(expected_cells) - seen) if expected_cells is not None else 0
        vals = [c.value(score_name) for c in cs if c.finite and math.isfinite(c.value(score_name))]
        n_nonfinite = len(cs) - len(vals) + n_missing
        score = float(np.mean(vals)) if vals else float("inf")
        out.append({"lemma_id": lid, "lemma_score": score, "n_cells_scored": len(vals),
                    "n_nonfinite": n_nonfinite,
                    "n_cells_expected": len(expected_cells) if expected_cells is not None else len(cs),
                    "all_nonfinite": not vals})
    return pd.DataFrame(out, columns=["lemma_id", "lemma_score", "n_cells_scored", "n_nonfinite",
                                      "n_cells_expected", "all_nonfinite"])


def rank_lemmas(scores: Mapping[str, float], tie_seed: int) -> List[str]:
    """Lemma ids by score descending (+inf first), ties by sha256 tie key."""
    for lid, s in scores.items():
        if s is None or (isinstance(s, float) and math.isnan(s)):
            raise ValueError(f"lemma {lid!r} has a NaN score; cannot rank")
    return sorted(scores, key=lambda lid: (-float(scores[lid]), tie_key(tie_seed, lid)))


def cell_summary_frame(cells: Iterable[CellScore]) -> pd.DataFrame:
    cols = ["lemma_id", "target_cell", "top_hyp", "surprisal_norm", "entropy", "finite", "n_hyps",
            "n_hyps_finite", "n_hyps_entropy"]
    return pd.DataFrame([{k: getattr(c, k) for k in cols} for c in cells], columns=cols)
