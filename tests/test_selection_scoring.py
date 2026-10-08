"""Score definitions, directions, aggregation, non-finite handling and tie-breaking."""

import math

import numpy as np
import pytest

from morph_ldl.seeds import tie_key
from morph_ldl.selection.model import Hypothesis
from morph_ldl.selection.scoring import (aggregate_lemmas, rank_lemmas, renormalise, score_cell,
                                         surprisal_norm, thresholded_entropy)


def H(segs, lp):
    segs = tuple(segs.split()) if segs else ()
    return Hypothesis(segs, lp, len(segs))


def test_surprisal_uses_predicted_length_plus_eos():
    # top hypothesis has 3 symbols -> divide by 4, regardless of any gold length
    c = score_cell("l", "c", [H("a b c", -2.0), H("a b c d e f g", -3.0)])
    assert c.top_hyp == "a b c"
    assert c.surprisal_norm == pytest.approx(2.0 / 4)
    assert surprisal_norm(-6.0, 5) == pytest.approx(1.0)


def test_top_hypothesis_is_highest_logprob_sum_not_input_order():
    c = score_cell("l", "c", [H("x", -5.0), H("y y", -1.0), H("z", -3.0)])
    assert c.top_hyp == "y y"
    assert [r["hyp_rank"] for r in c.rows] == [0, 1, 2]
    assert [r["hyp"] for r in c.rows] == ["y y", "z", "x"]


def test_entropy_hand_calculation_with_threshold():
    lps = [math.log(0.6), math.log(0.3), math.log(0.07), math.log(0.02), math.log(0.01)]
    c = score_cell("l", "c", [H(f"s{i}", lp) for i, lp in enumerate(lps)], min_prob=0.05)
    p = np.array([0.6, 0.3, 0.07])  # 0.02 and 0.01 fall below 0.05 and are excluded
    assert c.entropy == pytest.approx(float(-(p * np.log(p)).sum()), rel=1e-9)
    assert c.n_hyps_entropy == 3
    # no re-renormalisation after thresholding
    assert c.entropy != pytest.approx(float(-((p / p.sum()) * np.log(p / p.sum())).sum()))


def test_renormalisation_is_over_beam_only():
    lps = [-1.0, -2.0, -4.0]  # sum of exp < 1: renormalised over the beam
    p = renormalise(lps)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] / p[1] == pytest.approx(math.e)
    c = score_cell("l", "c", [H("a", -1.0), H("b", -2.0), H("c", -4.0)])
    assert [r["prob_renorm"] for r in c.rows] == pytest.approx(list(p))


def test_entropy_single_hypothesis_is_zero_and_threshold_boundary_inclusive():
    assert thresholded_entropy([1.0]) == 0.0
    assert thresholded_entropy([0.95, 0.05]) == pytest.approx(-(0.95 * math.log(0.95) + 0.05 * math.log(0.05)))


def test_nonfinite_hypotheses_dropped_and_cells_flagged():
    c = score_cell("l", "c", [H("a", float("nan")), H("b", -1.0), H("c", float("-inf"))])
    assert c.finite and c.n_hyps == 3 and c.n_hyps_finite == 1
    assert c.top_hyp == "b" and c.entropy == 0.0
    assert sum(math.isnan(r["prob_renorm"]) for r in c.rows) == 2
    dead = score_cell("l", "d", [H("a", float("nan"))])
    assert not dead.finite and math.isnan(dead.surprisal_norm)
    empty = score_cell("l", "e", [])
    assert not empty.finite


def test_aggregation_mean_over_finite_cells_and_inf_for_all_nonfinite():
    cells = [score_cell("A", "c1", [H("a", -1.0)]),          # surprisal 0.5
             score_cell("A", "c2", [H("a b", -3.0)]),        # surprisal 1.0
             score_cell("A", "c3", [H("a", float("nan"))]),  # non-finite
             score_cell("B", "c1", [H("a", float("nan"))]),
             score_cell("B", "c2", [])]
    agg = aggregate_lemmas(cells, "surprisal_norm", ["c1", "c2", "c3"]).set_index("lemma_id")
    assert agg.at["A", "lemma_score"] == pytest.approx(0.75)
    assert agg.at["A", "n_cells_scored"] == 2 and agg.at["A", "n_nonfinite"] == 1
    assert math.isinf(agg.at["B", "lemma_score"]) and agg.at["B", "lemma_score"] > 0
    assert bool(agg.at["B", "all_nonfinite"])
    assert agg.at["B", "n_nonfinite"] == 3  # 2 non-finite + 1 never scored
    # per-cell scores are retained on the cell objects
    assert [c.surprisal_norm for c in cells[:2]] == pytest.approx([0.5, 1.0])


def test_ranking_direction_higher_score_first_and_inf_first():
    order = rank_lemmas({"low": 0.1, "high": 2.0, "mid": 1.0, "dead": float("inf")}, tie_seed=1)
    assert order == ["dead", "high", "mid", "low"]


def test_low_confidence_and_high_entropy_directions_end_to_end():
    confident = [score_cell("conf", "c", [H("a b", -0.05), H("a c", -6.0)])]
    unsure = [score_cell("unsure", "c", [H("a b", -1.4), H("a c", -1.5), H("a d", -1.6)])]
    for name in ("surprisal_norm", "entropy"):
        agg = aggregate_lemmas(confident + unsure, name, ["c"])
        order = rank_lemmas(dict(zip(agg.lemma_id, agg.lemma_score)), tie_seed=0)
        assert order[0] == "unsure", name


def test_deterministic_tie_breaking_by_sha256_key():
    ids = [f"r:x::{i}" for i in range(20)]
    scores = {i: 1.0 for i in ids}
    o1 = rank_lemmas(scores, tie_seed=99)
    o2 = rank_lemmas(dict(reversed(list(scores.items()))), tie_seed=99)
    assert o1 == o2 == sorted(ids, key=lambda l: tie_key(99, l))
    assert rank_lemmas(scores, tie_seed=100) != o1  # depends on the tie seed


def test_nan_score_cannot_be_ranked():
    with pytest.raises(ValueError):
        rank_lemmas({"a": float("nan")}, 0)
