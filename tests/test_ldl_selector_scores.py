"""Declared LDL-selector uncertainty scores (configs/pcfp_v1.yaml, docs/SELECTION.md)."""

import json
import math

import pandas as pd
import pytest

from morph_ldl import seeds
from morph_ldl.config import PIPELINE_ROOT, load_config
from morph_ldl.ldl.selector import add_cell_scores, lemma_scores, selector_configs


def cells(rows):
    return pd.DataFrame([{"lemma_id": l, "target_cell": c, "status": st, "supports": json.dumps(s),
                          "semantic_seed_idx": i} for l, c, st, s, i in rows])


def test_low_confidence_and_entropy_definitions():
    c = add_cell_scores(cells([("a", "x", "ok", [0.9, 0.1], 0), ("a", "y", "ok", [0.5], 0),
                               ("b", "x", "no_candidate", [], 0), ("b", "y", "error", [], 0)]), 0.1, -1.0, 10)
    assert c.u_low_confidence.tolist() == pytest.approx([0.1, 0.5, 2.0, 2.0])
    w = [math.exp(8.0) / (math.exp(8.0) + 1), 1 / (math.exp(8.0) + 1)]          # softmax([.9,.1]/.1)
    h = -sum(p * math.log(p) for p in w)
    assert c.u_entropy.tolist() == pytest.approx([h, 0.0, math.log(10), math.log(10)])
    assert c.scored.tolist() == [True, True, False, False]


def test_lemma_score_means_over_cells_then_seeds():
    c = add_cell_scores(cells([("a", "x", "ok", [0.9], 0), ("a", "y", "ok", [0.7], 0),
                               ("a", "x", "ok", [0.5], 1), ("a", "y", "ok", [0.3], 1),
                               ("b", "x", "no_candidate", [], 0), ("b", "x", "ok", [0.8], 1)]), 0.1, -1.0, 10)
    lem = lemma_scores(c, "low_confidence").set_index("lemma_id")
    assert lem.loc["a", "lemma_score"] == pytest.approx(((0.1 + 0.3) / 2 + (0.5 + 0.7) / 2) / 2)
    assert lem.loc["b", "lemma_score"] == pytest.approx((2.0 + 0.2) / 2)
    assert lem.loc["b", "n_nonfinite"] == 1 and lem.loc["a", "n_seeds"] == 2


def test_selector_semantic_seeds():
    cfg = load_config(PIPELINE_ROOT / "configs" / "pcfp_v1.yaml")
    cs = selector_configs(cfg, "ita.V.orth.mgn", 0, 2, {"cue_ngram": 2, "sem_sd_inflection": 1.0}, 3)
    m = cfg["experiment"]["master_seed"]
    assert cs[0]["semantic_seed"] == seeds.derive(m, "semantic", "ita.V.orth.mgn", 0, 2)     # = evaluated LDL
    assert [c["semantic_seed"] for c in cs[1:]] == [seeds.derive(m, "selector_semantic", "ita.V.orth.mgn", 0, 2, j)
                                                    for j in (1, 2)]
    assert all(c["sem_sd_inflection"] == 1.0 and c["cue_ngram"] == 2 for c in cs)
