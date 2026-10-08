"""Scoring: variants accepted, symbol-level edit distance, failures counted."""

import pandas as pd

from cv_fixtures import make_forms
from morph_ldl.cv import evaluate


def test_levenshtein_symbols():
    assert evaluate.levenshtein(list("kitten"), list("sitting")) == 3
    assert evaluate.levenshtein(["aː", "b"], ["a", "b"]) == 1  # multi-char symbol is one unit
    assert evaluate.levenshtein([], list("abc")) == 3


def test_score_items_variants_and_failures():
    forms = make_forms(20, missing_every=0)
    gold = evaluate.gold_table(forms)
    lid = "test:xxx-v::v0003are"   # i % 10 == 3: two variants of PANEL[0]
    pred = pd.DataFrame([
        {"lemma_id": lid, "target_cell": "1;IND;PRS;SG", "prediction_segments": " ".join("v0003ox"), "status": "ok"},
        {"lemma_id": lid, "target_cell": "3;IND;PRS;SG", "prediction_segments": " ".join("v0003o"), "status": "ok"},
        {"lemma_id": lid, "target_cell": "3;IND;PL;PRS", "prediction_segments": "", "status": "ok"},
        {"lemma_id": lid, "target_cell": "NFIN", "prediction_segments": "", "status": "no_candidate"},
    ])
    meta = dict(unit_id="u", repetition=0, outer_fold=0, policy="random", pool_cap=50, budget=10, model="ldl")
    items = evaluate.score_items(pred, gold, meta, {lid: "g"})
    assert items["correct"].tolist() == [True, False, False, False]
    assert items["edit_distance"].tolist() == [0, 1, len("v0003ano"), len("v0003are")]
    assert items["status"].tolist() == ["ok", "ok", "missing", "no_candidate"]
    s = evaluate.summarize(items, ["policy"])
    assert int(s["n_missing"][0]) == 1 and int(s["n_failed"][0]) == 1 and int(s["n_items"][0]) == 4
