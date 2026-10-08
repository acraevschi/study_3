"""Acquisition loop: budgets, random policy, gold isolation, outputs, end-to-end runs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morph_ldl.schemas import ACQ_LOG_COLUMNS, CELL_SCORE_COLUMNS, FORMS_COLUMNS, ORDER_COLUMNS
from morph_ldl.selection.acquisition import (CandidatePoolView, CandidateQuery, GoldAccessError, Oracle,
                                             SelectionSeeds, SelectionTask, build_examples, policy_dir,
                                             predict_queries, round_plan, run_acquisition)
from morph_ldl.selection.fixtures import eligible, fixture_roles, toy_forms

FIX = Path(__file__).parent / "fixtures" / "selection"
TASK = SelectionTask("toy.V.orth.test", "NFIN", ("1;PRS;SG", "3;PRS;SG", "3;PL;PRS"))
TINY = dict(d_model=32, n_heads=2, n_enc_layers=1, n_dec_layers=1, d_ff=64, dropout=0.1, warmup_steps=20,
            max_steps=60, eval_every=30, early_stop_patience=1, batch_size=32, beam_size=3, num_threads=1,
            device="cpu")


def cfg(budgets=(25, 33), batch=6, **sel):
    return {"selection": {"batch_size": batch, "budgets": list(budgets), "entropy_min_prob": 0.05,
                          "aggregation": "mean_cell"},
            "selector": {**TINY, **sel}}


@pytest.fixture(scope="module")
def toy():
    forms = toy_forms(90, seed=4)
    ids = eligible(forms, TASK.source_cell, TASK.panel_cells)
    roles = fixture_roles(ids, {"test": 15, "dev": 10, "seed": 9, "pool": 50}, seed=1)
    return forms, roles


SEEDS = SelectionSeeds.derive(7, TASK.unit_id, 0, 0)


def run(policy, forms, roles, out, c=None, **kw):
    return run_acquisition(policy, roles["seed"], roles["pool"], roles["dev"], forms, TASK, c or cfg(), SEEDS,
                           out, test_ids=roles["test"], **kw)


# ----------------------------------------------------------------- budgets

def test_round_plan_remainder_and_shortfall():
    p = round_plan(20, [100], 20, 500)
    assert p["batch_sizes"] == [20, 20, 20, 20] and not p["remainder_last_round"]
    p = round_plan(20, [50, 75], 20, 500)
    assert p["batch_sizes"] == [20, 20, 15] and p["remainder_last_round"]
    p = round_plan(20, [100], 20, 30)
    assert p["batch_sizes"] == [20, 10] and p["shortfall"] == 50
    with pytest.raises(ValueError):
        round_plan(20, [10], 5, 100)


def test_budget_accounting_and_nested_prefixes(toy, tmp_path):
    forms, roles = toy
    res = run("low_confidence", forms, roles, tmp_path / "lc")
    order = pd.read_csv(tmp_path / "lc" / "order.csv")
    assert list(order.columns) == ORDER_COLUMNS
    assert len(order) == 33 and order["acquisition_rank"].tolist() == list(range(1, 34))
    assert order["lemma_id"].is_unique
    assert set(order.loc[order["round"] == 0, "lemma_id"]) == set(roles["seed"])
    # 9 seed + batches 6,6,6,6 then remainder 0 -> 33 = 9 + 4*6: rounds 1..4
    assert order.groupby("round").size().to_dict() == {0: 9, 1: 6, 2: 6, 3: 6, 4: 6}
    s25 = pd.read_csv(tmp_path / "lc" / "samples" / "budget_25_lemmas.csv")
    s33 = pd.read_csv(tmp_path / "lc" / "samples" / "budget_33_lemmas.csv")
    assert s25["lemma_id"].tolist() == s33["lemma_id"].tolist()[:25] == order["lemma_id"].tolist()[:25]
    assert (s33["weight"] == 1.0).all()
    b25 = pd.read_csv(tmp_path / "lc" / "samples" / "budget_25.csv", keep_default_na=False)
    assert list(b25.columns) == FORMS_COLUMNS
    assert set(b25["lemma_id"]) == set(s25["lemma_id"])
    assert set(b25["cell_norm"]) == {TASK.source_cell, *TASK.panel_cells}
    assert not set(b25["lemma_id"]) & set(roles["test"]) and not set(b25["lemma_id"]) & set(roles["dev"])
    summ = json.loads((tmp_path / "lc" / "selection_summary.json").read_text())
    assert summ["budgets"]["25"]["n_lemmas"] == 25
    assert summ["budgets"]["25"]["n_training_examples_panel"] == 25 * 3
    assert summ["budgets"]["25"]["partial_last_round"] is True
    rounds = json.loads((tmp_path / "lc" / "rounds.json").read_text())["rounds"]
    assert [r["n_train_lemmas"] for r in rounds] == [9, 15, 21, 27]
    assert [r["n_train_examples"] for r in rounds] == [27, 45, 63, 81]
    assert all(r["model_hash"] and r["dev_acc"] is not None for r in rounds)
    log = pd.read_csv(tmp_path / "lc" / "acquisition_log.csv")
    assert list(log.columns) == ACQ_LOG_COLUMNS
    assert log.groupby("round")["selected"].sum().tolist() == [6, 6, 6, 6]
    assert log.groupby("round").size().tolist() == [50, 44, 38, 32]
    # selected = top of the round ranking, ranking follows the score descending
    for _, g in log.groupby("round"):
        g = g.sort_values("rank_in_round")
        assert g["selected"].tolist() == [True] * 6 + [False] * (len(g) - 6)
        assert (np.diff(g["lemma_score"].to_numpy()) <= 0).all()
    cs = pd.read_csv(tmp_path / "lc" / "cell_scores.csv", keep_default_na=False)
    assert list(cs.columns) == CELL_SCORE_COLUMNS
    assert set(cs["target_cell"]) == set(TASK.panel_cells)


def test_remainder_round_logged(toy, tmp_path):
    forms, roles = toy
    run("random", forms, roles, tmp_path / "r", c=cfg(budgets=[20], batch=4))
    rounds = json.loads((tmp_path / "r" / "rounds.json").read_text())
    assert rounds["plan"]["batch_sizes"] == [4, 4, 3] and rounds["plan"]["remainder_last_round"]
    assert rounds["rounds"][-1]["is_remainder_round"] is True


# ----------------------------------------------------------------- random policy

def test_random_policy_reproducible_and_seed_dependent(toy, tmp_path):
    forms, roles = toy
    a = run("random", forms, roles, tmp_path / "a").order
    b = run("random", forms, roles, tmp_path / "b").order
    assert a["lemma_id"].tolist() == b["lemma_id"].tolist()
    other = SelectionSeeds(SEEDS.selector_init, SEEDS.random_policy + 1, SEEDS.tie)
    c = run_acquisition("random", roles["seed"], roles["pool"], roles["dev"], forms, TASK, cfg(), other,
                        tmp_path / "c").order
    assert a["lemma_id"].tolist()[:9] == c["lemma_id"].tolist()[:9]  # same seed lemmas
    assert a["lemma_id"].tolist()[9:] != c["lemma_id"].tolist()[9:]
    # matched design: random uses the same seed set and budget as the active run
    assert len(a) == 33 and set(a["lemma_id"][:9]) == set(roles["seed"])


def test_random_selection_independent_of_round_structure(toy, tmp_path):
    forms, roles = toy
    a = run("random", forms, roles, tmp_path / "a", c=cfg(budgets=[33], batch=6)).order
    b = run("random", forms, roles, tmp_path / "b", c=cfg(budgets=[33], batch=24)).order
    assert a["lemma_id"].tolist() == b["lemma_id"].tolist()


# ----------------------------------------------------------------- gold isolation

def test_pool_view_has_no_gold(toy):
    forms, roles = toy
    view = CandidatePoolView.from_forms(forms, roles["pool"], TASK)
    q = view.queries()[0]
    assert isinstance(q, CandidateQuery)
    assert set(vars(q)) == {"lemma_id", "source_form", "source_segments", "source_cell", "target_cells"}
    gold_forms = set(forms.loc[forms.lemma_id.isin(roles["pool"]) & forms.cell_norm.isin(TASK.panel_cells), "form"])
    assert not any(v in gold_forms for qq in view.queries() for v in vars(qq).values() if isinstance(v, str))


def test_oracle_refuses_unselected_and_logs_reveals(toy):
    forms, roles = toy
    o = Oracle(forms, roles["seed"] + roles["pool"])
    with pytest.raises(GoldAccessError):
        o.reveal([roles["pool"][0]])
    with pytest.raises(GoldAccessError):
        o.select_and_reveal([roles["test"][0]], 1, 1)
    rows = o.select_and_reveal([roles["pool"][0]], 1, 10)
    assert set(rows["lemma_id"]) == {roles["pool"][0]}
    assert len(o.reveal([roles["pool"][0]])) == len(rows)
    with pytest.raises(GoldAccessError):
        o.reveal(roles["pool"][:2])
    with pytest.raises(ValueError):
        o.select_and_reveal([roles["pool"][0]], 2, 11)
    log = o.reveal_log()
    assert log[["round", "lemma_id", "reason", "acquisition_rank"]].values.tolist() == [[1, roles["pool"][0], "selected", 10]]


def _permute_gold(forms, lemma_ids, seed):
    """Shuffle target (non-source) rows' forms among the given lemmas."""
    f = forms.copy()
    m = f.lemma_id.isin(set(lemma_ids)) & (f.cell_norm != TASK.source_cell)
    perm = np.random.default_rng(seed).permutation(m.sum())
    for col in ("form", "segments"):
        vals = f.loc[m, col].to_numpy()
        f.loc[m, col] = vals[perm]
    return f


@pytest.mark.parametrize("policy", ["low_confidence", "high_entropy"])
def test_trajectory_invariant_to_unrevealed_pool_gold(toy, tmp_path, policy):
    forms, roles = toy
    a = run(policy, forms, roles, tmp_path / "a")
    never = sorted(set(roles["pool"]) - set(a.order["lemma_id"]))
    assert len(never) >= 10
    forms_b = _permute_gold(forms, never, seed=3)
    assert not forms_b.equals(forms)
    b = run(policy, forms_b, roles, tmp_path / "b")
    assert a.order["lemma_id"].tolist() == b.order["lemma_id"].tolist()
    for name in ("acquisition_log.csv", "cell_scores.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    # and the first round ignores *all* pool gold (model trained on the seed only)
    forms_c = _permute_gold(forms, roles["pool"], seed=4)
    c = run(policy, forms_c, roles, tmp_path / "c", c=cfg(budgets=[15], batch=6))
    la = pd.read_csv(tmp_path / "a" / "acquisition_log.csv")
    lc = pd.read_csv(tmp_path / "c" / "acquisition_log.csv")
    pd.testing.assert_frame_equal(la[la["round"] == 1].reset_index(drop=True), lc.reset_index(drop=True))


def test_reveals_logged_and_only_for_selected(toy, tmp_path):
    forms, roles = toy
    res = run("high_entropy", forms, roles, tmp_path / "h")
    rev = pd.read_csv(tmp_path / "h" / "oracle_reveals.csv")
    sel = rev[rev.reason.isin(["seed", "selected"])]
    assert sel["lemma_id"].tolist() == res.order["lemma_id"].tolist()
    assert set(rev.loc[rev.reason == "dev_early_stopping", "lemma_id"]) == set(roles["dev"])
    assert "oracle_policy_scoring" not in set(rev.reason)


def test_overlapping_roles_rejected(toy, tmp_path):
    forms, roles = toy
    with pytest.raises(ValueError):
        run_acquisition("random", roles["seed"], roles["pool"] + roles["test"][:1], roles["dev"], forms, TASK,
                        cfg(), SEEDS, tmp_path / "x", test_ids=roles["test"])
    with pytest.raises(ValueError):
        run_acquisition("random", roles["seed"], roles["pool"] + roles["dev"][:1], roles["dev"], forms, TASK,
                        cfg(), SEEDS, tmp_path / "y")


# ----------------------------------------------------------------- end to end / hooks

def test_end_to_end_on_italian_fixture(tmp_path):
    forms = pd.read_csv(FIX / "ita_v_mini_forms.csv", keep_default_na=False)
    forms["is_missing"] = forms["is_missing"].astype(str).str.lower().eq("true")
    task = SelectionTask("ita.V.orth.mgn", "NFIN",
                         ("1;IND;PRS;SG", "3;IND;PRS;SG", "3;IND;PL;PRS", "1;IND;PFV;PST;SG", "3;IND;PFV;PST;SG",
                          "3;IND;PFV;PL;PST", "3;COND;SG", "2;IMP;POS;SG"))
    ids = eligible(forms, task.source_cell, task.panel_cells)
    roles = fixture_roles(ids, {"test": 10, "dev": 10, "seed": 10, "pool": 40}, seed=2)
    seeds = SelectionSeeds.derive(1, task.unit_id, 0, 1)
    out = policy_dir(tmp_path, task.unit_id, 0, 1, "low_confidence")
    res = run_acquisition("low_confidence", roles["seed"], roles["pool"], roles["dev"], forms, task,
                          cfg(budgets=[20], batch=5, share_embeddings=True, aux_copy="pool"), seeds, out,
                          test_ids=roles["test"])
    assert out.as_posix().endswith("selection/ita.V.orth.mgn/rep0/fold1/low_confidence")
    for name in ("order.csv", "acquisition_log.csv", "cell_scores.csv", "rounds.json", "oracle_reveals.csv",
                 "samples/budget_20.csv", "samples/budget_20_allforms.csv", "stage_manifest.json"):
        assert (out / name).exists(), name
    assert len(res.order) == 20
    rounds = res.rounds
    assert rounds[0]["n_aux_copy_examples"] == 50  # seed + pool source anchors, never dev/test
    cs = pd.read_csv(out / "cell_scores.csv", keep_default_na=False)
    assert cs.groupby(["round", "lemma_id", "target_cell"]).size().max() <= 3
    assert (cs["prob_renorm"].astype(float) >= 0).all()
    q = pd.DataFrame({"lemma_id": roles["test"][:2], "source_cell": "NFIN",
                      "source_form": ["x", "y"], "source_segments": ["a m a r e", "c a n t a r e"],
                      "target_cell": "3;IND;PRS;SG"})
    sample = pd.read_csv(out / "samples" / "budget_20.csv", keep_default_na=False)
    sample["is_missing"] = sample["is_missing"].astype(str).str.lower().eq("true")
    dev_rows = forms[forms.lemma_id.isin(roles["dev"])]
    pred = predict_queries(sample, q, task=task, dev_rows=dev_rows, cfg=cfg(), seed=seeds.selector_init)
    assert pred.columns.tolist()[:4] == ["lemma_id", "source_cell", "target_cell", "prediction"]
    assert len(pred) == 2 and set(pred["status"]) <= {"ok", "no_hypothesis"}


def test_predict_queries_with_trained_selector_ignores_gold_columns(toy, tmp_path):
    forms, roles = toy
    res = run("low_confidence", forms, roles, tmp_path / "p", c=cfg(budgets=[15]), keep_selectors=True)
    sel = res.selectors[1]
    q = pd.DataFrame({"lemma_id": roles["test"][:3], "source_cell": "NFIN", "source_form": "",
                      "source_segments": ["b a l a r e", "m e r e", "t o k a r e"], "target_cell": "3;PRS;SG",
                      "gold": ["SECRET", "SECRET", "SECRET"]})
    p1 = predict_queries(sel, q)
    p2 = predict_queries(sel, q.drop(columns="gold"))
    pd.testing.assert_frame_equal(p1, p2)
    p3 = predict_queries(sel, [CandidateQuery(roles["test"][0], "", ("b", "a", "l", "a", "r", "e"), "NFIN",
                                              ("3;PRS;SG",))])
    assert p3.loc[0, "prediction_segments"] == p1.loc[0, "prediction_segments"]


def test_oracle_policy_is_labelled_and_logged(toy, tmp_path):
    forms, roles = toy
    res = run("oracle_incorrect", forms, roles, tmp_path / "o", c=cfg(budgets=[15]))
    assert res.summary["is_oracle_policy"] is True
    rev = pd.read_csv(tmp_path / "o" / "oracle_reveals.csv")
    assert (rev.reason == "oracle_policy_scoring").sum() == 50
    assert set(res.order.loc[res.order["round"] > 0, "score_name"]) == {"oracle_norm_edit_distance"}


def test_all_cells_mode_reports_exposure(toy, tmp_path):
    forms, roles = toy
    rows = forms[forms.lemma_id.isin(roles["seed"])]
    panel, st_p = build_examples(rows, TASK, mode="panel")
    allc, st_a = build_examples(rows, TASK, mode="all_cells")
    assert st_p["n_examples"] == 9 * 3 and st_a["n_examples"] == 9 * 3  # toy has only panel cells
    extra = rows[rows.cell_norm == "3;PRS;SG"].assign(cell_norm="2;PRS;SG", cell_orig="2;PRS;SG")
    allc2, st_a2 = build_examples(pd.concat([rows, extra]), TASK, mode="all_cells")
    assert st_a2["n_examples"] == 9 * 4 and st_a2["n_target_cells_distinct"] == 4
