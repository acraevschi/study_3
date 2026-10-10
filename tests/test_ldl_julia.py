"""Real-model LDL tests (JudiLing via the Julia batch runner and selector server). Marked slow.

One session fixture launches a single run_ldl_jobs call (2 Julia processes) on synthetic
PCFP paradigms (known lexemes, hidden cells queried), then the tests compare outputs
across jobs. The selector tests drive one persistent selector process.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morph_ldl.config import PIPELINE_ROOT, load_config
from morph_ldl.ldl import runner, semantics, toy
from morph_ldl.ldl.selector import SelectorServer, add_cell_scores, lemma_scores

pytestmark = pytest.mark.slow
OVR = {"sem_dim": 120, "sem_sd_inflection": 2.0}   # 0.4 makes known-lexeme LDL copy shown forms
MARGIN = 4
IMP = "2;IMP;POS;SG"


@pytest.fixture(scope="session")
def cfg():
    return load_config(PIPELINE_ROOT / "configs" / "pcfp_v1.yaml")


@pytest.fixture(scope="session")
def world(tmp_path_factory, cfg):
    d = tmp_path_factory.mktemp("ldl")
    forms = toy.lexicon(80, seed=11)
    # a known lexeme whose hidden gold forms are far longer than any training form
    long_ = pd.DataFrame(toy.paradigm_rows("bal", "a"))
    long_.loc[long_.cell_norm != "NFIN", "form"] = "bal" + "x" * 40
    long_["segments"] = long_.form.map(" ".join)
    forms = pd.concat([forms, long_], ignore_index=True)
    train, q, gold_df, expo = toy.pcfp_tables(forms, 40, seed=5, max_shown=6)
    bal = long_.lemma_id.iloc[0]
    # 'bal' is known from its NFIN only; its hidden cells are queried
    train = pd.concat([train[train.lemma_id != bal], long_[long_.cell_norm == "NFIN"]], ignore_index=True)
    q = pd.concat([q, pd.DataFrame({"lemma_id": bal, "target_cell": toy.CELLS[1:]})], ignore_index=True)
    # no training form shows the imperative: its features (2, IMP, POS) are never trained
    train = train[train.cell_norm != IMP]
    q = q[q.lemma_id.isin(set(train.lemma_id))]
    ids = list(dict.fromkeys(q.lemma_id))
    gold = d / "gold.csv"
    runner.write_gold_csv(forms, q, gold)
    g = pd.read_csv(gold)
    # a variant-1 row and a missing row that must be ignored for training
    junk = train.iloc[[0, 1]].copy(); junk["variant_idx"] = [1, 0]; junk["is_missing"] = [False, True]
    junk["segments"] = ["q q q", ""]
    junk.iloc[1, junk.columns.get_loc("cell_norm")] = "FAKE"
    t1 = pd.concat([train, junk])
    lem_t1 = list(dict.fromkeys(train.lemma_id))
    t2 = train[train.lemma_id.isin(lem_t1[:20] + lem_t1[50:])]          # shares 20 verbs with t1
    q2 = q[q.lemma_id.isin(set(t2.lemma_id))]
    files = {"t1": t1, "t2": t2, "q": q, "q2": q2,
             "q_gold": q.merge(g, on=["lemma_id", "target_cell"]),
             "q_perm": q.merge(g.assign(gold_variants=g.gold_variants.sample(frac=1, random_state=1).values),
                               on=["lemma_id", "target_cell"]),
             "q_rev": q.iloc[::-1],
             "q_one": q[q.lemma_id == ids[3]],
             "q_unknown": pd.concat([q.head(3), pd.DataFrame({"lemma_id": ["toy:v::nope"], "target_cell": [toy.CELLS[1]]})])}
    for k, v in files.items():
        v.to_csv(d / f"{k}.csv", index=False)

    def job(name, train_, queries, extra=None):
        return {"train_csv": d / f"{train_}.csv", "queries_csv": d / f"{queries}.csv",
                "out_dir": d / name, "unit_id": "toy.V.orth.test", "repetition": 0, "fold": 0,
                "overrides": {**OVR, **(extra or {})}}
    jobs = [job("A", "t1", "q"), job("B", "t1", "q_gold"), job("C", "t1", "q_perm"),
            job("D", "t1", "q_rev"), job("E", "t1", "q_one"), job("F", "t2", "q2"),
            job("G", "t1", "q", {"predict_chunk": 7})]
    outs = runner.run_ldl_jobs(jobs, cfg, n_procs=2)
    bad = job("H", "t1", "q_unknown")
    runner.run_ldl_jobs([bad], cfg, n_procs=1, raise_on_error=False)
    return {"d": d, "cfg": cfg, "jobs": jobs, "outs": dict(zip("ABCDEFG", outs)), "forms": forms,
            "train": train, "t1": t1, "q": q, "ids": ids, "gold": gold, "bal": bal, "bad": bad}


def P(world, k):
    return runner.read_predictions(world["outs"][k])


def key(df):
    return df.sort_values(["lemma_id", "target_cell"]).reset_index(drop=True)


def test_outputs_and_contract_columns(world):
    p = P(world, "A")
    assert list(p.columns) == ["lemma_id", "target_cell", "prediction", "prediction_segments", "status",
                               "n_candidates", "top_candidates", "support", "unseen_target_features",
                               "n_train_forms_lemma", "max_t"]
    assert p[["lemma_id", "target_cell"]].equals(world["q"][["lemma_id", "target_cell"]].reset_index(drop=True))
    assert set(p.status) <= {"ok", "no_candidate", "error"}
    for r in p.itertuples():
        cands = json.loads(r.top_candidates)
        assert len(cands) == int(r.n_candidates) <= 10
        if cands:
            assert cands[0]["prediction"] == r.prediction == r.prediction_segments.replace(" ", "").replace("_", " ")
    n_train = len(world["train"])
    diag = json.loads((world["outs"]["A"] / "diagnostics.json").read_text())
    assert diag["n_train_rows"] == n_train and diag["n_train_rows_raw"] == n_train + 2
    assert diag["train_comprehension_accuracy"] > 0.9 and diag["train_production_accuracy"] > 0.9
    jc = json.loads((world["outs"]["A"] / "job_config.json").read_text())
    assert jc["ldl_config"]["semantic_seed"] == diag["semantic_seed"]
    tr = world["train"].groupby("lemma_id").size()
    assert (p.n_train_forms_lemma.astype(int).values == p.lemma_id.map(tr).values).all()


def test_known_lexeme_decoding_on_toy_paradigm(world):
    """Known-lexeme PCFP on a regular 3-class toy language: hidden cells of trained verbs
    are filled clearly above the copy baseline (cells with trained features only)."""
    p = P(world, "A")
    gold = pd.read_csv(world["gold"])
    m = p.merge(gold, on=["lemma_id", "target_cell"])
    m = m[(m.target_cell != IMP) & (m.lemma_id != world["bal"])]
    acc = (m.prediction_segments == m.gold_variants).mean()
    shown = world["train"].groupby("lemma_id").segments.agg(set)
    copy = np.mean([r.prediction_segments in shown[r.lemma_id] for r in m.itertuples()])
    assert acc > 0.15, acc
    assert acc > copy


def test_decoder_isolation_gold_withheld_present_permuted(world):
    a = key(P(world, "A"))
    pd.testing.assert_frame_equal(a, key(P(world, "B")))
    pd.testing.assert_frame_equal(a, key(P(world, "C")))


def test_order_and_chunk_invariance(world):
    a = key(P(world, "A"))
    pd.testing.assert_frame_equal(a, key(P(world, "D")))
    pd.testing.assert_frame_equal(a, key(P(world, "G")))
    one = key(P(world, "E"))
    pd.testing.assert_frame_equal(one, key(a[a.lemma_id == world["ids"][3]]))


def test_long_gold_does_not_change_anything_and_max_t_is_gold_free(world):
    p = P(world, "A")
    t = world["train"]
    tl = t.segments.str.split().str.len().max()
    assert (p.max_t.astype(int) == tl + MARGIN).all()
    assert (p[p.lemma_id == world["bal"]].max_t.astype(int) < 40).all()   # gold length 43 never used


def test_unseen_target_features_are_reported(world):
    p = P(world, "A")
    imp = p[p.target_cell == IMP]
    assert len(imp) and (imp.unseen_target_features != "").all()
    assert set(";".join(imp.unseen_target_features).split(";")) == {"2", "IMP", "POS"}
    assert (p[p.target_cell != IMP].unseen_target_features == "").all()
    diag = json.loads((world["outs"]["A"] / "diagnostics.json").read_text())
    assert diag["n_items_with_unseen_target_features"] == len(imp)
    assert set(diag["unseen_target_features"]) == {"2", "IMP", "POS"}


def test_unknown_lemma_fails_loudly(world):
    out = Path(world["bad"]["out_dir"])
    assert (out / "error.json").exists() and not (out / "job_config.json").exists()
    assert "not known lexemes" in json.loads((out / "error.json").read_text())["error"]


def test_semantic_stability_across_samples_and_python_parity(world):
    pa = json.loads((world["outs"]["A"] / "diagnostics.json").read_text())["semantic_probe"]
    pf = json.loads((world["outs"]["F"] / "diagnostics.json").read_text())["semantic_probe"]
    c = runner.job_config(world["jobs"][0], world["cfg"])["ldl_config"]
    shared = set(pa) & set(pf)
    assert shared
    for k in shared:
        assert pa[k] == pf[k]
    for k, v in pa.items():
        lid, cell = k.split("|")
        assert np.allclose(semantics.form_semantics(c, lid, cell)[:5], v, rtol=1e-12, atol=1e-12)


def test_julia_python_generator_parity():
    code = ('include("' + str(PIPELINE_ROOT / "julia" / "src" / "LDLRunner.jl") + '"); '
            'using .LDLRunner, JSON; print(JSON.json(['
            'gaussian_vector(seed_of(7, "lexeme", "mgn_data:ita-v::àbc"), 1001, 4.0), '
            'gaussian_vector(seed_of(7, "feature", "PRS"), 10, 0.4)]))')
    out = subprocess.run([runner.julia_executable(), f"--project={runner.JULIA_PROJECT}",
                          "--startup-file=no", "-e", code], capture_output=True, text=True, check=True)
    jl = json.loads(out.stdout.strip().splitlines()[-1])
    py = [semantics.gaussian_vector(semantics.seed_of(7, "lexeme", "mgn_data:ita-v::àbc"), 1001, 4.0),
          semantics.gaussian_vector(semantics.seed_of(7, "feature", "PRS"), 10, 0.4)]
    for a, b in zip(jl, py):
        assert np.allclose(a, b, rtol=1e-13, atol=1e-13)


def test_resumability_no_relaunch(world, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("relaunched a finished job")
    monkeypatch.setattr(runner, "_launch_shards", boom)
    assert runner.run_ldl_jobs(world["jobs"], world["cfg"], n_procs=2) == list(world["outs"].values())


def test_score_mapping_after_prediction(world):
    job = world["jobs"][0]
    before = (world["outs"]["A"] / "predictions.csv").read_bytes()
    [mq] = runner.score_mapping_jobs([job], [world["gold"]], world["cfg"])
    m = pd.read_csv(mq)
    p = P(world, "A")
    assert len(m) == len(p)
    assert {"chat_gold_cor", "n_gold_cues_outside_inventory", "n_gold_cues_below_threshold",
            "gold_in_top_candidates", "gold_rank_in_candidates", "gold_reachable"} <= set(m.columns)
    assert (world["outs"]["A"] / "predictions.csv").read_bytes() == before
    mp = m.merge(p, on=["lemma_id", "target_cell"])
    gold = pd.read_csv(world["gold"]).set_index(["lemma_id", "target_cell"]).gold_variants
    correct = np.array([r.prediction_segments == gold[(r.lemma_id, r.target_cell)] for r in mp.itertuples()])
    assert (mp.gold_rank_in_candidates[correct] == 1).all()
    lng = m[m.lemma_id == world["bal"]]
    assert lng.gold_path_longer_than_max_t.all() and (lng.n_gold_cues_outside_inventory > 0).all()


# ----------------------------------------------------------------------------- LDL selector

def test_rank_one_update_matches_full_refit(world, cfg):
    """add_row (Sherman-Morrison on G and F) equals a full JudiLing refit with the row."""
    code = f'''
include("{PIPELINE_ROOT / "julia" / "src" / "LDLRunner.jl"}")
using .LDLRunner, JSON
for grams in (2, 3)
    c = LDLConfig(Dict("cue_ngram" => grams, "semantic_seed" => 99, "sem_dim" => 150))
    tr, _ = read_training("{world["d"] / "t1.csv"}")
    bg = fit_background(tr, c)
    S = vcat([target_semantics(c, "toy:v::zzqueta", x)' for x in ["1;IND;PRS;SG", "3;COND;SG", "NFIN"]]...)
    Cf, Ff, Cr, Fr = full_refit_with_row(bg, "toy:v::zzqueta", "NFIN", "z z q u e t a r e", S)
    println(JSON.json(Dict("chat" => maximum(abs.(Cf .- Cr)), "F" => maximum(abs.(Ff .- Fr)),
                           "scale" => maximum(abs.(Cf)), "novel" => size(Cf, 2) - length(bg.f2i))))
end
'''
    out = subprocess.run([runner.julia_executable(), f"--project={runner.JULIA_PROJECT}", "--startup-file=no",
                          "-e", code], capture_output=True, text=True, check=True)
    for line in out.stdout.strip().splitlines()[-2:]:
        r = json.loads(line)
        assert r["novel"] > 0                       # the citation brings unseen cues
        assert r["chat"] < 1e-8 and r["F"] < 1e-8, r


@pytest.fixture(scope="module")
def selector_round(world, cfg, tmp_path_factory):
    d = tmp_path_factory.mktemp("sel")
    forms = toy.lexicon(95, seed=11)
    trained = set(world["train"].lemma_id)
    pool = [l for l in dict.fromkeys(forms.lemma_id) if l not in trained][:8]
    nfin = forms[forms.cell_norm == "NFIN"].set_index("lemma_id").segments
    rng = np.random.default_rng(0)
    cands = pd.DataFrame([{"lemma_id": l, "citation_cell": "NFIN", "citation_segments": nfin[l],
                           "shown_cells": "|".join(sorted(rng.choice(toy.CELLS[1:], 3, replace=False)))}
                          for l in pool])
    cfgs = [dict(runner.resolve_ldl_config(cfg, "toy.V.orth.test", 0, 0, OVR)),
            {**runner.resolve_ldl_config(cfg, "toy.V.orth.test", 0, 0, OVR), "semantic_seed": 4242}]

    def score(server, cand_df, name):
        cand_df.to_csv(d / f"{name}.csv", index=False)
        server.score(world["d"] / "t1.csv", d / f"{name}.csv", d / f"{name}_cells.csv", d / f"{name}_comp.csv", cfgs)
        cells = pd.read_csv(d / f"{name}_cells.csv", keep_default_na=False, dtype={"supports": str,
                                                                                    "top_prediction_segments": str})
        return cells.sort_values(["semantic_seed_idx", "lemma_id", "target_cell"]).reset_index(drop=True), \
            pd.read_csv(d / f"{name}_comp.csv").sort_values(["semantic_seed_idx", "lemma_id"]).reset_index(drop=True)

    altered = cands.copy()
    altered.loc[0, "citation_segments"] = "q u o r b a r e"
    with SelectorServer(d / "server.log", threads=1) as srv:
        res = {"base": score(srv, cands, "base"), "again": score(srv, cands, "again"),
               "rev": score(srv, cands.iloc[::-1], "rev"), "alt": score(srv, altered, "alt"),
               "sub": score(srv, cands.iloc[2:5], "sub")}
    return {"cands": cands, "res": res, "cfgs": cfgs}


def test_selector_scores_deterministic_and_order_free(selector_round):
    base_cells, base_comp = selector_round["res"]["base"]
    for name in ("again", "rev"):
        c, p = selector_round["res"][name]
        pd.testing.assert_frame_equal(base_cells, c)
        pd.testing.assert_frame_equal(base_comp, p)
    # each candidate is scored from the same background (citation row removed afterwards):
    # scoring a subset gives the same rows for those candidates
    sub_cells, _ = selector_round["res"]["sub"]
    ids = set(sub_cells.lemma_id)
    pd.testing.assert_frame_equal(sub_cells, base_cells[base_cells.lemma_id.isin(ids)].reset_index(drop=True))
    assert set(base_cells.semantic_seed_idx) == {0, 1}


def test_selector_only_the_citation_form_matters(selector_round):
    base_cells, _ = selector_round["res"]["base"]
    alt_cells, _ = selector_round["res"]["alt"]
    changed = selector_round["cands"].lemma_id.iloc[0]
    others_b = base_cells[base_cells.lemma_id != changed].reset_index(drop=True)
    others_a = alt_cells[alt_cells.lemma_id != changed].reset_index(drop=True)
    pd.testing.assert_frame_equal(others_b, others_a)
    assert not base_cells[base_cells.lemma_id == changed].supports.equals(
        alt_cells[alt_cells.lemma_id == changed].supports)


def test_selector_scores_are_usable(selector_round, cfg):
    cells, comp = selector_round["res"]["base"]
    s = add_cell_scores(cells, 0.1, -1.0, 10)
    for pol in ("low_confidence", "high_entropy"):
        lem = lemma_scores(s, pol)
        assert len(lem) == len(selector_round["cands"]) and lem.lemma_score.notna().all()
        assert lem.n_seeds.eq(2).all()
    assert comp.share_citation_cues_unseen.between(0, 1).all()


def test_cell_vector_and_tolerant_decoding(tmp_path, cfg):
    """sem_sd_cell adds V(cell) identically in Julia and Python; tolerant learn_paths runs
    end to end on known-lexeme queries."""
    forms = toy.lexicon(60, seed=3)
    train, q, _, _ = toy.pcfp_tables(forms, 30, seed=2, max_shown=6)
    q = q[q.lemma_id.isin(set(train.lemma_id))].head(40)
    train.to_csv(tmp_path / "train.csv", index=False); q.to_csv(tmp_path / "q.csv", index=False)
    ovr = {**OVR, "sem_sd_cell": 1.5, "tolerance": True, "tolerance_floor": 0.0, "max_tolerance": 1}
    job = dict(train_csv=str(tmp_path / "train.csv"), queries_csv=str(tmp_path / "q.csv"),
               out_dir=str(tmp_path / "tol"), unit_id="toy", repetition=0, fold=0, overrides=ovr)
    out = runner.run_ldl_jobs([job], cfg, n_procs=1)[0]
    pred = pd.read_csv(out / "predictions.csv", keep_default_na=False)
    assert len(pred) == len(q) and (pred.status == "ok").mean() > 0.9
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["tolerant"] is True and diag["max_tolerance"] == 1 and diag["sem_sd_cell"] == 1.5
    c = runner.job_config(job, cfg)["ldl_config"]
    for k, v in diag["semantic_probe"].items():
        lid, cell = k.split("|")
        assert np.allclose(semantics.form_semantics(c, lid, cell)[:5], v, rtol=1e-12, atol=1e-12)
