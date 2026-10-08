"""Real-model LDL tests (JudiLing via the Julia batch runner). Marked slow.

One session fixture launches a single run_ldl_jobs call (2 Julia processes) on synthetic
paradigms, then the tests compare outputs across jobs.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morph_ldl.config import PIPELINE_ROOT, load_config
from morph_ldl.ldl import runner, semantics, toy

pytestmark = pytest.mark.slow
OVR = {"sem_dim": 120}
MARGIN = 4


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("ldl")
    cfg = load_config(PIPELINE_ROOT / "configs" / "pilot.yaml")
    forms = toy.lexicon(70, seed=11)
    # a held-out lemma with symbols never seen in training (unseen-cue accounting)
    novel = pd.DataFrame(toy.paradigm_rows("zuqqit", "a"))
    # a held-out lemma whose gold targets are far longer than any training form
    long_ = pd.DataFrame(toy.paradigm_rows("bal", "a"))
    long_.loc[long_.cell_norm != "NFIN", "form"] = "bal" + "x" * 40
    long_["segments"] = long_.form.map(" ".join)
    forms = pd.concat([forms, novel, long_], ignore_index=True)
    ids = list(dict.fromkeys(forms.lemma_id))
    held = ids[60:70] + ids[-2:]
    t1 = forms[forms.lemma_id.isin(ids[:40])]
    t2 = forms[forms.lemma_id.isin(ids[:20] + ids[40:60])]        # shares 20 lemmas with t1
    # a variant-1 row and a missing row that must be ignored for training
    junk = t1.iloc[[0, 1]].copy(); junk["variant_idx"] = [1, 0]; junk["is_missing"] = [False, True]
    junk["segments"] = ["q q q", ""]
    junk.iloc[1, junk.columns.get_loc("cell_norm")] = "FAKE"
    t1 = pd.concat([t1, junk])
    q = toy.queries_for(forms, held)
    gold = d / "gold.csv"
    runner.write_gold_csv(forms, q, gold)
    g = pd.read_csv(gold)
    files = {"t1": t1, "t2": t2, "q": q,
             "q_gold": q.merge(g, on=["lemma_id", "target_cell"]),
             "q_perm": q.merge(g.assign(gold_variants=g.gold_variants.sample(frac=1, random_state=1).values),
                               on=["lemma_id", "target_cell"]),
             "q_rev": pd.concat([q[q.lemma_id == l] for l in reversed(held)]),
             "q_one": q[q.lemma_id == held[3]]}
    for k, v in files.items():
        v.to_csv(d / f"{k}.csv", index=False)

    def job(name, train, queries):
        return {"train_csv": d / f"{train}.csv", "queries_csv": d / f"{queries}.csv",
                "out_dir": d / name, "unit_id": "toy.V.orth.test", "repetition": 0, "fold": 0,
                "overrides": OVR}
    jobs = [job("A", "t1", "q"), job("B", "t1", "q_gold"), job("C", "t1", "q_perm"),
            job("D", "t1", "q_rev"), job("E", "t1", "q_one"), job("F", "t2", "q")]
    outs = runner.run_ldl_jobs(jobs, cfg, n_procs=2)
    return {"d": d, "cfg": cfg, "jobs": jobs, "outs": dict(zip("ABCDEF", outs)), "forms": forms,
            "t1": t1, "q": q, "held": held, "gold": gold}


def P(world, k):
    return runner.read_predictions(world["outs"][k])


def key(df):
    return df.sort_values(["lemma_id", "target_cell"]).reset_index(drop=True)


def test_outputs_and_contract_columns(world):
    p = P(world, "A")
    assert list(p.columns) == ["lemma_id", "target_cell", "prediction", "prediction_segments",
                               "status", "n_candidates", "top_candidates", "support",
                               "n_source_cues", "n_source_cues_unseen", "unseen_target_features",
                               "binding_fit", "max_t"]
    assert len(p) == len(world["q"])                                   # no item dropped
    assert set(p.status) <= {"ok", "no_candidate", "error"}
    for r in p.itertuples():
        cands = json.loads(r.top_candidates)
        assert len(cands) == int(r.n_candidates) <= 10
        if cands:
            assert cands[0]["prediction"] == r.prediction == r.prediction_segments.replace(" ", "").replace("_", " ")
    diag = json.loads((world["outs"]["A"] / "diagnostics.json").read_text())
    assert diag["n_train_rows"] == 40 * 9 and diag["n_train_rows_raw"] == 40 * 9 + 2
    assert diag["train_comprehension_accuracy"] > 0.9 and diag["train_production_accuracy"] > 0.9
    assert diag["sem_dim"] == 120
    jc = json.loads((world["outs"]["A"] / "job_config.json").read_text())
    assert jc["ldl_config"]["semantic_seed"] == diag["semantic_seed"]
    # the decoder produces real, sometimes correct forms (weak sanity floor only: trigram
    # wug_refit scores ~5-15% even on this regular toy language, see LDL_PROTOCOL §6)
    gold = pd.read_csv(world["gold"])
    m = p.merge(gold, on=["lemma_id", "target_cell"])
    assert (m.prediction_segments == m.gold_variants).sum() >= 1
    assert (p.status == "ok").mean() > 0.5


def test_decoder_isolation_gold_withheld_present_permuted(world):
    a = key(P(world, "A"))
    pd.testing.assert_frame_equal(a, key(P(world, "B")))
    pd.testing.assert_frame_equal(a, key(P(world, "C")))


def test_reset_and_order_invariance(world):
    a = key(P(world, "A"))
    pd.testing.assert_frame_equal(a, key(P(world, "D")))
    one = key(P(world, "E"))
    pd.testing.assert_frame_equal(one, key(a[a.lemma_id == world["held"][3]]))


def test_long_gold_does_not_change_anything_and_max_t_is_gold_free(world):
    p = P(world, "A")
    t = world["t1"]
    tl = t[(t.variant_idx == 0) & (~t.is_missing.astype(bool))].segments.str.split().str.len().max()
    for r in p.itertuples():
        src = world["q"][world["q"].lemma_id == r.lemma_id].source_segments.iloc[0]
        assert int(r.max_t) == max(tl, len(src.split())) + MARGIN
    longp = p[p.lemma_id.str.endswith("::balare")]
    assert (longp.max_t.astype(int) < 40).all()            # gold length 43 never used


def test_unseen_cue_accounting(world):
    p = P(world, "A")
    t = world["t1"]
    t = t[(t.variant_idx == 0) & (~t.is_missing.astype(bool))]

    def ngrams(segs, n=3):
        toks = ["#"] + segs.split() + ["#"]
        return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}
    inv = set().union(*[ngrams(s) for s in t.segments])
    for lid, sub in p.groupby("lemma_id"):
        src = world["q"][world["q"].lemma_id == lid].source_segments.iloc[0]
        g = ngrams(src)
        assert (sub.n_source_cues.astype(int) == len(g)).all()
        assert (sub.n_source_cues_unseen.astype(int) == len(g - inv)).all()
    z = p[p.lemma_id.str.endswith("::zuqqitare")]
    assert (z.n_source_cues_unseen.astype(int) > 0).all()
    diag = json.loads((world["outs"]["A"] / "diagnostics.json").read_text())
    assert diag["n_items_with_unseen_source_cues"] == int((p.n_source_cues_unseen.astype(int) > 0).sum())


def test_semantic_stability_across_samples_and_python_parity(world):
    pa = json.loads((world["outs"]["A"] / "diagnostics.json").read_text())["semantic_probe"]
    pf = json.loads((world["outs"]["F"] / "diagnostics.json").read_text())["semantic_probe"]
    c = runner.job_config(world["jobs"][0], world["cfg"])["ldl_config"]
    shared = set(pa) & set(pf)
    assert shared                                           # both samples start with the same lemma
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
    lng = m[m.lemma_id.str.endswith("::balare")]
    assert lng.gold_path_longer_than_max_t.all() and (lng.n_gold_cues_outside_inventory > 0).all()
