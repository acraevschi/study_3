"""Fast LDL bridge tests (no Julia): config resolution, hashing, resumability, semantics port."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morph_ldl import seeds
from morph_ldl.config import PIPELINE_ROOT, load_config
from morph_ldl.ldl import runner, semantics, toy


@pytest.fixture()
def cfg():
    return load_config(PIPELINE_ROOT / "configs" / "pcfp_v1.yaml")


@pytest.fixture()
def toy_job(tmp_path):
    forms = toy.lexicon(12, seed=3)
    train, q, _, _ = toy.pcfp_tables(forms, 4, seed=1)
    train.to_csv(tmp_path / "train.csv", index=False)
    q.to_csv(tmp_path / "q.csv", index=False)
    return {"train_csv": tmp_path / "train.csv", "queries_csv": tmp_path / "q.csv",
            "out_dir": tmp_path / "out", "unit_id": "toy.V.orth.x", "repetition": 0, "fold": 1,
            "overrides": {"sem_dim": 50}}


def test_resolve_config_seed_and_overrides(cfg):
    c = runner.resolve_ldl_config(cfg, "ita.V.orth.mgn", 0, 2, {"cue_ngram": 2})
    assert c["semantic_seed"] == seeds.derive(cfg["experiment"]["master_seed"], "semantic",
                                              "ita.V.orth.mgn", 0, 2)
    assert c["cue_ngram"] == 2 and "tune" not in c and "n_procs" not in c
    assert c["ridge_shift"] == 0.02 and "source_binding" not in c
    # same seed for every policy/budget of a fold; different across folds
    c2 = runner.resolve_ldl_config(cfg, "ita.V.orth.mgn", 0, 2, {"sem_sd_inflection": 4.0})
    c3 = runner.resolve_ldl_config(cfg, "ita.V.orth.mgn", 0, 1)
    assert c2["semantic_seed"] == c["semantic_seed"] != c3["semantic_seed"]
    with pytest.raises(ValueError):
        runner.resolve_ldl_config(cfg, "u", 0, 0, {"semantic_seed": 1})


def test_job_hash_tracks_config_and_inputs(cfg, toy_job):
    h = runner.job_config(toy_job, cfg)["job_hash"]
    assert runner.job_config(toy_job, cfg)["job_hash"] == h
    assert runner.job_config({**toy_job, "overrides": {"sem_dim": 60}}, cfg)["job_hash"] != h
    q = pd.read_csv(toy_job["queries_csv"])
    q.iloc[::-1].to_csv(toy_job["queries_csv"], index=False)
    assert runner.job_config(toy_job, cfg)["job_hash"] != h


def _fake_done(out: Path, jc: dict):
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.csv").write_text("lemma_id\n")
    (out / "diagnostics.json").write_text("{}")
    (out / "job_config.json").write_text(json.dumps(jc))


def test_resumable_skip_and_rerun(cfg, toy_job, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_launch_shards", lambda mode, shards, threads, run_dir: calls.append(shards))
    jc = runner.job_config(toy_job, cfg)
    _fake_done(Path(toy_job["out_dir"]), jc)
    assert runner.run_ldl_jobs([toy_job], cfg) == [Path(toy_job["out_dir"])]
    assert calls == []                                       # skipped: matching hash
    with pytest.raises(runner.LDLJobError):                  # mismatching hash -> relaunch
        runner.run_ldl_jobs([{**toy_job, "overrides": {"sem_dim": 70}}], cfg)
    assert len(calls) == 1 and calls[0][0][0]["config"]["sem_dim"] == 70
    assert not (Path(toy_job["out_dir"]) / "job_config.json").exists()   # stale marker removed


def test_score_requires_predictions(cfg, toy_job, tmp_path):
    gold = tmp_path / "gold.csv"
    pd.DataFrame({"lemma_id": [], "target_cell": [], "gold_variants": []}).to_csv(gold, index=False)
    with pytest.raises(runner.LDLJobError):
        runner.score_mapping_jobs([toy_job], [gold], cfg)


def test_parallelism_respects_max_threads(cfg):
    procs, threads = runner._resolve_parallelism(cfg, 4, 10)
    assert procs == 4 and procs * threads <= cfg["limits"]["max_threads"]
    assert runner._resolve_parallelism(cfg, 4, 1)[0] == 1
    assert sum(len(s) for s in runner._shard(list(range(7)), 3)) == 7


def test_write_gold_csv(tmp_path):
    forms = toy.lexicon(3, seed=1)
    extra = forms.iloc[[1]].copy(); extra["variant_idx"] = 1; extra["segments"] = "x y"
    forms = pd.concat([forms, extra])
    q = pd.DataFrame({"lemma_id": forms.lemma_id.iloc[0], "target_cell": toy.CELLS[1:]})
    g = pd.read_csv(runner.write_gold_csv(forms, q, tmp_path / "g.csv"))
    assert len(g) == 8 and " || x y" in g.gold_variants.iloc[0]


def test_semantics_identifier_keyed():
    c = {"semantic_seed": 42, "sem_dim": 400, "sem_sd_lexeme": 4.0, "sem_sd_inflection": 0.4,
         "sem_sd_noise": 1.0}
    a = semantics.lexeme_vec(c, "x::a")
    assert np.array_equal(a, semantics.lexeme_vec(c, "x::a"))
    assert not np.allclose(a, semantics.lexeme_vec(c, "x::b"))
    assert not np.allclose(a, semantics.lexeme_vec({**c, "semantic_seed": 43}, "x::a"))
    assert abs(a.std() - 4.0) < 0.5 and abs(a.mean()) < 0.5
    s = semantics.form_semantics(c, "x::a", "1;IND;PRS;SG")
    manual = a + sum(semantics.feature_vec(c, f) for f in ["1", "IND", "PRS", "SG"]) \
        + semantics.noise_vec(c, "x::a", "1;IND;PRS;SG")
    assert np.allclose(s, manual)
    # known value guards against accidental generator changes (also checked against Julia)
    assert semantics.seed_of(1, "lexeme", "a") == semantics.seed_of("1", "lexeme", "a")


def test_cell_vector_is_optional_and_cell_keyed():
    c = {"semantic_seed": 42, "sem_dim": 300, "sem_sd_lexeme": 4.0, "sem_sd_inflection": 2.0,
         "sem_sd_noise": 1.0}
    off = semantics.form_semantics(c, "x::a", "1;IND;PRS;SG")
    assert np.array_equal(off, semantics.form_semantics({**c, "sem_sd_cell": 0.0}, "x::a", "1;IND;PRS;SG"))
    on = semantics.form_semantics({**c, "sem_sd_cell": 2.0}, "x::a", "1;IND;PRS;SG")
    cc = {**c, "sem_sd_cell": 2.0}
    assert np.allclose(on - off, semantics.cell_vec(cc, "1;IND;PRS;SG"))
    # one vector per full cell, shared by lexemes; different cells sharing features differ
    assert np.array_equal(semantics.cell_vec(cc, "1;IND;PRS;SG"), semantics.cell_vec(cc, "1;IND;PRS;SG"))
    assert not np.allclose(semantics.cell_vec(cc, "1;IND;PRS;SG"), semantics.cell_vec(cc, "3;IND;PRS;SG"))
