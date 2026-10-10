"""PCFP acquisition loop with the LDL selector interface (fast; a fake selector server).

The fake server's scores depend on *everything* it is given (training rows and candidate
table), so any leak of a candidate's forms into its inputs would change the logs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_pcfp import make_forms
from morph_ldl import seeds as seedlib
from morph_ldl.cv import pcfp
from morph_ldl.selection import SelectionSeeds
from morph_ldl.selection.acquisition import GoldAccessError
from morph_ldl.selection.ldl_acquisition import CANDIDATE_COLUMNS, ShownOracle, run_ldl_acquisition

CFG = {"selection": {"batch_size": 10, "budgets": [30, 40], "entropy_temperature": 0.1,
                     "no_candidate_support": -1.0},
       "ldl": {"max_can": 10}}


class FakeServer:
    """Deterministic stand-in for SelectorServer; logs every input it receives."""
    calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def score(self, train_csv, cands_csv, out_cells, out_comp, configs):
        train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
        cands = pd.read_csv(cands_csv, dtype=str, keep_default_na=False)
        assert list(cands.columns) == CANDIDATE_COLUMNS
        FakeServer.calls.append({"train": train, "cands": cands})
        tsig = hashlib.sha256("".join(sorted(train["segments"])).encode()).hexdigest()
        rows, comp = [], []
        for i, c in enumerate(configs):
            for r in cands.itertuples():
                for cell in r.shown_cells.split("|"):
                    h = int(hashlib.sha256(f"{tsig}|{r.citation_segments}|{cell}|{i}".encode()).hexdigest()[:8], 16)
                    s1 = (h % 1000) / 1000
                    rows.append({"lemma_id": r.lemma_id, "target_cell": cell, "status": "ok", "n_candidates": 2,
                                 "supports": json.dumps([s1, s1 / 2]), "top_support": s1,
                                 "top_prediction_segments": r.citation_segments, "top_equals_citation": True,
                                 "semantic_seed_idx": i, "semantic_seed": c["semantic_seed"]})
                comp.append({"lemma_id": r.lemma_id, "comp_cor": 0.0, "comp_rel_dist": 1.0, "n_citation_cues": 3,
                             "n_citation_cues_unseen": 0, "share_citation_cues_unseen": 0.0,
                             "citation_len": len(r.citation_segments.split()),
                             "semantic_seed_idx": i, "semantic_seed": c["semantic_seed"]})
        pd.DataFrame(rows).to_csv(out_cells, index=False)
        pd.DataFrame(comp).to_csv(out_comp, index=False)
        return {"fit_seconds": [0.0], "score_seconds": [0.0], "n_cues": [1], "n_train_rows": len(train)}


@pytest.fixture(scope="module")
def setup():
    forms = make_forms(300)
    cells = pcfp.eligible_cells(forms)
    lem = pcfp.eligible_lemmas(forms, cells, "NFIN")
    expo = pcfp.exposure_from_frame(pcfp.build_exposure_manifest(lem, cells, "NFIN", 7, "u", 7))
    ids = sorted(expo)
    core, seed, pool = ids[:20], ids[20:30], ids[30:90]
    labels = pcfp.lemma_labels(forms, pool)
    return {"forms": forms, "expo": expo, "core": core, "seed": seed, "pool": pool, "labels": labels}


def run(setup, tmp, policy="low_confidence", forms=None, pool=None):
    FakeServer.calls = []
    seeds = SelectionSeeds.derive(7, "u", 0, 0)
    pool = pool or setup["pool"]
    return run_ldl_acquisition(policy, setup["core"], setup["seed"], pool,
                               setup["forms"] if forms is None else forms, setup["expo"], setup["labels"], "NFIN",
                               "orth", CFG, seeds, tmp, selector_configs=[{"semantic_seed": 1}, {"semantic_seed": 2}],
                               server_factory=FakeServer)


def _alter(forms, ids):
    f = forms.copy()
    m = f["lemma_id"].isin(set(ids))
    f.loc[m, "segments"] = f.loc[m, "segments"].map(lambda s: "z " + s)
    f.loc[m, "form"] = "z" + f.loc[m, "form"]
    return f


def read(d, name):
    return (Path(d) / name).read_bytes()


def test_candidate_forms_never_matter(setup, tmp_path):
    a = run(setup, tmp_path / "a")
    selected = set(a.order["lemma_id"])
    never = [l for l in setup["pool"] if l not in selected]
    # altering every form (shown and hidden) of never-selected candidates: identical logs
    run(setup, tmp_path / "b", forms=_alter(setup["forms"], never))
    for name in ("order.csv", "acquisition_log.csv", "cell_scores.csv"):
        assert read(tmp_path / "a", name) == read(tmp_path / "b", name), name
    # altering every pool form (also of later-selected verbs): round 1 is identical
    run(setup, tmp_path / "c", forms=_alter(setup["forms"], setup["pool"]))
    la = pd.read_csv(tmp_path / "a" / "acquisition_log.csv")
    lc = pd.read_csv(tmp_path / "c" / "acquisition_log.csv")
    pd.testing.assert_frame_equal(la[la["round"] == 1], lc[lc["round"] == 1])


def test_selector_inputs_contain_no_candidate_or_hidden_forms(setup, tmp_path):
    res = run(setup, tmp_path)
    expo = setup["expo"]
    order = res.order
    for r, call in enumerate(FakeServer.calls, start=1):
        before = set(order.loc[order["round"] < r, "lemma_id"])
        train = call["train"]
        assert set(train.lemma_id) == set(setup["core"]) | before        # core + acquired so far only
        assert all(c in expo[l].shown for l, c in zip(train.lemma_id, train.cell_norm))
        cands = call["cands"]
        assert set(cands.lemma_id) == set(setup["pool"]) - before
        assert all(s == pcfp.citation_segments(setup["labels"][l], "orth")
                   for l, s in zip(cands.lemma_id, cands.citation_segments))
        assert all(tuple(s.split("|")) == expo[l].shown for l, s in zip(cands.lemma_id, cands.shown_cells))


def test_samples_budgets_and_core_exposure(setup, tmp_path):
    res = run(setup, tmp_path)
    expo = setup["expo"]
    for b in (30, 40):
        smp = pd.read_csv(tmp_path / "samples" / f"budget_{b}.csv")
        lem = pd.read_csv(tmp_path / "samples" / f"budget_{b}_lemmas.csv")
        sel = res.order["lemma_id"].iloc[:b].tolist()
        assert set(smp.lemma_id) == set(setup["core"]) | set(sel)
        assert (lem.role != "core").sum() == b and (lem.role == "core").sum() == 20
        pairs = set(zip(smp.lemma_id, smp.cell_norm))
        assert pairs == {(l, c) for l in set(setup["core"]) | set(sel) for c in expo[l].shown}
        s = res.summary["budgets"][str(b)]
        assert s["n_forms_total"] == len(smp) and s["n_forms_core"] == sum(expo[l].k for l in setup["core"])
    # nested budgets: budget 30 is a prefix of budget 40
    l30 = pd.read_csv(tmp_path / "samples" / "budget_30_lemmas.csv")
    l40 = pd.read_csv(tmp_path / "samples" / "budget_40_lemmas.csv")
    assert set(l30.lemma_id) <= set(l40.lemma_id)


def test_random_policy_needs_no_selector_and_shares_the_seed(setup, tmp_path):
    a = run_ldl_acquisition("random", setup["core"], setup["seed"], setup["pool"], setup["forms"], setup["expo"],
                            setup["labels"], "NFIN", "orth", CFG, SelectionSeeds.derive(7, "u", 0, 0), tmp_path / "r")
    b = run(setup, tmp_path / "lc")
    assert set(a.order[a.order["round"] == 0].lemma_id) == set(b.order[b.order["round"] == 0].lemma_id)
    a2 = run_ldl_acquisition("random", setup["core"], setup["seed"], setup["pool"], setup["forms"], setup["expo"],
                             setup["labels"], "NFIN", "orth", CFG, SelectionSeeds.derive(7, "u", 0, 0), tmp_path / "r2")
    pd.testing.assert_frame_equal(a.order, a2.order)
    assert not (tmp_path / "r" / "rounds").exists()


def test_deterministic_scoring_and_tie_breaking(setup, tmp_path):
    a = run(setup, tmp_path / "a")
    b = run(setup, tmp_path / "b")
    pd.testing.assert_frame_equal(a.order, b.order)

    class TieServer(FakeServer):
        def score(self, train_csv, cands_csv, out_cells, out_comp, configs):
            r = super().score(train_csv, cands_csv, out_cells, out_comp, configs)
            c = pd.read_csv(out_cells)
            c["supports"] = "[0.5, 0.25]"
            c.to_csv(out_cells, index=False)
            return r
    seeds = SelectionSeeds.derive(7, "u", 0, 0)
    t = run_ldl_acquisition("high_entropy", setup["core"], setup["seed"], setup["pool"], setup["forms"], setup["expo"],
                            setup["labels"], "NFIN", "orth", CFG, seeds, tmp_path / "t",
                            selector_configs=[{"semantic_seed": 1}], server_factory=TieServer)
    r1 = t.order[t.order["round"] == 1].lemma_id.tolist()
    want = sorted(setup["pool"], key=lambda l: seedlib.tie_key(seeds.tie, l))[:10]
    assert r1 == want


def test_oracle_refuses_unacquired_and_holds_no_hidden_cells(setup):
    o = ShownOracle(setup["forms"], setup["expo"], setup["core"] + setup["pool"])
    with pytest.raises(GoldAccessError):
        o.reveal(setup["pool"][:1])
    rows = o._rows
    assert all(c in setup["expo"][l].shown for l, c in zip(rows.lemma_id, rows.cell_norm))
    o.select_and_reveal(setup["pool"][:2], 1, 1, "selected")
    assert set(o.reveal(setup["pool"][:2]).lemma_id) == set(setup["pool"][:2])


def test_overlapping_roles_rejected(setup, tmp_path):
    with pytest.raises(ValueError):
        run_ldl_acquisition("random", setup["core"], setup["core"][:5] + setup["seed"][5:], setup["pool"],
                            setup["forms"], setup["expo"], setup["labels"], "NFIN", "orth", CFG,
                            SelectionSeeds.derive(7, "u", 0, 0), tmp_path)
