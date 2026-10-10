"""Repeated-random PCFP design: fixed core folds x independent random draws."""

import numpy as np
import pandas as pd
import pytest

from morph_ldl.config import cv_design
from morph_ldl.cv import outcomes, pcfp, pipeline, splits
from morph_ldl.ldl.runner import resolve_ldl_config

from test_pcfp import make_forms


def cfg(**cv):
    base = {"design": "repeated_random", "inventory_size": 240, "n_folds": 4, "core_size": 30,
            "random_size": 40, "repetitions": [0, 1, 2]}
    base.update(cv)
    return {"experiment": {"id": "t", "master_seed": 7}, "cv": base}


@pytest.fixture(scope="module")
def lemmas():
    forms = make_forms()
    return pcfp.eligible_lemmas(forms, pcfp.eligible_cells(forms), "NFIN")[["lemma_id", "group_id"]]


@pytest.fixture(scope="module")
def mans(lemmas):
    return {d: splits.build_random_manifest(lemmas, cfg(), "u", d) for d in (0, 1, 2)}


def _set(man, k, role):
    return set(man[(man.outer_fold == k) & (man.role == role)].lemma_id)


def test_core_fixed_across_draws_random_shared_across_folds(mans):
    for d, man in mans.items():
        cores = [_set(man, k, "core") for k in range(4)]
        assert all(len(c) == 30 for c in cores)
        assert len(set().union(*cores)) == 120                               # disjoint
        rnd = [_set(man, k, "random") for k in range(4)]
        assert all(r == rnd[0] for r in rnd) and len(rnd[0]) == 40          # one draw, every fold
        core_groups = set(man[man.role == "core"].group_id)
        assert not core_groups & set(man[man.role == "random"].group_id)    # outside every core group
        for k in range(4):
            assert man[man.outer_fold == k].role.value_counts().to_dict() == {"unused": 170, "core": 30, "random": 40}
        # random role_rank is the draw order and is the same in every fold
        orders = [splits.roles(man, d, k)["random"] for k in range(4)]
        assert all(o == orders[0] for o in orders)
    for k in range(4):
        assert _set(mans[0], k, "core") == _set(mans[1], k, "core") == _set(mans[2], k, "core")
    assert _set(mans[0], 0, "random") != _set(mans[1], 0, "random")         # draws differ


def test_random_manifest_deterministic_and_roundtrip(tmp_path, lemmas, mans):
    pd.testing.assert_frame_equal(mans[1], splits.build_random_manifest(lemmas, cfg(), "u", 1))
    p = splits.write_manifest(mans[1], tmp_path)
    assert len(splits.load_manifest(p)) == len(mans[1])                      # dispatches to the random validator


def test_random_capacity_is_never_reduced(lemmas):
    with pytest.raises(splits.SplitError):
        splits.build_random_manifest(lemmas, cfg(random_size=121), "u", 0)  # 240 - 4*30 = 120 non-core
    with pytest.raises(splits.SplitError):
        splits.build_random_manifest(lemmas, cfg(core_size=60), "u", 0)     # cores fill the inventory


def test_random_validator_catches_tampering(mans):
    man = mans[0]
    # a fold whose random set differs from the others
    bad = man.copy()
    i = bad.index[(bad.outer_fold == 2) & (bad.role == "random")][0]
    j = bad.index[(bad.outer_fold == 2) & (bad.role == "unused")
                  & ~bad.group_id.isin(bad.loc[bad.role == "core", "group_id"])][0]
    bad.loc[i, "role"], bad.loc[j, "role"] = "unused", "random"
    with pytest.raises(splits.SplitError, match="random sets|straddles"):
        splits.validate_random_manifest(bad)
    # core sets that change between draws
    other = mans[1].copy()
    a = other.index[(other.outer_fold == 0) & (other.role == "core")][0]
    b = other.index[(other.outer_fold == 0) & (other.role == "unused")][0]
    other.loc[a, "role"], other.loc[b, "role"] = "unused", "core"
    with pytest.raises(splits.SplitError):
        splits.validate_random_manifest(pd.concat([man, other]))


def test_design_default_run_specs_and_comparisons():
    assert cv_design({"cv": {}}) == "repeated_random"
    with pytest.raises(ValueError):
        cv_design({"cv": {"design": "active"}})
    c = cfg()
    assert pipeline.run_specs(c) == [("random", 120, [40])]
    assert pipeline.comparisons(c) == [] and pipeline._policies(c) == ["random"]
    assert pipeline.run_specs(c, policies=["low_confidence"]) == []


def test_semantic_seed_scope_fold():
    base = {"experiment": {"master_seed": 7}, "ldl": {"sem_dim": 10}}
    per_fold = {**base, "cv": {"semantic_seed_scope": "fold"}}
    s = lambda c, r, k: resolve_ldl_config(c, "u", r, k)["semantic_seed"]
    assert s(per_fold, 0, 1) == s(per_fold, 3, 1) != s(per_fold, 0, 2)
    assert s(base, 0, 1) != s(base, 3, 1)                                    # default: per repetition and fold
    assert s(per_fold, 0, 1) == s(base, 0, 1)


def test_draw_variability_decomposition():
    rows = []
    draw_eff, fold_eff = [0.0, 0.1, -0.1], [0.05, -0.05]
    rng = np.random.default_rng(0)
    for r, de in enumerate(draw_eff):
        for k, fe in enumerate(fold_eff):
            p = 0.4 + de + fe
            n = 1000
            c = np.zeros(n, bool); c[: int(round(p * n))] = True
            rng.shuffle(c)
            rows.append(pd.DataFrame({"unit_id": "u", "item_set": "core", "policy": "random", "pool_cap": 120,
                                      "budget": 40, "model": "ldl", "repetition": r, "outer_fold": k,
                                      "correct": c, "edit_distance": 1.0}))
    items = pd.concat(rows, ignore_index=True)
    per_draw, summ = outcomes.draw_variability(items)
    assert per_draw.acc_micro.round(3).tolist() == [0.4, 0.5, 0.3]
    s = summ.iloc[0]
    assert s.n_draws == 3 and s.n_folds == 2
    assert s.fit_draw_effect_sd == pytest.approx(0.1) and s.fit_fold_effect_sd == pytest.approx(np.std([0.05, -0.05], ddof=1))
    assert s.fit_residual_sd == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        outcomes.draw_variability(items[~((items.repetition == 2) & (items.outer_fold == 1))])
