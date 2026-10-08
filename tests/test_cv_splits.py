"""Split manifests: grouping, determinism, exact sizes, nesting, no silent reduction."""

import pandas as pd
import pytest

from cv_fixtures import PANEL, SOURCE, make_forms, small_cfg
from morph_ldl.cv import splits


@pytest.fixture(scope="module")
def forms():
    return make_forms()


def _man(forms, cfg, rep=0):
    lem = splits.eligible_lemmas(forms, SOURCE, PANEL)
    return splits.build_split_manifest(lem, cfg, "xxx.V.orth.test", rep)


def test_eligibility_excludes_missing_panel_cells(forms):
    lem = splits.eligible_lemmas(forms, SOURCE, PANEL)
    missing = forms.loc[forms["is_missing"], "lemma_id"].unique()
    assert len(missing) > 0
    assert not set(missing) & set(lem["lemma_id"])
    assert len(lem) == forms["lemma_id"].nunique() - len(missing)


def test_manifest_deterministic_and_exact_sizes(forms):
    cfg = small_cfg()
    a, b = _man(forms, cfg), _man(forms, cfg)
    pd.testing.assert_frame_equal(a, b)
    for k in range(3):
        r = splits.roles(a, 0, k)
        assert (len(r["dev"]), len(r["seed"]), len(r["pool"])) == (20, 10, 100)
    tests = a[a["role"] == "test"]
    assert tests["lemma_id"].nunique() == 240 and not tests["lemma_id"].duplicated().any()


def test_groups_never_straddle_roles(forms):
    man = _man(forms, small_cfg())
    shared = forms.groupby("group_id")["lemma_id"].nunique()
    assert (shared > 1).any(), "fixture must contain multi-lemma groups"
    for _, fm in man.groupby("outer_fold"):
        assert (fm.groupby("group_id")["role"].nunique() == 1).all()


def test_repetitions_change_folds_not_inventory(forms):
    cfg = small_cfg()
    a, b = _man(forms, cfg, 0), _man(forms, cfg, 1)
    assert set(a["lemma_id"]) == set(b["lemma_id"])
    fa = a[a.role == "test"].set_index("lemma_id")["outer_fold"]
    fb = b[b.role == "test"].set_index("lemma_id")["outer_fold"]
    assert (fa.sort_index() != fb.sort_index()).any()


def test_nested_pool_caps(forms):
    man = _man(forms, small_cfg())
    full = splits.roles(man, 0, 0)["pool"]
    small = splits.roles(man, 0, 0, pool_cap=50)["pool"]
    assert small == full[:50]


def test_no_silent_reduction(forms):
    with pytest.raises(splits.SplitError):
        _man(forms, small_cfg(inventory_size=10_000))
    with pytest.raises(splits.SplitError):
        _man(forms, small_cfg(pool_cap=200))           # does not fit outside a test fold
    cfg = small_cfg(pool_cap=20, pool_cap_sensitivity=[])
    with pytest.raises(splits.SplitError):           # max budget 60 - seed 10 > pool 20
        _man(forms, cfg)


def test_manifest_roundtrip(tmp_path, forms):
    man = _man(forms, small_cfg())
    p = splits.write_manifest(man, tmp_path)
    back = splits.load_manifest(p)
    assert back["lemma_id"].tolist() == man["lemma_id"].tolist()


def test_auxiliary_set_disjoint_from_inventory_groups(forms):
    cfg = small_cfg(inventory_size=180, pool_cap=50, pool_cap_sensitivity=[])
    lem = splits.eligible_lemmas(forms, SOURCE, PANEL)
    man = splits.build_split_manifest(lem, cfg, "xxx.V.orth.test", 0)
    aux = splits.build_auxiliary_manifest(lem, man["lemma_id"].unique(),
                                          {"tune_background": 30, "tune_heldout": 20, "copy_anchor": 40},
                                          7, "xxx.V.orth.test")
    r = splits.aux_roles(aux)
    assert (len(r["tune_background"]), len(r["tune_heldout"]), len(r["copy_anchor"])) == (30, 20, 40)
    assert not set(aux["group_id"]) & set(man["group_id"])
    assert not set(aux["lemma_id"]) & set(man["lemma_id"])
    with pytest.raises(splits.SplitError):
        splits.build_auxiliary_manifest(lem, man["lemma_id"].unique(),
                                        {"tune_background": 500, "tune_heldout": 1, "copy_anchor": 1},
                                        7, "xxx.V.orth.test")
