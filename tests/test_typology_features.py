"""Feature-set declaration and counting rules of the typology stage."""

import copy

import pandas as pd
import pytest

from morph_ldl.typology import grambank as gb
from test_typology_helpers import FIX, default_block, fixture_typology

CORE = ("GB079 GB080 GB082 GB083 GB084 GB086 GB312 GB089 GB090 GB091 GB092 GB093 GB094 GB107 GB286 "
        "GB042 GB043 GB044 GB165 GB166 GB070 GB071 GB072 GB073 GB430 GB431 GB432 GB433 "
        "GB170 GB171 GB172 GB184 GB185 GB186 GB198").split()
EXCLUDED = "GB047 GB048 GB049 GB187 GB188 GB119 GB120 GB121 GB298 GB103 GB104 GB113 GB147 GB148 GB155 GB275".split()


def test_default_block_declares_the_35_core_features():
    fs = gb.parse_feature_set(default_block())
    assert fs["features"] == CORE and len(fs["features"]) == 35
    assert sorted(fs["excluded"]) == sorted(EXCLUDED)
    assert {g: c for g, c in fs["present_codes"].items()} == {g: ["1"] for g in CORE}
    assert len(fs["sensitivity"]["verbal"]) == 15
    assert len(fs["sensitivity"]["nominal"]) == 13
    assert len(fs["sensitivity"]["no_agreement"]) == 28
    assert not set(fs["sensitivity"]["no_agreement"]) & {"GB170", "GB171", "GB172", "GB184", "GB185", "GB186", "GB198"}


@pytest.mark.parametrize("mutate, msg", [
    (lambda t: t.pop("feature_set"), "feature_set is missing"),
    (lambda t: t["feature_set"].update(n_features=4), "n_features"),
    (lambda t: t["feature_set"]["domains"]["case"].append("GB080"), "listed twice"),
    (lambda t: t["feature_set"]["domains"]["case"].append("XX1"), "invalid Grambank ID"),
    (lambda t: t["feature_set"]["excluded"].update(bad=["GB044"]), "both included and excluded"),
    (lambda t: t["feature_set"].pop("present_codes"), "present_codes"),
    (lambda t: t["sensitivity_sets"].update(x=["nonexistent"]), "unknown domain"),
    (lambda t: t.pop("sensitivity_sets"), "sensitivity_sets"),
])
def test_ill_formed_feature_set_is_refused(mutate, msg):
    t = copy.deepcopy(fixture_typology())
    mutate(t)
    with pytest.raises(gb.FeatureSetError, match=msg):
        gb.parse_feature_set(t)


def _gb():
    log = gb.FileLog()
    d = FIX / "grambank" / "cldf"
    return {n: log.read_csv(d / f"{n}.csv") for n in ("values", "parameters", "codes", "languages")}


def test_multistate_feature_needs_explicit_present_codes():
    G = _gb()
    t = fixture_typology()
    t["feature_set"]["domains"]["case"].append("GB999")
    t["feature_set"]["n_features"] = 6
    fs = gb.parse_feature_set(t)
    with pytest.raises(gb.FeatureSetError, match="not binary"):
        gb.check_against_grambank(fs, G["parameters"], G["codes"])
    t["feature_set"]["present_codes"]["per_feature"] = {"GB999": ["1", "2"]}
    fs = gb.parse_feature_set(t)
    used = gb.check_against_grambank(fs, G["parameters"], G["codes"])
    assert used["GB999"]["present_codes"] == ["1", "2"]
    m = gb.feature_matrix(G["values"], fs["features"], {g: used[g]["codes"] for g in fs["features"]})
    c = gb.count_set(m, fs["features"], fs["present_codes"], {g: used[g]["codes"] for g in fs["features"]})
    assert c.loc["iso1", "n_present"] == 2           # GB080=1 and GB999=2
    t["feature_set"]["present_codes"]["per_feature"] = {"GB999": ["3"]}
    with pytest.raises(gb.FeatureSetError, match="not in codes.csv"):
        gb.check_against_grambank(gb.parse_feature_set(t), G["parameters"], G["codes"])


def test_counts_with_unknown_and_missing_values():
    G = _gb()
    fs = gb.parse_feature_set(fixture_typology())
    used = gb.check_against_grambank(fs, G["parameters"], G["codes"])
    valid = {g: used[g]["codes"] for g in fs["features"]}
    m = gb.feature_matrix(G["values"], fs["features"], valid)
    c = gb.count_set(m, fs["features"], fs["present_codes"], valid)
    # lang1: 1,1,1,?,0 -> coded 4, present 3
    assert c.loc["lang1"].to_dict() == {"n_features": 5, "n_coded": 4, "n_present": 3, "coverage": 0.8, "share": 0.75}
    # lang2: two zeros, three rows missing -> coded 2, present 0
    assert (c.loc["lang2", "n_coded"], c.loc["lang2", "n_present"], c.loc["lang2", "coverage"]) == (2, 0, 0.4)
    assert c.loc["lang2", "share"] == 0.0
    # iso1: 1,?,0,0,0 -> coded 4, present 1
    assert (c.loc["iso1", "n_coded"], c.loc["iso1", "n_present"]) == (4, 1)
    # a sub-set
    v = gb.count_set(m, fs["sensitivity"]["verbal"], fs["present_codes"], valid)
    assert (v.loc["iso1", "n_features"], v.loc["iso1", "n_coded"], v.loc["iso1", "n_present"]) == (2, 1, 1)


def test_unexpected_value_stops_counting():
    G = _gb()
    fs = gb.parse_feature_set(fixture_typology())
    vals = pd.concat([G["values"], pd.DataFrame([{"ID": "x", "Language_ID": "lang2", "Parameter_ID": "GB044",
                                                  "Value": "7"}])]).fillna("")
    with pytest.raises(ValueError, match="unexpected values"):
        gb.feature_matrix(vals, fs["features"], {g: ["0", "1"] for g in fs["features"]})
    dup = pd.concat([G["values"], G["values"].iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        gb.feature_matrix(dup, fs["features"], {g: ["0", "1"] for g in fs["features"]})
