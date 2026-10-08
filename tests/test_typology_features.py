"""Feature-set declaration and counting rules of the typology stage."""

import copy

import pandas as pd
import pytest

from morph_ldl.typology import grambank as gb
from test_typology_helpers import FIX, default_block, fixture_typology

CATEGORIES = {
    "tense": ["GB082", "GB083", "GB084"], "aspect": ["GB086"], "mood": ["GB312"],
    "person_indexing": ["GB089", "GB090", "GB091", "GB092", "GB093", "GB094"], "negation": ["GB107"],
    "polar_interrogation": ["GB285", "GB286"], "nominal_number": ["GB042", "GB043", "GB044", "GB165", "GB166"],
    "case": ["GB070", "GB071", "GB072", "GB073"], "possessor_affix": ["GB430", "GB432"],
    "possessed_affix": ["GB431", "GB433"], "gender_agreement": ["GB170", "GB171", "GB172", "GB198"],
    "number_agreement": ["GB184", "GB185", "GB186"]}
EXCLUDED = ("GB079 GB080 GB047 GB048 GB049 GB187 GB188 GB119 GB120 GB121 GB298 GB103 GB104 GB113 GB147 GB148 "
            "GB155 GB275").split()


def test_default_block_declares_the_12_inflection_categories():
    fs = gb.parse_feature_set(default_block())
    assert fs["features"] == list(CATEGORIES) and fs["categories"] == CATEGORIES
    assert sorted(fs["excluded"]) == sorted(EXCLUDED)
    assert fs["present_codes"] == {c: ["1"] for c in CATEGORIES}
    assert len(fs["sensitivity"]["verbal"]) == 6 and len(fs["sensitivity"]["nominal"]) == 4
    assert fs["sensitivity"]["no_agreement"] == list(CATEGORIES)[:10]


def _cat_block(cats, domains):
    t = fixture_typology()
    t["feature_set"] = {"id": "fixture_cats", "n_features": len(cats), "categories": cats, "domains": domains,
                        "excluded": {"other": ["GB020"]}, "present_codes": {"default": ["1"], "per_feature": {}}}
    t["sensitivity_sets"] = {"verbal": [next(iter(domains))]}
    return t


def test_category_or_merge_rule():
    G = _gb()
    t = _cat_block({"verb": ["GB080", "GB082"], "noun": ["GB044"], "mixed": ["GB070", "GB170"]},
                   {"verbal_tam": ["verb"], "nominal_number": ["noun"], "agreement": ["mixed"]})
    fs = gb.parse_feature_set(t)
    used = gb.check_against_grambank(fs, G["parameters"], G["codes"])
    assert used["verb"]["sources"].keys() == {"GB080", "GB082"} and used["verb"]["codes"] == ["0", "1"]
    valid = {c: used[c]["codes"] for c in fs["features"]}
    m = gb.category_matrix(G["values"], fs, valid)
    assert m.loc["lang1"].tolist() == ["1", "1", "?"]       # 1|1, 1, ?|0
    assert m.loc["iso1"].tolist() == ["1", "0", "0"]        # 1|?, 0, 0|0
    assert m.loc["lang2"].tolist() == ["0", "", ""]         # 0|0, no rows, no rows
    c = gb.count_set(m, fs["features"], fs["present_codes"], valid)
    assert (c.loc["lang1", "n_coded"], c.loc["lang1", "n_present"]) == (2, 2)
    assert (c.loc["iso1", "n_coded"], c.loc["iso1", "n_present"]) == (3, 1)


@pytest.mark.parametrize("cats, domains, extra, msg", [
    ({"a": ["GB080"]}, {"verbal_tam": ["b"]}, {}, "not a declared category"),
    ({"a": ["GB080"], "b": ["GB080"]}, {"verbal_tam": ["a", "b"]}, {}, "source of two categories"),
    ({"a": ["GB080"], "b": ["GB082"]}, {"verbal_tam": ["a"]}, {"n_features": 1}, "not placed in any domain"),
    ({"a": ["GB020"]}, {"verbal_tam": ["a"]}, {}, "both included and excluded"),
    ({"a": ["GB080"]}, {"verbal_tam": ["a"]}, {"present_codes": {"default": ["1", "2"]}}, "binary sources only"),
    ({"A": ["GB080"]}, {"verbal_tam": ["A"]}, {}, "invalid category name"),
])
def test_ill_formed_category_set_is_refused(cats, domains, extra, msg):
    t = _cat_block(cats, domains)
    t["feature_set"].update(extra)
    with pytest.raises(gb.FeatureSetError, match=msg):
        gb.parse_feature_set(t)


def test_category_sources_must_be_binary():
    G = _gb()
    t = _cat_block({"a": ["GB080", "GB999"]}, {"verbal_tam": ["a"]})
    with pytest.raises(gb.FeatureSetError, match="not binary"):
        gb.check_against_grambank(gb.parse_feature_set(t), G["parameters"], G["codes"])


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
