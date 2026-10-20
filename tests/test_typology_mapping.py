"""Glottolog roll-up / map-down and the population link table (fixtures only)."""

import pandas as pd
import pytest

from morph_ldl.typology import grambank as gb
from morph_ldl.typology.stage import build_links
from test_typology_helpers import FIX

GB_LANGS = {"lang1", "lang2", "lang3", "iso1"}


def _glotto():
    log = gb.FileLog()
    d = FIX / "glottolog-cldf" / "cldf"
    return gb.Glottolog.from_frames(log.read_csv(d / "languages.csv"),
                                    log.read_csv(d / "values.csv", usecols=["Language_ID", "Parameter_ID", "Value"]))


def test_resolve_code_rules():
    g = _glotto()
    r = lambda c: gb.resolve_code(c, g, GB_LANGS)
    assert r("lang1")["glottocode"] == "lang1" and r("lang1")["basis"] == "exact"
    assert (r("dia1")["glottocode"], r("dia1")["basis"]) == ("lang1", "dialect_rollup")
    # group with exactly one Grambank-coded language-level descendant (lang5 not in Grambank)
    assert (r("grp2")["glottocode"], r("grp2")["basis"]) == ("lang3", "group_map_down")
    # group with several Grambank-coded descendants: no automatic match
    amb = r("grp1")
    assert amb["glottocode"] == "" and amb["basis"] == "group_ambiguous" and amb["candidates"] == ["lang1", "lang2"]
    assert r("grp3")["basis"] == "group_no_grambank_descendant" and r("grp3")["glottocode"] == ""
    assert r("NA")["basis"] == "unresolved" and r("")["basis"] == "unresolved"
    assert r("zzzz9999")["basis"] == "unresolved"
    # roll-up never goes below/above language level
    assert g.language_of("fam1") == "" and g.family("lang3") == ("fam1", "Fam One", False)
    assert g.family("iso1") == ("iso1", "Isolate One", True)


def _links(reviews=(), manual=()):
    log = gb.FileLog()
    main = log.read_csv(FIX / "gelato" / "populations.csv")
    s1 = log.read_csv(FIX / "gelato" / "tableS1.csv")
    return build_links(main, s1, _glotto(), GB_LANGS, list(reviews), list(manual))


def test_population_links():
    lk = _links()
    by = {(r.population, r.link_basis): r for r in lk.itertuples()}
    assert by[("PopA", "exact")].glottocode == "lang1" and by[("PopA", "exact")].match_status == "candidate"
    assert ("PopA", "proxy_gbi_tli") not in by                      # proxy = own language: no extra row
    assert by[("PopB", "dialect_rollup")].glottocode == "lang1"
    assert by[("PopB", "dialect_rollup")].match_status == "candidate"   # roll-up is not acceptance
    assert by[("PopC", "group_map_down")].glottocode == "lang3"
    amb = by[("PopD", "group_ambiguous")]
    assert amb.glottocode == "" and amb.match_status == "ambiguous" and amb.candidate_glottocodes == "lang1;lang2"
    assert by[("PopG", "group_no_grambank_descendant")].match_status == "unmatched"
    assert by[("PopE", "unresolved")].glottocode == ""
    px = by[("PopE", "proxy_gbi")]
    assert px.glottocode == "iso1" and px.is_proxy and px.match_status == "ambiguous"
    assert by[("PopF", "proxy_tli")].glottocode == "lang2"
    assert by[("PopF", "exact")].glottocode == "lang6"
    assert "accepted" not in set(lk["match_status"])
    assert not {"ancestry_q_files", "derived_K23_file"} & set(lk.columns)


def test_review_needs_human_confirmation():
    minimal = {"id": "t", "glottocode": "lang1", "population": "PopA", "status": "excluded",
               "confirmed_by": "PI", "confirmed_date": "2026-10-16"}
    row = _links([minimal]).query("population == 'PopA'").iloc[0]
    assert row.match_status == "excluded" and row.confirmed_date == "2026-10-16" and row.proposed_status == ""
    prop = {k: v for k, v in minimal.items() if not k.startswith("confirmed")} | {"reviewer_note": "n"}
    row = _links([prop]).query("population == 'PopA'").iloc[0]
    assert row.match_status == "candidate" and row.proposed_status == "excluded" and row.decision_reason == "n"
    row = _links([{**minimal, "confirmed_by": ""}]).query("population == 'PopA'").iloc[0]
    assert row.match_status == "candidate" and "malformed" in row.unresolved_issues


def test_review_rule_same_in_both_stages():
    """The data-stage crosswalk and the typology links give the same result for one entry."""
    from morph_ldl.data import gelato
    base = {"match_status": "candidate", "decision_reason": "auto", "unresolved_issues": "",
            "glottocode": "lang1", "population": "PopA"}
    from morph_ldl.typology.stage import apply_review
    cases = [None,
             {"id": "a", "glottocode": "lang1", "population": "PopA", "status": "accepted",
              "confirmed_by": "PI", "confirmed_date": "2026-10-16"},
             {"id": "b", "glottocode": "lang1", "population": "PopA", "status": "ambiguous", "reviewer_note": "r"},
             {"id": "c", "glottocode": "lang1", "population": "PopA", "status": "nope"}]
    for rv in cases:
        o = gelato.review_outcome(rv, "candidate", "auto")
        t = apply_review(dict(base), [rv] if rv else [])
        for c in ("match_status", "proposed_status", "decision_reason", "review_ref"):
            assert t.get(c, "") == o[c], (rv, c)
        for c in ("confirmed_by", "confirmed_date"):
            assert t.get(c, "") == o[c]
        assert all(i in t["unresolved_issues"] for i in o["issues"])


def test_manual_link():
    lk = _links(manual=[{"population": "PopD", "glottocode": "lang2", "reason": "fixture", "by": "test"}])
    m = lk[(lk.population == "PopD") & (lk.link_basis == "manual")].iloc[0]
    assert m.glottocode == "lang2" and m.match_status == "candidate"


def test_dialect_entries_substitute_only_for_languages_without_an_entry():
    from morph_ldl.typology.stage import dialect_substitutes
    langs = pd.DataFrame([
        {"ID": "fam1", "Name": "Fam", "Level": "family", "Language_ID": "", "Family_ID": "", "Macroarea": "",
         "ISO639P3code": ""},
        *[{"ID": i, "Name": i, "Level": "language", "Language_ID": "", "Family_ID": "fam1", "Macroarea": "",
           "ISO639P3code": ""} for i in ("lang1", "lang2")],
        *[{"ID": i, "Name": i, "Level": "dialect", "Language_ID": lg, "Family_ID": "fam1", "Macroarea": "",
           "ISO639P3code": ""} for i, lg in (("dia1", "lang1"), ("dia2", "lang2"), ("dia3", "lang2"))],
    ])
    glotto = gb.Glottolog.from_frames(langs, None)
    feats = ["GB1", "GB2", "GB3"]
    fs = {"features": feats, "present_codes": {g: ["1"] for g in feats}}
    valid = {g: ["0", "1"] for g in feats}
    vals = pd.DataFrame([(e, g, v) for e, row in {"lang1": "101", "dia1": "111", "dia2": "1?0", "dia3": "011"}.items()
                         for g, v in zip(feats, row)], columns=["Language_ID", "Parameter_ID", "Value"])
    ids = ["lang1", "dia1", "dia2", "dia3"]
    assert dialect_substitutes(ids, {"lang1"}, glotto, vals, fs, valid, "ignore") == {}
    # lang1 has its own entry (dia1 unused); lang2 takes its best-covered dialect entry (dia3: 3 coded > 2)
    assert dialect_substitutes(ids, {"lang1"}, glotto, vals, fs, valid, "substitute") == {"lang2": "dia3"}
    with pytest.raises(ValueError):
        dialect_substitutes(ids, {"lang1"}, glotto, vals, fs, valid, "merge")
