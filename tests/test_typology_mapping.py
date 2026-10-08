"""Glottolog roll-up / map-down and the population link table (fixtures only)."""

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
    full = {"id": "t", "glottocode": "lang1", "population": "PopA", "status": "accepted",
            "variety_review": "v", "community_review": "c", "locality_review": "l",
            "source_period_review": "s", "evidence": "e", "reviewer_note": "n", "reviewer": "agent",
            "date": "2026-10-08"}
    row = _links([full]).query("population == 'PopA'").iloc[0]
    assert row.match_status == "candidate" and row.proposed_status == "accepted"
    row = _links([{**full, "confirmed_by": "PI", "confirmed_date": "2026-10-09"}]).query("population == 'PopA'").iloc[0]
    assert row.match_status == "accepted" and row.confirmed_by == "PI"
    partial = {k: v for k, v in full.items() if k != "locality_review"}
    row = _links([partial]).query("population == 'PopA'").iloc[0]
    assert row.match_status == "candidate" and "not applied" in row.unresolved_issues


def test_manual_link():
    lk = _links(manual=[{"population": "PopD", "glottocode": "lang2", "reason": "fixture", "by": "test"}])
    m = lk[(lk.population == "PopD") & (lk.link_basis == "manual")].iloc[0]
    assert m.glottocode == "lang2" and m.match_status == "candidate"
