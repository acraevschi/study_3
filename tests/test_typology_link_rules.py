"""Link rules of 2026-10-20, the decision sheet and the sensitivity analyses (fixtures only)."""

import json

import pandas as pd
import pytest

from morph_ldl.data import gelato
from morph_ldl.typology import audit_typology, stage_typology
from morph_ldl.typology import grambank as gb
from morph_ldl.typology import link_review as lr
from morph_ldl.typology.stage import (analysis_links, build_links, link_rules, typology_dir,
                                      yaml_conflicts)
from test_typology_helpers import FIX, fixture_cfg, fixture_typology

GB_LANGS = {"lang1", "lang2", "lang3", "iso1"}
RULES = {"accept_by_rule": ["exact", "dialect_rollup", "group_map_down", "graff_gbi_stand_in"],
         "endangered_min_aes": 4}


def _glotto(aes=(("lang3", 5), ("lang1", 3))):
    log = gb.FileLog()
    d = FIX / "glottolog-cldf" / "cldf"
    vals = log.read_csv(d / "values.csv", usecols=["Language_ID", "Parameter_ID", "Value"])
    vals = pd.concat([vals, pd.DataFrame([{"Language_ID": g, "Parameter_ID": "aes", "Value": str(a)} for g, a in aes])])
    return gb.Glottolog.from_frames(log.read_csv(d / "languages.csv"), vals)


# population -> (own code, GBI code, TLI code)
POPS = {"Pexact": ("lang1", "lang2", ""), "Proll": ("dia1", "", ""), "Pmap": ("grp2", "", ""),
        "Pgrp": ("grp1", "lang2", ""), "Pother": ("lang6", "iso1", ""), "Ptli": ("lang6", "", "lang2"),
        "Prelink": ("grp1", "", "")}


def _frames():
    main = pd.DataFrame([{"ID": f"HO-{i}", "Name": p, "samplesize": "6", "Glottocode": c[0]}
                         for i, (p, c) in enumerate(POPS.items())])
    s1 = pd.DataFrame([{"Population": p, "N_Individuals": "6", "GeLaTo_Glottocode": c[0],
                        "GBI_Glottocode": c[1] or "NA", "TLI_Glottocode": c[2] or "NA"} for p, c in POPS.items()])
    return main, s1


def _sheet(*rows):
    return pd.DataFrame([{**{c: "" for c in lr.SHEET_COLUMNS}, **r} for r in rows], columns=lr.SHEET_COLUMNS)


def _links(sheet=None, rules=RULES, glotto=None):
    main, s1 = _frames()
    return build_links(main, s1, glotto or _glotto(), GB_LANGS, [], [], sheet, rules)


def _by(lk):
    return {(r.population, r.glottocode): r for r in lk.itertuples()}


CONFIRM = {"confirmed_by": "PI", "confirmed_date": "2026-10-20"}


def test_rules_accept_without_a_person():
    by = _by(_links())
    for k, rule in ((("Pexact", "lang1"), "exact"), (("Proll", "lang1"), "dialect_rollup"),
                    (("Pmap", "lang3"), "group_map_down")):
        assert (by[k].match_status, by[k].accepted_by_rule, by[k].confirmed_by, by[k].in_main) == \
            ("accepted", rule, "", True)
    # without rules nothing is accepted (pcfp_v2 behaviour)
    assert "accepted" not in set(_links(rules=None)["match_status"])


def test_graff_stand_in_only_without_an_own_grambank_link():
    by = _by(_links())
    # own language in Grambank: the GBI stand-in stays a proxy (ambiguous)
    assert (by[("Pexact", "lang2")].match_status, by[("Pexact", "lang2")].in_main) == ("ambiguous", False)
    g = by[("Pgrp", "lang2")]                       # own code a group: stand-in accepted
    assert (g.match_status, g.accepted_by_rule, g.sensitivity_flags) == \
        ("accepted", "graff_gbi_stand_in", "stand_in_no_own_language")
    o = by[("Pother", "iso1")]                      # own language not in Grambank: accepted, flagged
    assert (o.match_status, o.sensitivity_flags) == ("accepted", "stand_in_other_language")
    assert by[("Ptli", "lang2")].match_status == "ambiguous"   # TLI stand-ins are never accepted by rule


def test_endangered_flag_follows_the_threshold():
    by = _by(_links())
    assert "endangered" in by[("Pmap", "lang3")].sensitivity_flags          # AES 5
    assert "endangered" not in by[("Pexact", "lang1")].sensitivity_flags    # AES 3
    by3 = _by(_links(rules={**RULES, "endangered_min_aes": 3}))
    assert "endangered" in by3[("Pexact", "lang1")].sensitivity_flags
    assert by[("Pexact", "lang1")].aes == 3
    with pytest.raises(ValueError, match="endangered_min_aes"):
        link_rules({"link_rules": {"endangered_min_aes": 7}})
    with pytest.raises(ValueError, match="unknown rules"):
        link_rules({"link_rules": {"accept_by_rule": ["proxy_tli"]}})


def test_sheet_decisions_need_confirmation():
    sh = _sheet({"population": "Pexact", "link_glottocode": "lang1", "decision": "exclude", **CONFIRM},
                {"population": "Proll", "link_glottocode": "lang1", "decision": "ambiguous"},
                {"population": "Ptli", "link_glottocode": "lang2", "decision": "accept", **CONFIRM})
    by = _by(_links(sh))
    e = by[("Pexact", "lang1")]
    assert (e.match_status, e.accepted_by_rule, e.confirmed_by, e.in_main) == ("excluded", "", "PI", False)
    p = by[("Proll", "lang1")]                      # proposal: rule status stays
    assert (p.match_status, p.proposed_status, p.accepted_by_rule) == ("accepted", "ambiguous", "dialect_rollup")
    assert "awaiting human confirmation" in p.unresolved_issues
    a = by[("Ptli", "lang2")]                       # a person may accept a TLI stand-in
    assert (a.match_status, a.confirmed_by, a.in_main, a.sensitivity_flags) == \
        ("accepted", "PI", True, "stand_in_other_language")


def test_relink_replaces_the_link():
    sh = _sheet({"population": "Prelink", "link_glottocode": "", "decision": "relink",
                 "decision_glottocode": "lang3", "reason": "r", **CONFIRM},
                {"population": "Pother", "link_glottocode": "lang6", "decision": "relink",
                 "decision_glottocode": "dia1"})
    lk = _links(sh)
    r = lk[lk["population"] == "Prelink"].set_index("link_basis")
    assert r.loc["group_ambiguous", "match_status"] == "excluded"
    m = r.loc["manual"]
    assert (m.glottocode, m.match_status, m.confirmed_by, m.in_main) == ("lang3", "accepted", "PI", True)
    assert "relink" in m.decision_reason
    o = lk[lk["population"] == "Pother"]            # unconfirmed relink: proposal, no manual link
    assert "manual" not in set(o["link_basis"])
    assert o.set_index("glottocode").loc["lang6", "proposed_status"] == "excluded"


def test_malformed_rows_are_not_applied_and_doubt_needs_no_confirmation():
    sh = _sheet({"population": "Pexact", "link_glottocode": "lang1", "decision": "drop", **CONFIRM},
                {"population": "Pmap", "link_glottocode": "lang3", "decision": "exclude", "confirmed_by": "PI"},
                {"population": "Pgrp", "link_glottocode": "", "decision": "relink", "decision_glottocode": "zzzz9999",
                 **CONFIRM},
                {"population": "Proll", "link_glottocode": "lang1", "doubt": "diaspora;wrong_code"})
    by = _by(_links(sh))
    for k in (("Pexact", "lang1"), ("Pmap", "lang3")):
        assert by[k].match_status == "accepted" and "malformed" in by[k].unresolved_issues
    assert "manual" not in {r.link_basis for r in by.values() if r.population == "Pgrp"}
    assert by[("Proll", "lang1")].sensitivity_flags == "diaspora;wrong_code"
    assert lr.row_problems(_sheet({"population": "x", "link_glottocode": "", "doubt": "Bad Label"}).iloc[0])


def test_read_sheet_refuses_wrong_columns_and_repeated_keys(tmp_path):
    p = tmp_path / "s.csv"
    _sheet({"population": "A", "link_glottocode": "x"}).drop(columns="doubt").to_csv(p, index=False)
    with pytest.raises(ValueError, match="columns"):
        lr.read_sheet(p)
    _sheet({"population": "A", "link_glottocode": "x"}, {"population": "A", "link_glottocode": "x"}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="repeated"):
        lr.read_sheet(p)


def test_yaml_decision_must_be_in_the_sheet():
    rv = {"id": "t", "population": "Pexact", "glottocode": "lang1", "status": "accepted", **CONFIRM}
    assert yaml_conflicts([rv], _sheet())
    assert yaml_conflicts([rv], _sheet({"population": "Pexact", "link_glottocode": "lang1", "decision": "accept"}))
    assert not yaml_conflicts([rv], _sheet({"population": "Pexact", "link_glottocode": "lang1",
                                            "decision": "accept", **CONFIRM}))
    assert not yaml_conflicts([{k: v for k, v in rv.items() if not k.startswith("confirmed")}], _sheet())


def test_sensitivity_analyses_only_drop_links():
    lk = _links(_sheet({"population": "Proll", "link_glottocode": "lang1", "doubt": "diaspora"}))
    oc = pd.DataFrame({"glottocode": sorted(set(lk["glottocode"]) - {""})}).assign(in_grambank=True, meets_coverage_main=True)
    al = analysis_links(lk, oc, {"no_doubt": ["endangered", "diaspora"], "no_other": ["stand_in_other_language"],
                                 "strict": ["endangered", "diaspora", "stand_in_other_language"]})
    sets = {a: set(zip(d["population"], d["glottocode"])) for a, d in al.groupby("analysis")}
    assert sets["main"] == set(zip(*lk[lk["in_main"]][["population", "glottocode"]].values.T))
    assert ("Pmap", "lang3") not in sets["no_doubt"] and ("Proll", "lang1") not in sets["no_doubt"]
    assert ("Pother", "iso1") not in sets["no_other"] and ("Pother", "iso1") in sets["no_doubt"]
    assert sets["strict"] == sets["no_doubt"] & sets["no_other"]
    assert all(s <= sets["main"] for s in sets.values())


def test_link_audit_accepts_rules_but_not_unconfirmed_people():
    lk = _links().astype(str)
    assert gelato.review_link_problems(lk, "t") == []
    bad = lk.copy()
    bad.loc[bad["accepted_by_rule"] != "", "accepted_by_rule"] = ""
    assert any("without confirmed_by" in p for p in gelato.review_link_problems(bad, "t"))


# --------------------------------------------------------------------------
# stage, regeneration
# --------------------------------------------------------------------------

def _cfg(tmp_path, sheet_rows=()):
    t = fixture_typology()
    sheet = tmp_path / "sheet.csv"
    lr.write_sheet(_sheet(*sheet_rows), sheet)
    t.update({"decision_sheet": str(sheet), "link_rules": RULES,
              "sensitivity_analyses": {"no_doubtful_language": ["endangered", "diaspora", "wrong_code"]}})
    return fixture_cfg(tmp_path, t), sheet


def test_stage_with_rules_and_sheet(tmp_path):
    cfg, _ = _cfg(tmp_path, [{"population": "PopB", "link_glottocode": "lang1", "doubt": "diaspora"},
                             {"population": "PopD", "link_glottocode": "", "decision": "relink",
                              "decision_glottocode": "lang2", **CONFIRM}])
    stage_typology(cfg)
    out = typology_dir(cfg)
    lk = pd.read_csv(out / "grambank_population_links.csv", dtype=str, keep_default_na=False)
    by = {(r.population, r.link_basis): r for r in lk.itertuples()}
    assert by[("PopA", "exact")].match_status == "accepted"           # the fixture YAML is not applied
    assert by[("PopE", "proxy_gbi")].accepted_by_rule == "graff_gbi_stand_in"
    assert by[("PopD", "manual")].glottocode == "lang2"
    al = pd.read_csv(out / "analysis_links.csv", dtype=str, keep_default_na=False)
    assert set(al["analysis"]) == {"main", "no_doubtful_language"}
    assert "PopB" not in set(al[al["analysis"] == "no_doubtful_language"]["population"])
    summ = json.loads((out / "coverage_summary.json").read_text())
    assert summ["decision_sheet"]["n_decisions_confirmed"] == 1 and summ["analyses"]["main"]["n_links"] > 0
    oc = pd.read_csv(out / "grambank_inflection.csv", dtype=str, keep_default_na=False).set_index("glottocode")
    assert oc.loc["lang2", "n_populations_main"] == "1"
    checks, problems = audit_typology(cfg)
    assert problems == [], problems
    first = {n: (out / n).read_bytes() for n in ("grambank_population_links.csv", "analysis_links.csv")}
    stage_typology(cfg)
    assert all((out / n).read_bytes() == b for n, b in first.items())   # deterministic


def test_stage_refuses_a_confirmed_yaml_decision_missing_from_the_sheet(tmp_path):
    cfg, _ = _cfg(tmp_path)
    y = tmp_path / "review.yaml"
    y.write_text("reviews:\n  - {id: t, glottocode: lang1, population: PopA, status: excluded, "
                 "confirmed_by: PI, confirmed_date: '2026-10-20'}\n")
    cfg["typology"]["review_file"] = str(y)
    with pytest.raises(ValueError, match="disagree"):
        stage_typology(cfg)


def test_regeneration_never_loses_a_decision(tmp_path):
    cfg, sheet = _cfg(tmp_path, [
        {"population": "PopA", "link_glottocode": "lang1", "decision": "accept", "reason": "kept", **CONFIRM},
        {"population": "PopX", "link_glottocode": "lang9", "decision": "exclude", "reason": "gone"},  # vanished, filled
        {"population": "PopY", "link_glottocode": "lang9"},                                         # vanished, empty
    ])
    stage_typology(cfg)
    lr.regenerate(cfg)
    new = lr.read_sheet(sheet).set_index(["population", "link_glottocode"])
    a = new.loc[("PopA", "lang1")]
    assert (a.decision, a.reason, a.confirmed_by, a.n_individuals) == ("accept", "kept", "PI", "10")  # evidence added
    assert new.loc[("PopX", "lang9"), "why_in_sheet"].startswith(lr.STALE)     # kept, marked stale
    assert ("PopY", "lang9") not in new.index                                  # empty and vanished: dropped
    assert new.loc[("PopD", ""), "why_in_sheet"] == lr.WHY_GROUP               # new case added
    # group candidates are in Grambank even without an outcome row: the sheet gives their counts
    assert "Lang One (lang1): Grambank 3 present of 4 coded, coverage 0.80" in new.loc[("PopD", ""), "options"]
    # a person fills the new row; regenerating again keeps it and changes nothing
    df = lr.read_sheet(sheet)
    df.loc[df["population"] == "PopD", ["decision", "decision_glottocode"]] = ["relink", "lang1"]
    lr.write_sheet(df, sheet)
    before = sheet.read_bytes()
    lr.regenerate(cfg)
    assert sheet.read_bytes() == before


def test_regeneration_refuses_to_drop_or_change_a_filled_row():
    old = _sheet({"population": "PopA", "link_glottocode": "lang1", "decision": "accept"})
    with pytest.raises(RuntimeError, match="drop filled rows"):
        lr.check_no_loss(old, _sheet())
    with pytest.raises(RuntimeError, match="change filled cells"):
        lr.check_no_loss(old, _sheet({"population": "PopA", "link_glottocode": "lang1", "decision": "exclude"}))
    lr.check_no_loss(old, old)
