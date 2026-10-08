"""End-to-end typology stage on fixtures: outputs, determinism, firewall, manifest, audit."""

import builtins
import copy
import io
import json
import os
import pathlib

import pandas as pd
import pytest

from morph_ldl.typology import audit_typology, stage_typology
from morph_ldl.typology import grambank as gb
from morph_ldl.typology.stage import typology_dir
from test_typology_helpers import FIX, fixture_cfg, fixture_typology

CSVS = ("grambank_inflection.csv", "grambank_population_links.csv", "missing_from_grambank.csv", "ldl_overlap.csv")


def _run(tmp_path, **kw):
    cfg = fixture_cfg(tmp_path, **kw)
    stage_typology(cfg)
    return cfg, typology_dir(cfg)


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_outcome_table(tmp_path):
    units = [{"unit_id": "lon.V.orth.mgn", "resource_id": "mgn_data:lon-v"},
             {"unit_id": "lsi.V.orth.mgn", "resource_id": "mgn_data:lsi-v"}]
    cfg = fixture_cfg(tmp_path, units=units)
    forms = typology_dir(cfg).parent / "data" / "forms"
    forms.mkdir(parents=True)
    pd.DataFrame({"unit_id": ["lon.V.orth.mgn"], "glottocode": ["dia1"]}).to_csv(forms / "lon.V.orth.mgn.csv", index=False)
    stage_typology(cfg)
    out = typology_dir(cfg)
    oc = _read(out / "grambank_inflection.csv").set_index("glottocode")
    assert sorted(oc.index) == ["iso1", "lang1", "lang2", "lang3", "lang6"]
    l1 = oc.loc["lang1"]
    assert (l1.n_features, l1.n_coded, l1.n_present, l1.coverage, l1.share) == ("5", "4", "3", "0.8", "0.75")
    assert l1.link_bases == "dialect_rollup;exact" and l1.link_basis_proxy_only == "False"
    assert l1.best_link_status == "candidate" and l1.has_accepted_link == "False"   # agent review only proposes
    assert l1.n_populations == "2" and l1.n_individuals_total == "13"
    assert l1.has_ldl_unit == "True" and l1.ldl_units == "lon.V.orth.mgn"           # dia1 rolled up
    assert l1.meets_coverage_main == "True" and l1.meets_coverage_75 == "True"
    assert (l1.verbal_n_features, l1.verbal_n_present, l1.no_agreement_n_features) == ("2", "2", "4")
    l2 = oc.loc["lang2"]
    assert l2.link_basis_proxy_only == "True" and l2.no_inflection == "True" and l2.meets_coverage_main == "False"
    assert l2.meets_coverage_50 == "False"
    assert oc.loc["lang3", "no_inflection"] == "True" and oc.loc["lang3", "link_bases"] == "group_map_down"
    assert oc.loc["iso1", "minimal_inflection"] == "True" and oc.loc["iso1", "no_inflection"] == "False"
    assert oc.loc["iso1", "family"] == "Isolate One" and oc.loc["iso1", "is_isolate"] == "True"
    l6 = oc.loc["lang6"]
    assert l6.in_grambank == "False" and l6.n_present == "" and l6.no_inflection == "" and l6.n_features == "5"
    # lsi unit has no data stage output and no ingest entry -> reported, not fatal
    man = json.loads((out / "stage_manifest.json").read_text())
    assert man["ldl_unit_glottocodes"]["lon.V.orth.mgn"]["source"] == "data_stage_forms"
    assert man["ldl_unit_glottocodes"]["lsi.V.orth.mgn"]["glottocode"] == ""
    # clitic flags (fixture heuristic: fam1 with n_present >= 3; profile on lang? none)
    assert oc.loc["lang1", "clitic_flag_family"] == "True"
    assert oc.loc["iso1", "clitic_flag_profile"] == "True"   # n_present 1, no number/agreement present
    miss = _read(out / "missing_from_grambank.csv")
    assert miss["glottocode"].tolist() == ["lang6"] and miss.loc[0, "grambank_dialect_entries"] == "dia6"
    assert miss.loc[0, "n_individuals_total"] == "20"
    summ = json.loads((out / "coverage_summary.json").read_text())
    assert summ["n_languages_gelato_linked"] == 5 and summ["n_languages_proxy_only"] == 2
    assert summ["n_nonproxy_in_grambank"] == 2
    assert summ["n_nonproxy_by_coverage"]["meets_coverage_main"] == 2
    assert sorted(summ["unlinked_populations"]) == ["PopD", "PopG"]
    checks, problems = audit_typology(cfg)
    assert problems == [] and checks


def test_deterministic_outputs(tmp_path):
    _, a = _run(tmp_path / "a")
    _, b = _run(tmp_path / "b")
    for name in CSVS:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    # feature_set.json differs only by the config hash (the output path is part of the config)
    ja, jb = (json.loads((d / "feature_set.json").read_text()) for d in (a, b))
    ja.pop("config_hash"); jb.pop("config_hash")
    assert ja == jb


def test_no_ancestry_file_is_opened(tmp_path, monkeypatch):
    # The fixture GeLaTo directory also holds Q matrices, GeneticInfoID and derived files.
    gdir = FIX / "gelato"
    assert list(gdir.rglob("*.Q")) and (gdir / "GeneticInfoID.csv").exists()
    opened = []

    def guard(p):
        s = os.fspath(p) if isinstance(p, (str, os.PathLike)) else ""
        if s:
            opened.append(s)
        if s and gb.is_forbidden(pathlib.Path(s).resolve()):
            raise AssertionError(f"ancestry file opened: {s}")

    real_open, real_read_csv, real_path_open = builtins.open, pd.read_csv, pathlib.Path.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (guard(f), real_open(f, *a, **k))[1])
    monkeypatch.setattr(io, "open", builtins.open)
    monkeypatch.setattr(pd, "read_csv", lambda f, *a, **k: (guard(f), real_read_csv(f, *a, **k))[1])
    monkeypatch.setattr(pathlib.Path, "open", lambda self, *a, **k: (guard(self), real_path_open(self, *a, **k))[1])
    cfg, out = _run(tmp_path)
    assert any(o.endswith("tableS1.csv") for o in opened)
    man = json.loads((out / "stage_manifest.json").read_text())
    assert man["files_opened"] and not [x for x in man["files_opened"] if gb.is_forbidden(x["path"])]
    assert all(gb.is_forbidden(p) for p in [gdir / "GeneticInfoID.csv", gdir / "fixture_K23.Q",
                                            gdir / "best_runs" / "fixture_K12.Q",
                                            gdir / "population_ancestry_K23_components.csv",
                                            gdir / "population_ancestry_K12_K30_diagnostics.csv",
                                            gdir / "population_crosswalk.csv"])
    with pytest.raises(gb.AncestryFileError):
        gb.FileLog().read_csv(gdir / "GeneticInfoID.csv")


def test_ancestry_source_path_is_refused(tmp_path):
    t = fixture_typology()
    t["sources"]["gelato_tableS1"]["path"] = str(FIX / "gelato" / "GeneticInfoID.csv")
    with pytest.raises(gb.AncestryFileError):
        stage_typology(fixture_cfg(tmp_path, typology=t))


def test_manifest_records_sources_and_pins(tmp_path):
    cfg, out = _run(tmp_path)
    man = json.loads((out / "stage_manifest.json").read_text())
    assert man["stage"] == "typology" and man["status"] == "ok"
    src = man["typology_sources"]
    assert src["grambank"]["tag"] == "v1.0.3" and src["glottolog"]["tag"] == "v5.3"
    assert src["grambank"]["declared_commit"] == "7ae000cf740f93cdb3e4ec67010668d6795337a9"
    assert {pathlib.Path(f["path"]).name for f in src["grambank"]["files"]} == {
        "values.csv", "parameters.csv", "codes.csv", "languages.csv"}
    assert {pathlib.Path(f["path"]).name for f in src["glottolog"]["files"]} == {"languages.csv", "values.csv"}
    assert all(len(f["sha256"]) == 64 for s in src.values() for f in s["files"])
    assert src["gelato_main_populations"]["files"] and src["gelato_tableS1"]["files"]
    assert man["feature_set_id"] == "fixture_5" and set(man["outputs"]) >= set(CSVS)
    fsj = json.loads((out / "feature_set.json").read_text())
    assert fsj["used"]["features"] == ["GB080", "GB082", "GB044", "GB070", "GB170"]
    assert fsj["declared"]["feature_set"] == cfg["typology"]["feature_set"]


def test_refuses_without_valid_feature_set(tmp_path):
    cfg = fixture_cfg(tmp_path)
    del cfg["typology"]
    with pytest.raises(KeyError, match="typology"):
        stage_typology(cfg)
    t = fixture_typology()
    t["feature_set"]["n_features"] = 35
    cfg = fixture_cfg(tmp_path, typology=t)
    with pytest.raises(gb.FeatureSetError):
        stage_typology(cfg)
    assert not typology_dir(cfg).exists() or not any(typology_dir(cfg).iterdir())


def test_audit_detects_changed_feature_set_and_forbidden_file(tmp_path):
    cfg, out = _run(tmp_path)
    assert audit_typology(cfg)[1] == []
    cfg2 = copy.deepcopy(cfg)
    cfg2["typology"]["feature_set"]["domains"]["case"] = ["GB071"]
    _, problems = audit_typology(cfg2)
    assert any("differs" in p for p in problems)
    man = json.loads((out / "stage_manifest.json").read_text())
    man["files_opened"].append({"path": str(FIX / "gelato" / "GeneticInfoID.csv"), "sha256": "x"})
    (out / "stage_manifest.json").write_text(json.dumps(man))
    _, problems = audit_typology(cfg)
    assert any("ancestry" in p for p in problems)


def test_audit_detects_duplicate_or_non_language_rows(tmp_path):
    cfg, out = _run(tmp_path)
    oc = _read(out / "grambank_inflection.csv")
    bad = pd.concat([oc, oc.iloc[[0]]])
    bad.loc[bad.index[-1], "glottocode"] = "dia1"
    bad.to_csv(out / "grambank_inflection.csv", index=False)
    _, problems = audit_typology(cfg)
    assert any("not language-level" in p for p in problems)
    pd.concat([oc, oc.iloc[[0]]]).to_csv(out / "grambank_inflection.csv", index=False)
    _, problems = audit_typology(cfg)
    assert any("duplicated" in p for p in problems)
