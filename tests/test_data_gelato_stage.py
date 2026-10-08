"""GeLaTo crosswalk status rules, linkage builder, and the data stage's write guard."""

import copy
from pathlib import Path

import pandas as pd
import pytest

from morph_ldl.config import PIPELINE_ROOT, config_hash
from morph_ldl.data import gelato
from morph_ldl.data.stage import load_forms, run_data_stage
from morph_ldl.data.util import RawWriteError, guard_path, sha256_file

FIX = Path(__file__).parent / "fixtures" / "data"
AUDIT = PIPELINE_ROOT / "analyses" / "gelato_feasibility_2026_10_01"

pytestmark = pytest.mark.skipif(not AUDIT.exists(), reason="cached GeLaTo audit sources absent")


def _res(rid, glotto, unit, extinct=False):
    return {"unit_id": unit, "resource_id": rid, "resource_version": "sha256:0", "language": "x",
            "variety_id": "x", "pos": "V", "representation": "orth", "original_id": "x",
            "iso639_3": "x", "glottocode": glotto, "extinct": extinct}


CFG = {"paths": {"gelato_audit": str(AUDIT)}}
RES = pd.DataFrame([
    _res("mgn_data:ita-v", "ital1282", "ita.V.orth.mgn"),
    _res("mgn_data:fin-v", "finn1318", "fin.V.orth.mgn"),
    _res("mgn_custom:navajo-v", "nava1243", "nav.V.phon_custom.mgn"),
    _res("mgn_custom:latin-v", "lati1261", "lat.V.phon_custom.mgn", extinct=True),
])


def test_no_automatic_acceptance():
    xw = gelato.build_crosswalk(RES, CFG, reviews=[])
    assert "accepted" not in set(xw["match_status"])
    ita = xw[xw.resource_id == "mgn_data:ita-v"].set_index("population")
    assert ita.loc["Tuscan", "match_status"] == "candidate"        # exact Glottocode
    assert ita.loc["Bergamo", "match_status"] == "candidate"
    assert ita.loc["Sicilian_East", "match_status"] == "ambiguous"  # TLI proxy only
    assert xw.loc[xw.resource_id == "mgn_custom:navajo-v", "match_status"].tolist() == ["unmatched"]
    assert set(xw.loc[xw.resource_id == "mgn_custom:latin-v", "match_status"]) == {"excluded"}
    # one row per population, never aggregated
    assert ita.index.is_unique and len(ita) >= 2


def test_review_rules():
    full = {"id": "t", "glottocode": "finn1318", "population": "Finnish", "status": "accepted",
            **{f: "x" for f in gelato.REQUIRED_REVIEW_FIELDS}}
    xw = gelato.build_crosswalk(RES, CFG, reviews=[full])
    row = xw.loc[xw.population == "Finnish"].iloc[0]
    # An agent review only proposes acceptance; a named human must confirm it.
    assert row["match_status"] == "candidate" and row["proposed_status"] == "accepted"
    confirmed = {**full, "confirmed_by": "PI name", "confirmed_date": "2026-10-07"}
    xw = gelato.build_crosswalk(RES, CFG, reviews=[confirmed])
    assert xw.loc[xw.population == "Finnish", "match_status"].item() == "accepted"
    partial = {k: v for k, v in full.items() if k != "locality_review"}
    xw = gelato.build_crosswalk(RES, CFG, reviews=[partial])
    row = xw.loc[xw.population == "Finnish"].iloc[0]
    assert row["match_status"] == "candidate" and "review entry incomplete" in row["unresolved_issues"]
    # every accepted entry in the curated review file is complete
    for r in gelato.load_review():
        assert not gelato.review_problems(r), r.get("id")


def test_crosswalk_carries_identifiers_not_ancestry_values():
    xw = gelato.build_crosswalk(RES, CFG, reviews=[])
    assert not [c for c in xw.columns if c.startswith("q") and c[1:].isdigit()]
    for banned in ("largest_share", "ancestry_heterogeneity", "two_component_screen_only"):
        assert banned not in xw.columns
    fin = xw[xw.population == "Finnish"].iloc[0]
    assert fin["ancestry_K_available"] == "12-30" and fin["geneticinfo_n_rows"] == 8
    assert fin["derived_K23_row_key"] == "population=Finnish"
    links = gelato.links_from_crosswalk(xw, ["ita.V.orth.mgn", "fin.V.orth.mgn"])
    assert {"unit_id", "population", "match_status"} <= set(links.columns)
    assert not {"form", "accuracy", "cell_norm"} & set(links.columns)
    with pytest.raises(KeyError):
        gelato.links_from_crosswalk(xw, ["nope.V.orth.mgn"])


def _fixture_cfg(tmp_path):
    cfg = {
        "experiment": {"id": "fixture_exp", "master_seed": 1},
        "paths": {"repo_root": str(PIPELINE_ROOT), "mgn_data": str(FIX / "mgn/data"),
                  "mgn_custom": str(FIX / "mgn/data-custom"), "gelato_audit": str(AUDIT),
                  "outputs": str(tmp_path / "outputs"), "external": str(tmp_path / "ext")},
        "resources": {"ingest": [
            {"adapter": "mgn_wide", "file": "toy-v.csv", "iso": "ita", "pos": "V", "representation": "orth"},
            {"adapter": "mgn_long", "file": "toyish-v.csv", "iso": "ara", "pos": "V", "representation": "phon_custom"},
        ]},
        "task": {"name": "source_known_completion", "source_slot": "NFIN", "panel_slots": ["PRS.1SG", "PRS.3SG"]},
        "units": [{"unit_id": "ita.V.orth.mgn", "resource_id": "mgn_data:toy-v", "role": "smoke_only",
                   "cells": {"NFIN": "NFIN", "PRS.1SG": "1;IND;PRS;SG", "PRS.3SG": "3;IND;PRS;SG"}}],
        "cv": {"inventory_size": 2},
    }
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def test_stage_on_fixtures_never_writes_raw(tmp_path):
    raw_files = sorted(p for p in (FIX / "mgn").rglob("*") if p.is_file())
    before = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in raw_files}
    cfg = _fixture_cfg(tmp_path)
    res = run_data_stage(cfg, with_registry=False)
    after = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in raw_files}
    assert before == after
    assert sorted(p for p in (FIX / "mgn").rglob("*") if p.is_file()) == raw_files
    out = Path(res["output_dir"])
    assert out == (tmp_path / "outputs" / "fixture_exp").resolve()
    f = load_forms(cfg, "ita.V.orth.mgn")
    assert f["is_missing"].dtype == bool and f["variant_idx"].dtype.kind == "i"
    # amare / amere share the NFIN form 'amare' -> one leakage group
    g = f.drop_duplicates("lemma_id").set_index("lemma_label")["group_id"]
    assert g["amare"] == g["amere"]
    s = pd.read_csv(out / "eligibility" / "summary.csv").iloc[0]
    # eligible: lavare, lavarsi, dire#1, amare (amere lacks nothing either) -> 5 lemmas, dire#2 missing PRS.1SG
    assert s["n_eligible_lemmas"] == 5
    assert (out / "data" / "stage_manifest.json").exists()
    with pytest.raises(RawWriteError):
        guard_path(FIX / "mgn" / "data" / "x.csv", cfg)


def test_mixed_or_relabelled_representation_rejected(tmp_path):
    cfg = _fixture_cfg(tmp_path)
    cfg = copy.deepcopy(cfg)
    cfg["resources"]["ingest"][0]["representation"] = "ipa_epitran"
    with pytest.raises(ValueError, match="representation"):
        run_data_stage(cfg, with_registry=False)
