"""Unit and end-to-end R `PARA_DRY=1` / `ENT_DRY=1` verification tests for Track 2 Stage-2
Bayesian preparation, Berkson regression, equivalence (`BF_01`/ROPE), and token-frequency
entrenchment (`analyses/paradigm/stage2/`).
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import analyses.common.genetics as genetics_mod
import analyses.paradigm.stage2.prepare_stage2 as prepare_stage2_mod
from analyses.paradigm.stage2.prepare_stage2 import (
    DEFAULT_LINKS_CSV,
    DEFAULT_MGL_RESULTS_DIR,
    DEFAULT_NEURAL_RESULTS_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_PARADIGM_DATA_DIR,
    SETS,
    TRACK1_DOMAIN_GATES,
    TRACKED_LINKS_CSV,
    build_smoke_test_inputs,
    classify_source_provenance,
    load_track1_overlap,
    prepare_stage2_inputs,
    resolve_links_for_glottocodes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_source_provenance_classification() -> None:
    assert classify_source_provenance("paralex_leffi", "1_paralex") == "paralex"
    assert classify_source_provenance("celex_eng_v", "1_celex") == "paralex"
    assert classify_source_provenance("national_polimorf", "2_national") == "national"
    assert classify_source_provenance("sigmorphon2020_uralic", "3_sigmorphon") == "unimorph"
    assert classify_source_provenance("unimorph_fin", "3_unimorph") == "unimorph"


def test_link_resolution_nonproxy_precedence_and_standin_filtering() -> None:
    """Verify 1-to-1 population -> glottocode resolution on duplicate proxy populations."""
    # Case A: Both `neap1235` (own language of Italian_Calabria) and `ital1282` (proxy) in target set
    links_both, in_set_both = resolve_links_for_glottocodes(
        DEFAULT_LINKS_CSV, {"ital1282", "neap1235", "turo1239", "mlah1239"}
    )
    pop_map_both = {r["population"]: r["glottocode"] for r in links_both}
    assert len(pop_map_both) == len(links_both)
    assert pop_map_both["Bergamo"] == "ital1282"
    assert pop_map_both["Italian_Calabria"] == "neap1235"
    assert pop_map_both["Syrian"] == "turo1239"
    for s in SETS:
        assert in_set_both[s] <= in_set_both["main"]

    # Case B: Only `ital1282` in target set -> Italian_Calabria uses stand-in in `main`, dropped in `strict`
    links_ita_only, in_set_ita = resolve_links_for_glottocodes(DEFAULT_LINKS_CSV, {"ital1282"})
    pop_map_ita = {r["population"]: r["glottocode"] for r in links_ita_only}
    assert pop_map_ita["Italian_Calabria"] == "ital1282"
    assert "Italian_Calabria" in in_set_ita["main"]
    assert "Italian_Calabria" not in in_set_ita["no_other_language_stand_ins"]
    assert "Italian_Calabria" not in in_set_ita["strict"]
    assert "Bergamo" in in_set_ita["main"]
    assert "Bergamo" not in in_set_ita["strict"]
    assert "Tuscan" in in_set_ita["strict"]


def test_load_track1_overlap_5domain_gating_and_legacy_fallback(tmp_path: Path) -> None:
    """Verify >=60% gating for 5-domain Track-1 proportions and graceful legacy fallback."""
    # 1. New 5-domain schema
    gb_new = tmp_path / "gb_new"
    gb_new.mkdir()
    with open(gb_new / "language_data.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "glottocode", "in_main",
                "v_breadth_present", "v_breadth_coded",
                "n_breadth_present", "n_breadth_coded",
                "v_irreg_present", "v_irreg_coded",
                "n_irreg_present", "n_irreg_coded",
                "np_agr_present", "np_agr_coded",
            ],
        )
        w.writeheader()
        w.writerow({
            "glottocode": "taga1270", "in_main": "1",
            "v_breadth_present": "4", "v_breadth_coded": "6",  # >=6/9 passes
            "n_breadth_present": "1", "n_breadth_coded": "2",  # <3/5 gated to NaN
            "v_irreg_present": "2", "v_irreg_coded": "3",      # >=3/5 passes
            "n_irreg_present": "0", "n_irreg_coded": "1",      # <2/3 gated to NaN
            "np_agr_present": "3", "np_agr_coded": "5",        # >=5/7 passes
        })
        w.writerow({
            "glottocode": "lowc1234", "in_main": "1",
            "v_breadth_present": "2", "v_breadth_coded": "5",
            "n_breadth_present": "1", "n_breadth_coded": "2",
            "v_irreg_present": "0", "v_irreg_coded": "2",
            "n_irreg_present": "0", "n_irreg_coded": "1",
            "np_agr_present": "1", "np_agr_coded": "4",
        })
    ov_new = load_track1_overlap(gb_data_dir=gb_new, gb_sem_dir=tmp_path / "no_sem")
    assert ov_new["taga1270"]["in_track1"] == 1
    assert abs(float(ov_new["taga1270"]["v_breadth_prop"]) - 4.0 / 6.0) < 1e-9
    assert math.isnan(float(ov_new["taga1270"]["n_breadth_prop"]))
    assert abs(float(ov_new["taga1270"]["v_irreg_prop"]) - 2.0 / 3.0) < 1e-9
    assert math.isnan(float(ov_new["taga1270"]["n_irreg_prop"]))
    assert abs(float(ov_new["taga1270"]["np_agr_prop"]) - 3.0 / 5.0) < 1e-9
    assert ov_new["lowc1234"]["in_track1"] == 0

    # 2. Legacy schema fallback
    gb_old = tmp_path / "gb_old"
    gb_old.mkdir()
    with open(gb_old / "language_data.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["glottocode", "in_main", "n_present", "n_coded", "irreg13_present", "irreg13_coded"],
        )
        w.writeheader()
        w.writerow({
            "glottocode": "finn1318", "in_main": "1",
            "n_present": "10", "n_coded": "12",
            "irreg13_present": "4", "irreg13_coded": "13",
        })
    ov_old = load_track1_overlap(gb_data_dir=gb_old, gb_sem_dir=tmp_path / "no_sem")
    assert ov_old["finn1318"]["in_track1"] == 1
    assert abs(float(ov_old["finn1318"]["irreg13_prop"]) - 4.0 / 13.0) < 1e-9
    assert math.isnan(float(ov_old["finn1318"]["v_breadth_prop"]))


def test_prepare_stage2_and_r_dry_validation(tmp_path: Path) -> None:
    """Build Stage-2 V/N tables on real GeLaTo genetics/trees and validate harmonized R Stan models."""
    prep_dir = build_smoke_test_inputs(tmp_path)
    out_dir = tmp_path / "stage2_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    required_pop_cols = {
        "f3_fixed", "f3_fixed_iso", "f3_fixed_raw", "f3_fixed_se", "f3_all_fixed",
        "f4_est", "f4_se", "neigh_flow_out", "median_fst_all",
        "regime_a_enclave", "regime_b_shift", "barbieri_mismatch",
    }

    for pos in ("V", "N"):
        ld_path = prep_dir / f"language_data_{pos}.csv"
        ind_path = prep_dir / f"individuals_{pos}.csv"
        pop_path = prep_dir / f"populations_{pos}.csv"
        nwk_path = prep_dir / f"phylo_tree_{pos}.nwk"
        lf_path = prep_dir / f"lemma_freq_data_{pos}.csv"

        assert ld_path.exists() and ind_path.exists() and pop_path.exists()
        assert nwk_path.exists() and lf_path.exists()

        with open(ld_path, encoding="utf-8") as f:
            ld_reader = csv.DictReader(f)
            ld_cols = set(ld_reader.fieldnames or [])
            ld_rows = list(ld_reader)
        assert len(ld_rows) == 18
        # Strict 1-row-per-glottocode invariant & absence of cross-language mean_token_freq
        gcs = [r["glottocode"] for r in ld_rows]
        assert len(set(gcs)) == len(gcs)
        assert "mean_token_freq" not in ld_cols
        for dom, _, _ in TRACK1_DOMAIN_GATES:
            assert f"{dom}_prop" in ld_cols
        assert {"f3_fixed_maxp", "median_fst_all", "regime_a_enclave", "regime_b_shift"} <= ld_cols

        # Track 1 overlap populated with 5-domain proportions
        n_t1 = sum(int(r["in_track1"]) for r in ld_rows)
        assert n_t1 >= 10
        assert any(
            r["v_breadth_prop"] != "" and math.isfinite(float(r["v_breadth_prop"]))
            for r in ld_rows
        )

        # Population table exposes harmonized f3_fixed & FST regime columns
        with open(pop_path, encoding="utf-8") as f:
            pop_reader = csv.DictReader(f)
            assert required_pop_cols <= set(pop_reader.fieldnames or [])

        # Within-language z_freq_within must have exact mean 0 and unit SD per glottocode
        with open(lf_path, encoding="utf-8") as f:
            lf_rows = list(csv.DictReader(f))
        by_gc: dict[str, list[float]] = defaultdict(list)
        for r in lf_rows:
            by_gc[r["glottocode"]].append(float(r["z_freq_within"]))
        assert len(by_gc) == 18
        for gc, z_vals in by_gc.items():
            arr = np.array(z_vals, dtype=np.float64)
            assert abs(float(np.mean(arr))) < 1e-4, f"{gc} z_freq mean={np.mean(arr)}"
            assert abs(float(np.std(arr, ddof=1)) - 1.0) < 1e-4, f"{gc} z_freq sd={np.std(arr, ddof=1)}"

    # Mock SEM latent EAP table to verify construct correlations ingest theta_*_eap & theta0_*_eap
    sem_eap_mock = tmp_path / "sem_latent_eap.csv"
    with open(prep_dir / "language_data_V.csv", encoding="utf-8") as f:
        mock_gcs = [r["glottocode"] for r in csv.DictReader(f)]
    with open(sem_eap_mock, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["glottocode", "theta_v_breadth_eap", "theta0_v_breadth_eap", "theta_n_breadth_eap"],
        )
        w.writeheader()
        for idx, gc in enumerate(mock_gcs):
            w.writerow({
                "glottocode": gc,
                "theta_v_breadth_eap": f"{(idx - 8.5) * 0.15:.4f}",
                "theta0_v_breadth_eap": f"{(idx - 8.5) * 0.12:.4f}",
                "theta_n_breadth_eap": f"{(idx - 8.5) * 0.10:.4f}",
            })

    # Run `PARA_DRY=1 Rscript analyses/paradigm/stage2/fit_stage2.R` with harmonized defaults (`f3_fixed`)
    env = {
        **os.environ,
        "PARA_DRY": "1",
        "PARA_POS": "V,N",
        "PARA_SETS": "main",
        "PARA_MEASURES": "f3_fixed",
        "PARA_DATA_DIR": str(prep_dir),
        "PARA_OUT_DIR": str(out_dir),
        "PARA_SEM_EAP_CSV": str(sem_eap_mock),
    }
    res_s2 = subprocess.run(
        ["Rscript", str(REPO_ROOT / "analyses/paradigm/stage2/fit_stage2.R"), "--cores=2"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_s2.returncode == 0, f"fit_stage2.R failed:\nSTDOUT:\n{res_s2.stdout}\nSTDERR:\n{res_s2.stderr}"

    block1_groups = {
        "maximal", "loo_no_space", "loo_no_cdist", "loo_no_demog",
        "loo_no_lexprov", "mundlak_wb", "null_baseline",
    }
    block2_expected = {
        "full_sample_PS", "cutoff_n5", "strict_links",
        "bouckaert_stem", "bouckaert_crown", "no_inner_asia",
        "tier2_uniform480", "topfreq120", "pcfp_k1", "pcfp_k3",
        "reliability", "neural_monotonic",
        "f3_all_fixed", "f3_fixed_iso", "f3_fixed_maxp", "f3_fixed_raw",
        "f4_est", "neigh_flow_out",
        "drop_regime_a", "drop_regime_b", "plus_regime_b",
        "me_baseline", "berkson_20draw",
    }

    for pos in ("V", "N"):
        dry_csv = out_dir / f"stage2_results_{pos}_dry.csv"
        assert dry_csv.exists()
        with open(dry_csv, encoding="utf-8") as f:
            s2_rows = list(csv.DictReader(f))
        spec_groups = {r["spec_group"] for r in s2_rows}
        assert block1_groups <= spec_groups, f"Missing Block-1 specs: {block1_groups - spec_groups}"
        assert block2_expected <= spec_groups, f"Missing Block-2 specs: {block2_expected - spec_groups}"

        # Block-1 locked complete-case N invariant: all 7 Block-1 specs must share identical n_lang
        b1_rows = [r for r in s2_rows if r["block"] == "block1"]
        assert len(b1_rows) == 7
        assert len({int(r["n_lang"]) for r in b1_rows}) == 1, f"Block-1 n_lang mismatch: {b1_rows}"

        # Verify mean_token_freq is never used in any Stage-2 regression formula
        assert all("mean_token_freq" not in r["formula"] for r in s2_rows)

        # Verify f3_fixed_raw and neigh_flow_out include z_median_fst_all
        by_grp = {r["spec_group"]: r for r in s2_rows}
        assert "z_median_fst_all" in by_grp["f3_fixed_raw"]["formula"]
        assert "z_median_fst_all" in by_grp["neigh_flow_out"]["formula"]
        assert "me(x, x_sd)" in by_grp["me_baseline"]["formula"]
        assert "gp(sx, sy, sz)" in by_grp["maximal"]["formula"]
        assert "gp(sx, sy, sz)" not in by_grp["loo_no_space"]["formula"]
        assert "z_log_C_distinct" not in by_grp["loo_no_cdist"]["formula"]

    corr_csv = out_dir / "cross_track_correlations_dry.csv"
    assert corr_csv.exists()
    assert not (out_dir / "cross_track_correlations.csv").exists()
    with open(corr_csv, encoding="utf-8") as f:
        corr_rows = list(csv.DictReader(f))
    comp_names = {r["comparison"] for r in corr_rows}
    assert "H_tree_V_vs_Track1_v_breadth_prop" in comp_names
    assert "log_C_distinct_N_vs_Track1_n_breadth_prop" in comp_names
    assert "H_tree_N_vs_Track1_n_irreg_prop" in comp_names
    assert "H_tree_V_vs_Track1_theta_v_breadth_eap" in comp_names
    assert "log_C_distinct_V_vs_Track1_theta0_v_breadth_eap" in comp_names
    assert "H_tree_V_vs_H_tree_N" in comp_names

    # Verify spatial GP lscale prior inv_gamma(3, 0.4) is literally emitted in Stan code (0 warnings)
    res_lscale = subprocess.run(
        [
            "Rscript", "-e",
            "suppressPackageStartupMessages(library(brms)); "
            "source_lines <- readLines('analyses/paradigm/stage2/fit_stage2.R'); "
            "i1 <- grep('^make_stage2_prior <-', source_lines)[1]; "
            "i2 <- grep('^fit_stage2_brm <-', source_lines)[1] - 1; "
            "eval(parse(text = paste(source_lines[i1:i2], collapse = '\\n'))); "
            "pr <- make_stage2_prior(TRUE, TRUE, TRUE); "
            "stopifnot(any(pr$class == 'lscale' & pr$coef == 'gpsxsysz' & pr$prior == 'inv_gamma(3, 0.4)')); "
            "df <- data.frame(y=rnorm(10), se_y=rep(0.1,10), x=rnorm(10), sx=rnorm(10), sy=rnorm(10), sz=rnorm(10), lang=paste0('l',1:10)); "
            "sc <- make_stancode(bf(y | se(se_y, sigma = TRUE) ~ 1 + x + gp(sx, sy, sz)), data=df, family=gaussian(), prior=make_stage2_prior(TRUE, FALSE, TRUE)); "
            "stopifnot(grepl('inv_gamma_lpdf(lscale_1[1][1] | 3, 0.4)', sc, fixed = TRUE)); "
            "cat('OK_LSCALE\\n')",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_lscale.returncode == 0 and "OK_LSCALE" in res_lscale.stdout, res_lscale.stderr

    # Run `ENT_DRY=1 Rscript analyses/paradigm/stage2/fit_entrenchment.R`
    env_ent = {
        **os.environ,
        "ENT_DRY": "1",
        "ENT_CORES": "2",
        "ENT_POS": "V,N",
        "ENT_MEASURES": "f3_fixed",
        "ENT_CUTOFFS": "3,5",
        "ENT_DATA_DIR": str(prep_dir),
        "ENT_OUT_DIR": str(out_dir),
    }
    res_ent = subprocess.run(
        ["Rscript", str(REPO_ROOT / "analyses/paradigm/stage2/fit_entrenchment.R")],
        cwd=REPO_ROOT,
        env=env_ent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_ent.returncode == 0, f"fit_entrenchment.R failed:\nSTDOUT:\n{res_ent.stdout}\nSTDERR:\n{res_ent.stderr}"
    for pos in ("V", "N"):
        ent_csv = out_dir / f"entrenchment_results_{pos}_dry.csv"
        assert ent_csv.exists()
        with open(ent_csv, encoding="utf-8") as f:
            ent_rows = list(csv.DictReader(f))
        assert {r["sample"] for r in ent_rows} == {"all_lemmas", "attested_only"}
        assert {int(r["min_n_ind"]) for r in ent_rows} == {3, 5}
        assert {r["measure"] for r in ent_rows} == {"f3_fixed"}

    # Verify `unit_outcomes_uniform120_malouf.csv` ingestion + singleton `source_prov` collapse on subset
    mgl_dir = tmp_path / "mgl_results"
    with open(mgl_dir / "unit_outcomes_uniform120.csv", encoding="utf-8") as f:
        u120_orig = list(csv.DictReader(f))
    mal_rows = [
        {
            **r,
            "model": "malouf",
            "h_tree_bits": "0.4123" if r["glottocode"] == "finn1318" else str(round(float(r["h_tree_bits"]) * 0.95, 5)),
        }
        for r in u120_orig
        if r["glottocode"] in ("finn1318", "nucl1301", "ital1282", "stan1290", "russ1263", "poli1260", "ukra1253")
    ]
    with open(mgl_dir / "unit_outcomes_uniform120_malouf.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mal_rows[0].keys()))
        w.writeheader()
        w.writerows(mal_rows)

    prep_dir_2 = tmp_path / "prepared_edge"
    prepare_stage2_inputs(
        paradigm_data_dir=tmp_path / "paradigm_data",
        mgl_results_dir=mgl_dir,
        neural_results_dir=tmp_path / "neu_results",
        out_dir=prep_dir_2,
        pos_list=("V",),
    )
    with open(prep_dir_2 / "language_data_V.csv", encoding="utf-8") as f:
        ld_v2 = list(csv.DictReader(f))
    by_gc_v2 = {r["glottocode"]: r for r in ld_v2}
    assert by_gc_v2["finn1318"]["has_malouf"] == "1"
    assert abs(float(by_gc_v2["finn1318"]["h_tree_malouf_bits"]) - 0.4123) < 1e-4

    # Make `malt1254` the ONLY `"national"` source_prov row (and drop its f3_fixed via n_individuals=1)
    for r in ld_v2:
        r["source_prov"] = "national" if r["glottocode"] == "malt1254" else "unimorph"
    with open(prep_dir_2 / "language_data_V.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ld_v2[0].keys()))
        w.writeheader()
        w.writerows(ld_v2)

    with open(prep_dir_2 / "populations_V.csv", encoding="utf-8") as f:
        pops_v2 = list(csv.DictReader(f))
    for pr in pops_v2:
        if pr["glottocode"] == "malt1254":
            pr["n_individuals"] = "1"
    with open(prep_dir_2 / "populations_V.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pops_v2[0].keys()))
        w.writeheader()
        w.writerows(pops_v2)

    out_dir_2 = tmp_path / "stage2_out_edge"
    out_dir_2.mkdir()
    env_edge = {
        **env,
        "PARA_POS": "V",
        "PARA_SETS": "main",
        "PARA_MEASURES": "f3_fixed",
        "PARA_DATA_DIR": str(prep_dir_2),
        "PARA_OUT_DIR": str(out_dir_2),
    }
    res_edge = subprocess.run(
        ["Rscript", str(REPO_ROOT / "analyses/paradigm/stage2/fit_stage2.R")],
        cwd=REPO_ROOT,
        env=env_edge,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_edge.returncode == 0, (
        f"fit_stage2.R failed on singleton source_prov subset:\nSTDOUT:\n{res_edge.stdout}\nSTDERR:\n{res_edge.stderr}"
    )
    with open(out_dir_2 / "stage2_results_V_dry.csv", encoding="utf-8") as f:
        edge_rows = list(csv.DictReader(f))
    edge_by_grp = {r["spec_group"]: r for r in edge_rows}
    assert "malouf_harmony" in edge_by_grp
    assert "source_prov" not in edge_by_grp["maximal"]["formula"]
    assert int(edge_by_grp["maximal"]["n_lang"]) == 17


def test_prepare_stage2_cold_start_fallback_matches_primary_and_updates_neural(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cold-start `prepare_stage2_inputs` without raw ADMIXTURE matches primary outputs
    identically and updates Neural outcome columns when new Neural runs arrive.
    """
    assert TRACKED_LINKS_CSV.is_file()
    assert DEFAULT_LINKS_CSV == TRACKED_LINKS_CSV

    if (genetics_mod.ADM_DIR / "GeneticInfoID.csv").is_file() and genetics_mod.GELATO_DATA_CSV.is_file():
        primary_dir = tmp_path / "primary_prepared"
        prepare_stage2_inputs(
            paradigm_data_dir=DEFAULT_PARADIGM_DATA_DIR,
            mgl_results_dir=DEFAULT_MGL_RESULTS_DIR,
            neural_results_dir=DEFAULT_NEURAL_RESULTS_DIR,
            links_csv=TRACKED_LINKS_CSV,
            out_dir=primary_dir,
        )
    else:
        primary_dir = DEFAULT_OUT_DIR

    missing_root = tmp_path / "missing_gelato_sources"
    missing_adm = missing_root / "ADMIXTURE"
    missing_gelato_csv = missing_root / "data.csv"
    missing_arch = missing_root / "arch"
    for mod in (genetics_mod, prepare_stage2_mod):
        monkeypatch.setattr(mod, "ADM_DIR", missing_adm)
        monkeypatch.setattr(mod, "GELATO_DATA_CSV", missing_gelato_csv)
        monkeypatch.setattr(mod, "ARCH_DIR", missing_arch)

    # 1. Cold-start into a fresh directory using tracked `analysis_links.csv` + tracked genetics
    cold_dir = tmp_path / "cold_prepared"
    prepare_stage2_inputs(
        paradigm_data_dir=DEFAULT_PARADIGM_DATA_DIR,
        mgl_results_dir=DEFAULT_MGL_RESULTS_DIR,
        neural_results_dir=DEFAULT_NEURAL_RESULTS_DIR,
        links_csv=TRACKED_LINKS_CSV,
        out_dir=cold_dir,
        adm_dir=missing_adm,
        fallback_prepared_dir=primary_dir,
    )

    for pos in ("V", "N"):
        for stem in ("language_data", "individuals", "populations", "phylo_tree"):
            ext = "nwk" if stem == "phylo_tree" else "csv"
            p_file = primary_dir / f"{stem}_{pos}.{ext}"
            c_file = cold_dir / f"{stem}_{pos}.{ext}"
            assert c_file.is_file()
            assert c_file.read_text(encoding="utf-8") == p_file.read_text(encoding="utf-8"), (
                f"Cold-start {c_file.name} differed from primary output"
            )

    # 2. Simulate full Neural re-run arriving on a fresh clone (only `individuals_*` & `populations_*` pre-existing)
    inplace_dir = tmp_path / "inplace_prepared"
    inplace_dir.mkdir(parents=True)
    for pos in ("V", "N"):
        shutil.copy2(primary_dir / f"individuals_{pos}.csv", inplace_dir / f"individuals_{pos}.csv")
        shutil.copy2(primary_dir / f"populations_{pos}.csv", inplace_dir / f"populations_{pos}.csv")

    neu_sim_dir = tmp_path / "neural_updated"
    neu_sim_dir.mkdir(parents=True)
    with open(DEFAULT_NEURAL_RESULTS_DIR / "unit_outcomes_uniform120.csv", encoding="utf-8") as f:
        neu_rows = list(csv.DictReader(f))
        neu_cols = list(f.seek(0) or csv.reader(f).__next__())
    existing_uids = {r["unit_id"] for r in neu_rows}
    template = dict(neu_rows[0])

    for new_uid, new_gc, new_pos, new_h, new_se in (
        ("amha1245_V_orth", "amha1245", "V", "1.4321", "0.0654"),
        ("bela1254_N_orth", "bela1254", "N", "0.8765", "0.0432"),
    ):
        assert new_uid not in existing_uids
        neu_rows.append({
            **template,
            "unit_id": new_uid,
            "glottocode": new_gc,
            "pos": new_pos,
            "representation": "orth",
            "h_tree_bits": new_h,
            "h_tree_se": new_se,
        })
    with open(neu_sim_dir / "unit_outcomes_uniform120.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=neu_cols, lineterminator="\n")
        w.writeheader()
        w.writerows(neu_rows)

    neu_480_rows = [
        {
            **template,
            "unit_id": "cusc1236_V_orth",
            "glottocode": "cusc1236",
            "pos": "V",
            "representation": "orth",
            "regime": "uniform480",
            "h_tree_bits": "0.1234",
            "h_tree_se": "0.0198",
        }
    ]
    with open(neu_sim_dir / "unit_outcomes_uniform480.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=neu_cols, lineterminator="\n")
        w.writeheader()
        w.writerows(neu_480_rows)

    summ_upd = prepare_stage2_inputs(
        paradigm_data_dir=DEFAULT_PARADIGM_DATA_DIR,
        mgl_results_dir=DEFAULT_MGL_RESULTS_DIR,
        neural_results_dir=neu_sim_dir,
        links_csv=TRACKED_LINKS_CSV,
        out_dir=inplace_dir,
        adm_dir=missing_adm,
        fallback_prepared_dir=None,
    )

    for pos in ("V", "N"):
        assert (inplace_dir / f"individuals_{pos}.csv").read_text(encoding="utf-8") == (
            primary_dir / f"individuals_{pos}.csv"
        ).read_text(encoding="utf-8")
        assert (inplace_dir / f"populations_{pos}.csv").read_text(encoding="utf-8") == (
            primary_dir / f"populations_{pos}.csv"
        ).read_text(encoding="utf-8")

    with open(inplace_dir / "language_data_V.csv", encoding="utf-8") as f:
        ld_v = {r["glottocode"]: r for r in csv.DictReader(f)}
    with open(inplace_dir / "language_data_N.csv", encoding="utf-8") as f:
        ld_n = {r["glottocode"]: r for r in csv.DictReader(f)}

    assert ld_v["amha1245"]["has_neural"] == "1"
    assert abs(float(ld_v["amha1245"]["h_tree_neural_bits"]) - 1.4321) < 1e-6
    assert abs(float(ld_v["amha1245"]["h_tree_neural_se"]) - 0.0654) < 1e-6
    assert abs(float(ld_v["cusc1236"]["h_tree_neural_480_bits"]) - 0.1234) < 1e-6
    assert abs(float(ld_v["cusc1236"]["h_tree_neural_480_se"]) - 0.0198) < 1e-6

    assert ld_n["bela1254"]["has_neural"] == "1"
    assert abs(float(ld_n["bela1254"]["h_tree_neural_bits"]) - 0.8765) < 1e-6
    assert abs(float(ld_n["bela1254"]["h_tree_neural_se"]) - 0.0432) < 1e-6
    assert summ_upd["pos"]["V"]["n_neural"] == 10  # type: ignore[index]
    assert summ_upd["pos"]["N"]["n_neural"] == 4  # type: ignore[index]
