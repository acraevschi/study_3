"""Unit and end-to-end R `PARA_DRY=1` verification tests for Track 2 Stage-2 Bayesian
preparation, Berkson regression, equivalence (`BF_01`/ROPE), and token-frequency entrenchment
(`analyses/paradigm/stage2/`).
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
    TRACKED_LINKS_CSV,
    build_smoke_test_inputs,
    classify_source_provenance,
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
    assert "Bergamo" in in_set_ita["strict"]


def test_prepare_stage2_and_r_dry_validation(tmp_path: Path) -> None:
    """Build Stage-2 V/N tables on real GeLaTo genetics/trees and validate R Stan models."""
    prep_dir = build_smoke_test_inputs(tmp_path)
    out_dir = tmp_path / "stage2_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pos in ("V", "N"):
        ld_path = prep_dir / f"language_data_{pos}.csv"
        ind_path = prep_dir / f"individuals_{pos}.csv"
        pop_path = prep_dir / f"populations_{pos}.csv"
        nwk_path = prep_dir / f"phylo_tree_{pos}.nwk"
        lf_path = prep_dir / f"lemma_freq_data_{pos}.csv"

        assert ld_path.exists() and ind_path.exists() and pop_path.exists()
        assert nwk_path.exists() and lf_path.exists()

        with open(ld_path, encoding="utf-8") as f:
            ld_rows = list(csv.DictReader(f))
        assert len(ld_rows) == 18
        # Strict 1-row-per-glottocode invariant
        gcs = [r["glottocode"] for r in ld_rows]
        assert len(set(gcs)) == len(gcs)
        # Track 1 overlap populated (e.g. ital1282, finn1318, russ1263 are in Grambank)
        n_t1 = sum(int(r["in_track1"]) for r in ld_rows)
        assert n_t1 >= 10
        assert any(
            r["theta_irreg13_eap"] != "" and math.isfinite(float(r["theta_irreg13_eap"]))
            for r in ld_rows
        )

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

    # Run `PARA_DRY=1 Rscript analyses/paradigm/stage2/fit_stage2.R`
    env = {
        **os.environ,
        "PARA_DRY": "1",
        "PARA_CORES": "2",
        "PARA_POS": "V,N",
        "PARA_SETS": "main,strict",
        "PARA_MEASURES": "ent,sqrt_het,f3k_est_pos",
        "PARA_DATA_DIR": str(prep_dir),
        "PARA_OUT_DIR": str(out_dir),
    }
    res_s2 = subprocess.run(
        ["Rscript", str(REPO_ROOT / "analyses/paradigm/stage2/fit_stage2.R")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_s2.returncode == 0, f"fit_stage2.R failed:\nSTDOUT:\n{res_s2.stdout}\nSTDERR:\n{res_s2.stderr}"

    for pos in ("V", "N"):
        dry_csv = out_dir / f"stage2_results_{pos}_dry.csv"
        assert dry_csv.exists()
        with open(dry_csv, encoding="utf-8") as f:
            s2_rows = list(csv.DictReader(f))
        spec_groups = {r["spec_group"] for r in s2_rows}
        expected_groups = {
            "baseline",
            "full_covs",
            "full_covs_mundlak",
            "primary_demog",
            "primary_demog_mundlak",
            "bouckaert",
            "ctrl_C_distinct",
            "tier2_uniform480",
            "topfreq120",
            "pcfp_k1",
            "pcfp_k3",
            "reliability",
            "neural_monotonic",
            "no_turk_mong",
            "no_inner_asia",
            "link_sensitivity",
            "berkson_20draw",
        }
        assert expected_groups <= spec_groups, f"Missing spec groups: {expected_groups - spec_groups}"

    corr_csv = out_dir / "cross_track_correlations.csv"
    assert corr_csv.exists()
    with open(corr_csv, encoding="utf-8") as f:
        corr_rows = list(csv.DictReader(f))
    comp_names = {r["comparison"] for r in corr_rows}
    assert "H_tree_V_vs_Track1_irreg13_prop" in comp_names
    assert "H_tree_N_vs_Track1_theta_irreg13_EAP" in comp_names
    assert "H_tree_V_vs_H_tree_N" in comp_names

    # Run `PARA_DRY=1 Rscript analyses/paradigm/stage2/fit_entrenchment.R`
    res_ent = subprocess.run(
        ["Rscript", str(REPO_ROOT / "analyses/paradigm/stage2/fit_entrenchment.R")],
        cwd=REPO_ROOT,
        env=env,
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

    # Verify `unit_outcomes_uniform120_malouf.csv` ingestion + singleton `source_prov` collapse on subset
    mgl_dir = tmp_path / "mgl_results"
    with open(mgl_dir / "unit_outcomes_uniform120.csv", encoding="utf-8") as f:
        u120_orig = list(csv.DictReader(f))
    mal_rows = [
        {**r, "model": "malouf", "h_tree_bits": "0.4123"}
        for r in u120_orig
        if r["glottocode"] in ("finn1318", "nucl1301")
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

    # Make `malt1254` the ONLY `"national"` source_prov row, and set its `het`/`sqrt_het` to NA
    # so `ent` sees 2 `source_prov` levels (N=18) while `sqrt_het` sees only 1 level (`unimorph`, N=17).
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
            for col_na in ("het", "het_se", "sqrt_het", "sqrt_het_se", "f3_z", "f3_est", "f3k_z", "f3k_est", "f3k_est_pos"):
                pr[col_na] = ""
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
        "PARA_MEASURES": "ent,sqrt_het,f3k_est_pos",
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
    expected_n_f3 = len({pr["glottocode"] for pr in pops_v2 if pr["f3_z"] not in ("", "NA")})
    assert expected_n_f3 <= 17
    fc_ent = [r for r in edge_rows if r["spec_group"] == "full_covs" and r["measure"] == "ent"]
    fc_het = [r for r in edge_rows if r["spec_group"] == "full_covs" and r["measure"] == "sqrt_het"]
    fc_f3 = [r for r in edge_rows if r["spec_group"] == "full_covs" and r["measure"] == "f3k_est_pos"]
    assert len(fc_ent) == 4 and all(int(r["n_lang"]) == 18 for r in fc_ent)
    assert len(fc_het) == 4 and all(int(r["n_lang"]) == 17 for r in fc_het)
    assert len(fc_f3) == 4 and all(int(r["n_lang"]) == expected_n_f3 for r in fc_f3)


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
