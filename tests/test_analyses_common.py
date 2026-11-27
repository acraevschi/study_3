"""Tests for `analyses/common/` and `analyses/grambank/prepare.py`.

Verifies:
  1. Canonical Grambank tables in `analyses/grambank/data/` have exact row/column counts
     and invariants (and match the reference fixtures bit-for-bit when present).
  2. `analyses.common.demography`, `analyses.common.genetics`, and `analyses.common.phylogeny`
     work cleanly on an arbitrary LDL/Wug language sample (e.g. Italian, Finnish, German,
     Russian, French) and `(population, glottocode)` link pairs.
  3. `analyses/common/data/gelato_populations_summary.csv` covers all 558 unique GeLaTo
     ADMIXTURE populations.

Compatible with both `pytest` (`.venv/bin/python -m pytest`) and `unittest`
(`/usr/bin/python3 -m unittest tests/test_analyses_common.py`).
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import unittest
from pathlib import Path

from analyses.common.demography import (
    resolve_demography_df,
    resolve_demography_rows,
)
from analyses.common.genetics import (
    DEFAULT_MIN_N_IND,
    SENSITIVITY_MIN_N_IND,
    compute_discrete_k_diagnostics,
    compute_grambank_nonmorph_coverage,
    compute_individual_measures,
    compute_language_f3_maxp,
    compute_population_measures,
    load_admixture_q,
    load_f3_fixed_scores,
    load_f3_scores,
    load_fst_neighbor_regimes,
    load_gelato_metadata,
    load_graff_targets,
)
from analyses.common.phylogeny import (
    build_newick,
    resolve_glottolog_coords_rows,
    resolve_glottolog_paths,
)

ROOT = Path(__file__).resolve().parents[1]
GB_DATA = ROOT / "analyses/grambank/data"
COMMON_DATA = ROOT / "analyses/common/data"


class TestAnalysesCommonAndGrambank(unittest.TestCase):
    def test_grambank_canonical_tables_and_invariants(self) -> None:
        required_files = [
            "language_data.csv",
            "populations.csv",
            "individuals.csv",
            "phylo_tree.nwk",
            "prep_summary.json",
            "items_metadata.csv",
            "item_observations.csv",
            "language_table.csv",
            "latent_prep_summary.json",
        ]
        for fname in required_files:
            self.assertTrue((GB_DATA / fname).exists(), f"Missing {GB_DATA / fname}")

        with open(GB_DATA / "language_data.csv", encoding="utf-8") as f:
            langs = list(csv.DictReader(f))
        self.assertEqual(len(langs), 225)
        self.assertEqual(sum(int(r["has_pop"]) for r in langs if int(r["in_main"]) == 1), 218)
        self.assertEqual(sum(int(r["has_area"]) for r in langs if int(r["in_main"]) == 1), 215)

        with open(GB_DATA / "populations.csv", encoding="utf-8") as f:
            pops = list(csv.DictReader(f))
        self.assertEqual(len(pops), 364)

        with open(GB_DATA / "individuals.csv", encoding="utf-8") as f:
            inds = list(csv.DictReader(f))
        self.assertEqual(len(inds), 3166)

        with open(GB_DATA / "items_metadata.csv", encoding="utf-8") as f:
            items = list(csv.DictReader(f))
        self.assertEqual(len(items), 29)
        self.assertEqual(sum(1 for r in items if r["item_group"] == "v_breadth"), 9)
        self.assertEqual(sum(1 for r in items if r["item_group"] == "n_breadth"), 5)
        self.assertEqual(sum(1 for r in items if r["item_group"] == "v_irreg"), 5)
        self.assertEqual(sum(1 for r in items if r["item_group"] == "n_irreg"), 3)
        self.assertEqual(sum(1 for r in items if r["item_group"] == "np_agr"), 7)

        with open(GB_DATA / "item_observations.csv", encoding="utf-8") as f:
            obs = list(csv.DictReader(f))
        self.assertEqual(len(obs), 5411)

        with open(GB_DATA / "language_table.csv", encoding="utf-8") as f:
            ltab = list(csv.DictReader(f))
        self.assertEqual(len(ltab), 225)
        self.assertIn("grambank_entry", ltab[0])

        summ = json.loads((GB_DATA / "prep_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summ["sets"]["main"]["languages"], 225)
        self.assertEqual(summ["sets"]["main"]["populations"], 364)
        self.assertEqual(summ["sets"]["main"]["individuals"], 3166)
        self.assertEqual(summ["n_nonmorph_features_total"], 151)

        lat_summ = json.loads((GB_DATA / "latent_prep_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(lat_summ["n_languages_total"], 225)
        self.assertEqual(lat_summ["n_items_total"], 29)
        self.assertEqual(lat_summ["n_nonmorph_features_total"], 151)
        self.assertEqual(lat_summ["n_total_observations"], 5411)
        self.assertEqual(lat_summ["domains"]["v_breadth"]["n_lang_f3_n3"], 176)
        self.assertEqual(lat_summ["domains"]["v_breadth"]["n_lang_f3_n5"], 167)
        self.assertEqual(lat_summ["domains"]["n_breadth"]["n_lang_f3_n3"], 177)
        self.assertEqual(lat_summ["domains"]["n_breadth"]["n_lang_f3_n5"], 167)
        self.assertEqual(lat_summ["domains"]["v_irreg"]["n_lang_f3_n3"], 140)
        self.assertEqual(lat_summ["domains"]["v_irreg"]["n_lang_f3_n5"], 133)
        self.assertEqual(lat_summ["domains"]["n_irreg"]["n_lang_f3_n3"], 113)
        self.assertEqual(lat_summ["domains"]["n_irreg"]["n_lang_f3_n5"], 108)
        self.assertEqual(lat_summ["domains"]["np_agr"]["n_lang_f3_n3"], 167)
        self.assertEqual(lat_summ["domains"]["np_agr"]["n_lang_f3_n5"], 160)

        # Verify running analyses.grambank.prepare.main() reproduces all 9 files bit-for-bit
        before_bytes = {fname: (GB_DATA / fname).read_bytes() for fname in required_files}
        from analyses.grambank import prepare as gb_prepare
        gb_prepare.main()
        for fname in required_files:
            self.assertEqual(
                (GB_DATA / fname).read_bytes(),
                before_bytes[fname],
                f"prepare.py did not reproduce {fname} bit-for-bit",
            )

    def test_common_demography_and_phylogeny_on_ldl_sample(self) -> None:
        ldl_glottocodes = ["ital1282", "finn1318", "deut1234", "russ1263", "stan1290"]
        demog_rows = resolve_demography_rows(ldl_glottocodes)
        self.assertEqual([r["glottocode"] for r in demog_rows], ldl_glottocodes)
        by_gc = {r["glottocode"]: r for r in demog_rows}
        # Italian, Finnish, Russian, and French all have measured L1 population and range area
        for gc in ("ital1282", "finn1318", "russ1263", "stan1290"):
            self.assertEqual(by_gc[gc]["has_pop"], 1, f"Expected has_pop=1 for {gc}")
            self.assertEqual(by_gc[gc]["has_area"], 1, f"Expected has_area=1 for {gc}")
            self.assertGreater(float(by_gc[gc]["population_l1"]), 1000.0)
            self.assertGreater(float(by_gc[gc]["area_km2"]), 100.0)

        demog_df = resolve_demography_df(ldl_glottocodes)
        self.assertEqual(len(demog_df), 5)

        paths, src = resolve_glottolog_paths(ldl_glottocodes)
        self.assertEqual(set(paths.keys()), set(ldl_glottocodes))
        self.assertEqual(set(src.keys()), set(ldl_glottocodes))
        nwk_str = build_newick(paths)
        self.assertTrue(nwk_str.endswith(")root;"))
        for gc in ldl_glottocodes:
            self.assertIn(gc, nwk_str)

        coords = resolve_glottolog_coords_rows(ldl_glottocodes)
        self.assertEqual(len(coords), 5)
        for r in coords:
            if r["glottocode"] in ("ital1282", "finn1318", "russ1263", "stan1290"):
                self.assertFalse(math.isnan(float(r["lat"])))
                self.assertFalse(math.isnan(float(r["lon"])))

    def test_common_genetics_on_ldl_population_links(self) -> None:
        ldl_links = [
            {"population": "Bergamo", "glottocode": "ital1282"},
            {"population": "Tuscan", "glottocode": "ital1282"},
            {"population": "Finnish", "glottocode": "finn1318"},
            {"population": "Russian", "glottocode": "russ1263"},
            {"population": "French_Central", "glottocode": "stan1290"},
        ]
        info, Q = load_admixture_q()
        ind = compute_individual_measures(info, Q, links_df=ldl_links)
        ind_rows = ind if isinstance(ind, list) else ind.to_dict("records")
        self.assertGreater(len(ind_rows), 30)
        self.assertEqual({r["glottocode"] for r in ind_rows}, {"ital1282", "finn1318", "russ1263", "stan1290"})

        pops, pooled = compute_population_measures(
            info,
            Q,
            [r["population"] for r in ldl_links],
            n_boot=50,
            seed=20261021,
            include_extended_metadata=True,
            warn_missing_f3=False,
        )
        pop_rows = pops if isinstance(pops, list) else pops.to_dict("records")
        self.assertEqual(len(pop_rows), 5)
        for m in ("pop_ent", "het", "support", "amount", "sqrt_het"):
            self.assertIn(m, pooled)
            self.assertGreater(pooled[m], 0.0)
        for r in pop_rows:
            self.assertGreater(float(r["pop_ent"]), 0.0)
            self.assertIn("harmonicMean", r)
            self.assertIn("f3k_est_pos", r)

        # Verify shared all-GeLaTo-population summary table exists and has all 558 populations
        gel_sum_path = COMMON_DATA / "gelato_populations_summary.csv"
        self.assertTrue(gel_sum_path.exists())
        with open(gel_sum_path, encoding="utf-8") as f:
            gel_rows = list(csv.DictReader(f))
        self.assertEqual(len(gel_rows), 558)
        self.assertGreater(len(load_graff_targets()), 50)
        self.assertGreater(len(load_gelato_metadata()), 100)
        self.assertGreater(len(load_f3_scores()), 300)

    def test_strict_link_flags_and_main_invariance(self) -> None:
        links_path = COMMON_DATA / "analysis_links.csv"
        with open(links_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        main_rows = [r for r in rows if r["analysis"] == "main"]
        strict_rows = [r for r in rows if r["analysis"] == "strict"]
        self.assertEqual(len(main_rows), 533)
        strict_keys = {(r["population"], r["glottocode"]) for r in strict_rows}
        main_keys = {(r["population"], r["glottocode"]) for r in main_rows}

        must_exclude_strict = [
            ("Bergamo", "ital1282"),
            ("French_East", "stan1290"),
            ("French_Northwest", "stan1290"),
            ("French_South", "stan1290"),
            ("French_West", "stan1290"),
            ("GujaratiA", "guja1252"),
            ("GujaratiB", "guja1252"),
            ("GujaratiC", "guja1252"),
            ("Ashkenazi_Jew", "east2295"),
            ("Jew_Iraqi", "jude1266"),
            ("Jew_Libyan", "jude1264"),
            ("Jew_Moroccan", "jude1265"),
            ("Jew_Tunisian", "tuni1259"),
            ("Jew_Turkish", "ladi1251"),
            ("Jew_Yemenite", "jude1267"),
            ("Albanian", "gheg1238"),
            ("Greek_Athens", "mode1248"),
            ("Greek_Thessaloniki", "mode1248"),
            ("Greek_WGA", "mode1248"),
            ("Baining_Marabu", "mali1284"),
        ]
        for key in must_exclude_strict:
            self.assertIn(key, main_keys, f"{key} must remain in main")
            self.assertNotIn(key, strict_keys, f"{key} must be excluded from strict")

        for keep_key in (("Tuscan", "ital1282"), ("French_Central", "stan1290"), ("Druze", "nort3139")):
            self.assertIn(keep_key, strict_keys, f"{keep_key} should remain in strict")

    def test_f3_fixed_fst_regimes_discrete_k_and_nonmorph_coverage(self) -> None:
        # 1. Precomputed fixed f3 scores across all 517 linked populations
        f3f = load_f3_fixed_scores()
        f3f_rows = f3f if isinstance(f3f, list) else f3f.to_dict("records")
        self.assertEqual(len(f3f_rows), 517)
        by_pop = {str(r["population"]): r for r in f3f_rows}
        # Track 2 non-Grambank & rescued benchmark populations must have primary f3_fixed scores
        for p in (
            "German_Lipsian", "Spanish_Andalucia", "Romanian",
            "Turkish_Istanbul", "Hungarian1", "Finnish", "Yakut", "Veps", "Russian",
        ):
            self.assertIn(p, by_pop)
            self.assertFalse(math.isnan(float(by_pop[p]["f3_fixed"])), f"Missing f3_fixed for {p}")
            self.assertGreaterEqual(float(by_pop[p]["f3_fixed"]), 0.0)
            self.assertNotEqual(str(by_pop[p]["best_pop2"]), p)
            self.assertEqual(int(by_pop[p]["iso_fallback_used"]), 0)
        # Multi-glottocode conspecificity guard: Taa_North & Taa_West must not pick Taa_East as B*
        for taa_p in ("Taa_North", "Taa_West"):
            self.assertNotEqual(str(by_pop[taa_p]["best_pop2"]), "Taa_East")
        # African populations must have NA for Mbuti-outgroup f4
        self.assertTrue(math.isnan(float(by_pop["Hadza"]["f4_est"])))
        # Multi-pop isolates use fallback ONLY in f3_fixed_iso
        for iso_p in ("Basque", "Hadza", "Ket"):
            self.assertTrue(math.isnan(float(by_pop[iso_p]["f3_fixed"])))
            self.assertEqual(int(by_pop[iso_p]["iso_fallback_used"]), 1)
            self.assertFalse(math.isnan(float(by_pop[iso_p]["f3_fixed_iso"])))

        # 2. FST neighbor flow & Barbieri Regime A/B table
        fstr = load_fst_neighbor_regimes()
        fstr_rows = fstr if isinstance(fstr, list) else fstr.to_dict("records")
        self.assertEqual(len(fstr_rows), 558)
        fstr_by_pop = {str(r["population"]): r for r in fstr_rows}
        n_by_pop = {str(r["population"]): int(r["n_individuals"]) for r in fstr_rows}
        # Every closest-FST reference proxy must satisfy n_individuals >= 5
        for r in fstr_rows:
            for col in ("close_pop_same_family", "close_pop_diff_family", "close_pop_diff_1000km"):
                ref_p = str(r[col])
                if ref_p:
                    self.assertGreaterEqual(n_by_pop[ref_p], 5, f"{r['population']} {col}={ref_p} has n<5")
            bm = r["barbieri_mismatch"]
            if bm is not None and not (isinstance(bm, float) and math.isnan(bm)):
                ra, rb, bmi = int(r["regime_a_enclave"]), int(r["regime_b_shift"]), int(bm)
                self.assertEqual(ra + rb, bmi, f"Non-exclusive regimes on {r['population']}: {ra}+{rb}!={bmi}")
        for taa_p in ("Taa_North", "Taa_West"):
            self.assertNotEqual(str(fstr_by_pop[taa_p]["close_pop_same_family"]), "Taa_East")
        for non_mm in ("Atayal", "Ain_Touta_WGA"):
            self.assertEqual(int(fstr_by_pop[non_mm]["barbieri_mismatch"]), 0)
            self.assertEqual(int(fstr_by_pop[non_mm]["regime_a_enclave"]), 0)
            self.assertEqual(int(fstr_by_pop[non_mm]["regime_b_shift"]), 0)
        for shift_p in ("Hungarian1", "Hungarian2", "Turkish_Istanbul", "Turkish_Trabzon", "Gagauz", "Maltese", "Brahui"):
            self.assertEqual(int(fstr_by_pop[shift_p]["barbieri_mismatch"]), 1)
            self.assertEqual(int(fstr_by_pop[shift_p]["regime_b_shift"]), 1)
            self.assertEqual(int(fstr_by_pop[shift_p]["regime_a_enclave"]), 0)
        for enc_p in ("Damara", "Jew_Cochin", "Druze"):
            self.assertEqual(int(fstr_by_pop[enc_p]["barbieri_mismatch"]), 1)
            self.assertEqual(int(fstr_by_pop[enc_p]["regime_a_enclave"]), 1)
            self.assertEqual(int(fstr_by_pop[enc_p]["regime_b_shift"]), 0)

        # 3. Discrete K=12, 20, 30 diagnostics & min_n_ind filtering + max_p unmasking
        info, Q = load_admixture_q()
        test_pops = ["Maltese", "Finnish", "Uygur", "Negidal_Orochi", "BedouinA", "BedouinB"]
        dk = compute_discrete_k_diagnostics(info, Q, test_pops, min_n_rmsd=SENSITIVITY_MIN_N_IND)
        dk_rows = dk if isinstance(dk, list) else dk.to_dict("records")
        dk_by_pop = {str(r["population"]): r for r in dk_rows}
        self.assertTrue(math.isnan(float(dk_by_pop["Negidal_Orochi"]["rmsd_K20"])))  # n = 4 < 5
        self.assertLess(float(dk_by_pop["Maltese"]["rmsd_K20"]), float(dk_by_pop["Finnish"]["rmsd_K20"]))
        dk_csv_path = COMMON_DATA / "discrete_k_diagnostics.csv"
        self.assertTrue(dk_csv_path.exists())
        with open(dk_csv_path, encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 558)

        pop_n3, _ = compute_population_measures(
            info,
            Q,
            ["Negidal_Orochi", "Wayuu", "Maltese", "BedouinA", "BedouinB"],
            n_boot=20,
            seed=20261021,
            min_n_ind=DEFAULT_MIN_N_IND,
            include_fixed_f3=True,
            include_fst_regimes=True,
            include_discrete_k=True,
            warn_missing_f3=False,
        )
        pop_n3_rows = pop_n3 if isinstance(pop_n3, list) else pop_n3.to_dict("records")
        self.assertNotIn("Wayuu", {str(r["population"]) for r in pop_n3_rows})  # n = 1 < 3
        self.assertIn("Negidal_Orochi", {str(r["population"]) for r in pop_n3_rows})  # n = 4 >= 3

        pop_n5, _ = compute_population_measures(
            info,
            Q,
            ["Negidal_Orochi", "Maltese"],
            n_boot=20,
            seed=20261021,
            min_n_ind=SENSITIVITY_MIN_N_IND,
            warn_missing_f3=False,
        )
        pop_n5_rows = pop_n5 if isinstance(pop_n5, list) else pop_n5.to_dict("records")
        self.assertEqual([str(r["population"]) for r in pop_n5_rows], ["Maltese"])

        maxp = compute_language_f3_maxp(
            [
                {"population": "BedouinA", "glottocode": "east2690", "n_individuals": 24, "f3_fixed": float(by_pop["BedouinA"]["f3_fixed"])},
                {"population": "BedouinB", "glottocode": "east2690", "n_individuals": 20, "f3_fixed": float(by_pop["BedouinB"]["f3_fixed"])},
            ],
            col="f3_fixed",
            min_n_ind=DEFAULT_MIN_N_IND,
        )
        maxp_rows = maxp if isinstance(maxp, list) else maxp.to_dict("records")
        self.assertEqual(len(maxp_rows), 1)
        self.assertEqual(maxp_rows[0]["f3_fixed_maxp_pop"], "BedouinA")
        self.assertGreater(float(maxp_rows[0]["f3_fixed_maxp"]), 0.0)

        # 4. Track-1 non-morphological Grambank coverage helper (z_coded_nonmorph)
        cov = compute_grambank_nonmorph_coverage()
        cov_rows = cov if isinstance(cov, list) else cov.to_dict("records")
        self.assertEqual(len(cov_rows), 225)
        self.assertTrue(all(int(r["n_nonmorph_total"]) == 151 for r in cov_rows))
        z_vals = [float(r["z_coded_nonmorph"]) for r in cov_rows]
        self.assertAlmostEqual(sum(z_vals) / len(z_vals), 0.0, places=6)

        # 5. R stage1_exposure.R dry-run check across ALL registered specs + rbind(diag) schema
        r_code = (
            'source("analyses/common/stage1_exposure.R"); '
            'ind <- read.csv("analyses/grambank/data/individuals.csv", stringsAsFactors=FALSE); '
            'pop <- read.csv("analyses/grambank/data/populations.csv", stringsAsFactors=FALSE); '
            'pop <- pop[pop$in_main == 1, ]; '
            'pop <- augment_population_f3_columns(pop, '
            '  k5_file="analyses/common/data/f3_scores_k5.csv", '
            '  f3_fixed_file="analyses/common/data/f3_fixed_scores.csv", '
            '  fst_regimes_file="analyses/common/data/fst_neighbor_regimes.csv", '
            '  discrete_k_file="analyses/common/data/discrete_k_diagnostics.csv"); '
            'all_m <- names(get_stage1_measure_specs()); '
            'res_list <- lapply(all_m, function(m) { '
            '  mn <- if (grepl("^rmsd_", m)) 5L else 3L; '
            '  fit_stage1_exposure(m, ind, pop, set_name="main", min_n_ind=mn, dry=TRUE) '
            '}); '
            'diag_all <- do.call(rbind, lapply(res_list, "[[", "diag")); '
            'stopifnot(nrow(diag_all) == length(all_m), ncol(diag_all) == 11L, all(diag_all$n_lang >= 100L)); '
            'expo_all <- do.call(rbind, lapply(res_list, "[[", "e")); '
            'stopifnot(length(unique(expo_all$measure)) == length(all_m));'
        )
        proc = subprocess.run(["Rscript", "-e", r_code], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, f"R stage1_exposure.R dry check failed:\n{proc.stderr}\n{proc.stdout}")


if __name__ == "__main__":
    unittest.main()
