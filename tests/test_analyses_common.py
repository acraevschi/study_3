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
import unittest
from pathlib import Path

from analyses.common.demography import (
    resolve_demography_df,
    resolve_demography_rows,
)
from analyses.common.genetics import (
    compute_individual_measures,
    compute_population_measures,
    load_admixture_q,
    load_f3_scores,
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
        self.assertEqual(len(langs), 202)
        self.assertEqual(sum(int(r["has_pop"]) for r in langs if int(r["in_main"]) == 1), 195)
        self.assertEqual(sum(int(r["has_area"]) for r in langs if int(r["in_main"]) == 1), 192)

        with open(GB_DATA / "populations.csv", encoding="utf-8") as f:
            pops = list(csv.DictReader(f))
        self.assertEqual(len(pops), 335)

        with open(GB_DATA / "individuals.csv", encoding="utf-8") as f:
            inds = list(csv.DictReader(f))
        self.assertEqual(len(inds), 2943)

        with open(GB_DATA / "items_metadata.csv", encoding="utf-8") as f:
            items = list(csv.DictReader(f))
        self.assertEqual(len(items), 38)
        self.assertEqual(sum(int(r["in_cat12"]) for r in items), 12)
        self.assertEqual(sum(int(r["in_irreg13"]) for r in items), 13)
        self.assertEqual(sum(int(r["in_flex26"]) for r in items), 26)

        with open(GB_DATA / "item_observations.csv", encoding="utf-8") as f:
            obs = list(csv.DictReader(f))
        self.assertEqual(len(obs), 6821)

        with open(GB_DATA / "language_table.csv", encoding="utf-8") as f:
            ltab = list(csv.DictReader(f))
        self.assertEqual(len(ltab), 202)
        self.assertIn("grambank_entry", ltab[0])

        summ = json.loads((GB_DATA / "prep_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summ["sets"]["main"]["languages"], 202)
        self.assertEqual(summ["sets"]["main"]["populations"], 335)
        self.assertEqual(summ["sets"]["main"]["individuals"], 2943)

        lat_summ = json.loads((GB_DATA / "latent_prep_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(lat_summ["n_languages_main_cat12_cov60"], 202)
        self.assertEqual(lat_summ["n_languages_main_cat12_irreg13_cov60"], 175)
        self.assertEqual(lat_summ["n_languages_main_cat12_flex26_cov60"], 186)
        self.assertEqual(lat_summ["n_total_observations"], 6821)

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

        # If reference fixtures are present, verify bit-for-bit equality
        adm_ref = ROOT / "analyses/admixture_links_2026_10_21"
        if adm_ref.exists():
            for fname in ("language_data.csv", "populations.csv", "individuals.csv", "phylo_tree.nwk", "prep_summary.json"):
                self.assertEqual(
                    (GB_DATA / fname).read_bytes(),
                    (adm_ref / fname).read_bytes(),
                    f"Bit-for-bit mismatch on {fname}",
                )
        lat_ref = ROOT / "analyses/latent_morphology_2026_10_30"
        if lat_ref.exists():
            for fname in ("items_metadata.csv", "item_observations.csv", "language_table.csv"):
                self.assertEqual(
                    (GB_DATA / fname).read_bytes(),
                    (lat_ref / fname).read_bytes(),
                    f"Bit-for-bit mismatch on {fname}",
                )
            self.assertEqual(
                (GB_DATA / "latent_prep_summary.json").read_bytes(),
                (lat_ref / "prep_summary.json").read_bytes(),
                "Bit-for-bit mismatch on latent_prep_summary.json",
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


if __name__ == "__main__":
    unittest.main()
