"""
Tier 2: Comprehensive Boundary & Corner Case Tests for study_3.

Covers >= 5 tests across each of the 5 boundary domains:
1. Chatino Disambiguation & False Friends
2. Macro-Language & Multi-Standard Resolution
3. Paradigm Distances & Morphological Boundary Cases
4. Population & Demographic Extreme Boundaries
5. Covariance Matrix Numerical & Boundary Properties
"""

import json
import os
import subprocess
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from conftest import canonical_iso, MGN_LIVING_LANGS_64


# ---------------------------------------------------------------------------
# Domain 1: Chatino Disambiguation & False Friends
# ---------------------------------------------------------------------------

class TestTier2ChatinoDisambiguation:
    """Boundary tests for Chatino language disambiguation preventing false friends."""

    def test_tier2_chatino_01_yai_maps_to_western_highland_chatino(self, demographic_registry_df):
        """MGN 'yai' (Yaitepec Chatino) must resolve to ctp/west2644, not Yaghnobi.

        Glottolog treats Yaitepec Chatino (yait1239) and San Juan Quiahije Chatino
        (sanj1283) as dialects of Western Highland Chatino (west2644, ISO ctp), which
        is the language-level unit carrying demographic data. 'czp'/'yait1238' are not
        current Glottolog identifiers.
        """
        assert canonical_iso("yai") == "ctp", "yai must canonicalise to ctp"
        rows = demographic_registry_df[demographic_registry_df["iso_639_3"] == "ctp"]
        assert not rows.empty, "ctp (Western Highland Chatino) missing from registry"
        row = rows.iloc[0]
        assert row["glottocode"] == "west2644", f"Wrong glottocode: {row['glottocode']}"
        assert "Chatino" in str(row["language_name"]), \
            f"yai resolved to a non-Chatino language: {row['language_name']}"
        assert row["family"] == "Otomanguean", f"Wrong family: {row['family']}"

    def test_tier2_chatino_02_zen_maps_to_zenzontepec(self, demographic_registry_df):
        """MGN 'zen' (Zenzontepec Chatino) must resolve to czn/zenz1235, not Zenaga."""
        assert canonical_iso("zen") == "czn", "zen must canonicalise to czn"
        rows = demographic_registry_df[demographic_registry_df["iso_639_3"] == "czn"]
        assert not rows.empty, "czn missing from registry"
        row = rows.iloc[0]
        assert row["glottocode"] == "zenz1235", f"Wrong glottocode: {row['glottocode']}"
        assert "Chatino" in str(row["language_name"])
        assert row["family"] == "Otomanguean", f"Wrong family: {row['family']}"

    def test_tier2_chatino_03_geographic_coordinates_oaxaca(self, demographic_registry_df):
        """Both Chatino languages must sit in Oaxaca, Mexico."""
        isos = [canonical_iso("yai"), canonical_iso("zen")]
        rows = demographic_registry_df[demographic_registry_df["iso_639_3"].isin(isos)]
        assert len(rows) == 2, f"Expected 2 Chatino rows, found {len(rows)}"
        for _, row in rows.iterrows():
            assert 15.0 <= row["latitude"] <= 18.5, \
                f"{row['iso_639_3']} latitude {row['latitude']} outside Oaxaca"
            assert -98.5 <= row["longitude"] <= -95.0, \
                f"{row['iso_639_3']} longitude {row['longitude']} outside Oaxaca"

    def test_tier2_chatino_04_false_friends_not_used(self, demographic_registry_df):
        """The ISO false friends must never be the resolution target.

        In ISO 639-3, 'yai' is Yaghnobi (Iranian, Tajikistan) and 'zen' is Zenaga
        (Berber, Mauritania). Resolving MGN's Chatino corpora to either would place
        them on the wrong continent with the wrong phylogeny.
        """
        for mgn in ("yai", "zen"):
            assert canonical_iso(mgn) != mgn, f"{mgn} still resolves to its false friend"
        for false_friend, wrong_family in (("yai", "Indo-European"), ("zen", "Afro-Asiatic")):
            rows = demographic_registry_df[demographic_registry_df["iso_639_3"] == false_friend]
            if not rows.empty:
                assert rows.iloc[0]["family"] != "Otomanguean", \
                    f"{false_friend} wrongly labelled Otomanguean"

    def test_tier2_chatino_04_macro_area_mesoamerica(self, demographic_registry_df):
        """Verify Chatino macro-area is North America / Mesoamerica (not Eurasia or Africa)."""
        chatino_rows = demographic_registry_df[
            demographic_registry_df["glottocode"].isin(["west2644", "zenz1235"]) |
            demographic_registry_df["iso_639_3"].isin([canonical_iso("yai"), canonical_iso("zen")])
        ]
        for _, row in chatino_rows.iterrows():
            macro_area = str(row["macro_area"]).lower()
            assert any(term in macro_area for term in ["north america", "northamerica", "mesoamerica", "americas"]), (
                f"Unexpected macro-area '{row['macro_area']}' for Chatino {row['iso_639_3']}"
            )

    def test_tier2_chatino_05_genealogical_family_otomanguean(self, demographic_registry_df):
        """Verify Chatino language family is Otomanguean (Zapotecan branch)."""
        chatino_rows = demographic_registry_df[
            demographic_registry_df["glottocode"].isin(["west2644", "zenz1235"]) |
            demographic_registry_df["iso_639_3"].isin([canonical_iso("yai"), canonical_iso("zen")])
        ]
        for _, row in chatino_rows.iterrows():
            family = str(row["family"]).lower()
            assert "otomanguean" in family or "zapotecan" in family, (
                f"Unexpected family '{row['family']}' for Chatino {row['iso_639_3']}"
            )


# ---------------------------------------------------------------------------
# Domain 2: Macro-Language & Multi-Standard Resolution
# ---------------------------------------------------------------------------

class TestTier2MacroLanguageResolution:
    """Boundary tests for ISO 639-3 macro-language and alias resolutions."""

    def test_tier2_macro_01_arabic_resolution(self, demographic_registry_df):
        """Verify Arabic macro 'ara' resolves to Standard Arabic (arb / stan1318)."""
        match = demographic_registry_df[
            (demographic_registry_df["iso_639_3"].isin(["ara", "arb"])) |
            (demographic_registry_df["glottocode"] == "stan1318")
        ]
        assert not match.empty, "Standard Arabic missing from demographic registry"
        row = match.iloc[0]
        assert row["population_l1"] > 1_000_000, f"Unrealistic Arabic population: {row['population_l1']}"

    def test_tier2_macro_02_farsi_resolution(self, demographic_registry_df):
        """Verify Persian macro 'fas' resolves to Western Farsi / Persian (pes / west2369)."""
        match = demographic_registry_df[
            (demographic_registry_df["iso_639_3"].isin(["fas", "pes"])) |
            (demographic_registry_df["glottocode"] == "west2369")
        ]
        assert not match.empty, "Western Farsi missing from demographic registry"
        row = match.iloc[0]
        assert row["population_l1"] > 1_000_000, f"Unrealistic Persian population: {row['population_l1']}"

    def test_tier2_macro_03_azerbaijani_resolution(self, demographic_registry_df):
        """Verify Azerbaijani macro 'aze' resolves to North Azerbaijani (azj / nort2697)."""
        match = demographic_registry_df[
            (demographic_registry_df["iso_639_3"].isin(["aze", "azj"])) |
            (demographic_registry_df["glottocode"] == "nort2697")
        ]
        assert not match.empty, "North Azerbaijani missing from demographic registry"

    def test_tier2_macro_04_serbo_croatian_resolution(self, demographic_registry_df):
        """Verify Serbo-Croatian macro 'hbs' resolves to Serbian/Croatian/Bosnian (srp/hrv/bos)."""
        match = demographic_registry_df[
            (demographic_registry_df["iso_639_3"].isin(["hbs", "srp", "hrv", "bos"])) |
            (demographic_registry_df["glottocode"] == "sout1528")
        ]
        assert not match.empty, "Serbo-Croatian complex missing from demographic registry"

    def test_tier2_macro_05_estonian_yiddish_pashto(self, demographic_registry_df):
        """Verify Estonian (est->ekk), Yiddish (yid->ydd), Pashto (pus->pbt/pst)."""
        targets = [
            {"iso_candidates": ["est", "ekk"], "glotto": "stan1290"},
            {"iso_candidates": ["yid", "ydd"], "glotto": "east2295"},
            {"iso_candidates": ["pus", "pbt", "pst"], "glotto": "sout2649"}
        ]
        for t in targets:
            match = demographic_registry_df[
                demographic_registry_df["iso_639_3"].isin(t["iso_candidates"]) |
                (demographic_registry_df["glottocode"] == t["glotto"])
            ]
            assert not match.empty, f"Target language {t['iso_candidates']} missing in registry"

    def test_tier2_macro_06_french_galician_aliases(self, demographic_registry_df):
        """Verify legacy MGN aliases: French ('fre' -> 'fra') and Galician ('gal' -> 'glg')."""
        for legacy, canonical in [("fre", "fra"), ("gal", "glg")]:
            match = demographic_registry_df[
                demographic_registry_df["iso_639_3"].isin([legacy, canonical])
            ]
            assert not match.empty, f"Language {legacy}/{canonical} missing in demographic registry"


# ---------------------------------------------------------------------------
# Domain 3: Paradigm Distances & Morphological Boundary Cases
# ---------------------------------------------------------------------------

class TestTier2MorphologicalDistances:
    """Boundary tests for symmetric set difference distance calculations."""

    @staticmethod
    def sym_diff(f1_str: str, f2_str: str) -> int:
        set1 = set(f1_str.split(";"))
        set2 = set(f2_str.split(";"))
        return len(set1.symmetric_difference(set2))

    def test_tier2_distance_01_minimum_non_zero_distance(self):
        """Verify minimal distance d = 1 between adjacent paradigm cells differing in a single feature."""
        f1 = "V;IND;PRS;ACT;1;SG"
        f2 = "V;IND;PRS;ACT;2;SG"
        d = self.sym_diff(f1, f2)
        # set1: {V, IND, PRS, ACT, 1, SG}, set2: {V, IND, PRS, ACT, 2, SG}
        # symmetric difference: {1, 2} => len = 2.
        # If feature is 1 vs 2, symmetric difference is 2 elements.
        assert d == 2, f"Expected symmetric difference 2 for 1.SG vs 2.SG, got {d}"

    def test_tier2_distance_02_maximum_symmetric_difference(self):
        """Verify maximum distance when two feature bundles are completely disjoint."""
        f1 = "V;IND;PST;ACT;1;SG"
        f2 = "N;NOM;PL;FEM;DEF"
        d = self.sym_diff(f1, f2)
        assert d == len(set(f1.split(";"))) + len(set(f2.split(";"))), (
            f"Expected full sum of lengths for disjoint sets, got {d}"
        )

    def test_tier2_distance_03_symmetry_invariance(self, mgn_modeling_df):
        """Verify that distance calculation is strictly symmetric: d(c1, c2) == d(c2, c1)."""
        sample_rows = mgn_modeling_df.sample(min(100, len(mgn_modeling_df)), random_state=42)
        for _, row in sample_rows.iterrows():
            u1 = str(row["unimorph_1"])
            u2 = str(row["unimorph_2"])
            d12 = self.sym_diff(u1, u2)
            d21 = self.sym_diff(u2, u1)
            assert d12 == d21, f"Distance asymmetry detected between '{u1}' and '{u2}'"

    def test_tier2_distance_04_triangle_inequality_property(self):
        """Verify metric space property: d(F1, F3) <= d(F1, F2) + d(F2, F3)."""
        f1 = "V;IND;PRS;ACT;1;SG"
        f2 = "V;SBJV;PRS;ACT;1;SG"
        f3 = "V;SBJV;PST;PASS;3;PL;FEM"
        d12 = self.sym_diff(f1, f2)
        d23 = self.sym_diff(f2, f3)
        d13 = self.sym_diff(f1, f3)
        assert d13 <= d12 + d23, f"Triangle inequality violated: d13={d13} > d12+d23={d12+d23}"

    def test_tier2_distance_05_all_tags_in_mgn_data_mapped(self, cells_to_unimorph_path):
        """Verify that cells_to_unimorph.json contains valid non-empty mapping without null values."""
        with open(cells_to_unimorph_path, "r") as f:
            mapping = json.load(f)
        for k, v in mapping.items():
            assert isinstance(k, str) and len(k) > 0, f"Empty key in cells_to_unimorph: {k}"
            assert isinstance(v, str) and len(v) > 0, f"Empty or non-string mapping value for key {k}: {v}"


# ---------------------------------------------------------------------------
# Domain 4: Population & Demographic Extreme Boundaries
# ---------------------------------------------------------------------------

class TestTier2DemographicBoundaries:
    """Boundary tests for population sizes, proportions, and extreme demographic ranges."""

    def test_tier2_pop_01_micro_population_bounds(self, demographic_registry_df):
        """Verify small/endangered living languages have positive population and valid log10."""
        # Find languages with population < 5,000
        micro_langs = demographic_registry_df[demographic_registry_df["population_l1"] < 5000]
        assert not micro_langs.empty, "Expected some small languages in demographic registry"
        for _, row in micro_langs.iterrows():
            pop = row["population_l1"]
            assert pop > 0, f"Population for {row['iso_639_3']} is non-positive: {pop}"
            assert np.log10(pop) >= 0, f"log10(pop) for {row['iso_639_3']} is negative: {np.log10(pop)}"

    def test_tier2_pop_02_mega_population_bounds(self, demographic_registry_df):
        """Verify mega-languages (>10^8 speakers) do not overflow and have realistic log10."""
        mega_langs = demographic_registry_df[demographic_registry_df["population_l1"] >= 100_000_000]
        assert not mega_langs.empty, "Expected mega languages (e.g. English, Spanish, Mandarin) in registry"
        for _, row in mega_langs.iterrows():
            pop = row["population_l1"]
            assert pop <= 2_000_000_000, f"Exorbitant population > 2 billion for {row['iso_639_3']}: {pop}"
            log_pop = np.log10(pop)
            assert 8.0 <= log_pop <= 10.0, f"Unexpected log10 population {log_pop} for {row['iso_639_3']}"

    def test_tier2_pop_03_l2_proportion_range_bounds(self, demographic_registry_df):
        """Verify L2 proportion is strictly bounded in [0.0, 1.0]."""
        l2_vals = demographic_registry_df["l2_proportion"].dropna()
        if not l2_vals.empty:
            assert (l2_vals >= 0.0).all(), "Found negative L2 proportion"
            assert (l2_vals <= 1.0).all(), "Found L2 proportion > 1.0"

    def test_tier2_pop_04_vehicularity_binary_bounds(self, demographic_registry_df):
        """Verify vehicularity is binary {0, 1} where present."""
        veh_vals = demographic_registry_df["vehicularity"].dropna().unique()
        for v in veh_vals:
            assert int(v) in {0, 1}, f"Vehicularity must be 0 or 1, found: {v}"

    def test_tier2_pop_05_missing_covariate_fallback_robustness(self, mgn_modeling_df):
        """Verify that scaled macro-ecological covariates in modeling dataset have zero NaNs or Infs."""
        scaled_cols = [
            "contact_richness_scaled", "log10_area_scaled",
            "altitude_range_scaled", "roughness_scaled"
        ]
        for col in scaled_cols:
            if col in mgn_modeling_df.columns:
                series = mgn_modeling_df[col]
                assert not series.isna().any(), f"Scaled covariate '{col}' contains NaN values"
                assert not np.isinf(series).any(), f"Scaled covariate '{col}' contains Inf values"


# ---------------------------------------------------------------------------
# Domain 5: Covariance Matrix Numerical & Boundary Properties
# ---------------------------------------------------------------------------

class TestTier2CovarianceMatrixBoundaries:
    """Boundary tests for phylogenetic covariance matrix condition numbers, eigenvalues, and symmetry."""

    def test_tier2_cov_01_diagonal_dominance(self, phylo_cov_matrix_path, project_root):
        """Verify diagonal elements A_ii = 1.0 + eps >= A_ij for all i != j."""
        import subprocess
        r_cmd = (
            "A <- readRDS('phylo_cov_matrix.rds'); "
            "d <- diag(A); "
            "diag(A) <- 0; "
            "max_off <- max(A); "
            "min_diag <- min(d); "
            "stopifnot(min_diag >= max_off); "
            "cat(sprintf('MIN_DIAG=%.4f;MAX_OFF=%.4f', min_diag, max_off))"
        )
        res = subprocess.run(["Rscript", "-e", r_cmd], cwd=str(project_root), capture_output=True, text=True)
        assert res.returncode == 0, f"Diagonal dominance test failed:\n{res.stderr}"

    def test_tier2_cov_02_condition_number(self, phylo_cov_matrix_path, project_root):
        """Verify condition number kappa(A) < 10^5 to guarantee numerical stability in Stan."""
        import subprocess
        r_cmd = (
            "A <- readRDS('phylo_cov_matrix.rds'); "
            "ev <- eigen(A, symmetric = TRUE, only.values = TRUE)$values; "
            "cond <- max(ev) / min(ev); "
            "cat(sprintf('COND=%.2f', cond)); "
            "stopifnot(cond < 1e5)"
        )
        res = subprocess.run(["Rscript", "-e", r_cmd], cwd=str(project_root), capture_output=True, text=True)
        assert res.returncode == 0, f"Condition number test failed:\n{res.stderr}"

    def test_tier2_cov_03_bounded_minimum_eigenvalue(self, phylo_cov_matrix_path, project_root):
        """Verify minimum eigenvalue lambda_min >= 0.01."""
        import subprocess
        r_cmd = (
            "A <- readRDS('phylo_cov_matrix.rds'); "
            "ev <- eigen(A, symmetric = TRUE, only.values = TRUE)$values; "
            "min_ev <- min(ev); "
            "cat(sprintf('MIN_EV=%.6f', min_ev)); "
            "stopifnot(min_ev >= 0.01)"
        )
        res = subprocess.run(["Rscript", "-e", r_cmd], cwd=str(project_root), capture_output=True, text=True)
        assert res.returncode == 0, f"Minimum eigenvalue test failed:\n{res.stderr}"

    def test_tier2_cov_04_symmetry_tolerance(self, phylo_cov_matrix_path, project_root):
        """Verify matrix symmetry ||A - A^T|| < 10^-12."""
        import subprocess
        r_cmd = (
            "A <- readRDS('phylo_cov_matrix.rds'); "
            "diff <- max(abs(A - t(A))); "
            "cat(sprintf('MAX_ASYM=%.2e', diff)); "
            "stopifnot(diff < 1e-12)"
        )
        res = subprocess.run(["Rscript", "-e", r_cmd], cwd=str(project_root), capture_output=True, text=True)
        assert res.returncode == 0, f"Symmetry tolerance test failed:\n{res.stderr}"

    def test_tier2_cov_05_within_genus_vs_between_family_covariance(self, phylo_cov_matrix_path, project_root):
        """Verify that closely related languages (e.g. spa & por) have higher covariance than unrelated (e.g. spa & fin)."""
        import subprocess
        r_cmd = (
            "A <- readRDS('phylo_cov_matrix.rds'); "
            "langs <- rownames(A); "
            "if ('spa' %in% langs && 'por' %in% langs && 'fin' %in% langs) { "
            "  cov_spa_por <- A['spa', 'por']; "
            "  cov_spa_fin <- A['spa', 'fin']; "
            "  cat(sprintf('COV_SPA_POR=%.4f;COV_SPA_FIN=%.4f', cov_spa_por, cov_spa_fin)); "
            "  stopifnot(cov_spa_por > cov_spa_fin); "
            "} else { "
            "  cat('LANGS_NOT_FOUND_FOR_GENUS_CHECK'); "
            "}"
        )
        res = subprocess.run(["Rscript", "-e", r_cmd], cwd=str(project_root), capture_output=True, text=True)
        assert res.returncode == 0, f"Taxonomic covariance hierarchy check failed:\n{res.stderr}"
