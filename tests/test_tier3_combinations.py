"""
Tier 3: Cross-Feature Combinations & Invariant Integration Tests for study_3.

Verifies cross-table integrity between:
- Demographic Registry (M2) ↔ MGN Modeling Dataset (M3)
- Modeling Dataset (M3) ↔ Phylogenetic Covariance Matrix (M4)
- Spatial GP Coordinates ↔ Geographic Range Properties
- Monotonic Distance Factor Ordering vs brms formula constraints
- Multi-POS demographic stability
- Predictor monotonic scaling and trial bounds
"""

import subprocess
from pathlib import Path
import pytest
import pandas as pd
import numpy as np


class TestTier3CrossFeatureCombinations:
    """Integration and cross-feature invariant tests across pipeline stages."""

    def test_tier3_01_all_modeling_languages_present_in_registry(
        self, mgn_modeling_df, demographic_registry_df
    ):
        """Ensure 100% of languages present in mgn_modeling_dataset exist in global_demographic_registry."""
        model_isos = set(mgn_modeling_df["iso_sanitized"].dropna().unique())
        registry_isos = set(demographic_registry_df["iso_639_3"].dropna().unique())
        
        missing = model_isos - registry_isos
        assert not missing, f"Languages in modeling dataset not found in demographic registry: {missing}"

    def test_tier3_02_population_and_coordinates_consistency(
        self, mgn_modeling_df, demographic_registry_df
    ):
        """Verify that lat, lon, and log10_pop in modeling dataset match values in demographic registry."""
        registry_indexed = demographic_registry_df.set_index("iso_639_3")
        
        sample_rows = mgn_modeling_df.sample(min(200, len(mgn_modeling_df)), random_state=42)
        for _, row in sample_rows.iterrows():
            iso = row["iso_sanitized"]
            assert iso in registry_indexed.index, f"ISO {iso} missing from registry index"
            reg_row = registry_indexed.loc[iso]
            if isinstance(reg_row, pd.DataFrame):
                reg_row = reg_row.iloc[0]
            
            # Check coordinates
            assert np.isclose(row["lat"], reg_row["latitude"], atol=1e-3), (
                f"Latitude mismatch for {iso}: modeling={row['lat']} vs registry={reg_row['latitude']}"
            )
            assert np.isclose(row["lon"], reg_row["longitude"], atol=1e-3), (
                f"Longitude mismatch for {iso}: modeling={row['lon']} vs registry={reg_row['longitude']}"
            )
            # Check log10 population
            expected_log10_pop = np.log10(reg_row["population_l1"])
            assert np.isclose(row["log10_pop"], expected_log10_pop, atol=1e-3), (
                f"log10_pop mismatch for {iso}: modeling={row['log10_pop']} vs expected={expected_log10_pop}"
            )

    def test_tier3_03_glottocode_consistency(
        self, mgn_modeling_df, demographic_registry_df
    ):
        """Verify that glottocodes match between modeling dataset and demographic registry."""
        registry_indexed = demographic_registry_df.set_index("iso_639_3")
        sample_rows = mgn_modeling_df.sample(min(100, len(mgn_modeling_df)), random_state=42)
        for _, row in sample_rows.iterrows():
            iso = row["iso_sanitized"]
            reg_row = registry_indexed.loc[iso]
            if isinstance(reg_row, pd.DataFrame):
                reg_row = reg_row.iloc[0]
            assert str(row["glottocode"]) == str(reg_row["glottocode"]), (
                f"Glottocode mismatch for {iso}: modeling={row['glottocode']} vs registry={reg_row['glottocode']}"
            )

    def test_tier3_04_matrix_dimnames_match_modeling_dataset_languages(
        self, mgn_modeling_df, phylo_cov_matrix_path, project_root
    ):
        """Verify that all modeling dataset languages are covered by the phylogenetic covariance matrix dimnames."""
        assert phylo_cov_matrix_path.exists(), f"phylo_cov_matrix.rds missing at {phylo_cov_matrix_path}"
        
        r_cmd = "A <- readRDS('phylo_cov_matrix.rds'); cat(paste(rownames(A), collapse=','))"
        res = subprocess.run(["Rscript", "-e", r_cmd], cwd=str(project_root), capture_output=True, text=True)
        assert res.returncode == 0, f"Failed to extract dimnames from phylo_cov_matrix.rds:\n{res.stderr}"
        
        matrix_langs = set(res.stdout.strip().split(","))
        model_langs = set(mgn_modeling_df["iso_sanitized"].dropna().unique())
        
        missing_in_matrix = model_langs - matrix_langs
        assert not missing_in_matrix, (
            f"Languages in modeling dataset not covered by phylogenetic covariance matrix: {missing_in_matrix}"
        )

    def test_tier3_05_matrix_dimnames_equal_colnames(
        self, phylo_cov_matrix_path, project_root
    ):
        """Verify that rownames and colnames of matrix A are identical in order and content."""
        r_cmd = (
            "A <- readRDS('phylo_cov_matrix.rds'); "
            "stopifnot(identical(rownames(A), colnames(A))); "
            "cat('DIMNAMES_IDENTICAL')"
        )
        res = subprocess.run(["Rscript", "-e", r_cmd], cwd=str(project_root), capture_output=True, text=True)
        assert res.returncode == 0, f"Matrix dimnames mismatch between rows and columns:\n{res.stderr}"

    def test_tier3_06_monotonic_distance_integer_validity(self, mgn_modeling_df):
        """Verify distance values are valid positive integers suitable for brms mo() monotonic terms."""
        dists = mgn_modeling_df["distance"]
        assert (dists >= 1).all(), "Distance values must be >= 1"
        assert (dists % 1 == 0).all(), "Distance values must be integer-valued"

    def test_tier3_07_spatial_gp_coordinate_uniqueness_per_iso(self, mgn_modeling_df):
        """Ensure each language has exactly one unique (lat, lon) coordinate in the modeling dataset."""
        grouped = mgn_modeling_df.groupby("iso_sanitized")[["lat", "lon"]].nunique()
        assert (grouped["lat"] == 1).all(), "Multiple differing latitudes found for a single language"
        assert (grouped["lon"] == 1).all(), "Multiple differing longitudes found for a single language"

    def test_tier3_08_spatial_coordinates_non_null_island(self, mgn_modeling_df):
        """Ensure coordinates are not default/placeholder (0.0, 0.0) Null Island coordinates."""
        zero_coords = mgn_modeling_df[
            (np.isclose(mgn_modeling_df["lat"], 0.0)) & 
            (np.isclose(mgn_modeling_df["lon"], 0.0))
        ]
        assert len(zero_coords) == 0, f"Found {len(zero_coords)} rows with Null Island coordinates (0.0, 0.0)"

    def test_tier3_09_cross_pos_demographic_stability(self, mgn_modeling_df):
        """Verify that languages with multiple POS share identical demographic predictors across POS partitions."""
        pos_counts = mgn_modeling_df.groupby("iso_sanitized")["pos"].nunique()
        multi_pos_langs = pos_counts[pos_counts > 1].index
        
        for lang in multi_pos_langs:
            lang_df = mgn_modeling_df[mgn_modeling_df["iso_sanitized"] == lang]
            assert lang_df["log10_pop"].nunique() == 1, f"Inconsistent log10_pop across POS for language {lang}"
            assert lang_df["contact_richness_scaled"].nunique() == 1, (
                f"Inconsistent contact_richness_scaled across POS for language {lang}"
            )

    def test_tier3_10_log_transformation_monotonicity(self, mgn_modeling_df):
        """Verify strict monotonic correspondence between log10_pop and log10_pop_z."""
        sub = mgn_modeling_df[["log10_pop", "log10_pop_z"]].drop_duplicates().sort_values("log10_pop")
        assert sub["log10_pop_z"].is_monotonic_increasing, (
            "Standardized log10_pop_z does not monotonically preserve log10_pop order"
        )

    def test_tier3_11_binomial_trials_non_empty_predictions(self, mgn_modeling_df):
        """Verify prediction trials have valid total >= 1 and plausible non-negative accuracy."""
        assert (mgn_modeling_df["total"] >= 1).all(), "Found total trial count < 1"
        assert (mgn_modeling_df["correct"] >= 0).all(), "Found negative correct trial count"
        acc = mgn_modeling_df["correct"] / mgn_modeling_df["total"]
        assert (acc >= 0.0).all() and (acc <= 1.0).all(), "Trial accuracies out of [0, 1] range"
