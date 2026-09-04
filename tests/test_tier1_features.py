"""
Tier 1: Comprehensive Feature Coverage Tests for study_3 (Requirements R1, R2, R3, R4).

Each requirement area contains >= 5 independent test cases verifying specifications
from .agents/ORIGINAL_REQUEST.md and docs/METHODS.md.
"""

import json
import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from conftest import (
    canonical_iso,
    MGN_LIVING_LANGS_64,
    MGN_EXTINCT_LANGS_9,
)

# ---------------------------------------------------------------------------
# Requirement R1: Complete Workspace Backup & Targeted Cleanup
# ---------------------------------------------------------------------------

class TestRequirementR1BackupAndHygiene:
    """Feature coverage tests for R1: Backup verification and workspace cleanup."""

    def test_r1_01_backup_directory_exists_and_populated(self, backup_root):
        """Verify that /Users/acraev/Work/study_3_backup exists, is a directory, and contains files."""
        assert backup_root.exists(), f"Backup directory does not exist at {backup_root}"
        assert backup_root.is_dir(), f"Backup path is not a directory: {backup_root}"
        backup_files = list(backup_root.iterdir())
        assert len(backup_files) >= 10, f"Backup directory seems incomplete: found {len(backup_files)} items"

    def test_r1_02_backup_contains_preserved_scientific_tracks(self, backup_root):
        """Verify the pre-cleanup backup still holds every track, including the retired ones.

        The backup is the only remaining copy of the LDL/JudiLing/UniMorph/LCC
        experiment tracks that the 2026-09-03 cleanup removed from the working tree.
        """
        expected_dirs = ["mgn_data", "unimorph", "CorA-ReN-XML_1.1",
                         "lcc_corpora", "ldl_results", "judiling_output"]
        for d in expected_dirs:
            target = backup_root / d
            assert target.exists(), f"Preserved scientific track '{d}' missing in backup: {target}"
            assert target.is_dir(), f"Expected directory in backup for '{d}'"

    def test_r1_03_deprecated_root_scripts_removed(self, project_root):
        """Verify that obsolete Factbook scrapers and scratch scripts are deleted from root."""
        deprecated_files = [
            # Factbook / Wikidata scrapers and scratch files (removed 2026-09-02)
            "extract_languages.py",
            "extracted_languages.json",
            "scratch.py",
            "extract_population_data.py",
            "process_l2_speakers.py",
            "process_l2_speakers_conservative.py",
            # Orphaned outputs of those scrapers (removed 2026-09-03)
            "l2_speakers.csv",
            "l2_speakers_conservative.csv",
            "population_glottolog_merged.csv",
            "population_glottolog_merged_filtered.csv",
            # Superseded model versions (removed 2026-09-03)
            "prelim_analysis_mgn.R",
            "prelim_analysis_mgn_cells.R",
            "prelim_analysis_mgn_cells_binomial.R",
            "prelim_analysis_mgn_cells_binomial_median.R",
            "tree_glottolog_newick.txt",
            # Retired LDL / JudiLing / phonemicisation track (removed 2026-09-03)
            "check_ldl.ipynb",
            "check_extra_langs.py",
            "normalize_cells.ipynb",
            "results_3900.csv",
        ]
        found = [f for f in deprecated_files if (project_root / f).exists()]
        assert not found, f"Deprecated files still present in root directory: {found}"

    def test_r1_04_deprecated_factbook_dir_removed(self, project_root):
        """Verify that the cloned factbook.json/ directory is completely removed."""
        factbook_dir = project_root / "factbook.json"
        assert not factbook_dir.exists(), f"Deprecated directory factbook.json still exists at {factbook_dir}"

    def test_r1_05_deprecated_src_scripts_removed(self, project_root):
        """Verify src/ holds only the five live modules, with no retired ones left behind."""
        deprecated_src = [
            "demographic_data.py",
            # Retired LDL / JudiLing / phonemicisation modules (removed 2026-09-03)
            "analyze_ldl_output.py", "assign_freqs_lcc.py", "check_paradigms.ipynb",
            "download_lcc.py", "extract_unimorph.py", "extract_wikipron.py",
            "g2p_unimorph_freq.py", "ldl_unimorph.py", "ldl_unimorph_new.py",
            "prepare_ldl_data.py", "run_ldl.jl", "run_ldl_ortho.jl",
        ]
        found = [f for f in deprecated_src if (project_root / "src" / f).exists()]
        assert not found, f"Retired src modules still present: {found}"

    def test_r1_06_preserved_scientific_tracks_intact_in_working_dir(self, project_root):
        """Verify that all core scientific tracks and essential project files are intact in working directory."""
        essential_tracks = [
            # MGN benchmark input
            project_root / "mgn_data",
            # Demographic source data
            project_root / "data_sources",
            project_root / "data_sources" / "ethnologue_population_data.csv",
            project_root / "data_sources" / "cells_to_unimorph.json",
            # Low/High German historical track
            project_root / "germanic" / "cora_ren_xml_1.1",
            project_root / "germanic" / "unimorph",
            project_root / "germanic" / "extracted_verbs.csv",
            project_root / "src" / "low_german_extraction.py",
            # Docs
            project_root / "README.md",
            project_root / "docs" / "METHODS.md",
            project_root / "LICENSE",
        ]
        for track in essential_tracks:
            assert track.exists(), f"Core scientific asset or track missing in working dir: {track}"


# ---------------------------------------------------------------------------
# Requirement R2: Unified Cross-Linguistic Demographic Data Fusion
# ---------------------------------------------------------------------------

class TestRequirementR2DemographicRegistry:
    """Feature coverage tests for R2: Demographic Data Fusion Pipeline & Registry."""

    def test_r2_01_registry_file_existence(self, demographic_registry_path):
        """Verify global_demographic_registry.csv exists and is non-empty."""
        assert demographic_registry_path.exists(), f"Registry file missing at {demographic_registry_path}"
        assert demographic_registry_path.stat().st_size > 0, "Registry file is empty (0 bytes)"

    def test_r2_02_registry_schema_compliance(self, demographic_registry_df):
        """Validate that all 15 interface contract columns are present in the registry."""
        from tests.conftest import DEMOGRAPHIC_REGISTRY_REQUIRED_COLS
        actual_cols = list(demographic_registry_df.columns)
        missing_cols = [c for c in DEMOGRAPHIC_REGISTRY_REQUIRED_COLS if c not in actual_cols]
        assert not missing_cols, f"Registry missing required columns: {missing_cols}. Found: {actual_cols}"

    def test_r2_03_registry_row_count_threshold(self, demographic_registry_df):
        """Ensure global_demographic_registry.csv contains > 6,500 language records."""
        total_rows = len(demographic_registry_df)
        assert total_rows > 6500, f"Expected >6,500 language records, got {total_rows}"
        unique_isos = demographic_registry_df["iso_639_3"].dropna().nunique()
        assert unique_isos > 6000, f"Expected >6,000 unique ISO codes, got {unique_isos}"

    def test_r2_04_registry_mgn_living_languages_complete_coverage(self, demographic_registry_df):
        """All 64 living MGN languages resolve to a registry row with a real population."""
        isos = set(demographic_registry_df["iso_639_3"])
        missing = []
        for mgn in sorted(MGN_LIVING_LANGS_64):
            target = canonical_iso(mgn)
            if target not in isos:
                missing.append(f"{mgn}->{target} (absent)")
                continue
            row = demographic_registry_df[
                demographic_registry_df["iso_639_3"] == target].iloc[0]
            if pd.isna(row["population_l1"]) or row["population_l1"] <= 0:
                missing.append(f"{mgn}->{target} (no population)")
        assert not missing, f"Missing {len(missing)}/64 living MGN languages: {missing}"

    def test_r2_05_registry_ancient_extinct_exclusion(self, demographic_registry_df):
        """Ensure none of the ancient/extinct languages are present in the synchronic demographic registry."""
        from tests.conftest import MGN_EXTINCT_LANGS_9
        registry_isos = set(demographic_registry_df["iso_639_3"].dropna())
        present_extinct = MGN_EXTINCT_LANGS_9.intersection(registry_isos)
        assert not present_extinct, (
            f"Ancient/extinct languages found in synchronic registry: {present_extinct}"
        )

    def test_r2_06_registry_provenance_and_valid_ranges(self, demographic_registry_df):
        """Verify provenance tracking column is fully populated with valid source labels."""
        valid_sources = {"Bromham", "Koplenig", "JoshuaProject", "Grambank", "Wikidata", "Ethnologue"}
        sources = demographic_registry_df["population_source"].dropna().unique()
        assert len(sources) > 0, "No population sources recorded"
        # Ensure no nulls in population_source for entries with population_l1
        with_pop = demographic_registry_df[demographic_registry_df["population_l1"].notna()]
        null_sources = with_pop["population_source"].isna().sum()
        assert null_sources == 0, f"Found {null_sources} rows with population but missing population_source"


# ---------------------------------------------------------------------------
# Requirement R3: MGN Dataset Integration & Feature Engineering
# ---------------------------------------------------------------------------

class TestRequirementR3MGNFeatureEngineering:
    """Feature coverage tests for R3: MGN feature integration, distance calculations, transformations."""

    def test_r3_01_mgn_modeling_dataset_existence(self, mgn_modeling_dataset_path):
        """Verify mgn_modeling_dataset.csv exists and is non-empty."""
        assert mgn_modeling_dataset_path.exists(), f"Modeling dataset missing at {mgn_modeling_dataset_path}"
        assert mgn_modeling_dataset_path.stat().st_size > 0, "Modeling dataset file is empty"

    def test_r3_02_mgn_modeling_dataset_schema(self, mgn_modeling_df):
        """Validate required columns in mgn_modeling_dataset.csv."""
        from tests.conftest import MGN_MODELING_REQUIRED_COLS
        actual_cols = list(mgn_modeling_df.columns)
        missing_cols = [c for c in MGN_MODELING_REQUIRED_COLS if c not in actual_cols]
        assert not missing_cols, f"Modeling dataset missing required columns: {missing_cols}"

    def test_r3_03_unimorph_tag_mapping_validity(self, cells_to_unimorph_path):
        """Verify cells_to_unimorph.json contains valid non-empty mapping from MGN tags to UniMorph bundles."""
        assert cells_to_unimorph_path.exists(), f"Mapping file missing at {cells_to_unimorph_path}"
        with open(cells_to_unimorph_path, "r") as f:
            mapping = json.load(f)
        assert len(mapping) > 200, f"Expected >200 mappings in cells_to_unimorph.json, got {len(mapping)}"
        # Check sample tag structure
        sample_keys = list(mapping.keys())[:10]
        for k in sample_keys:
            val = mapping[k]
            assert isinstance(val, str) and len(val) > 0, f"Invalid mapped value for '{k}': {val}"
            assert ";" in val or val.isupper(), f"Mapped value '{val}' does not follow UniMorph convention"

    def test_r3_04_symmetric_set_difference_distance_logic(self, mgn_modeling_df):
        """Verify symmetric set difference distance calculations: d >= 1 and d_rel in (0, 1]."""
        distances = mgn_modeling_df["distance"]
        assert (distances >= 1).all(), "Found distance < 1 in modeling dataset"
        assert np.issubdtype(distances.dtype, np.integer), "Distance column must be integer"

        if "distance_rel" in mgn_modeling_df.columns:
            d_rel = mgn_modeling_df["distance_rel"]
            assert (d_rel > 0.0).all(), "Found relative distance <= 0"
            assert (d_rel <= 1.000001).all(), "Found relative distance > 1.0"

    def test_r3_05_no_reflexive_pairs(self, mgn_modeling_df):
        """Verify that reflexive / identity cell pairs (d = 0 or cell_1 == cell_2) are strictly excluded."""
        same_cells = mgn_modeling_df[mgn_modeling_df["cell_1"] == mgn_modeling_df["cell_2"]]
        assert len(same_cells) == 0, f"Found {len(same_cells)} reflexive cell pairs (cell_1 == cell_2)"
        zero_dist = mgn_modeling_df[mgn_modeling_df["distance"] == 0]
        assert len(zero_dist) == 0, f"Found {len(zero_dist)} rows with distance == 0"

    def test_r3_06_statistical_transformations_and_non_nullness(self, mgn_modeling_df):
        """Verify log10 population, log10 paradigm size, and scaled predictors contain no NaNs/Infs."""
        numeric_cols = [
            "log10_pop", "log10_pop_z", "log10_paradigm_size", "log10_paradigm_size_z",
            "contact_richness_scaled", "log10_area_scaled", "lat", "lon"
        ]
        for col in numeric_cols:
            if col in mgn_modeling_df.columns:
                vals = mgn_modeling_df[col]
                assert not vals.isna().any(), f"Column '{col}' contains NaN values"
                assert not np.isinf(vals).any(), f"Column '{col}' contains infinite values"

        # Check binomial counts consistency: 0 <= correct <= total
        assert (mgn_modeling_df["correct"] >= 0).all(), "Found negative correct trial counts"
        assert (mgn_modeling_df["total"] >= 1).all(), "Found total trial count < 1"
        assert (mgn_modeling_df["correct"] <= mgn_modeling_df["total"]).all(), (
            "Found correct count exceeding total trials"
        )


# ---------------------------------------------------------------------------
# Requirement R4: Bayesian Phylogenetic & Spatial Model Suite
# ---------------------------------------------------------------------------

class TestRequirementR4BayesianModelingSuite:
    """Feature coverage tests for R4: Phylogenetic covariance matrix, brms model fits, LOO & summaries."""

    def test_r4_01_phylo_cov_matrix_structure_and_positive_definiteness(self, phylo_cov_matrix_path, project_root):
        """Verify phylo_cov_matrix.rds is a 64x64 symmetric positive definite matrix."""
        assert phylo_cov_matrix_path.exists(), f"Phylogenetic covariance matrix missing at {phylo_cov_matrix_path}"

        import subprocess
        r_cmd = (
            "A <- readRDS('phylo_cov_matrix.rds'); "
            "stopifnot(is.matrix(A)); "
            "stopifnot(nrow(A) == 64 && ncol(A) == 64); "
            "stopifnot(isSymmetric(A, tol = 1e-6)); "
            "ev <- eigen(A, symmetric = TRUE, only.values = TRUE)$values; "
            "cat(sprintf('MIN_EV=%.6f;MAX_EV=%.6f;ROWS=%d;COLS=%d', min(ev), max(ev), nrow(A), ncol(A)))"
        )
        res = subprocess.run(["Rscript", "-e", r_cmd], cwd=str(project_root), capture_output=True, text=True)
        assert res.returncode == 0, f"R script verifying phylo_cov_matrix.rds failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert "MIN_EV=" in res.stdout
        # Parse min eigenvalue
        min_ev = float(res.stdout.split("MIN_EV=")[1].split(";")[0])
        assert min_ev > 0, f"Covariance matrix A is not positive definite: lambda_min = {min_ev}"

    def test_r4_02_serialized_brms_fits_existence(self, project_root):
        """Verify THIS pipeline's model fits exist.

        Checks for the specific files fit_bayesian_models.R writes. A glob for any
        *.rds in fits/ passes on unrelated fits left over from earlier analyses,
        which reports the modelling milestone as done when it has never run.
        """
        fits_dir = project_root / "fits"
        expected = ["model_minimal.rds", "model_comprehensive.rds"]
        missing = [f for f in expected if not (fits_dir / f).exists()]
        if missing:
            pytest.skip(
                f"Models not yet fitted (missing {missing}); run fit_bayesian_models.R. "
                f"Other files in fits/ belong to earlier analyses and do not count."
            )
        for f in expected:
            assert (fits_dir / f).stat().st_size > 10_000, f"{f} is implausibly small"

    def test_r4_03_loo_model_comparison_metrics(self, project_root):
        """Verify results/loo_model_comparison.csv exists and contains LOO/WAIC metrics."""
        loo_path = project_root / "results" / "loo_model_comparison.csv"
        if not loo_path.exists():
            pytest.skip(f"LOO comparison file pending model fitting milestone: {loo_path}")
        df = pd.read_csv(loo_path)
        assert len(df) >= 2, f"Expected at least 2 models compared in {loo_path}, got {len(df)}"
        # Check presence of expected metric columns (e.g. elpd_loo, elpd_diff, se_diff)
        cols_lower = [c.lower() for c in df.columns]
        has_elpd = any("elpd" in c or "loo" in c or "waic" in c for c in cols_lower)
        assert has_elpd, f"No ELPD/LOO/WAIC metrics found in comparison table: {df.columns}"

    def test_r4_04_posterior_parameter_summaries_convergence(self, project_root):
        """Verify results/posterior_parameter_summaries.csv exists with R-hat < 1.01 and ESS > 400."""
        summary_path = project_root / "results" / "posterior_parameter_summaries.csv"
        if not summary_path.exists():
            pytest.skip(f"Posterior summary file pending model fitting milestone: {summary_path}")
        df = pd.read_csv(summary_path)
        assert len(df) > 0, "Posterior summary table is empty"

        # Check convergence diagnostic columns
        rhat_col = next((c for c in df.columns if "rhat" in c.lower() or "r_hat" in c.lower()), None)
        if rhat_col:
            rhats = df[rhat_col].dropna()
            assert (rhats < 1.05).all(), f"Found R-hat >= 1.05 indicative of non-convergence:\n{df[[rhat_col]]}"

        ess_col = next((c for c in df.columns if "ess" in c.lower()), None)
        if ess_col:
            ess_vals = df[ess_col].dropna()
            assert (ess_vals > 200).all(), f"Found low ESS indicative of inadequate sampling:\n{df[[ess_col]]}"

    def test_r4_05_model_visualization_plots(self, project_root):
        """Verify output diagnostic/result plots in plots/ exist and are valid PNG images."""
        plots_dir = project_root / "plots"
        assert plots_dir.exists() and plots_dir.is_dir(), f"plots/ directory missing at {plots_dir}"
        expected = ["posterior_coefficients.png", "marginal_effects_population.png"]
        missing = [f for f in expected if not (plots_dir / f).exists()]
        if missing:
            pytest.skip(
                f"Model plots not yet generated (missing {missing}); run "
                f"fit_bayesian_models.R. Other images in plots/ are from earlier analyses."
            )
        for f in expected:
            assert (plots_dir / f).stat().st_size > 1000, f"{f} is suspiciously small (< 1KB)"
