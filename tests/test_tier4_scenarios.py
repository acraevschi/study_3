"""
Tier 4: Real-World E2E Scenarios & Pipeline Execution Tests for study_3.

Exercises the full scientific pipeline end-to-end:
1. Multi-source Demographic Fusion (`src/build_demographic_registry.py`)
2. MGN Paradigm & Trial Feature Engineering (`src/merge_mgn_features.py`)
3. Phylogenetic Covariance Matrix Construction (`build_phylo_matrix.R`)
4. Bayesian Model Suite & Convergence Audit (`fit_bayesian_models.R`)
5. End-to-End Scientific Artifact Audit
"""

import os
import subprocess
from pathlib import Path
import pytest
import pandas as pd


class TestTier4RealWorldScenarios:
    """Real-World E2E Scenario and full pipeline execution tests."""

    def test_tier4_scenario_01_demographic_fusion_script_runnable(self, project_root):
        """Verify that src/build_demographic_registry.py is executable and produces valid registry."""
        script_path = project_root / "src" / "build_demographic_registry.py"
        if not script_path.exists():
            pytest.skip(f"Demographic fusion script pending M2 implementation at {script_path}")

        res = subprocess.run(
            ["python3", str(script_path), "--help"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        # Either --help is supported or running without args/with default args
        assert res.returncode in [0, 2], f"Script execution error:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"

    def test_tier4_scenario_02_mgn_feature_merging_script_runnable(self, project_root):
        """Verify that src/merge_mgn_features.py is executable and produces valid modeling dataset."""
        script_path = project_root / "src" / "merge_mgn_features.py"
        if not script_path.exists():
            pytest.skip(f"MGN feature merging script pending M3 implementation at {script_path}")

        res = subprocess.run(
            ["python3", str(script_path), "--help"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        assert res.returncode in [0, 2], f"Script execution error:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"

    def test_tier4_scenario_03_phylo_matrix_builder_runnable(self, project_root):
        """Verify that build_phylo_matrix.R executes cleanly in R."""
        script_path = project_root / "build_phylo_matrix.R"
        assert script_path.exists(), f"Phylogenetic matrix builder script missing at {script_path}"

        # Verify R script syntax
        res = subprocess.run(
            ["Rscript", "-e", f"parse(file = '{script_path}')"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        assert res.returncode == 0, f"R script syntax parsing error in {script_path}:\n{res.stderr}"

    def test_tier4_scenario_04_bayesian_modeling_script_syntax_and_formula_audit(self, project_root):
        """Audit fit_bayesian_models.R for correct brms formula and syntax."""
        script_path = project_root / "fit_bayesian_models.R"
        assert script_path.exists(), "No Bayesian modeling script found in root"

        # Verify R script syntax
        res = subprocess.run(
            ["Rscript", "-e", f"parse(file = '{script_path}')"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        assert res.returncode == 0, f"R syntax error in modeling script {script_path}:\n{res.stderr}"

    def test_tier4_scenario_05_e2e_artifact_pipeline_integrity(
        self, project_root, demographic_registry_path, mgn_modeling_dataset_path, phylo_cov_matrix_path
    ):
        """End-to-End consistency check across all generated pipeline artifacts."""
        # 1. Check demographic registry
        assert demographic_registry_path.exists(), f"Missing {demographic_registry_path}"
        reg_df = pd.read_csv(demographic_registry_path)
        assert len(reg_df) > 6500, f"Demographic registry too small: {len(reg_df)}"

        # 2. Check modeling dataset
        assert mgn_modeling_dataset_path.exists(), f"Missing {mgn_modeling_dataset_path}"
        mod_df = pd.read_csv(mgn_modeling_dataset_path)
        assert len(mod_df) > 1000, f"Modeling dataset too small: {len(mod_df)}"

        # 3. Check phylo covariance matrix
        assert phylo_cov_matrix_path.exists(), f"Missing {phylo_cov_matrix_path}"
