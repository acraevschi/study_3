"""
Shared fixtures, configuration, constants, and helper utilities for study_3 E2E test suite.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_ROOT = Path("/Users/acraev/Work/study_3_backup").resolve()

# The language inventory and MGN -> ISO mapping live in exactly one place.
# Tests import it rather than restating it, so a test can never silently
# validate against a mapping the pipeline no longer uses. These names are
# re-exported for the test modules; __all__ keeps linters from stripping them.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from mgn_language_map import (  # noqa: E402
    MGN_LIVING_LANGS_64,
    MGN_EXTINCT_LANGS_9,
    MGN_TO_ISO as ISO_SANITIZED_MAP,
    canonical_iso,
)

__all__ = [
    "PROJECT_ROOT",
    "BACKUP_ROOT",
    "MGN_LIVING_LANGS_64",
    "MGN_EXTINCT_LANGS_9",
    "ISO_SANITIZED_MAP",
    "canonical_iso",
    "DEMOGRAPHIC_REGISTRY_REQUIRED_COLS",
    "MGN_MODELING_REQUIRED_COLS",
]

# Expected schema for global_demographic_registry.csv
DEMOGRAPHIC_REGISTRY_REQUIRED_COLS = [
    "iso_639_3",
    "glottocode",
    "language_name",
    "family",
    "macro_area",
    "population_l1",
    "population_source",
    "population_scope",
    "l2_proportion",
    "vehicularity",
    "div_bordering_languages",
    "area_km2",
    "altitude_range",
    "roughness",
    "eco_imputed",
    "latitude",
    "longitude",
]

# Expected schema for mgn_modeling_dataset.csv
MGN_MODELING_REQUIRED_COLS = [
    "lang",
    "iso_sanitized",
    "glottocode",
    "language_name",
    "family",
    "macro_area",
    "pos",
    "num_used",
    "cell_1",
    "cell_2",
    "unimorph_1",
    "unimorph_2",
    "distance",
    "distance_rel",
    "correct",
    "total",
    "nvar",
    "nph",
    "nmarkers",
    "n_pairs",
    "log10_nvar",
    "log10_nvar_z",
    "population_l1",
    "population_source",
    "l2_proportion",
    "vehicularity",
    "log10_pop",
    "log10_pop_z",
    "paradigm_size",
    "log10_paradigm_size",
    "log10_paradigm_size_z",
    "eco_imputed",
    "contact_richness_scaled",
    "log10_area_scaled",
    "altitude_range_scaled",
    "roughness_scaled",
    "lat",
    "lon"
]

@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def backup_root() -> Path:
    return BACKUP_ROOT

@pytest.fixture(scope="session")
def demographic_registry_path() -> Path:
    return PROJECT_ROOT / "global_demographic_registry.csv"

@pytest.fixture(scope="session")
def mgn_modeling_dataset_path() -> Path:
    return PROJECT_ROOT / "mgn_modeling_dataset.csv"

@pytest.fixture(scope="session")
def phylo_cov_matrix_path() -> Path:
    return PROJECT_ROOT / "phylo_cov_matrix.rds"

@pytest.fixture(scope="session")
def cells_to_unimorph_path() -> Path:
    return PROJECT_ROOT / "data_sources" / "cells_to_unimorph.json"

@pytest.fixture(scope="session")
def demographic_registry_df(demographic_registry_path):
    if not demographic_registry_path.exists():
        pytest.skip(f"Demographic registry not found at {demographic_registry_path}")
    return pd.read_csv(demographic_registry_path)

@pytest.fixture(scope="session")
def mgn_modeling_df(mgn_modeling_dataset_path):
    if not mgn_modeling_dataset_path.exists():
        pytest.skip(f"MGN modeling dataset not found at {mgn_modeling_dataset_path}")
    return pd.read_csv(mgn_modeling_dataset_path)
