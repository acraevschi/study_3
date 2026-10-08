"""Shared fixture config for the typology tests (no network, fixture CLDF files only)."""

import copy
from pathlib import Path

import yaml

from morph_ldl.config import PIPELINE_ROOT, config_hash

FIX = Path(__file__).parent / "fixtures" / "typology"
DEFAULT_BLOCK = PIPELINE_ROOT / "morph_ldl" / "typology" / "default_config.yaml"


def fixture_typology() -> dict:
    return {
        "sources": {
            "grambank": {"path": str(FIX / "grambank"), "repo": "fixture", "tag": "v1.0.3",
                         "commit": "7ae000cf740f93cdb3e4ec67010668d6795337a9"},
            "glottolog": {"path": str(FIX / "glottolog-cldf"), "repo": "fixture", "tag": "v5.3",
                          "commit": "072ca0d0410039fb8b779be8fc165bac575d2cda"},
            "gelato_main_populations": {"path": str(FIX / "gelato" / "populations.csv"), "version": "fixture"},
            "gelato_tableS1": {"path": str(FIX / "gelato" / "tableS1.csv"), "version": "fixture"},
        },
        "require_git_pin": False,
        "review_file": str(FIX / "review.yaml"),
        "manual_links": [],
        "ldl_eligibility_audit": None,
        "feature_set": {
            "id": "fixture_5",
            "n_features": 5,
            "domains": {"verbal_tam": ["GB080", "GB082"], "nominal_number": ["GB044"],
                        "case": ["GB070"], "agreement": ["GB170"]},
            "excluded": {"other": ["GB020"]},
            "present_codes": {"default": ["1"], "per_feature": {}},
        },
        "sensitivity_sets": {"verbal": ["verbal_tam"], "nominal": ["nominal_number", "case"],
                             "no_agreement": ["verbal_tam", "nominal_number", "case"]},
        "coverage_threshold": 0.6,
        "report_coverage_thresholds": [0.5, 0.75],
        "minimal_inflection_max": 1,
        "clitic_heuristic": {"analytic_families": ["fam1"], "analytic_family_min_present": 3,
                             "profile_min_present": 1, "profile_zero_domains": ["nominal_number", "agreement"],
                             "profile_min_domain_coverage": 0.5},
    }


def fixture_cfg(tmp_path: Path, typology: dict = None, units=None) -> dict:
    cfg = {
        "experiment": {"id": "typology_test", "master_seed": 1},
        "paths": {"repo_root": str(PIPELINE_ROOT), "outputs": str(tmp_path / "outputs")},
        "resources": {"ingest": []},
        "units": units or [],
        "typology": typology if typology is not None else fixture_typology(),
    }
    cfg = copy.deepcopy(cfg)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def default_block() -> dict:
    return yaml.safe_load(DEFAULT_BLOCK.read_text(encoding="utf-8"))["typology"]
