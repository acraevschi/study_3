#!/usr/bin/env python3
"""
Feature engineering and dataset assembly for MGN cell-pair prediction trials.

Merges Guzman Naranjo's (2024) morphological prediction trials with
`global_demographic_registry.csv` and writes `mgn_modeling_dataset.csv`.

Design notes
------------
Training-set size (`num`)
    MGN reports each cell pair at several training-set sizes (200/500/1000/2000/5000).
    Accuracy rises with `num`, so the size must be held constant across languages or
    the complexity measure is confounded with corpus availability. We therefore
    prefer `num == 200` (the paper shows estimates are already stable there, p.1799).
    Two datasets have no 200 condition -- Navajo verbs (500/1000/2000) and Belarusian
    verbs (500) -- so instead of dropping them we fall back to the SMALLEST available
    size for that (language, POS) and record it in `num_used`. Navajo is one of the
    two headline high-complexity languages in the paper and must not be lost to a
    sampling artefact.

POS tag normalisation
    MGN cell labels come in two conventions: dotted MGN-internal labels that
    `cells_to_unimorph.json` expands into a bundle carrying an explicit POS feature
    (`imp.act.f.2.s` -> `V;IMP;ACT;2;SG;FEM`), and labels that are already UniMorph
    bundles with no POS feature (`1;IND;PL;PRF;PRS`). Comparing one against the other
    used to add a spurious +1 to the symmetric-difference distance. Bare POS features
    (V, N, ADJ, ADV) are stripped before the distance is computed -- POS is already a
    column. Sub-POS features that carry real morphosyntax (V.PTCP, V.CVB, V.MSDR) are
    kept.

Chance level
    Raw accuracy is not comparable across cell pairs: a pair with two inflection
    classes has a 0.5 chance baseline, a pair with twenty has 0.05. MGN's `nvar`
    (mean number of variants) tracks this and correlates -0.32 with accuracy, so it
    is carried through as `log10_nvar` for use as a model control. `nph`, `nmarkers`
    and `n_pairs` are carried through as well.

Standardisation
    z-scores for language-level covariates are computed over the LANGUAGES, not over
    the trials. Trial counts per language range from ~6 to ~12,000, so trial-level
    standardisation would silently weight the centring by paradigm size.

Nothing here imputes a missing covariate. A language whose demographic record lacks
a required field raises rather than falling back to a magic constant.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from cell_normalization import normalize_cell
from mgn_language_map import (
    MGN_EXTINCT_LANGS_9,
    MGN_LIVING_LANGS_64,
    canonical_iso,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREFERRED_NUM = 200

# Bare POS features stripped before distance computation (redundant with `pos`).
# Sub-POS tags such as V.PTCP / V.CVB / V.MSDR are morphosyntactically contentful
# and are deliberately NOT in this set.
BARE_POS_FEATURES: Set[str] = {"V", "N", "ADJ", "ADV", "PROPN", "AUX"}

# Demographic fields that must be present; no defaults, no silent imputation.
REQUIRED_DEMO_FIELDS = [
    "population_l1", "latitude", "longitude",
    "div_bordering_languages", "area_km2", "altitude_range", "roughness",
]

MGN_MODELING_COLS = [
    # identity
    "lang", "iso_sanitized", "glottocode", "language_name", "family", "macro_area",
    # trial
    "pos", "num_used", "cell_1", "cell_2", "unimorph_1", "unimorph_2",
    "distance", "distance_rel", "correct", "total",
    # MGN paradigm structure / chance level
    "nvar", "nph", "nmarkers", "n_pairs", "log10_nvar", "log10_nvar_z",
    # demography
    "population_l1", "population_source", "l2_proportion", "vehicularity",
    "log10_pop", "log10_pop_z",
    # paradigm size
    "paradigm_size", "log10_paradigm_size", "log10_paradigm_size_z",
    # macro-ecology
    "eco_imputed", "contact_richness_scaled", "log10_area_scaled",
    "altitude_range_scaled", "roughness_scaled",
    # geography
    "lat", "lon",
]


def parse_unimorph_features(cell: str, cell_map: Dict[str, str]) -> Optional[Set[str]]:
    """Normalise a cell label to a UniMorph feature set, minus bare POS features.

    Returns None if the label cannot be parsed under any known convention; see
    src/cell_normalization.py.
    """
    feats = normalize_cell(cell, cell_map)
    if feats is None:
        return None
    feats = feats - BARE_POS_FEATURES
    return feats or None


def sym_diff_dist(set1: Set[str], set2: Set[str]) -> int:
    """Symmetric set difference distance |A \\ B| + |B \\ A|."""
    return len(set1.symmetric_difference(set2))


def load_demographic_registry(registry_path: Path) -> Dict[str, dict]:
    """Load global_demographic_registry.csv indexed by iso_639_3."""
    if not registry_path.exists():
        raise FileNotFoundError(f"Demographic registry missing at {registry_path}")
    reg_df = pd.read_csv(registry_path)
    out = {}
    for _, row in reg_df.iterrows():
        iso = str(row["iso_639_3"]).strip().lower()
        if iso and iso != "nan":
            out[iso] = row.to_dict()
    return out


def load_unimorph_mapping(mapping_path: Path) -> Dict[str, str]:
    """Load cells_to_unimorph.json."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Cell tag mapping file missing at {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_num_per_group(
    df: pd.DataFrame, pos: str, preferred: int = PREFERRED_NUM
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Keep one training-set size per language: `preferred`, else the smallest available.

    Returns the filtered frame and {lang: num_used}.
    """
    chosen: Dict[str, int] = {}
    keep_idx: List[pd.Index] = []
    for lang, grp in df.groupby("lang", sort=True):
        available = sorted(grp["num"].unique())
        num = preferred if preferred in available else available[0]
        chosen[str(lang)] = int(num)
        keep_idx.append(grp.index[grp["num"] == num])
        if num != preferred:
            print(
                f"  [num fallback] {lang}/{pos}: no num=={preferred}; "
                f"using num=={num} (available: {available})"
            )
    if not keep_idx:
        return df.iloc[0:0].copy(), chosen
    out = df.loc[np.concatenate([i.values for i in keep_idx])].copy()
    out["num_used"] = out["lang"].map(chosen)
    return out, chosen


def _zscore(series: pd.Series) -> pd.Series:
    sd = series.std()
    return (series - series.mean()) / (sd if sd and sd > 0 else 1.0)


def build_mgn_modeling_dataset(
    output_path: Optional[Path] = None,
    num_sample_size: int = PREFERRED_NUM,
    registry_path: Optional[Path] = None,
    mapping_path: Optional[Path] = None,
    mgn_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Ingest MGN cell pairs, compute distances, merge demography, export dataset."""
    output_path = output_path or PROJECT_ROOT / "mgn_modeling_dataset.csv"
    registry_path = registry_path or PROJECT_ROOT / "global_demographic_registry.csv"
    mapping_path = mapping_path or PROJECT_ROOT / "data_sources" / "cells_to_unimorph.json"
    mgn_dir = mgn_dir or PROJECT_ROOT / "mgn_data" / "results-final"

    print(f"Loading demographic registry from {registry_path}...")
    reg_by_iso = load_demographic_registry(registry_path)
    print(f"Loading UniMorph cell tag mappings from {mapping_path}...")
    cell_map = load_unimorph_mapping(mapping_path)

    # The verb file is read from its gzip: the plain .csv is 48 MB and the .gz is
    # 5 MB with byte-identical contents (verified by md5), so only the .gz is kept
    # under version control. pandas decompresses by extension. The noun file has
    # no such equivalence -- the shipped .csv and .csv.gz differ in float
    # precision -- so the .csv is used and committed as-is.
    pos_file_configs = [
        ("v", ["compressed-lang-pairs-v.csv.gz", "compressed-lang-pairs-v.csv"]),
        ("n", ["compressed-lang-pairs-n.csv"]),
        ("adj", ["compressed-lang-pairs-adj.csv.gz"]),
    ]

    all_pairs = []
    for pos, candidates in pos_file_configs:
        fpath = next((mgn_dir / c for c in candidates if (mgn_dir / c).exists()), None)
        if fpath is None:
            print(f"Warning: none of {candidates} found in {mgn_dir}, skipping POS '{pos}'")
            continue
        fname = fpath.name
        print(f"Reading {fname} (POS: {pos})...")
        df = pd.read_csv(fpath)
        df = df[df["lang"].astype(str).str.lower().isin(MGN_LIVING_LANGS_64)]
        df, _ = select_num_per_group(df, pos, num_sample_size)
        df["pos"] = pos
        all_pairs.append(df)

    if not all_pairs:
        raise RuntimeError("No MGN trial files found to process.")

    combined_df = pd.concat(all_pairs, ignore_index=True)
    print(f"Total raw trials after training-size selection: {len(combined_df)}")

    stats = {"reflexive": 0, "zero_distance": 0, "unparseable": 0,
             "bad_counts": 0, "no_demographics": 0}
    unparseable_cells: Counter = Counter()
    missing_demo: Set[str] = set()
    rows = []

    for _, row in combined_df.iterrows():
        lang = str(row["lang"]).strip().lower()
        if lang in MGN_EXTINCT_LANGS_9 or lang not in MGN_LIVING_LANGS_64:
            continue

        cell_1 = str(row["cell_1"]).strip()
        cell_2 = str(row["cell_2"]).strip()

        if cell_1 == cell_2:
            stats["reflexive"] += 1
            continue

        f1 = parse_unimorph_features(cell_1, cell_map)
        f2 = parse_unimorph_features(cell_2, cell_map)
        if f1 is None or f2 is None:
            stats["unparseable"] += 1
            for c, f in ((cell_1, f1), (cell_2, f2)):
                if f is None:
                    unparseable_cells[c] += 1
            continue

        dist = sym_diff_dist(f1, f2)
        if dist == 0:
            # Distinct labels, identical normalised feature bundles: no
            # morphosyntactic contrast to measure.
            stats["zero_distance"] += 1
            continue

        correct, total = int(row["correct"]), int(row["total"])
        if total <= 0 or correct < 0 or correct > total:
            stats["bad_counts"] += 1
            continue

        iso = canonical_iso(lang)
        demo = reg_by_iso.get(iso)
        if not demo:
            stats["no_demographics"] += 1
            missing_demo.add(f"{lang}->{iso}")
            continue
        for field in REQUIRED_DEMO_FIELDS:
            if pd.isna(demo.get(field)):
                raise ValueError(
                    f"Language {lang} ({iso}) has no {field} in the registry. "
                    "Fix the registry rather than imputing here."
                )

        rows.append({
            "lang": lang,
            "iso_sanitized": iso,
            "glottocode": str(demo.get("glottocode", "")),
            "language_name": demo.get("language_name", ""),
            "family": demo.get("family", ""),
            "macro_area": demo.get("macro_area", ""),
            "pos": row["pos"],
            "num_used": int(row["num_used"]),
            "cell_1": cell_1,
            "cell_2": cell_2,
            "unimorph_1": ";".join(sorted(f1)),
            "unimorph_2": ";".join(sorted(f2)),
            "distance": int(dist),
            "correct": correct,
            "total": total,
            "nvar": float(row["nvar"]),
            "nph": float(row["nph"]),
            "nmarkers": float(row["nmarkers"]),
            "n_pairs": float(row["n_pairs"]),
            "population_l1": float(demo["population_l1"]),
            "population_source": demo.get("population_source", ""),
            "l2_proportion": demo.get("l2_proportion", np.nan),
            "vehicularity": int(demo.get("vehicularity", 0)),
            "eco_imputed": bool(demo.get("eco_imputed", True)),
            "div_bordering_languages": float(demo["div_bordering_languages"]),
            "area_km2": float(demo["area_km2"]),
            "altitude_range": float(demo["altitude_range"]),
            "roughness": float(demo["roughness"]),
            "lat": float(demo["latitude"]),
            "lon": float(demo["longitude"]),
        })

    out_df = pd.DataFrame(rows)
    print("\nFiltering summary:")
    for k, v in stats.items():
        print(f"  {k:18} {v}")
    if unparseable_cells:
        print(f"  unparseable cell labels ({len(unparseable_cells)} distinct): "
              f"{[c for c, _ in unparseable_cells.most_common(12)]}")
    if missing_demo:
        print(f"  languages lacking demographics: {sorted(missing_demo)}")
    print(f"\nValid trials: {len(out_df)} across "
          f"{out_df['iso_sanitized'].nunique()} languages")

    # -- paradigm-level features ------------------------------------------
    grouped = out_df.groupby(["iso_sanitized", "pos"])

    # Paradigm size: distinct cells appearing on either side of a pair.
    sizes = (
        out_df.groupby(["iso_sanitized", "pos"])[["cell_1", "cell_2"]]
        .apply(lambda g: len(set(g["cell_1"]) | set(g["cell_2"])))
        .rename("paradigm_size")
    )
    out_df = out_df.merge(sizes, on=["iso_sanitized", "pos"], how="left")
    out_df["log10_paradigm_size"] = np.log10(out_df["paradigm_size"].clip(lower=1))

    # Relative distance within a paradigm.
    max_dists = grouped["distance"].transform("max")
    out_df["distance_rel"] = (out_df["distance"] / max_dists).clip(1e-6, 1.0)

    # -- transformations ---------------------------------------------------
    out_df["log10_pop"] = np.log10(out_df["population_l1"].clip(lower=1.0))
    out_df["log10_nvar"] = np.log10(out_df["nvar"].clip(lower=1.0))
    log_area = np.log10(out_df["area_km2"].clip(lower=1.0))

    # Language-level standardisation: derive the moments from one row per
    # language so that trial counts do not weight the centring.
    lang_level = out_df.drop_duplicates("iso_sanitized").set_index("iso_sanitized")

    def lang_scale(col: pd.Series, key: str) -> pd.Series:
        ref = lang_level[key] if key in lang_level.columns else None
        mu, sd = (ref.mean(), ref.std()) if ref is not None else (col.mean(), col.std())
        return (col - mu) / (sd if sd and sd > 0 else 1.0)

    out_df["log10_pop_z"] = lang_scale(out_df["log10_pop"], "log10_pop")
    out_df["contact_richness_scaled"] = lang_scale(
        out_df["div_bordering_languages"], "div_bordering_languages")
    out_df["altitude_range_scaled"] = lang_scale(out_df["altitude_range"], "altitude_range")
    out_df["roughness_scaled"] = lang_scale(out_df["roughness"], "roughness")

    lang_level_area = np.log10(lang_level["area_km2"].clip(lower=1.0))
    out_df["log10_area_scaled"] = (
        (log_area - lang_level_area.mean())
        / (lang_level_area.std() if lang_level_area.std() > 0 else 1.0)
    )

    # Trial-level covariates keep trial-level standardisation.
    out_df["log10_paradigm_size_z"] = _zscore(out_df["log10_paradigm_size"])
    out_df["log10_nvar_z"] = _zscore(out_df["log10_nvar"])

    final_df = out_df[MGN_MODELING_COLS].copy()
    for c in ["distance", "correct", "total", "paradigm_size", "num_used", "vehicularity"]:
        final_df[c] = final_df[c].astype(int)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(final_df)} rows x {len(final_df.columns)} cols -> {output_path}")

    covered = set(final_df["lang"].unique())
    missing = sorted(MGN_LIVING_LANGS_64 - covered)
    print(f"MGN language coverage: {len(covered)}/{len(MGN_LIVING_LANGS_64)}"
          + (f"  MISSING: {missing}" if missing else "  (complete)"))
    return final_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge MGN prediction trials with demographic covariates.")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "mgn_modeling_dataset.csv"))
    parser.add_argument("--num", type=int, default=PREFERRED_NUM,
                        help=f"Preferred training-set size (default: {PREFERRED_NUM}); "
                             "languages lacking it fall back to their smallest.")
    parser.add_argument("--registry", type=str,
                        default=str(PROJECT_ROOT / "global_demographic_registry.csv"))
    parser.add_argument("--mapping", type=str,
                        default=str(PROJECT_ROOT / "data_sources" / "cells_to_unimorph.json"))
    parser.add_argument("--mgn-dir", type=str,
                        default=str(PROJECT_ROOT / "mgn_data" / "results-final"))
    args = parser.parse_args()

    build_mgn_modeling_dataset(
        output_path=Path(args.output),
        num_sample_size=args.num,
        registry_path=Path(args.registry),
        mapping_path=Path(args.mapping),
        mgn_dir=Path(args.mgn_dir),
    )


if __name__ == "__main__":
    main()
