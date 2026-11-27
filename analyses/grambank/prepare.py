"""Unified data preparation for Track 1: GeLaTo Admixture -> Grambank 5-Domain Morphology.

Calls `analyses.common` (`genetics`, `demography`, `phylogeny`) to write all canonical
Grambank data tables into `analyses/grambank/data/`:
  - `language_data.csv`          (All 225 GeLaTo-linked Grambank languages with 5-domain
                                  counts, raw unmasked irregularity counts, non-morphological
                                  documentation coverage, demography, and legacy counts)
  - `populations.csv`            (364 main-set populations linked to Grambank languages)
  - `individuals.csv`            (All individuals belonging to those 364 populations)
  - `phylo_tree.nwk`             (Glottolog topology Newick for all 225 languages)
  - `prep_summary.json`          (Dataset, 5-domain eligibility, and demography summary)
  - `items_metadata.csv`         (29 clean items across v_breadth, n_breadth, v_irreg, n_irreg, np_agr)
  - `item_observations.csv`      (Observed 0/1 cells across the 225 languages; domain-inapplicable
                                  irregularity cells are NA and omitted)
  - `language_table.csv`         (225 languages with grambank_entry, 5-domain counts, and coverage)
  - `latent_prep_summary.json`   (Item-level & domain-level descriptive summary)

Also ensures `analyses/common/data/gelato_populations_summary.csv` (all 558 unique
GeLaTo ADMIXTURE populations) is written.

Run:
  .venv/bin/python analyses/grambank/prepare.py
  # or inside a stdlib-only sandbox:
  /usr/bin/python3 analyses/grambank/prepare.py
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyses.common.demography import (  # noqa: E402
    DEMOG_COLS,
    GROUP_ISO_MAP,
    PLURICENTRIC_AREA_MEMBERS,
    resolve_demography_rows,
)
from analyses.common.genetics import (  # noqa: E402
    EXCLUDE_MORPH_IDS,
    F3_FIXED_CSV,
    KS,
    N_BOOT,
    SEED,
    build_gelato_populations_summary,
    compute_grambank_nonmorph_coverage,
    compute_individual_measures,
    compute_population_measures,
    load_admixture_q,
    load_f3_fixed_scores,
    write_rows_csv,
)
from analyses.common.phylogeny import (  # noqa: E402
    build_newick,
    resolve_glottolog_paths,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
COMMON_DATA_DIR = ROOT / "analyses/common/data"
TYP_DIR = ROOT / "outputs/pcfp_v2_links_2026_10_20/typology"
GB_DIR = ROOT / "external/grambank/cldf"
GLOT_DIR = ROOT / "external/glottolog-cldf/cldf"

# Legacy 12 merged inflectional categories (from morph_ldl/typology/default_config.yaml),
# used to resolve dialect substitutes and retain legacy n_present/n_coded columns directly
# from tracked Grambank and Glottolog CLDF tables (no dependency on local outputs/).
LEGACY_CAT12_SOURCES: dict[str, tuple[str, ...]] = {
    "tense": ("GB082", "GB083", "GB084"),
    "aspect": ("GB086",),
    "mood": ("GB312",),
    "person_indexing": ("GB089", "GB090", "GB091", "GB092", "GB093", "GB094"),
    "negation": ("GB107",),
    "polar_interrogation": ("GB285", "GB286"),
    "nominal_number": ("GB042", "GB043", "GB044", "GB165", "GB166"),
    "case": ("GB070", "GB071", "GB072", "GB073"),
    "possessor_affix": ("GB430", "GB432"),
    "possessed_affix": ("GB431", "GB433"),
    "gender_agreement": ("GB170", "GB171", "GB172", "GB198"),
    "number_agreement": ("GB184", "GB185", "GB186"),
}

SETS = ["main", "no_doubtful_language", "no_other_language_stand_ins", "strict"]

# Canonical 5-domain Track-1 feature batteries (29 items from 36 outcome IDs + 8 prereq IDs = 44 raw IDs)
# Each tuple: (item_id, item_group, domain, testlet, sources, agg_rule, prereq_key, description)
DOMAIN_ITEM_SPECS: list[tuple[str, str, str, str, tuple[str, ...], str, str | None, str]] = [
    # A. Verbal Enumerative Breadth (v_breadth, k = 9, eligible >= 6)
    ("present_tense", "v_breadth", "verbal_tense", "verbal_tense", ("GB082",), "exact", None,
     "Overt morphological present tense on verbs (GB082)"),
    ("past_tense", "v_breadth", "verbal_tense", "verbal_tense", ("GB083",), "exact", None,
     "Dedicated morphological past tense on verbs (GB083)"),
    ("future_tense", "v_breadth", "verbal_tense", "verbal_tense", ("GB084",), "exact", None,
     "Dedicated morphological future tense on verbs (GB084)"),
    ("aspect", "v_breadth", "verbal_aspect", "none", ("GB086",), "exact", None,
     "Morphological perfective/imperfective aspect on verbs (GB086)"),
    ("mood", "v_breadth", "verbal_mood", "none", ("GB312",), "exact", None,
     "Dedicated morphological mood marking on verbs (GB312)"),
    ("subj_indexing", "v_breadth", "verbal_subj_idx", "none", ("GB089", "GB090", "GB091", "GB092"), "or_exact", None,
     "S or A prefix/suffix indexing on verbs (OR_exact GB089..GB092)"),
    ("obj_indexing", "v_breadth", "verbal_obj_idx", "none", ("GB093", "GB094"), "or_exact", None,
     "P prefix/suffix indexing on verbs (OR_exact GB093, GB094)"),
    ("negation", "v_breadth", "verbal_neg", "none", ("GB107",), "exact", None,
     "Bound verbal negation affix, clitic, or stem modification (GB107)"),
    ("polar_q_bound", "v_breadth", "verbal_polar_q", "none", ("GB286",), "exact", None,
     "Polar interrogation via overt verbal morphology only (GB286)"),
    # B. Nominal Enumerative Breadth (n_breadth, k = 5, eligible >= 3)
    ("noun_plural", "n_breadth", "noun_plural", "none", ("GB044",), "exact", None,
     "Productive morphological plural on common nouns (GB044)"),
    ("noun_dual", "n_breadth", "noun_dual", "none", ("GB043",), "exact", None,
     "Productive morphological dual on common nouns (GB043)"),
    ("noun_case_core", "n_breadth", "noun_case", "noun_case", ("GB070",), "exact", None,
     "Non-pronominal core (S/A/P) morphological case on nouns (GB070)"),
    ("noun_case_oblique", "n_breadth", "noun_case", "noun_case", ("GB072",), "exact", None,
     "Non-pronominal oblique morphological case on nouns (GB072)"),
    ("possessed_affix", "n_breadth", "noun_possessed", "none", ("GB431", "GB433"), "or_exact", None,
     "Head-marked possessive prefix/suffix on possessed noun (OR_exact GB431, GB433)"),
    # C. Verbal Paradigmatic Irregularity (v_irreg, k = 5 domain-conditioned, eligible >= 3)
    ("conj_classes", "v_irreg", "v_conj_classes", "none", ("GB111",), "exact", "any_v_morph",
     "Conjugation classes on verbs (GB111 | any_v_morph==1)"),
    ("tam_suppletion", "v_irreg", "v_tam_suppl", "none", ("GB110",), "exact", "any_v_tam",
     "Verb stem suppletion for tense/aspect (GB110 | any_v_tam==1)"),
    ("pers_stem_alt", "v_irreg", "v_pers_stem_alt", "none", ("GB099",), "exact", "person_idx",
     "Verb stem alternation by participant person (GB099 | person_idx==1)"),
    ("part_num_suppl", "v_irreg", "v_part_num_suppl", "none", ("GB109",), "exact", "any_v_morph",
     "Verb stem suppletion for participant number (GB109 | any_v_morph==1)"),
    ("lex_verb_suppl", "v_irreg", "v_lex_suppl", "none", ("GB300", "GB402", "GB403"), "or_lenient", "any_v_morph",
     "Suppletive forms in give/see/come (OR_lenient GB300, GB402, GB403 | any_v_morph==1)"),
    # D. Nominal Paradigmatic Opacity (n_irreg, k = 3 domain-conditioned, eligible >= 2)
    ("noun_num_allom", "n_irreg", "n_num_allom", "none", ("GB039",), "exact", "any_noun_num",
     "Non-phonological allomorphy of noun number markers (GB039 | any_noun_num==1)"),
    ("noun_num_suppl", "n_irreg", "n_num_suppl", "none", ("GB041",), "exact", "any_noun_num",
     ">3 nouns suppletive for number (GB041 | any_noun_num==1)"),
    ("opaque_gender", "n_irreg", "n_opaque_gender", "none", ("GB321",), "exact", "any_gender",
     "Large class of nouns with unpredictable gender/noun-class (GB321 | any_gender==1)"),
    # E. Contextual NP Agreement (np_agr, k = 7, eligible >= 5)
    ("agr_gender_prop", "np_agr", "gender_agr", "gender_agr", ("GB170",), "exact", None,
     "Gender agreement on adnominal property words (GB170)"),
    ("agr_gender_dem", "np_agr", "gender_agr", "gender_agr", ("GB171",), "exact", None,
     "Gender agreement on adnominal demonstratives (GB171)"),
    ("agr_gender_art", "np_agr", "gender_agr", "gender_agr", ("GB172",), "exact", None,
     "Gender agreement on articles (GB172)"),
    ("agr_gender_num", "np_agr", "gender_agr", "gender_agr", ("GB198",), "exact", None,
     "Gender agreement on numerals (GB198)"),
    ("agr_num_prop", "np_agr", "num_agr", "num_agr", ("GB184",), "exact", None,
     "Number agreement on adnominal property words (GB184)"),
    ("agr_num_dem", "np_agr", "num_agr", "num_agr", ("GB185",), "exact", None,
     "Number agreement on adnominal demonstratives (GB185)"),
    ("agr_num_art", "np_agr", "num_agr", "num_agr", ("GB186",), "exact", None,
     "Number agreement on articles (GB186)"),
]

DOMAIN_ORDER: tuple[str, ...] = ("v_breadth", "n_breadth", "v_irreg", "n_irreg", "np_agr")
DOMAIN_FLOORS: dict[str, tuple[int, int]] = {
    "v_breadth": (6, 9),
    "n_breadth": (3, 5),
    "v_irreg": (3, 5),
    "n_irreg": (2, 3),
    "np_agr": (5, 7),
}
RAW_IRREG_FLOORS: dict[str, tuple[int, int]] = {
    "v_irreg_raw": (3, 5),
    "n_irreg_raw": (2, 3),
}

PREREQ_SPECS: dict[str, tuple[str, ...]] = {
    "any_v_morph": (
        "GB082", "GB083", "GB084", "GB086", "GB312",
        "GB089", "GB090", "GB091", "GB092", "GB093", "GB094", "GB107", "GB286",
    ),
    "any_v_tam": ("GB082", "GB083", "GB084", "GB086", "GB312"),
    "person_idx": ("GB089", "GB090", "GB091", "GB092", "GB093", "GB094"),
    "any_noun_num": ("GB042", "GB043", "GB044", "GB165", "GB166"),
    # Broad gender presence: semantic gender cues OR gender agreement on NP modifiers
    # (captures opaque-only noun class languages like Fe'fe' where GB051..054,192 are 0 but GB171=1, GB321=1)
    "any_gender": ("GB051", "GB052", "GB053", "GB054", "GB192", "GB170", "GB171", "GB172", "GB198"),
}

# Legacy sets preserved strictly for backward-compatible columns in language_data.csv
IRREG13_IDS = {
    "GB039", "GB041", "GB095", "GB096", "GB098", "GB099",
    "GB109", "GB110", "GB111", "GB300", "GB321", "GB402", "GB403",
}
OVERLAP4_IDS = {"GB170", "GB171", "GB172", "GB198"}


def _or_exact(raw_vals: list[str]) -> int | None:
    """Exact OR: 1 if any '1'; 0 if all coded and all '0'; None otherwise."""
    if "1" in raw_vals:
        return 1
    if raw_vals and all(v == "0" for v in raw_vals):
        return 0
    return None


def _or_lenient(raw_vals: list[str]) -> int | None:
    """Lenient OR: 1 if any '1'; 0 if >=1 coded and all coded are '0'; None if none coded."""
    coded = [v for v in raw_vals if v in ("0", "1")]
    if not coded:
        return None
    return 1 if "1" in coded else 0


def _eval_item(raw_map: dict[str, str], sources: tuple[str, ...], agg_rule: str) -> int | None:
    vals = [raw_map.get(pid, "") for pid in sources]
    if agg_rule == "exact":
        v = vals[0]
        return int(v) if v in ("0", "1") else None
    if agg_rule == "or_exact":
        return _or_exact(vals)
    if agg_rule == "or_lenient":
        return _or_lenient(vals)
    raise ValueError(f"Unknown agg_rule: {agg_rule}")


def score_language_5domains(raw_map: dict[str, str]) -> tuple[dict[str, int | None], dict[str, int]]:
    """Compute all 29 domain-conditioned items + summary counts (incl. raw unmasked irregularity)."""
    prereqs = {
        k: _or_exact([raw_map.get(pid, "") for pid in pids])
        for k, pids in PREREQ_SPECS.items()
    }
    item_vals: dict[str, int | None] = {}
    counts: dict[str, int] = {}
    for dom in DOMAIN_ORDER:
        counts[f"{dom}_present"] = 0
        counts[f"{dom}_coded"] = 0
    for rdom in ("v_irreg_raw", "n_irreg_raw"):
        counts[f"{rdom}_present"] = 0
        counts[f"{rdom}_coded"] = 0

    for item_id, group, _dom, _testlet, sources, agg_rule, prereq_key, _desc in DOMAIN_ITEM_SPECS:
        raw_v = _eval_item(raw_map, sources, agg_rule)
        if group == "v_irreg" and raw_v is not None:
            counts["v_irreg_raw_coded"] += 1
            counts["v_irreg_raw_present"] += raw_v
        elif group == "n_irreg" and raw_v is not None:
            counts["n_irreg_raw_coded"] += 1
            counts["n_irreg_raw_present"] += raw_v

        if prereq_key is not None:
            v = raw_v if prereqs[prereq_key] == 1 else None
        else:
            v = raw_v

        item_vals[item_id] = v
        if v is not None:
            counts[f"{group}_coded"] += 1
            counts[f"{group}_present"] += v

    return item_vals, counts


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den > 0 else 0.0


def _describe_series(vals: list[int | float]) -> dict[str, float]:
    n = len(vals)
    if n == 0:
        return {"count": 0.0}
    s_vals = sorted(float(x) for x in vals)
    mean_v = sum(s_vals) / n
    std_v = math.sqrt(sum((x - mean_v) ** 2 for x in s_vals) / (n - 1)) if n >= 2 else 0.0

    def quant(p: float) -> float:
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        return s_vals[lo] + (pos - lo) * (s_vals[hi] - s_vals[lo])

    return {
        "count": round(float(n), 2),
        "mean": round(mean_v, 2),
        "std": round(std_v, 2),
        "min": round(s_vals[0], 2),
        "25%": round(quant(0.25), 2),
        "50%": round(quant(0.50), 2),
        "75%": round(quant(0.75), 2),
        "max": round(s_vals[-1], 2),
    }


def _rank_average(vals: list[float]) -> list[float]:
    idx_vals = sorted(enumerate(vals), key=lambda t: t[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(idx_vals):
        j = i
        while j + 1 < len(idx_vals) and idx_vals[j + 1][1] == idx_vals[i][1]:
            j += 1
        avg_r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx_vals[k][0]] = avg_r
        i = j + 1
    return ranks


def _spearman_matrix_dict(rows: list[dict[str, object]], cols: list[str]) -> dict[str, dict[str, float]]:
    col_ranks: dict[str, dict[int, float]] = {}
    for c in cols:
        valid_idx: list[int] = []
        valid_vals: list[float] = []
        for idx, r in enumerate(rows):
            v = r.get(c)
            if v is None or v == "":
                continue
            fv = float(v)
            if not math.isnan(fv):
                valid_idx.append(idx)
                valid_vals.append(fv)
        rks = _rank_average(valid_vals)
        col_ranks[c] = dict(zip(valid_idx, rks))

    out: dict[str, dict[str, float]] = {c: {} for c in cols}
    for c1 in cols:
        r1 = col_ranks[c1]
        for c2 in cols:
            if c1 == c2:
                out[c1][c2] = 1.0
                continue
            r2 = col_ranks[c2]
            common = [i for i in r1 if i in r2]
            xs = [r1[i] for i in common]
            ys = [r2[i] for i in common]
            out[c1][c2] = round(pearson_r(xs, ys), 2)
    return out


def update_existing_language_data(out_dir: Path = DATA_DIR) -> None:
    """Attach audited demographic columns to `language_data.csv` and `prep_summary.json`."""
    ld_path = out_dir / "language_data.csv"
    with open(ld_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    demog_by_gc = {
        str(r["glottocode"]): r for r in resolve_demography_rows([r["glottocode"] for r in rows])
    }
    base_cols = [c for c in rows[0].keys() if c not in DEMOG_COLS]
    fieldnames = base_cols + DEMOG_COLS
    merged_rows = []
    for r in rows:
        dm = demog_by_gc[r["glottocode"]]
        merged = {c: r[c] for c in base_cols}
        for c in DEMOG_COLS:
            merged[c] = dm[c]
        merged_rows.append(merged)

    with open(ld_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(merged_rows)

    main_rows = [r for r in merged_rows if int(r["in_main"]) == 1]
    assert len(main_rows) == 225, f"Expected 225 main Grambank languages, got {len(main_rows)}"
    n_pop_main = sum(int(r["has_pop"]) for r in main_rows)
    n_area_main = sum(int(r["has_area"]) for r in main_rows)
    assert n_pop_main >= 215 and n_area_main >= 210

    both_main = [r for r in main_rows if int(r["has_area"]) == 1]
    lp = [float(r["log_pop"]) for r in both_main]
    la = [float(r["log_area"]) for r in both_main]
    r_pa = pearson_r(lp, la)

    pop_src_cnt = dict(Counter(r["population_source"] or "MISSING" for r in main_rows))
    area_src_cnt = dict(Counter(r["area_source"] or "MISSING" for r in main_rows))
    missing_pop_gcs = [r["glottocode"] for r in main_rows if int(r["has_pop"]) == 0]
    missing_area_only_gcs = [
        r["glottocode"] for r in main_rows if int(r["has_pop"]) == 1 and int(r["has_area"]) == 0
    ]

    summ_path = out_dir / "prep_summary.json"
    summ = json.loads(summ_path.read_text(encoding="utf-8"))
    summ["demography"] = {
        "sets_all_grambank": {
            s: {
                "languages_total": sum(1 for r in merged_rows if int(r[f"in_{s}"]) == 1),
                "languages_has_pop": sum(int(r["has_pop"]) for r in merged_rows if int(r[f"in_{s}"]) == 1),
                "languages_has_pop_and_area": sum(int(r["has_area"]) for r in merged_rows if int(r[f"in_{s}"]) == 1),
            }
            for s in SETS
        },
        "domains_main_cov60": {
            dom: {
                "floor": DOMAIN_FLOORS[dom][0],
                "k_items": DOMAIN_FLOORS[dom][1],
                "full": sum(1 for r in main_rows if int(r[f"{dom}_coded"]) >= DOMAIN_FLOORS[dom][0]),
                "has_pop": sum(
                    int(r["has_pop"]) for r in main_rows if int(r[f"{dom}_coded"]) >= DOMAIN_FLOORS[dom][0]
                ),
                "has_pop_and_area": sum(
                    int(r["has_area"]) for r in main_rows if int(r[f"{dom}_coded"]) >= DOMAIN_FLOORS[dom][0]
                ),
            }
            for dom in DOMAIN_ORDER
        },
        "population_source_main": pop_src_cnt,
        "area_source_main": area_src_cnt,
        "group_iso_mappings": GROUP_ISO_MAP,
        "pluricentric_area_members": PLURICENTRIC_AREA_MEMBERS,
        "missing_pop_glottocodes_main": missing_pop_gcs,
        "missing_area_only_glottocodes_main": missing_area_only_gcs,
        "pearson_r_log_pop_log_area_main": round(r_pa, 4),
        "vif_log_pop_log_area_main": round(1.0 / (1.0 - r_pa ** 2), 4),
    }
    summ_path.write_text(json.dumps(summ, indent=2) + "\n", encoding="utf-8")


def prepare_item_tables(out_dir: Path = DATA_DIR) -> None:
    """Prepare canonical 29-item 5-domain tables (`items_metadata.csv`, `item_observations.csv`,
    `language_table.csv`, `latent_prep_summary.json`) across all 225 GeLaTo-linked Grambank languages."""
    gb_codes: dict[str, set[str]] = {}
    with open(GB_DIR / "codes.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gb_codes.setdefault(r["Parameter_ID"], set()).add(r["Name"])

    all_needed_gb_ids = set(EXCLUDE_MORPH_IDS)
    assert len(all_needed_gb_ids) == 44
    for pid in all_needed_gb_ids:
        assert gb_codes.get(pid) == {"0", "1"}, f"Non-binary Grambank feature {pid}: {gb_codes.get(pid)}"

    with open(out_dir / "language_data.csv", encoding="utf-8") as f:
        lang_rows = list(csv.DictReader(f))
    assert len(lang_rows) == 225

    needed_entries = {r["grambank_entry"] for r in lang_rows}
    raw_by_entry: dict[str, dict[str, str]] = {e: {} for e in needed_entries}
    with open(GB_DIR / "values.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lid = r["Language_ID"]
            pid = r["Parameter_ID"]
            if lid in raw_by_entry and pid in all_needed_gb_ids:
                val = r["Value"]
                if val in ("0", "1", "?"):
                    assert pid not in raw_by_entry[lid], f"Duplicate Grambank value for {(lid, pid)}"
                    raw_by_entry[lid][pid] = val

    items_meta: list[dict[str, str | int]] = []
    for idx, (item_id, group, dom, testlet, sources, agg_rule, prereq_key, desc) in enumerate(
        DOMAIN_ITEM_SPECS, start=1
    ):
        items_meta.append({
            "item_order": idx,
            "item_id": item_id,
            "item_group": group,
            "domain": dom,
            "testlet": testlet,
            "agg_rule": agg_rule,
            "prereq": prereq_key or "none",
            "sources": ";".join(sources),
            "description": desc,
        })
    assert len(items_meta) == 29

    obs_rows: list[dict[str, str | int]] = []
    by_lang_item: dict[tuple[str, str], int] = {}

    for r in lang_rows:
        gc = r["glottocode"]
        entry = r["grambank_entry"]
        item_vals, counts = score_language_5domains(raw_by_entry[entry])
        for dom in DOMAIN_ORDER:
            assert counts[f"{dom}_present"] == int(r[f"{dom}_present"])
            assert counts[f"{dom}_coded"] == int(r[f"{dom}_coded"])
        for rdom in ("v_irreg_raw", "n_irreg_raw"):
            assert counts[f"{rdom}_present"] == int(r[f"{rdom}_present"])
            assert counts[f"{rdom}_coded"] == int(r[f"{rdom}_coded"])

        for item_id, group, dom, testlet, _sources, _agg, _pre, _desc in DOMAIN_ITEM_SPECS:
            v = item_vals[item_id]
            if v is not None:
                by_lang_item[(gc, item_id)] = v
                obs_rows.append({
                    "glottocode": gc,
                    "item_id": item_id,
                    "item_group": group,
                    "domain": dom,
                    "testlet": testlet,
                    "value": v,
                })

    with open(out_dir / "items_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(items_meta[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(items_meta)

    with open(out_dir / "item_observations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(obs_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(obs_rows)

    with open(out_dir / "language_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lang_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(lang_rows)

    # Compute exact domain eligibility counts across all 225, n_A >= 3 w/ f3_fixed, and n_A >= 5 w/ f3_fixed
    f3f_rows = load_f3_fixed_scores(F3_FIXED_CSV)
    f3f_iter = f3f_rows if isinstance(f3f_rows, list) else f3f_rows.to_dict("records")
    pop_has_f3 = {
        str(r["population"]): not math.isnan(float(r["f3_fixed"]))
        for r in f3f_iter
    }
    has_f3_n3: set[str] = set()
    has_f3_n5: set[str] = set()
    with open(out_dir / "populations.csv", encoding="utf-8") as f:
        for pr in csv.DictReader(f):
            if int(pr["in_main"]) != 1:
                continue
            p_nm = pr["population"]
            gc = pr["glottocode"]
            ni = int(pr["n_individuals"])
            if pop_has_f3.get(p_nm, False):
                if ni >= 3:
                    has_f3_n3.add(gc)
                if ni >= 5:
                    has_f3_n5.add(gc)

    domain_summary: dict[str, dict[str, object]] = {}
    item_stats: dict[str, dict[str, object]] = {}
    for dom in DOMAIN_ORDER:
        floor_n, k_items = DOMAIN_FLOORS[dom]
        elig_gcs = [r["glottocode"] for r in lang_rows if int(r[f"{dom}_coded"]) >= floor_n]
        dom_items = [spec[0] for spec in DOMAIN_ITEM_SPECS if spec[1] == dom]
        domain_summary[dom] = {
            "k_items": k_items,
            "min_coded": floor_n,
            "n_lang_all": len(elig_gcs),
            "n_lang_f3_n3": sum(1 for g in elig_gcs if g in has_f3_n3),
            "n_lang_f3_n5": sum(1 for g in elig_gcs if g in has_f3_n5),
        }
        for it in dom_items:
            vals_it = [by_lang_item[(gc, it)] for gc in elig_gcs if (gc, it) in by_lang_item]
            xs, rest_means = [], []
            for gc in elig_gcs:
                if (gc, it) in by_lang_item:
                    others = [by_lang_item[(gc, o)] for o in dom_items if o != it and (gc, o) in by_lang_item]
                    if others:
                        xs.append(float(by_lang_item[(gc, it)]))
                        rest_means.append(sum(others) / len(others))
            item_stats[it] = {
                "item_group": dom,
                "n_coded": len(vals_it),
                "n_present": sum(vals_it),
                "prop_present": round(sum(vals_it) / len(vals_it), 4) if vals_it else None,
                "r_item_rest": round(pearson_r(xs, rest_means), 4) if len(xs) >= 3 else None,
            }

    summary = {
        "n_languages_total": len(lang_rows),
        "n_items_total": len(items_meta),
        "n_nonmorph_features_total": 195 - len(EXCLUDE_MORPH_IDS),
        "n_total_observations": len(obs_rows),
        "domains": domain_summary,
        "item_stats": item_stats,
    }
    (out_dir / "latent_prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def derive_grambank_metadata_from_cldf(
    main_gcs: list[str],
    gb_dir: Path = GB_DIR,
    glot_dir: Path = GLOT_DIR,
) -> list[dict[str, object]]:
    """Derive grambank_entry, name, family, macroarea, and legacy cat12 n_present/n_coded
    directly from tracked Grambank & Glottolog CLDF tables (matching morph_ldl.typology.stage)."""
    with open(glot_dir / "languages.csv", encoding="utf-8") as f:
        gl_by_id = {r["ID"]: r for r in csv.DictReader(f)}

    with open(gb_dir / "languages.csv", encoding="utf-8") as f:
        gb_ids = [r["ID"] for r in csv.DictReader(f)]

    gb_lang_ids = {eid for eid in gb_ids if gl_by_id.get(eid, {}).get("Level") == "language"}
    dial_cands: dict[str, list[str]] = {}
    for eid in gb_ids:
        if eid in gb_lang_ids:
            continue
        gl_e = gl_by_id.get(eid)
        if gl_e and gl_e.get("Level") == "dialect":
            lg = (gl_e.get("Language_ID") or "").strip()
            if lg and gl_by_id.get(lg, {}).get("Level") == "language" and lg not in gb_lang_ids:
                dial_cands.setdefault(lg, []).append(eid)

    needed_entries: set[str] = set()
    for gc in main_gcs:
        if gc in gb_lang_ids:
            needed_entries.add(gc)
        elif gc in dial_cands:
            needed_entries.update(dial_cands[gc])
        else:
            raise KeyError(
                f"Glottocode {gc!r} has neither a language-level nor a dialect-level Grambank entry."
            )

    cat12_param_ids = {pid for srcs in LEGACY_CAT12_SOURCES.values() for pid in srcs}
    raw_cat12: dict[str, dict[str, str]] = {e: {} for e in needed_entries}
    with open(gb_dir / "values.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lid = r["Language_ID"]
            pid = r["Parameter_ID"]
            if lid in raw_cat12 and pid in cat12_param_ids:
                raw_cat12[lid][pid] = r["Value"]

    def score_cat12(vals: dict[str, str]) -> tuple[int, int]:
        n_pres = 0
        n_cod = 0
        for srcs in LEGACY_CAT12_SOURCES.values():
            vs = [vals.get(pid, "") for pid in srcs]
            if any(v == "1" for v in vs):
                n_pres += 1
                n_cod += 1
            elif all(v == "0" for v in vs):
                n_cod += 1
        return n_pres, n_cod

    cat12_scores = {e: score_cat12(raw_cat12[e]) for e in needed_entries}

    g_rows: list[dict[str, object]] = []
    for gc in main_gcs:
        if gc in gb_lang_ids:
            entry = gc
        else:
            entry = sorted(dial_cands[gc], key=lambda e: (-cat12_scores[e][1], e))[0]
        n_pres, n_cod = cat12_scores[entry]
        gl_r = gl_by_id[gc]
        fam_id = (gl_r.get("Family_ID") or "").strip()
        family = gl_by_id[fam_id]["Name"] if (fam_id and fam_id in gl_by_id) else (gl_r.get("Name") or "isolate")
        g_rows.append({
            "glottocode": gc,
            "grambank_entry": entry,
            "name": gl_r.get("Name") or gc,
            "family": family,
            "macroarea": gl_r.get("Macroarea") or "",
            "n_present": n_pres,
            "n_coded": n_cod,
        })
    return g_rows


def main(out_dir: Path = DATA_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read ALL GeLaTo-linked Grambank rows (ungated from meets_coverage_main!)
    links_path = COMMON_DATA_DIR / "analysis_links.csv"
    if not links_path.is_file():
        links_path = TYP_DIR / "analysis_links.csv"
    with open(links_path, encoding="utf-8") as f:
        al_rows = [r for r in csv.DictReader(f) if r["in_grambank"] == "True"]
    main_links = [r for r in al_rows if r["analysis"] == "main"]
    main_link_by_pop = {r["population"]: r for r in main_links}
    assert len(main_link_by_pop) == len(main_links)
    assert len(main_links) == 364, f"Expected 364 main Grambank links, got {len(main_links)}"
    in_set = {s: {r["population"] for r in al_rows if r["analysis"] == s} for s in SETS}
    assert all(in_set[s] <= in_set["main"] for s in SETS)

    main_gcs = sorted({r["glottocode"] for r in main_links})
    assert len(main_gcs) == 225, f"Expected 225 unique Grambank languages, got {len(main_gcs)}"

    # Derive grambank_entry & metadata directly from tracked Grambank & Glottolog CLDF tables
    g_rows = derive_grambank_metadata_from_cldf(main_gcs, gb_dir=GB_DIR, glot_dir=GLOT_DIR)

    # Read Grambank parameters and values for both 5-domain batteries (44 IDs) and legacy flexivity (30 IDs)
    with open(GB_DIR / "parameters.csv", encoding="utf-8") as f:
        flex_ids = {r["ID"] for r in csv.DictReader(f) if r.get("Flexivity") == "1"}
    assert len(flex_ids) == 30
    flex26_ids = flex_ids - OVERLAP4_IDS
    assert len(flex26_ids) == 26 and IRREG13_IDS <= flex26_ids

    needed_gb_entries = {str(r["grambank_entry"]) for r in g_rows}
    needed_gb_params = set(EXCLUDE_MORPH_IDS) | flex_ids
    raw_by_entry: dict[str, dict[str, str]] = {e: {} for e in needed_gb_entries}
    flex_sums: dict[str, dict[str, list[int]]] = {
        "flex": {e: [0, 0] for e in needed_gb_entries},
        "flex26": {e: [0, 0] for e in needed_gb_entries},
        "irreg13": {e: [0, 0] for e in needed_gb_entries},
    }
    with open(GB_DIR / "values.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lid = r["Language_ID"]
            pid = r["Parameter_ID"]
            val = r["Value"]
            if lid not in raw_by_entry or pid not in needed_gb_params:
                continue
            if val in ("0", "1", "?"):
                raw_by_entry[lid][pid] = val
            if pid in flex_ids and val in ("0", "1"):
                iv = int(val)
                flex_sums["flex"][lid][0] += iv
                flex_sums["flex"][lid][1] += 1
                if pid in flex26_ids:
                    flex_sums["flex26"][lid][0] += iv
                    flex_sums["flex26"][lid][1] += 1
                if pid in IRREG13_IDS:
                    flex_sums["irreg13"][lid][0] += iv
                    flex_sums["irreg13"][lid][1] += 1

    # Compute strictly non-outcome Grambank documentation coverage (151 features)
    gb_entry_map = {str(r["glottocode"]): str(r["grambank_entry"]) for r in g_rows}
    cov_res = compute_grambank_nonmorph_coverage(
        main_gcs,
        exclude_ids=EXCLUDE_MORPH_IDS,
        gb_dir=GB_DIR,
        entry_map=gb_entry_map,
    )
    cov_iter = cov_res if isinstance(cov_res, list) else cov_res.to_dict("records")
    cov_by_gc = {str(r["glottocode"]): r for r in cov_iter}

    for r in g_rows:
        gc = str(r["glottocode"])
        e = str(r["grambank_entry"])
        for pfx in ("flex", "flex26", "irreg13"):
            r[f"{pfx}_present"] = flex_sums[pfx][e][0]
            r[f"{pfx}_coded"] = flex_sums[pfx][e][1]
        _, dom_counts = score_language_5domains(raw_by_entry[e])
        r.update(dom_counts)
        r["n_coded_nonmorph"] = int(cov_by_gc[gc]["n_nonmorph_coded"])
        r["prop_coded_nonmorph"] = float(cov_by_gc[gc]["prop_coded_nonmorph"])

    # Compute individual & population measures via analyses.common.genetics
    info, Q = load_admixture_q(ks=KS)
    ind_res = compute_individual_measures(info, Q, links_df=main_links, ks=KS)
    ind_cols = ["population", "ind", "ent", "nonmax", "ent_K12", "ent_K20", "ent_K30", "glottocode"]
    if pd is not None and isinstance(ind_res, pd.DataFrame):
        ind_res.round(6).to_csv(out_dir / "individuals.csv", index=False)
        ind_rows = ind_res.to_dict("records")
    else:
        ind_rows = ind_res
        write_rows_csv(ind_rows, out_dir / "individuals.csv", ind_cols, ndigits=6)

    all_main_pops = {r["population"] for r in main_links}
    pop_res, pooled = compute_population_measures(
        info,
        Q,
        [r["population"] for r in main_links],
        priority_populations=all_main_pops,
        pooled_ref_populations=all_main_pops,
        n_boot=N_BOOT,
        seed=SEED,
        ks=KS,
        f3_path=COMMON_DATA_DIR / "f3_scores.csv",
        k5_path=COMMON_DATA_DIR / "f3_scores_k5.csv",
        include_extended_metadata=False,
        warn_missing_f3=True,
    )

    pop_cols = [
        "population", "n_individuals", "pop_ent", "het", "support", "amount", "sqrt_het",
        "pop_ent_se", "het_se", "support_se", "amount_se", "sqrt_het_se",
        "graff_target", "neigh_flow", "n_B", "n_C", "n_triples", "n_z_lt_m3",
        "min_pop2", "min_pop3", "f3_z", "f3_est",
        "glottocode", "link_basis", "accepted_by", "sensitivity_flags",
        "in_main", "in_no_doubtful_language", "in_no_other_language_stand_ins", "in_strict",
    ]
    if pd is not None and isinstance(pop_res, pd.DataFrame):
        main_links_df = pd.DataFrame(main_links)[[
            "population", "glottocode", "link_basis", "accepted_by", "sensitivity_flags"
        ]]
        pop_df = pop_res.merge(main_links_df, on="population")
        for s in SETS:
            pop_df[f"in_{s}"] = pop_df.population.isin(in_set[s]).astype(int)
        pop_df.round(7).to_csv(out_dir / "populations.csv", index=False)
        pop_rows = pop_df.to_dict("records")
    else:
        pop_rows = []
        for r in pop_res:
            p = str(r["population"])
            ml = main_link_by_pop[p]
            merged_p = {
                **r,
                "glottocode": ml["glottocode"],
                "link_basis": ml["link_basis"],
                "accepted_by": ml["accepted_by"],
                "sensitivity_flags": ml["sensitivity_flags"],
            }
            for s in SETS:
                merged_p[f"in_{s}"] = 1 if p in in_set[s] else 0
            pop_rows.append(merged_p)
        write_rows_csv(pop_rows, out_dir / "populations.csv", pop_cols, ndigits=7)

    # Language table, coordinates, and Glottolog phylogeny across all 225 languages
    with open(ROOT / "external/glottolog-cldf/cldf/languages.csv", encoding="utf-8") as f:
        gl_by_id = {r["ID"]: r for r in csv.DictReader(f)}

    pops_by_gc: dict[str, list[dict[str, object]]] = {}
    for r in pop_rows:
        pops_by_gc.setdefault(str(r["glottocode"]), []).append(r)

    inds_by_gc: dict[str, list[dict[str, object]]] = {}
    for r in ind_rows:
        inds_by_gc.setdefault(str(r["glottocode"]), []).append(r)

    paths, tree_src = resolve_glottolog_paths(main_gcs, root_dir=ROOT)
    (out_dir / "phylo_tree.nwk").write_text(build_newick(paths) + "\n", encoding="utf-8")

    lang_base_cols = [
        "glottocode", "grambank_entry", "name", "family", "macroarea",
        "v_breadth_present", "v_breadth_coded",
        "n_breadth_present", "n_breadth_coded",
        "v_irreg_present", "v_irreg_coded",
        "n_irreg_present", "n_irreg_coded",
        "np_agr_present", "np_agr_coded",
        "v_irreg_raw_present", "v_irreg_raw_coded",
        "n_irreg_raw_present", "n_irreg_raw_coded",
        "n_coded_nonmorph", "prop_coded_nonmorph",
        "n_present", "n_coded",
        "flex_present", "flex_coded", "flex26_present", "flex26_coded",
        "irreg13_present", "irreg13_coded",
        "lat", "lon",
        "n_populations", "n_individuals",
        "in_main", "n_individuals_main",
        "in_no_doubtful_language", "n_individuals_no_doubtful_language",
        "in_no_other_language_stand_ins", "n_individuals_no_other_language_stand_ins",
        "in_strict", "n_individuals_strict",
        "ent", "nonmax", "tree_source",
    ]
    lang_out_rows: list[dict[str, object]] = []
    for r in g_rows:
        gc = str(r["glottocode"])
        gl_r = gl_by_id[gc]
        p_list = pops_by_gc[gc]
        i_list = inds_by_gc[gc]
        row_l: dict[str, object] = {
            "glottocode": gc,
            "grambank_entry": r["grambank_entry"],
            "name": r["name"],
            "family": r["family"] or "isolate",
            "macroarea": r["macroarea"],
            "v_breadth_present": int(r["v_breadth_present"]),
            "v_breadth_coded": int(r["v_breadth_coded"]),
            "n_breadth_present": int(r["n_breadth_present"]),
            "n_breadth_coded": int(r["n_breadth_coded"]),
            "v_irreg_present": int(r["v_irreg_present"]),
            "v_irreg_coded": int(r["v_irreg_coded"]),
            "n_irreg_present": int(r["n_irreg_present"]),
            "n_irreg_coded": int(r["n_irreg_coded"]),
            "np_agr_present": int(r["np_agr_present"]),
            "np_agr_coded": int(r["np_agr_coded"]),
            "v_irreg_raw_present": int(r["v_irreg_raw_present"]),
            "v_irreg_raw_coded": int(r["v_irreg_raw_coded"]),
            "n_irreg_raw_present": int(r["n_irreg_raw_present"]),
            "n_irreg_raw_coded": int(r["n_irreg_raw_coded"]),
            "n_coded_nonmorph": int(r["n_coded_nonmorph"]),
            "prop_coded_nonmorph": float(r["prop_coded_nonmorph"]),
            "n_present": int(r["n_present"]),
            "n_coded": int(r["n_coded"]),
            "flex_present": int(r["flex_present"]),
            "flex_coded": int(r["flex_coded"]),
            "flex26_present": int(r["flex26_present"]),
            "flex26_coded": int(r["flex26_coded"]),
            "irreg13_present": int(r["irreg13_present"]),
            "irreg13_coded": int(r["irreg13_coded"]),
            "lat": float(gl_r["Latitude"]),
            "lon": float(gl_r["Longitude"]),
            "n_populations": len(p_list),
            "n_individuals": sum(int(p["n_individuals"]) for p in p_list),
        }
        for s in SETS:
            sub_p = [p for p in p_list if int(p[f"in_{s}"]) == 1]
            row_l[f"in_{s}"] = 1 if sub_p else 0
            row_l[f"n_individuals_{s}"] = sum(int(p["n_individuals"]) for p in sub_p)
        row_l["ent"] = sum(float(x["ent"]) for x in i_list) / len(i_list)
        row_l["nonmax"] = sum(float(x["nonmax"]) for x in i_list) / len(i_list)
        row_l["tree_source"] = tree_src[gc]
        lang_out_rows.append(row_l)

    write_rows_csv(lang_out_rows, out_dir / "language_data.csv", lang_base_cols, ndigits=6)

    # Summary JSON
    inds_by_pop: dict[str, list[dict[str, object]]] = {}
    for r in ind_rows:
        inds_by_pop.setdefault(str(r["population"]), []).append(r)

    pi_rows: list[dict[str, object]] = []
    for p_r in pop_rows:
        p_name = str(p_r["population"])
        il = inds_by_pop[p_name]
        pi_rows.append({
            **p_r,
            "ent": sum(float(x["ent"]) for x in il) / len(il),
            "nonmax": sum(float(x["nonmax"]) for x in il) / len(il),
            "ent_K12": sum(float(x["ent_K12"]) for x in il) / len(il),
            "ent_K30": sum(float(x["ent_K30"]) for x in il) / len(il),
        })

    spearman_cols = [
        "ent", "nonmax", "pop_ent", "het", "sqrt_het", "support", "amount",
        "graff_target", "neigh_flow", "f3_z", "f3_est", "n_individuals", "n_triples",
    ]
    tree_src_counts = dict(Counter(str(r["tree_source"]) for r in lang_out_rows).most_common())

    def is_na_val(v: object) -> bool:
        if v is None or v == "":
            return True
        try:
            return math.isnan(float(v))
        except (TypeError, ValueError):
            return True

    summ = {
        "sets": {
            s: {
                "languages": sum(int(r[f"in_{s}"]) for r in lang_out_rows),
                "populations": sum(int(r[f"in_{s}"]) for r in pop_rows),
                "individuals": sum(int(r["n_individuals"]) for r in pop_rows if int(r[f"in_{s}"]) == 1),
            }
            for s in SETS
        },
        "n_families": len({str(r["family"]) for r in lang_out_rows}),
        "tree_source": tree_src_counts,
        "individuals_per_language": _describe_series([int(r["n_individuals"]) for r in lang_out_rows]),
        "n_nonmorph_features_total": 195 - len(EXCLUDE_MORPH_IDS),
        "populations_without_het": sum(1 for r in pop_rows if is_na_val(r.get("het"))),
        "populations_without_f3": sum(1 for r in pop_rows if is_na_val(r.get("f3_z"))),
        "populations_without_neigh_flow": sum(1 for r in pop_rows if is_na_val(r.get("neigh_flow"))),
        "pooled_per_individual_sd": pooled,
        "n_boot": N_BOOT,
        "domains": {
            dom: {
                "k_items": DOMAIN_FLOORS[dom][1],
                "min_coded": DOMAIN_FLOORS[dom][0],
                "languages_eligible": sum(1 for r in lang_out_rows if int(r[f"{dom}_coded"]) >= DOMAIN_FLOORS[dom][0]),
                "coded_per_language": _describe_series([int(r[f"{dom}_coded"]) for r in lang_out_rows]),
            }
            for dom in DOMAIN_ORDER
        },
        "raw_irreg": {
            rdom: {
                "k_items": RAW_IRREG_FLOORS[rdom][1],
                "min_coded": RAW_IRREG_FLOORS[rdom][0],
                "languages_eligible": sum(1 for r in lang_out_rows if int(r[f"{rdom}_coded"]) >= RAW_IRREG_FLOORS[rdom][0]),
                "coded_per_language": _describe_series([int(r[f"{rdom}_coded"]) for r in lang_out_rows]),
            }
            for rdom in ("v_irreg_raw", "n_irreg_raw")
        },
        "population_spearman": _spearman_matrix_dict(pi_rows, spearman_cols),
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summ, indent=2) + "\n", encoding="utf-8")
    update_existing_language_data(out_dir=out_dir)

    # 29-item 5-domain tables for latent_sem
    prepare_item_tables(out_dir=out_dir)

    # Shared all-GeLaTo-population summary in analyses/common/data/
    gel_sum_path = COMMON_DATA_DIR / "gelato_populations_summary.csv"
    if not gel_sum_path.exists() or "--refresh-gelato-summary" in sys.argv:
        build_gelato_populations_summary(out_path=gel_sum_path)

    print("Prepared Grambank canonical 5-domain tables (225 languages, 364 populations) in:", out_dir)


if __name__ == "__main__":
    if "--demog-only" in sys.argv:
        update_existing_language_data()
    else:
        main()
