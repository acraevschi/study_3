"""Unified data preparation for Track 1: GeLaTo Admixture -> Grambank Morphology.

Calls `analyses.common` (`genetics`, `demography`, `phylogeny`) to write all canonical
Grambank data tables into `analyses/grambank/data/`:
  - `language_data.csv`          (202 main-coverage Grambank languages)
  - `populations.csv`            (335 main-set populations)
  - `individuals.csv`            (2,943 main-set individuals)
  - `phylo_tree.nwk`             (Glottolog topology Newick for Grafen branch lengths)
  - `prep_summary.json`          (Dataset & demography summary matching baseline)
  - `items_metadata.csv`         (38 binary items: 12 cat12 + 26 flex26, incl. 13 irreg13)
  - `item_observations.csv`      (6,821 observed binary language x item cells)
  - `language_table.csv`         (202 cat12 languages with grambank_entry)
  - `latent_prep_summary.json`   (Item-level descriptive summary)

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
import statistics
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
    KS,
    N_BOOT,
    SEED,
    build_gelato_populations_summary,
    compute_individual_measures,
    compute_population_measures,
    load_admixture_q,
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

SETS = ["main", "no_doubtful_language", "no_other_language_stand_ins", "strict"]

IRREG13_IDS = {
    "GB039", "GB041", "GB095", "GB096", "GB098", "GB099",
    "GB109", "GB110", "GB111", "GB300", "GB321", "GB402", "GB403",
}
OVERLAP4_IDS = {"GB170", "GB171", "GB172", "GB198"}

FLEX_SUBDOMAINS: dict[str, str] = {
    "GB039": "noun_irreg",
    "GB041": "noun_irreg",
    "GB321": "noun_irreg",
    "GB095": "arg_split_classes",
    "GB096": "arg_split_classes",
    "GB098": "arg_split_classes",
    "GB111": "arg_split_classes",
    "GB099": "verb_suppletion_stem",
    "GB109": "verb_suppletion_stem",
    "GB110": "verb_suppletion_stem",
    "GB300": "verb_suppletion_stem",
    "GB402": "verb_suppletion_stem",
    "GB403": "verb_suppletion_stem",
    "GB030": "gender_nc",
    "GB051": "gender_nc",
    "GB052": "gender_nc",
    "GB053": "gender_nc",
    "GB054": "gender_nc",
    "GB192": "gender_nc",
    "GB038": "classifiers",
    "GB057": "classifiers",
    "GB058": "classifiers",
    "GB116": "classifiers",
    "GB177": "evaluative_anim",
    "GB187": "evaluative_anim",
    "GB188": "evaluative_anim",
}


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
    s_vals = sorted(float(x) for x in vals)
    mean_v = sum(s_vals) / n
    std_v = math.sqrt(sum((x - mean_v) ** 2 for x in s_vals) / (n - 1))

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
    # Pandas .rank().corr() ranks each column independently over its non-NA entries,
    # then computes pairwise Pearson r on the overlapping non-NA rows of each pair.
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged_rows)

    c12 = [r for r in merged_rows if int(r["n_coded"]) >= 8]
    assert sum(int(r["has_pop"]) for r in c12 if int(r["in_main"]) == 1) == 195
    assert sum(int(r["has_area"]) for r in c12 if int(r["in_main"]) == 1) == 192

    both_c12 = [r for r in c12 if int(r["in_main"]) == 1 and int(r["has_area"]) == 1]
    lp = [float(r["log_pop"]) for r in both_c12]
    la = [float(r["log_area"]) for r in both_c12]
    mp, ma = sum(lp) / len(lp), sum(la) / len(la)
    cov = sum((x - mp) * (y - ma) for x, y in zip(lp, la)) / (len(lp) - 1)
    sp = (sum((x - mp) ** 2 for x in lp) / (len(lp) - 1)) ** 0.5
    sa = (sum((y - ma) ** 2 for y in la) / (len(la) - 1)) ** 0.5
    r_pa = cov / (sp * sa)

    pop_src_cnt = dict(Counter(r["population_source"] or "MISSING" for r in c12 if int(r["in_main"]) == 1))
    area_src_cnt = dict(Counter(r["area_source"] or "MISSING" for r in c12 if int(r["in_main"]) == 1))
    missing_pop_gcs = [r["glottocode"] for r in c12 if int(r["in_main"]) == 1 and int(r["has_pop"]) == 0]
    missing_area_only_gcs = [
        r["glottocode"]
        for r in c12
        if int(r["in_main"]) == 1 and int(r["has_pop"]) == 1 and int(r["has_area"]) == 0
    ]

    summ_path = out_dir / "prep_summary.json"
    summ = json.loads(summ_path.read_text(encoding="utf-8"))
    summ["demography"] = {
        "sets_cat12": {
            s: {
                "languages_total": sum(1 for r in c12 if int(r[f"in_{s}"]) == 1),
                "languages_has_pop": sum(int(r["has_pop"]) for r in c12 if int(r[f"in_{s}"]) == 1),
                "languages_has_pop_and_area": sum(int(r["has_area"]) for r in c12 if int(r[f"in_{s}"]) == 1),
            }
            for s in SETS
        },
        "outcomes_main_cat12_denom": {
            "cat12_cov60": {
                "full": len(c12),
                "has_pop": sum(int(r["has_pop"]) for r in c12),
                "has_pop_and_area": sum(int(r["has_area"]) for r in c12),
            },
            "flex_cov60": {
                "full": sum(1 for r in c12 if int(r["flex_coded"]) >= 18),
                "has_pop": sum(int(r["has_pop"]) for r in c12 if int(r["flex_coded"]) >= 18),
                "has_pop_and_area": sum(int(r["has_area"]) for r in c12 if int(r["flex_coded"]) >= 18),
            },
            "irreg13_cov60": {
                "full": sum(1 for r in c12 if int(r["irreg13_coded"]) >= 8),
                "has_pop": sum(int(r["has_pop"]) for r in c12 if int(r["irreg13_coded"]) >= 8),
                "has_pop_and_area": sum(int(r["has_area"]) for r in c12 if int(r["irreg13_coded"]) >= 8),
            },
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
    """Prepare item-level binary data tables (`items_metadata.csv`, `item_observations.csv`,
    `language_table.csv`, `latent_prep_summary.json`)."""
    with open(TYP_DIR / "feature_set.json", encoding="utf-8") as f:
        fjson = json.load(f)
    fs_decl = fjson["declared"]["feature_set"]
    categories: dict[str, list[str]] = fs_decl["categories"]
    cat_domains: dict[str, str] = {}
    cat_order: list[str] = []
    for dom, clist in fs_decl["domains"].items():
        for c in clist:
            cat_domains[c] = dom
            cat_order.append(c)
    assert len(cat_order) == 12

    gb_params: dict[str, dict[str, str]] = {}
    with open(GB_DIR / "parameters.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gb_params[r["ID"]] = r

    gb_codes: dict[str, set[str]] = {}
    with open(GB_DIR / "codes.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gb_codes.setdefault(r["Parameter_ID"], set()).add(r["Name"])

    flex_all_ids = sorted(pid for pid, r in gb_params.items() if r.get("Flexivity") == "1")
    assert len(flex_all_ids) == 30
    flex26_ids = [pid for pid in flex_all_ids if pid not in OVERLAP4_IDS]
    assert len(flex26_ids) == 26
    assert set(flex26_ids) == set(FLEX_SUBDOMAINS.keys())
    assert IRREG13_IDS <= set(flex26_ids) and len(IRREG13_IDS) == 13

    all_needed_gb_ids = set(flex26_ids) | {g for srcs in categories.values() for g in srcs}
    for pid in all_needed_gb_ids:
        assert gb_codes.get(pid) == {"0", "1"}, f"Non-binary Grambank feature {pid}: {gb_codes.get(pid)}"

    with open(out_dir / "language_data.csv", encoding="utf-8") as f:
        raw_lang_rows = [r for r in csv.DictReader(f) if int(r["n_coded"]) >= 8]
    assert len(raw_lang_rows) == 202

    gb_entry_map: dict[str, str] = {}
    with open(TYP_DIR / "grambank_inflection.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["glottocode"] and r["grambank_entry"]:
                gb_entry_map[r["glottocode"]] = r["grambank_entry"]

    lang_rows = []
    for r in raw_lang_rows:
        gc = r["glottocode"]
        assert gc in gb_entry_map, f"Missing grambank_entry for {gc}"
        row_clean = {k: v for k, v in r.items() if k not in DEMOG_COLS}
        row_clean["grambank_entry"] = gb_entry_map[gc]
        lang_rows.append(row_clean)

    needed_entries = {r["grambank_entry"] for r in lang_rows}

    raw_vals: dict[tuple[str, str], str] = {}
    with open(GB_DIR / "values.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lid = r["Language_ID"]
            pid = r["Parameter_ID"]
            if lid in needed_entries and pid in all_needed_gb_ids:
                val = r["Value"]
                if val in ("0", "1", "?"):
                    assert (lid, pid) not in raw_vals, f"Duplicate Grambank value for {(lid, pid)}"
                    raw_vals[(lid, pid)] = val

    domain_counts: dict[str, int] = {}
    for c in cat_order:
        d = cat_domains[c]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    for pid in flex26_ids:
        d = FLEX_SUBDOMAINS[pid]
        domain_counts[d] = domain_counts.get(d, 0) + 1

    items_meta: list[dict[str, str | int]] = []
    for idx, c in enumerate(cat_order, start=1):
        dom = cat_domains[c]
        testlet = dom if domain_counts[dom] >= 2 else "none"
        items_meta.append({
            "item_order": idx,
            "item_id": c,
            "item_group": "cat12",
            "in_cat12": 1,
            "in_irreg13": 0,
            "in_flex26": 0,
            "domain": dom,
            "testlet": testlet,
            "is_agreement": 1 if dom == "agreement" else 0,
            "sources": ";".join(categories[c]),
            "description": f"GBI category: {c} ({', '.join(categories[c])})",
        })

    for offset, pid in enumerate(flex26_ids, start=1):
        dom = FLEX_SUBDOMAINS[pid]
        testlet = dom if domain_counts[dom] >= 2 else "none"
        items_meta.append({
            "item_order": 12 + offset,
            "item_id": pid,
            "item_group": "flex26",
            "in_cat12": 0,
            "in_irreg13": 1 if pid in IRREG13_IDS else 0,
            "in_flex26": 1,
            "domain": dom,
            "testlet": testlet,
            "is_agreement": 0,
            "sources": pid,
            "description": gb_params[pid]["Name"],
        })

    obs_rows: list[dict[str, str | int]] = []
    by_lang_item: dict[tuple[str, str], int] = {}

    for r in lang_rows:
        gc = r["glottocode"]
        entry = r["grambank_entry"]

        c12_pres, c12_coded = 0, 0
        for c in cat_order:
            srcs = categories[c]
            vals = [raw_vals.get((entry, s), "") for s in srcs]
            if "1" in vals:
                v = 1
            elif vals and all(x == "0" for x in vals):
                v = 0
            else:
                v = None
            if v is not None:
                c12_coded += 1
                c12_pres += v
                by_lang_item[(gc, c)] = v
                obs_rows.append({
                    "glottocode": gc,
                    "item_id": c,
                    "item_group": "cat12",
                    "domain": cat_domains[c],
                    "value": v,
                })
        assert c12_pres == int(r["n_present"]) and c12_coded == int(r["n_coded"])

        f26_pres, f26_coded = 0, 0
        ir13_pres, ir13_coded = 0, 0
        for pid in flex26_ids:
            val = raw_vals.get((entry, pid), "")
            if val in ("0", "1"):
                v = int(val)
                f26_coded += 1
                f26_pres += v
                if pid in IRREG13_IDS:
                    ir13_coded += 1
                    ir13_pres += v
                by_lang_item[(gc, pid)] = v
                obs_rows.append({
                    "glottocode": gc,
                    "item_id": pid,
                    "item_group": "flex26",
                    "domain": FLEX_SUBDOMAINS[pid],
                    "value": v,
                })
        assert f26_pres == int(r["flex26_present"]) and f26_coded == int(r["flex26_coded"])
        assert ir13_pres == int(r["irreg13_present"]) and ir13_coded == int(r["irreg13_coded"])

    with open(out_dir / "items_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(items_meta[0].keys()))
        w.writeheader()
        w.writerows(items_meta)

    with open(out_dir / "item_observations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(obs_rows[0].keys()))
        w.writeheader()
        w.writerows(obs_rows)

    with open(out_dir / "language_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lang_rows[0].keys()))
        w.writeheader()
        w.writerows(lang_rows)

    main_cat12_langs = [r["glottocode"] for r in lang_rows if r["in_main"] == "1" and int(r["n_coded"]) >= 8]
    assert len(main_cat12_langs) == 202

    cat12_stats = {}
    for c in cat_order:
        vals_c = [by_lang_item[(gc, c)] for gc in main_cat12_langs if (gc, c) in by_lang_item]
        xs, rest_means = [], []
        for gc in main_cat12_langs:
            if (gc, c) in by_lang_item:
                others = [by_lang_item[(gc, o)] for o in cat_order if o != c and (gc, o) in by_lang_item]
                xs.append(float(by_lang_item[(gc, c)]))
                rest_means.append(sum(others) / len(others))
        cat12_stats[c] = {
            "domain": cat_domains[c],
            "n_coded": len(vals_c),
            "n_present": sum(vals_c),
            "prop_present": round(sum(vals_c) / len(vals_c), 4),
            "r_item_rest": round(pearson_r(xs, rest_means), 4),
        }

    summary = {
        "n_languages_total": len(lang_rows),
        "n_languages_main_cat12_cov60": len(main_cat12_langs),
        "n_languages_main_cat12_irreg13_cov60": sum(
            1 for r in lang_rows
            if r["in_main"] == "1" and int(r["n_coded"]) >= 8 and int(r["irreg13_coded"]) >= 8
        ),
        "n_languages_main_cat12_flex26_cov60": sum(
            1 for r in lang_rows
            if r["in_main"] == "1" and int(r["n_coded"]) >= 8 and int(r["flex26_coded"]) >= 16
        ),
        "n_items_cat12": len(cat_order),
        "n_items_irreg13": len(IRREG13_IDS),
        "n_items_flex26": len(flex26_ids),
        "n_total_observations": len(obs_rows),
        "cat12_item_stats_main": cat12_stats,
    }
    (out_dir / "latent_prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(out_dir: Path = DATA_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read analysis_links.csv (202 main-coverage Grambank languages, 335 populations)
    with open(TYP_DIR / "analysis_links.csv", encoding="utf-8") as f:
        al_rows = [
            r for r in csv.DictReader(f)
            if r["in_grambank"] == "True" and r["meets_coverage_main"] == "True"
        ]
    main_links = [r for r in al_rows if r["analysis"] == "main"]
    main_link_by_pop = {r["population"]: r for r in main_links}
    assert len(main_link_by_pop) == len(main_links)
    in_set = {s: {r["population"] for r in al_rows if r["analysis"] == s} for s in SETS}
    assert all(in_set[s] <= in_set["main"] for s in SETS)

    # Read grambank_inflection.csv
    with open(TYP_DIR / "grambank_inflection.csv", encoding="utf-8") as f:
        g_rows = [
            r for r in csv.DictReader(f)
            if r["in_grambank"] == "True"
            and r["meets_coverage_main"] == "True"
            and int(r["n_populations_main"]) > 0
        ]
    assert {r["glottocode"] for r in g_rows} == {r["glottocode"] for r in main_links}

    # Read Grambank flexivity parameters & values
    with open(GB_DIR / "parameters.csv", encoding="utf-8") as f:
        flex_ids = {r["ID"] for r in csv.DictReader(f) if r.get("Flexivity") == "1"}
    assert len(flex_ids) == 30
    flex26_ids = flex_ids - OVERLAP4_IDS
    assert len(flex26_ids) == 26 and IRREG13_IDS <= flex26_ids

    needed_gb_entries = {r["grambank_entry"] for r in g_rows}
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
            if lid in needed_gb_entries and pid in flex_ids and val in ("0", "1"):
                iv = int(val)
                flex_sums["flex"][lid][0] += iv
                flex_sums["flex"][lid][1] += 1
                if pid in flex26_ids:
                    flex_sums["flex26"][lid][0] += iv
                    flex_sums["flex26"][lid][1] += 1
                if pid in IRREG13_IDS:
                    flex_sums["irreg13"][lid][0] += iv
                    flex_sums["irreg13"][lid][1] += 1

    for r in g_rows:
        e = r["grambank_entry"]
        for pfx in ("flex", "flex26", "irreg13"):
            r[f"{pfx}_present"] = flex_sums[pfx][e][0]
            r[f"{pfx}_coded"] = flex_sums[pfx][e][1]

    # Compute individual & population measures via analyses.common.genetics
    info, Q = load_admixture_q(ks=KS)
    ind_res = compute_individual_measures(info, Q, links_df=main_links, ks=KS)
    ind_cols = ["population", "ind", "ent", "nonmax", "ent_K12", "ent_K30", "glottocode"]
    if pd is not None and isinstance(ind_res, pd.DataFrame):
        ind_res.round(6).to_csv(out_dir / "individuals.csv", index=False)
        ind_rows = ind_res.to_dict("records")
    else:
        ind_rows = ind_res
        write_rows_csv(ind_rows, out_dir / "individuals.csv", ind_cols, ndigits=6)

    cat12_pops = {r["population"] for r in main_links if r["meets_coverage_main"] == "True"}
    pop_res, pooled = compute_population_measures(
        info,
        Q,
        [r["population"] for r in main_links],
        priority_populations=cat12_pops,
        pooled_ref_populations=cat12_pops,
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

    # Language table, coordinates, and Glottolog phylogeny
    with open(ROOT / "external/glottolog-cldf/cldf/languages.csv", encoding="utf-8") as f:
        gl_by_id = {r["ID"]: r for r in csv.DictReader(f)}

    pops_by_gc: dict[str, list[dict[str, object]]] = {}
    for r in pop_rows:
        pops_by_gc.setdefault(str(r["glottocode"]), []).append(r)

    inds_by_gc: dict[str, list[dict[str, object]]] = {}
    for r in ind_rows:
        inds_by_gc.setdefault(str(r["glottocode"]), []).append(r)

    g_codes = [r["glottocode"] for r in g_rows]
    paths, tree_src = resolve_glottolog_paths(g_codes, root_dir=ROOT)
    (out_dir / "phylo_tree.nwk").write_text(build_newick(paths) + "\n", encoding="utf-8")

    lang_base_cols = [
        "glottocode", "name", "family", "macroarea", "n_present", "n_coded",
        "flex_present", "flex_coded", "flex26_present", "flex26_coded",
        "irreg13_present", "irreg13_coded", "lat", "lon",
        "n_populations", "n_individuals",
        "in_main", "n_individuals_main",
        "in_no_doubtful_language", "n_individuals_no_doubtful_language",
        "in_no_other_language_stand_ins", "n_individuals_no_other_language_stand_ins",
        "in_strict", "n_individuals_strict",
        "ent", "nonmax", "tree_source",
    ]
    lang_out_rows: list[dict[str, object]] = []
    for r in g_rows:
        gc = r["glottocode"]
        gl_r = gl_by_id[gc]
        p_list = pops_by_gc[gc]
        i_list = inds_by_gc[gc]
        row_l: dict[str, object] = {
            "glottocode": gc,
            "name": r["name"],
            "family": r["family"] or "isolate",
            "macroarea": r["macroarea"],
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
    d_c12 = [r for r in lang_out_rows if int(r["n_coded"]) >= 8]
    pop_c12 = [r for r in pop_rows if str(r["population"]) in cat12_pops]
    inds_by_pop: dict[str, list[dict[str, object]]] = {}
    for r in ind_rows:
        inds_by_pop.setdefault(str(r["population"]), []).append(r)

    pi_rows: list[dict[str, object]] = []
    for p_r in pop_c12:
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
    tree_src_counts = dict(Counter(str(r["tree_source"]) for r in d_c12).most_common())

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
                "languages": sum(int(r[f"in_{s}"]) for r in d_c12),
                "populations": sum(int(r[f"in_{s}"]) for r in pop_c12),
                "individuals": sum(int(r["n_individuals"]) for r in pop_c12 if int(r[f"in_{s}"]) == 1),
            }
            for s in SETS
        },
        "n_families": len({str(r["family"]) for r in d_c12}),
        "tree_source": tree_src_counts,
        "individuals_per_language": _describe_series([int(r["n_individuals"]) for r in d_c12]),
        "populations_without_het": sum(1 for r in pop_c12 if is_na_val(r.get("het"))),
        "populations_without_f3": sum(1 for r in pop_c12 if is_na_val(r.get("f3_z"))),
        "populations_without_neigh_flow": sum(1 for r in pop_c12 if is_na_val(r.get("neigh_flow"))),
        "pooled_per_individual_sd": pooled,
        "n_boot": N_BOOT,
        "flexivity_outcomes": {
            "flex": {
                "n_features": 30,
                "languages_cov60": sum(1 for r in d_c12 if int(r["flex_coded"]) >= 18),
                "languages_cov50": sum(1 for r in d_c12 if int(r["flex_coded"]) >= 15),
                "languages_cov75": sum(1 for r in d_c12 if int(r["flex_coded"]) >= 23),
                "coded_per_language": _describe_series([int(r["flex_coded"]) for r in d_c12]),
            },
            "flex26": {
                "n_features": 26,
                "languages_cov60": sum(1 for r in d_c12 if int(r["flex26_coded"]) >= 16),
                "coded_per_language": _describe_series([int(r["flex26_coded"]) for r in d_c12]),
            },
            "irreg13": {
                "n_features": 13,
                "languages_cov60": sum(1 for r in d_c12 if int(r["irreg13_coded"]) >= 8),
                "coded_per_language": _describe_series([int(r["irreg13_coded"]) for r in d_c12]),
            },
        },
        "population_spearman": _spearman_matrix_dict(pi_rows, spearman_cols),
    }
    (out_dir / "prep_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    update_existing_language_data(out_dir=out_dir)

    # Item-level tables for latent_sem
    prepare_item_tables(out_dir=out_dir)

    # Shared all-GeLaTo-population summary in analyses/common/data/
    gel_sum_path = COMMON_DATA_DIR / "gelato_populations_summary.csv"
    if not gel_sum_path.exists() or "--refresh-gelato-summary" in sys.argv:
        build_gelato_populations_summary(out_path=gel_sum_path)

    print("Prepared Grambank canonical tables in:", out_dir)


if __name__ == "__main__":
    if "--demog-only" in sys.argv:
        update_existing_language_data()
    else:
        main()
