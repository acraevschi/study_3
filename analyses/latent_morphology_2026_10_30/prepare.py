"""Latent-variable and multi-item morphological outcomes: data preparation (EXPLORATORY).

Builds the item-level binary data tables for the 1-factor (1PL, 2PL IRT/MIMIC, and
hierarchical varying-slopes) and 2-factor (Breadth + Irregularity/Flexivity) models on
the 2026-10-20 GeLaTo-Grambank links.

Reads (read-only):
  analyses/admixture_links_2026_10_21/language_data.csv
  outputs/pcfp_v2_links_2026_10_20/typology/grambank_inflection.csv
  outputs/pcfp_v2_links_2026_10_20/typology/feature_set.json
  external/grambank/cldf/parameters.csv
  external/grambank/cldf/codes.csv
  external/grambank/cldf/values.csv

Writes, next to this script:
  items_metadata.csv     one row per binary item (12 cat12 categories + 26 flex26 features,
                         with domain, testlet cluster, factor assignment, and in_irreg13 flag)
  item_observations.csv  long-format observed binary cells (glottocode, item_id, value 0/1)
  language_table.csv     language-level table aligned with admixture_links_2026_10_21
  prep_summary.json      item base rates, item-rest correlations, and dataset counts

Run:
  /usr/bin/python3 analyses/latent_morphology_2026_10_30/prepare.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ADM_DIR = ROOT / "analyses/admixture_links_2026_10_21"
TYP_DIR = ROOT / "outputs/pcfp_v2_links_2026_10_20/typology"
GB_DIR = ROOT / "external/grambank/cldf"

IRREG13_IDS = {
    "GB039", "GB041", "GB095", "GB096", "GB098", "GB099",
    "GB109", "GB110", "GB111", "GB300", "GB321", "GB402", "GB403",
}
OVERLAP4_IDS = {"GB170", "GB171", "GB172", "GB198"}

# Sub-domain classification for the 26 non-overlapping Flexivity features (README §12):
#   - noun_irreg: noun number allomorphy (GB039), noun number suppletion (GB041),
#                 unpredictable noun gender/class assignment (GB321)
#   - arg_split_classes: core-argument marking splits by TAM (GB095), verb class (GB096),
#                        person (GB098), and conjugation classes (GB111)
#   - verb_suppletion_stem: verb stem alternation by person (GB099), suppletion for
#                           participant number (GB109), TAM (GB110), 'give' (GB300),
#                           'see' (GB402), 'come' (GB403)
#   - gender_nc: gender/noun-class presence and semantic/phonological assignment
#                (GB030, GB051, GB052, GB053, GB054, GB192)
#   - classifiers: demonstrative, numeral, possessive, and verbal classificatory stems
#                  (GB038, GB057, GB058, GB116)
#   - evaluative_anim: productive diminutive/augmentative and verbal animacy marking
#                      (GB177, GB187, GB188)
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


def main() -> None:
    # 1. Load declared 12-category feature set
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

    # 2. Load Grambank parameters and verify binary coding of all sources and flexivity features
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

    # 3. Load language_data.csv from admixture_links_2026_10_21 and grambank_entry mapping
    with open(ADM_DIR / "language_data.csv", encoding="utf-8") as f:
        lang_rows = [r for r in csv.DictReader(f) if int(r["n_coded"]) >= 8]
    assert len(lang_rows) == 202

    gb_entry_map: dict[str, str] = {}
    with open(TYP_DIR / "grambank_inflection.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["glottocode"] and r["grambank_entry"]:
                gb_entry_map[r["glottocode"]] = r["grambank_entry"]

    for r in lang_rows:
        gc = r["glottocode"]
        assert gc in gb_entry_map, f"Missing grambank_entry for {gc}"
        r["grambank_entry"] = gb_entry_map[gc]

    needed_entries = {r["grambank_entry"] for r in lang_rows}

    # 4. Read Grambank values.csv for all 225 languages and needed features
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

    # 5. Build items_metadata.csv (12 cat12 items + 26 flex26 items = 38 items total)
    # Count items per domain to identify multi-item testlets (domains with >= 2 items)
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

    # 6. Build long-format item_observations.csv and verify exact match with language_data.csv counts
    obs_rows: list[dict[str, str | int]] = []
    by_lang_item: dict[tuple[str, str], int] = {}

    for r in lang_rows:
        gc = r["glottocode"]
        entry = r["grambank_entry"]

        # 6a. 12 GBI categories (OR rule: 1 if any source is '1', 0 if all sources are '0')
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
        assert c12_pres == int(r["n_present"]) and c12_coded == int(r["n_coded"]), (
            f"cat12 count mismatch for {gc}: got {c12_pres}/{c12_coded}, "
            f"expected {r['n_present']}/{r['n_coded']}"
        )

        # 6b. 26 Flexivity features (and 13 Irregularity subset)
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
        assert f26_pres == int(r["flex26_present"]) and f26_coded == int(r["flex26_coded"]), (
            f"flex26 count mismatch for {gc}: got {f26_pres}/{f26_coded}, "
            f"expected {r['flex26_present']}/{r['flex26_coded']}"
        )
        assert ir13_pres == int(r["irreg13_present"]) and ir13_coded == int(r["irreg13_coded"]), (
            f"irreg13 count mismatch for {gc}: got {ir13_pres}/{ir13_coded}, "
            f"expected {r['irreg13_present']}/{r['irreg13_coded']}"
        )

    # 7. Write CSV outputs
    with open(OUT / "items_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(items_meta[0].keys()))
        w.writeheader()
        w.writerows(items_meta)

    with open(OUT / "item_observations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(obs_rows[0].keys()))
        w.writeheader()
        w.writerows(obs_rows)

    with open(OUT / "language_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lang_rows[0].keys()))
        w.writeheader()
        w.writerows(lang_rows)

    # 8. Compute descriptive item summary on the main set (N=202 for cat12, N=186/175 for joint)
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
    (OUT / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
