"""Reproduce the GeLaTo/MGN coverage audit from pinned local snapshots.

Run with the bundled Python runtime. No network, fitting, or pipeline mutations.
Catalogue compatibility and genetic evidence of contact are separate outputs.
"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
GENE = BASE / "sources/gelato_c625fdc"
EXPANDED = BASE / "sources/zenodo_15263706/geneticAdmixture-linguisticDiffusion"
OUT = BASE / "outputs"
MISSING = {"", "NA", "NaN", "ND"}


def read_csv(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    mgn = read_csv(ROOT / "mgn_modeling_dataset.csv")
    languages = {}
    by_pos = defaultdict(set)
    by_language = defaultdict(list)
    for row in mgn:
        code = row["glottocode"]
        if code in languages:
            assert all(languages[code][field] == row[field]
                       for field in ["lang", "iso_sanitized", "language_name", "family"])
        languages[code] = row
        by_pos[row["pos"]].add(code)
        by_language[code].append(row)
    assert len(languages) == len({row["iso_sanitized"] for row in languages.values()}) == 64
    lookup = {row["glottocode"]: row for row in read_csv(
        ROOT / "data_sources/Glottolog_lookup_table_Heti_edition.tsv", "\t")}
    current = read_csv(GENE / "cldf/populations.csv")
    expanded = read_csv(EXPANDED / "tables/tableS1.csv")
    assert len({row["ID"] for row in current}) == len(current)
    assert len({row["Population"] for row in expanded}) == len(expanded)
    expanded_by_pop = {row["Population"]: row for row in expanded}
    scalars = {row["PopName"]: row for row in read_csv(GENE / "datasets/HumanOrigins_AutosomalSNP/data.csv")}

    population_rows = []
    for panel, rows in [("main_397", current), ("expanded_558", expanded)]:
        for row in rows:
            code = row["Glottocode"] if panel == "main_397" else row["GeLaTo_Glottocode"]
            name = row["Name"] if panel == "main_397" else row["Population"]
            n = int(row["samplesize"] if panel == "main_397" else row["N_Individuals"])
            match, status, basis = "", "unmatched", ""
            if code in languages:
                match, status, basis = code, "exact", "identical curated base Glottocode"
            elif lookup.get(code, {}).get("Language_level_ID") in languages:
                match = lookup[code]["Language_level_ID"]
                status, basis = "descendant_candidate", "local Glottolog lookup Language_level_ID; community compatibility unverified"
            elif (code == "gree1276" and name.startswith("Greek_")
                  and expanded_by_pop.get(name, {}).get("TLI_Glottocode") == "mode1248"):
                match, status = "mode1248", "curated_greek_candidate"
                basis = "published Table S1 maps this Greek population to Modern Greek in GBI and TLI"
            population_rows.append({
                "panel": panel, "population": name, "base_glottocode": code,
                "n_individuals": n, "at_least_five": n >= 5,
                "match_status": status, "candidate_mgn_glottocode": match,
                "mgn_iso": languages[match]["iso_sanitized"] if match else "",
                "mgn_language": languages[match]["language_name"] if match else "",
                "mapping_basis": basis,
                "source": row.get("Source", row.get("Reference", "")),
                "curation_note": row.get("curation_notes_linguistics", ""),
                "latitude": row.get("Latitude", ""), "longitude": row.get("Longitude", ""),
                "median_FST_neighbor": scalars.get(name, {}).get("MedianFSTAdjustedNeighbors", "") if panel == "main_397" else "",
                "Ne": scalars.get(name, {}).get("harmonicMean", "") if panel == "main_397" else "",
            })
    write_csv("population_crosswalk.csv", population_rows)

    def coverage(panel, mode="exact", min_n=0):
        selected = [row for row in population_rows if row["panel"] == panel
                    and row["candidate_mgn_glottocode"] and row["n_individuals"] >= min_n
                    and (mode != "exact" or row["match_status"] == "exact")]
        codes = {row["candidate_mgn_glottocode"] for row in selected}
        return {
            "languages": len(codes), "populations": len(selected),
            "individuals": sum(row["n_individuals"] for row in selected),
            "by_pos": {pos: len(codes & available) for pos, available in by_pos.items()},
            "by_family": dict(Counter(languages[code]["family"] for code in sorted(codes))),
            "by_macro_area": dict(Counter(languages[code]["macro_area"] for code in sorted(codes))),
            "cell_pair_rows": sum(len(by_language[code]) for code in codes),
            "language_list": sorted(languages[code]["language_name"] for code in codes),
        }

    coverage_rows = {}
    for panel in ["main_397", "expanded_558"]:
        for mode in ["exact", "screened_candidates"]:
            for minimum in [0, 5]:
                coverage_rows[f"{panel}_{mode}_min{minimum}"] = coverage(panel, mode, minimum)

    tree_rows = []
    for path in sorted((BASE / "sources/trees").glob("*.gz")):
        text = gzip.decompress(path.read_bytes()).decode()
        labels = re.search(r"Taxlabels(.*?);", text, re.S | re.I).group(1).split()
        codes = {label.strip("'").split("_")[0] for label in labels}
        assert len(codes) == len(labels)
        tree_rows.append({"file": path.name, "tips": len(labels),
                          "mgn_exact_tips": len(codes & languages.keys()),
                          "missing_mgn_codes": sorted(languages.keys() - codes)})

    language_rows = []
    for code, row in sorted(languages.items(), key=lambda item: item[1]["language_name"]):
        result = {"mgn_iso": row["iso_sanitized"], "mgn_source_code": row["lang"],
                  "glottocode": code, "language": row["language_name"],
                  "family": row["family"], "macro_area": row["macro_area"],
                  "parts_of_speech": ";".join(sorted({r["pos"] for r in by_language[code]})),
                  "cell_pair_rows": len(by_language[code])}
        for panel in ["main_397", "expanded_558"]:
            hits = [r for r in population_rows if r["panel"] == panel and r["candidate_mgn_glottocode"] == code]
            exact = [r for r in hits if r["match_status"] == "exact"]
            result.update({f"{panel}_exact_populations": len(exact),
                           f"{panel}_exact_individuals": sum(r["n_individuals"] for r in exact),
                           f"{panel}_exact_populations_min5": sum(r["n_individuals"] >= 5 for r in exact),
                           f"{panel}_all_candidate_populations": len(hits),
                           f"{panel}_population_names": ";".join(r["population"] for r in hits)})
        language_rows.append(result)
    write_csv("mgn_language_coverage.csv", language_rows)

    pairs = read_csv(EXPANDED / "tables/tableS2.csv")
    contact_rows = []
    for row in pairs:
        exact_code = expanded_by_pop.get(row["TargetPop"], {}).get("GeLaTo_Glottocode", "")
        feature_matches = {row["FinalTargetGlottocodeGBI"], row["FinalTargetGlottocodeTLI"]} & languages.keys()
        if exact_code in languages or feature_matches:
            code = exact_code if exact_code in languages else next(iter(feature_matches))
            source_code = expanded_by_pop.get(row["AlterFamilySourcePopGelatoOrManual"], {}).get("GeLaTo_Glottocode", "")
            contact_rows.append({
                "pair_id": row["pair_id"], "target_population": row["TargetPop"],
                "mgn_target": languages[code]["language_name"], "mgn_glottocode": code,
                "mapping": "exact_target_population" if exact_code in languages else "published_linguistic_proxy_only",
                "n_genetic_individuals": expanded_by_pop.get(row["TargetPop"], {}).get("N_Individuals", ""),
                "source_population": row["AlterFamilySourcePopGelatoOrManual"],
                "source_base_glottocode": source_code,
                "genetic_source_proxy_also_exact_mgn": source_code in languages,
                "published_source_language_GBI": row["FinalSourceGlottocodeGBIForPlot"],
                "published_source_language_TLI": row["FinalSourceGlottocodeTLIForPlot"],
                "source_clade": row["LinguisticCladeFromWhichToResampleSource"],
                "evidence_source": row["Reference"], "F3": row["MostNegF3"],
                "F3_Z": row["MostNegF3Z"],
                "F3_Z_below_minus3": row["MostNegF3Z"] not in MISSING and float(row["MostNegF3Z"]) < -3,
            })
    write_csv("published_contact_case_overlap.csv", contact_rows)
    longlist = read_csv(EXPANDED / "output/longlists/longlist_for_manual_curation.csv")
    long_matches = [row for row in longlist if row["TargetPopGlottocodeBase"] in languages]
    write_csv("unvalidated_candidate_contact_overlap.csv", long_matches)

    ids = read_csv(EXPANDED / "input/MegaAdmixtureCatalogue/ADMIXTURE/GeneticInfoID.csv")
    assert all(int(row["Order"]) == i + 1 for i, row in enumerate(ids))
    groups = defaultdict(list)
    for index, row in enumerate(ids):
        groups[row["Population"]].append(index)
    assert len(groups) == 558
    assert all(len(indices) == int(expanded_by_pop[name]["N_Individuals"]) for name, indices in groups.items())
    diagnostics, q23_rows = [], []
    for k in range(12, 31):
        path = EXPANDED / f"input/MegaAdmixtureCatalogue/ADMIXTURE/best_runs/GelatoHO_mergedSetMarchBEDnorelatives_pruned_autosomes_K{k}.Q"
        q = np.loadtxt(path)
        assert q.shape == (len(ids), k)
        assert np.all(q >= 0) and np.all(q <= 1) and np.max(abs(q.sum(axis=1) - 1)) < 0.00001
        for name, indices in groups.items():
            mean = q[indices].mean(axis=0)
            order = np.argsort(mean)[::-1]
            metadata = expanded_by_pop[name]
            if k == 23:
                q23_rows.append({"population": name, "curated_glottocode": metadata["GeLaTo_Glottocode"],
                                 "n_individuals": len(indices), **{f"q{i+1}": float(value) for i, value in enumerate(mean)}})
            if metadata["GeLaTo_Glottocode"] in languages:
                diagnostics.append({"population": name, "glottocode": metadata["GeLaTo_Glottocode"],
                    "n_individuals": len(indices), "K": k, "largest_component_id": int(order[0] + 1),
                    "largest_share": float(mean[order[0]]), "second_component_id": int(order[1] + 1),
                    "second_share": float(mean[order[1]]), "two_largest_sum": float(mean[order[:2]].sum()),
                    "ancestry_heterogeneity": float(1 - (mean**2).sum()),
                    "mean_individual_ancestry_heterogeneity": float((1 - (q[indices]**2).sum(axis=1)).mean()),
                    "two_component_screen_only": bool(mean[order[:2]].sum() > 0.7 and mean[order[1]] > 0.05)})
    write_csv("population_ancestry_K12_K30_diagnostics.csv", diagnostics)
    write_csv("population_ancestry_K23_components.csv", q23_rows)
    stability = []
    for name in sorted({row["population"] for row in diagnostics}):
        rows = [row for row in diagnostics if row["population"] == name]
        stability.append({"population": name, "glottocode": rows[0]["glottocode"], "n_individuals": rows[0]["n_individuals"],
            "min_largest_share": min(row["largest_share"] for row in rows),
            "max_largest_share": max(row["largest_share"] for row in rows),
            "min_heterogeneity": min(row["ancestry_heterogeneity"] for row in rows),
            "max_heterogeneity": max(row["ancestry_heterogeneity"] for row in rows),
            "number_of_K_passing_component_screen": sum(row["two_component_screen_only"] for row in rows)})
    write_csv("ancestry_K_sensitivity.csv", stability)

    raw_mgn = read_csv(ROOT / "mgn_data/results-final/all-accuracies.csv")
    summary = {
        "audit_date": "2026-10-01", "mgn_languages": len(languages), "mgn_cell_pair_rows": len(mgn),
        "mgn_by_pos": {pos: len(codes) for pos, codes in by_pos.items()},
        "upstream_mgn_language_codes": sorted({r['lang'] for r in raw_mgn}),
        "upstream_mgn_codes_outside_current_sample": sorted({r['lang'] for r in raw_mgn} - {r['lang'] for r in mgn}),
        "main_panel": {"populations": len(current), "valid_glottocodes": len({r['Glottocode'] for r in current} - MISSING),
                       "individuals": sum(int(r['samplesize']) for r in current)},
        "expanded_panel": {"populations": len(expanded), "valid_glottocodes": len({r['GeLaTo_Glottocode'] for r in expanded} - MISSING),
                           "individuals": sum(int(r['N_Individuals']) for r in expanded)},
        "coverage": coverage_rows, "trees": tree_rows,
        "published_contact_overlap": {
            "exact_target_languages": len({r['mgn_glottocode'] for r in contact_rows if r['mapping'] == 'exact_target_population'}),
            "exact_target_pairs": sum(r['mapping'] == 'exact_target_population' for r in contact_rows),
            "including_published_proxy_target_languages": len({r['mgn_glottocode'] for r in contact_rows}),
            "including_published_proxy_target_pairs": len(contact_rows),
            "exact_target_pairs_F3_Z_below_minus3": sum(r['mapping'] == 'exact_target_population' and r['F3_Z_below_minus3'] for r in contact_rows),
        },
        "unvalidated_longlist_overlap": {"rows": len(long_matches), "target_languages": len({r['TargetPopGlottocodeBase'] for r in long_matches})},
        "ancestry_Q_validation": {"K_values": list(range(12, 31)), "individual_rows_per_K": len(ids), "population_means": len(groups)},
        "limitations": ["Exact identifiers do not establish community representativeness.",
                        "Screened descendants and Greek mappings are candidates, not accepted analysis rows.",
                        "Ancestry heterogeneity is a diagnostic, not a non-native ancestry measure.",
                        "A positive F3 or absent published contact pair is not evidence of no admixture.",
                        "Tree coverage counts taxon labels only; no ancestral reconstruction or model fitting was performed."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    sources = [ROOT / "mgn_modeling_dataset.csv", ROOT / "mgn_language_map.json",
               ROOT / "data_sources/Glottolog_lookup_table_Heti_edition.tsv",
               ROOT / "mgn_data/results-final/all-accuracies.csv", ROOT / "mgn_data/functions.R"]
    sources += sorted(path for path in (BASE / "sources").rglob("*") if path.is_file())
    manifest = {"GeLaTo_commit": "c625fdcf0225142cc03ae3a1635edf322e9c7778", "GeLaTo_url": "https://github.com/gelato-org/gelato-data",
                "Zenodo_record": "https://zenodo.org/records/15263706",
                "tree_release_url": "https://github.com/rbouckaert/global-language-tree-pipeline/releases",
                "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources]}
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"coverage": {key: {k: v for k, v in value.items() if k != 'language_list'} for key, value in coverage_rows.items()},
                      "contacts": summary['published_contact_overlap'], "trees": tree_rows}, indent=2))


if __name__ == "__main__":
    main()
