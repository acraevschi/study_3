"""Admixture -> Grambank inflection: data preparation for the 2026-10-20 links (EXPLORATORY).

Successor of analyses/draft_admixture_model_2026_10_10/prepare.py (left unchanged). It
lifts the ancestry firewall for this folder only; the admixture values were unblinded on
2026-10-10, so nothing here is a blind or pre-registered test.

What changes against the draft:
  * links, languages and the three sensitivity sets come from
    outputs/pcfp_v2_links_2026_10_20/typology/analysis_links.csv (rule-accepted links,
    Graff stand-ins, the user-reviewed decision sheet);
  * new population measures: `het` (between-individual spread inside a population) and
    `f3_z`, `f3_est` (f3.R, must be run first);
  * population measures get a bootstrap standard error over their individuals, so the
    number of individuals enters the exposure model for them too.

Writes, next to this script: language_data.csv, individuals.csv, populations.csv,
phylo_tree.nwk, prep_summary.json.

Run: .venv/bin/python analyses/admixture_links_2026_10_21/prepare.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DEMOG_DIR = OUT / "demography_sources"

# Group-level / retired-ISO mappings for the two group_map_down languages whose
# parent group ISO carries measured L1 population and polygon area in Bromham/Koplenig.
GROUP_ISO_MAP = {
    "minh1238": "mjg",  # Mangghuer <- Tu (tuuu1240, ISO mjg)
    "tanz1241": "ngo",  # Tanzanian Ngoni <- Ngoni (ngon1269, retired ISO ngo)
}
PLURICENTRIC_AREA_MEMBERS = {
    "hbs": ["hrv", "srp"],  # Serbian-Croatian-Bosnian: Koplenig Rangesize on hrv + srp
}


def resolve_demography_rows(glottocodes: list[str]) -> list[dict[str, object]]:
    """Resolve audited L1 speaker population and measured range area (km^2) per Glottocode.

    Invariants:
      1. No population is ever invented or proxy-substituted from another Grambank language.
         Only direct Glottocode matches in global_demographic_registry.csv and the two
         group_map_down parent ISOs (minh1238 -> mjg, tanz1241 -> ngo) are accepted.
      2. No family/macroarea-imputed range area ever enters area_km2. Only measured polygon
         areas from Bromham (lang_polygon.area_lang) or Koplenig (Rangesize, including the
         hrv + srp sum for pluricentric hbs) are accepted; the 3 Ethnologue-multiISO rows
         without a polygon in Bromham or Koplenig (avl, ayl, zlm) get area_km2 = None.
    """
    with open(DEMOG_DIR / "global_demographic_registry.csv", encoding="utf-8-sig") as f:
        reg_by_gc = {r["glottocode"]: r for r in csv.DictReader(f) if r.get("glottocode")}
    with open(DEMOG_DIR / "bromham_extracted.csv", encoding="utf-8-sig") as f:
        brom_by_iso = {r["iso_639_3"]: r for r in csv.DictReader(f) if r.get("iso_639_3")}
    with open(DEMOG_DIR / "ethnologue_population_data.csv", encoding="utf-8-sig") as f:
        kop_by_iso = {r["ISO"]: r for r in csv.DictReader(f) if r.get("ISO")}

    out: list[dict[str, object]] = []
    for gc in glottocodes:
        iso = ""
        pop_l1: float | None = None
        pop_src = ""
        area_km2: float | None = None
        area_src = ""

        if gc in GROUP_ISO_MAP:
            iso = GROUP_ISO_MAP[gc]
            b = brom_by_iso[iso]
            pop_l1 = float(b["population_l1"])
            pop_src = "Bromham_group_ISO"
            area_km2 = float(b["area_km2"])
            area_src = "Bromham_group_ISO"
        elif gc in reg_by_gc:
            rr = reg_by_gc[gc]
            iso = rr["iso_639_3"]
            if rr.get("population_l1") not in (None, "", "nan", "NaN"):
                pop_l1 = float(rr["population_l1"])
                pop_src = rr["population_source"]
            b_area = brom_by_iso.get(iso, {}).get("area_km2")
            k_area = kop_by_iso.get(iso, {}).get("Rangesize")
            if b_area not in (None, "", "NA", "0"):
                area_km2 = float(b_area)
                area_src = "Bromham"
            elif k_area not in (None, "", "NA", "0"):
                area_km2 = float(k_area)
                area_src = "Koplenig"
            elif iso in PLURICENTRIC_AREA_MEMBERS:
                members = PLURICENTRIC_AREA_MEMBERS[iso]
                area_km2 = sum(float(kop_by_iso[m]["Rangesize"]) for m in members)
                area_src = "Aggregate_Koplenig"

        has_pop = 1 if (pop_l1 is not None and pop_l1 > 0) else 0
        has_area = 1 if (has_pop == 1 and area_km2 is not None and area_km2 > 0) else 0
        out.append({
            "glottocode": gc,
            "iso_639_3": iso,
            "population_l1": round(pop_l1, 6) if pop_l1 is not None else "",
            "population_source": pop_src,
            "area_km2": round(area_km2, 6) if area_km2 is not None else "",
            "area_source": area_src,
            "log_pop": round(math.log(pop_l1), 6) if pop_l1 is not None and pop_l1 > 0 else "",
            "log_area": round(math.log(area_km2), 6) if area_km2 is not None and area_km2 > 0 else "",
            "has_pop": has_pop,
            "has_area": has_area,
        })
    return out


def update_existing_language_data() -> None:
    """Attach audited demographic columns to existing language_data.csv and prep_summary.json."""
    ld_path = OUT / "language_data.csv"
    with open(ld_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    demog_by_gc = {str(r["glottocode"]): r for r in resolve_demography_rows([r["glottocode"] for r in rows])}
    demog_cols = ["iso_639_3", "population_l1", "population_source", "area_km2", "area_source",
                  "log_pop", "log_area", "has_pop", "has_area"]
    base_cols = [c for c in rows[0].keys() if c not in demog_cols]
    fieldnames = base_cols + demog_cols
    merged_rows = []
    for r in rows:
        dm = demog_by_gc[r["glottocode"]]
        merged = {c: r[c] for c in base_cols}
        for c in demog_cols:
            merged[c] = dm[c]
        merged_rows.append(merged)

    with open(ld_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged_rows)

    c12 = [r for r in merged_rows if int(r["n_coded"]) >= 8]
    assert sum(int(r["has_pop"]) for r in c12 if int(r["in_main"]) == 1) == 195
    assert sum(int(r["has_area"]) for r in c12 if int(r["in_main"]) == 1) == 192

    # Compute correlation between log_pop and log_area on the 192 main cat12 languages
    both_c12 = [r for r in c12 if int(r["in_main"]) == 1 and int(r["has_area"]) == 1]
    lp = [float(r["log_pop"]) for r in both_c12]
    la = [float(r["log_area"]) for r in both_c12]
    mp, ma = sum(lp) / len(lp), sum(la) / len(la)
    cov = sum((x - mp) * (y - ma) for x, y in zip(lp, la)) / (len(lp) - 1)
    sp = (sum((x - mp) ** 2 for x in lp) / (len(lp) - 1)) ** 0.5
    sa = (sum((y - ma) ** 2 for y in la) / (len(la) - 1)) ** 0.5
    r_pa = cov / (sp * sa)

    from collections import Counter
    pop_src_cnt = dict(Counter(r["population_source"] or "MISSING" for r in c12 if int(r["in_main"]) == 1))
    area_src_cnt = dict(Counter(r["area_source"] or "MISSING" for r in c12 if int(r["in_main"]) == 1))
    missing_pop_gcs = [r["glottocode"] for r in c12 if int(r["in_main"]) == 1 and int(r["has_pop"]) == 0]
    missing_area_only_gcs = [r["glottocode"] for r in c12 if int(r["in_main"]) == 1 and int(r["has_pop"]) == 1 and int(r["has_area"]) == 0]

    summ_path = OUT / "prep_summary.json"
    summ = json.loads(summ_path.read_text(encoding="utf-8"))
    summ["demography"] = {
        "sets_cat12": {
            s: {
                "languages_total": sum(1 for r in c12 if int(r[f"in_{s}"]) == 1),
                "languages_has_pop": sum(int(r["has_pop"]) for r in c12 if int(r[f"in_{s}"]) == 1),
                "languages_has_pop_and_area": sum(int(r["has_area"]) for r in c12 if int(r[f"in_{s}"]) == 1),
            }
            for s in ("main", "no_doubtful_language", "no_other_language_stand_ins", "strict")
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
    print(json.dumps(summ["demography"], indent=2))

ARCH = ROOT / "analyses/gelato_feasibility_2026_10_01/sources/zenodo_15263706/geneticAdmixture-linguisticDiffusion"
ADM = ARCH / "input/MegaAdmixtureCatalogue/ADMIXTURE"
TYP = ROOT / "outputs/pcfp_v2_links_2026_10_20/typology"
KS = list(range(12, 31))
SETS = ["main", "no_doubtful_language", "no_other_language_stand_ins", "strict"]
N_BOOT = 400
SEED = 20261021

sys.path.insert(0, str(ROOT / "external/languages-of-the-world/src"))


def entropy(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-12, 1)
    return -(q * np.log(q)).sum(axis=-1)


def pop_stats(q_by_k: list[np.ndarray], idx: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Population measures, averaged over K, for one population.
    q_by_k: per K an (n, K) matrix of individual ancestry proportions.
    idx: (B, n) resampled individual indices (bootstrap) or None (the observed sample).
    Returns arrays of shape (B,) (or (1,) when idx is None)."""
    n = q_by_k[0].shape[0]
    acc = {k: 0.0 for k in ("pop_ent", "het", "support", "amount")}
    for q in q_by_k:
        qs = q[idx] if idx is not None else q[None]                       # (B, n, K)
        mean = qs.mean(axis=1)                                            # (B, K)
        h_pop = entropy(mean)
        h_ind = entropy(qs).mean(axis=1)
        # plug-in Jensen gap H(mean) - mean H is low by the factor (n-1)/n; undo it
        het = (h_pop - h_ind) * n / (n - 1) if n >= 2 else np.full_like(h_pop, np.nan)
        srt = -np.sort(-mean, axis=1)
        flags = np.stack([(srt[:, 0] + srt[:, 1] > t) & (srt[:, 1] > 0.05) for t in (0.7, 0.8, 0.9)])
        acc["pop_ent"] = acc["pop_ent"] + h_pop
        acc["het"] = acc["het"] + het
        acc["support"] = acc["support"] + flags.mean(0)
        acc["amount"] = acc["amount"] + (flags * srt[:, 1]).mean(0)
    out = {k: v / len(q_by_k) for k, v in acc.items()}
    out["sqrt_het"] = np.sqrt(np.maximum(0.0, out["het"]))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)

    # --- links and sets (all Grambank-linked languages; coverage applied in fit.R) -------
    al = pd.read_csv(TYP / "analysis_links.csv", keep_default_na=False, dtype=str)
    al = al[al.in_grambank == "True"]
    assert al.groupby(["analysis", "population"]).glottocode.nunique().max() == 1
    main_links = al[al.analysis == "main"][["population", "glottocode", "link_basis", "accepted_by", "sensitivity_flags", "meets_coverage_main"]]
    assert main_links.population.is_unique
    in_set = {s: set(al[al.analysis == s].population) for s in SETS}
    assert all(in_set[s] <= in_set["main"] for s in SETS)

    # --- outcome: all Grambank-linked languages of the typology stage (225 in main) ------
    g = pd.read_csv(TYP / "grambank_inflection.csv", keep_default_na=False, dtype=str)
    g = g[(g.in_grambank == "True") & (g.n_populations_main.astype(int) > 0)].copy()
    for c in ("n_present", "n_coded"):
        g[c] = g[c].astype(int)
    assert set(g.glottocode) == set(main_links.glottocode), "outcome languages and linked languages differ"
    assert ((g.n_coded >= 8) == (g.meets_coverage_main == "True")).all()

    # --- Grambank Flexivity outcomes (30 full, 26 non-overlapping, 13 irregularity) -------
    gb_dir = ROOT / "external/grambank/cldf"
    gb_params = pd.read_csv(gb_dir / "parameters.csv", keep_default_na=False, dtype=str)
    gb_codes = pd.read_csv(gb_dir / "codes.csv", keep_default_na=False, dtype=str)
    flex_ids = set(gb_params.loc[gb_params.Flexivity == "1", "ID"])
    assert len(flex_ids) == 30
    for pid in flex_ids:
        assert set(gb_codes.loc[gb_codes.Parameter_ID == pid, "Name"]) == {"0", "1"}
    flex26_ids = flex_ids - {"GB170", "GB171", "GB172", "GB198"}
    irreg13_ids = {"GB039", "GB041", "GB095", "GB096", "GB098", "GB099", "GB109",
                   "GB110", "GB111", "GB300", "GB321", "GB402", "GB403"}
    assert len(flex26_ids) == 26 and irreg13_ids <= flex26_ids and len(irreg13_ids) == 13
    gb_vals = pd.read_csv(gb_dir / "values.csv", keep_default_na=False, dtype=str)
    gb_vals = gb_vals[(gb_vals.Parameter_ID.isin(flex_ids)) & (gb_vals.Value.isin(["0", "1"]))].copy()
    gb_vals["v"] = gb_vals.Value.astype(int)
    for prefix, fset in (("flex", flex_ids), ("flex26", flex26_ids), ("irreg13", irreg13_ids)):
        sub = gb_vals[gb_vals.Parameter_ID.isin(fset)].groupby("Language_ID")["v"]
        g[f"{prefix}_present"] = g.grambank_entry.map(sub.sum()).fillna(0).astype(int)
        g[f"{prefix}_coded"] = g.grambank_entry.map(sub.count()).fillna(0).astype(int)

    # --- individuals and Q matrices ------------------------------------------------------
    info = pd.read_csv(ADM / "GeneticInfoID.csv", keep_default_na=False)
    assert info.Order.astype(int).tolist() == list(range(1, len(info) + 1))
    info = info[["Population"]].rename(columns={"Population": "population"}).reset_index(drop=True)
    assert set(main_links.population) <= set(info.population)
    Q = {}
    for k in KS:
        q = np.loadtxt(ADM / f"best_runs/GelatoHO_mergedSetMarchBEDnorelatives_pruned_autosomes_K{k}.Q")
        assert q.shape == (len(info), k) and np.allclose(q.sum(1), 1, atol=1e-3)
        Q[k] = q / q.sum(1, keepdims=True)

    # --- individual measures (K-averaged) -------------------------------------------------
    ent = np.stack([entropy(Q[k]) for k in KS])                 # (K, N)
    nonmax = np.stack([1 - Q[k].max(1) for k in KS])
    ind = info.assign(ind=np.arange(len(info)), ent=ent.mean(0), nonmax=nonmax.mean(0),
                      ent_K12=ent[KS.index(12)], ent_K30=ent[KS.index(30)])
    ind = ind.merge(main_links[["population", "glottocode"]], on="population")   # main-set individuals only
    ind.round(6).to_csv(OUT / "individuals.csv", index=False)

    # --- population measures with bootstrap errors -----------------------------------------
    # Bootstrap the 335 cat12-main populations first in sorted order so their RNG stream
    # and pooled fallback SDs remain bit-for-bit identical to the 202-language build.
    cat12_pops = set(main_links.loc[main_links.meets_coverage_main == "True", "population"])
    rows = []
    for pop_list in (sorted(cat12_pops), sorted(set(main_links.population) - cat12_pops)):
        for pop_name in pop_list:
            grp = ind[ind.population == pop_name]
            ii = grp.ind.to_numpy()
            n = len(ii)
            qk = [Q[k][ii] for k in KS]
            obs = {k: float(v[0]) for k, v in pop_stats(qk).items()}
            boot = pop_stats(qk, rng.integers(0, n, size=(N_BOOT, n)))
            rows.append({"population": pop_name, "n_individuals": n, **obs,
                         **{f"{k}_se": float(np.nanstd(v, ddof=1)) if n >= 2 else np.nan for k, v in boot.items()}})
    pop = pd.DataFrame(rows).sort_values("population", kind="mergesort").reset_index(drop=True)
    # an SE from fewer than 3 individuals is unreliable: use the pooled per-individual SD / sqrt(n),
    # as the draft did for the language-level measurement error
    pooled = {}
    for m in ("pop_ent", "het", "support", "amount", "sqrt_het"):
        big = pop[pop.population.isin(cat12_pops) & (pop.n_individuals >= 3)]
        pooled[m] = float((big[f"{m}_se"] * np.sqrt(big.n_individuals)).median())
        small = pop.n_individuals < 3
        pop.loc[small, f"{m}_se"] = pooled[m] / np.sqrt(pop.loc[small, "n_individuals"])
    pop.loc[pop.n_individuals < 2, ["het", "het_se", "sqrt_het", "sqrt_het_se"]] = np.nan

    # Graff et al. 2025 curated admixture targets (Table S2; contact between families)
    s2 = pd.read_csv(ARCH / "tables/tableS2.csv", keep_default_na=False, dtype=str)
    def names(v: str) -> set[str]:
        # a bracketed comment may contain commas ("Aleut_Tlingit (admixture detected ..., but ...)")
        return {p.strip().replace(" ", "_") for p in re.sub(r"\([^)]*\)", "", v).split(",") if p.strip()}
    tpops = set().union(*(names(v) for v in s2.TargetPop), *(names(v) for v in s2.AlternativeTargetPop))
    pop["graff_target"] = pop.population.isin(tpops).astype(int)

    # GeLaTo neighbour FST: -log median FST to populations within 1,000 km (drifted ones excluded)
    fst = pd.read_csv(ROOT / "analyses/gelato_feasibility_2026_10_01/sources/gelato_c625fdc/datasets/"
                      "HumanOrigins_AutosomalSNP/data.csv", encoding="utf-8-sig")
    fst = fst[["PopName", "MedianFSTAdjustedNeighbors"]].rename(columns={"PopName": "population"})
    fst = fst[pd.to_numeric(fst.MedianFSTAdjustedNeighbors, errors="coerce") > 0]
    fst["neigh_flow"] = -np.log(fst.MedianFSTAdjustedNeighbors.astype(float))
    pop = pop.merge(fst[["population", "neigh_flow"]], on="population", how="left")

    # f3 scores (f3.R): larger = stronger evidence of admixture
    f3 = pd.read_csv(OUT / "f3_scores.csv")
    missing_f3 = set(main_links.population) - set(f3.population)
    if missing_f3:
        print(f"WARNING: {len(missing_f3)} target populations missing from f3_scores.csv (re-run f3.R to score all {len(main_links)} targets)", file=sys.stderr)
    k5_path = OUT / "f3_scores_k5.csv"
    if k5_path.exists():
        k5 = pd.read_csv(k5_path)
        missing_k5 = set(main_links.population) - set(k5.population)
        if missing_k5:
            print(f"WARNING: {len(missing_k5)} target populations missing from f3_scores_k5.csv (re-run 'f3.R ... 3000 5' to score all {len(main_links)} targets)", file=sys.stderr)
    f3["f3_z"] = -f3.min_z
    f3["f3_est"] = -f3.min_est
    pop = pop.merge(f3[["population", "n_B", "n_C", "n_triples", "n_z_lt_m3", "min_pop2", "min_pop3", "f3_z", "f3_est"]],
                    on="population", how="left")
    for ic in ("n_B", "n_C", "n_triples", "n_z_lt_m3"):
        pop[ic] = pd.array(pop[ic].tolist(), dtype="Int64")

    pop = pop.merge(main_links.drop(columns=["meets_coverage_main"]), on="population")
    for s in SETS:
        pop[f"in_{s}"] = pop.population.isin(in_set[s]).astype(int)
    pop.round(7).to_csv(OUT / "populations.csv", index=False)

    # --- language table, coordinates, phylogeny ---------------------------------------------
    gl = pd.read_csv(ROOT / "external/glottolog-cldf/cldf/languages.csv", keep_default_na=False, dtype=str).set_index("ID")
    d = g.set_index("glottocode")[["name", "family", "macroarea", "n_present", "n_coded",
                                   "flex_present", "flex_coded", "flex26_present", "flex26_coded",
                                   "irreg13_present", "irreg13_coded"]]
    d["lat"] = gl.loc[d.index, "Latitude"].astype(float)
    d["lon"] = gl.loc[d.index, "Longitude"].astype(float)
    d["family"] = d.family.replace("", "isolate")
    per_lang = pop.groupby("glottocode")
    d["n_populations"] = per_lang.size()
    d["n_individuals"] = per_lang.n_individuals.sum()
    for s in SETS:
        sub = pop[pop[f"in_{s}"] == 1].groupby("glottocode")
        d[f"in_{s}"] = d.index.isin(sub.groups.keys()).astype(int)
        d[f"n_individuals_{s}"] = sub.n_individuals.sum().reindex(d.index).fillna(0).astype(int)
    li = ind.groupby("glottocode")[["ent", "nonmax"]].mean()
    d = d.join(li)

    import low  # noqa: E402
    by_code = {l.glottocode: l for l in low.LanguagesOfTheWorld().languages if l.glottocode}
    vals = pd.read_csv(ROOT / "external/glottolog-cldf/cldf/values.csv", keep_default_na=False, dtype=str)
    lineage_gl = vals[vals.Parameter_ID == "classification"].set_index("Language_ID").Value.to_dict()
    paths, src = {}, {}
    for code in d.index:
        lang = by_code.get(code)
        if lang is not None and lang.family is not None:
            fam = lang.family
            paths[code] = [a.glottocode for a in reversed(fam.ancestors)] + [fam.glottocode]
            src[code] = "low"
        else:
            paths[code] = [x for x in lineage_gl.get(code, "").split("/") if x]
            src[code] = "glottolog_5.3"
    d["tree_source"] = pd.Series(src)
    (OUT / "phylo_tree.nwk").write_text(newick(paths) + "\n")
    d.index.name = "glottocode"
    d.reset_index().round(6).to_csv(OUT / "language_data.csv", index=False)

    # --- summary ----------------------------------------------------------------------------
    d_c12 = d[d.n_coded >= 8]
    pop_c12 = pop[pop.population.isin(cat12_pops)]
    pi = pop_c12.merge(ind.groupby("population")[["ent", "nonmax", "ent_K12", "ent_K30"]].mean(), on="population", suffixes=("_p", ""))
    summ = {
        "sets": {s: {"languages": int(d_c12[f"in_{s}"].sum()), "populations": int(pop_c12[f"in_{s}"].sum()),
                     "individuals": int(pop_c12.loc[pop_c12[f"in_{s}"] == 1, "n_individuals"].sum())} for s in SETS},
        "sets_all_grambank": {s: {"languages": int(d[f"in_{s}"].sum()), "populations": int(pop[f"in_{s}"].sum()),
                                  "individuals": int(pop.loc[pop[f"in_{s}"] == 1, "n_individuals"].sum())} for s in SETS},
        "n_families": int(d_c12.family.nunique()),
        "n_families_all_grambank": int(d.family.nunique()),
        "tree_source": d_c12.tree_source.value_counts().to_dict(),
        "individuals_per_language": d_c12.n_individuals.describe().round(2).to_dict(),
        "populations_without_het": int(pop_c12.het.isna().sum()),
        "populations_without_f3": int(pop_c12.f3_z.isna().sum()),
        "populations_without_neigh_flow": int(pop_c12.neigh_flow.isna().sum()),
        "pooled_per_individual_sd": pooled,
        "n_boot": N_BOOT,
        "flexivity_outcomes": {
            "flex": {"n_features": 30,
                     "own_cov60": int((d.flex_coded >= 18).sum()),
                     "intersect_cat12_cov60": int(((d.n_coded >= 8) & (d.flex_coded >= 18)).sum()),
                     "own_cov50": int((d.flex_coded >= 15).sum()),
                     "own_cov75": int((d.flex_coded >= 23).sum()),
                     "own_all_coded": int((d.flex_coded > 0).sum()),
                     "coded_per_language_all_grambank": d.flex_coded.describe().round(2).to_dict()},
            "flex26": {"n_features": 26,
                       "own_cov60": int((d.flex26_coded >= 16).sum()),
                       "intersect_cat12_cov60": int(((d.n_coded >= 8) & (d.flex26_coded >= 16)).sum()),
                       "coded_per_language_all_grambank": d.flex26_coded.describe().round(2).to_dict()},
            "irreg13": {"n_features": 13,
                        "own_cov60": int((d.irreg13_coded >= 8).sum()),
                        "intersect_cat12_cov60": int(((d.n_coded >= 8) & (d.irreg13_coded >= 8)).sum()),
                        "coded_per_language_all_grambank": d.irreg13_coded.describe().round(2).to_dict()},
        },
        "population_spearman": pi[["ent", "nonmax", "pop_ent", "het", "sqrt_het", "support", "amount", "graff_target", "neigh_flow",
                                   "f3_z", "f3_est", "n_individuals", "n_triples"]].astype(float).rank().corr().round(2).to_dict(),
    }
    (OUT / "prep_summary.json").write_text(json.dumps(summ, indent=2))
    update_existing_language_data()
    print(json.dumps({k: v for k, v in json.loads((OUT / "prep_summary.json").read_text()).items() if k != "population_spearman"}, indent=2))


def newick(paths: dict[str, list[str]]) -> str:
    """Topology-only Newick: internal nodes are Glottolog groups, tips are languages.
    Families hang from a common root; branch lengths are set later (Grafen, in R)."""
    tree: dict = {}
    for tip, path in paths.items():
        node = tree
        for a in path:
            node = node.setdefault(a, {})
        node[tip] = None

    def emit(name, children):
        if children is None:
            return name
        kids = [emit(k, v) for k, v in children.items()]
        return kids[0] if len(kids) == 1 else "(" + ",".join(kids) + ")" + name

    return emit("root", tree) + ";"


if __name__ == "__main__":
    if "--demog-only" in sys.argv or np is None:
        update_existing_language_data()
    else:
        main()
