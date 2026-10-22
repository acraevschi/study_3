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

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCH = ROOT / "analyses/gelato_feasibility_2026_10_01/sources/zenodo_15263706/geneticAdmixture-linguisticDiffusion"
ADM = ARCH / "input/MegaAdmixtureCatalogue/ADMIXTURE"
TYP = ROOT / "outputs/pcfp_v2_links_2026_10_20/typology"
KS = list(range(12, 31))
SETS = ["main", "no_doubtful_language", "no_other_language_stand_ins", "strict"]
N_BOOT = 400
SEED = 20261021

sys.path.insert(0, str(ROOT / "external/languages-of-the-world/src"))
import low  # noqa: E402


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
    return {k: v / len(q_by_k) for k, v in acc.items()}


def main() -> None:
    rng = np.random.default_rng(SEED)

    # --- links and sets ------------------------------------------------------------------
    al = pd.read_csv(TYP / "analysis_links.csv", keep_default_na=False, dtype=str)
    al = al[(al.in_grambank == "True") & (al.meets_coverage_main == "True")]
    assert al.groupby(["analysis", "population"]).glottocode.nunique().max() == 1
    main_links = al[al.analysis == "main"][["population", "glottocode", "link_basis", "accepted_by", "sensitivity_flags"]]
    assert main_links.population.is_unique
    in_set = {s: set(al[al.analysis == s].population) for s in SETS}
    assert all(in_set[s] <= in_set["main"] for s in SETS)

    # --- outcome: main analysis set of the typology stage ---------------------------------
    g = pd.read_csv(TYP / "grambank_inflection.csv", keep_default_na=False, dtype=str)
    g = g[g.in_main_analysis == "True"].copy()
    for c in ("n_present", "n_coded"):
        g[c] = g[c].astype(int)
    assert set(g.glottocode) == set(main_links.glottocode), "outcome languages and linked languages differ"

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
    rows = []
    for pop, grp in ind.groupby("population"):
        ii = grp.ind.to_numpy()
        n = len(ii)
        qk = [Q[k][ii] for k in KS]
        obs = {k: float(v[0]) for k, v in pop_stats(qk).items()}
        boot = pop_stats(qk, rng.integers(0, n, size=(N_BOOT, n)))
        rows.append({"population": pop, "n_individuals": n, **obs,
                     **{f"{k}_se": float(np.nanstd(v, ddof=1)) if n >= 2 else np.nan for k, v in boot.items()}})
    pop = pd.DataFrame(rows)
    # an SE from fewer than 3 individuals is unreliable: use the pooled per-individual SD / sqrt(n),
    # as the draft did for the language-level measurement error
    pooled = {}
    for m in ("pop_ent", "het", "support", "amount"):
        big = pop[pop.n_individuals >= 3]
        pooled[m] = float((big[f"{m}_se"] * np.sqrt(big.n_individuals)).median())
        small = pop.n_individuals < 3
        pop.loc[small, f"{m}_se"] = pooled[m] / np.sqrt(pop.loc[small, "n_individuals"])
    pop.loc[pop.n_individuals < 2, ["het", "het_se"]] = np.nan

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
    f3["f3_z"] = -f3.min_z
    f3["f3_est"] = -f3.min_est
    pop = pop.merge(f3[["population", "n_B", "n_C", "n_triples", "n_z_lt_m3", "min_pop2", "min_pop3", "f3_z", "f3_est"]],
                    on="population", how="left")

    pop = pop.merge(main_links, on="population")
    for s in SETS:
        pop[f"in_{s}"] = pop.population.isin(in_set[s]).astype(int)
    pop.round(7).to_csv(OUT / "populations.csv", index=False)

    # --- language table, coordinates, phylogeny ---------------------------------------------
    gl = pd.read_csv(ROOT / "external/glottolog-cldf/cldf/languages.csv", keep_default_na=False, dtype=str).set_index("ID")
    d = g.set_index("glottocode")[["name", "family", "macroarea", "n_present", "n_coded"]]
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
    meas = ["ent", "nonmax", "pop_ent", "het", "support", "amount", "graff_target", "neigh_flow", "f3_z", "f3_est"]
    pi = pop.merge(ind.groupby("population")[["ent", "nonmax", "ent_K12", "ent_K30"]].mean(), on="population", suffixes=("_p", ""))
    summ = {
        "sets": {s: {"languages": int(d[f"in_{s}"].sum()), "populations": int(pop[f"in_{s}"].sum()),
                     "individuals": int(pop.loc[pop[f"in_{s}"] == 1, "n_individuals"].sum())} for s in SETS},
        "n_families": int(d.family.nunique()),
        "tree_source": d.tree_source.value_counts().to_dict(),
        "individuals_per_language": d.n_individuals.describe().round(2).to_dict(),
        "populations_without_het": int(pop.het.isna().sum()),
        "populations_without_f3": int(pop.f3_z.isna().sum()),
        "populations_without_neigh_flow": int(pop.neigh_flow.isna().sum()),
        "pooled_per_individual_sd": pooled,
        "n_boot": N_BOOT,
        "population_spearman": pi[["ent", "nonmax", "pop_ent", "het", "support", "amount", "graff_target", "neigh_flow", "f3_z", "f3_est",
                                    "n_individuals", "n_triples"]].rank().corr().round(2).to_dict(),
    }
    (OUT / "prep_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "population_spearman"}, indent=2))


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
    main()
