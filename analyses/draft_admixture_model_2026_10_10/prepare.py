"""Draft admixture -> Grambank inflection model: data preparation.

EXPLORATORY. Written on the user's request of 2026-10-10 to fit a first draft model; it
lifts the ancestry firewall of the pipeline for this folder only. Links are taken as
reliable (all non-proxy candidate links), as instructed.

Writes, next to this script:
  language_data.csv      one row per language: Grambank outcome, admixture measures,
                         coordinates (Glottolog 5.3 point), family
  admixture_by_K.csv     the continuous measures per language and K (K sensitivity)
  individuals.csv        one row per individual (K-averaged entropy and 1 - max; K = 12, 30)
  populations.csv        one row per population (population entropy, Graff support/amount,
                         Graff curated-target flag, neighbour FST)
  phylo_tree.nwk         Glottolog tree of the analysed languages (topology from `low`,
                         Glottolog 5.3 classification where `low` lacks the language)
  prep_summary.json      counts and coverage

Run: .venv/bin/python analyses/draft_admixture_model_2026_10_10/prepare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCH = ROOT / "analyses/gelato_feasibility_2026_10_01/sources/zenodo_15263706/geneticAdmixture-linguisticDiffusion"
ADM = ARCH / "input/MegaAdmixtureCatalogue/ADMIXTURE"
TYP = ROOT / "outputs/pcfp_v2/typology"
KS = range(12, 31)
NONPROXY = {"exact", "dialect_rollup", "group_map_down", "manual"}

sys.path.insert(0, str(ROOT / "external/languages-of-the-world/src"))
import low  # noqa: E402


def entropy(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-12, 1)
    return -(q * np.log(q)).sum(axis=1)


def main() -> None:
    # --- outcome: main analysis set of the typology stage --------------------------------
    g = pd.read_csv(TYP / "grambank_inflection.csv", keep_default_na=False, dtype=str)
    g = g[(g.meets_coverage_main == "True") & (g.link_basis_proxy_only == "False")].copy()
    for c in ("n_present", "n_coded"):
        g[c] = g[c].astype(int)

    # --- population -> language (non-proxy, not excluded) --------------------------------
    links = pd.read_csv(TYP / "grambank_population_links.csv", keep_default_na=False, dtype=str)
    links = links[links.link_basis.isin(NONPROXY) & (links.match_status != "excluded")
                  & links.glottocode.isin(g.glottocode)][["population", "glottocode"]].drop_duplicates()

    # --- individuals and Q matrices ------------------------------------------------------
    info = pd.read_csv(ADM / "GeneticInfoID.csv", keep_default_na=False)
    assert info.Order.astype(int).tolist() == list(range(1, len(info) + 1))
    info = info[["Population"]].rename(columns={"Population": "population"}).reset_index(drop=True)
    ind_rows, pop_rows = [], []
    for k in KS:
        q = np.loadtxt(ADM / f"best_runs/GelatoHO_mergedSetMarchBEDnorelatives_pruned_autosomes_K{k}.Q")
        assert q.shape == (len(info), k) and np.allclose(q.sum(1), 1, atol=1e-3)
        ind = info.assign(K=k, ind=np.arange(len(info)), ent=entropy(q), nonmax=1 - q.max(1))
        ind_rows.append(ind)
        pm = pd.DataFrame(q).groupby(info.population).mean()
        # Graff et al. detection rule per K: top two components > t and second > 5%;
        # support = share of thresholds t in {0.7, 0.8, 0.9} passed, amount = the second
        # component's share weighted by the same flags
        srt = -np.sort(-pm.to_numpy(), axis=1)
        flags = np.stack([(srt[:, 0] + srt[:, 1] > t) & (srt[:, 1] > 0.05) for t in (0.7, 0.8, 0.9)])
        pop_rows.append(pd.DataFrame({"population": pm.index, "K": k, "pop_ent": entropy(pm.to_numpy()),
                                      "support": flags.mean(0), "amount": (flags * srt[:, 1]).mean(0)}))
    ind = pd.concat(ind_rows)
    pop = pd.concat(pop_rows)

    # individual measures: mean over the individuals of all linked populations (an
    # individual-level metric can be averaged without creating spurious mixture);
    # population measure: per population, then mean over the language's populations.
    li = ind.merge(links, on="population")
    by_k = (li.groupby(["glottocode", "K"]).agg(ent=("ent", "mean"), nonmax=("nonmax", "mean"))
            .join(pop.merge(links, on="population").groupby(["glottocode", "K"]).pop_ent.mean()).reset_index())
    by_k.round(6).to_csv(OUT / "admixture_by_K.csv", index=False)
    meas = by_k.groupby("glottocode")[["ent", "nonmax", "pop_ent"]].mean()
    # sampling uncertainty of the K-averaged individual entropy (for a measurement-error
    # sensitivity): SD over individuals / sqrt(n), with a pooled SD when n < 3
    ind_k = li.groupby(["glottocode", "ind"]).ent.mean().reset_index()
    sd_pool = ind_k.groupby("glottocode").ent.std().median()
    agg = ind_k.groupby("glottocode").ent.agg(["std", "count"])
    agg["std"] = np.where(agg["count"] >= 3, agg["std"].fillna(sd_pool), sd_pool)
    meas["ent_se"] = agg["std"] / np.sqrt(agg["count"])
    meas["n_individuals"] = agg["count"]
    meas["n_populations"] = links.groupby("glottocode").population.nunique()

    # Graff et al. 2025 curated admixture targets (Table S2; contact between families)
    s2 = pd.read_csv(ARCH / "tables/tableS2.csv", keep_default_na=False, dtype=str)
    tpops = {p.strip().replace(" ", "_") for v in s2.TargetPop for p in v.split(",")} | \
            {p.strip().replace(" ", "_") for v in s2.AlternativeTargetPop for p in v.split(",") if p.strip()}
    tlangs = set(links[links.population.isin(tpops)].glottocode) | set(s2.FinalTargetGlottocodeGBI) - {""}
    meas["graff_target"] = meas.index.isin(tlangs).astype(int)

    # GeLaTo (main panel only): median FST to populations within 1,000 km, drifted
    # populations excluded. Low FST = genetically close to neighbours (gene flow or shared
    # recent ancestry). Measure: -log FST, mean over the language's populations.
    fst = pd.read_csv(ROOT / "analyses/gelato_feasibility_2026_10_01/sources/gelato_c625fdc/datasets/"
                      "HumanOrigins_AutosomalSNP/data.csv", encoding="utf-8-sig")
    fst = fst[["PopName", "MedianFSTAdjustedNeighbors"]].rename(columns={"PopName": "population"})
    fst = fst[pd.to_numeric(fst.MedianFSTAdjustedNeighbors, errors="coerce") > 0]
    fst["neigh_flow"] = -np.log(fst.MedianFSTAdjustedNeighbors.astype(float))
    meas["neigh_flow"] = fst.merge(links, on="population").groupby("glottocode").neigh_flow.mean()

    # --- long tables for the hierarchical exposure model (individual -> population -> language)
    ind_l = li.groupby(["ind", "population", "glottocode"]).agg(ent=("ent", "mean"), nonmax=("nonmax", "mean"))
    for k in (12, 30):
        ind_l[f"ent_K{k}"] = li[li.K == k].set_index(["ind", "population", "glottocode"]).ent
    ind_l.reset_index().round(6).to_csv(OUT / "individuals.csv", index=False)
    pop_l = (pop.groupby("population")[["pop_ent", "support", "amount"]].mean()
             .join(links.set_index("population"), how="inner"))
    pop_l["n_individuals"] = ind_l.reset_index().groupby("population").size()
    pop_l["graff_target"] = pop_l.index.isin(tpops).astype(int)
    pop_l = pop_l.join(fst.set_index("population").neigh_flow)
    pop_l.index.name = "population"
    pop_l.reset_index().round(6).to_csv(OUT / "populations.csv", index=False)
    meas["support"] = pop_l.groupby("glottocode").support.mean()
    meas["amount"] = pop_l.groupby("glottocode").amount.mean()

    # --- coordinates (Glottolog 5.3 language point) and families -------------------------
    gl = pd.read_csv(ROOT / "external/glottolog-cldf/cldf/languages.csv", keep_default_na=False, dtype=str).set_index("ID")
    d = g.set_index("glottocode")[["name", "family", "macroarea", "n_present", "n_coded"]].join(meas, how="inner")
    d["lat"] = gl.loc[d.index, "Latitude"].astype(float)
    d["lon"] = gl.loc[d.index, "Longitude"].astype(float)
    d["family"] = d.family.replace("", "isolate")

    # --- phylogeny: Glottolog tree via `low` ---------------------------------------------
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
    summ = {"n_languages": len(d), "n_outcome_languages": len(g),
            "n_without_admixture": int(len(g) - len(d)),
            "tree_source": d.tree_source.value_counts().to_dict(),
            "n_families": int(d.family.nunique()), "n_graff_targets": int(d.graff_target.sum()), "n_with_neigh_flow": int(d.neigh_flow.notna().sum()),
            "individuals_per_language": d.n_individuals.describe().round(2).to_dict(),
            "corr_measures_spearman": d[["ent", "nonmax", "pop_ent", "graff_target", "neigh_flow", "support", "amount"]].rank().corr().round(3).to_dict()}
    (OUT / "prep_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


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
