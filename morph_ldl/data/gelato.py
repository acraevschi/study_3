"""GeLaTo population-language crosswalk for morphology units (identifiers only).

Reads the cached sources of analyses/gelato_feasibility_2026_10_01 (never downloads,
never writes there). Three source layers are kept apart (gelato/ancestry_sources.csv):

  population_metadata  GeLaTo c625fdc cldf/populations.csv (main_397 panel) and the
                       Zenodo 15263706 Table S1 / population-glottocode mapping
                       (expanded_558 panel, curated language assignments)
  ancestry_inference   Zenodo 15263706 ADMIXTURE best-run Q matrices K=12..30, rows in
                       GeneticInfoID.csv order
  derived_summary      audit.py outputs (population means at K=23; K12-K30 diagnostics)

Ancestry proportions/heterogeneity are never read here: only population labels / keys
are taken from the derived files (pandas `usecols`), so no ancestry value can enter a
matching decision.

Status rules
  exact Glottocode (main panel Glottocode or Table S1 GeLaTo_Glottocode) -> candidate
  only Table S1 GBI/TLI proxy, or audit descendant/Greek candidate      -> ambiguous
  no population                                                        -> unmatched
  extinct/historical resource                                          -> excluded
  `accepted` (or any other override) only via a complete review entry in
  morph_ldl/data/gelato_review.yaml. Nothing is accepted automatically, and an
  `accepted` review takes effect only with a human `confirmed_by` + `confirmed_date`;
  otherwise the row stays `candidate` with `proposed_status: accepted`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from morph_ldl.config import resolve

from .util import read_text_csv, sha256_file

GELATO_COMMIT = "c625fdcf0225142cc03ae3a1635edf322e9c7778"
ZENODO_RECORD = "https://zenodo.org/records/15263706"
Q_PATTERN = "GelatoHO_mergedSetMarchBEDnorelatives_pruned_autosomes_K{K}.Q"
K_VALUES = list(range(12, 31))
REVIEW_FILE = Path(__file__).with_name("gelato_review.yaml")
REQUIRED_REVIEW_FIELDS = ("variety_review", "community_review", "locality_review",
                          "source_period_review", "evidence", "reviewer_note", "reviewer", "date")
STATUSES = ("accepted", "candidate", "ambiguous", "unmatched", "excluded")

CROSSWALK_COLUMNS = [
    "unit_id", "resource_id", "resource_version", "language", "variety_id", "pos", "representation",
    "original_id", "iso639_3", "glottocode",
    "population", "panels", "main_population_id", "main_glottocode", "expanded_population_key",
    "tableS1_gelato_glottocode", "tableS1_gbi_glottocode", "tableS1_tli_glottocode",
    "geneticinfo_population_label", "geneticinfo_glottocode_base", "geneticinfo_n_rows",
    "n_individuals_main", "n_individuals_expanded",
    "latitude", "longitude", "country", "region", "location", "source_publication",
    "curation_notes_linguistics", "curation_notes_genetics", "mapping_comment_gbi", "mapping_comment_tli",
    "ancestry_record", "ancestry_K_available", "ancestry_q_files", "ancestry_row_key",
    "derived_K23_file", "derived_K23_row_key", "derived_diagnostics_file", "derived_diagnostics_row_key",
    "match_basis", "match_status", "proposed_status", "confirmed_by", "evidence", "unresolved_issues", "decision_reason", "review_ref",
]


def _audit_dir(cfg) -> Path:
    return resolve(cfg, "gelato_audit")


@lru_cache(maxsize=4)
def _load(audit_dir: str) -> Dict[str, Any]:
    p = _paths_dir(Path(audit_dir))
    main = read_text_csv(p["main_populations"])
    s1 = read_text_csv(p["tableS1"])
    zmap = read_text_csv(p["zenodo_mapping"])
    gi = read_text_csv(p["geneticinfo"], usecols=["Population", "Publication", "glottocodeBase"])
    audit = read_text_csv(p["audit_crosswalk"])
    k23 = read_text_csv(p["derived_K23"], usecols=["population"])          # keys only
    diag = read_text_csv(p["derived_diag"], usecols=["population", "K"])   # keys only
    return {"main": main, "s1": s1, "zmap": zmap, "gi": gi, "audit": audit,
            "k23_keys": set(k23["population"]), "diag_keys": set(diag["population"]), "paths": p}


def _paths_dir(a: Path) -> Dict[str, Path]:
    z = a / "sources/zenodo_15263706/geneticAdmixture-linguisticDiffusion"
    g = a / "sources/gelato_c625fdc"
    return {
        "main_populations": g / "cldf/populations.csv",
        "main_genetic_summaries": g / "datasets/HumanOrigins_AutosomalSNP/data.csv",
        "tableS1": z / "tables/tableS1.csv",
        "zenodo_mapping": z / "input/GeLaTo-population-glottocode-mapping.csv",
        "geneticinfo": z / "input/MegaAdmixtureCatalogue/ADMIXTURE/GeneticInfoID.csv",
        "q_dir": z / "input/MegaAdmixtureCatalogue/ADMIXTURE/best_runs",
        "audit_crosswalk": a / "outputs/population_crosswalk.csv",
        "derived_K23": a / "outputs/population_ancestry_K23_components.csv",
        "derived_diag": a / "outputs/population_ancestry_K12_K30_diagnostics.csv",
        "derived_Ksens": a / "outputs/ancestry_K_sensitivity.csv",
        "audit_provenance": a / "outputs/provenance.json",
    }


def load_sources(cfg) -> Dict[str, Any]:
    return _load(str(_audit_dir(cfg)))


# --------------------------------------------------------------------------
# Review file
# --------------------------------------------------------------------------

def load_review(path: Path = REVIEW_FILE) -> List[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return doc.get("reviews", [])


def review_problems(entry: dict) -> List[str]:
    probs = []
    st = entry.get("status")
    if st not in STATUSES:
        probs.append(f"invalid status {st!r}")
    if st == "accepted":
        for f in REQUIRED_REVIEW_FIELDS:
            if not str(entry.get(f, "") or "").strip():
                probs.append(f"missing {f}")
    else:
        for f in ("evidence", "reviewer_note"):
            if not str(entry.get(f, "") or "").strip():
                probs.append(f"missing {f}")
    return probs


def _find_review(reviews: List[dict], glottocode: str, population: str, resource_id: str) -> Optional[dict]:
    for r in reviews:
        if r.get("glottocode") != glottocode or r.get("population") != population:
            continue
        res = r.get("resources", "all")
        if res == "all" or resource_id in (res or []):
            return r
    return None


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------

def population_candidates(glottocode: str, src: Dict[str, Any]) -> pd.DataFrame:
    """All populations linked to a Glottocode, one row per population name."""
    if not glottocode:
        return pd.DataFrame()
    main, s1, zmap, gi, audit = src["main"], src["s1"], src["zmap"], src["gi"], src["audit"]
    recs: Dict[str, dict] = {}

    def rec(name: str) -> dict:
        return recs.setdefault(name, {"population": name, "panels": set(), "basis": []})

    for r in main[main["Glottocode"] == glottocode].itertuples():
        d = rec(r.Name)
        d["panels"].add("main_397")
        d["basis"].append("main_397 Glottocode exact")
    for r in s1[s1["GeLaTo_Glottocode"] == glottocode].itertuples():
        d = rec(r.Population)
        d["panels"].add("expanded_558")
        d["basis"].append("Table S1 GeLaTo_Glottocode exact")
    prox = s1[((s1["GBI_Glottocode"] == glottocode) | (s1["TLI_Glottocode"] == glottocode))
              & (s1["GeLaTo_Glottocode"] != glottocode)]
    for r in prox.itertuples():
        d = rec(r.Population)
        d["panels"].add("expanded_558")
        which = [n for n, v in (("GBI", r.GBI_Glottocode), ("TLI", r.TLI_Glottocode)) if v == glottocode]
        d["basis"].append(f"Table S1 {'/'.join(which)} proxy (base {r.GeLaTo_Glottocode})")
    au = audit[(audit["candidate_mgn_glottocode"] == glottocode) & (audit["match_status"] != "exact")]
    for r in au.itertuples():
        d = rec(r.population)
        d["panels"].add(r.panel)
        d["basis"].append(f"audit {r.match_status}: {r.mapping_basis}")
    # if a proxy-only population is also in the main panel under another code, record its panel
    for name, d in recs.items():
        if "main_397" not in d["panels"] and (main["Name"] == name).any():
            d["panels"].add("main_397")
    rows = []
    gi_counts = gi.groupby("Population").size()
    gi_glotto = gi.groupby("Population")["glottocodeBase"].agg(lambda s: ";".join(sorted(set(s))))
    for name, d in sorted(recs.items()):
        m = main[main["Name"] == name]
        s = s1[s1["Population"] == name]
        z = zmap[zmap["PopName"] == name]
        mr = m.iloc[0] if len(m) else None
        sr = s.iloc[0] if len(s) else None
        zr = z.iloc[0] if len(z) else None
        exact = any("exact" in b for b in d["basis"])
        rows.append({
            "population": name,
            "panels": ";".join(sorted(d["panels"])),
            "main_population_id": mr["ID"] if mr is not None else "",
            "main_glottocode": mr["Glottocode"] if mr is not None else "",
            "expanded_population_key": name if sr is not None else "",
            "tableS1_gelato_glottocode": sr["GeLaTo_Glottocode"] if sr is not None else "",
            "tableS1_gbi_glottocode": sr["GBI_Glottocode"] if sr is not None else "",
            "tableS1_tli_glottocode": sr["TLI_Glottocode"] if sr is not None else "",
            "geneticinfo_population_label": name if name in gi_counts.index else "",
            "geneticinfo_glottocode_base": gi_glotto.get(name, ""),
            "geneticinfo_n_rows": int(gi_counts.get(name, 0)),
            "n_individuals_main": int(mr["samplesize"]) if mr is not None and mr["samplesize"] else "",
            "n_individuals_expanded": int(sr["N_Individuals"]) if sr is not None else "",
            "latitude": (mr["Latitude"] if mr is not None else (zr["lat"] if zr is not None else "")),
            "longitude": (mr["Longitude"] if mr is not None else (zr["lon"] if zr is not None else "")),
            "country": (mr["country"] if mr is not None else (zr["country"] if zr is not None else "")),
            "region": mr["geographicRegion"] if mr is not None else "",
            "location": zr["Location"] if zr is not None else "",
            "source_publication": (mr["Source"] if mr is not None else "") or (sr["Reference"] if sr is not None else ""),
            "curation_notes_linguistics": mr["curation_notes_linguistics"] if mr is not None else "",
            "curation_notes_genetics": mr["curation_notes_genetics"] if mr is not None else "",
            "mapping_comment_gbi": zr["comment.gbi.proxies"] if zr is not None else "",
            "mapping_comment_tli": zr["comment.tli.proxies"] if zr is not None else "",
            "ancestry_record": ZENODO_RECORD if name in gi_counts.index else "",
            "ancestry_K_available": "12-30" if name in gi_counts.index else "",
            "ancestry_q_files": Q_PATTERN.replace("{K}", "{12..30}") if name in gi_counts.index else "",
            "ancestry_row_key": (f"GeneticInfoID.csv rows with Population == '{name}' (Q rows in GeneticInfoID Order)"
                                 if name in gi_counts.index else ""),
            "derived_K23_file": "analyses/gelato_feasibility_2026_10_01/outputs/population_ancestry_K23_components.csv" if name in src["k23_keys"] else "",
            "derived_K23_row_key": f"population={name}" if name in src["k23_keys"] else "",
            "derived_diagnostics_file": "analyses/gelato_feasibility_2026_10_01/outputs/population_ancestry_K12_K30_diagnostics.csv" if name in src["diag_keys"] else "",
            "derived_diagnostics_row_key": f"population={name};K=12..30" if name in src["diag_keys"] else "",
            "match_basis": " | ".join(d["basis"]),
            "_exact": exact,
        })
    return pd.DataFrame(rows)


def _issues(row: dict) -> List[str]:
    out = []
    ns = [n for n in (row.get("n_individuals_main"), row.get("n_individuals_expanded")) if n != ""]
    if ns and max(ns) < 5:
        out.append("small genetic sample (<5 individuals)")
    if row.get("geneticinfo_glottocode_base") in ("NA", "") and row.get("geneticinfo_population_label"):
        out.append("GeneticInfoID glottocodeBase is NA; language assignment comes from Table S1 only")
    if row.get("main_glottocode") and row.get("tableS1_gelato_glottocode") and \
            row["main_glottocode"] != row["tableS1_gelato_glottocode"]:
        out.append("main panel and Table S1 Glottocodes differ")
    if not row.get("geneticinfo_population_label"):
        out.append("no ADMIXTURE rows in Zenodo 15263706 (main panel only)")
    return out


def build_crosswalk(resources: pd.DataFrame, cfg, reviews: Optional[List[dict]] = None) -> pd.DataFrame:
    """resources: registry rows with unit_id, resource_id, resource_version, language,
    variety_id, pos, representation, original_id, iso639_3, glottocode, extinct."""
    src = load_sources(cfg)
    reviews = load_review() if reviews is None else reviews
    rows = []
    for res in resources.to_dict("records"):
        base = {k: res.get(k, "") for k in ("unit_id", "resource_id", "resource_version", "language",
                                            "variety_id", "pos", "representation", "original_id",
                                            "iso639_3", "glottocode")}
        cands = population_candidates(res.get("glottocode", ""), src)
        if res.get("extinct"):
            for c in (cands.to_dict("records") if len(cands) else [{}]):
                rows.append({**base, **{k: v for k, v in c.items() if not k.startswith("_")},
                             "match_status": "excluded", "evidence": "",
                             "unresolved_issues": "", "review_ref": "",
                             "decision_reason": "extinct/historical language: no living sampled community"})
            continue
        if not len(cands):
            rows.append({**base, "match_status": "unmatched", "evidence": "",
                         "unresolved_issues": "no GeLaTo population with this Glottocode or as GBI/TLI proxy",
                         "review_ref": "",
                         "decision_reason": "no population candidate in main_397 or expanded_558"})
            continue
        for c in cands.to_dict("records"):
            exact = c.pop("_exact")
            status = "candidate" if exact else "ambiguous"
            reason = ("exact Glottocode; variety/community/locality/period not yet reviewed" if exact
                      else "linked only through a proxy or descendant mapping; not the same language-level unit")
            issues = _issues(c)
            evidence, ref, proposed, confirmed = "", "", "", ""
            rv = _find_review(reviews, res.get("glottocode", ""), c["population"], res["resource_id"])
            if rv is not None:
                probs = review_problems(rv)
                if probs:
                    issues.append("review entry incomplete (" + "; ".join(probs) + "); not applied")
                else:
                    status = rv["status"]
                    reason = rv["reviewer_note"].strip()
                    proposed = ""
                    confirmed = str(rv.get("confirmed_by", "") or "").strip()
                    if status == "accepted" and not (confirmed and str(rv.get("confirmed_date", "") or "").strip()):
                        # Main-agent rule after review: an agent-written review can only
                        # PROPOSE acceptance; a named human must confirm it.
                        status, proposed = "candidate", "accepted"
                        issues.append("acceptance proposed by an agent review; awaiting human "
                                      "confirmation (confirmed_by + confirmed_date in gelato_review.yaml)")
                    evidence = " ".join(str(rv.get(f, "")).strip() for f in
                                        ("evidence", "variety_review", "community_review",
                                         "locality_review", "source_period_review") if rv.get(f))
                    ref = f"gelato_review.yaml:{rv.get('id', '')}"
                    issues += [str(x) for x in rv.get("unresolved", []) or []]
            rows.append({**base, **c, "match_status": status, "proposed_status": proposed,
                         "confirmed_by": confirmed, "evidence": evidence,
                         "unresolved_issues": "; ".join(dict.fromkeys(issues)), "decision_reason": reason,
                         "review_ref": ref})
    out = pd.DataFrame(rows)
    for c in CROSSWALK_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    return out[CROSSWALK_COLUMNS]


def ancestry_sources(cfg) -> pd.DataFrame:
    import json
    p = _paths_dir(_audit_dir(cfg))
    root = _audit_dir(cfg).parents[1]
    prov = {}
    if p["audit_provenance"].exists():
        prov = {d["path"]: d["sha256"] for d in json.loads(p["audit_provenance"].read_text())["inputs"]}

    def row(source_id, layer, path: Path, origin, key, notes, K="", ref=""):
        rel = str(path.relative_to(root))
        sha = sha256_file(path) if path.exists() else ""
        return {"source_id": source_id, "layer": layer, "path": rel, "sha256": sha,
                "sha256_matches_audit_provenance": (prov.get(rel) == sha) if rel in prov else "",
                "origin": origin, "K": K, "reference_setting": ref, "row_key": key, "notes": notes}

    gh = f"https://github.com/gelato-org/gelato-data/tree/{GELATO_COMMIT}"
    rows = [
        row("gelato_main_populations", "population_metadata", p["main_populations"], gh, "ID / Name",
            "397 Human Origins populations: sample size, coordinates, Glottocode, curation notes, source"),
        row("gelato_main_genetic_summaries", "genetic_summary_gelato", p["main_genetic_summaries"], gh, "PopName",
            "GeLaTo FST/Ne summaries (not ancestry proportions); not read by the data stage"),
        row("zenodo_tableS1", "population_metadata", p["tableS1"], ZENODO_RECORD, "Population",
            "558 analysed populations; curated GeLaTo/GBI/TLI Glottocodes; N_Individuals; reference"),
        row("zenodo_population_mapping", "population_metadata", p["zenodo_mapping"], ZENODO_RECORD, "PopName",
            "653-row auxiliary mapping (location, proxy comments); wider than the analysed set"),
        row("zenodo_geneticinfo", "ancestry_inference_index", p["geneticinfo"], ZENODO_RECORD, "Order / Population",
            "individual rows; row i of every Q matrix is Order i"),
    ]
    for k in K_VALUES:
        rows.append(row(f"zenodo_Q_K{k}", "ancestry_inference_Q", p["q_dir"] / Q_PATTERN.format(K=k), ZENODO_RECORD,
                        "GeneticInfoID Order", "ADMIXTURE best run; component ids local to this K",
                        K=k, ref="GelatoHO merged set (March), no relatives, LD-pruned autosomes; 4,768 individuals"))
    rows += [
        row("audit_K23_population_means", "derived_summary", p["derived_K23"],
            "analyses/gelato_feasibility_2026_10_01/audit.py", "population",
            "mean of individual Q rows per Table S1 population; K=23 only; descriptive", K=23,
            ref="as zenodo_Q_K23"),
        row("audit_K12_K30_diagnostics", "derived_summary", p["derived_diag"],
            "analyses/gelato_feasibility_2026_10_01/audit.py", "population + K",
            "diagnostics for populations whose Table S1 Glottocode is an MGN language; descriptive", K="12-30",
            ref="as zenodo_Q_K12..30"),
        row("audit_K_sensitivity", "derived_summary", p["derived_Ksens"],
            "analyses/gelato_feasibility_2026_10_01/audit.py", "population",
            "range across K=12..30; descriptive", K="12-30", ref="as zenodo_Q_K12..30"),
    ]
    return pd.DataFrame(rows)


def links_from_crosswalk(xw: pd.DataFrame, outcome_units: List[str]) -> pd.DataFrame:
    cols = ["unit_id", "resource_id", "glottocode", "population", "panels", "match_status",
            "main_population_id", "expanded_population_key", "n_individuals_main", "n_individuals_expanded",
            "geneticinfo_population_label", "geneticinfo_n_rows", "ancestry_record", "ancestry_K_available",
            "ancestry_q_files", "ancestry_row_key", "derived_K23_file", "derived_K23_row_key",
            "proposed_status", "confirmed_by", "decision_reason", "unresolved_issues", "review_ref"]
    sub = xw[xw["unit_id"].isin(outcome_units)]
    missing = sorted(set(outcome_units) - set(sub["unit_id"]))
    if missing:
        raise KeyError(f"units not in GeLaTo crosswalk: {missing}")
    return sub[cols].reset_index(drop=True)
