"""`typology` stage: Grambank inflection extent for GeLaTo-linked languages.

Independent of select/ldl/evaluate. Reads only Grambank/Glottolog CLDF, the two GeLaTo
population-metadata tables, the GeLaTo review YAML, and (optionally) the data stage's
unit Glottocodes. See docs/TYPOLOGY.md.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from morph_ldl.config import PIPELINE_ROOT, output_dir
from morph_ldl.data import gelato
from morph_ldl.data.util import sha256_file, write_csv, write_json
from morph_ldl.provenance import StageRecorder
from morph_ldl.typology import grambank as gb

STAGE = "typology"
GRAMBANK_FILES = ("values.csv", "parameters.csv", "codes.csv", "languages.csv")
STATUS_RANK = {"accepted": 0, "candidate": 1, "ambiguous": 2, "unmatched": 3, "excluded": 4}
NONPROXY_MATCH = ("exact", "dialect_rollup", "group_map_down", "manual")

LINK_COLUMNS = [
    "population", "panels", "in_main_397", "in_expanded_558", "main_population_id",
    "n_individuals_main", "n_individuals_expanded",
    "main_glottocode", "tableS1_gelato_glottocode", "tableS1_gbi_glottocode", "tableS1_tli_glottocode",
    "source_glottocode", "source_glottocode_field", "source_glottocode_level",
    "glottocode", "language_name", "link_basis", "is_proxy", "candidate_glottocodes",
    "match_status", "proposed_status", "confirmed_by", "decision_reason", "unresolved_issues", "review_ref",
]

SET_METRICS = ("n_features", "n_coded", "n_present", "coverage", "share")


def typology_dir(cfg: dict) -> Path:
    return output_dir(cfg) / "typology"


def _p(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PIPELINE_ROOT / p)


def _tcfg(cfg: dict) -> dict:
    t = cfg.get("typology")
    if not isinstance(t, dict):
        raise KeyError("cfg['typology'] is missing: add the typology block "
                       "(morph_ldl/typology/default_config.yaml) to the experiment config")
    for k in ("sources", "feature_set", "sensitivity_sets", "coverage_threshold", "report_coverage_thresholds"):
        if k not in t:
            raise KeyError(f"cfg['typology'] lacks {k!r}")
    for s in ("grambank", "glottolog", "gelato_main_populations", "gelato_tableS1"):
        if s not in t["sources"] or "path" not in t["sources"][s]:
            raise KeyError(f"typology.sources.{s}.path is missing")
    return t


def _git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _repo_pin(name: str, scfg: dict, require: bool) -> dict:
    root = _p(scfg["path"])
    rec = {"source": name, "path": str(root), "repo": scfg.get("repo", ""), "tag": scfg.get("tag", ""),
           "declared_commit": scfg.get("commit", ""), "resolved_commit": None, "resolved_tag": None,
           "commit_verified": False}
    if (root / ".git").exists():
        rec["resolved_commit"] = _git(["rev-parse", "HEAD"], root)
        rec["resolved_tag"] = _git(["describe", "--tags", "--exact-match"], root)
        if scfg.get("commit") and rec["resolved_commit"] != scfg["commit"]:
            raise RuntimeError(f"{name}: checkout {root} is at {rec['resolved_commit']}, pinned {scfg['commit']} "
                               "(run scripts/fetch_external.sh)")
        rec["commit_verified"] = bool(scfg.get("commit"))
    elif require:
        raise RuntimeError(f"{name}: {root} is not a git checkout; cannot verify pinned commit "
                           "(run scripts/fetch_external.sh or set typology.require_git_pin: false)")
    return rec


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_inputs(cfg: dict, log: gb.FileLog) -> dict:
    t = _tcfg(cfg)
    src = t["sources"]
    require = bool(t.get("require_git_pin", True))
    pins = {"grambank": _repo_pin("grambank", src["grambank"], require),
            "glottolog": _repo_pin("glottolog", src["glottolog"], require)}
    gdir = _p(src["grambank"]["path"]) / "cldf"
    grambank = {f.split(".")[0]: log.read_csv(gdir / f) for f in GRAMBANK_FILES}
    odir = _p(src["glottolog"]["path"]) / "cldf"
    glang = log.read_csv(odir / "languages.csv")
    gvals = log.read_csv(odir / "values.csv", usecols=["Language_ID", "Parameter_ID", "Value"])
    glotto = gb.Glottolog.from_frames(glang, gvals)
    main = log.read_csv(_p(src["gelato_main_populations"]["path"]))
    s1 = log.read_csv(_p(src["gelato_tableS1"]["path"]))
    for name in ("gelato_main_populations", "gelato_tableS1"):
        pins[name] = {"source": name, "path": str(_p(src[name]["path"])), "origin": src[name].get("origin", ""),
                      "version": src[name].get("version", "")}
    return {"grambank": grambank, "glotto": glotto, "main": main, "s1": s1, "pins": pins}


def load_reviews(cfg: dict, log: gb.FileLog) -> List[dict]:
    path = _p(_tcfg(cfg).get("review_file") or str(gelato.REVIEW_FILE))
    if not path.exists():
        return []
    log.record(path)
    return gelato.load_review(path)


# --------------------------------------------------------------------------
# Population links
# --------------------------------------------------------------------------

def _int(x) -> object:
    x = str(x or "").strip()
    return int(float(x)) if x and x not in gb.MISSING_CODES else ""


def apply_review(link: dict, reviews: List[dict]) -> dict:
    """Same rules as morph_ldl.data.gelato.build_crosswalk (contract §9)."""
    issues = [i for i in link["unresolved_issues"].split("; ") if i]
    rv = gelato._find_review(reviews, link["glottocode"], link["population"], "grambank")
    if rv is None or not link["glottocode"]:
        return link
    probs = gelato.review_problems(rv)
    if probs:
        issues.append("review entry incomplete (" + "; ".join(probs) + "); not applied")
    else:
        status = rv["status"]
        proposed = ""
        confirmed = str(rv.get("confirmed_by", "") or "").strip()
        if status == "accepted" and not (confirmed and str(rv.get("confirmed_date", "") or "").strip()):
            status, proposed = "candidate", "accepted"
            issues.append("acceptance proposed by an agent review; awaiting human confirmation "
                          "(confirmed_by + confirmed_date in gelato_review.yaml)")
        issues += [str(x) for x in rv.get("unresolved", []) or []]
        link = {**link, "match_status": status, "proposed_status": proposed, "confirmed_by": confirmed,
                "decision_reason": str(rv["reviewer_note"]).strip(),
                "review_ref": f"gelato_review.yaml:{rv.get('id', '')}"}
    link["unresolved_issues"] = "; ".join(dict.fromkeys(issues))
    return link


def build_links(main: pd.DataFrame, s1: pd.DataFrame, glotto: gb.Glottolog, grambank_lang_ids: set,
                reviews: List[dict], manual_links: Iterable[dict] = ()) -> pd.DataFrame:
    mrows = {r["Name"]: r for r in main.to_dict("records")}
    srows = {r["Population"]: r for r in s1.to_dict("records")}
    rows = []
    for pop in sorted(set(mrows) | set(srows)):
        m, s = mrows.get(pop), srows.get(pop)
        clean = lambda v: "" if v is None or str(v).strip() in gb.MISSING_CODES else str(v).strip()
        meta = {
            "population": pop,
            "panels": ";".join(p for p, ok in (("expanded_558", s is not None), ("main_397", m is not None)) if ok),
            "in_main_397": m is not None, "in_expanded_558": s is not None,
            "main_population_id": m["ID"] if m else "",
            "n_individuals_main": _int(m.get("samplesize")) if m else "",
            "n_individuals_expanded": _int(s.get("N_Individuals")) if s else "",
            "main_glottocode": clean(m.get("Glottocode")) if m else "",
            "tableS1_gelato_glottocode": clean(s.get("GeLaTo_Glottocode")) if s else "",
            "tableS1_gbi_glottocode": clean(s.get("GBI_Glottocode")) if s else "",
            "tableS1_tli_glottocode": clean(s.get("TLI_Glottocode")) if s else "",
        }
        base_issues = []
        ns = [n for n in (meta["n_individuals_main"], meta["n_individuals_expanded"]) if n != ""]
        if ns and max(ns) < 5:
            base_issues.append("small genetic sample (<5 individuals)")
        own: Dict[str, List[str]] = {}
        for fld in ("main_glottocode", "tableS1_gelato_glottocode"):
            if meta[fld]:
                own.setdefault(meta[fld], []).append(fld)
        if len(own) > 1:
            base_issues.append("main panel and Table S1 Glottocodes differ")
        pop_rows = []
        own_langs = set()
        if not own:
            pop_rows.append({**meta, "source_glottocode": "", "source_glottocode_field": "",
                             "source_glottocode_level": "", "glottocode": "", "link_basis": "unresolved",
                             "is_proxy": False, "candidate_glottocodes": "", "match_status": "unmatched",
                             "decision_reason": "population has no own Glottocode (NA) in main_397 / Table S1",
                             "unresolved_issues": "; ".join(base_issues)})
        for code, flds in sorted(own.items()):
            r = gb.resolve_code(code, glotto, grambank_lang_ids)
            issues = list(base_issues)
            if r["basis"] == "dialect_rollup":
                issues.append("dialect-level population rolled up to its language; variety not reviewed")
            if r["basis"] == "group_map_down":
                issues.append("group-level population mapped to its only Grambank-coded language-level descendant")
            status, reason = {
                "exact": ("candidate", "own Glottocode is language-level; variety/community/locality not yet reviewed"),
                "dialect_rollup": ("candidate", "own Glottocode is a dialect; rolled up with Glottolog Language_ID"),
                "group_map_down": ("candidate", "own Glottocode is a group with exactly one Grambank-coded language"),
                "group_ambiguous": ("ambiguous", "own Glottocode is a group with several Grambank-coded languages; no automatic match"),
                "group_no_grambank_descendant": ("unmatched", "own Glottocode is a group with no Grambank-coded language"),
                "unresolved": ("unmatched", "own Glottocode not resolvable to a language in Glottolog 5.3"),
            }[r["basis"]]
            if r["glottocode"]:
                own_langs.add(r["glottocode"])
            pop_rows.append({**meta, "source_glottocode": code, "source_glottocode_field": ";".join(flds),
                             "source_glottocode_level": r["level"], "glottocode": r["glottocode"],
                             "link_basis": r["basis"], "is_proxy": False,
                             "candidate_glottocodes": ";".join(r["candidates"]) if r["basis"] == "group_ambiguous" else "",
                             "match_status": status, "decision_reason": reason,
                             "unresolved_issues": "; ".join(issues)})
        proxies: Dict[str, dict] = {}
        for fld, lab in (("tableS1_gbi_glottocode", "gbi"), ("tableS1_tli_glottocode", "tli")):
            code = meta[fld]
            if not code:
                continue
            r = gb.resolve_code(code, glotto, grambank_lang_ids)
            if not r["glottocode"] or r["glottocode"] in own_langs:
                continue
            d = proxies.setdefault(r["glottocode"], {"labels": [], "codes": [], "levels": [], "res": []})
            d["labels"].append(lab); d["codes"].append(code); d["levels"].append(r["level"]); d["res"].append(r["basis"])
        for lg, d in sorted(proxies.items()):
            issues = list(base_issues) + [f"proxy code resolved by {'/'.join(sorted(set(d['res'])))}"]
            pop_rows.append({**meta, "source_glottocode": ";".join(dict.fromkeys(d["codes"])),
                             "source_glottocode_field": ";".join(f"tableS1_{x}_glottocode" for x in d["labels"]),
                             "source_glottocode_level": ";".join(dict.fromkeys(d["levels"])),
                             "glottocode": lg, "link_basis": "proxy_" + "_".join(d["labels"]), "is_proxy": True,
                             "candidate_glottocodes": "", "match_status": "ambiguous",
                             "decision_reason": "linked only through a Table S1 GBI/TLI database proxy; "
                                                "not the population's own language",
                             "unresolved_issues": "; ".join(issues)})
        for ml in manual_links:
            if ml.get("population") != pop:
                continue
            lg = glotto.language_of(str(ml.get("glottocode", "")))
            if not lg:
                raise ValueError(f"manual link {ml}: glottocode is not a language/dialect in Glottolog")
            pop_rows.append({**meta, "source_glottocode": ml["glottocode"], "source_glottocode_field": "manual",
                             "source_glottocode_level": glotto.level(ml["glottocode"]), "glottocode": lg,
                             "link_basis": "manual", "is_proxy": False, "candidate_glottocodes": "",
                             "match_status": "candidate",
                             "decision_reason": f"manual link: {ml.get('reason', '')} ({ml.get('by', '')})",
                             "unresolved_issues": "; ".join(base_issues)})
        for r in pop_rows:
            r.setdefault("proposed_status", ""); r.setdefault("confirmed_by", ""); r.setdefault("review_ref", "")
            r["language_name"] = glotto.name(r["glottocode"])
            rows.append(apply_review(r, reviews))
    out = pd.DataFrame(rows, columns=LINK_COLUMNS)
    unknown = set(m.get("population") for m in manual_links) - set(out["population"])
    if unknown:
        raise ValueError(f"manual links for unknown populations: {sorted(unknown)}")
    return out.sort_values(["population", "is_proxy", "glottocode", "link_basis"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# LDL units
# --------------------------------------------------------------------------

def ldl_unit_glottocodes(cfg: dict, units: Optional[Iterable[str]], log: gb.FileLog) -> Dict[str, dict]:
    """unit_id -> {"glottocode", "source"} (raw unit Glottocode, before roll-up)."""
    wanted = set(units) if units else None
    out = {}
    reg_path = output_dir(cfg) / "registry" / "registry.csv"
    reg = None
    for u in cfg.get("units", []) or []:
        uid = u["unit_id"]
        if wanted is not None and uid not in wanted:
            continue
        fpath = output_dir(cfg) / "data" / "forms" / f"{uid}.csv"
        gc, source = "", ""
        if fpath.exists():
            df = log.read_csv(fpath, usecols=["glottocode"], nrows=1)
            if len(df):
                gc, source = df["glottocode"].iloc[0], "data_stage_forms"
        if not gc and reg_path.exists():
            if reg is None:
                reg = log.read_csv(reg_path, usecols=["resource_id", "glottocode"])
            hit = reg[reg["resource_id"] == u.get("resource_id")]
            if len(hit):
                gc, source = hit["glottocode"].iloc[0], "registry"
        if not gc:
            try:
                from morph_ldl.data import identifiers
                rid = u.get("resource_id", "")
                entry = next((e for e in cfg.get("resources", {}).get("ingest", [])
                              if rid.endswith(":" + Path(e["file"]).stem)), None)
                if entry is not None:
                    gc = identifiers.resolve_identifier(entry["iso"])["final_glottocode"]
                    source = "identifier_crosswalk"
            except Exception as exc:  # pragma: no cover - environment dependent
                source = f"unavailable ({type(exc).__name__})"
        out[uid] = {"glottocode": gc or "", "source": source or "unavailable"}
    return out


# --------------------------------------------------------------------------
# Outcome table
# --------------------------------------------------------------------------

def _best_status(statuses: Iterable[str]) -> str:
    st = sorted(set(statuses), key=lambda s: STATUS_RANK.get(s, 9))
    return st[0] if st else ""


def build_outcome(links: pd.DataFrame, counts: Dict[str, pd.DataFrame], domain_counts: Dict[str, pd.DataFrame],
                  glotto: gb.Glottolog, grambank_lang_ids: set, tcfg: dict, ldl_units: Dict[str, str],
                  set_sizes: Dict[str, int]) -> pd.DataFrame:
    """One row per language-level Glottocode. ldl_units: unit_id -> language-level glottocode."""
    live = links[(links["glottocode"] != "") & (links["match_status"] != "excluded")]
    langs = set(live["glottocode"]) | {g for g in ldl_units.values() if g}
    thr = float(tcfg["coverage_threshold"])
    rep = sorted(float(x) for x in tcfg["report_coverage_thresholds"])
    min_max = int(tcfg.get("minimal_inflection_max", 1))
    hcfg = tcfg.get("clitic_heuristic", {}) or {}
    rows = []
    for gc in sorted(langs):
        lk = live[live["glottocode"] == gc]
        nonproxy = lk[~lk["is_proxy"].astype(bool)]
        fam_id, fam, iso_flag = glotto.family(gc)
        units = sorted(u for u, g in ldl_units.items() if g == gc)
        def n_ind(df) -> int:  # per population the larger panel size; populations never averaged
            tot = 0
            for r in df.drop_duplicates("population").itertuples():
                ns = [n for n in (r.n_individuals_main, r.n_individuals_expanded) if n != ""]
                tot += max(ns) if ns else 0
            return tot
        proxy_only_pops = lk[~lk["population"].isin(set(nonproxy["population"]))]
        row = {
            "glottocode": gc, "name": glotto.name(gc), "iso639_3": glotto.iso(gc),
            "glottolog_level": glotto.level(gc), "family_id": fam_id, "family": fam, "is_isolate": iso_flag,
            "macroarea": glotto.macroarea(gc),
            "in_scope_reason": ";".join(x for x, ok in (("gelato_linked", len(lk) > 0), ("ldl_unit", bool(units))) if ok),
            "gelato_linked": len(lk) > 0,
            "link_bases": ";".join(sorted(set(lk["link_basis"]))),
            "link_basis_proxy_only": bool(len(lk) > 0 and len(nonproxy) == 0),
            "best_link_status": _best_status(nonproxy["match_status"] if len(nonproxy) else lk["match_status"]),
            "has_accepted_link": bool((nonproxy["match_status"] == "accepted").any()),
            "n_populations": int(lk["population"].nunique()),
            "n_populations_nonproxy": int(nonproxy["population"].nunique()),
            "populations": ";".join(sorted(set(lk["population"]))),
            "n_individuals_total": n_ind(nonproxy),
            "n_individuals_proxy_links": n_ind(proxy_only_pops),
            "has_ldl_unit": bool(units), "ldl_units": ";".join(units),
            "in_grambank": gc in grambank_lang_ids,
        }
        for sname, cdf in counts.items():
            pre = "" if sname == "main" else f"{sname}_"
            has = gc in cdf.index and row["in_grambank"]
            for m in SET_METRICS:
                row[pre + m] = (cdf.at[gc, m] if has else None)
            row[pre + "n_features"] = set_sizes[sname]
        cov = row["coverage"]
        row["meets_coverage_main"] = bool(cov is not None and not pd.isna(cov) and cov >= thr)
        for t in rep:
            row[f"meets_coverage_{int(round(t * 100))}"] = bool(cov is not None and not pd.isna(cov) and cov >= t)
        npres = row["n_present"]
        row["no_inflection"] = (None if npres is None or pd.isna(npres) else bool(npres == 0))
        row["minimal_inflection"] = (None if npres is None or pd.isna(npres) else bool(npres <= min_max))
        dc = {d: ((int(df.at[gc, "n_features"]), int(df.at[gc, "n_coded"]), int(df.at[gc, "n_present"]))
                  if row["in_grambank"] and gc in df.index else (0, 0, 0)) for d, df in domain_counts.items()}
        ff, pf, reason = gb.clitic_flags(row, hcfg, dc) if row["in_grambank"] else (False, False, "")
        row.update({"clitic_flag_family": ff, "clitic_flag_profile": pf, "clitic_flag": ff or pf,
                    "clitic_flag_reason": reason})
        rows.append(row)
    out = pd.DataFrame(rows)
    int_cols = [c for c in out.columns if c.endswith(("n_features", "n_coded", "n_present"))] + ["n_individuals_total", "n_individuals_proxy_links"]
    for c in int_cols:
        out[c] = pd.array(out[c].tolist(), dtype="Int64")
    for c in [c for c in out.columns if c.endswith(("coverage", "share"))]:
        out[c] = pd.to_numeric(out[c]).round(6)
    for c in ("no_inflection", "minimal_inflection"):
        out[c] = pd.array(out[c].tolist(), dtype="boolean")
    return out


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------

def _set_counts(matrix: pd.DataFrame, fs: dict, used: dict) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    valid = {g: used[g]["codes"] for g in fs["features"]}
    present = fs["present_codes"]
    sets = {"main": fs["features"], **fs["sensitivity"]}
    counts = {n: gb.count_set(matrix, ids, present, valid) for n, ids in sets.items()}
    doms = {d: gb.count_set(matrix, ids, present, valid) for d, ids in fs["domains"].items()}
    return counts, doms


def _feature_set_record(tcfg: dict, fs: dict, used: dict, cfg: dict) -> dict:
    return {
        "declared": {"feature_set": tcfg["feature_set"], "sensitivity_sets": tcfg["sensitivity_sets"],
                     "coverage_threshold": tcfg["coverage_threshold"],
                     "report_coverage_thresholds": tcfg["report_coverage_thresholds"],
                     "minimal_inflection_max": tcfg.get("minimal_inflection_max", 1),
                     "clitic_heuristic": tcfg.get("clitic_heuristic", {})},
        "used": {"id": fs["id"], "n_features": len(fs["features"]), "features": fs["features"],
                 "feature_details": used,
                 "sensitivity_sets": {k: v for k, v in fs["sensitivity"].items()},
                 "excluded": fs["excluded"]},
        "config_hash": cfg.get("_config_hash"),
        "grambank_version": tcfg["sources"]["grambank"].get("tag", ""),
    }


def _ldl_overlap(cfg: dict, tcfg: dict, outcome: pd.DataFrame, glotto: gb.Glottolog, ldl_units: Dict[str, str],
                 log: gb.FileLog) -> Tuple[pd.DataFrame, str]:
    rows = [{"kind": "configured_unit", "id": u, "language": glotto.name(g), "glottocode": g, "suitability": "configured"}
            for u, g in sorted(ldl_units.items())]
    path_cfg = tcfg.get("ldl_eligibility_audit")
    path = _p(path_cfg) if path_cfg else output_dir(cfg) / "eligibility" / "broad_audit.csv"
    note = ""
    if path.exists():
        ba = log.read_csv(path, usecols=["resource_id", "language", "glottocode", "pos", "suitability"])
        levels = set(tcfg.get("ldl_eligibility_levels", ["suitable", "suitable_with_caveats", "limited"]))
        ba = ba[(ba["pos"] == "V") & ba["suitability"].isin(levels)]
        for r in ba.sort_values("resource_id").itertuples():
            rows.append({"kind": "broad_audit_verb_resource", "id": r.resource_id, "language": r.language,
                         "glottocode": glotto.language_of(r.glottocode) or "", "suitability": r.suitability})
    else:
        note = f"broad audit not found at {path}; only configured units reported"
    df = pd.DataFrame(rows, columns=["kind", "id", "language", "glottocode", "suitability"])
    o = outcome.set_index("glottocode")
    look = lambda g, c, default: (o.at[g, c] if g in o.index else default)
    df["gelato_linked"] = [bool(look(g, "gelato_linked", False)) for g in df["glottocode"]]
    df["link_basis_proxy_only"] = [bool(look(g, "link_basis_proxy_only", False)) for g in df["glottocode"]]
    df["best_link_status"] = [look(g, "best_link_status", "") for g in df["glottocode"]]
    df["in_grambank"] = [bool(look(g, "in_grambank", False)) for g in df["glottocode"]]
    df["meets_coverage_main"] = [bool(look(g, "meets_coverage_main", False)) for g in df["glottocode"]]
    df["n_present"] = pd.array([look(g, "n_present", None) for g in df["glottocode"]], dtype="Int64")
    df["n_coded"] = pd.array([look(g, "n_coded", None) for g in df["glottocode"]], dtype="Int64")
    return df, note


def _summary(links: pd.DataFrame, outcome: pd.DataFrame, missing: pd.DataFrame, overlap: pd.DataFrame,
             tcfg: dict, ldl_raw: Dict[str, dict], overlap_note: str) -> dict:
    g = outcome[outcome["gelato_linked"]]
    gn = g[~g["link_basis_proxy_only"]]
    ing = gn[gn["in_grambank"]]

    def names(df):
        return [{"glottocode": r.glottocode, "name": r.name, "family": r.family, "macroarea": r.macroarea,
                 "n_present": int(r.n_present), "n_coded": int(r.n_coded), "meets_coverage_main": bool(r.meets_coverage_main)}
                for r in df.itertuples()]

    thr_cols = sorted(c for c in outcome.columns if c.startswith("meets_coverage_"))
    vb = links.groupby(["link_basis", "match_status"]).size().reset_index(name="n")
    return {
        "n_populations": int(links["population"].nunique()),
        "links_by_basis_status": [{"link_basis": r.link_basis, "match_status": r.match_status, "n": int(r.n)}
                                  for r in vb.itertuples()],
        "n_populations_without_language": int(links.groupby("population")["glottocode"].apply(lambda s: (s == "").all()).sum()),
        "unlinked_populations": sorted(links.groupby("population").filter(lambda d: (d["glottocode"] == "").all())["population"].unique().tolist()),
        "n_languages_gelato_linked": int(len(g)),
        "n_languages_gelato_linked_nonproxy": int(len(gn)),
        "n_languages_proxy_only": int(g["link_basis_proxy_only"].sum()),
        "n_nonproxy_in_grambank": int(len(ing)),
        "n_nonproxy_missing_from_grambank": int((~gn["in_grambank"]).sum()),
        "n_proxy_only_in_grambank": int(g[g["link_basis_proxy_only"]]["in_grambank"].sum()),
        "coverage_threshold_main": tcfg["coverage_threshold"],
        "n_nonproxy_by_coverage": {c: int(ing[c].sum()) for c in thr_cols},
        "n_all_linked_by_coverage": {c: int(g[c].sum()) for c in thr_cols},
        "no_inflection_nonproxy": names(ing[ing["no_inflection"].fillna(False).astype(bool)]),
        "minimal_inflection_nonproxy": names(ing[ing["minimal_inflection"].fillna(False).astype(bool)]),
        "clitic_flagged_nonproxy": [{"glottocode": r.glottocode, "name": r.name, "family": r.family,
                                     "n_present": int(r.n_present), "n_coded": int(r.n_coded),
                                     "reason": r.clitic_flag_reason}
                                    for r in ing[ing["clitic_flag"]].itertuples()],
        "missing_from_grambank_top20": missing.head(20)[["glottocode", "name", "n_individuals_total", "populations"]].to_dict("records"),
        "ldl_units": ldl_raw,
        "ldl_overlap": overlap.to_dict("records"),
        "ldl_overlap_note": overlap_note,
    }


def dialect_substitutes(entry_ids, gb_lang_ids: set, glotto, values: pd.DataFrame, fs: dict, valid: dict,
                        mode: str) -> Dict[str, str]:
    """{language-level Glottocode: Grambank entry} for languages coded in Grambank only below
    language level (mode ``substitute``; ``ignore`` returns {}). Candidates are Grambank
    entries that are not language-level in Glottolog and roll up to a language without its
    own entry (dialect entries, and Grambank "languages" that Glottolog treats as dialects).
    With several candidates, the one with most coded main-set features is used (ties: lowest
    ID). Entries are never merged."""
    if mode == "ignore":
        return {}
    if mode != "substitute":
        raise ValueError(f"typology.dialect_entries must be 'ignore' or 'substitute', got {mode!r}")
    cand: Dict[str, List[str]] = {}
    for i in entry_ids:
        if i in gb_lang_ids:
            continue
        lg = glotto.language_of(i)
        if lg and lg not in gb_lang_ids:
            cand.setdefault(lg, []).append(i)
    if not cand:
        return {}
    ids = sorted(i for v in cand.values() for i in v)
    coded = gb.count_set(gb.category_matrix(values, fs, valid, ids), fs["features"],
                         fs["present_codes"], valid)["n_coded"]
    return {lg: sorted(v, key=lambda i: (-int(coded.get(i, 0)), i))[0] for lg, v in sorted(cand.items())}


def stage_typology(cfg: dict, units=None, folds=None, policies=None) -> None:
    tcfg = _tcfg(cfg)
    fs = gb.parse_feature_set(tcfg)      # refuses ill-formed declarations before any read
    out = typology_dir(cfg)
    log = gb.FileLog()
    with StageRecorder(STAGE, cfg, out) as rec:
        inp = load_inputs(cfg, log)
        G = inp["grambank"]
        used = gb.check_against_grambank(fs, G["parameters"], G["codes"])
        langs = G["languages"]
        gb_lang_ids = {i for i in langs["ID"] if inp["glotto"].level(i) == "language"}
        reviews = load_reviews(cfg, log)
        # 1. declared + used feature set first; nothing else is written if this fails
        write_json(_feature_set_record(tcfg, fs, used, cfg), out / "feature_set.json", cfg)
        # 2. counts (with typology.dialect_entries: substitute, a language without a
        #    language-level entry is represented by one of its Grambank dialect entries)
        valid = {g: used[g]["codes"] for g in fs["features"]}
        subst = dialect_substitutes(langs["ID"], gb_lang_ids, inp["glotto"], G["values"], fs, valid,
                                    tcfg.get("dialect_entries", "ignore"))
        matrix = gb.category_matrix(G["values"], fs, valid, gb_lang_ids | set(subst.values()))
        matrix = matrix.rename(index={e: lg for lg, e in subst.items()}).sort_index()
        gb_lang_ids = gb_lang_ids | set(subst)
        counts, doms = _set_counts(matrix, fs, used)
        # 3. links and outcome
        links = build_links(inp["main"], inp["s1"], inp["glotto"], gb_lang_ids, reviews,
                            tcfg.get("manual_links") or [])
        ldl_raw = ldl_unit_glottocodes(cfg, units, log)
        ldl_units = {u: inp["glotto"].language_of(d["glottocode"]) for u, d in ldl_raw.items()}
        for u, d in ldl_raw.items():
            d["language_glottocode"] = ldl_units[u]
        set_sizes = {"main": len(fs["features"]), **{k: len(v) for k, v in fs["sensitivity"].items()}}
        outcome = build_outcome(links, counts, doms, inp["glotto"], gb_lang_ids, tcfg, ldl_units, set_sizes)
        outcome["grambank_entry"] = [subst.get(g, g) if ok else "" for g, ok in zip(outcome["glottocode"],
                                                                                   outcome["in_grambank"])]
        outcome["grambank_entry_level"] = ["dialect" if g in subst else ("language" if ok else "")
                                           for g, ok in zip(outcome["glottocode"], outcome["in_grambank"])]
        dialect_gb = langs[[inp["glotto"].level(i) != "language" for i in langs["ID"]]]
        dial_map: Dict[str, List[str]] = {}
        for i in dialect_gb["ID"]:
            lg = inp["glotto"].language_of(i)
            if lg:
                dial_map.setdefault(lg, []).append(i)
        miss = outcome[outcome["gelato_linked"] & ~outcome["in_grambank"]].copy()
        miss["grambank_dialect_entries"] = [";".join(sorted(dial_map.get(g, []))) for g in miss["glottocode"]]
        miss = miss[["glottocode", "name", "family", "macroarea", "n_individuals_total", "n_individuals_proxy_links",
                     "n_populations",
                     "populations", "link_bases", "link_basis_proxy_only", "best_link_status",
                     "grambank_dialect_entries", "has_ldl_unit"]]
        miss = miss.sort_values(["link_basis_proxy_only", "n_individuals_total", "n_individuals_proxy_links", "glottocode"],
                                ascending=[True, False, False, True], kind="mergesort")
        overlap, note = _ldl_overlap(cfg, tcfg, outcome, inp["glotto"], ldl_units, log)
        paths = {
            "grambank_population_links.csv": write_csv(links, out / "grambank_population_links.csv", cfg),
            "grambank_inflection.csv": write_csv(outcome, out / "grambank_inflection.csv", cfg),
            "missing_from_grambank.csv": write_csv(miss, out / "missing_from_grambank.csv", cfg),
            "ldl_overlap.csv": write_csv(overlap, out / "ldl_overlap.csv", cfg),
        }
        summ = _summary(links, outcome, miss, overlap, tcfg, ldl_raw, note)
        summ["dialect_entries"] = tcfg.get("dialect_entries", "ignore")
        summ["dialect_substitutes"] = [
            {"glottocode": lg, "name": inp["glotto"].name(lg), "grambank_entry": e,
             "grambank_entry_name": inp["glotto"].name(e),
             "gelato_linked": bool(outcome.set_index("glottocode")["gelato_linked"].get(lg, False))}
            for lg, e in sorted(subst.items())]
        paths["coverage_summary.json"] = write_json(summ, out / "coverage_summary.json", cfg)
        rec.inputs = [Path(p) for p in sorted(log.opened)]
        files_by_source = {}
        for name, d in inp["pins"].items():
            root = Path(d["path"])
            files_by_source[name] = {**d, "files": [x for x in log.listing()
                                                    if x["path"] == str(root.resolve()) or x["path"].startswith(str(root.resolve()) + "/")]}
        rec.extra.update({
            "typology_sources": files_by_source,
            "files_opened": log.listing(),
            "forbidden_path_patterns": list(gb.FORBIDDEN_PATTERNS),
            "feature_set_id": fs["id"],
            "ldl_unit_glottocodes": ldl_raw,
            "outputs": {k: sha256_file(v) for k, v in paths.items()},
        })
    print(f"typology: {len(links)} link rows, {len(outcome)} languages "
          f"({int(outcome['in_grambank'].sum())} in Grambank) -> {out}")


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def audit_typology(cfg: dict) -> Tuple[list, list]:
    checks, problems = [], []
    out = typology_dir(cfg)
    try:
        tcfg = _tcfg(cfg)
        fs = gb.parse_feature_set(tcfg)
    except Exception as exc:
        return checks, [f"typology: config invalid: {exc}"]
    need = ["feature_set.json", "grambank_inflection.csv", "grambank_population_links.csv", "stage_manifest.json"]
    missing = [n for n in need if not (out / n).exists()]
    if missing:
        return checks, [f"typology: missing outputs {missing}"]
    fsj = json.loads((out / "feature_set.json").read_text(encoding="utf-8"))
    man = json.loads((out / "stage_manifest.json").read_text(encoding="utf-8"))
    # 1. declared == used == current config
    used = fsj["used"]
    dec = gb.parse_feature_set({"feature_set": fsj["declared"]["feature_set"],
                                "sensitivity_sets": fsj["declared"]["sensitivity_sets"]})
    if dec["features"] != used["features"] or dec["id"] != used["id"]:
        problems.append("typology: declared feature set differs from the set used")
    if {g: d["present_codes"] for g, d in used["feature_details"].items()} != dec["present_codes"]:
        problems.append("typology: declared present codes differ from those used")
    if dec["sensitivity"] != used["sensitivity_sets"]:
        problems.append("typology: declared sensitivity sets differ from those used")
    if dec["categories"] and {c: list(d.get("sources", {})) for c, d in used["feature_details"].items()} != dec["categories"]:
        problems.append("typology: declared category sources differ from those used")
    if fs["categories"] != dec["categories"]:
        problems.append("typology: categories in the current config differ from feature_set.json (re-run the stage)")
    if fs["features"] != used["features"] or fs["present_codes"] != dec["present_codes"] or fs["sensitivity"] != dec["sensitivity"]:
        problems.append("typology: feature set in the current config differs from feature_set.json (re-run the stage)")
    checks.append("typology: feature set declared == used")
    # 2. manifest: status, pins, no forbidden file
    if man.get("status") != "ok":
        problems.append(f"typology: stage manifest status {man.get('status')}")
    srcs = man.get("typology_sources", {})
    for name in ("grambank", "glottolog"):
        s = srcs.get(name)
        if not s:
            problems.append(f"typology: source {name} not recorded in manifest")
            continue
        pin = tcfg["sources"][name]
        if s.get("tag") != pin.get("tag") or s.get("declared_commit") != pin.get("commit"):
            problems.append(f"typology: {name} pin in manifest differs from config")
        if s.get("resolved_commit") and s["resolved_commit"] != pin.get("commit"):
            problems.append(f"typology: {name} resolved commit {s['resolved_commit']} != pinned {pin.get('commit')}")
        if not s.get("files"):
            problems.append(f"typology: no files recorded for {name}")
        if not s.get("commit_verified"):
            checks.append(f"typology: {name} commit not verifiable (not a git checkout)")
    opened = [x["path"] for x in man.get("files_opened", [])]
    if not opened:
        problems.append("typology: manifest lists no opened files")
    bad = [p for p in opened + list(man.get("inputs", {})) if gb.is_forbidden(p)]
    if bad:
        problems.append(f"typology: ancestry/genetic files were opened: {bad}")
    checks.append(f"typology: {len(opened)} opened files, none forbidden")
    # 3. outcome rows: one language-level Glottocode each
    oc = pd.read_csv(out / "grambank_inflection.csv", dtype=str, keep_default_na=False)
    if (oc["glottocode"] == "").any():
        problems.append("typology: outcome rows without glottocode")
    if oc["glottocode"].duplicated().any():
        problems.append(f"typology: duplicated glottocodes {sorted(oc.loc[oc['glottocode'].duplicated(), 'glottocode'])[:10]}")
    glang_path = _p(tcfg["sources"]["glottolog"]["path"]) / "cldf" / "languages.csv"
    if glang_path.exists():
        lv = pd.read_csv(glang_path, dtype=str, keep_default_na=False, usecols=["ID", "Level"]).set_index("ID")["Level"]
        notlang = sorted(g for g in oc["glottocode"] if lv.get(g) != "language")
        if notlang:
            problems.append(f"typology: outcome glottocodes not language-level in Glottolog: {notlang[:10]}")
    else:
        notlang = sorted(oc.loc[oc["glottolog_level"] != "language", "glottocode"])
        if notlang:
            problems.append(f"typology: outcome glottocodes not language-level: {notlang[:10]}")
        checks.append("typology: Glottolog languages.csv absent; used glottolog_level column")
    for pre in [""] + [f"{s}_" for s in fs["sensitivity"]]:
        sub = oc[oc["in_grambank"] == "True"]
        nf, nc, npr = (pd.to_numeric(sub[pre + c]) for c in ("n_features", "n_coded", "n_present"))
        if not ((npr >= 0) & (npr <= nc) & (nc <= nf)).all():
            problems.append(f"typology: count bounds violated for set '{pre or 'main'}'")
    checks.append(f"typology: {len(oc)} outcome rows, unique language-level glottocodes")
    # 4. links
    lk = pd.read_csv(out / "grambank_population_links.csv", dtype=str, keep_default_na=False)
    live = lk[(lk["glottocode"] != "") & (lk["match_status"] != "excluded")]
    orphan = sorted(set(live["glottocode"]) - set(oc["glottocode"]))
    if orphan:
        problems.append(f"typology: link glottocodes without outcome row: {orphan[:10]}")
    if ((lk["match_status"] == "accepted") & (lk["confirmed_by"].str.strip() == "")).any():
        problems.append("typology: accepted link without human confirmed_by")
    checks.append(f"typology: {len(lk)} link rows consistent")
    return checks, problems
