"""The GeLaTo link decision sheet (morph_ldl/data/gelato_link_review.csv).

Since 2026-10-20 the sheet is the only place where a person decides about a GeLaTo ->
Grambank link (docs/TYPOLOGY.md §2). Links that a rule accepts (exact, dialect roll-up,
group map-down, Graff et al.'s Grambank-side stand-in) need no row. The sheet holds the
cases a rule cannot settle, the recorded decisions, and doubt labels for the sensitivity
analyses.

One row per link, keyed by (population, link_glottocode); link_glottocode is empty for a
population without a language link (for example a group code with several candidates).

  decision   accept | relink | exclude | ambiguous (empty: no decision)
             relink: the link is replaced by a manual link to decision_glottocode
  doubt      labels for the whole population (for example diaspora; wrong_code), used by
             the sensitivity analyses; they never change the main analysis and need no
             confirmation
  confirmed_by + confirmed_date
             a decision takes effect only with both (rule of 2026-10-16); without them it
             is a proposal

Regenerating the sheet (`python -m morph_ldl.cli link_review --config ...`) never
loses a decision: rows are merged by key, filled cells are kept, new cases are added, and
a filled row whose link no longer exists is kept and marked STALE.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

SHEET_FILE = Path(__file__).resolve().parents[1] / "data" / "gelato_link_review.csv"
SHEET_COLUMNS = [
    "population", "link_glottocode", "n_individuals", "why_in_sheet", "gelato_language", "options",
    "place", "notes", "doubt", "decision", "decision_glottocode", "reason", "confirmed_by", "confirmed_date",
]
EVIDENCE_COLUMNS = ("n_individuals", "gelato_language", "options", "place", "notes")
USER_COLUMNS = ("doubt", "decision", "decision_glottocode", "reason", "confirmed_by", "confirmed_date")
DECISIONS = ("accept", "relink", "exclude", "ambiguous")
DECISION_STATUS = {"accept": "accepted", "exclude": "excluded", "ambiguous": "ambiguous", "relink": "excluded"}
STALE = "STALE (no such link)"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
LABEL_RE = re.compile(r"[a-z][a-z0-9_]*")
WHY_GROUP = "group code, no automatic choice"

Key = Tuple[str, str]


# --------------------------------------------------------------------------
# Reading and checking
# --------------------------------------------------------------------------

def read_sheet(path: Path) -> pd.DataFrame:
    """The sheet as strings. Wrong columns or a repeated key stop the run."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    if list(df.columns) != SHEET_COLUMNS:
        raise ValueError(f"{path}: columns must be {SHEET_COLUMNS}, got {list(df.columns)}")
    df = df.apply(lambda c: c.str.strip())
    dup = df[df.duplicated(["population", "link_glottocode"], keep=False)]
    if len(dup):
        raise ValueError(f"{path}: repeated (population, link_glottocode): "
                         f"{sorted(set(zip(dup['population'], dup['link_glottocode'])))}")
    return df


def empty_sheet() -> pd.DataFrame:
    return pd.DataFrame(columns=SHEET_COLUMNS, dtype=str)


def doubt_labels(row) -> List[str]:
    return [x.strip() for x in str(row["doubt"]).split(";") if x.strip()]


def is_confirmed(row) -> bool:
    return bool(row["confirmed_by"] and row["confirmed_date"])


def row_problems(row, language_of=None) -> List[str]:
    """Why a sheet row cannot be applied. language_of: Glottocode -> language-level code ('' if none)."""
    probs = []
    dec, gc = row["decision"], row["decision_glottocode"]
    if dec and dec not in DECISIONS:
        probs.append(f"decision {dec!r} is not one of {', '.join(DECISIONS)}")
    by, date = row["confirmed_by"], row["confirmed_date"]
    if bool(by) != bool(date):
        probs.append("only one of confirmed_by / confirmed_date is filled")
    if date and not DATE_RE.fullmatch(date):
        probs.append(f"confirmed_date {date!r} is not YYYY-MM-DD")
    if (by or date) and not dec:
        probs.append("confirmed without a decision")
    if dec == "relink":
        if not gc:
            probs.append("relink without decision_glottocode")
        elif language_of is not None and not language_of(gc):
            probs.append(f"decision_glottocode {gc!r} is not a language or dialect in Glottolog")
    elif gc:
        probs.append("decision_glottocode is only used with relink")
    if dec in ("accept",) and not row["link_glottocode"]:
        probs.append("accept needs a link with a language (link_glottocode)")
    bad = [x for x in doubt_labels(row) if not LABEL_RE.fullmatch(x)]
    if bad:
        probs.append(f"doubt labels must be lower-case words: {bad}")
    return probs


def filled(row) -> bool:
    return any(str(row[c]).strip() for c in USER_COLUMNS)


# --------------------------------------------------------------------------
# Building and merging
# --------------------------------------------------------------------------

def _n(r) -> str:
    ns = [int(float(x)) for x in (r["n_individuals_main"], r["n_individuals_expanded"]) if str(x).strip()]
    return str(max(ns)) if ns else ""


def mechanical_keys(links: pd.DataFrame, in_grambank: set) -> Dict[Key, str]:
    """Cases no rule settles: a population whose own code is a group with several Grambank
    languages and which has no main-analysis link to a Grambank language."""
    out: Dict[Key, str] = {}
    main_gb = set(links[(links["in_main"] == "True") & links["glottocode"].isin(in_grambank)]["population"])
    for r in links[links["link_basis"] == "group_ambiguous"].itertuples():
        if r.population not in main_gb:
            out[(r.population, r.glottocode)] = WHY_GROUP
    return out


def evidence(links: pd.DataFrame, outcome: pd.DataFrame, meta: Dict[str, dict], keys: Iterable[Key],
             names: Optional[Dict[str, str]] = None) -> Dict[Key, dict]:
    """Evidence cells for each key that is a link (population, glottocode) in `links`."""
    oc = outcome.set_index("glottocode")
    names = names or {}
    lang = lambda g: f"{oc.at[g, 'name'] if g in oc.index else names.get(g, '')} ({g})".strip()

    def grambank_note(g: str) -> str:
        if g not in oc.index or oc.at[g, "in_grambank"] != "True":
            return "not in Grambank"
        return "in main set" if oc.at[g, "meets_coverage_main"] == "True" else "in Grambank, below coverage"

    out = {}
    by_key = {(r["population"], r["glottocode"]): r for r in links.to_dict("records")}
    for k in keys:
        r = by_key.get(k)
        if r is None:
            continue
        pop = k[0]
        own = links[(links["population"] == pop) & (links["is_proxy"] != "True")]
        src = sorted({c for c in own["source_glottocode"] if c})
        m = meta.get(pop, {})
        if r["link_basis"] == "group_ambiguous":
            opts = "; ".join(f"{lang(c)}: {grambank_note(c)}" for c in r["candidate_glottocodes"].split(";") if c)
        else:
            aes = f", AES {r['aes']}" if r.get("aes") else ""
            opts = f"linked: {lang(r['glottocode'])}, {r['link_basis']}, {r['match_status']}{aes}; {grambank_note(r['glottocode'])}"
        notes = [x for x in (f"GeLaTo curation: {m['curation']}" if m.get("curation") else "",
                             f"Graff Grambank-side comment: {m['graff_comment']}" if m.get("graff_comment") else "",
                             r["unresolved_issues"]) if x]
        out[k] = {
            "n_individuals": _n(r),
            "gelato_language": "; ".join(f"{m.get('language_name') or ''} ({c})".strip() for c in src) or "none",
            "options": opts,
            "place": ", ".join(x for x in (m.get("location", ""), m.get("country", ""), m.get("coords", "")) if x),
            "notes": " | ".join(notes),
        }
    return out


def merge(old: pd.DataFrame, links: pd.DataFrame, outcome: pd.DataFrame, meta: Dict[str, dict],
          in_grambank: set, names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """New sheet: mechanical cases plus every old row, evidence refreshed, decisions kept.
    Raises if a filled old row would be lost."""
    mech = mechanical_keys(links, in_grambank)
    link_keys = set(zip(links["population"], links["glottocode"]))
    old = old if len(old) else empty_sheet()
    old_rows = {(r["population"], r["link_glottocode"]): r for r in old.to_dict("records")}
    keys = list(dict.fromkeys(list(old_rows) + sorted(mech)))
    ev = evidence(links, outcome, meta, keys, names)
    rows = []
    for k in keys:
        o = old_rows.get(k)
        if k not in link_keys:
            if o is None or not filled(o):
                continue                                     # vanished and empty: dropped
            r = dict(o)
            if not r["why_in_sheet"].startswith(STALE):
                r["why_in_sheet"] = "; ".join(x for x in (STALE, r["why_in_sheet"]) if x)
            rows.append(r)
            continue
        r = {c: "" for c in SHEET_COLUMNS} if o is None else dict(o)
        r["population"], r["link_glottocode"] = k
        why = [w for w in r["why_in_sheet"].split("; ") + [mech.get(k, "")] if w and w != STALE]
        r["why_in_sheet"] = "; ".join(dict.fromkeys(why))
        r.update(ev.get(k, {}))
        rows.append(r)
    new = pd.DataFrame(rows, columns=SHEET_COLUMNS).fillna("")
    check_no_loss(old, new)
    order = new["why_in_sheet"].str.startswith(STALE)
    return new.assign(_s=order).sort_values(["_s", "population", "link_glottocode"], kind="mergesort") \
              .drop(columns="_s").reset_index(drop=True)


def check_no_loss(old: pd.DataFrame, new: pd.DataFrame) -> None:
    """Every filled old row is in `new` with the same filled cells; otherwise nothing is written."""
    have = {(r["population"], r["link_glottocode"]): r for r in new.to_dict("records")}
    lost, changed = [], []
    for o in old.to_dict("records"):
        if not filled(o):
            continue
        k = (o["population"], o["link_glottocode"])
        n = have.get(k)
        if n is None:
            lost.append(k)
        elif any(str(n[c]) != str(o[c]) for c in USER_COLUMNS):
            changed.append(k)
    if lost:
        raise RuntimeError(f"regeneration would drop filled rows {lost}; sheet not written")
    if changed:
        raise RuntimeError(f"regeneration would change filled cells of {changed}; sheet not written")


def population_meta(main: pd.DataFrame, mapping: Optional[pd.DataFrame]) -> Dict[str, dict]:
    """Place and curation notes per population (GeLaTo main panel, else the Graff et al. mapping)."""
    meta: Dict[str, dict] = {}
    if mapping is not None:
        for r in mapping.to_dict("records"):
            meta[r["PopName"]] = {"location": r.get("Location", ""), "country": r.get("country", ""),
                                  "coords": f"{r['lat']}, {r['lon']}" if r.get("lat") else "",
                                  "graff_comment": r.get("comment.gbi.proxies", "")}
    for r in main.to_dict("records"):
        d = meta.setdefault(r["Name"], {})
        d.update({"language_name": r.get("Language_Name", ""), "curation": r.get("curation_notes_linguistics", ""),
                  "country": d.get("country") or r.get("country", "")})
        if not d.get("coords") and r.get("Latitude"):
            d["coords"] = f"{r['Latitude']}, {r['Longitude']}"
    for d in meta.values():
        for k, v in list(d.items()):
            d[k] = "" if str(v).strip() in ("NA", "..", "nan") else str(v).strip()
    return meta


def write_sheet(df: pd.DataFrame, path: Path) -> None:
    df[SHEET_COLUMNS].to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


# --------------------------------------------------------------------------
# Regeneration (python -m morph_ldl.cli link_review --config ...)
# --------------------------------------------------------------------------

def regenerate(cfg: dict, out: Optional[Path] = None) -> Path:
    """Rebuild the sheet from the typology outputs of cfg's experiment and write it."""
    from morph_ldl.typology.stage import _p, typology_dir
    t = cfg["typology"]
    tdir = typology_dir(cfg)
    links = pd.read_csv(tdir / "grambank_population_links.csv", dtype=str, keep_default_na=False)
    outcome = pd.read_csv(tdir / "grambank_inflection.csv", dtype=str, keep_default_na=False)
    main_pops = pd.read_csv(_p(t["sources"]["gelato_main_populations"]["path"]), dtype=str, keep_default_na=False)
    mp = t["sources"].get("graff_mapping", {}).get("path")
    mapping = pd.read_csv(_p(mp), dtype=str, keep_default_na=False) if mp else None
    glang = pd.read_csv(_p(t["sources"]["glottolog"]["path"]) / "cldf" / "languages.csv", dtype=str,
                        keep_default_na=False, usecols=["ID", "Name"])
    sheet = _p(t.get("decision_sheet") or str(SHEET_FILE))
    old = read_sheet(sheet) if sheet.exists() else empty_sheet()
    new = merge(old, links, outcome, population_meta(main_pops, mapping),
                set(outcome.loc[outcome["in_grambank"] == "True", "glottocode"]),
                dict(zip(glang["ID"], glang["Name"])))
    out = Path(out) if out else sheet
    write_sheet(new, out)
    print(f"link_review: {len(new)} rows -> {out} ({int(new['why_in_sheet'].str.startswith(STALE).sum())} stale)")
    return out
