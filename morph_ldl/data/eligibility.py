"""Eligibility audit (contract §4) and resource profiling for the registry / broad audit.

Unit audit (configured units, from forms tables):
  eligibility/<unit_id>_cells.csv     per-cell coverage, missing and variant rates, role
  eligibility/<unit_id>_duplicates.csv duplicate labels and multi-lemma groups
  eligibility/summary.csv             one row per unit
  eligibility/config_cell_check.csv   configured cell_norm strings vs normaliser output

Broad audit (all MGN V/N resources): a *plausible* task specification is derived per
resource without any outcome information:
  verbs: source NFIN; panel = the pilot's 8 abstract slots, each mapped to the cell whose
         features contain the slot's required features with the fewest extra features
         (cells with NEG/PASS/SBJV/participle/converb/masdar/PRF/PROG features avoided),
         ties broken by coverage; an unfillable slot falls back to the best-covered
         remaining cell (flagged).
  nouns: source NOM;SG (fallbacks INDF;SG, then SG with fewest features); panel = the 8
         best-covered other cells among lemmas with the source present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .cells import feature_set, normalize_label
from .grouping import UnionFind, norm_label

VERB_SLOTS: List[Tuple[str, frozenset]] = [
    ("PRS.1SG", frozenset({"PRS", "1", "SG"})),
    ("PRS.3SG", frozenset({"PRS", "3", "SG"})),
    ("PRS.3PL", frozenset({"PRS", "3", "PL"})),
    ("PST.1SG", frozenset({"PST", "1", "SG"})),
    ("PST.3SG", frozenset({"PST", "3", "SG"})),
    ("PST.3PL", frozenset({"PST", "3", "PL"})),
    ("COND.3SG", frozenset({"COND", "3", "SG"})),
    ("IMP.2SG", frozenset({"IMP", "2", "SG"})),
]
AVOID = frozenset({"NEG", "PASS", "SBJV", "V.PTCP", "V.CVB", "V.MSDR", "PRF", "PROG", "LGSPEC1", "LGSPEC2"})
MOODS = frozenset({"COND", "IMP", "POT", "OPT", "SBJV", "JUS", "ADM", "IRR", "QUOT", "INFR"})
# (required, allowed extra features) in order of preference. ACC;SG is a stand-in only
# because MGN drops a column identical to an earlier one (e.g. fin NOM;SG == ACC;SG).
NOUN_SOURCES = [
    (frozenset({"NOM", "SG"}), frozenset({"INDF", "NDEF"})),
    (frozenset({"INDF", "SG"}), frozenset()),
    (frozenset({"NDEF", "SG"}), frozenset()),
    (frozenset({"SG"}), frozenset()),
    (frozenset({"ACC", "SG"}), frozenset({"INDF", "NDEF"})),
]
LEXEME_CELL = "__LEXEME__"


def _choose_exact(cands: Dict[str, float], req: frozenset, allowed: frozenset) -> Optional[str]:
    best = None
    for cn, cov in cands.items():
        fs = feature_set(cn)
        if req <= fs and (fs - req) <= allowed:
            key = (len(fs - req), -cov, cn)
            if best is None or key < best[0]:
                best = (key, cn)
    return best[1] if best else None


def _choose(cands: Dict[str, float], req: frozenset, used: set, avoid=AVOID) -> Optional[str]:
    best = None
    for cn, cov in cands.items():
        if cn in used:
            continue
        fs = feature_set(cn)
        if not req <= fs:
            continue
        if (fs - req) & avoid:
            continue
        if (fs - req) & (MOODS - req) and avoid is AVOID:
            continue
        key = (len(fs - req), -cov, cn)
        if best is None or key < best[0]:
            best = (key, cn)
    return best[1] if best else None


@dataclass
class TaskSpec:
    source: Optional[str]
    panel: List[Tuple[str, str]] = field(default_factory=list)  # (slot, cell_norm)
    method: str = ""
    fallbacks: List[str] = field(default_factory=list)


def plausible_spec(pos: str, coverage: Dict[str, float], source_cov: Dict[str, float] | None = None) -> TaskSpec:
    """coverage: cell_norm -> share of lemmas with the cell filled."""
    cells = {c: v for c, v in coverage.items() if c}
    if pos == "V":
        src = "NFIN" if "NFIN" in cells else None
        if src is None:
            return TaskSpec(None, method="no NFIN cell")
        used = {src}
        panel, fb = [], []
        for slot, req in VERB_SLOTS:
            c = _choose(cells, req, used)
            if c is None:
                rest = sorted((-v, k) for k, v in cells.items() if k not in used)
                if not rest:
                    continue
                c = rest[0][1]
                fb.append(slot)
            used.add(c)
            panel.append((slot, c))
        return TaskSpec(src, panel, "verb slot template", fb)
    if pos == "N":
        src, fb = None, []
        for i, (req, allowed) in enumerate(NOUN_SOURCES):
            src = _choose_exact(cells, req, allowed)
            if src:
                if i:
                    fb.append(f"NOM;SG absent; used {src}")
                break
        if src is None:
            return TaskSpec(None, method="no base singular cell")
        cov = source_cov or cells
        rest = sorted(((-v, k) for k, v in cov.items() if k and k != src and k != LEXEME_CELL))[:8]
        return TaskSpec(src, [(f"N{i+1}", k) for i, (_, k) in enumerate(rest)], "noun top-coverage panel", fb)
    return TaskSpec(None, method=f"POS {pos} not audited")


# --------------------------------------------------------------------------
# Profile of one resource from its raw long records
# --------------------------------------------------------------------------

def _with_lexeme_source(raw: pd.DataFrame, cell_norm: str) -> pd.DataFrame:
    """Add a synthetic source cell whose form is the lexeme label (audit only)."""
    lem = raw.drop_duplicates("lemma_key")[["lemma_key", "lemma_label", "lemma_occurrence", "source_file", "source_row"]].copy()
    lem["cell_orig"] = LEXEME_CELL
    lem["cell_norm"] = cell_norm
    lem["form_orig"] = lem["lemma_label"]
    lem["variant_raw"] = lem["lemma_label"]
    lem["variant_idx"] = 0
    lem["n_variants"] = 1
    lem["is_missing"] = False
    lem["pos_hint"] = ""
    lem["_synthetic"] = True
    return pd.concat([raw, lem[raw.columns]], ignore_index=True)


def profile_raw(raw: pd.DataFrame, file_stem: str, pos: str, representation: str,
                min_groups: int, lexeme_recovery: bool = False) -> dict:
    """Profile one resource. With `lexeme_recovery` (MGN wide, orth only), a source cell
    absent from the file is reconstructed from the lexeme label *for the audit only*:
    MGN's process_subset drops any column identical to an earlier column, including the
    `lexeme` column itself, so a citation-form cell equal to the label everywhere vanishes.
    """
    raw = raw.copy()
    raw["lemma_key"] = raw["lemma_label"] + "\x1f" + raw["lemma_occurrence"].astype(str)
    labels = raw["cell_orig"].unique()
    norm = {c: normalize_label(c, file_stem)[0] for c in labels}
    raw["cell_norm"] = raw["cell_orig"].map(norm)
    raw["_synthetic"] = False
    recovered = None
    if lexeme_recovery and representation == "orth" and pos in ("V", "N"):
        cov0 = set(raw.loc[~raw["is_missing"], "cell_norm"])
        target = "NFIN" if pos == "V" else "NOM;SG"
        if plausible_spec(pos, {c: 1.0 for c in cov0 if c}).source is None:
            raw = _with_lexeme_source(raw, target)
            recovered = target
    n_lemmas = raw["lemma_key"].nunique()
    n_dup = int((raw.groupby("lemma_label")["lemma_occurrence"].max() > 1).sum())
    orig = raw[~raw["_synthetic"]]
    slots = orig.drop_duplicates(["lemma_key", "cell_orig"])
    filled = slots[~slots["is_missing"]]
    n_cells = len(labels)  # source labels only (synthetic lexeme cell excluded)
    missing_rate = 1 - len(filled) / max(1, n_lemmas * n_cells)
    variant_rate = float((filled["n_variants"] > 1).mean()) if len(filled) else 0.0
    vals = orig.loc[~orig["is_missing"], "variant_raw"]
    multi = float(vals.str.strip().str.contains(r"\s", regex=True).mean()) if (len(vals) and representation != "phon_custom") else 0.0
    n_hash = int(vals.str.contains("#", regex=False).sum())
    n_us = int(vals.str.contains("_", regex=False).sum())
    sample = vals.sample(min(3000, len(vals)), random_state=0).tolist() if len(vals) else []
    # coverage per cell_norm (a norm may come from several labels: union)
    all_filled = raw.drop_duplicates(["lemma_key", "cell_orig"])
    all_filled = all_filled[~all_filled["is_missing"]]
    cn_filled = all_filled[all_filled["cell_norm"] != ""].drop_duplicates(["lemma_key", "cell_norm"])
    coverage = (cn_filled.groupby("cell_norm")["lemma_key"].nunique() / max(1, n_lemmas)).to_dict()
    spec = plausible_spec(pos, coverage)
    if recovered:
        spec.fallbacks.append(f"source {recovered} reconstructed from lexeme label (column absent; "
                              "MGN drops columns identical to `lexeme`; unverified)")
    elig_lemmas = elig_groups = 0
    src_cov = None
    if spec.source and pos == "N":
        has_src = set(cn_filled.loc[cn_filled["cell_norm"] == spec.source, "lemma_key"])
        sub = cn_filled[cn_filled["lemma_key"].isin(has_src)]
        src_cov = (sub.groupby("cell_norm")["lemma_key"].nunique() / max(1, len(has_src))).to_dict()
        fb = spec.fallbacks
        spec = plausible_spec(pos, coverage, src_cov)
        spec.fallbacks = list(dict.fromkeys(fb + spec.fallbacks))
    if spec.source and spec.panel:
        need = [spec.source] + [c for _, c in spec.panel]
        have = cn_filled[cn_filled["cell_norm"].isin(need)].groupby("lemma_key")["cell_norm"].nunique()
        elig = set(have[have == len(set(need))].index)
        elig_lemmas = len(elig)
        src_rows = raw[(raw["cell_norm"] == spec.source) & (~raw["is_missing"]) & raw["lemma_key"].isin(elig)]
        lab_of = raw.drop_duplicates("lemma_key").set_index("lemma_key")["lemma_label"]
        elig_groups = _groups_label_and_form(lab_of, elig, src_rows)
    panel_multi = 0.0
    if spec.panel and representation != "phon_custom":
        pv = raw[(raw["cell_norm"].isin([c for _, c in spec.panel])) & (~raw["is_missing"])]
        panel_multi = float(pv["variant_raw"].str.strip().str.contains(r"\s", regex=True).mean()) if len(pv) else 0.0
    return {
        "n_lemmas": n_lemmas, "n_duplicate_labels": n_dup, "n_records": len(raw),
        "n_cells": n_cells, "n_cells_norm": len({v for v in norm.values() if v}),
        "unparseable_cells": sorted(c for c, v in norm.items() if not v),
        "missing_rate": round(missing_rate, 4), "variant_rate": round(variant_rate, 4),
        "multiword_rate": round(multi, 4), "n_forms_with_hash": n_hash,
        "n_forms_with_underscore": n_us, "sample_forms": sample, "coverage": coverage,
        "spec": spec, "n_eligible_lemmas": elig_lemmas, "n_eligible_groups": elig_groups,
        "panel_multiword_rate": round(panel_multi, 4), "min_groups": min_groups,
    }


def _groups_label_and_form(lab_of: pd.Series, elig: set, src_rows: pd.DataFrame) -> int:
    keys = sorted(elig)
    labels = pd.Series([lab_of[k] for k in keys], index=keys)
    uf = UnionFind(keys)
    first: Dict[str, str] = {}
    for k, lab in labels.items():
        nl = "L:" + norm_label(lab)
        if nl in first:
            uf.union(first[nl], k)
        else:
            first[nl] = k
    for k, form in zip(src_rows["lemma_key"], src_rows["variant_raw"].str.strip()):
        fk = "F:" + form
        if fk in first:
            uf.union(first[fk], k)
        else:
            first[fk] = k
    return len({uf.find(k) for k in keys})


# --------------------------------------------------------------------------
# Unit audit from forms tables
# --------------------------------------------------------------------------

def unit_cell_table(forms: pd.DataFrame, source: str, panel: List[Tuple[str, str]]) -> pd.DataFrame:
    slots = forms.drop_duplicates(["lemma_id", "cell_orig"])
    n_lem = forms["lemma_id"].nunique()
    role = {source: "source"}
    slot_of = {source: "SOURCE"}
    for s, c in panel:
        role[c] = "panel"
        slot_of[c] = s
    rows = []
    for (co, cn), g in slots.groupby(["cell_orig", "cell_norm"], sort=False):
        nm = g[~g["is_missing"]]
        allv = forms[(forms["cell_orig"] == co) & (~forms["is_missing"])]
        rows.append({
            "cell_orig": co, "cell_norm": cn, "role": role.get(cn, "other"), "slot": slot_of.get(cn, ""),
            "n_lemmas": n_lem, "n_present": len(nm), "coverage": round(len(nm) / max(1, n_lem), 4),
            "n_missing_marked": int(g["is_missing"].sum()),
            "missing_rate": round(1 - len(nm) / max(1, n_lem), 4),
            "variant_rate": round(float((nm["n_variants"] > 1).mean()) if len(nm) else 0.0, 4),
            "max_variants": int(nm["n_variants"].max()) if len(nm) else 0,
            "multiword_rate": round(float(allv["form"].str.contains(" ", regex=False).mean()) if len(allv) else 0.0, 4),
            "unparseable": cn == "",
        })
    out = pd.DataFrame(rows)
    order = {"source": 0, "panel": 1, "other": 2}
    return out.sort_values(["role", "coverage"], key=lambda s: s.map(order) if s.name == "role" else -s).reset_index(drop=True)


def eligible_lemmas(forms: pd.DataFrame, source: str, panel_cells: List[str]) -> pd.Index:
    need = set([source] + list(panel_cells))
    nm = forms[(~forms["is_missing"]) & (forms["variant_idx"] == 0) & forms["cell_norm"].isin(need)]
    have = nm.groupby("lemma_id")["cell_norm"].nunique()
    return have[have == len(need)].index


def duplicate_table(forms: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    lem = forms.drop_duplicates("lemma_id")[["lemma_id", "lemma_label", "resource_id"]].merge(groups, on="lemma_id")
    multi = lem[lem["group_size"] > 1].sort_values(["group_id", "lemma_id"])
    return multi[["group_id", "group_size", "lemma_id", "lemma_label", "resource_id", "join_reasons"]]


def config_cell_check(unit_id: str, slots: Dict[str, str], present: set, coverage: Dict[str, float]) -> List[dict]:
    rows = []
    for slot, cn in slots.items():
        found = cn in present
        sugg = ""
        if not found:
            want = feature_set(cn)
            cand = sorted(present, key=lambda c: (-len(feature_set(c) & want), len(feature_set(c) ^ want), c))[:3]
            sugg = " | ".join(cand)
        rows.append({"unit_id": unit_id, "slot": slot, "configured_cell_norm": cn, "found": found,
                     "coverage": round(coverage.get(cn, 0.0), 4), "closest_present": sugg})
    return rows


# --------------------------------------------------------------------------
# Derived-paradigm diagnostics (near-duplicate lemmas the contract §3 rule misses)
# --------------------------------------------------------------------------

import re as _re

# Italian pronominal / clitic infinitives: abbandonarsi, andarsene, farcela ... whose
# finite forms are the base verb's forms with proclitics ("mi abbandonai").
_ITA_CLITIC = _re.compile(r"^(.*?r)(si|sene|ci|cene|ne|sela|selo|sele|seli|cela|celo|gliela|glielo|la|le|li|lo)$")


def derived_paradigms(forms: pd.DataFrame, iso: str, source: str) -> pd.DataFrame:
    """Lemmas that are derived/near-duplicate paradigms of another lemma in the resource.

    kinds: 'pronominal_of' (ita clitic infinitive with base verb present),
           'pronominal_no_base' (clitic infinitive, base verb absent),
           'multiword_source' (source form contains a space: phrasal/idiomatic lemma),
           'multiword_label' (label contains a space but the source form does not).
    """
    lem = forms.drop_duplicates("lemma_id")[["lemma_id", "lemma_label", "group_id"]]
    by_label = dict(zip(lem["lemma_label"], lem["lemma_id"]))
    rows = []
    if iso == "ita":
        for lid, lab, gid in lem.itertuples(index=False):
            m = _ITA_CLITIC.match(lab)
            if not m:
                continue
            stem = m.group(1)
            base = next((b for b in (stem + "e", stem + "re") if b in by_label and b != lab), None)
            rows.append({"lemma_id": lid, "lemma_label": lab, "group_id": gid,
                         "kind": "pronominal_of" if base else "pronominal_no_base",
                         "base_lemma_id": by_label.get(base, "") if base else "", "clitic": m.group(2)})
    src = forms[(forms["cell_norm"] == source) & (~forms["is_missing"]) & (forms["variant_idx"] == 0)]
    for lid, lab, gid in src[src["form"].str.contains(" ", regex=False)][["lemma_id", "lemma_label", "group_id"]].itertuples(index=False):
        rows.append({"lemma_id": lid, "lemma_label": lab, "group_id": gid, "kind": "multiword_source",
                     "base_lemma_id": "", "clitic": ""})
    seen = {r["lemma_id"] for r in rows if r["kind"] == "multiword_source"}
    for lid, lab, gid in lem.itertuples(index=False):
        if lid not in seen and _re.search(r"\s", lab.strip()):
            rows.append({"lemma_id": lid, "lemma_label": lab, "group_id": gid, "kind": "multiword_label",
                         "base_lemma_id": "", "clitic": ""})
    return pd.DataFrame(rows, columns=["lemma_id", "lemma_label", "group_id", "kind", "base_lemma_id", "clitic"])


def groups_if_joined(lemmas: pd.DataFrame, derived: pd.DataFrame, eligible: set) -> int:
    """Eligible group count if 'pronominal_of' lemmas were joined to their base's group."""
    gid = dict(zip(lemmas["lemma_id"], lemmas["group_id"]))
    uf = UnionFind(sorted(set(gid.values())))
    for r in derived[derived["kind"] == "pronominal_of"].itertuples():
        if r.base_lemma_id in gid:
            uf.union(gid[r.lemma_id], gid[r.base_lemma_id])
    return len({uf.find(gid[l]) for l in eligible})
