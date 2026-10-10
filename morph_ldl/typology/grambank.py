"""Pure functions for the Grambank inflection-extent outcome (docs/TYPOLOGY.md).

Nothing here reads ancestry values. All file access of the typology stage goes through
``FileLog.read_csv``, which refuses ancestry/genetic-value paths and records every file
it opens (path + sha256) for the stage manifest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from morph_ldl.data.util import read_text_csv, sha256_file

# --------------------------------------------------------------------------
# Ancestry firewall
# --------------------------------------------------------------------------

FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    # matched against the posix path; file-name patterns are anchored to the last component
    r"\.Q$",                              # ADMIXTURE Q matrices
    r"GeneticInfoID[^/]*$",               # Q-row index of individuals
    r"/best_runs/",                       # Q-matrix directory
    r"ancestry[^/]*$",                    # derived ancestry summaries (K23 means, K sensitivity)
    r"K12_K30[^/]*$",                     # K12-K30 diagnostics
    r"diagnostics[^/]*$",
    r"(^|/)population_crosswalk\.csv$",   # audit crosswalk carries FST / Ne values
    r"/datasets/[^/]+/data\.csv$",        # GeLaTo genetic summaries (FST/Ne)
)
_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]


class AncestryFileError(RuntimeError):
    pass


def is_forbidden(path) -> bool:
    s = Path(path).as_posix()
    return any(r.search(s) for r in _FORBIDDEN_RE)


@dataclass
class FileLog:
    """Single gate for all reads of the typology stage."""
    opened: Dict[str, str] = field(default_factory=dict)  # resolved path -> sha256

    def check(self, path) -> Path:
        p = Path(path).resolve()
        if is_forbidden(p):
            raise AncestryFileError(f"typology stage must not open ancestry/genetic files: {p}")
        return p

    def record(self, path) -> Path:
        p = self.check(path)
        if not p.exists():
            raise FileNotFoundError(p)
        self.opened[str(p)] = sha256_file(p)
        return p

    def read_csv(self, path, **kw) -> pd.DataFrame:
        p = self.record(path)
        return read_text_csv(p, **kw)

    def listing(self) -> List[dict]:
        return [{"path": k, "sha256": v} for k, v in sorted(self.opened.items())]


# --------------------------------------------------------------------------
# Feature set declaration
# --------------------------------------------------------------------------

class FeatureSetError(ValueError):
    pass


_GB_ID = re.compile(r"^GB\d{3}$")
_CAT_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_feature_set(tcfg: dict) -> dict:
    """Validate cfg['typology'] feature declarations (shape only, no data).

    Returns {"id", "features" (ordered), "domain_of", "domains", "excluded",
    "present_codes" (per feature), "sensitivity" (name -> ordered ids), "categories"}.

    With ``feature_set.categories`` (category -> Grambank IDs), the counted features are
    categories: domains list category names, and each category is the logical OR of its
    binary Grambank sources (``category_matrix``; the merge rule of the GBI curation,
    Graff et al. 2025, Sci. Data 12:106). Without it, domains list Grambank IDs directly.
    """
    if not isinstance(tcfg, dict):
        raise FeatureSetError("cfg['typology'] must be a mapping")
    fs = tcfg.get("feature_set")
    if not isinstance(fs, dict):
        raise FeatureSetError("typology.feature_set is missing")
    fid = fs.get("id")
    if not fid or not isinstance(fid, str):
        raise FeatureSetError("typology.feature_set.id is missing")
    domains = fs.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise FeatureSetError("typology.feature_set.domains must be a non-empty mapping")
    cats_cfg = fs.get("categories")
    categories: Dict[str, List[str]] = {}
    if cats_cfg is not None:
        if not isinstance(cats_cfg, dict) or not cats_cfg:
            raise FeatureSetError("typology.feature_set.categories must be a non-empty mapping")
        source_of: Dict[str, str] = {}
        for cname, ids in cats_cfg.items():
            if not isinstance(cname, str) or not _CAT_ID.match(cname):
                raise FeatureSetError(f"invalid category name {cname!r}")
            if not isinstance(ids, list) or not ids:
                raise FeatureSetError(f"category {cname!r} must list Grambank IDs")
            for g in ids:
                if not isinstance(g, str) or not _GB_ID.match(g):
                    raise FeatureSetError(f"invalid Grambank ID {g!r} in category {cname!r}")
                if g in source_of:
                    raise FeatureSetError(f"{g} is a source of two categories ({source_of[g]}, {cname})")
                source_of[g] = cname
            categories[cname] = list(ids)
    features: List[str] = []
    domain_of: Dict[str, str] = {}
    for dname, ids in domains.items():
        if not isinstance(ids, list) or not ids:
            raise FeatureSetError(f"domain {dname!r} must be a non-empty list")
        for g in ids:
            if categories:
                if g not in categories:
                    raise FeatureSetError(f"domain {dname!r} lists {g!r}, which is not a declared category")
            elif not isinstance(g, str) or not _GB_ID.match(g):
                raise FeatureSetError(f"invalid Grambank ID {g!r} in domain {dname!r}")
            if g in domain_of:
                raise FeatureSetError(f"{g} listed twice ({domain_of[g]}, {dname})")
            domain_of[g] = dname
            features.append(g)
    n = fs.get("n_features")
    if not isinstance(n, int) or n != len(features):
        raise FeatureSetError(f"n_features={n!r} but {len(features)} features declared")
    if categories and set(categories) != set(features):
        raise FeatureSetError(f"categories not placed in any domain: {sorted(set(categories) - set(features))}")
    sources = [g for c in features for g in categories.get(c, [])]
    excl = fs.get("excluded", {}) or {}
    if not isinstance(excl, dict):
        raise FeatureSetError("typology.feature_set.excluded must be a mapping")
    excluded = {}
    for reason, ids in excl.items():
        for g in ids or []:
            if not isinstance(g, str) or not _GB_ID.match(g):
                raise FeatureSetError(f"invalid excluded ID {g!r}")
            if g in domain_of or g in sources:
                raise FeatureSetError(f"{g} is both included and excluded")
            excluded[g] = reason
    pc = fs.get("present_codes")
    if not isinstance(pc, dict) or "default" not in pc:
        raise FeatureSetError("typology.feature_set.present_codes.default is missing")
    default = [str(x) for x in pc["default"]]
    per = {str(k): [str(x) for x in v] for k, v in (pc.get("per_feature") or {}).items()}
    if categories and (per or default != ["1"]):
        raise FeatureSetError("a category set counts 'present' as '1' (binary sources only); "
                              "present_codes must be {default: ['1']}")
    for g in per:
        if g not in domain_of:
            raise FeatureSetError(f"present_codes.per_feature has undeclared feature {g}")
    present = {g: per.get(g, default) for g in features}
    for g, codes in present.items():
        if not codes or any(c == "?" for c in codes):
            raise FeatureSetError(f"present codes for {g} must be non-empty and not '?'")
    sens_cfg = tcfg.get("sensitivity_sets")
    if not isinstance(sens_cfg, dict) or not sens_cfg:
        raise FeatureSetError("typology.sensitivity_sets is missing")
    sensitivity = {}
    for name, doms in sens_cfg.items():
        if name == "main" or not re.match(r"^[a-z][a-z0-9_]*$", str(name)):
            raise FeatureSetError(f"invalid sensitivity set name {name!r}")
        if not isinstance(doms, list) or not doms:
            raise FeatureSetError(f"sensitivity set {name!r} must list domains")
        for d in doms:
            if d not in domains:
                raise FeatureSetError(f"sensitivity set {name!r}: unknown domain {d!r}")
        sensitivity[name] = [g for g in features if domain_of[g] in set(doms)]
    return {"id": fid, "features": features, "domain_of": domain_of,
            "domains": {d: list(v) for d, v in domains.items()}, "excluded": excluded,
            "present_codes": present, "explicit_present": sorted(per), "sensitivity": sensitivity,
            "categories": categories}


def check_against_grambank(fs: dict, parameters: pd.DataFrame, codes: pd.DataFrame) -> Dict[str, dict]:
    """Every declared feature exists, codes are known; multistate needs explicit rules.

    Returns {feature: {"name", "codes", "present_codes", "domain"}} (the set used).
    """
    pnames = dict(zip(parameters["ID"], parameters["Name"]))
    code_lists = codes.groupby("Parameter_ID")["Name"].apply(lambda s: sorted(set(s))).to_dict()
    if fs.get("categories"):
        used = {}
        for c in fs["features"]:
            src = {}
            for g in fs["categories"][c]:
                if g not in pnames:
                    raise FeatureSetError(f"{g} (category {c}) not in Grambank parameters.csv")
                if set(code_lists.get(g, [])) != {"0", "1"}:
                    raise FeatureSetError(f"{g} (category {c}) is not binary {code_lists.get(g, [])}")
                src[g] = pnames[g]
            used[c] = {"name": c, "domain": fs["domain_of"][c], "codes": ["0", "1"], "present_codes": ["1"],
                       "rule": "OR: 1 if any source is 1; 0 if every source is 0; otherwise '?'",
                       "sources": src}
        for g in sorted(fs["excluded"]):
            if g not in pnames:
                raise FeatureSetError(f"{g} not in Grambank parameters.csv")
        return used
    used = {}
    for g in fs["features"] + sorted(fs["excluded"]):
        if g not in pnames:
            raise FeatureSetError(f"{g} not in Grambank parameters.csv")
    for g in fs["features"]:
        cl = code_lists.get(g, [])
        if not cl:
            raise FeatureSetError(f"{g} has no codes in codes.csv")
        pres = fs["present_codes"][g]
        if set(cl) != {"0", "1"} and g not in fs["explicit_present"]:
            raise FeatureSetError(f"{g} is not binary {cl}: declare present_codes.per_feature")
        bad = [c for c in pres if c not in cl]
        if bad:
            raise FeatureSetError(f"{g}: present codes {bad} not in codes.csv {cl}")
        used[g] = {"name": pnames[g], "domain": fs["domain_of"][g], "codes": cl, "present_codes": list(pres)}
    return used


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------

def feature_matrix(values: pd.DataFrame, features: Sequence[str], valid_codes: Dict[str, List[str]],
                   language_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Language x feature table of raw values ('' = no row). '?' and '' stay as they are.

    Raises on duplicate (language, feature) rows and on values that are neither a code
    of the feature nor '?'/''.
    """
    v = values[values["Parameter_ID"].isin(list(features))]
    if language_ids is not None:
        v = v[v["Language_ID"].isin(set(language_ids))]
    dup = v.duplicated(["Language_ID", "Parameter_ID"])
    if dup.any():
        raise ValueError(f"duplicate Grambank values: {v[dup][['Language_ID', 'Parameter_ID']].head().values.tolist()}")
    for g, sub in v.groupby("Parameter_ID"):
        bad = sorted(set(sub["Value"]) - set(valid_codes[g]) - {"?", ""})
        if bad:
            raise ValueError(f"{g}: unexpected values {bad}")
    m = v.pivot(index="Language_ID", columns="Parameter_ID", values="Value")
    m = m.reindex(columns=list(features)).fillna("")
    return m.sort_index()


def category_matrix(values: pd.DataFrame, fs: dict, valid_codes: Dict[str, List[str]],
                    language_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Language x counted-feature table. For a category set, each category is the logical
    OR of its binary sources (GBI merge rule): '1' if any source is '1', '0' if every source
    is '0', '' if no source has a row, otherwise '?'. Without categories this is
    ``feature_matrix``."""
    cats = fs.get("categories") or {}
    if not cats:
        return feature_matrix(values, fs["features"], valid_codes, language_ids)
    srcs = [g for c in fs["features"] for g in cats[c]]
    raw = feature_matrix(values, srcs, {g: ["0", "1"] for g in srcs}, language_ids)
    out = pd.DataFrame(index=raw.index)
    for c in fs["features"]:
        sub = raw[cats[c]]
        any1 = sub.eq("1").any(axis=1)
        all0 = sub.eq("0").all(axis=1)
        none = sub.eq("").all(axis=1)
        out[c] = np.select([any1, all0, none], ["1", "0", ""], default="?")
    return out


def count_set(matrix: pd.DataFrame, features: Sequence[str], present_codes: Dict[str, List[str]],
              valid_codes: Dict[str, List[str]]) -> pd.DataFrame:
    """n_features, n_coded, n_present, coverage, share per language for one feature set."""
    feats = list(features)
    sub = matrix.reindex(columns=feats).fillna("")
    coded = pd.DataFrame({g: sub[g].isin(valid_codes[g]) for g in feats}, index=sub.index)
    present = pd.DataFrame({g: sub[g].isin(present_codes[g]) for g in feats}, index=sub.index)
    out = pd.DataFrame(index=sub.index)
    out["n_features"] = len(feats)
    out["n_coded"] = coded.sum(axis=1).astype(int)
    out["n_present"] = present.sum(axis=1).astype(int)
    out["coverage"] = (out["n_coded"] / len(feats)).round(6)
    out["share"] = (out["n_present"] / out["n_coded"].where(out["n_coded"] > 0)).round(6)
    return out


# --------------------------------------------------------------------------
# Glottolog
# --------------------------------------------------------------------------

@dataclass
class Glottolog:
    languages: pd.DataFrame                      # indexed by ID
    classification: Dict[str, List[str]]          # ID -> ancestor glottocodes (top first)

    @classmethod
    def from_frames(cls, languages: pd.DataFrame, values: Optional[pd.DataFrame]) -> "Glottolog":
        langs = languages.set_index("ID", drop=False)
        cls_map: Dict[str, List[str]] = {}
        if values is not None:
            c = values[values["Parameter_ID"] == "classification"]
            cls_map = {k: [x for x in str(v).split("/") if x] for k, v in zip(c["Language_ID"], c["Value"])}
        return cls(langs, cls_map)

    def level(self, gc: str) -> str:
        return self.languages["Level"].get(gc, "") if gc else ""

    def name(self, gc: str) -> str:
        return self.languages["Name"].get(gc, "") if gc else ""

    def language_of(self, gc: str) -> str:
        lv = self.level(gc)
        if lv == "language":
            return gc
        if lv == "dialect":
            lg = self.languages["Language_ID"].get(gc, "")
            return lg if self.level(lg) == "language" else ""
        return ""

    def descendant_languages(self, group: str) -> List[str]:
        out = [k for k, anc in self.classification.items() if group in anc and self.level(k) == "language"]
        return sorted(out)

    def family(self, gc: str) -> Tuple[str, str, bool]:
        """(family_id, family_name, is_isolate) for a language-level code."""
        fam = self.languages["Family_ID"].get(gc, "")
        if not fam:
            return gc, self.name(gc), True
        return fam, self.name(fam), False

    def macroarea(self, gc: str) -> str:
        return self.languages["Macroarea"].get(gc, "") if gc else ""

    def iso(self, gc: str) -> str:
        return self.languages["ISO639P3code"].get(gc, "") if gc else ""


MISSING_CODES = {"", "NA", "na", "N/A"}


def resolve_code(gc: str, glotto: Glottolog, grambank_language_ids: set) -> dict:
    """Resolve a population Glottocode to one language-level Glottocode (docs/TYPOLOGY.md §2).

    Returns {"glottocode", "basis", "level", "candidates"}; glottocode '' if no automatic match.
    """
    gc = (gc or "").strip()
    if gc in MISSING_CODES:
        return {"glottocode": "", "basis": "unresolved", "level": "", "candidates": []}
    lv = glotto.level(gc)
    if lv == "language":
        return {"glottocode": gc, "basis": "exact", "level": lv, "candidates": []}
    if lv == "dialect":
        lg = glotto.language_of(gc)
        if lg:
            return {"glottocode": lg, "basis": "dialect_rollup", "level": lv, "candidates": []}
        return {"glottocode": "", "basis": "unresolved", "level": lv, "candidates": []}
    if lv == "family":
        cands = [d for d in glotto.descendant_languages(gc) if d in grambank_language_ids]
        if len(cands) == 1:
            return {"glottocode": cands[0], "basis": "group_map_down", "level": lv, "candidates": cands}
        if len(cands) > 1:
            return {"glottocode": "", "basis": "group_ambiguous", "level": lv, "candidates": cands}
        return {"glottocode": "", "basis": "group_no_grambank_descendant", "level": lv, "candidates": []}
    return {"glottocode": "", "basis": "unresolved", "level": "not_in_glottolog", "candidates": []}


# --------------------------------------------------------------------------
# Clitic heuristic
# --------------------------------------------------------------------------

def clitic_flags(row: dict, hcfg: dict, domain_counts: Dict[str, Tuple[int, int, int]]) -> Tuple[bool, bool, str]:
    """(family_flag, profile_flag, reason). domain_counts: domain -> (n_features, n_coded, n_present)."""
    n_present = row.get("n_present")
    if n_present is None or pd.isna(n_present):
        return False, False, ""
    reasons = []
    fam_flag = (row.get("family_id") in set(hcfg.get("analytic_families", []))
                and n_present >= int(hcfg.get("analytic_family_min_present", 5)))
    if fam_flag:
        reasons.append(f"family {row.get('family')} (analytic-prone list) with n_present={int(n_present)}")
    prof_flag = n_present >= int(hcfg.get("profile_min_present", 5))
    for d in hcfg.get("profile_zero_domains", []):
        nf, nc, npres = domain_counts[d]
        if not (npres == 0 and nf and nc / nf >= float(hcfg.get("profile_min_domain_coverage", 0.5))):
            prof_flag = False
    if prof_flag:
        reasons.append(f"n_present={int(n_present)} with no {'/'.join(hcfg.get('profile_zero_domains', []))} features present")
    return bool(fam_flag), bool(prof_flag), "; ".join(reasons)
