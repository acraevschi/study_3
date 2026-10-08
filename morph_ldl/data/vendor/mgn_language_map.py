#!/usr/bin/env python3
"""
Canonical MGN <-> ISO 639-3 <-> Glottolog mapping.

Single source of truth for language identity across the whole Study 3 pipeline.
Previously this mapping existed in three divergent copies (build_demographic_registry.py,
merge_mgn_features.py, build_phylo_matrix.R) which disagreed on hbs/lav/nob/pus and
forced the registry to invent placeholder rows for MGN-internal codes.

`dump_json()` writes mgn_language_map.json so the R scripts read the same mapping
rather than maintaining a fourth copy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = PROJECT_ROOT / "mgn_language_map.json"

# ---------------------------------------------------------------------------
# Language inventory
# ---------------------------------------------------------------------------

# 64 living synchronic MGN languages (codes as they appear in the MGN result files)
MGN_LIVING_LANGS_64: Set[str] = {
    "ady", "ara", "aze", "bak", "bel", "bul", "cat", "ces", "ckb", "crh",
    "cym", "dan", "deu", "dsb", "ell", "eng", "est", "fao", "fas", "fin",
    "fre", "fur", "gal", "gle", "hbs", "heb", "hin", "hun", "hye", "isl",
    "ita", "kan", "kat", "kbd", "klr", "kmr", "lav", "lit", "lld", "mkd",
    "nav", "nld", "nob", "oci", "pol", "por", "pus", "ron", "rus", "slv",
    "sme", "spa", "sqi", "swe", "tat", "tel", "tur", "ukr", "urd", "vec",
    "yai", "yid", "zen", "zul",
}

# 9 ancient/extinct MGN languages, excluded from synchronic demographic modelling
MGN_EXTINCT_LANGS_9: Set[str] = {
    "ang", "fro", "frm", "osx", "lat", "grc", "san", "syc", "xcl",
}

# Comprehensive extinct & historical blacklist for the global registry
EXTINCT_BLACKLIST: Set[str] = MGN_EXTINCT_LANGS_9 | {
    "got", "chu", "non", "goh", "ave", "orv", "akk", "hbo", "pal", "peo",
    "sux", "hit", "cop", "egy", "liv", "dum", "mga", "gmh", "pro", "osp",
    "odt", "ofs", "gml", "peq", "xve", "ett", "uga", "arc", "qaa", "xcr",
    "xum", "olt", "xvo", "xpg", "xfa", "xib", "xct", "sga", "owl", "wlm",
    "oge",
}

# ---------------------------------------------------------------------------
# MGN code -> canonical ISO 639-3
# ---------------------------------------------------------------------------
# Every entry maps an MGN-internal or ISO 639-2/macrolanguage code onto the
# individual ISO 639-3 code that actually carries demographic data. Codes not
# listed here are already canonical ISO 639-3 and map to themselves.
#
# NOTE: hbs/lav/nob/pus/sqi were previously left unmapped, which meant the
# registry had no real row for them and placeholder rows had to be invented.
MGN_TO_ISO: Dict[str, str] = {
    # Chatino false friends: in ISO 639-3 'yai' is Yaghnobi (Iranian, Tajikistan)
    # and 'zen' is Zenaga (Berber, Mauritania). In MGN (Guzman Naranjo 2024, p.417,
    # 1450) they are Oaxaca Chatino fieldwork corpora.
    #
    # 'yai' is Yaitepec Chatino. Glottolog treats Yaitepec Chatino (yait1239) and
    # San Juan Quiahije Chatino (sanj1283) as DIALECTS of Western Highland Chatino
    # (west2644, ISO ctp) -- so ctp is the language-level unit that carries data.
    # NB: an earlier version of this pipeline mapped yai -> czp/yait1238 and then
    # resolved it against Bromham's 'cly' (east2558, Eastern Highland Chatino),
    # which is a DIFFERENT Chatino language. 'czp' and 'yait1238' are not current
    # Glottolog identifiers at all.
    "yai": "ctp",   # Yaitepec / San Juan Quiahije -> Western Highland Chatino
    "zen": "czn",   # Zenzontepec Chatino (zenz1235) - a language-level ISO
    # ISO 639-2 / legacy aliases
    "gal": "glg",   # Galician
    "fre": "fra",   # French
    # Macrolanguage -> individual variety the MGN paradigm dataset represents
    "est": "ekk",   # Standard Estonian
    "fas": "pes",   # Western (Iranian) Persian
    "ara": "arb",   # Standard Arabic
    "aze": "azj",   # North Azerbaijani
    "yid": "ydd",   # Eastern Yiddish
    # hbs and lav are kept as-is: Glottolog carries Serbian-Croatian-Bosnian
    # (sout1528) and Latvian (latv1249) as language-level units, whereas hrv/srp/
    # bos and lvs are only dialects and have no language-level record to attach to.
    "nob": "nor",   # Norwegian Bokmal -> Norwegian (norw1258; nob is a dialect)
    "pus": "pbt",   # Southern Pashto
    "sqi": "als",   # Albanian -> Tosk, the basis of the standard
}

# Bromham ships some varieties under a different ISO code than Glottolog/UniMorph.
# Currently empty: the one former entry (cly -> czp) conflated two distinct
# Chatino languages and has been removed. Kept as an extension point.
BROMHAM_ISO_FIXUPS: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Pluricentric standards
# ---------------------------------------------------------------------------
# A few MGN paradigm datasets describe one shared standard used across several
# ISO 639-3 varieties. For those, L1 population is the sum over the member
# varieties rather than the anchor variety alone; otherwise the population is
# badly understated relative to the speech community that uses the paradigm.
#
# Everything not listed here uses its single anchor variety.
PLURICENTRIC_MEMBERS: Dict[str, List[str]] = {
    # Serbo-Croatian: one inflectional system, several national standards.
    # The anchor (hbs/sout1528) has no population of its own in any source;
    # hrv/srp/bos each do.
    "hbs": ["hrv", "srp", "bos"],
    # Standard Albanian is Tosk-based but is the written standard for Gheg
    # speakers too, so both feed the population that uses this paradigm.
    "als": ["als", "aln"],
    # Modern Standard Arabic has no native speakers by definition. The relevant
    # community is the aggregate of Arabic vernacular speakers. Koplenig covers
    # 9 of ~30 Ethnologue Arabic varieties, so this is an explicit LOWER BOUND
    # (see docs/METHODS.md section 3.3).
    "arb": ["acm", "aeb", "afb", "ajp", "apc", "ary", "arz", "ayn", "shu"],
}

# ---------------------------------------------------------------------------
# Manual population overrides
# ---------------------------------------------------------------------------
# PROVENANCE WARNING -- READ BEFORE PUBLICATION.
#
# These are the ONLY two numbers in the entire pipeline that do not come from a
# data file in this repository or its download cache. Every other population is
# read from Bromham, the Ethnologue multi-ISO table, or Koplenig, and can be
# traced by re-running the pipeline.
#
# WHERE THEY CAME FROM: they were entered by Claude (Anthropic's assistant)
# during the pipeline audit on 2026-09-03, from its own knowledge of standard
# Ethnologue figures. They were NOT read from, or checked against, any primary
# source. They are round numbers of the right order of magnitude, not verified
# citations.
#
# WHY THEY EXIST: both languages are absent from all three population sources,
# and both are needed for the MGN sample. The alternative was to drop Latvian
# and Yiddish from the study.
#
# WHAT TO DO: before publication, look both up in Ethnologue (or another source
# you can cite) and replace these values with the cited figure and edition.
# Until then, treat any result that depends on them as provisional.
#
# HOW TO AUDIT THEM: they carry the provenance label "Manual_Ethnologue" in
# `global_demographic_registry.csv` and in `mgn_modeling_dataset.csv`, so
# `population_source == "Manual_Ethnologue"` isolates every affected row.
# The two languages contribute 78 (lav) and 36 (ydd) of 111,315 trials.
MANUAL_POPULATION_OVERRIDES: Dict[str, Dict[str, object]] = {
    "lav": {
        "population_l1": 1_500_000.0,
        "language_name": "Latvian",
        "note": "UNVERIFIED, entered by Claude 2026-09-03. Latvian (latv1249) is "
                "absent from Bromham, Koplenig and the Ethnologue multi-ISO table. "
                "Approximate L1 figure, order of magnitude reliable. Needs a citation.",
    },
    "ydd": {
        "population_l1": 600_000.0,
        "language_name": "Eastern Yiddish",
        "note": "UNVERIFIED, entered by Claude 2026-09-03. Bromham carries only "
                "Western Yiddish (yih, 5,000), a different variety. Approximate L1 "
                "figure for Eastern Yiddish. Needs a citation.",
    },
}


def canonical_iso(mgn_code: str) -> str:
    """Map an MGN language code onto its canonical ISO 639-3 code."""
    code = str(mgn_code).strip().lower()
    return MGN_TO_ISO.get(code, code)


def population_members(iso: str) -> List[str]:
    """ISO codes whose L1 populations are summed for this language."""
    return PLURICENTRIC_MEMBERS.get(iso, [iso])


def mgn_canonical_isos() -> Dict[str, str]:
    """The full 64-entry MGN -> canonical ISO mapping."""
    return {m: canonical_iso(m) for m in sorted(MGN_LIVING_LANGS_64)}


def dump_json(path: Path = JSON_PATH) -> Path:
    """Write the mapping to JSON so the R scripts share this single definition."""
    payload = {
        "mgn_living_langs_64": sorted(MGN_LIVING_LANGS_64),
        "mgn_extinct_langs_9": sorted(MGN_EXTINCT_LANGS_9),
        "mgn_to_iso": MGN_TO_ISO,
        "mgn_canonical_isos": mgn_canonical_isos(),
        "pluricentric_members": PLURICENTRIC_MEMBERS,
        "bromham_iso_fixups": BROMHAM_ISO_FIXUPS,
    }
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    return path


if __name__ == "__main__":
    p = dump_json()
    print(f"Wrote {p} ({len(MGN_LIVING_LANGS_64)} living MGN languages)")
