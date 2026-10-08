"""Identifier crosswalk: resource code -> canonical ISO 639-3 -> Glottocode.

Layering:
  1. original id from the file name (MGN code, or a data-custom language name mapped
     to the MGN code used in MGN's own results);
  2. project correction `src/mgn_language_map.canonical_iso` (MGN_TO_ISO) - read-only;
  3. `low` (languages-of-the-world, pinned) as the ISO -> Glottocode lookup;
  4. data-stage variety subtags (contract §1 `variety_id`).
Every place where layer 2 changes what `low` would give for the original code is
recorded as an override with its reason.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import pandas as pd

from .util import project_module

# data-custom file language names -> MGN code (as used in MGN results / mgn_language_map)
CUSTOM_NAME_TO_MGN = {
    "arabic": "ara", "english": "eng", "french": "fre", "hungarian": "hun", "latin": "lat",
    "latvian": "lav", "navajo": "nav", "portuguese": "por", "russian": "rus",
    "yaitepec-chatino": "yai", "zenzontepec-chatino": "zen",
}

# Reasons for project ISO corrections (summarised from src/mgn_language_map.py comments).
OVERRIDE_REASONS = {
    "yai": "Chatino false friend: ISO yai is Yaghnobi; MGN 'yai' is Yaitepec Chatino, a dialect of Western Highland Chatino (ctp)",
    "zen": "Chatino false friend: ISO zen is Zenaga; MGN 'zen' is Zenzontepec Chatino (czn)",
    "gal": "ISO 639-2/legacy alias: 'gal' is Galoli in ISO 639-3; MGN 'gal' is Galician (glg)",
    "fre": "ISO 639-2/B alias for French (fra)",
    "est": "macrolanguage -> Standard Estonian (ekk)",
    "fas": "macrolanguage -> Western Persian (pes)",
    "ara": "macrolanguage -> Standard Arabic (arb)",
    "aze": "macrolanguage -> North Azerbaijani (azj)",
    "yid": "macrolanguage -> Eastern Yiddish (ydd)",
    "nob": "Norwegian Bokmal mapped to Norwegian (nor/norw1258); nob is a dialect-level code in Glottolog",
    "pus": "macrolanguage -> Southern Pashto (pbt)",
    "sqi": "macrolanguage -> Tosk Albanian (als), the basis of the standard",
}

# Contract §1 variety subtags: resource documents a sub-variety of the canonical ISO.
VARIETY_SUBTAG = {"nob": "nor-bokmal"}


@lru_cache(maxsize=1)
def _low():
    import low  # pinned languages-of-the-world 0.2.0
    return low.LanguagesOfTheWorld()


def low_glottocode(iso: str) -> Optional[str]:
    lang = _low().languages.get(iso)
    return getattr(lang, "glottocode", None) if lang is not None else None


def low_label(iso: str) -> Optional[str]:
    lang = _low().languages.get(iso)
    return getattr(lang, "label", None) if lang is not None else None


def original_id(file_stem: str, collection: str) -> str:
    lang = file_stem.rsplit("-", 1)[0]
    if collection == "mgn_data-custom":
        return CUSTOM_NAME_TO_MGN.get(lang, lang)
    return lang


def resolve_identifier(orig: str) -> Dict[str, object]:
    lm = project_module("mgn_language_map")
    canon = lm.canonical_iso(orig)
    g_orig = low_glottocode(orig)
    g_canon = low_glottocode(canon)
    overridden = canon != orig
    flags = []
    if g_canon is None:
        flags.append("canonical_iso_not_in_low")
    if overridden and g_orig is not None and g_orig != g_canon:
        flags.append("original_code_is_false_friend_in_low")
    if overridden and g_orig is None:
        flags.append("original_code_not_in_low")
    extinct = orig in lm.EXTINCT_BLACKLIST or canon in lm.EXTINCT_BLACKLIST
    if extinct:
        flags.append("extinct_or_historical")
    return {
        "original_id": orig,
        "canonical_iso639_3": canon,
        "variety_id": VARIETY_SUBTAG.get(orig, canon),
        "low_glottocode_for_original": g_orig or "",
        "low_glottocode_for_canonical": g_canon or "",
        "project_override": "yes" if overridden else "no",
        "override_reason": OVERRIDE_REASONS.get(orig, "MGN_TO_ISO entry" if overridden else ""),
        "final_glottocode": g_canon or "",
        "language_label_low": low_label(canon) or "",
        "in_mgn_living_64": orig in lm.MGN_LIVING_LANGS_64,
        "extinct": extinct,
        "unresolved_flags": ";".join(flags),
    }


def identifier_crosswalk(stems_by_collection: Dict[str, list]) -> pd.DataFrame:
    rows = []
    seen = set()
    for coll, stems in stems_by_collection.items():
        for stem in stems:
            orig = original_id(stem, coll)
            key = (coll, orig, stem.rsplit("-", 1)[0])
            if key in seen:
                continue
            seen.add(key)
            r = resolve_identifier(orig)
            rows.append({"collection": coll, "file_language": stem.rsplit("-", 1)[0], **r})
    return pd.DataFrame(rows)
