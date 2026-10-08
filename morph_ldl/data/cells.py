"""Cell-label normalisation to contract `cell_norm` (docs/CONTRACT.md §1).

`cell_norm` = UniMorph features, upper-case, de-duplicated, sorted alphabetically,
';'-joined, bare POS removed.

Order of attempts:
  1. resource-specific parsers for MGN data-custom label conventions that the
     project normaliser does not cover (English, French, Hungarian, Latvian,
     Russian, Latin nouns, Portuguese, Chatino). Defined here, documented in DATA.md.
  2. the project normaliser `src/cell_normalization.normalize_cell` with
     `data_sources/cells_to_unimorph.json` (read-only import): curated dotted map,
     Navajo, native UniMorph `;` labels, Polish colon labels, uncurated dotted
     labels, bare single features.
Anything else -> '' (unparseable; the row is kept and the label is reported).
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Set, Tuple

from .util import project_cell_map, project_module

BARE_POS = {"V", "N", "ADJ", "ADV"}
# Spelling variants of UniMorph tags produced by lower-case source labels.
TAG_ALIASES = {"INST": "INS", "SING": "SG", "PLUR": "PL"}


def finalize(feats: Set[str]) -> str:
    out = set()
    for f in feats:
        f = f.strip().upper()
        if not f:
            continue
        f = TAG_ALIASES.get(f, f)
        if f in BARE_POS:
            continue
        out.add(f)
    return ";".join(sorted(out))


# --------------------------------------------------------------------------
# data-custom parsers
# --------------------------------------------------------------------------

_ENGLISH = {
    "inf": {"NFIN"},
    "pres1s": {"PRS", "1", "SG"},
    "pres3s": {"PRS", "3", "SG"},
    # present form of all other person/number combinations
    "presothers": {"PRS", "LGSPEC1"},
    # past form used in 1sg/3sg (differs from pastnot13 only for 'be')
    "past13": {"PST", "LGSPEC1"},
    "pastnot13": {"PST", "LGSPEC2"},
    "ppart": {"V.PTCP", "PST"},
    "prespart": {"V.PTCP", "PRS"},
}

_PN = re.compile(r"^([123])(sg|pl)$")
_FRENCH_TOK = {
    "prs": "PRS", "pst": "PST", "fut": "FUT", "ipfv": "IPFV", "cond": "COND", "imp": "IMP",
    "sbjv": "SBJV", "inf": "NFIN", "ptcp": "V.PTCP", "m": "MASC", "f": "FEM", "sg": "SG", "pl": "PL",
}


def _parse_french(label: str) -> Optional[Set[str]]:
    feats: Set[str] = set()
    for tok in label.lower().split("."):
        m = _PN.match(tok)
        if m:
            feats |= {m.group(1), m.group(2).upper()}
        elif tok in _FRENCH_TOK:
            feats.add(_FRENCH_TOK[tok])
        else:
            return None
    return feats or None


_HUN_TOK = {"inst": "INS", "prp": "PRP", "frml": "FRML", "term": "TERM", "trans": "TRANS"}


def _parse_hungarian(label: str) -> Optional[Set[str]]:
    feats: Set[str] = set()
    for tok in label.split(";"):
        tok = tok.strip()
        if not tok:
            return None
        feats.add(_HUN_TOK.get(tok, tok.upper()))
    return feats


_CASE_WORDS = {
    "nom": "NOM", "nominative": "NOM", "gen": "GEN", "genitive": "GEN", "dat": "DAT",
    "dative": "DAT", "acc": "ACC", "accusative": "ACC", "inst": "INS", "instrumental": "INS",
    "loc": "LOC", "voc": "VOC", "vocative": "VOC",
    # UniMorph Russian tags the prepositional case ESS
    "prepositional": "ESS",
    "sg": "SG", "singular": "SG", "pl": "PL", "plural": "PL",
}


def _parse_spaced_case(label: str) -> Optional[Set[str]]:
    toks = label.lower().split()
    if not toks or not all(t in _CASE_WORDS for t in toks):
        return None
    return {_CASE_WORDS[t] for t in toks}


_LATIN_N = re.compile(r"^NOUN:(Nom|Gen|Dat|Acc|Abl|Voc)\+(Sing|Plur)$")


def _parse_latin_n(label: str) -> Optional[Set[str]]:
    m = _LATIN_N.match(label)
    if not m:
        return None
    return {m.group(1).upper(), {"Sing": "SG", "Plur": "PL"}[m.group(2)]}


_PT_PREFIX = {
    "Condicional": {"COND"},
    "FutConj": {"FUT", "SBJV"},
    "FutImpIndic": {"FUT", "IND"},
    "Imperativo": {"IMP"},
    "InfinitPessoal": {"NFIN", "LGSPEC1"},   # personal (inflected) infinitive
    "PresConj": {"PRS", "SBJV"},
    "PresIndic": {"PRS", "IND"},
    "PretImpConj": {"PST", "IPFV", "SBJV"},
    "PretImpIndic": {"PST", "IPFV", "IND"},
    "PretMqpfIndic": {"PST", "PRF", "IND"},  # pluperfect
    "PretPerfIndic": {"PST", "PFV", "IND"},
}
_PT_PN = {"1": {"1", "SG"}, "2": {"2", "SG"}, "3": {"3", "SG"},
          "4": {"1", "PL"}, "5": {"2", "PL"}, "6": {"3", "PL"}}
_PT_SPECIAL = {"Infinitivo": {"NFIN"}, "Gerúndio": {"V.CVB"},
               "PartPasssm": {"V.PTCP", "PST", "MASC", "SG"}}


def _parse_portuguese(label: str) -> Optional[Set[str]]:
    if label in _PT_SPECIAL:
        return set(_PT_SPECIAL[label])
    m = re.match(r"^([A-Za-z]+)([1-6])$", label)
    if not m or m.group(1) not in _PT_PREFIX:
        return None
    return set(_PT_PREFIX[m.group(1)]) | _PT_PN[m.group(2)]


_CHATINO = {"cpl": "CPL", "hab": "HAB", "pot": "POT", "prog": "PROG", "opt": "OPT"}


def _parse_chatino(label: str) -> Optional[Set[str]]:
    feats: Set[str] = set()
    for tok in label.split():
        m = re.match(r"^([123]?)([A-Za-z]+)$", tok)
        if not m or m.group(2).lower() not in _CHATINO:
            return None
        if m.group(1):
            feats.add(m.group(1))
        feats.add(_CHATINO[m.group(2).lower()])
    return feats or None


CUSTOM_PARSERS: Dict[str, Tuple[str, Callable[[str], Optional[Set[str]]]]] = {
    "english-v": ("custom_english", lambda s: set(_ENGLISH[s]) if s in _ENGLISH else None),
    "french-v": ("custom_french_dotted", _parse_french),
    "hungarian-n": ("custom_hungarian", _parse_hungarian),
    "latvian-n": ("custom_case_words", _parse_spaced_case),
    "russian-n": ("custom_case_words", _parse_spaced_case),
    "latin-n": ("custom_latin_noun", _parse_latin_n),
    "portuguese-v": ("custom_portuguese", _parse_portuguese),
    "yaitepec-chatino-v": ("custom_chatino", _parse_chatino),
    "zenzontepec-chatino-v": ("custom_chatino", _parse_chatino),
}


def normalize_label(label: str, file_stem: str | None = None) -> Tuple[str, str]:
    """Return (cell_norm, method). cell_norm == '' means unparseable."""
    if not isinstance(label, str) or not label.strip():
        return "", "empty"
    lab = label.strip()
    if file_stem in CUSTOM_PARSERS:
        method, fn = CUSTOM_PARSERS[file_stem]
        feats = fn(lab)
        if feats:
            return finalize(feats), method
    cn = project_module("cell_normalization")
    feats = cn.normalize_cell(lab, project_cell_map())
    if feats:
        out = finalize(feats)
        if out:
            if lab in project_cell_map():
                return out, "project_curated_map"
            if ";" in lab:
                return out, "unimorph_native"
            return out, "project_normaliser"
    if re.fullmatch(r"[A-Z][A-Z0-9+.]*", lab):
        # bare upper-case UniMorph-style tag not in the project's single-feature list
        return finalize({lab}), "bare_uppercase_token"
    return "", "unparseable"


def feature_set(cell_norm: str) -> frozenset:
    return frozenset(cell_norm.split(";")) if cell_norm else frozenset()
