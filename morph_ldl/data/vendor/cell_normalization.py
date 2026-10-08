#!/usr/bin/env python3
"""
Normalisation of MGN paradigm cell labels into UniMorph feature sets.

MGN cell labels are not written in one convention. Guzman Naranjo's datasets come
from many sources, and each kept the labelling of its origin corpus. Before any
symmetric-difference distance can be computed the labels have to be put on a common
footing, otherwise the distance measures notational differences instead of
morphosyntactic ones.

Conventions handled, in the order they are tried:

  1. `cells_to_unimorph.json`     curated map for dotted MGN labels
                                  'imp.act.f.2.s' -> 'V;IMP;ACT;2;SG;FEM'
  2. Navajo                       'FUT.3apl:IPA'  -> {FUT, 4, PL}
  3. Native UniMorph              '1;IND;PL;PRF;PRS' -> {1, IND, PL, PRF, PRS}
  4. Polish colon format          'pl:acc:m1.p1'  -> {PL, ACC, MASC}
  5. Dotted, uncurated            'pst.ptcp.f.pl' -> {V.PTCP, PST, FEM, PL}
  6. Bare single feature          'GEN'           -> {GEN}

Anything else is reported as unparseable and its trials are dropped rather than
being turned into an opaque one-token "feature", which would give every pair
involving it the same meaningless distance.

Bare POS features (V, N, ADJ, ADV) are stripped by the caller: POS is already a
column, and it is present in convention 1 but absent from convention 3, so leaving
it in adds a spurious +1 to every cross-convention distance.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Set

# --------------------------------------------------------------------------
# Navajo (nav): '<TAM>.<person><number>:IPA'
# --------------------------------------------------------------------------
# Guzman Naranjo (2024, p.455): Navajo verbs inflect for 7 persons
# (1, 2, 3, 3o, 3a "fourth person", 3s "space", 3i "indefinite"),
# 3 numbers (singular, dual, plural) and 5 TAM categories.
# The ':IPA' suffix marks the phonemic transcription variant and is constant
# across every Navajo cell, so it carries no contrast.
NAVAJO_TAM: Set[str] = {"FUT", "IPFV", "ITER", "OPT", "PFV"}

# Person component -> feature. UniMorph '4' is the standard tag for fourth person.
# '3o' and '3s' have no UniMorph equivalent and keep language-specific tags so the
# contrast between them is preserved without asserting a false analysis.
NAVAJO_PERSON: Dict[str, str] = {
    "1": "1",
    "2": "2",
    "3": "3",
    "3a": "4",       # fourth person
    "3i": "INDF",    # indefinite
    "3o": "3o",      # Navajo-specific third person category
    "3s": "3s",      # spatial
}
NAVAJO_NUMBER: Dict[str, str] = {"": "SG", "dl": "DU", "pl": "PL"}

_NAVAJO_RE = re.compile(r"^(?P<tam>[A-Z]+)\.(?P<pn>[0-9a-z]+):IPA$")


def _parse_navajo(cell: str) -> Optional[Set[str]]:
    m = _NAVAJO_RE.match(cell)
    if not m or m.group("tam") not in NAVAJO_TAM:
        return None
    pn = m.group("pn")
    # Longest-matching person prefix, remainder is the number suffix.
    for person in sorted(NAVAJO_PERSON, key=len, reverse=True):
        if pn.startswith(person):
            number = pn[len(person):]
            if number in NAVAJO_NUMBER:
                return {m.group("tam"), NAVAJO_PERSON[person], NAVAJO_NUMBER[number]}
    return None


# --------------------------------------------------------------------------
# Polish (pol): '<number>:<case(s)>[:<gender classes>]'
# --------------------------------------------------------------------------
# From the Grammatical Dictionary of Polish tagset. The third field lists the
# gender/animacy classes that share this form (m1 masc personal, m2 masc animate,
# m3 masc inanimate, f feminine, n1/n2 neuter, p1/p2/p3 plural-only classes).
# That is a syncretism annotation, so it is reduced to the set of genders involved
# rather than the raw class list, whose length would otherwise drive the distance.
POLISH_NUMBER: Dict[str, str] = {"sg": "SG", "pl": "PL"}
POLISH_CASE: Dict[str, str] = {
    "nom": "NOM", "gen": "GEN", "dat": "DAT", "acc": "ACC",
    "inst": "INS", "loc": "LOC", "voc": "VOC",
}
POLISH_GENDER_PREFIX: Dict[str, str] = {"m": "MASC", "f": "FEM", "n": "NEUT"}


def _parse_polish(cell: str) -> Optional[Set[str]]:
    if ":" not in cell:
        return None
    parts = cell.split(":")
    if len(parts) not in (2, 3):
        return None
    if parts[0] not in POLISH_NUMBER:
        return None
    feats = {POLISH_NUMBER[parts[0]]}
    cases = [c for c in parts[1].split(".") if c]
    if not cases or not all(c in POLISH_CASE for c in cases):
        return None
    feats |= {POLISH_CASE[c] for c in cases}
    if len(parts) == 3:
        for cls in parts[2].split("."):
            if cls and cls[0] in POLISH_GENDER_PREFIX:
                feats.add(POLISH_GENDER_PREFIX[cls[0]])
            # 'p1'/'p2'/'p3' are plural-only classes: number is already encoded.
    return feats


# --------------------------------------------------------------------------
# Dotted MGN labels not present in cells_to_unimorph.json
# --------------------------------------------------------------------------
DOTTED_TOKEN_MAP: Dict[str, str] = {
    # person
    "1": "1", "2": "2", "3": "3",
    # number
    "s": "SG", "sg": "SG", "p": "PL", "pl": "PL", "d": "DU", "du": "DU",
    # gender
    "m": "MASC", "f": "FEM", "n": "NEUT",
    "m/f": "MASC|FEM",     # syncretic masculine/feminine, one contrast
    # tense / aspect / mood
    "prs": "PRS", "pst": "PST", "fut": "FUT", "prf": "PRF", "ipfv": "IPFV",
    "pfv": "PFV", "prog": "PROG", "hab": "HAB", "aor": "AOR", "impf": "IPFV",
    "ind": "IND", "sbjv": "SBJV", "imp": "IMP", "cond": "COND", "opt": "OPT",
    "juss": "JUS", "pot": "POT",
    # voice
    "act": "ACT", "pass": "PASS", "mid": "MID",
    # non-finite
    "ptcp": "V.PTCP", "inf": "NFIN", "cvb": "V.CVB", "msdr": "V.MSDR",
    "ger": "V.MSDR",
    # case
    "nom": "NOM", "gen": "GEN", "dat": "DAT", "acc": "ACC", "voc": "VOC",
    "ins": "INS", "inst": "INS", "loc": "LOC", "abl": "ABL",
    # definiteness / polarity / degree
    "def": "DEF", "indf": "INDF", "neg": "NEG", "pos": "POS",
    "cmpr": "CMPR", "sprl": "SPRL",
}


def _parse_dotted(cell: str) -> Optional[Set[str]]:
    """Parse an uncurated dotted label such as 'pst.ptcp.f.pl' or 'sbjv..pass.m/f.2.d'."""
    if "." not in cell:
        return None
    tokens = [t for t in cell.lower().split(".") if t]  # empty tokens from '..'
    if not tokens:
        return None
    feats: Set[str] = set()
    for tok in tokens:
        if tok not in DOTTED_TOKEN_MAP:
            return None
        feats.add(DOTTED_TOKEN_MAP[tok])
    return feats


# --------------------------------------------------------------------------
# Bare single UniMorph features
# --------------------------------------------------------------------------
KNOWN_SINGLE_FEATURES: Set[str] = {
    "NOM", "GEN", "DAT", "ACC", "INS", "LOC", "VOC", "ABL", "ESS", "TRANS",
    "DEF", "INDF", "CMPR", "SPRL", "POS", "NEG",
    "SG", "PL", "DU", "PROG", "PRF", "IPFV", "PFV", "HAB",
}


def _parse_bare(cell: str) -> Optional[Set[str]]:
    tok = cell.strip().upper()
    return {tok} if tok in KNOWN_SINGLE_FEATURES else None


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def normalize_cell(cell: str, cell_map: Dict[str, str]) -> Optional[Set[str]]:
    """Return the UniMorph feature set for a cell label, or None if unparseable.

    Bare POS features are NOT stripped here; the caller does that so the rule
    lives next to the `pos` column it is redundant with.
    """
    if not isinstance(cell, str):
        return None
    cell = cell.strip()
    if not cell:
        return None

    mapped = cell_map.get(cell)
    if mapped:
        return {f.strip() for f in mapped.split(";") if f.strip()}

    nav = _parse_navajo(cell)
    if nav is not None:
        return nav

    if ";" in cell:
        feats = {f.strip() for f in cell.split(";") if f.strip()}
        return feats or None

    pol = _parse_polish(cell)
    if pol is not None:
        return pol

    dot = _parse_dotted(cell)
    if dot is not None:
        return dot

    return _parse_bare(cell)


if __name__ == "__main__":
    demo = [
        ("FUT.3apl:IPA", {}), ("IPFV.1dl:IPA", {}), ("OPT.3i:IPA", {}),
        ("1;IND;PL;PRF;PRS", {}), ("pl:acc:m1.p1", {}), ("sg:nom.voc:f", {}),
        ("pst.ptcp.f.pl", {}), ("sbjv..pass.m/f.2.d", {}), ("GEN", {}),
        ("prepositional plural", {}), ("pastnot13", {}),
    ]
    for c, m in demo:
        print(f"{c:26} -> {normalize_cell(c, m)}")
