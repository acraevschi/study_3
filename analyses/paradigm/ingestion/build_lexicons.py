#!/usr/bin/env python3
"""Build cleaned, deduplicated, root-grouped Track 2 morphological lexicons and CV splits.

Implements the locked-in Track 2 morphological paradigm pipeline:
  1. Priority source cascade per `(glottocode, pos)` across accepted GeLaTo Glottocodes:
       - Priority 1: Paralex v1 Zenodo releases & documented CELEX English V (exporting BOTH
         `{glottocode}_{pos}_orth` and `{glottocode}_{pos}_phon` + `{unit_id}_sounds.csv`
         for the paired Orth-vs-IPA check).
       - Priority 2: Curated open national lexicons (`Morph-it! v0.48` for Italian V/N,
         `PoliMorf 0.6.7` for Polish V/N, `VESUM v6.8.6` for Ukrainian V/N).
       - Priority 3: Official raw orthographic `github.com/unimorph/*` repositories +
         `SIGMORPHON 2020 Task 0` Uralic tables (`mdf`, `myv`, `udm`).
  2. NFC normalization, `#DEF#` defective-cell handling, periphrastic multiword cell stripping
     (`max_multiword_share <= 0.5`), canonical synthetic cell grid extraction, lexicon-wide
     100%-syncretic cell deduplication (`data/cell_maps/{unit_id}_cells.csv`), and complete
     synthetic paradigm retention (`variant_idx == 0` non-missing across all retained cells;
     additional variants `variant_idx > 0` preserved for overabundance awareness).
  3. Three-rule Union-Find root-cognate grouping (`grouping.py`) -> `leakage_group_id`,
     per-lemma syncretism partition hash -> `shape_id`, keeping units with
     `n_cells_distinct >= 2` and `n_leakage_groups >= 150` (`tier3`: 150-599; `tier2`: >=600).
  4. Deterministic SHA-256 CV splits (`splits.py`) -> `uniform120` and `uniform480`.
  5. Malouf (2017) unsupervised harmony pre-abstraction (`harmony_malouf.py`) ->
     `data/lexicons_malouf/{unit_id}.csv` for Uralic/Turkic/Mongolic/Tungusic/Chukotko-Kamchatkan.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import hashlib
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from analyses.paradigm.ingestion.fetch_sources import (
        EXTERNAL_ROOT,
        PARALEX_SPECS,
        PROJECT_ROOT,
        UNIMORPH_SPECS,
        find_unimorph_data_files,
    )
    from analyses.paradigm.ingestion.grouping import IPA_MODIFIERS, assign_leakage_groups, nfc_clean
    from analyses.paradigm.ingestion.harmony_malouf import HARMONY_FAMILIES, apply_malouf_abstraction
    from analyses.paradigm.ingestion.splits import export_unit_splits
else:
    from .fetch_sources import (
        EXTERNAL_ROOT,
        PARALEX_SPECS,
        PROJECT_ROOT,
        UNIMORPH_SPECS,
        find_unimorph_data_files,
    )
    from .grouping import IPA_MODIFIERS, assign_leakage_groups, nfc_clean
    from .harmony_malouf import HARMONY_FAMILIES, apply_malouf_abstraction
    from .splits import export_unit_splits

DATA_DIR = PROJECT_ROOT / "analyses" / "paradigm" / "data"
LEXICONS_DIR = DATA_DIR / "lexicons"
MALOUF_DIR = DATA_DIR / "lexicons_malouf"
CELL_MAPS_DIR = DATA_DIR / "cell_maps"
SPLITS_DIR = DATA_DIR / "splits"
SUMMARY_PATH = DATA_DIR / "unit_summary.csv"
EXCLUDED_LOG_PATH = EXTERNAL_ROOT / "excluded_candidates_log.csv"

LEXICON_COLUMNS = [
    "unit_id",
    "glottocode",
    "iso639_3",
    "pos",
    "representation",
    "source_id",
    "lemma_id",
    "lemma_label",
    "leakage_group_id",
    "cell",
    "raw_form",
    "segmented_form",
    "variant_idx",
    "shape_id",
]

SUMMARY_COLUMNS = [
    "unit_id",
    "glottocode",
    "iso639_3",
    "language_name",
    "family_id",
    "family_name",
    "pos",
    "representation",
    "priority_tier",
    "source_id",
    "tier",
    "n_lemmas",
    "n_leakage_groups",
    "n_multi_lemma_groups",
    "n_cells_raw",
    "n_cells_synthetic",
    "n_cells_distinct",
    "n_forms_total",
    "overabundance_rate",
    "has_uniform120",
    "has_uniform480",
    "has_malouf",
    "sounds_file",
]

MISSING_STRINGS: Set[str] = {
    "",
    "NA",
    "N/A",
    "NULL",
    "None",
    "-",
    "--",
    "—",
    "?",
    "*",
    "#DEF#",
    "DEFECTIVE",
}

INHERENT_NOUN_GENDERS: Set[str] = {"MASC", "FEM", "NEUT", "MASC+FEM"}
INHERENT_NOUN_ANIMACY: Set[str] = {"ANIM", "INAN", "HUM", "NHUM"}
BARE_POS_TAGS: Set[str] = {"V", "N", "ADJ", "ADV", "NOUN", "VERB"}
TAG_SYNONYMS: Dict[str, str] = {"INST": "INS", "SING": "SG", "PLUR": "PL"}

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Glottolog & GeLaTo Metadata Helpers
# ---------------------------------------------------------------------------

def load_glottolog_and_gelato() -> Tuple[Set[str], Dict[str, dict], Dict[str, str]]:
    links_path = PROJECT_ROOT / "outputs" / "pcfp_v2_links_2026_10_20" / "typology" / "analysis_links.csv"
    links_df = pd.read_csv(links_path, keep_default_na=False)
    accepted_gcs = set(links_df["glottocode"].unique()) - {""}

    gl_path = PROJECT_ROOT / "external" / "glottolog-cldf" / "cldf" / "languages.csv"
    gl_df = pd.read_csv(gl_path, keep_default_na=False)
    gl_by_id: Dict[str, dict] = {}
    fam_names: Dict[str, str] = {}
    for row in gl_df.to_dict("records"):
        gl_by_id[row["ID"]] = row
        if row.get("Level") == "family":
            fam_names[row["ID"]] = row["Name"]

    return accepted_gcs, gl_by_id, fam_names


# ---------------------------------------------------------------------------
# Segmentation & Cell Normalization
# ---------------------------------------------------------------------------

def segment_form(raw_form: str, representation: str) -> str:
    """Segment an NFC-normalized surface or phonemic form into space-separated symbols."""
    s = nfc_clean(raw_form)
    if not s:
        return ""
    if representation == "phon":
        if " " in s:
            return " ".join(tok for tok in _WS.split(s) if tok)
        out_p: List[str] = []
        for ch in s:
            cat = unicodedata.category(ch)
            if out_p and (cat in ("Mn", "Me") or ch in IPA_MODIFIERS):
                out_p[-1] += ch
            else:
                out_p.append(ch)
        return " ".join(out_p)

    # Orthographic: grapheme clusters per word, with internal word spaces mapped to `_`
    syms: List[str] = []
    for w_idx, word in enumerate(_WS.split(s)):
        if w_idx > 0:
            syms.append("_")
        chars: List[str] = []
        for ch in word:
            cat = unicodedata.category(ch)
            if chars and cat in ("Mn", "Me"):
                chars[-1] += ch
            else:
                chars.append(ch)
        syms.extend(chars)
    return " ".join(syms)


def is_multiword_form(raw_form: str, representation: str, orth_hint: str = "") -> bool:
    if orth_hint:
        return bool(_WS.search(orth_hint.strip()))
    if representation == "phon":
        # In Paralex phonemic notation, word boundaries inside periphrastic forms use `#` or `_`
        return ("_" in raw_form) or ("#" in raw_form and raw_form.strip() != "#DEF#")
    return bool(_WS.search(raw_form.strip()))


def normalize_cell_tag(raw_tag: str) -> str:
    parts = [p.strip().upper() for p in re.split(r"[;,.|\s]+", str(raw_tag).strip()) if p.strip()]
    cleaned = set()
    for p in parts:
        p = TAG_SYNONYMS.get(p, p)
        if p in BARE_POS_TAGS:
            continue
        cleaned.add(p)
    return ";".join(sorted(cleaned))


_CYRILLIC_STRESS_ISOS: Set[str] = {
    "ady", "bak", "bel", "bul", "ckt", "evn", "itl", "kaz", "kbd", "ket",
    "khk", "kir", "kjh", "mdf", "mkd", "myv", "rus", "sah", "tat", "tgk",
    "tyv", "udm", "ukr",
}

_NON_LATIN_SCRIPT_ISOS: Set[str] = {
    "ady", "afb", "amh", "arz", "bak", "bel", "ben", "bul", "ckt", "ell",
    "guj", "hye", "jpn", "kat", "kaz", "kbd", "ket", "khk",
    "kjh", "kor", "mdf", "myv", "pbt", "rus", "sah", "tel",
    "tgk", "tyv", "udm", "ukr", "ydd",
}

_WIKTIONARY_HEADER_GARBAGE: Set[str] = {
    "indefinite",
    "definite",
    "singular",
    "plural",
    "no-table-tags",
    "table-tags",
    "countable",
    "uncountable",
    "plurale tantum",
    "singulare tantum",
    "file",
    "audio",
}

_ASCII_LATIN_RE = re.compile(r"[A-Za-z]")


def strip_cyrillic_stress(s: str) -> str:
    """Remove dictionary acute/grave stress combining marks on Cyrillic orthography (keeping ё/й/ў!)."""
    nfd = unicodedata.normalize("NFD", s)
    filtered = "".join(ch for ch in nfd if ch not in ("\u0301", "\u0300"))
    return unicodedata.normalize("NFC", filtered)


def normalize_orth_surface(s: str, iso639_3: str) -> str:
    """Normalize NFC and strip non-orthographic dictionary stress/pitch diacritics in `orth`."""
    s = nfc_clean(s)
    if not s:
        return ""
    if iso639_3 in _CYRILLIC_STRESS_ISOS:
        return strip_cyrillic_stress(s)
    if iso639_3 == "lit":
        # Strip Baltic lexical pitch-accent marks (grave U+0300, acute U+0301, tilde U+0303)
        # while preserving Lithuanian orthographic marks: ogonek U+0328 (ąęįų), dot-above U+0307 (ė),
        # macron U+0304 (ū), and caron U+030C (čšž).
        nfd = unicodedata.normalize("NFD", s)
        filtered = "".join(ch for ch in nfd if ch not in ("\u0300", "\u0301", "\u0303"))
        return unicodedata.normalize("NFC", filtered)
    if iso639_3 == "hbs":
        # Strip Serbo-Croatian dictionary pitch/length marks (U+0300, U+0302, U+0304, U+030F, U+0311)
        # while preserving acute U+0301 (ć) and caron U+030C (čšž).
        nfd = unicodedata.normalize("NFD", s)
        filtered = "".join(ch for ch in nfd if ch not in ("\u0300", "\u0302", "\u0304", "\u030F", "\u0311"))
        return unicodedata.normalize("NFC", filtered)
    return s


def _is_clean_hangul_word(s: str) -> bool:
    for ch in s:
        cp = ord(ch)
        if (0xAC00 <= cp <= 0xD7A3) or (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F):
            continue
        return False
    return bool(s)


def is_polluted_entry(
    lemma_label: str,
    raw_form: str,
    representation: str,
    iso639_3: str,
) -> bool:
    """Reject Wiktionary HTML headers, bound affixes (`-`), multiword lemmas, and cross-script typos."""
    if not lemma_label or not raw_form:
        return True
    # Bound derivational affixes (`-csál`, `-뜨리다`, `rêze-`, `اخّـ`) and multiword citation idioms (`Aa machen`)
    if (
        lemma_label.startswith(("-", "\u0640"))
        or lemma_label.endswith(("-", "\u0640"))
        or bool(_WS.search(lemma_label))
    ):
        return True
    if representation != "orth":
        return False
    if raw_form.startswith(("-", "\u0640")) or raw_form.endswith(("-", "\u0640")):
        return True
    rf_cf = raw_form.casefold()
    if rf_cf in _WIKTIONARY_HEADER_GARBAGE and rf_cf != lemma_label.casefold():
        return True
    if iso639_3 == "kor":
        if not (_is_clean_hangul_word(lemma_label) and _is_clean_hangul_word(raw_form)):
            return True
    elif iso639_3 in _NON_LATIN_SCRIPT_ISOS:
        if _ASCII_LATIN_RE.search(lemma_label) or _ASCII_LATIN_RE.search(raw_form):
            return True
    return False


# ---------------------------------------------------------------------------
# Raw Source Readers -> Standardized Long Records
#   Record tuple: (lemma_key, lemma_label, raw_cell, raw_form, orth_hint, pref_rank)
# ---------------------------------------------------------------------------

def read_paralex_records(
    source_id: str,
    ext_dir: Path,
    pos_target: str,
    representation: str,
    strip_cyr_stress: bool = False,
) -> Tuple[List[Tuple[str, str, str, str, str, int]], Optional[Path]]:
    """Read a Paralex package directory into standardized long records for `(pos_target, representation)`."""
    csv_files = [
        p for p in sorted(ext_dir.rglob("*.csv"))
        if "evaluation" not in p.parts and "raw" not in p.parts and "epitran" not in p.parts and "etc" not in p.parts and "sources" not in p.parts
    ]
    cells_files = [p for p in csv_files if p.name.lower().endswith("cells.csv")]
    lex_files = [p for p in csv_files if p.name.lower().endswith("lexemes.csv")]
    sounds_files = [p for p in csv_files if p.name.lower().endswith("sounds.csv")]
    forms_files = [
        p for p in csv_files
        if re.search(r"(?:forms\d*|paradigms)\.csv$", p.name.lower())
    ]
    if not forms_files:
        return [], None

    # Map cell_id (both exact and lowercased) -> (pos_str, normalized UniMorph tag)
    cell_map: Dict[str, Tuple[str, str]] = {}
    if cells_files:
        cdf = pd.read_csv(cells_files[0], keep_default_na=False)
        pos_col = next((c for c in cdf.columns if c.lower() == "pos"), None)
        um_col = next((c for c in cdf.columns if c.lower() == "unimorph"), None)
        for row in cdf.to_dict("records"):
            cid = str(row.get("cell_id", "")).strip()
            if not cid:
                continue
            cpos_raw = str(row.get(pos_col, "")).strip().lower() if pos_col else ""
            um_raw = str(row.get(um_col, "")).strip() if um_col else ""
            cpos = ""
            if cpos_raw.startswith("v") or "V" in um_raw.split(";"):
                cpos = "V"
            elif cpos_raw.startswith("n") or "N" in um_raw.split(";"):
                cpos = "N"
            cnorm = normalize_cell_tag(um_raw if um_raw else cid)
            cell_map[cid] = (cpos, cnorm)
            cell_map[cid.lower()] = (cpos, cnorm)

    # Map lexeme_id -> (lemma_label, pos_str)
    lex_map: Dict[str, Tuple[str, str]] = {}
    if lex_files:
        ldf = pd.read_csv(lex_files[0], keep_default_na=False)
        id_col = "lexeme_id" if "lexeme_id" in ldf.columns else ldf.columns[0]
        lab_col = next((c for c in ("label", "lemma", "lexeme_id") if c in ldf.columns), id_col)
        pos_col_l = next((c for c in ldf.columns if c.lower() in ("pos", "post")), None)
        for row in ldf.to_dict("records"):
            lid = str(row[id_col]).strip()
            lab = nfc_clean(row.get(lab_col, lid)) or lid
            if strip_cyr_stress:
                lab = strip_cyrillic_stress(lab)
            lpos_raw = str(row.get(pos_col_l, "")).strip().lower() if pos_col_l else ""
            lpos = "V" if lpos_raw.startswith("v") else ("N" if lpos_raw.startswith("n") else "")
            lex_map[lid] = (lab, lpos)

    col_want = "orth_form" if representation == "orth" else "phon_form"
    records: List[Tuple[str, str, str, str, str, int]] = []
    for fpath in forms_files:
        fdf = pd.read_csv(fpath, keep_default_na=False)
        if col_want not in fdf.columns:
            return [], (sounds_files[0] if sounds_files else None)
        has_orth = "orth_form" in fdf.columns
        has_def_tag = "defectiveness_tag" in fdf.columns
        for row in fdf.to_dict("records"):
            if has_def_tag and str(row.get("defectiveness_tag", "")).strip().lower() in ("defective", "#def#", "missing"):
                continue
            val = nfc_clean(row.get(col_want, ""))
            if not val or val in MISSING_STRINGS:
                continue
            if strip_cyr_stress and representation == "orth":
                val = strip_cyrillic_stress(val)
            lid = str(row.get("lexeme", "")).strip()
            cid = str(row.get("cell", "")).strip()
            if not lid or not cid:
                continue
            lab, lpos = lex_map.get(lid, (lid, ""))
            cpos, cnorm = cell_map.get(cid, cell_map.get(cid.lower(), ("", normalize_cell_tag(cid))))
            # Determine POS from lexeme table, cell table, or dataset default
            eff_pos = lpos or cpos
            if not eff_pos:
                # Single-POS verb Paralex datasets (LeFFI, Vlexique, VeLeSpa, VeLeCa, VeLeRo)
                eff_pos = "N" if ("zal" in source_id or "hung" in source_id or "naslex" in source_id or "parafin" in source_id) else "V"
            if eff_pos != pos_target or not cnorm:
                continue
            orth_hint = nfc_clean(row.get("orth_form", "")) if has_orth else ""
            records.append((lid, lab, cnorm, val, orth_hint, 0))

    return records, (sounds_files[0] if sounds_files else None)


def read_celex_english_v() -> List[Tuple[str, str, str, str, str, int]]:
    path = PROJECT_ROOT / "mgn_data" / "data-custom" / "english-v.csv"
    eng_cell_map = {
        "inf": "NFIN",
        "pres1s": "1;PRS;SG",
        "pres3s": "3;PRS;SG",
        "presothers": "LGSPEC1;PRS",
        "past13": "LGSPEC1;PST",
        "pastnot13": "LGSPEC2;PST",
        "ppart": "PST;V.PTCP",
        "prespart": "PRS;V.PTCP",
    }
    df = pd.read_csv(path, keep_default_na=False)
    records: List[Tuple[str, str, str, str, str, int]] = []
    for row in df.to_dict("records"):
        lem = nfc_clean(row["lexeme"])
        c_raw = str(row["cell"]).strip()
        form = nfc_clean(row["form"])
        if not lem or not form or form in MISSING_STRINGS:
            continue
        cnorm = eng_cell_map.get(c_raw, normalize_cell_tag(c_raw))
        # Skip multiword phrasal verbs in citation label (e.g. "abut on") for clean single-word parity
        orth_hint = lem if " " in lem else ""
        records.append((lem, lem, cnorm, form, orth_hint, 0))
    return records


_MORPHIT_VER_MAP = {
    "inf+pres": "NFIN",
    "ger+pres": "PRS;V.CVB",
    "part+pres+s+m": "MASC;PRS;SG;V.PTCP",
    "part+pres+s+f": "FEM;PRS;SG;V.PTCP",
    "part+pres+p+m": "MASC;PL;PRS;V.PTCP",
    "part+pres+p+f": "FEM;PL;PRS;V.PTCP",
    "part+past+s+m": "MASC;PST;SG;V.PTCP",
    "part+past+s+f": "FEM;PST;SG;V.PTCP",
    "part+past+p+m": "MASC;PL;PST;V.PTCP",
    "part+past+p+f": "FEM;PL;PST;V.PTCP",
}


def _parse_morphit_ver_tag(tag_body: str) -> Optional[str]:
    if tag_body in _MORPHIT_VER_MAP:
        return _MORPHIT_VER_MAP[tag_body]
    parts = tag_body.split("+")
    # Exclude cliticized forms (e.g. inf+pres+si, ind+pres+3+s+lo -> >4 parts or clitic tokens)
    if len(parts) != 4:
        return None
    mood_m = {"ind": "IND", "sub": "SBJV", "cond": "COND", "impr": "IMP"}.get(parts[0])
    tense_m = {
        "pres": "PRS",
        "impf": "IPFV;PST",
        "past": "PFV;PST",
        "fut": "FUT",
    }.get(parts[1])
    pers_m = parts[2] if parts[2] in ("1", "2", "3") else None
    num_m = {"s": "SG", "p": "PL"}.get(parts[3])
    if not (mood_m and tense_m and pers_m and num_m):
        return None
    return normalize_cell_tag(f"{mood_m};{tense_m};{pers_m};{num_m}")


def read_morphit_records(pos_target: str) -> List[Tuple[str, str, str, str, str, int]]:
    path = PROJECT_ROOT / "external" / "morph-it" / "morph-it_048.utf8.txt"
    records: List[Tuple[str, str, str, str, str, int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 3:
                continue
            form_raw, lem_raw, feat = parts[0], parts[1], parts[2]
            form = nfc_clean(form_raw)
            lem = nfc_clean(lem_raw)
            if not form or not lem or form in MISSING_STRINGS:
                continue
            # Demote abbreviations (`Avv.` for `avvocato`) behind genuine lowercase lexical forms
            pref = 1 if ("." in form or (form[0].isupper() and not lem[0].isupper())) else 0

            if pos_target == "N" and feat.startswith("NOUN-"):
                # NOUN-M:s, NOUN-M:p, NOUN-F:s, NOUN-F:p
                if ":" not in feat:
                    continue
                gpart, npart = feat.split(":", 1)
                gender = gpart.split("-", 1)[1] if "-" in gpart else ""
                cnorm = {"s": "SG", "p": "PL"}.get(npart, "")
                if not cnorm:
                    continue
                if pref == 1:
                    continue  # Never treat `Avv.` title abbreviation as an inflected form of `avvocato`
                lkey = f"{lem}#{gender}" if gender else lem
                records.append((lkey, lem, cnorm, form, form, 0))
            elif pos_target == "V" and feat.startswith("VER:"):
                cnorm = _parse_morphit_ver_tag(feat[4:])
                if not cnorm or pref == 1:
                    continue
                records.append((lem, lem, cnorm, form, form, 0))
    return records


_POL_CASE = {"nom": "NOM", "gen": "GEN", "dat": "DAT", "acc": "ACC", "inst": "INS", "loc": "LOC", "voc": "VOC"}
_POL_NUM = {"sg": "SG", "pl": "PL"}
_POL_PERS = {"pri": "1", "sec": "2", "ter": "3"}


def read_polimorf_records(pos_target: str) -> List[Tuple[str, str, str, str, str, int]]:
    gz_path = EXTERNAL_ROOT / "national" / "polimorf" / "PoliMorf-0.6.7.tab.gz"
    records: List[Tuple[str, str, str, str, str, int]] = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[3] != "pospolita":
                continue
            form = nfc_clean(parts[0])
            lem = nfc_clean(parts[1])
            tag = parts[2]
            if not form or not lem or form in MISSING_STRINGS:
                continue
            tparts = tag.split(":")
            t0 = tparts[0]
            if pos_target == "N" and t0 == "subst" and len(tparts) >= 4:
                nums = [ _POL_NUM[x] for x in tparts[1].split(".") if x in _POL_NUM ]
                cases = [ _POL_CASE[x] for x in tparts[2].split(".") if x in _POL_CASE ]
                gend = tparts[3].split(".")[0]
                lkey = f"{lem}#{gend}"
                for nm in nums:
                    for cs in cases:
                        records.append((lkey, lem, f"{cs};{nm}", form, form, 0))
            elif pos_target == "V":
                if t0 == "inf":
                    records.append((lem, lem, "NFIN", form, form, 0))
                elif t0 in ("fin", "bedzie") and len(tparts) >= 3:
                    # Synthetic non-past (`fin` for imperf/perf, `bedzie` for `być` synthetic future; for `być` keep `fin` as PRS/NPST)
                    if t0 == "bedzie" and lem == "być":
                        cnorm_base = "FUT"
                    elif t0 == "fin":
                        cnorm_base = "NPST"
                    else:
                        continue
                    nm = _POL_NUM.get(tparts[1])
                    ps = _POL_PERS.get(tparts[2])
                    if nm and ps and cnorm_base == "NPST":
                        records.append((lem, lem, f"IND;{cnorm_base};{ps};{nm}", form, form, 0))
                elif t0 == "impt" and len(tparts) >= 3:
                    nm = _POL_NUM.get(tparts[1])
                    ps = _POL_PERS.get(tparts[2])
                    if nm and ps and f"{ps};{nm}" in ("2;SG", "1;PL", "2;PL"):
                        records.append((lem, lem, f"IMP;{ps};{nm}", form, form, 0))
                elif t0 == "praet" and len(tparts) >= 3:
                    nm = _POL_NUM.get(tparts[1])
                    gset = set(tparts[2].split("."))
                    if nm == "SG":
                        if gset & {"m1", "m2", "m3"}:
                            records.append((lem, lem, "MASC;PST;SG", form, form, 0))
                        if "f" in gset:
                            records.append((lem, lem, "FEM;PST;SG", form, form, 0))
                        if gset & {"n1", "n2"}:
                            records.append((lem, lem, "NEUT;PST;SG", form, form, 0))
                    elif nm == "PL":
                        if gset & {"m1", "p1"}:
                            records.append((lem, lem, "HUM;PL;PST", form, form, 0))
                        if gset & {"m2", "m3", "f", "n1", "n2", "p2", "p3"}:
                            records.append((lem, lem, "NHUM;PL;PST", form, form, 0))
    return records


_UKR_CASE = {
    "v_naz": "NOM",
    "v_rod": "GEN",
    "v_dav": "DAT",
    "v_zna": "ACC",
    "v_oru": "INS",
    "v_mis": "LOC",
    "v_kly": "VOC",
}
_UKR_DEMOTE_FLAGS = {"rare", "arch", "short", "slang", "bad", "subst", "coll", "dial", "alt", "var"}
_UKR_SKIP_FLAGS = {"prop", "abbr", "fname", "lname", "pname", "geo", "nv", "ns", "np"}


def read_vesum_records(pos_target: str) -> List[Tuple[str, str, str, str, str, int]]:
    bz_path = EXTERNAL_ROOT / "national" / "vesum" / "dict_corp_vis.txt.bz2"
    records: List[Tuple[str, str, str, str, str, int]] = []
    cur_lemma = ""
    cur_key = ""
    skip_block = False

    with bz2.open(bz_path, "rt", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line or line.lstrip().startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            is_indented = raw_line.startswith((" ", "\t"))
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            form = nfc_clean(toks[0])
            tag = toks[1]
            tset = set(tag.split(":"))

            if not is_indented:
                cur_lemma = form
                skip_block = bool(tset & _UKR_SKIP_FLAGS)
                if tag.startswith("noun:"):
                    gend = next((g for g in ("m", "f", "n") if g in tset), "")
                    anim = "anim" if "anim" in tset else "inanim"
                    cur_key = f"{cur_lemma}#{gend}_{anim}"
                else:
                    cur_key = cur_lemma

            if skip_block or not form or not cur_lemma:
                continue
            if tset & {"prop", "abbr", "bad", "arch", "dial", "slang"}:
                continue
            pref = 1 if (tset & _UKR_DEMOTE_FLAGS) else 0

            if pos_target == "N" and tag.startswith("noun:"):
                case_u = next((_UKR_CASE[k] for k in tset if k in _UKR_CASE), None)
                num_u = "PL" if "p" in tset else ("SG" if (tset & {"m", "f", "n", "s"}) else None)
                if case_u and num_u:
                    records.append((cur_key, cur_lemma, f"{case_u};{num_u}", form, form, pref))
            elif pos_target == "V" and tag.startswith("verb:"):
                if "inf" in tset:
                    records.append((cur_key, cur_lemma, "NFIN", form, form, pref))
                elif ("pres" in tset or ("futr" in tset and "perf" in tset)) and (tset & {"1", "2", "3"}):
                    # Synthetic non-past: imperfective `pres` and perfective simple `futr`
                    ps = next(p for p in ("1", "2", "3") if p in tset)
                    nm = "SG" if "s" in tset else ("PL" if "p" in tset else "")
                    if nm:
                        records.append((cur_key, cur_lemma, f"IND;NPST;{ps};{nm}", form, form, pref))
                elif "impr" in tset and (tset & {"1", "2"}):
                    ps = next(p for p in ("1", "2") if p in tset)
                    nm = "SG" if "s" in tset else ("PL" if "p" in tset else "")
                    if nm and f"{ps};{nm}" in ("2;SG", "1;PL", "2;PL"):
                        records.append((cur_key, cur_lemma, f"IMP;{ps};{nm}", form, form, pref))
                elif "past" in tset:
                    if "m" in tset:
                        records.append((cur_key, cur_lemma, "MASC;PST;SG", form, form, pref))
                    elif "f" in tset:
                        records.append((cur_key, cur_lemma, "FEM;PST;SG", form, form, pref))
                    elif "n" in tset:
                        records.append((cur_key, cur_lemma, "NEUT;PST;SG", form, form, pref))
                    elif "p" in tset:
                        records.append((cur_key, cur_lemma, "PL;PST", form, form, pref))
    return records


def _repair_romanian_noun_triples(
    raw_triples: List[Tuple[str, str, Set[str]]],
) -> List[Tuple[str, str, Set[str]]]:
    """Repair upstream `unimorph/ron` nominal `PL;DEF` tag collision on definite singular vs plural.

    In `github.com/unimorph/ron`, both definite singular (`greblei`, `grebla`) and definite
    plural (`greblelor`, `greblele`) were emitted with `PL;DEF`. In Romanian morphology,
    `GEN/DAT;DEF;PL` strictly ends in `-lor` (`greblelor`), whereas `GEN/DAT;DEF;SG` ends in
    `-lui`/`-ei`/`-ii` (`greblei`, `grecului`), and `NOM/ACC;DEF;SG` precedes the `-lor`
    plural block while `NOM/ACC;DEF;PL` follows it.
    """
    by_lem: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)
    for lem, form, ctags in raw_triples:
        by_lem[lem].append((form, ctags))

    repaired: List[Tuple[str, str, Set[str]]] = []
    for lem, rows in by_lem.items():
        gd_def_indices = [
            i for i, (_, ct) in enumerate(rows)
            if {"DEF", "PL"} <= ct and ("GEN/DAT" in ct or {"GEN", "DAT"} <= ct)
        ]
        na_def_indices = [
            i for i, (_, ct) in enumerate(rows)
            if {"DEF", "PL"} <= ct and ("NOM/ACC" in ct or {"NOM", "ACC"} <= ct)
        ]
        has_any_lor_gd = any(rows[j][0].lower().endswith("lor") for j in gd_def_indices)

        for idx, (form, ctags) in enumerate(rows):
            ct = set(ctags)
            if idx in gd_def_indices:
                ct.discard("PL")
                ct.discard("SG")
                ct.add("PL" if form.lower().endswith("lor") else "SG")
            elif idx in na_def_indices:
                has_lor_before = any(
                    rows[j][0].lower().endswith("lor")
                    and (j in gd_def_indices or {"PL", "VOC"} <= rows[j][1])
                    for j in range(idx)
                )
                ct.discard("PL")
                ct.discard("SG")
                if has_lor_before or (len(na_def_indices) == 1 and has_any_lor_gd):
                    ct.add("PL")
                else:
                    ct.add("SG")
            repaired.append((lem, form, ct))
    return repaired


def read_unimorph_files_records(
    paths: Sequence[Path],
    pos_target: str,
    iso_code: str = "",
) -> List[Tuple[str, str, str, str, str, int]]:
    """Read one or more standard 3-column UniMorph TSV files for `pos_target in {'V', 'N'}`."""
    raw_triples: List[Tuple[str, str, Set[str]]] = []
    seen_exact: Set[Tuple[str, str, str]] = set()

    for fp in paths:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                lem = nfc_clean(parts[0])
                form = nfc_clean(parts[1])
                feats = parts[2].strip()
                if not lem or not form or form in MISSING_STRINGS:
                    continue
                tkey = (lem, form, feats)
                if tkey in seen_exact:
                    continue
                seen_exact.add(tkey)
                tags = {
                    TAG_SYNONYMS.get(t.strip().upper(), t.strip().upper())
                    for t in re.split(r"[;|]+", feats)
                    if t.strip()
                }
                is_v = ("V" in tags) or any(t.startswith("V.") for t in tags)
                is_n = ("N" in tags) and not is_v
                if (pos_target == "V" and not is_v) or (pos_target == "N" and not is_n):
                    continue
                clean_tags = {t for t in tags if t not in BARE_POS_TAGS}
                if not clean_tags:
                    continue
                raw_triples.append((lem, form, clean_tags))

    if not raw_triples:
        return []

    if iso_code == "ron" and pos_target == "N":
        raw_triples = _repair_romanian_noun_triples(raw_triples)

    # For Nouns (`N`), check whether grammatical gender (`MASC`, `FEM`, `NEUT`) or animacy
    # (`ANIM`, `INAN`) is an inherent lexeme feature (meaning <25% of multi-cell lemmas attest
    # multiple values, as in German `N;NOM;MASC;SG`, Spanish `N;MASC;SG`, Russian `N;ANIM;...`).
    factor_gender = False
    factor_anim = False
    if pos_target == "N":
        genders_by_lem: Dict[str, Set[str]] = defaultdict(set)
        anims_by_lem: Dict[str, Set[str]] = defaultdict(set)
        cells_by_lem: Dict[str, int] = Counter()
        for lem, _, ctags in raw_triples:
            cells_by_lem[lem] += 1
            g_hit = ctags & INHERENT_NOUN_GENDERS
            if g_hit:
                genders_by_lem[lem].update(g_hit)
            a_hit = ctags & INHERENT_NOUN_ANIMACY
            if a_hit:
                anims_by_lem[lem].update(a_hit)
        multi_lems = [l for l, cnt in cells_by_lem.items() if cnt >= 2]
        if multi_lems and genders_by_lem:
            multi_g_share = sum(1 for l in multi_lems if len(genders_by_lem.get(l, ())) > 1) / len(multi_lems)
            if multi_g_share < 0.25:
                factor_gender = True
        if multi_lems and anims_by_lem:
            multi_a_share = sum(1 for l in multi_lems if len(anims_by_lem.get(l, ())) > 1) / len(multi_lems)
            if multi_a_share < 0.25:
                factor_anim = True

    records: List[Tuple[str, str, str, str, str, int]] = []
    for lem, form, ctags in raw_triples:
        rem = set(ctags)
        sub_parts: List[str] = []
        if factor_gender:
            gh = sorted(rem & INHERENT_NOUN_GENDERS)
            if gh:
                sub_parts.append("+".join(gh))
                rem -= INHERENT_NOUN_GENDERS
        if factor_anim:
            ah = sorted(rem & INHERENT_NOUN_ANIMACY)
            if ah:
                sub_parts.append("+".join(ah))
                rem -= INHERENT_NOUN_ANIMACY
        if not rem:
            continue
        lkey = f"{lem}#{'_'.join(sub_parts)}" if sub_parts else lem
        cnorm = ";".join(sorted(rem))
        records.append((lkey, lem, cnorm, form, form, 0))

    return records


# ---------------------------------------------------------------------------
# Core Cleaning, Cell Deduplication, Completeness & Shape Hashing
# ---------------------------------------------------------------------------

def choose_canonical_cell_name(syncretic_group: Sequence[str], pos: str) -> str:
    """Pick deterministic canonical cell tag when multiple cells are 100% syncretic lexicon-wide.

    Prefers standard citation tags (`NFIN` for V; `NOM;SG` for N) so e.g. Finnish nominal
    `NOM;SG == ACC;SG` retains `NOM;SG` rather than alphabetically earlier `ACC;SG`.
    """
    gset = set(syncretic_group)
    if pos == "V":
        for pref in ("NFIN", "IND;PRS;3;SG", "PRS;3;SG"):
            if pref in gset:
                return pref
    elif pos == "N":
        for pref in ("NOM;SG", "INDF;NOM;SG", "SG", "ABS;SG"):
            if pref in gset:
                return pref
    return sorted(syncretic_group, key=lambda c: (len(c.split(";")), len(c), c))[0]


def compute_shape_id(forms_ordered: Sequence[str]) -> str:
    """Hash the cell-syncretism equivalence partition among a lemma's `variant_idx == 0` forms."""
    seen: Dict[str, int] = {}
    part: List[int] = []
    for f in forms_ordered:
        if f not in seen:
            seen[f] = len(seen)
        part.append(seen[f])
    payload = ",".join(str(x) for x in part).encode("utf-8")
    return "sh_" + hashlib.sha256(payload).hexdigest()[:12]


def process_candidate_unit(
    unit_id: str,
    glottocode: str,
    iso639_3: str,
    pos: str,
    representation: str,
    source_id: str,
    priority_tier: str,
    records: List[Tuple[str, str, str, str, str, int]],
    max_multiword_share: float = 0.50,
    min_groups: int = 150,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], dict]:
    """Clean a candidate `(glottocode, pos, representation)` dataset and check eligibility."""
    diag = {
        "unit_id": unit_id,
        "glottocode": glottocode,
        "iso639_3": iso639_3,
        "pos": pos,
        "representation": representation,
        "source_id": source_id,
        "priority_tier": priority_tier,
        "n_raw_records": len(records),
        "reason": "",
    }
    if not records:
        diag["reason"] = "empty_records"
        return None, None, diag

    # Normalize dictionary diacritics (`orth`), filter Wiktionary headers / bound affixes / multiword
    # citation idioms, and deduplicate exact `(lkey, cell, raw_form)` keeping lowest `pref_rank`
    slot_variants_raw: Dict[Tuple[str, str], List[Tuple[int, int, str, str]]] = defaultdict(list)
    label_of_key: Dict[str, str] = {}
    seen_slot_form: Set[Tuple[str, str, str]] = set()

    for order_idx, (lkey, llabel, cell, rform, ohint, prank) in enumerate(records):
        if representation == "orth":
            llabel = normalize_orth_surface(llabel, iso639_3)
            rform = normalize_orth_surface(rform, iso639_3)
            if "#" in lkey:
                lkey = f"{llabel}#{lkey.split('#', 1)[1]}"
            else:
                lkey = normalize_orth_surface(lkey, iso639_3) if lkey == nfc_clean(lkey) else lkey
        if not rform or rform in MISSING_STRINGS:
            continue
        if is_polluted_entry(llabel, rform, representation, iso639_3):
            continue
        label_of_key.setdefault(lkey, llabel)
        sf = (lkey, cell, rform)
        if sf in seen_slot_form:
            continue
        seen_slot_form.add(sf)
        slot_variants_raw[(lkey, cell)].append((prank, order_idx, rform, ohint))

    # Compute raw cell counts and multiword shares across all cells (for `cell_maps/` logging)
    all_raw_cells: Counter = Counter()
    mw_raw_cells: Counter = Counter()

    for (lkey, cell), vlist in slot_variants_raw.items():
        vlist.sort(key=lambda t: (t[0], t[1]))
        rform0, ohint0 = vlist[0][2], vlist[0][3]
        all_raw_cells[cell] += 1
        if is_multiword_form(rform0, representation, ohint0):
            mw_raw_cells[cell] += 1

    n_cells_raw = len(all_raw_cells)
    if n_cells_raw < 2:
        diag["reason"] = f"fewer_than_2_raw_cells ({n_cells_raw})"
        return None, None, diag

    # Step 1: Strip periphrastic multiword cells (`multiword_share > 0.50`) AND drop any residual
    # space-containing / multiword periphrastic rows inside retained synthetic cells
    synth_cells: Set[str] = set()
    cell_status: Dict[str, Tuple[str, str]] = {}
    for cell, cnt in all_raw_cells.items():
        mw_sh = mw_raw_cells[cell] / max(1, cnt)
        if mw_sh <= max_multiword_share:
            synth_cells.add(cell)
        else:
            cell_status[cell] = ("dropped_periphrastic", "")

    if len(synth_cells) < 2:
        diag["reason"] = f"fewer_than_2_synthetic_cells ({len(synth_cells)})"
        return None, None, diag

    slot_variants: Dict[Tuple[str, str], List[Tuple[int, int, str, str]]] = {}
    v0_by_lem: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
    for (lkey, cell), vlist in slot_variants_raw.items():
        if cell not in synth_cells:
            continue
        clean_vlist = [
            tup for tup in vlist
            if not is_multiword_form(tup[2], representation, tup[3])
        ]
        if not clean_vlist:
            continue
        slot_variants[(lkey, cell)] = clean_vlist
        v0_by_lem[lkey][cell] = (clean_vlist[0][2], clean_vlist[0][3])

    # Filter out 1-cell stub lemmas before computing synthetic cell grid coverage
    multi_lems = {
        lkey: {c: pair[0] for c, pair in cdict.items() if c in synth_cells}
        for lkey, cdict in v0_by_lem.items()
    }
    multi_lems = {lkey: cdict for lkey, cdict in multi_lems.items() if len(cdict) >= 2}
    if len(multi_lems) < min_groups:
        diag["reason"] = f"fewer_than_{min_groups}_multicell_lemmas ({len(multi_lems)})"
        return None, None, diag

    synth_counts = Counter(c for cdict in multi_lems.values() for c in cdict)
    max_cnt = max(synth_counts.values())

    # Select canonical synthetic cell grid: try standard `0.50 * max_cnt` first; if sub-POS or
    # sparse field sampling drops complete paradigms below `min_groups`, try stricter core cell
    # coverage thresholds `(0.65, 0.80, 0.90)` or greedy core grid >= min_groups.
    retained_synth_cells: List[str] = []
    complete_lkeys: List[str] = []

    for cov_frac in (0.50, 0.65, 0.80, 0.90):
        cand_cells = sorted(
            c for c in synth_cells
            if synth_counts[c] >= max(min_groups, int(cov_frac * max_cnt))
        )
        if len(cand_cells) < 2:
            continue
        cset = set(cand_cells)
        comp = sorted(lk for lk, d in multi_lems.items() if cset <= d.keys())
        if len(comp) >= min_groups:
            retained_synth_cells = cand_cells
            complete_lkeys = comp
            break

    if not retained_synth_cells:
        # Fallback greedy frequent-cell core grid (for field/SIGMORPHON tables like mdf/myv/udm/zul)
        ordered_c = [c for c, cnt in synth_counts.most_common() if cnt >= min_groups]
        comp_set = set(multi_lems.keys())
        grid_list: List[str] = []
        for c in ordered_c:
            nxt = {lk for lk in comp_set if c in multi_lems[lk]}
            if len(nxt) >= min_groups:
                grid_list.append(c)
                comp_set = nxt
        if len(grid_list) >= 2 and len(comp_set) >= min_groups:
            retained_synth_cells = sorted(grid_list)
            complete_lkeys = sorted(comp_set)

    if len(retained_synth_cells) < 2 or len(complete_lkeys) < min_groups:
        diag["reason"] = (
            f"insufficient_complete_paradigms (multi_lems={len(multi_lems)}, "
            f"complete={len(complete_lkeys)}, grid={len(retained_synth_cells)})"
        )
        return None, None, diag

    retained_set = set(retained_synth_cells)
    for c in synth_cells - retained_set:
        cell_status[c] = ("dropped_sparse_coverage", "")

    # Step 2: Deduplicate lexicon-wide 100%-syncretic cells across `complete_lkeys`
    col_sigs: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for c in retained_synth_cells:
        sig = tuple(multi_lems[lk][c] for lk in complete_lkeys)
        col_sigs[sig].append(c)

    distinct_cells: List[str] = []
    for _, syn_group in col_sigs.items():
        canon_c = choose_canonical_cell_name(syn_group, pos)
        distinct_cells.append(canon_c)
        cell_status[canon_c] = ("retained_canonical", canon_c)
        for other_c in syn_group:
            if other_c != canon_c:
                cell_status[other_c] = ("merged_100pct_syncretic", canon_c)

    distinct_cells.sort()
    if len(distinct_cells) < 2:
        diag["reason"] = f"pure_syncretic_or_isolating_fewer_than_2_distinct_cells ({len(distinct_cells)})"
        return None, None, diag

    # Build cell_map DataFrame
    cmap_rows: List[dict] = []
    for c in sorted(all_raw_cells.keys()):
        st, target_c = cell_status.get(c, ("dropped_sparse_coverage", ""))
        cmap_rows.append({
            "unit_id": unit_id,
            "raw_cell": c,
            "canonical_cell": target_c if st in ("retained_canonical", "merged_100pct_syncretic") else "",
            "status": st,
            "n_lemmas_filled": int(all_raw_cells[c]),
            "multiword_share": round(mw_raw_cells[c] / max(1, all_raw_cells[c]), 6),
        })
    cell_map_df = pd.DataFrame(cmap_rows)

    # Disambiguate `lemma_id`: `{source_id}::{lemma_label}` (with `#subtag` only if a label has >1 paradigm)
    label_counts = Counter(label_of_key[lk] for lk in complete_lkeys)
    lemma_id_of: Dict[str, str] = {}
    occ_seen: Counter = Counter()
    for lk in complete_lkeys:
        lab = label_of_key[lk]
        if label_counts[lab] == 1:
            lemma_id_of[lk] = f"{source_id}::{lab}"
        else:
            occ_seen[lab] += 1
            suffix = lk.split("#", 1)[1] if "#" in lk else str(occ_seen[lab])
            lemma_id_of[lk] = f"{source_id}::{lab}#{suffix}"

    # Assemble long lexicon rows (including `variant_idx > 0` for overabundance awareness)
    seg_cache: Dict[str, str] = {}
    long_rows: List[dict] = []
    n_overabundant_slots = 0
    n_total_slots = len(complete_lkeys) * len(distinct_cells)

    for lk in complete_lkeys:
        lid = lemma_id_of[lk]
        llab = label_of_key[lk]
        v0_forms_ordered = [multi_lems[lk][c] for c in distinct_cells]
        sh_id = compute_shape_id(v0_forms_ordered)
        for c in distinct_cells:
            vlist = slot_variants[(lk, c)]
            if len(vlist) > 1:
                n_overabundant_slots += 1
            for v_idx, (_, _, rform, _) in enumerate(vlist):
                if rform not in seg_cache:
                    seg_cache[rform] = segment_form(rform, representation)
                long_rows.append({
                    "unit_id": unit_id,
                    "glottocode": glottocode,
                    "iso639_3": iso639_3,
                    "pos": pos,
                    "representation": representation,
                    "source_id": source_id,
                    "lemma_id": lid,
                    "lemma_label": llab,
                    "leakage_group_id": "",
                    "cell": c,
                    "raw_form": rform,
                    "segmented_form": seg_cache[rform],
                    "variant_idx": v_idx,
                    "shape_id": sh_id,
                })

    df_lex = pd.DataFrame(long_rows, columns=LEXICON_COLUMNS)

    # Step 3: Three-rule Union-Find root-cognate grouping (`grouping.py`)
    group_map, _ = assign_leakage_groups(
        df_lex,
        glottocode=glottocode,
        pos=pos,
        representation=representation,
        cells_ordered=distinct_cells,
    )
    df_lex["leakage_group_id"] = df_lex["lemma_id"].map(group_map)

    n_groups = df_lex["leakage_group_id"].nunique()
    if n_groups < min_groups:
        diag["reason"] = f"fewer_than_{min_groups}_leakage_groups ({n_groups} from {len(complete_lkeys)} lemmas)"
        return None, None, diag

    # Count multi-lemma root groups
    lem_grp = df_lex[["lemma_id", "leakage_group_id"]].drop_duplicates("lemma_id")
    grp_sizes = lem_grp.groupby("leakage_group_id").size()
    n_multi_groups = int((grp_sizes > 1).sum())

    diag.update({
        "reason": "accepted",
        "n_lemmas": len(complete_lkeys),
        "n_leakage_groups": int(n_groups),
        "n_multi_lemma_groups": n_multi_groups,
        "n_cells_raw": n_cells_raw,
        "n_cells_synthetic": len(retained_synth_cells),
        "n_cells_distinct": len(distinct_cells),
        "n_forms_total": len(df_lex),
        "overabundance_rate": round(n_overabundant_slots / max(1, n_total_slots), 6),
    })
    return df_lex, cell_map_df, diag


# ---------------------------------------------------------------------------
# Candidate Registry & Priority Cascade Execution
# ---------------------------------------------------------------------------

def collect_candidates(accepted_gcs: Set[str]) -> Dict[Tuple[str, str, str], List[Tuple[str, str, str, callable]]]:
    """Build ordered candidate list per `(glottocode, pos, representation)` obeying Priority 1 > 2 > 3."""
    by_slot: Dict[Tuple[str, str, str], List[Tuple[str, str, str, callable]]] = defaultdict(list)

    # 1. Priority 1: Paralex Zenodo packages
    paralex_by_id = {s.source_id: s for s in PARALEX_SPECS}
    paralex_targets = [
        ("paralex_leffi", "ital1282", "ita", ["V"], ["phon"], False),
        ("paralex_vlexique", "stan1290", "fra", ["V"], ["orth", "phon"], False),
        ("paralex_velespa", "stan1288", "spa", ["V"], ["phon"], False),
        ("paralex_veleca", "stan1289", "cat", ["V"], ["orth", "phon"], False),
        ("paralex_velero", "roma1327", "ron", ["V"], ["phon"], False),
        ("paralex_eesthetic", "esto1258", "ekk", ["N", "V"], ["orth", "phon"], False),
        ("paralex_parafin", "finn1318", "fin", ["N"], ["orth", "phon"], False),
        ("paralex_zalilex", "russ1263", "rus", ["N"], ["orth", "phon"], True),
        ("paralex_parahungarian", "hung1274", "hun", ["N"], ["phon"], False),
        ("paralex_naslex", "sout1528", "hbs", ["N"], ["orth", "phon"], False),
    ]
    for sid, gc, iso, pos_list, rep_list, strip_stress in paralex_targets:
        if gc not in accepted_gcs or sid not in paralex_by_id:
            continue
        ext_dir = EXTERNAL_ROOT / "paralex" / sid / "extracted"
        for pos in pos_list:
            for rep in rep_list:
                def _loader(s=sid, ed=ext_dir, p=pos, r=rep, st=strip_stress):
                    return read_paralex_records(s, ed, p, r, strip_cyr_stress=st)
                by_slot[(gc, pos, rep)].append(("1_paralex", sid, iso, _loader))

    # CELEX English Verbs (`phon`)
    if "stan1293" in accepted_gcs:
        by_slot[("stan1293", "V", "phon")].append(
            ("1_celex", "celex_eng_v", "eng", lambda: (read_celex_english_v(), None))
        )

    # 2. Priority 2: Curated National Lexicons (`orth`)
    if "ital1282" in accepted_gcs:
        for pos in ("V", "N"):
            by_slot[("ital1282", pos, "orth")].append(
                ("2_national", "national_morphit", "ita", lambda p=pos: (read_morphit_records(p), None))
            )
    if "poli1260" in accepted_gcs:
        for pos in ("V", "N"):
            by_slot[("poli1260", pos, "orth")].append(
                ("2_national", "national_polimorf", "pol", lambda p=pos: (read_polimorf_records(p), None))
            )
    if "ukra1253" in accepted_gcs:
        for pos in ("V", "N"):
            by_slot[("ukra1253", pos, "orth")].append(
                ("2_national", "national_vesum", "ukr", lambda p=pos: (read_vesum_records(p), None))
            )

    # 3. Priority 3: Official `github.com/unimorph/*` (`orth`)
    for code, gc, iso in UNIMORPH_SPECS:
        if gc not in accepted_gcs:
            continue
        rdir = (PROJECT_ROOT / "external" / f"unimorph-{code}") if code in ("ita", "fin") else (EXTERNAL_ROOT / "unimorph" / code)
        if not rdir.exists():
            continue
        files = find_unimorph_data_files(rdir, code)
        if not files:
            continue
        for pos in ("V", "N"):
            if code == "fra":
                # Keep v3 (`fra`) and v4 (`fra.um4`) strictly non-overlapping per POS
                pos_files = [f for f in files if (f.name == "fra.um4") == (pos == "N")]
            else:
                pos_files = files
            if not pos_files:
                continue
            by_slot[(gc, pos, "orth")].append(
                ("3_unimorph", f"unimorph_{code}", iso, lambda fs=pos_files, p=pos, ic=iso: (read_unimorph_files_records(fs, p, ic), None))
            )

    # 4. Priority 3 supplement: SIGMORPHON 2020 Uralic (`mdf`, `myv`, `udm`)
    sig_root = EXTERNAL_ROOT / "sigmorphon2020" / "task0-data"
    if sig_root.exists():
        for code, gc, iso in (("mdf", "moks1248", "mdf"), ("myv", "erzy1239", "myv"), ("udm", "udmu1245", "udm")):
            if gc not in accepted_gcs:
                continue
            sfiles = sorted(
                p for p in sig_root.rglob(f"{code}.*")
                if p.suffix in (".trn", ".dev", ".tst", ".gold")
            )
            if not sfiles:
                continue
            for pos in ("V", "N"):
                by_slot[(gc, pos, "orth")].append(
                    ("3_sigmorphon", "sigmorphon2020_uralic", iso, lambda fs=sfiles, p=pos, ic=iso: (read_unimorph_files_records(fs, p, ic), None))
                )

    return by_slot


def build_all(only_glottocodes: Optional[Set[str]] = None) -> pd.DataFrame:
    for d in (LEXICONS_DIR, MALOUF_DIR, CELL_MAPS_DIR, SPLITS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not only_glottocodes:
        # Clean any stale CSVs before full deterministic rebuild
        for d in (LEXICONS_DIR, MALOUF_DIR, CELL_MAPS_DIR, SPLITS_DIR):
            for old in d.glob("*.csv"):
                old.unlink()
    else:
        for gc_del in only_glottocodes:
            for d in (LEXICONS_DIR, MALOUF_DIR, CELL_MAPS_DIR, SPLITS_DIR):
                for old in d.glob(f"{gc_del}_*.csv"):
                    old.unlink()

    accepted_gcs, gl_by_id, fam_names = load_glottolog_and_gelato()
    candidates = collect_candidates(accepted_gcs)

    summary_rows: List[dict] = []
    excluded_rows: List[dict] = []

    if only_glottocodes and SUMMARY_PATH.exists():
        prev_sum = pd.read_csv(SUMMARY_PATH, keep_default_na=False)
        for r in prev_sum.to_dict("records"):
            if r["glottocode"] not in only_glottocodes:
                r["has_uniform120"] = str(r["has_uniform120"]).lower() == "true"
                r["has_uniform480"] = str(r["has_uniform480"]).lower() == "true"
                r["has_malouf"] = str(r["has_malouf"]).lower() == "true"
                summary_rows.append(r)
    if only_glottocodes and EXCLUDED_LOG_PATH.exists():
        prev_ex = pd.read_csv(EXCLUDED_LOG_PATH, keep_default_na=False)
        for r in prev_ex.to_dict("records"):
            if r["glottocode"] not in only_glottocodes:
                excluded_rows.append(r)

    for (gc, pos, rep) in sorted(candidates.keys()):
        if only_glottocodes and gc not in only_glottocodes:
            continue
        unit_id = f"{gc}_{pos}_{rep}"
        cand_list = candidates[(gc, pos, rep)]
        accepted_unit = False

        for prio, sid, iso, loader_fn in cand_list:
            records, sounds_src = loader_fn()
            df_lex, cell_map_df, diag = process_candidate_unit(
                unit_id=unit_id,
                glottocode=gc,
                iso639_3=iso,
                pos=pos,
                representation=rep,
                source_id=sid,
                priority_tier=prio,
                records=records,
            )
            if df_lex is None or cell_map_df is None:
                excluded_rows.append(diag)
                continue

            # Accepted! Write lexicon, cell map, optional sounds.csv, splits, and optional Malouf table
            lex_path = LEXICONS_DIR / f"{unit_id}.csv"
            df_lex.to_csv(lex_path, index=False)

            cmap_path = CELL_MAPS_DIR / f"{unit_id}_cells.csv"
            cell_map_df.to_csv(cmap_path, index=False)

            sounds_rel = ""
            sounds_df: Optional[pd.DataFrame] = None
            if sounds_src is not None and sounds_src.exists():
                sdest = CELL_MAPS_DIR / f"{unit_id}_sounds.csv"
                shutil.copy2(sounds_src, sdest)
                sounds_rel = str(sdest.relative_to(PROJECT_ROOT))
                sounds_df = pd.read_csv(sdest, keep_default_na=False)

            split_paths = export_unit_splits(
                unit_id=unit_id,
                lex_df=df_lex,
                splits_dir=SPLITS_DIR,
                representation=rep,
            )

            gl_info = gl_by_id.get(gc, {})
            fam_id = gl_info.get("Family_ID", "") or "isolate"
            fam_name = gl_info.get("Name", gc) if fam_id == "isolate" else fam_names.get(fam_id, fam_id)
            lang_name = gl_info.get("Name", gc)

            has_malouf = False
            if fam_id in HARMONY_FAMILIES:
                mal_df, _ = apply_malouf_abstraction(df_lex, sounds_df=sounds_df)
                mal_df.to_csv(MALOUF_DIR / f"{unit_id}.csv", index=False)
                has_malouf = True

            n_groups = int(diag["n_leakage_groups"])
            tier_label = "tier2" if n_groups >= 600 else "tier3"

            summary_rows.append({
                "unit_id": unit_id,
                "glottocode": gc,
                "iso639_3": iso,
                "language_name": lang_name,
                "family_id": fam_id,
                "family_name": fam_name,
                "pos": pos,
                "representation": rep,
                "priority_tier": prio,
                "source_id": sid,
                "tier": tier_label,
                "n_lemmas": int(diag["n_lemmas"]),
                "n_leakage_groups": n_groups,
                "n_multi_lemma_groups": int(diag["n_multi_lemma_groups"]),
                "n_cells_raw": int(diag["n_cells_raw"]),
                "n_cells_synthetic": int(diag["n_cells_synthetic"]),
                "n_cells_distinct": int(diag["n_cells_distinct"]),
                "n_forms_total": int(diag["n_forms_total"]),
                "overabundance_rate": float(diag["overabundance_rate"]),
                "has_uniform120": split_paths["uniform120"] is not None,
                "has_uniform480": split_paths["uniform480"] is not None,
                "has_malouf": has_malouf,
                "sounds_file": sounds_rel,
            })
            print(
                f"  [accepted] {unit_id:20s} ({lang_name:22s} | {sid:18s}): "
                f"lemmas={diag['n_lemmas']:6d}, groups={n_groups:6d} ({tier_label}), "
                f"distinct_cells={diag['n_cells_distinct']:3d}"
            )
            accepted_unit = True
            break

        if not accepted_unit and cand_list:
            last_d = excluded_rows[-1]
            print(f"  [excluded] {unit_id:20s}: {last_d['reason']}")

    summary_rows.sort(key=lambda r: r["unit_id"])
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    if excluded_rows:
        excluded_rows.sort(key=lambda r: (r["unit_id"], r["source_id"]))
        pd.DataFrame(excluded_rows).to_csv(EXCLUDED_LOG_PATH, index=False)

    orth_df = summary_df[summary_df["representation"] == "orth"]
    phon_df = summary_df[summary_df["representation"] == "phon"]
    print(
        f"\n[build_lexicons] Complete! Exported {len(summary_df)} total units "
        f"({len(orth_df)} orth across {orth_df['glottocode'].nunique()} unique Glottocodes / "
        f"{orth_df['family_id'].nunique()} families; {len(phon_df)} paired phon units)."
    )
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track 2 morphological lexicons and CV splits.")
    parser.add_argument(
        "--glottocodes",
        type=str,
        default="",
        help="Optional comma-separated Glottocodes for targeted rebuild.",
    )
    args = parser.parse_args()
    only_gcs = {g.strip() for g in args.glottocodes.split(",") if g.strip()} or None
    build_all(only_glottocodes=only_gcs)


if __name__ == "__main__":
    main()
