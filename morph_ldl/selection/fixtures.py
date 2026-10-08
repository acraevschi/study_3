"""TEMPORARY FIXTURES for selection tests and smoke runs only.

These helpers stand in for the data stage (``forms.csv``) and the main agent's split
manifests until those exist. They are *not* the pipeline's adapters or splitters:
``mgn_wide_to_forms`` is a minimal reader of the MGN wide CSV (``lexeme`` column +
UniMorph cell columns, variants ``;``-joined, missing ``NA``), groups are one lemma each,
and ``fixture_roles`` is a plain seeded shuffle. Never use them for substantive runs.
"""

from __future__ import annotations

import hashlib
import random
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from morph_ldl.schemas import FORMS_COLUMNS


def norm_cell(label: str, pos: str = "V") -> str:
    feats = sorted({f.strip().upper() for f in label.split(";") if f.strip()} - {pos.upper()})
    return ";".join(feats)


def segment(form: str) -> str:
    # one symbol per character (combining marks are attached to the previous symbol)
    syms: List[str] = []
    for ch in unicodedata.normalize("NFC", form):
        if unicodedata.combining(ch) and syms:
            syms[-1] += ch
        else:
            syms.append("_" if ch == " " else ch)
    return " ".join(syms)


def mgn_wide_to_forms(path: Path, unit_id: str, resource_id: str, iso: str, pos: str = "V",
                      cells: Sequence[str] | None = None, lemmas: Sequence[str] | None = None) -> pd.DataFrame:
    """FIXTURE: long forms table (FORMS_COLUMNS) from an MGN wide CSV."""
    path = Path(path)
    wide = pd.read_csv(path, dtype=str, keep_default_na=False)
    version = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if lemmas is not None:
        wide = wide[wide["lexeme"].isin(set(lemmas))]
    cell_cols = [c for c in wide.columns if c != "lexeme"]
    if cells is not None:
        cell_cols = [c for c in cell_cols if norm_cell(c, pos) in set(cells)]
    seen: Dict[str, int] = {}
    rows = []
    for ridx, r in wide.iterrows():
        label = r["lexeme"]
        seen[label] = seen.get(label, 0) + 1
        lid = f"{resource_id}::{label}" + (f"#{seen[label]}" if seen[label] > 1 else "")
        for c in cell_cols:
            raw = r[c]
            missing = raw.strip() in ("", "NA")
            variants = [""] if missing else [unicodedata.normalize("NFC", v).strip() for v in raw.split(";")]
            for vi, v in enumerate(variants):
                rows.append(dict(unit_id=unit_id, resource_id=resource_id, resource_version=version,
                                 variety_id=iso, iso639_3=iso, glottocode="", pos=pos, representation="orth",
                                 lemma_id=lid, lemma_label=label, group_id=f"{iso}.{pos}::{lid}",
                                 cell_orig=c, cell_norm=norm_cell(c, pos), form_orig=raw, variant_idx=vi,
                                 n_variants=len(variants), form=v, segments="" if missing else segment(v),
                                 is_missing=missing, source_file=str(path.name), source_row=int(ridx) + 2))
    return pd.DataFrame(rows, columns=FORMS_COLUMNS)


def toy_forms(n_lemmas: int = 80, seed: int = 0, unit: str = "toy.V.orth.test") -> pd.DataFrame:
    """FIXTURE: synthetic regular verbs with two conjugation classes (NFIN + 3 targets)."""
    rng = random.Random(seed)
    cons, vows = "ptkbdgmnlrs", "aeiou"
    cells = ["NFIN", "1;PRS;SG", "3;PRS;SG", "3;PL;PRS"]
    rows = []
    used = set()
    while len(used) < n_lemmas:
        stem = "".join(rng.choice(cons) + rng.choice(vows) for _ in range(rng.randint(1, 2))) + rng.choice(cons)
        if stem in used:
            continue
        used.add(stem)
    for i, stem in enumerate(sorted(used)):
        cls = i % 2
        forms = ([stem + "are", stem + "o", stem + "a", stem + "ano"] if cls == 0
                 else [stem + "ere", stem + "o", stem + "e", stem + "ono"])
        lid = f"toy:v::{forms[0]}"
        for c, f in zip(cells, forms):
            variants = [f, f + "i"] if (c == "1;PRS;SG" and i % 7 == 3) else [f]
            for vi, v in enumerate(variants):
                rows.append(dict(unit_id=unit, resource_id="toy:v", resource_version="t", variety_id="toy",
                                 iso639_3="toy", glottocode="", pos="V", representation="orth", lemma_id=lid,
                                 lemma_label=forms[0], group_id=f"toy.V::{lid}", cell_orig=c, cell_norm=c,
                                 form_orig=";".join(variants), variant_idx=vi, n_variants=len(variants), form=v,
                                 segments=" ".join(v), is_missing=False, source_file="toy", source_row=i))
    return pd.DataFrame(rows, columns=FORMS_COLUMNS)


def eligible(forms: pd.DataFrame, source_cell: str, panel: Sequence[str]) -> List[str]:
    need = [source_cell, *panel]
    ok = forms[(forms["variant_idx"] == 0) & (~forms["is_missing"].astype(bool)) & forms["cell_norm"].isin(need)
               & (forms["form"] != "")]
    have = ok.groupby("lemma_id")["cell_norm"].nunique()
    return sorted(have[have == len(need)].index)


def fixture_roles(lemma_ids: Sequence[str], sizes: Dict[str, int], seed: int) -> Dict[str, List[str]]:
    """FIXTURE: seeded shuffle dealt into roles in the order given by ``sizes``."""
    ids = sorted(lemma_ids)
    random.Random(seed).shuffle(ids)
    out, i = {}, 0
    for role, n in sizes.items():
        if i + n > len(ids):
            raise ValueError(f"not enough lemmas for role {role}")
        out[role] = ids[i: i + n]
        i += n
    return out
