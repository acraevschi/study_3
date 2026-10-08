"""Synthetic Italian-like verb paradigms for LDL runner tests (no real-data lemmas).

Three regular conjugation classes over the pilot NFIN + 8 panel cells. Output tables use
the CONTRACT column names needed by the LDL runner (forms-like training rows, §6 queries,
and a gold table in the ``write_gold_csv`` format).
"""

from __future__ import annotations

import random
from typing import Dict, List

import pandas as pd

CELLS = ["NFIN", "1;IND;PRS;SG", "3;IND;PRS;SG", "3;IND;PL;PRS", "1;IND;PFV;PST;SG",
         "3;IND;PFV;PST;SG", "3;IND;PFV;PL;PST", "3;COND;SG", "2;IMP;POS;SG"]
SUFFIXES = {
    "a": ["are", "o", "a", "ano", "ai", "ò", "arono", "erebbe", "a"],
    "e": ["ere", "o", "e", "ono", "ei", "é", "erono", "erebbe", "i"],
    "i": ["ire", "o", "e", "ono", "ii", "ì", "irono", "irebbe", "i"],
}
ONSETS = list("bcdfglmnprstv") + ["br", "tr", "sp", "st"]
VOWELS = list("aeiou")


def make_stems(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    out: List[str] = []
    while len(out) < n:
        k = rng.choice([2, 2, 3])
        stem = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) for _ in range(k))
        stem = stem + rng.choice(["t", "n", "r", "l", "st", "nd"])
        if stem not in out:
            out.append(stem)
    return out


def paradigm_rows(stem: str, cls: str, prefix: str = "toy:v") -> List[Dict]:
    lemma = stem + SUFFIXES[cls][0]
    rows = []
    for cell, suf in zip(CELLS, SUFFIXES[cls]):
        form = stem + suf
        rows.append({"lemma_id": f"{prefix}::{lemma}", "cell_norm": cell, "form": form,
                     "segments": " ".join(form), "variant_idx": 0, "is_missing": False})
    return rows


def lexicon(n: int, seed: int = 0) -> pd.DataFrame:
    stems = make_stems(n, seed)
    rng = random.Random(seed + 1)
    rows = []
    for s in stems:
        rows += paradigm_rows(s, rng.choice(["a", "a", "e", "i"]))
    return pd.DataFrame(rows)


def queries_for(forms: pd.DataFrame, lemma_ids: List[str]) -> pd.DataFrame:
    rows = []
    for lid in lemma_ids:
        src = forms[(forms.lemma_id == lid) & (forms.cell_norm == "NFIN")].iloc[0]
        for cell in CELLS[1:]:
            rows.append({"lemma_id": lid, "source_cell": "NFIN", "source_form": src.form,
                         "source_segments": src.segments, "target_cell": cell})
    return pd.DataFrame(rows)
