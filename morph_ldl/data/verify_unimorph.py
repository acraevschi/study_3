"""Verify MGN wide files against a local UniMorph clone (provenance check, read-only).

For one language/POS: lemma overlap, cell inventory (UniMorph cells with >100 forms vs
MGN columns; cells absent from MGN are classified as 'dropped_<=100_forms' or
'dropped_as_duplicate_column' by re-applying MGN's process_subset rules), and exact
agreement of forms for shared (lemma, cell) pairs (variant sets compared as sets).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .adapters import read_mgn_wide


def _unimorph_pos(paths: List[Path], pos: str) -> pd.DataFrame:
    frames = [pd.read_csv(p, sep="\t", header=None, names=["lemma", "form", "cell"], dtype=str,
                          keep_default_na=False, quoting=3) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["cell"].str.startswith(pos + ";")].copy()
    df["cell"] = df["cell"].str[len(pos) + 1:]
    return df


def verify(mgn_file: Path, unimorph_files: List[Path], pos: str, commit: str) -> Dict[str, object]:
    um = _unimorph_pos(unimorph_files, pos)
    counts = um["cell"].value_counts()
    kept = set(counts[counts > 100].index)
    wide = (um[um["cell"].isin(kept)].groupby(["lemma", "cell"])["form"]
            .agg(lambda x: ";".join(sorted(set(x)))).unstack("cell"))
    mg = read_mgn_wide(mgn_file)
    mg_cells = set(mg["cell_orig"])
    dup_cols = set()
    seen = {}
    for c in wide.columns:  # duplicate columns across the whole table (order-insensitive check)
        key = tuple(wide[c].fillna("\x00").tolist())
        if key in seen:
            dup_cols.add(c)
        else:
            seen[key] = c
    absent = sorted(kept - mg_cells)
    mg_lem = set(mg["lemma_label"])
    um_lem = set(wide.index)
    pairs = mg[~mg["is_missing"]].drop_duplicates(["lemma_label", "cell_orig"])
    pairs = pairs[pairs["lemma_label"].isin(um_lem) & pairs["cell_orig"].isin(wide.columns)]
    um_vals = wide.stack()
    agree = 0
    for lab, cell, fo in pairs[["lemma_label", "cell_orig", "form_orig"]].itertuples(index=False):
        v = um_vals.get((lab, cell))
        if v is not None and set(v.split(";")) == set(fo.split(";")):
            agree += 1
    return {
        "mgn_file": mgn_file.name, "unimorph_files": ";".join(p.name for p in unimorph_files),
        "unimorph_commit": commit, "pos": pos,
        "mgn_lemmas": len(mg_lem), "unimorph_lemmas": len(um_lem),
        "lemmas_shared": len(mg_lem & um_lem), "lemmas_mgn_only": len(mg_lem - um_lem),
        "mgn_cells": len(mg_cells), "unimorph_cells_all": int(counts.size),
        "unimorph_cells_gt100": len(kept),
        "unimorph_cells_le100": " | ".join(sorted(set(counts.index) - kept)),
        "cells_absent_from_mgn": " | ".join(absent),
        "absent_cells_duplicate_in_current_unimorph": " | ".join(sorted(set(absent) & dup_cols)),
        "mgn_cells_not_in_unimorph": " | ".join(sorted(mg_cells - set(counts.index))),
        "shared_pairs_compared": len(pairs), "shared_pairs_identical": agree,
        "identical_rate": round(agree / max(1, len(pairs)), 4),
    }
