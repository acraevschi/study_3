"""Synthetic contract-conformant forms tables for cross-validation tests."""

from __future__ import annotations

import pandas as pd

from morph_ldl.schemas import FORMS_COLUMNS

PANEL = ["1;IND;PRS;SG", "3;IND;PRS;SG", "3;IND;PL;PRS"]
SOURCE = "NFIN"


def make_forms(n_lemmas: int = 300, unit: str = "xxx.V.orth.test", shared_every: int = 25,
               missing_every: int = 40) -> pd.DataFrame:
    """Regular toy verbs; every ``shared_every``-th lemma shares a group with its
    predecessor; every ``missing_every``-th lemma lacks one panel cell."""
    rows = []
    for i in range(n_lemmas):
        stem = f"v{i:04d}"
        lid = f"test:xxx-v::{stem}are"
        gid = f"xxx.V::g{(i - 1) if (shared_every and i % shared_every == 0 and i > 0) else i:05d}"
        cells = {SOURCE: stem + "are", PANEL[0]: stem + "o", PANEL[1]: stem + "a", PANEL[2]: stem + "ano"}
        for j, (cell, form) in enumerate(cells.items()):
            missing = missing_every and i % missing_every == 7 and cell == PANEL[2]
            variants = [form, form + "x"] if (i % 10 == 3 and cell == PANEL[0]) else [form]
            for v, f in enumerate(variants):
                rows.append(dict(unit_id=unit, resource_id="test:xxx-v", resource_version="t",
                                 variety_id="xxx", iso639_3="xxx", glottocode="xxxx1234", pos="V",
                                 representation="orth", lemma_id=lid, lemma_label=stem + "are",
                                 group_id=gid, cell_orig=cell, cell_norm=cell,
                                 form_orig=";".join(variants), variant_idx=v,
                                 n_variants=len(variants), form="" if missing else f,
                                 segments="" if missing else " ".join(f), is_missing=bool(missing),
                                 source_file="fixture", source_row=i))
    return pd.DataFrame(rows)[FORMS_COLUMNS]


def small_cfg(**cv) -> dict:
    base = {"inventory_size": 240, "n_folds": 3, "dev_size": 20, "seed_size": 10,
            "pool_cap": 100, "pool_cap_sensitivity": [50], "repetitions": [0]}
    base.update(cv)
    return {"experiment": {"id": "t", "master_seed": 7}, "cv": base,
            "selection": {"budgets": [40, 60], "batch_size": 10}}
