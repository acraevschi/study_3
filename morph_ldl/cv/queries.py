"""The single gate through which held-out lemmas reach prediction code.

``build_queries`` emits, for each held-out lemma and target cell, only the permitted
source anchor (variant 0 of the source cell, its segments and cell) and the requested
target cell. Gold targets are kept in a separate table for evaluation.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from morph_ldl.schemas import TEST_QUERY_COLUMNS


def build_queries(forms: pd.DataFrame, lemma_ids: Iterable[str], source_cell: str,
                  target_cells: Sequence[str]) -> pd.DataFrame:
    ids = list(dict.fromkeys(lemma_ids))
    if source_cell in target_cells:
        raise ValueError("the source cell cannot also be a target cell")
    src = forms[(forms["lemma_id"].isin(ids)) & (forms["cell_norm"] == source_cell)
                & (forms["variant_idx"] == 0) & (~forms["is_missing"].astype(bool))]
    src = src.drop_duplicates("lemma_id").set_index("lemma_id")
    missing = [i for i in ids if i not in src.index]
    if missing:
        raise KeyError(f"{len(missing)} lemmas lack a source anchor, e.g. {missing[:3]}")
    rows = [{"lemma_id": lid, "source_cell": source_cell, "source_form": src.at[lid, "form"],
             "source_segments": src.at[lid, "segments"], "target_cell": cell}
            for lid in ids for cell in target_cells]
    q = pd.DataFrame(rows, columns=TEST_QUERY_COLUMNS)
    assert list(q.columns) == TEST_QUERY_COLUMNS
    return q


def source_information_budget(queries: pd.DataFrame) -> dict:
    """Held-out information supplied at test time, reported apart from the training budget."""
    return {"n_test_lemmas": int(queries["lemma_id"].nunique()),
            "n_source_forms_supplied": int(queries["lemma_id"].nunique()),
            "n_target_queries": int(len(queries))}
