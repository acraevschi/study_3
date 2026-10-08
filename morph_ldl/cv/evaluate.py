"""Item-level scoring of out-of-fold predictions (docs/CONTRACT.md §7).

Gold forms are read here and nowhere in prediction code. A prediction is correct when
its segment string equals any documented variant of the gold cell. Edit distance is the
Levenshtein distance over segments (symbols, not bytes) to the closest variant.
Missing predictions and decoder failures are scored as incorrect with edit distance
equal to the length of the gold variant 0 and are also counted separately.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from morph_ldl.schemas import ITEM_COLUMNS

VARIANT_SEP = " || "


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Unit-cost Levenshtein distance between two symbol sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def symbols(segments: str) -> List[str]:
    return [s for s in str(segments).split(" ") if s != ""]


def gold_table(forms: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
    """{(lemma_id, cell_norm): [segments of variant 0, variant 1, ...]} for non-missing rows."""
    ok = forms[~forms["is_missing"].astype(bool)].sort_values(["lemma_id", "cell_norm", "variant_idx"])
    gold: Dict[Tuple[str, str], List[str]] = {}
    for (lid, cell), sub in ok.groupby(["lemma_id", "cell_norm"], sort=False):
        gold[(lid, cell)] = sub["segments"].astype(str).tolist()
    return gold


def score_items(pred: pd.DataFrame, gold: Dict[Tuple[str, str], List[str]],
                meta: Dict[str, object], group_of: Dict[str, str]) -> pd.DataFrame:
    """Score a predictions table with columns lemma_id, target_cell, prediction_segments
    (space-separated; empty = missing) and status. ``meta`` fills unit/fold/policy/budget/model."""
    rows = []
    for r in pred.itertuples(index=False):
        key = (r.lemma_id, r.target_cell)
        if key not in gold:
            raise KeyError(f"no gold form for {key}")
        variants = gold[key]
        status = str(getattr(r, "status", "ok") or "ok")
        pseg = getattr(r, "prediction_segments", "")
        pseg = "" if (pseg is None or (isinstance(pseg, float) and np.isnan(pseg))) else str(pseg)
        if status == "ok" and pseg.strip() == "":
            status = "missing"
        if status != "ok":
            correct = False
            ed = len(symbols(variants[0]))
            denom = max(1, ed)
            prediction = ""
        else:
            p = symbols(pseg)
            dists = [levenshtein(p, symbols(v)) for v in variants]
            best = int(np.argmin(dists))
            ed = dists[best]
            correct = ed == 0
            denom = max(1, len(symbols(variants[best])))
            prediction = pseg
        rows.append({
            **meta,
            "lemma_id": r.lemma_id,
            "group_id": group_of.get(r.lemma_id, ""),
            "target_cell": r.target_cell,
            "gold_variants": VARIANT_SEP.join(variants),
            "prediction": prediction,
            "status": status,
            "correct": bool(correct),
            "edit_distance": int(ed),
            "norm_edit_distance": float(ed) / denom,
        })
    out = pd.DataFrame(rows)
    return out[ITEM_COLUMNS] if len(out) else pd.DataFrame(columns=ITEM_COLUMNS)


def summarize(items: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Point summaries with denominators. Macro accuracy averages lemma means."""
    def agg(df: pd.DataFrame) -> pd.Series:
        lem = df.groupby("lemma_id").agg(acc=("correct", "mean"), ed=("edit_distance", "mean"),
                                        ned=("norm_edit_distance", "mean"))
        return pd.Series({
            "n_lemmas": df["lemma_id"].nunique(),
            "n_items": len(df),
            "acc_micro": df["correct"].mean(),
            "acc_macro_lemma": lem["acc"].mean(),
            "edit_distance_mean": df["edit_distance"].mean(),
            "norm_edit_distance_mean": df["norm_edit_distance"].mean(),
            "n_missing": int((df["status"] == "missing").sum()),
            "n_failed": int((~df["status"].isin(["ok", "missing"])).sum()),
        })
    return items.groupby(list(by), dropna=False).apply(agg, include_groups=False).reset_index()
