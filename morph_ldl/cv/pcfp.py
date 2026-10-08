"""Paradigm cell filling (PCFP): cell inventory, eligibility, exposure draw, queries.

The main agent owns this module (docs/PROTOCOL.md §2, docs/CONTRACT.md §4).

* **Cell inventory.** A cell is eligible when at most ``max_multiword_share`` of its
  variant-0, non-missing forms (over lemmas that are not declared derived paradigms) are
  multiword. Any remaining multiword form counts as unavailable.
* **Eligible lemma.** Not a derived paradigm and, with ``require_complete_paradigm``, every
  eligible cell non-missing and single-word at variant 0. Every eligible lemma therefore
  has the same ``n_cells``.
* **Exposure draw.** Once per lemma, from ``seeds.derive(master, "exposure", unit_id,
  lemma_id)``: k ~ Uniform{1..min(max_shown, n_cells - 1)}, then k shown cells uniformly
  without replacement from the sorted eligible cells. The other eligible cells are hidden.
  The draw depends on nothing else (not on policy, fold, budget, pool cap or repetition).
* **Test items.** The hidden cells of a verb, minus the citation cell
  (``citation_cell_rule: exclude_from_test``). The citation cell may be shown and then
  counts in k.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from morph_ldl import seeds as seedlib

SEP = "|"                       # separator of cell lists in manifests (cell_norm uses ';')
WORD_SEP = "_"
EXPOSURE_COLUMNS = ["unit_id", "lemma_id", "group_id", "n_cells", "k_max", "k", "shown_cells",
                    "hidden_cells", "citation_shown", "n_test_cells", "exposure_seed"]
QUERY_COLUMNS = ["lemma_id", "target_cell"]


class ExposureError(RuntimeError):
    """The exposure manifest is inconsistent with its declared rule."""


def _v0(forms: pd.DataFrame) -> pd.DataFrame:
    return forms[(forms["variant_idx"] == 0) & (~forms["is_missing"].astype(bool))
                 & (forms["form"].astype(str) != "")]


def is_multiword_segments(segments: pd.Series) -> pd.Series:
    return segments.astype(str).str.split(" ").map(lambda s: WORD_SEP in s)


# ----------------------------------------------------------------------------- cells

def cell_inventory_table(forms: pd.DataFrame, exclude_ids: Iterable[str] = (),
                         max_multiword_share: float = 0.5) -> pd.DataFrame:
    """Per cell: variant-0 form count, multiword count/share and the eligibility decision."""
    v0 = _v0(forms[~forms["lemma_id"].isin(set(exclude_ids))])
    v0 = v0[v0["cell_norm"].astype(str) != ""]
    t = (v0.assign(mw=is_multiword_segments(v0["segments"]))
         .groupby("cell_norm").agg(n_forms_v0=("mw", "size"), n_multiword=("mw", "sum")).reset_index())
    t["multiword_share"] = t["n_multiword"] / t["n_forms_v0"]
    t["eligible"] = t["multiword_share"] <= max_multiword_share
    return t.sort_values("cell_norm").reset_index(drop=True)


def eligible_cells(forms: pd.DataFrame, exclude_ids: Iterable[str] = (),
                   max_multiword_share: float = 0.5) -> List[str]:
    t = cell_inventory_table(forms, exclude_ids, max_multiword_share)
    return sorted(t.loc[t["eligible"], "cell_norm"])


def eligible_lemmas(forms: pd.DataFrame, cells: Sequence[str], citation_cell: str,
                    exclude_ids: Iterable[str] = (), require_complete: bool = True) -> pd.DataFrame:
    """One row per eligible lemma (lemma_id, group_id, n_cells_available).

    A cell is available for a lemma when its variant-0 form exists, is non-missing and is
    single-word. With ``require_complete`` every declared cell must be available; otherwise
    the citation cell plus at least one other cell are required."""
    cells = sorted(cells)
    if citation_cell not in cells:
        raise ValueError("the citation cell must be an eligible cell")
    v0 = _v0(forms[forms["cell_norm"].isin(cells)])
    v0 = v0[~is_multiword_segments(v0["segments"])]
    have = v0.groupby("lemma_id")["cell_norm"].nunique()
    has_cit = set(v0.loc[v0["cell_norm"] == citation_cell, "lemma_id"])
    if require_complete:
        keep = set(have[have == len(cells)].index)
    else:
        keep = set(have[have >= 2].index) & has_cit
    keep -= set(exclude_ids)
    lem = (forms.loc[forms["lemma_id"].isin(keep), ["lemma_id", "group_id"]]
           .drop_duplicates().sort_values("lemma_id").reset_index(drop=True))
    if lem["lemma_id"].duplicated().any():
        raise ValueError("lemmas with several group_ids")
    lem["n_cells_available"] = lem["lemma_id"].map(have).astype(int)
    return lem


# ----------------------------------------------------------------------------- exposure

@dataclass(frozen=True)
class Exposure:
    lemma_id: str
    k: int
    shown: Tuple[str, ...]
    hidden: Tuple[str, ...]

    def test_cells(self, citation_cell: str) -> Tuple[str, ...]:
        return tuple(c for c in self.hidden if c != citation_cell)


def k_max(n_cells: int, max_shown: int) -> int:
    return min(int(max_shown), int(n_cells) - 1)


def draw_exposure(master_seed: int, unit_id: str, lemma_id: str, cells: Sequence[str],
                  max_shown: int) -> Exposure:
    """The per-verb exposure draw. Depends only on (master seed, unit, lemma, cells, cap)."""
    cells = sorted(cells)
    km = k_max(len(cells), max_shown)
    if km < 1:
        raise ExposureError(f"{lemma_id}: {len(cells)} cells leave no hidden cell")
    rng = np.random.default_rng(seedlib.derive(master_seed, "exposure", unit_id, lemma_id))
    k = int(rng.integers(1, km + 1))
    idx = rng.choice(len(cells), size=k, replace=False)
    shown = tuple(sorted(cells[i] for i in idx))
    hidden = tuple(c for c in cells if c not in set(shown))
    return Exposure(lemma_id, k, shown, hidden)


def build_exposure_manifest(lemmas: pd.DataFrame, cells: Sequence[str], citation_cell: str,
                            master_seed: int, unit_id: str, max_shown: int) -> pd.DataFrame:
    rows = []
    for r in lemmas.sort_values("lemma_id").itertuples(index=False):
        e = draw_exposure(master_seed, unit_id, r.lemma_id, cells, max_shown)
        rows.append({"unit_id": unit_id, "lemma_id": r.lemma_id, "group_id": r.group_id,
                     "n_cells": len(cells), "k_max": k_max(len(cells), max_shown), "k": e.k,
                     "shown_cells": SEP.join(e.shown), "hidden_cells": SEP.join(e.hidden),
                     "citation_shown": citation_cell in e.shown,
                     "n_test_cells": len(e.test_cells(citation_cell)),
                     "exposure_seed": seedlib.derive(master_seed, "exposure", unit_id, r.lemma_id)})
    return pd.DataFrame(rows, columns=EXPOSURE_COLUMNS)


def validate_exposure_manifest(man: pd.DataFrame, cells: Sequence[str], citation_cell: str,
                               master_seed: int, unit_id: str, max_shown: int) -> Dict[str, object]:
    """Re-derive every draw and check the partition; returns summary counts."""
    cells = sorted(cells)
    if man["lemma_id"].duplicated().any():
        raise ExposureError("lemma listed twice in the exposure manifest")
    km = k_max(len(cells), max_shown)
    for r in man.itertuples(index=False):
        shown, hidden = tuple(str(r.shown_cells).split(SEP)), tuple(str(r.hidden_cells).split(SEP))
        if set(shown) & set(hidden) or sorted(shown + hidden) != cells:
            raise ExposureError(f"{r.lemma_id}: shown/hidden do not partition the eligible cells")
        if not (1 <= int(r.k) <= km) or len(shown) != int(r.k):
            raise ExposureError(f"{r.lemma_id}: k={r.k} outside 1..{km} or != number shown")
        e = draw_exposure(master_seed, unit_id, r.lemma_id, cells, max_shown)
        if e.shown != shown or e.k != int(r.k):
            raise ExposureError(f"{r.lemma_id}: exposure draw does not re-derive from its seed")
        if int(r.n_test_cells) != len(e.test_cells(citation_cell)):
            raise ExposureError(f"{r.lemma_id}: n_test_cells inconsistent")
    return {"n_lemmas": len(man), "n_cells": len(cells), "k_max": km,
            "k_counts": {int(k): int(v) for k, v in man["k"].value_counts().sort_index().items()},
            "k_mean": float(man["k"].mean()),
            "citation_shown_share": float(man["citation_shown"].astype(str).str.lower().eq("true").mean()),
            "n_test_cells_mean": float(man["n_test_cells"].mean())}


def load_exposure(path: Path | str) -> Dict[str, Exposure]:
    man = pd.read_csv(path, dtype={"lemma_id": str, "shown_cells": str, "hidden_cells": str})
    return exposure_from_frame(man)


def exposure_from_frame(man: pd.DataFrame) -> Dict[str, Exposure]:
    return {r.lemma_id: Exposure(r.lemma_id, int(r.k), tuple(str(r.shown_cells).split(SEP)),
                                 tuple(str(r.hidden_cells).split(SEP)))
            for r in man.itertuples(index=False)}


# ----------------------------------------------------------------------------- rows / queries

def shown_rows(forms: pd.DataFrame, exposure: Mapping[str, Exposure], lemma_ids: Iterable[str]) -> pd.DataFrame:
    """Forms rows (all variants) of the shown cells of the given lemmas. Hidden cells of
    these lemmas never appear in the result."""
    ids = list(dict.fromkeys(lemma_ids))
    missing = [i for i in ids if i not in exposure]
    if missing:
        raise ExposureError(f"{len(missing)} lemmas lack an exposure draw, e.g. {missing[:3]}")
    keep = {(l, c) for l in ids for c in exposure[l].shown}
    sub = forms[forms["lemma_id"].isin(set(ids))]
    mask = [(l, c) in keep for l, c in zip(sub["lemma_id"], sub["cell_norm"])]
    out = sub[mask]
    got = set(zip(out.loc[out["variant_idx"] == 0, "lemma_id"], out.loc[out["variant_idx"] == 0, "cell_norm"]))
    if got != keep:
        raise ExposureError(f"{len(keep - got)} shown cells lack a variant-0 form")
    return out


def hidden_queries(exposure: Mapping[str, Exposure], lemma_ids: Iterable[str], citation_cell: str) -> pd.DataFrame:
    """Gold-free test queries: the hidden cells of each lemma except the citation cell."""
    rows = [{"lemma_id": l, "target_cell": c} for l in dict.fromkeys(lemma_ids)
            for c in exposure[l].test_cells(citation_cell)]
    return pd.DataFrame(rows, columns=QUERY_COLUMNS)


def citation_segments(label: str, representation: str) -> str:
    """Segments of a lemma label (the citation form the selector may know)."""
    from morph_ldl.data.segments import segment
    if representation != "orth":
        raise ValueError("the citation-row selector needs labels in the unit's representation (orth only)")
    return segment(label, representation)


def lemma_labels(forms: pd.DataFrame, lemma_ids: Iterable[str]) -> Dict[str, str]:
    lab = forms.loc[forms["lemma_id"].isin(set(lemma_ids)), ["lemma_id", "lemma_label"]].drop_duplicates("lemma_id")
    return dict(zip(lab["lemma_id"], lab["lemma_label"].astype(str)))


# ----------------------------------------------------------------------------- class proxies (reporting only)

def inflection_class(unit_id: str, label: str) -> str:
    """Coarse inflection-class proxy from the citation form (reporting only).

    Italian: conjugation by infinitive ending (-are, -ere, -ire, -rre). Finnish: an
    infinitive-ending proxy for the Kotus types (no class labels in MGN): 'VV' (type 1,
    e.g. sanoa), 'da' (type 2, saada), 'CCa' (type 3: -lla/-nna/-rra/-sta), 'Vta' (types 4-6:
    haluta, tarvita, vanheta), other."""
    s = str(label).lower()
    if unit_id.startswith("ita."):
        for e in ("are", "ere", "ire", "rre"):
            if s.endswith(e):
                return "-" + e
        return "other"
    if unit_id.startswith("fin."):
        if len(s) < 3:
            return "other"
        stem, last = s[:-1], s[-1]
        if last not in "aä":
            return "other"
        v = set("aeiouyäö")
        if stem.endswith(("d",)):
            return "da"
        if stem.endswith(("ll", "nn", "rr", "st")):
            return "CCa"
        if stem.endswith("t") and len(stem) >= 2 and stem[-2] in v:
            return "Vta"
        if stem[-1] in v:
            return "VV"
        return "other"
    return "na"
