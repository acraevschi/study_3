"""Deterministic SHA-256 cross-validation split generator (`splits.py`).

Generates zero-leakage fixed-budget paradigm splits across eligible units:
  - `{unit_id}_uniform120.csv` (all units with `n_leakage_groups >= 150`):
      `R = 5` random lexical draws × `K = 5` disjoint test folds of `150` distinct
      `leakage_group_id`s (1 canonical simplex lemma per group):
        * `role` in `{"train", "test"}` (`120` train / `30` test)
        * `inner_role` in `{"subtrain", "dev", "test"}` (`96` subtrain / `24` dev / `30` test)
  - `{unit_id}_uniform480.csv` (`tier2` units with `n_leakage_groups >= 600`):
      `R = 3` random lexical draws × `K = 5` disjoint test folds of `600` distinct
      `leakage_group_id`s (1 canonical simplex lemma per group):
        * `role` in `{"train", "test"}` (`480` train / `120` test)
        * `inner_role` in `{"subtrain", "dev", "test"}` (`384` subtrain / `96` dev / `120` test)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .grouping import to_graphemes

SPLIT_COLUMNS = [
    "unit_id",
    "regime",
    "draw_idx",
    "fold_idx",
    "lemma_id",
    "leakage_group_id",
    "role",
    "inner_role",
]


def _rng_from_tag(tag: str) -> np.random.Generator:
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(seed_int)


def select_canonical_lemma_per_group(
    lex_df: pd.DataFrame,
    representation: str = "orth",
) -> pd.DataFrame:
    """Select exactly 1 canonical (simplex/shortest-label) `lemma_id` per `leakage_group_id`.

    Ordering within each `leakage_group_id` prefers:
      1. Single-word citation label over multiword label,
      2. Shortest grapheme length of `lemma_label` (selecting base simplex verb/noun such as
         `porre` over `comporre`, `stand` over `understand`, `abbandonare` over `abbandonarsi`),
      3. Lexicographically smallest `lemma_id` for 100% determinism.
    """
    lem = (
        lex_df[["lemma_id", "lemma_label", "leakage_group_id"]]
        .drop_duplicates("lemma_id")
        .copy()
    )
    lem["_has_space"] = lem["lemma_label"].astype(str).str.strip().str.contains(r"\s", regex=True).astype(int)
    lem["_glen"] = [len(to_graphemes(str(x), representation)) for x in lem["lemma_label"]]
    lem = lem.sort_values(
        ["leakage_group_id", "_has_space", "_glen", "lemma_id"],
        kind="mergesort",
    )
    canon = lem.drop_duplicates("leakage_group_id", keep="first")[
        ["leakage_group_id", "lemma_id", "lemma_label"]
    ].reset_index(drop=True)
    return canon


def build_regime_splits(
    unit_id: str,
    canon_groups: pd.DataFrame,
    regime: str,
    n_sample_groups: int,
    n_draws: int,
    n_folds: int = 5,
    dev_fraction_of_train: float = 0.20,
) -> pd.DataFrame:
    """Create deterministic `R x K` split manifest DataFrame for one `(unit_id, regime)`."""
    canon_sorted = canon_groups.sort_values("leakage_group_id", kind="mergesort").reset_index(drop=True)
    n_avail = len(canon_sorted)
    if n_avail < n_sample_groups:
        raise ValueError(
            f"{unit_id} ({regime}): requires >= {n_sample_groups} leakage groups, got {n_avail}"
        )
    if n_sample_groups % n_folds != 0:
        raise ValueError(f"n_sample_groups={n_sample_groups} must be divisible by n_folds={n_folds}")

    fold_size = n_sample_groups // n_folds
    train_size = n_sample_groups - fold_size
    dev_size = int(round(train_size * dev_fraction_of_train))
    subtrain_size = train_size - dev_size

    gids = canon_sorted["leakage_group_id"].to_numpy()
    lids = canon_sorted["lemma_id"].to_numpy()

    rows: List[dict] = []
    for r in range(n_draws):
        rng_draw = _rng_from_tag(f"study3_track2_split_v1::{unit_id}::{regime}::draw={r}")
        chosen_idx = rng_draw.choice(n_avail, size=n_sample_groups, replace=False)
        # Shuffle chosen indices into K equal-sized test partitions
        perm_idx = rng_draw.permutation(chosen_idx)
        partitions = [perm_idx[k * fold_size : (k + 1) * fold_size] for k in range(n_folds)]

        for k in range(n_folds):
            test_idx = partitions[k]
            train_pool = np.concatenate([partitions[j] for j in range(n_folds) if j != k])
            rng_fold = _rng_from_tag(
                f"study3_track2_split_v1::{unit_id}::{regime}::draw={r}::fold={k}"
            )
            train_perm = rng_fold.permutation(train_pool)
            subtrain_idx = train_perm[:subtrain_size]
            dev_idx = train_perm[subtrain_size : subtrain_size + dev_size]

            for idx in sorted(subtrain_idx.tolist(), key=lambda i: gids[i]):
                rows.append({
                    "unit_id": unit_id,
                    "regime": regime,
                    "draw_idx": r,
                    "fold_idx": k,
                    "lemma_id": lids[idx],
                    "leakage_group_id": gids[idx],
                    "role": "train",
                    "inner_role": "subtrain",
                })
            for idx in sorted(dev_idx.tolist(), key=lambda i: gids[i]):
                rows.append({
                    "unit_id": unit_id,
                    "regime": regime,
                    "draw_idx": r,
                    "fold_idx": k,
                    "lemma_id": lids[idx],
                    "leakage_group_id": gids[idx],
                    "role": "train",
                    "inner_role": "dev",
                })
            for idx in sorted(test_idx.tolist(), key=lambda i: gids[i]):
                rows.append({
                    "unit_id": unit_id,
                    "regime": regime,
                    "draw_idx": r,
                    "fold_idx": k,
                    "lemma_id": lids[idx],
                    "leakage_group_id": gids[idx],
                    "role": "test",
                    "inner_role": "test",
                })

    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def export_unit_splits(
    unit_id: str,
    lex_df: pd.DataFrame,
    splits_dir: Path,
    representation: str = "orth",
) -> Dict[str, Optional[Path]]:
    """Export `uniform120` (if `>=150` groups) and `uniform480` (if `>=600` groups) CSVs."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    canon = select_canonical_lemma_per_group(lex_df, representation=representation)
    n_groups = len(canon)
    out: Dict[str, Optional[Path]] = {"uniform120": None, "uniform480": None}

    if n_groups >= 150:
        df120 = build_regime_splits(
            unit_id=unit_id,
            canon_groups=canon,
            regime="uniform120",
            n_sample_groups=150,
            n_draws=5,
            n_folds=5,
        )
        p120 = splits_dir / f"{unit_id}_uniform120.csv"
        df120.to_csv(p120, index=False)
        out["uniform120"] = p120

    if n_groups >= 600:
        df480 = build_regime_splits(
            unit_id=unit_id,
            canon_groups=canon,
            regime="uniform480",
            n_sample_groups=600,
            n_draws=3,
            n_folds=5,
        )
        p480 = splits_dir / f"{unit_id}_uniform480.csv"
        df480.to_csv(p480, index=False)
        out["uniform480"] = p480

    return out
