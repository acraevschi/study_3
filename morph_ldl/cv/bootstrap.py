"""Lemma-cluster bootstrap intervals from cached out-of-fold predictions.

All targets of a lemma (and, with several repetitions, all its predictions across
repetitions) belong to one cluster. Clusters are leakage groups (``group_id``; in the
pilot almost always a single lemma), so lemmas sharing a group are resampled together.
The bootstrap resamples clusters with replacement and recomputes the statistic from the
resampled clusters' items. "Macro" averages lemma means within clusters' items. Paired active-minus-random
differences resample the same lemma draw for both policies.

These intervals are conditional on the fitted fold models (selection and LDL are not
re-run inside the bootstrap). Variability across split / acquisition / semantic seeds
is reported separately (see ``fold_variability``), never as a standard error.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

METRICS = ("correct", "edit_distance", "norm_edit_distance")


def _cluster_col(items: pd.DataFrame) -> str:
    return "group_id" if "group_id" in items and (items["group_id"].astype(str) != "").all() else "lemma_id"


def _lemma_sums(items: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster sums of each metric, of lemma means (for macro), and item/lemma counts."""
    col = _cluster_col(items)
    lem = items.groupby([col, "lemma_id"])[list(METRICS)].mean()
    g = items.groupby(col)
    sums = g[list(METRICS)].sum()
    sums["n"] = g.size()
    lm = lem.groupby(level=0).sum()
    for m in METRICS:
        sums[f"{m}__lemma_mean_sum"] = lm[m]
    sums["n_lemmas"] = lem.groupby(level=0).size()
    return sums.sort_index()


def _stats_from_sums(sums: pd.DataFrame, w: np.ndarray) -> Dict[str, np.ndarray]:
    """Statistics for bootstrap weight vectors ``w`` (n_boot x n_clusters counts).

    micro = sum over items / item count; macro = mean over lemmas of lemma means.
    """
    n = sums["n"].to_numpy(float)
    nl = sums["n_lemmas"].to_numpy(float)
    out = {}
    for m in METRICS:
        out[f"{m}_micro"] = (w @ sums[m].to_numpy(float)) / (w @ n)
        out[f"{m}_macro"] = (w @ sums[f"{m}__lemma_mean_sum"].to_numpy(float)) / (w @ nl)
    return out


def cluster_bootstrap(items: pd.DataFrame, n_boot: int, seed: int, level: float = 0.95) -> pd.DataFrame:
    """Point estimates and percentile intervals for one cell of the design."""
    sums = _lemma_sums(items)
    n = sums["n"].to_numpy(float)
    L = len(sums)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, L, L), minlength=L) for _ in range(n_boot)]).astype(float)
    point = _stats_from_sums(sums, np.ones((1, L)))
    boot = _stats_from_sums(sums, w)
    a = (1 - level) / 2
    rows = []
    for k in point:
        rows.append({"statistic": k, "estimate": float(point[k][0]),
                     "ci_low": float(np.quantile(boot[k], a)), "ci_high": float(np.quantile(boot[k], 1 - a)),
                     "n_clusters": L, "n_lemmas": int(sums["n_lemmas"].sum()), "n_items": int(n.sum()),
                     "n_boot": n_boot, "boot_seed": seed})
    return pd.DataFrame(rows)


def paired_bootstrap(items_a: pd.DataFrame, items_b: pd.DataFrame, n_boot: int, seed: int,
                     level: float = 0.95) -> pd.DataFrame:
    """Difference a - b on identical (lemma, cell) items, resampling lemmas jointly."""
    key = ["lemma_id", "target_cell"] + (["repetition"] if "repetition" in items_a else [])
    ka = items_a.set_index(key).index
    kb = items_b.set_index(key).index
    if not ka.sort_values().equals(kb.sort_values()):
        raise ValueError("paired comparison requires identical evaluation items")
    sa, sb = _lemma_sums(items_a), _lemma_sums(items_b)
    assert sa.index.equals(sb.index) and np.array_equal(sa["n"], sb["n"])
    n = sa["n"].to_numpy(float)
    L = len(sa)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, L, L), minlength=L) for _ in range(n_boot)]).astype(float)
    pa, pb = _stats_from_sums(sa, np.ones((1, L))), _stats_from_sums(sb, np.ones((1, L)))
    ba, bb = _stats_from_sums(sa, w), _stats_from_sums(sb, w)
    a = (1 - level) / 2
    rows = []
    for k in pa:
        d = ba[k] - bb[k]
        rows.append({"statistic": k, "estimate_a": float(pa[k][0]), "estimate_b": float(pb[k][0]),
                     "difference": float(pa[k][0] - pb[k][0]),
                     "ci_low": float(np.quantile(d, a)), "ci_high": float(np.quantile(d, 1 - a)),
                     "prop_boot_gt0": float((d > 0).mean()),
                     "n_clusters": L, "n_lemmas": int(sa["n_lemmas"].sum()), "n_items": int(n.sum()),
                     "n_boot": n_boot, "boot_seed": seed,
                     "interval_note": "paired cluster bootstrap over cached out-of-fold predictions; "
                                      "conditional on the fitted fold models"})
    return pd.DataFrame(rows)


def fold_variability(items: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Per fold/repetition point estimates and their spread (descriptive only)."""
    per = (items.groupby(list(by) + ["repetition", "outer_fold"])
           .agg(acc_micro=("correct", "mean"), edit_distance=("edit_distance", "mean"),
                n_items=("correct", "size")).reset_index())
    spread = (per.groupby(list(by))
              .agg(n_fold_runs=("acc_micro", "size"), acc_fold_min=("acc_micro", "min"),
                   acc_fold_max=("acc_micro", "max"), acc_fold_sd=("acc_micro", "std"),
                   ed_fold_sd=("edit_distance", "std")).reset_index())
    return per, spread
