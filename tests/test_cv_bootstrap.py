"""Lemma-cluster bootstrap: clustering, pairing, determinism."""

import numpy as np
import pandas as pd
import pytest

from morph_ldl.cv import bootstrap


def _items(acc_by_lemma, cells=4, rep=0):
    rows = []
    for lid, acc in acc_by_lemma.items():
        for c in range(cells):
            ok = c < round(acc * cells)
            rows.append(dict(lemma_id=lid, target_cell=f"c{c}", repetition=rep, outer_fold=0,
                             correct=ok, edit_distance=0 if ok else 2, norm_edit_distance=0 if ok else .5))
    return pd.DataFrame(rows)


def test_point_estimates_and_determinism():
    it = _items({f"l{i}": (i % 5) / 4 for i in range(50)})
    a = bootstrap.cluster_bootstrap(it, 500, seed=1)
    b = bootstrap.cluster_bootstrap(it, 500, seed=1)
    pd.testing.assert_frame_equal(a, b)
    est = a.set_index("statistic")["estimate"]
    assert est["correct_micro"] == pytest.approx(it["correct"].mean())
    row = a.set_index("statistic").loc["correct_micro"]
    assert row.ci_low <= row.estimate <= row.ci_high


def test_cluster_not_item_resampling():
    # Perfectly within-lemma-correlated outcomes: cluster CI must be wider than an
    # item-level CI computed as if the 8 cells were independent.
    rng = np.random.default_rng(0)
    it = _items({f"l{i}": float(rng.integers(0, 2)) for i in range(60)}, cells=8)
    ci = bootstrap.cluster_bootstrap(it, 2000, seed=3).set_index("statistic").loc["correct_micro"]
    p = it["correct"].mean()
    naive_half = 1.96 * np.sqrt(p * (1 - p) / len(it))
    assert (ci.ci_high - ci.ci_low) / 2 > 1.5 * naive_half


def test_paired_difference_requires_identical_items():
    a = _items({f"l{i}": 1.0 for i in range(30)})
    b = _items({f"l{i}": 0.5 for i in range(30)})
    d = bootstrap.paired_bootstrap(a, b, 500, seed=2).set_index("statistic").loc["correct_micro"]
    assert d.difference == pytest.approx(0.5) and d.ci_low > 0
    with pytest.raises(ValueError):
        bootstrap.paired_bootstrap(a, b.iloc[:-1], 100, seed=2)
