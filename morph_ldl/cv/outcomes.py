"""Analysis-ready outcome table, paired differences and the population linkage table.

One outcome row = one morphology observation design cell: (unit/variety, POS, task,
resource, representation, sampling policy, pool cap, training budget, model). GeLaTo
populations are attached only through the separate linkage table, so several
populations never duplicate a morphology observation and no aggregation rule is fixed.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from morph_ldl import seeds as seedlib
from morph_ldl.cv import bootstrap

DESIGN = ["unit_id", "item_set", "policy", "pool_cap", "budget", "model"]
ITEM_SET_NOTE = {"core": "primary: hidden cells of the fixed core verbs (identical items for every policy)",
                 "selected": "secondary, policy-dependent: hidden cells of the selected verbs (seed + acquired)"}


def _unit_meta(cfg: dict, forms_meta: Dict[str, dict]) -> Dict[str, dict]:
    t = cfg["task"]
    out = {}
    for u in cfg["units"]:
        m = dict(forms_meta.get(u["unit_id"], {}))
        m.update({"unit_role": u.get("role", "substantive"), "resource_id": u["resource_id"],
                  "task": t["name"], "citation_cell": u.get("citation_cell", ""),
                  "n_eligible_cells": len(u["cells"]),
                  "cell_rule": f"max_multiword_share<={t['cell_rule']['max_multiword_share']}; complete single-word paradigm",
                  "exposure_max_shown": int(t["exposure"]["max_shown"]),
                  "k_distribution": f"uniform{{1..min({t['exposure']['max_shown']}, n_cells-1)}}",
                  "citation_cell_rule": t["citation_cell_rule"]})
        out[u["unit_id"]] = m
    return out


def outcome_table(items: pd.DataFrame, cfg: dict, forms_meta: Dict[str, dict]) -> pd.DataFrame:
    """Bootstrap estimates per design cell (wide format, one row per cell)."""
    ev = cfg["evaluation"]
    master = int(cfg["experiment"]["master_seed"])
    meta = _unit_meta(cfg, forms_meta)
    rows = []
    for key, sub in items.groupby(DESIGN, sort=True):
        d = dict(zip(DESIGN, key))
        seed = seedlib.derive(master, "bootstrap", *key)
        b = bootstrap.cluster_bootstrap(sub, int(ev["bootstrap_reps"]), seed, float(ev["ci_level"]))
        row = {**d, **meta.get(d["unit_id"], {}), "item_set_note": ITEM_SET_NOTE.get(d.get("item_set"), "")}
        for r in b.itertuples(index=False):
            row[f"{r.statistic}"] = r.estimate
            row[f"{r.statistic}_ci_low"] = r.ci_low
            row[f"{r.statistic}_ci_high"] = r.ci_high
        row.update({
            "n_test_lemmas": sub["lemma_id"].nunique(), "n_items": len(sub),
            "k_shown_mean": float(sub.drop_duplicates("lemma_id")["k_shown"].mean()) if "k_shown" in sub else None,
            "n_test_cells_per_verb_mean": float(sub.groupby("lemma_id").size().mean()),
            "copy_shown_form_rate": float(sub["pred_equals_shown_form"].mean()) if "pred_equals_shown_form" in sub else None,
            "n_missing": int((sub["status"] == "missing").sum()),
            "n_failed": int((~sub["status"].isin(["ok", "missing"])).sum()),
            "n_folds": sub["outer_fold"].nunique(), "n_repetitions": sub["repetition"].nunique(),
            "boot_seed": seed, "n_boot": int(ev["bootstrap_reps"]), "ci_level": float(ev["ci_level"]),
            "interval_note": "lemma-cluster percentile bootstrap over cached out-of-fold "
                             "predictions; conditional on fitted fold models",
            "config_hash": cfg.get("_config_hash"),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    out.columns = [c.replace("correct_micro", "accuracy_micro").replace("correct_macro", "accuracy_macro_lemma")
                   for c in out.columns]
    return out


def paired_table(items: pd.DataFrame, cfg: dict, comparisons: Sequence[tuple]) -> pd.DataFrame:
    """Paired differences for (policy_a, pool_a, policy_b, pool_b) at each unit/budget/model."""
    ev = cfg["evaluation"]
    master = int(cfg["experiment"]["master_seed"])
    rows = []
    if "item_set" in items:
        items = items[items["item_set"] == "core"]       # paired only on identical (core) items
    for (unit, budget, model), sub in items.groupby(["unit_id", "budget", "model"]):
        for pa, ca, pb, cb in comparisons:
            a = sub[(sub["policy"] == pa) & (sub["pool_cap"] == ca)]
            b = sub[(sub["policy"] == pb) & (sub["pool_cap"] == cb)]
            if a.empty or b.empty:
                continue
            seed = seedlib.derive(master, "bootstrap", unit, budget, model, pa, ca, pb, cb)
            d = bootstrap.paired_bootstrap(a, b, int(ev["bootstrap_reps"]), seed, float(ev["ci_level"]))
            d.insert(0, "comparison", f"{pa}@{ca} - {pb}@{cb}")
            for k, v in (("unit_id", unit), ("budget", budget), ("model", model)):
                d.insert(0, k, v)
            rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
