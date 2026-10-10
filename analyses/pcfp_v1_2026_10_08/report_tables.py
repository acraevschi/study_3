"""Report tables for docs/REPORT_pcfp_v1.md (reads outputs/pcfp_v1 and outputs/pilot_v1; writes here).

Run: .venv/bin/python analyses/pcfp_v1_2026_10_08/report_tables.py
Nothing under outputs/ is modified. Ancestry data are not read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from morph_ldl.cv import pcfp  # noqa: E402

OUT = Path(__file__).resolve().parent
P = ROOT / "outputs" / "pcfp_v1"
PILOT = ROOT / "outputs" / "pilot_v1"


def labels_for(exp: Path, uid: str) -> dict:
    f = pd.read_csv(exp / "data" / "forms" / f"{uid}.csv", usecols=["lemma_id", "lemma_label"], dtype=str)
    return dict(zip(f.lemma_id, f.lemma_label))


def accuracy_tables() -> None:
    o = pd.read_csv(P / "outcomes" / "ldl_outcomes.csv")
    cols = ["unit_id", "item_set", "policy", "pool_cap", "budget", "n_test_lemmas", "n_items",
            "accuracy_micro", "accuracy_micro_ci_low", "accuracy_micro_ci_high",
            "accuracy_macro_lemma", "accuracy_macro_lemma_ci_low", "accuracy_macro_lemma_ci_high",
            "edit_distance_micro", "edit_distance_micro_ci_low", "edit_distance_micro_ci_high",
            "norm_edit_distance_micro", "copy_shown_form_rate", "k_shown_mean", "n_test_cells_per_verb_mean",
            "n_failed", "n_missing"]
    o[cols].to_csv(OUT / "accuracy.csv", index=False)
    d = pd.read_csv(P / "outcomes" / "paired_differences.csv")
    d[d.statistic.isin(["correct_micro", "correct_macro", "edit_distance_micro"])].to_csv(OUT / "paired.csv", index=False)


def copy_rates() -> None:
    rows = []
    c = pd.read_csv(P / "eval" / "copy_rates.csv")
    for r in c.itertuples():
        rows.append({"experiment": "pcfp_v1", "unit_id": r.unit_id, "item_set": r.item_set, "policy": r.policy,
                     "pool_cap": r.pool_cap, "copy_any_shown_form": r.copy_shown_form_rate,
                     "copy_citation_form": r.copy_citation_rate, "n_items": r.n_items})
    # pilot_v1: top LDL candidate equals the supplied infinitive (source form)
    items = pd.read_csv(PILOT / "eval" / "item_predictions.csv", keep_default_na=False)
    items = items[items.model == "ldl"]
    src = []
    for q in (PILOT / "queries").rglob("test_queries.csv"):
        t = pd.read_csv(q, keep_default_na=False)
        src.append(t[["lemma_id", "source_segments"]].drop_duplicates())
    src = pd.concat(src).drop_duplicates("lemma_id").set_index("lemma_id")["source_segments"]
    items["copy"] = items.prediction == items.lemma_id.map(src)
    for (u, pol, cap), g in items.groupby(["unit_id", "policy", "pool_cap"]):
        rows.append({"experiment": "pilot_v1", "unit_id": u, "item_set": "heldout_new_verbs", "policy": pol,
                     "pool_cap": cap, "copy_any_shown_form": g["copy"].mean(), "copy_citation_form": g["copy"].mean(),
                     "n_items": len(g)})
    pd.DataFrame(rows).to_csv(OUT / "copy_rates_vs_pilot.csv", index=False)


def composition() -> None:
    """Inflection-class composition of acquired verbs (pcfp_v1 LDL selector vs random) and
    of the pilot_v1 Transformer's acquired verbs, against the pool."""
    rows = []
    for exp, name in ((P, "pcfp_v1"), (PILOT, "pilot_v1")):
        for uid in ("ita.V.orth.mgn", "fin.V.orth.mgn"):
            lab = labels_for(exp, uid)
            for lp in sorted((exp / "selection" / uid).glob("rep0/fold*/*@*/samples/budget_100_lemmas.csv")):
                tag = lp.parents[1].name
                fold = lp.parents[2].name
                t = pd.read_csv(lp)
                acq = t[(t["round"] > 0)] if "round" in t else t
                if "role" in t:
                    acq = t[t.role == "acquired"]
                cls = acq.lemma_id.map(lambda l: pcfp.inflection_class(uid, lab[l])).value_counts()
                rows.append({"experiment": name, "unit_id": uid, "fold": fold, "run": tag, "n_acquired": len(acq),
                             **{f"class_{k}": int(v) for k, v in cls.items()},
                             "citation_len_mean": acq.lemma_id.map(lambda l: len(lab[l])).mean()})
            # pool composition (pcfp_v1 split manifest; pilot pool differs slightly but is from the same resource)
            if exp == P:
                man = pd.read_csv(exp / "splits" / uid / "rep0" / "split_manifest.csv")
                for k, fm in man.groupby("outer_fold"):
                    pool = fm[fm.role == "pool"].lemma_id
                    cls = pool.map(lambda l: pcfp.inflection_class(uid, lab[l])).value_counts()
                    rows.append({"experiment": name, "unit_id": uid, "fold": f"fold{k}", "run": "POOL(500)",
                                 "n_acquired": len(pool), **{f"class_{c}": int(v) for c, v in cls.items()},
                                 "citation_len_mean": pool.map(lambda l: len(lab[l])).mean()})
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(OUT / "selection_composition.csv", index=False)
    cls_cols = [c for c in df.columns if c.startswith("class_")]
    share = df.copy()
    share[cls_cols] = share[cls_cols].div(share["n_acquired"], axis=0)
    (share.groupby(["experiment", "unit_id", "run"])[cls_cols + ["citation_len_mean"]].mean().round(3)
     .to_csv(OUT / "selection_composition_share.csv"))
    # overlap between policies within a fold (pcfp_v1)
    ov = []
    for uid in ("ita.V.orth.mgn", "fin.V.orth.mgn"):
        for fd in sorted((P / "selection" / uid / "rep0").glob("fold*")):
            sets = {}
            for lp in fd.glob("*@*/samples/budget_100_lemmas.csv"):
                t = pd.read_csv(lp)
                sets[lp.parents[1].name] = set(t[t.role == "acquired"].lemma_id)
            names = sorted(sets)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    ov.append({"unit_id": uid, "fold": fd.name, "a": a, "b": b, "overlap": len(sets[a] & sets[b])})
    pd.DataFrame(ov).to_csv(OUT / "selection_overlap.csv", index=False)


def breakdowns() -> None:
    k = pd.read_csv(P / "eval" / "summary_by_k.csv")
    k.to_csv(OUT / "accuracy_by_k.csv", index=False)
    c = pd.read_csv(P / "eval" / "summary_by_cell.csv")
    c = c[c.item_set == "core"]
    rows = []
    for (u, pol, cap), g in c.groupby(["unit_id", "policy", "pool_cap"]):
        g = g.sort_values("acc_micro")
        for r in pd.concat([g.head(5), g.tail(5)]).itertuples():
            rows.append({"unit_id": u, "policy": pol, "pool_cap": cap, "cell": r.target_cell, "acc": r.acc_micro,
                         "n_items": r.n_items})
    pd.DataFrame(rows).to_csv(OUT / "cell_extremes.csv", index=False)
    c.to_csv(OUT / "accuracy_by_cell_core.csv", index=False)


def selector_checks() -> None:
    s = pd.read_csv(P / "eval" / "selector_checks.csv")
    cols = [c for c in s.columns if c.startswith(("spearman_", "eta2_", "r2_", "adj_r2_"))] + [
        "score_cv", "top_equals_citation_rate", "no_candidate_rate"]
    s.groupby(["unit_id", "policy", "pool_cap"])[cols].mean().round(3).to_csv(OUT / "selector_checks_summary.csv")


if __name__ == "__main__":
    accuracy_tables()
    copy_rates()
    composition()
    breakdowns()
    selector_checks()
    print("written to", OUT)
