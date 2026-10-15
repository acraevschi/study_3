"""Summary of the pcfp_grid_v1 LDL run (cell-grid pilot): accuracy by language and cell type,
and by whether the gold form was copyable from a shown form of the same verb.

Run from the repository root after the evaluate stage:
    .venv/bin/python analyses/cell_grid_pilot_2026_10_14/ldl/summarize_ldl.py [experiment_id]
Writes ldl_summary_*.csv next to this script (suffixed with the experiment id unless it is
pcfp_grid_v1).
Confidence intervals: cluster bootstrap over core verbs (all their items in all draws), 2000 reps.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = sys.argv[1] if len(sys.argv) > 1 else "pcfp_grid_v1"
SUF = "" if EXP == "pcfp_grid_v1" else "_" + EXP.removeprefix("pcfp_grid_v1_")
EVAL = ROOT / f"outputs/{EXP}/eval"
HERE = Path(__file__).resolve().parent
LANG = {"ita.V.orth.grid": "ita", "fin.V.orth.grid": "fin", "eng.V.phon_custom.grid": "eng",
        "hye.V.orth.grid": "hye", "hbs.V.orth.grid": "hbs"}
AUX = {"ita": "PST", "hye": "PRS", "hbs": "PST"}   # auxiliary-carried block (option B forms)

items = pd.read_csv(EVAL / "item_predictions.csv", keep_default_na=False)
items = items[items["item_set"] == "core"].copy()
items["lang"] = items["unit_id"].map(LANG)
items["correct"] = items["correct"].astype(str).eq("True")


def cell_type(r):
    c = r["target_cell"]
    if c == "NFIN":
        return "citation"
    if "IMP" in c:
        return "imperative"
    tense = "PST" if "PST" in c else "PRS"
    return "aux_block" if AUX.get(r["lang"]) == tense else f"plain_{tense}"


items["cell_type"] = items.apply(cell_type, axis=1)

# gold copyable from a shown form of the same verb (variant 0 of the shown cells)
forms = {u: pd.read_csv(ROOT / f"outputs/{EXP}/data/forms/{u}.csv", keep_default_na=False,
                        usecols=["lemma_id", "cell_norm", "variant_idx", "segments"]) for u in LANG}
expo = {u: pd.read_csv(ROOT / f"outputs/{EXP}/splits/{u}/exposure_manifest.csv", keep_default_na=False)
        for u in LANG}
shown = {}
for u in LANG:
    f = forms[u][forms[u]["variant_idx"] == 0].set_index(["lemma_id", "cell_norm"])["segments"]
    for r in expo[u].itertuples():
        shown[(u, r.lemma_id)] = {f.get((r.lemma_id, c), "") for c in str(r.shown_cells).split("|")}
items["copyable"] = [any(g in shown[(u, l)] for g in str(gv).split("|"))
                     for u, l, gv in zip(items["unit_id"], items["lemma_id"], items["gold_variants"])]


def boot(df, reps=2000, seed=1):
    by = df.groupby("lemma_id")["correct"].agg(["sum", "size"])
    s, n = by["sum"].to_numpy(), by["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by), size=(reps, len(by)))
    acc = s[idx].sum(1) / n[idx].sum(1)
    return np.quantile(acc, [0.025, 0.975])


def table(keys):
    rows = []
    for k, g in items.groupby(keys):
        lo, hi = boot(g)
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "n_items": len(g),
                     "accuracy": round(g["correct"].mean(), 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                     "edit_distance": round(g["edit_distance"].astype(float).mean(), 2),
                     "copyable_share": round(g["copyable"].mean(), 3)})
    return pd.DataFrame(rows)


overall = table(["lang"])
by_type = table(["lang", "cell_type"])
copy = table(["lang", "copyable"])
overall.to_csv(HERE / f"ldl_summary_overall{SUF}.csv", index=False)
by_type.to_csv(HERE / f"ldl_summary_by_type{SUF}.csv", index=False)
copy.to_csv(HERE / f"ldl_summary_copyable{SUF}.csv", index=False)
draw = (items.groupby(["lang", "repetition"])["correct"].mean().unstack().round(3))
draw.to_csv(HERE / f"ldl_accuracy_by_draw{SUF}.csv")
for t in (overall, by_type, copy):
    print(t.to_string(index=False), "\n")
print(draw.assign(sd_pt=(draw.std(axis=1) * 100).round(2)).to_string())
