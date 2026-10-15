"""Non-LDL baselines for the grid PCFP runs, on exactly the LDL test items.

Each baseline sees, per fit, only what the LDL model saw: the shown forms (variant 0) of
that fit's core and random verbs. It predicts every hidden test cell of the core verbs.

* copy: predict the shown form of the verb's shown cell s that is most often identical to
  the target cell t among training verbs where both s and t are shown (syncretism-aware
  copying). No other knowledge.
* rules: an edit-rule (analogy) baseline. For each cell pair s -> t, every training verb with
  both cells shown gives a rule A -> C (prefix) and B -> D (suffix) around the longest common
  substring of the two forms (works for "ho parlato", "pisao sam" and phoneme strings).
  For a test verb, each shown source form x proposes the majority rule among training pairs
  whose source form ends like x (longest matching ending, 4 down to 1 segments) and whose
  rule applies to x. The source with the highest majority share wins (ties: longer
  matching ending, more evidence). Without any applicable rule, the copy prediction is used.

Run from the repository root:
    .venv/bin/python analyses/cell_grid_pilot_2026_10_14/ldl/baselines.py [experiment_id]
"""
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP = sys.argv[1] if len(sys.argv) > 1 else "pcfp_grid_v1"
OUT = ROOT / "outputs" / EXP
MAXCTX = 4


def rule_of(src, tgt):
    m = SequenceMatcher(None, src, tgt, autojunk=False).find_longest_match(0, len(src), 0, len(tgt))
    if m.size == 0:
        return (tuple(src), tuple(tgt), True)          # whole-form replacement
    return (tuple(src[:m.a]), tuple(tgt[:m.b]), tuple(src[m.a + m.size:]), tuple(tgt[m.b + m.size:]))


def applies(rule, x):
    if len(rule) == 3:
        return tuple(x) == rule[0]
    a, _, b, _ = rule
    return len(x) > len(a) + len(b) and tuple(x[:len(a)]) == a and (not b or tuple(x[-len(b):]) == b)


def apply(rule, x):
    if len(rule) == 3:
        return list(rule[1])
    a, c, b, d = rule
    mid = x[len(a):len(x) - len(b)] if b else x[len(a):]
    return list(c) + list(mid) + list(d)


def run_unit(unit):
    forms = pd.read_csv(OUT / "data/forms" / f"{unit}.csv", keep_default_na=False,
                        usecols=["lemma_id", "cell_norm", "variant_idx", "segments", "is_missing"])
    v0 = forms[(forms["variant_idx"] == 0) & (forms["is_missing"].astype(str) != "True")]
    seg = {(l, c): s.split(" ") for l, c, s in zip(v0["lemma_id"], v0["cell_norm"], v0["segments"])}
    expo = pd.read_csv(OUT / "splits" / unit / "exposure_manifest.csv", keep_default_na=False)
    shown = {r.lemma_id: str(r.shown_cells).split("|") for r in expo.itertuples()}
    items = pd.read_csv(OUT / "eval/item_predictions.csv", keep_default_na=False)
    items = items[(items["unit_id"] == unit) & (items["item_set"] == "core")]
    rows = []
    for (rep, fold), it in items.groupby(["repetition", "outer_fold"]):
        man = pd.read_csv(OUT / "splits" / unit / f"rep{rep}" / "split_manifest.csv")
        man = man[(man["outer_fold"] == fold) & man["role"].isin(["core", "random"])]
        train = man["lemma_id"].tolist()
        same, tot = Counter(), Counter()
        ctx = defaultdict(lambda: defaultdict(Counter))   # (s, t) -> ending -> Counter(rule)
        for l in train:
            cs = shown[l]
            for s in cs:
                for t in cs:
                    if s == t:
                        continue
                    x, y = seg[(l, s)], seg[(l, t)]
                    tot[(s, t)] += 1
                    same[(s, t)] += x == y
                    r = rule_of(x, y)
                    for k in range(1, MAXCTX + 1):
                        ctx[(s, t)][tuple(x[-k:])][r] += 1
        for r in it.itertuples():
            l, t = r.lemma_id, r.target_cell
            src = [s for s in shown[l] if s != t]
            ident = sorted(src, key=lambda s: (same[(s, t)] / tot[(s, t)] if tot[(s, t)] else -1), reverse=True)
            copy_pred = seg[(l, ident[0])]
            best = None
            for s in src:
                x = seg[(l, s)]
                for k in range(min(MAXCTX, len(x)), 0, -1):
                    cnt = Counter({ru: n for ru, n in ctx[(s, t)].get(tuple(x[-k:]), {}).items() if applies(ru, x)})
                    if cnt:
                        ru, n = cnt.most_common(1)[0]
                        key = (n / sum(cnt.values()), k, sum(cnt.values()))
                        if best is None or key > best[0]:
                            best = (key, apply(ru, x))
                        break
            rule_pred = best[1] if best else copy_pred
            gold = set(str(r.gold_variants).split("|"))
            rows.append({"unit_id": unit, "repetition": rep, "outer_fold": fold, "lemma_id": l,
                         "target_cell": t, "ldl_correct": str(r.correct) == "True",
                         "copy_correct": " ".join(copy_pred) in gold, "rules_correct": " ".join(rule_pred) in gold})
    return pd.DataFrame(rows)


def boot(df, col, reps=2000, seed=1):
    by = df.groupby("lemma_id")[col].agg(["sum", "size"])
    s, n = by["sum"].to_numpy(), by["size"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(by), size=(reps, len(by)))
    return np.quantile(s[idx].sum(1) / n[idx].sum(1), [0.025, 0.975])


if __name__ == "__main__":
    units = sorted(p.name for p in (OUT / "splits").iterdir() if p.is_dir())
    res = pd.concat([run_unit(u) for u in units], ignore_index=True)
    res.to_csv(OUT / "eval" / "baseline_items.csv", index=False)
    rows = []
    for u, g in res.groupby("unit_id"):
        row = {"unit_id": u, "n_items": len(g)}
        for m in ("ldl", "copy", "rules"):
            lo, hi = boot(g, f"{m}_correct")
            row[m] = round(g[f"{m}_correct"].mean(), 3)
            row[f"{m}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(HERE / f"baselines_{EXP}.csv", index=False)
    print(summ.to_string(index=False))
