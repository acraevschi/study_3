"""LDL ablation on the pcfp_v2 tuning sample (auxiliary verbs only; no outer-test verb is read).

Why: pcfp_v2 LDL accuracy is 0.17 (Italian) / 0.14 (Finnish), and the gold form is among
the decoder's 10 candidates for only about a fifth of items. Three suspected causes are
varied here, one stage at a time; each stage starts from the best setting of the stages
before it (criterion as in ldl_tune: mean held-out accuracy over the two units, ties by
lower mean edit distance).

  space    meaning space: sem_dim x sem_sd_noise x ridge_shift
  cell     a cell-specific meaning vector V(cell) next to the feature vectors (sem_sd_cell),
           plus cells-only references (sem_sd_inflection = 0)
  decoder  learn_paths threshold, and tolerant mode (up to 1 n-gram per path with support
           in (0, threshold])

Data: outputs/pcfp_v2/ldl_tune/<unit>/{train.csv, tune_queries.csv} (shown forms of the 100
tune_core + 100 tune_extra auxiliary verbs; queries = hidden cells of the tune_core verbs),
read-only, with the tuning semantic seed (repetition 0, fold -1). Base setting = the pcfp_v2
ldl_tune choice (bigram cues, inflection SD 2.0) with the pcfp_v2 ldl block.

Fits:   outputs/ldl_ablation_v1/<stage>/<unit>/<setting>/
Tables: analyses/ldl_ablation_2026_10_09/<stage>_by_unit.csv, <stage>_summary.csv,
        inputs.json (input hashes)

Run: .venv/bin/python analyses/ldl_ablation_2026_10_09/ablation.py space|cell|decoder
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from morph_ldl.config import PIPELINE_ROOT, load_config
from morph_ldl.cv import evaluate, pipeline
from morph_ldl.ldl.runner import run_ldl_jobs

HERE = Path(__file__).resolve().parent
CFG = PIPELINE_ROOT / "configs" / "pcfp_v2.yaml"
FITS = PIPELINE_ROOT / "outputs" / "ldl_ablation_v1"
STAGES = ("space", "cell", "decoder")


def grid(stage: str, base: dict) -> list[dict]:
    if stage == "space":
        g = {"sem_dim": [100, 200, 400, 1000], "sem_sd_noise": [0.0, 0.5, 1.0], "ridge_shift": [0.02, 1.0, 10.0, 100.0]}
        return [dict(zip(g, v)) for v in itertools.product(*g.values())]
    if stage == "cell":
        infl = float(base.get("sem_sd_inflection", 2.0))
        return ([{"sem_sd_cell": x} for x in (0.0, 0.5, 1.0, 2.0, 4.0)]
                + [{"sem_sd_inflection": 0.0, "sem_sd_cell": x} for x in (infl, 2 * infl)])
    if stage == "decoder":
        # build_paths was dropped: with bigram cues its paths loop (e.g. "ssubisasubise") and
        # one Italian item took 129 s with 3 neighbours (timing on 3 tuning items, 2026-10-09)
        # threshold 0.01 (both units) and max_tolerance 2 (Finnish) were stopped after 60 min
        # per fit (> 40x the current decoder; 2026-10-09) and are not in the grid. Italian
        # max_tolerance 2 had finished: accuracy 0.430 vs 0.440 with max_tolerance 1.
        return ([{"threshold": t} for t in (0.1, 0.05, 0.02)]
                + [{"tolerance": True, "tolerance_floor": 0.0, "max_tolerance": 1}])
    raise ValueError(stage)


def tag(combo: dict) -> str:
    return "_".join(f"{k}-{v}" for k, v in sorted(combo.items()))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def base_setting(cfg: dict, stage: str) -> dict:
    """pcfp_v2 ldl_tune choice, then the winners of the earlier stages in order."""
    chosen = json.loads((pipeline.output_dir(cfg) / "ldl_tune" / "chosen_settings.json").read_text())
    base = {"cue_ngram": int(chosen["cue_ngram"]), "sem_sd_inflection": float(chosen["sem_sd_inflection"])}
    for s in STAGES[:STAGES.index(stage)]:
        p = HERE / f"{s}_summary.csv"
        if not p.exists():
            raise FileNotFoundError(f"run stage {s} first")
        best = json.loads(pd.read_csv(p).iloc[0]["setting_json"])
        base.update(best)
    return base


def main(stage: str) -> None:
    cfg = load_config(CFG)
    tune = cfg["ldl"]["tune"]
    rep, fold = int(tune["repetition"]), int(tune["fold"])
    base = base_setting(cfg, stage)
    combos = grid(stage, base)
    tdir = pipeline.output_dir(cfg) / "ldl_tune"
    jobs, meta, prep, inputs = [], [], {}, {}
    for unit in cfg["units"]:
        uid = unit["unit_id"]
        train, queries = tdir / uid / "train.csv", tdir / uid / "tune_queries.csv"
        inputs[uid] = {"train_csv": [str(train.relative_to(PIPELINE_ROOT)), sha(train)],
                       "queries_csv": [str(queries.relative_to(PIPELINE_ROOT)), sha(queries)]}
        forms = pipeline.load_unit_forms(cfg, uid)
        expo = pipeline.load_exposure(cfg, uid)
        aux = pipeline.load_aux(cfg, uid)
        q = pd.read_csv(queries, keep_default_na=False)
        assert set(q["lemma_id"]) <= set(aux["tune_core"]), "tuning queries must be tune_core verbs"
        prep[uid] = (forms, q, dict(zip(forms["lemma_id"], forms["group_id"])), expo)
        for combo in combos:
            setting = {**base, **combo}
            jobs.append(dict(train_csv=str(train), queries_csv=str(queries),
                             out_dir=str(FITS / stage / uid / tag(setting)), unit_id=uid,
                             repetition=rep, fold=fold, overrides=setting))
            meta.append((uid, setting))
    (HERE / "inputs.json").write_text(json.dumps(inputs, indent=2))
    run_ldl_jobs(jobs, cfg)

    rows = []
    for (uid, setting), job in zip(meta, jobs):
        forms, q, group_of, expo = prep[uid]
        out = Path(job["out_dir"])
        pred = pd.read_csv(out / "predictions.csv", keep_default_na=False)
        gold = evaluate.gold_table(forms[forms["lemma_id"].isin(set(q["lemma_id"]))])
        items = evaluate.score_items(pred, gold, dict(unit_id=uid, repetition=rep, outer_fold=fold, policy="ablation",
                                                      pool_cap=0, budget=0, model="ldl"), group_of)
        shown = pipeline._shown_segments(forms, expo, q["lemma_id"].unique())
        copy = np.mean([p in shown.get(l, set()) for l, p in zip(items["lemma_id"], items["prediction"])])
        # gold among the decoder's candidates (forms; "_" is a word space in segments)
        in_cand, rank1 = [], []
        for r in pred.itertuples(index=False):
            golds = {"".join(v.split()).replace("_", " ") for v in gold[(r.lemma_id, r.target_cell)]}
            cands = [c["prediction"] for c in json.loads(r.top_candidates or "[]")]
            in_cand.append(any(c in golds for c in cands))
            rank1.append(bool(cands) and cands[0] in golds)
        in_cand, rank1 = np.array(in_cand), np.array(rank1)
        diag = json.loads((out / "diagnostics.json").read_text())
        rows.append({"unit_id": uid, "setting": tag(setting), "setting_json": json.dumps(setting, sort_keys=True),
                     **{k: setting.get(k) for k in sorted(set(k for c in combos for k in c))},
                     "accuracy": items["correct"].mean(), "edit_distance": items["edit_distance"].mean(),
                     "copy_shown_form_rate": copy, "gold_in_candidates": in_cand.mean(),
                     "top1_given_in_candidates": rank1[in_cand].mean() if in_cand.any() else np.nan,
                     "n_items": len(items), "n_failed": int((items["status"] != "ok").sum()),
                     "train_production_accuracy": diag.get("train_production_accuracy"),
                     "train_comprehension_accuracy": diag.get("train_comprehension_accuracy"),
                     "n_train_rows": diag.get("n_train_rows"),
                     "fit_seconds": diag.get("background_fit_seconds"), "predict_seconds": diag.get("predict_seconds")})
    res = pd.DataFrame(rows)
    keys = [c for c in res.columns if c not in ("unit_id",) and c in sorted(set(k for c in combos for k in c))]
    agg = (res.groupby(["setting", "setting_json", *keys], dropna=False)
           .agg(mean_accuracy=("accuracy", "mean"), mean_edit_distance=("edit_distance", "mean"),
                mean_copy=("copy_shown_form_rate", "mean"), mean_gold_in_candidates=("gold_in_candidates", "mean"),
                mean_train_production=("train_production_accuracy", "mean"))
           .reset_index().sort_values(["mean_accuracy", "mean_edit_distance"], ascending=[False, True]))
    wide = res.pivot(index="setting", columns="unit_id", values="accuracy").add_prefix("acc_")
    agg = agg.merge(wide, left_on="setting", right_index=True)
    res.round(6).to_csv(HERE / f"{stage}_by_unit.csv", index=False)
    agg.round(6).to_csv(HERE / f"{stage}_summary.csv", index=False)
    print("base:", base)
    print(agg.drop(columns=["setting", "setting_json"]).round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: ablation.py {'|'.join(STAGES)}")
    main(sys.argv[1])
