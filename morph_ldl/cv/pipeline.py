"""Stage orchestration (main agent). Every stage reads what earlier stages wrote.

Layout under outputs/<experiment_id>/:
  data/ eligibility/ registry/ gelato/      data stage (morph_ldl.data)
  splits/<unit>/rep{r}/split_manifest.csv   splits stage
  selection/<unit>/rep{r}/fold{k}/<policy>@<pool>/   select stage (morph_ldl.selection)
  ldl/<unit>/rep{r}/fold{k}/<policy>@<pool>/budget_{B}/  ldl stage (morph_ldl.ldl)
  queries/<unit>/rep{r}/fold{k}/test_queries.csv     gold-free held-out queries
  eval/ item_predictions.csv, summaries, bootstrap, paired differences
  outcomes/ ldl_outcomes.csv, paired_differences.csv, population_links.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from morph_ldl import seeds as seedlib
from morph_ldl.config import output_dir, unit_cells
from morph_ldl.cv import splits
from morph_ldl.cv.queries import build_queries, source_information_budget
from morph_ldl.provenance import StageRecorder


# ----------------------------------------------------------------------------- helpers

def _units(cfg: dict, units: Optional[Iterable[str]] = None) -> List[dict]:
    us = [u for u in cfg["units"] if not units or u["unit_id"] in set(units)]
    if not us:
        raise ValueError(f"no configured unit matches {units}")
    return us


def _folds(cfg: dict, folds: Optional[Iterable[int]] = None) -> List[int]:
    ks = list(range(int(cfg["cv"]["n_folds"])))
    return [k for k in ks if not folds or k in set(folds)]


def _policies(cfg: dict, policies: Optional[Iterable[str]] = None) -> List[str]:
    ps = list(cfg["selection"]["policies"])
    return [p for p in ps if not policies or p in set(policies)]


def run_specs(cfg: dict, policies: Optional[Iterable[str]] = None) -> List[tuple]:
    """(policy, pool_cap, budgets) acquisition runs: every policy at the primary pool cap,
    plus the pool-size sensitivity runs (active policies, smallest budget only)."""
    primary = int(cfg["cv"]["pool_cap"])
    budgets = sorted(int(b) for b in cfg["selection"]["budgets"])
    specs = [(p, primary, budgets) for p in _policies(cfg, policies)]
    sens_policies = cfg["cv"]["pool_cap_sensitivity_policies"]
    for cap in cfg["cv"].get("pool_cap_sensitivity", []):
        for p in _policies(cfg, policies):
            if p in sens_policies:
                specs.append((p, int(cap), budgets[:1]))
    return specs


def run_tag(policy: str, pool_cap: int) -> str:
    return f"{policy}@{pool_cap}"


def split_path(cfg: dict, unit_id: str, rep: int) -> Path:
    return output_dir(cfg) / "splits" / unit_id / f"rep{rep}" / "split_manifest.csv"


def selection_dir(cfg: dict, unit_id: str, rep: int, fold: int, policy: str, pool_cap: int) -> Path:
    return output_dir(cfg) / "selection" / unit_id / f"rep{rep}" / f"fold{fold}" / run_tag(policy, pool_cap)


def ldl_dir(cfg: dict, unit_id: str, rep: int, fold: int, policy: str, pool_cap: int, budget: int) -> Path:
    return (output_dir(cfg) / "ldl" / unit_id / f"rep{rep}" / f"fold{fold}" / run_tag(policy, pool_cap)
            / f"budget_{budget}")


def aux_path(cfg: dict, unit_id: str) -> Path:
    return output_dir(cfg) / "splits" / unit_id / "auxiliary_manifest.csv"


def aux_sizes(cfg: dict) -> Dict[str, int]:
    t = cfg["ldl"]["tune"]
    return {"tune_background": int(t["background_size"]), "tune_heldout": int(t["heldout_size"]),
            "copy_anchor": int(cfg["selection"]["aux_copy_anchor_size"])}


def load_aux(cfg: dict, unit_id: str) -> Dict[str, List[str]]:
    p = aux_path(cfg, unit_id)
    if not p.exists():
        raise FileNotFoundError(f"{p} (run stage splits first)")
    return splits.aux_roles(pd.read_csv(p))


def file_sha(path: Path) -> str:
    from morph_ldl.provenance import sha256_file
    return sha256_file(Path(path))[:16]


def queries_path(cfg: dict, unit_id: str, rep: int, fold: int) -> Path:
    return output_dir(cfg) / "queries" / unit_id / f"rep{rep}" / f"fold{fold}" / "test_queries.csv"


def load_unit_forms(cfg: dict, unit_id: str) -> pd.DataFrame:
    from morph_ldl.data import load_forms
    return load_forms(cfg, unit_id)


def eligible_for_unit(cfg: dict, unit: dict, forms: pd.DataFrame) -> pd.DataFrame:
    c = unit_cells(unit, cfg)
    el = cfg["task"].get("eligibility", {})
    exclude: set = set()
    if el.get("exclude_derived_paradigms"):
        p = output_dir(cfg) / "eligibility" / f"{unit['unit_id']}_derived_paradigms.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} (run the data stage first)")
        exclude = set(pd.read_csv(p)["lemma_id"])
    return splits.eligible_lemmas(forms, c["source"], [x for _, x in c["panel"]],
                                  exclude_ids=exclude,
                                  exclude_multiword=bool(el.get("exclude_multiword_task_forms")))


# ----------------------------------------------------------------------------- stages

def stage_data(cfg: dict, units=None, folds=None, policies=None) -> None:
    from morph_ldl.data import run_data_stage
    res = run_data_stage(cfg)
    print(json.dumps({k: v for k, v in res.items() if k != "summary"}, default=str)[:2000])


def stage_splits(cfg: dict, units=None, folds=None, policies=None) -> None:
    out = output_dir(cfg) / "splits"
    with StageRecorder("splits", cfg, out) as rec:
        summary = []
        for unit in _units(cfg, units):
            uid = unit["unit_id"]
            forms = load_unit_forms(cfg, uid)
            lem = eligible_for_unit(cfg, unit, forms)
            for rep in cfg["cv"]["repetitions"]:
                man = splits.build_split_manifest(lem, cfg, uid, int(rep))
                path = splits.write_manifest(man, split_path(cfg, uid, int(rep)).parent)
                rec.inputs.append(path)
                if rep == cfg["cv"]["repetitions"][0]:
                    aux = splits.build_auxiliary_manifest(
                        lem, man["lemma_id"].unique(), aux_sizes(cfg),
                        int(cfg["experiment"]["master_seed"]), uid)
                    aux.to_csv(aux_path(cfg, uid), index=False)
                    rec.inputs.append(aux_path(cfg, uid))
                counts = man.groupby(["outer_fold", "role"]).size().unstack(fill_value=0)
                for k, row in counts.iterrows():
                    summary.append({"unit_id": uid, "repetition": rep, "outer_fold": k,
                                    "n_eligible_lemmas": len(lem), **row.to_dict()})
        df = pd.DataFrame(summary)
        df.to_csv(out / "split_summary.csv", index=False)
        rec.extra["summary"] = df.to_dict("records")
        print(df.to_string(index=False))


def task_cells(cfg: dict, unit: dict) -> tuple:
    c = unit_cells(unit, cfg)
    return c["source"], [x for _, x in c["panel"]]


def training_rows(forms: pd.DataFrame, lemma_ids: List[str], cells: List[str]) -> pd.DataFrame:
    """Variant-0 rows of the given cells for the given lemmas (panel training mode)."""
    sub = forms[forms["lemma_id"].isin(lemma_ids) & forms["cell_norm"].isin(cells)
                & (forms["variant_idx"] == 0) & (~forms["is_missing"].astype(bool))]
    n = sub.groupby("lemma_id").size()
    bad = n[n != len(cells)]
    if len(bad) or set(n.index) != set(lemma_ids):
        raise ValueError(f"incomplete training paradigms: {list(bad.index)[:5]}")
    return sub


def write_queries(cfg: dict, unit: dict, forms: pd.DataFrame, lemma_ids: List[str], path: Path) -> Path:
    src, panel = task_cells(cfg, unit)
    q = build_queries(forms, lemma_ids, src, panel)
    path.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(path, index=False)
    with open(path.with_suffix(".budget.json"), "w") as fh:
        json.dump(source_information_budget(q), fh)
    return path


def ldl_semantic_seed(cfg: dict, unit_id: str, rep: int, fold: int) -> int:
    return seedlib.derive(int(cfg["experiment"]["master_seed"]), "semantic", unit_id, rep, fold)


def stage_ldl_tune(cfg: dict, units=None, folds=None, policies=None) -> None:
    """Choose cue_ngram x sem_sd_inflection on auxiliary (never-tested) lemmas (PROTOCOL.md §6)."""
    import itertools
    from morph_ldl.cv import evaluate
    from morph_ldl.ldl.runner import run_ldl_jobs

    tune = cfg["ldl"]["tune"]
    rep, fold = int(tune["repetition"]), int(tune["fold"])
    out = output_dir(cfg) / "ldl_tune"
    if (output_dir(cfg) / "ldl").exists() and any((output_dir(cfg) / "ldl").rglob("predictions.csv")):
        raise RuntimeError("outer-test LDL outputs exist: settings are frozen; re-tuning is refused")
    grid = tune["grid"]
    keys = sorted(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    with StageRecorder("ldl_tune", cfg, out) as rec:
        jobs, meta = [], []
        golds = {}
        for unit in _units(cfg, units):
            uid = unit["unit_id"]
            forms = load_unit_forms(cfg, uid)
            aux = load_aux(cfg, uid)
            bg, held = aux["tune_background"], aux["tune_heldout"]
            src, panel = task_cells(cfg, unit)
            udir = out / uid
            udir.mkdir(parents=True, exist_ok=True)
            training_rows(forms, bg, [src, *panel]).to_csv(udir / "train.csv", index=False)
            qp = write_queries(cfg, unit, forms, held, udir / "tune_heldout_queries.csv")
            golds[uid] = (evaluate.gold_table(forms[forms["lemma_id"].isin(held)]),
                          dict(zip(forms["lemma_id"], forms["group_id"])))
            for combo in combos:
                tag = "_".join(f"{k}-{combo[k]}" for k in keys)
                jobs.append(dict(train_csv=str(udir / "train.csv"), queries_csv=str(qp),
                                 out_dir=str(udir / tag), unit_id=uid, repetition=rep, fold=fold,
                                 overrides=dict(combo)))
                meta.append((uid, tag, combo))
        run_ldl_jobs(jobs, cfg)
        rows = []
        for (uid, tag, combo), job in zip(meta, jobs):
            pred = pd.read_csv(Path(job["out_dir"]) / "predictions.csv", keep_default_na=False)
            gold, group_of = golds[uid]
            items = evaluate.score_items(pred, gold, dict(unit_id=uid, repetition=rep, outer_fold=fold,
                                                          policy="tune_background", pool_cap=0, budget=int(tune["background_size"]),
                                                          model="ldl"), group_of)
            rows.append({"unit_id": uid, "setting": tag, **combo, "dev_accuracy": items["correct"].mean(),
                         "dev_edit_distance": items["edit_distance"].mean(), "n_items": len(items),
                         "n_failed": int((items["status"] != "ok").sum())})
        res = pd.DataFrame(rows)
        agg = (res.groupby(["setting", *keys]).agg(mean_dev_accuracy=("dev_accuracy", "mean"),
                                                   mean_dev_edit_distance=("dev_edit_distance", "mean"))
               .reset_index().sort_values(["mean_dev_accuracy", "mean_dev_edit_distance"],
                                          ascending=[False, True]))
        chosen = {k: agg.iloc[0][k] for k in keys}
        chosen = {k: (int(v) if k == "cue_ngram" else float(v)) for k, v in chosen.items()}
        res.to_csv(out / "tune_by_unit.csv", index=False)
        agg.to_csv(out / "tune_summary.csv", index=False)
        with open(out / "chosen_settings.json", "w") as fh:
            json.dump(chosen, fh, indent=2)
        rec.extra["chosen"] = chosen
        print(res.to_string(index=False)); print(agg.to_string(index=False)); print("chosen:", chosen)


def frozen_ldl_settings(cfg: dict) -> dict:
    """Settings chosen by ldl_tune; required before any outer-test LDL fit."""
    p = output_dir(cfg) / "ldl_tune" / "chosen_settings.json"
    if not p.exists():
        raise FileNotFoundError("run stage ldl_tune first: outer-test fits need frozen settings")
    with open(p) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- selection

def _relevant_hash(cfg: dict, keys: Iterable[str], extra: object = None) -> str:
    import hashlib
    blob = json.dumps({k: cfg.get(k) for k in keys} | {"extra": extra, "master": cfg["experiment"]["master_seed"]},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


_FORMS_CACHE: Dict[str, pd.DataFrame] = {}


def _cached_forms(cfg: dict, unit_id: str) -> pd.DataFrame:
    if unit_id not in _FORMS_CACHE:
        _FORMS_CACHE[unit_id] = load_unit_forms(cfg, unit_id)
    return _FORMS_CACHE[unit_id]


def _anchor_ids(cfg: dict, unit_id: str) -> List[str]:
    """Fixed copy-anchor lemmas: auxiliary non-inventory lemmas (source forms only)."""
    return load_aux(cfg, unit_id)["copy_anchor"]


def code_hash(subdirs: Iterable[str]) -> str:
    import hashlib
    from morph_ldl.config import PIPELINE_ROOT
    h = hashlib.sha256()
    for sub in subdirs:
        for p in sorted((PIPELINE_ROOT / sub).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".jl", ".toml"} and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(PIPELINE_ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def unit_cfg(cfg: dict, unit_id: str) -> dict:
    return next(u for u in cfg["units"] if u["unit_id"] == unit_id)


def _selection_job(cfg: dict, unit_id: str, rep: int, fold: int, policy: str, cap: int, budgets: List[int]) -> dict:
    from morph_ldl.selection import SelectionSeeds, SelectionTask, run_acquisition
    out = selection_dir(cfg, unit_id, rep, fold, policy, cap)
    from morph_ldl.data.stage import forms_path
    h = _relevant_hash(cfg, ["selection", "selector", "cv", "task"],
                       [unit_id, rep, fold, policy, cap, budgets, unit_cfg(cfg, unit_id),
                        file_sha(split_path(cfg, unit_id, rep)), file_sha(aux_path(cfg, unit_id)),
                        file_sha(forms_path(cfg, unit_id)), code_hash(["morph_ldl/selection"])])
    done = out / "_done.json"
    if done.exists() and json.load(open(done)).get("hash") == h:
        return {"out": str(out), "skipped": True}
    forms = _cached_forms(cfg, unit_id)
    man = splits.load_manifest(split_path(cfg, unit_id, rep))
    r = splits.roles(man, rep, fold, pool_cap=cap)
    full = splits.roles(man, rep, fold)
    sub_cfg = json.loads(json.dumps(cfg))
    sub_cfg["selection"]["budgets"] = budgets
    task = SelectionTask.from_config(sub_cfg, unit_id)
    seeds = SelectionSeeds.derive(int(cfg["experiment"]["master_seed"]), unit_id, rep, fold)
    res = run_acquisition(policy, r["seed"], r["pool"], r["dev"], forms, task, sub_cfg, seeds, out,
                          test_ids=r["test"], anchor_ids=_anchor_ids(cfg, unit_id))
    with open(done, "w") as fh:
        json.dump({"hash": h, "summary": res.summary}, fh, default=str)
    return {"out": str(out), "skipped": False}


def _parallel(fn, argsets: List[tuple], n_procs: int) -> List[object]:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    if n_procs <= 1:
        return [fn(*a) for a in argsets]
    with ProcessPoolExecutor(max_workers=n_procs, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(fn, *a) for a in argsets]
        out = []
        for a, f in zip(argsets, futs):
            out.append(f.result())
            print("  done", a[1:], flush=True)
        return out


def stage_select(cfg: dict, units=None, folds=None, policies=None) -> None:
    out = output_dir(cfg) / "selection"
    argsets = []
    for unit in _units(cfg, units):
        for rep in cfg["cv"]["repetitions"]:
            for k in _folds(cfg, folds):
                for policy, cap, budgets in run_specs(cfg, policies):
                    argsets.append((cfg, unit["unit_id"], int(rep), k, policy, cap, budgets))
    with StageRecorder("select", cfg, out) as rec:
        res = _parallel(_selection_job, argsets, int(cfg["selection"].get("n_procs", 1)))
        rec.extra["jobs"] = res


# ----------------------------------------------------------------------------- LDL

def _ldl_jobs(cfg: dict, units=None, folds=None, policies=None) -> List[dict]:
    from morph_ldl.ldl.runner import write_gold_csv
    frozen = frozen_ldl_settings(cfg)
    jobs = []
    for unit in _units(cfg, units):
        uid = unit["unit_id"]
        forms = None
        for rep in cfg["cv"]["repetitions"]:
            rep = int(rep)
            man = splits.load_manifest(split_path(cfg, uid, rep))
            for k in _folds(cfg, folds):
                qp = queries_path(cfg, uid, rep, k)
                gp = output_dir(cfg) / "eval" / "gold" / uid / f"rep{rep}" / f"fold{k}" / "gold.csv"
                stamp = qp.with_suffix(".manifest_sha")
                want = file_sha(split_path(cfg, uid, rep)) + ":" + json.dumps(unit_cfg(cfg, uid)["cells"], sort_keys=True)
                if not qp.exists() or not gp.exists() or not stamp.exists() or stamp.read_text() != want:
                    forms = forms if forms is not None else load_unit_forms(cfg, uid)
                    r = splits.roles(man, rep, k)
                    write_queries(cfg, unit, forms, r["test"], qp)
                    gp.parent.mkdir(parents=True, exist_ok=True)
                    write_gold_csv(forms, pd.read_csv(qp, keep_default_na=False), gp)
                    stamp.write_text(want)
                for policy, cap, budgets in run_specs(cfg, policies):
                    sdir = selection_dir(cfg, uid, rep, k, policy, cap)
                    for b in budgets:
                        mode = cfg["task"].get("training_mode", "panel")
                        sample = sdir / "samples" / (f"budget_{b}.csv" if mode == "panel" else f"budget_{b}_allforms.csv")
                        if not sample.exists():
                            raise FileNotFoundError(f"{sample} (run stage select first)")
                        jobs.append(dict(train_csv=str(sample), queries_csv=str(qp),
                                         out_dir=str(ldl_dir(cfg, uid, rep, k, policy, cap, b)),
                                         unit_id=uid, repetition=rep, fold=k, overrides=dict(frozen),
                                         _gold=str(gp), _policy=policy, _cap=cap, _budget=b))
    return jobs


def stage_ldl(cfg: dict, units=None, folds=None, policies=None) -> None:
    from morph_ldl.ldl.runner import run_ldl_jobs, score_mapping_jobs
    jobs = _ldl_jobs(cfg, units, folds, policies)
    clean = [{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs]
    with StageRecorder("ldl", cfg, output_dir(cfg) / "ldl") as rec:
        run_ldl_jobs(clean, cfg)                                   # gold-free prediction
        score_mapping_jobs(clean, [j["_gold"] for j in jobs], cfg)  # gold-side diagnostics, afterwards
        rec.extra["n_jobs"] = len(jobs)
        rec.extra["frozen_settings"] = frozen_ldl_settings(cfg)


# ----------------------------------------------------------------------------- selector on test

def _selector_job(cfg: dict, unit_id: str, rep: int, fold: int, policy: str, cap: int, budget: int) -> dict:
    from morph_ldl.selection import SelectionSeeds, SelectionTask, predict_queries
    out = output_dir(cfg) / "selector_eval" / unit_id / f"rep{rep}" / f"fold{fold}" / run_tag(policy, cap) / f"budget_{budget}"
    pred_path = out / "predictions.csv"
    sample_csv = selection_dir(cfg, unit_id, rep, fold, policy, cap) / "samples" / f"budget_{budget}.csv"
    h = _relevant_hash(cfg, ["selector", "task"],
                       [unit_id, rep, fold, policy, cap, budget, unit_cfg(cfg, unit_id), file_sha(sample_csv),
                        file_sha(queries_path(cfg, unit_id, rep, fold)), file_sha(aux_path(cfg, unit_id)),
                        file_sha(split_path(cfg, unit_id, rep)), code_hash(["morph_ldl/selection"])])
    done = out / "_done.json"
    if pred_path.exists() and done.exists() and json.load(open(done)).get("hash") == h:
        return {"out": str(out), "skipped": True}
    forms = _cached_forms(cfg, unit_id)
    man = splits.load_manifest(split_path(cfg, unit_id, rep))
    full = splits.roles(man, rep, fold)
    sample = pd.read_csv(selection_dir(cfg, unit_id, rep, fold, policy, cap) / "samples" / f"budget_{budget}.csv",
                         keep_default_na=False, dtype={"segments": str, "form": str})
    sample["is_missing"] = sample["is_missing"].astype(str).str.lower().isin(["true", "1"])
    dev_rows = forms[forms["lemma_id"].isin(full["dev"])]
    anchor_rows = forms[forms["lemma_id"].isin(_anchor_ids(cfg, unit_id))]
    task = SelectionTask.from_config(cfg, unit_id)
    seed = SelectionSeeds.derive(int(cfg["experiment"]["master_seed"]), unit_id, rep, fold).selector_init
    queries = pd.read_csv(queries_path(cfg, unit_id, rep, fold), keep_default_na=False)
    pred = predict_queries(sample, queries, task=task, dev_rows=dev_rows, cfg=cfg, seed=seed,
                           anchor_rows=anchor_rows)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(pred_path, index=False)
    with open(done, "w") as fh:
        json.dump({"hash": h}, fh)
    return {"out": str(out), "skipped": False}


def stage_selector(cfg: dict, units=None, folds=None, policies=None) -> None:
    argsets = []
    for unit in _units(cfg, units):
        for rep in cfg["cv"]["repetitions"]:
            for k in _folds(cfg, folds):
                for policy, cap, budgets in run_specs(cfg, policies):
                    for b in budgets:
                        argsets.append((cfg, unit["unit_id"], int(rep), k, policy, cap, b))
    with StageRecorder("selector_eval", cfg, output_dir(cfg) / "selector_eval") as rec:
        rec.extra["jobs"] = _parallel(_selector_job, argsets, int(cfg["selection"].get("n_procs", 1)))


# ----------------------------------------------------------------------------- evaluation

COMPARISONS_DEFAULT = "active_vs_random_and_pool_sensitivity"


def comparisons(cfg: dict) -> List[tuple]:
    primary = int(cfg["cv"]["pool_cap"])
    out = [(p, primary, "random", primary) for p in cfg["selection"]["policies"] if p != "random"]
    for cap in cfg["cv"].get("pool_cap_sensitivity", []):
        for p in cfg["cv"]["pool_cap_sensitivity_policies"]:
            if p in cfg["selection"]["policies"]:
                out.append((p, primary, p, int(cap)))
                out.append((p, int(cap), "random", primary))
    return out


def _sample_composition(cfg: dict, uid: str, rep: int, k: int, policy: str, cap: int, budget: int,
                        forms: pd.DataFrame) -> dict:
    sdir = selection_dir(cfg, uid, rep, k, policy, cap)
    summ = json.load(open(sdir / "selection_summary.json"))
    b = summ["budgets"][str(budget)]
    lem = pd.read_csv(sdir / "samples" / f"budget_{budget}_lemmas.csv")
    src_cell, panel = task_cells(cfg, next(u for u in cfg["units"] if u["unit_id"] == uid))
    src = forms[forms["lemma_id"].isin(lem["lemma_id"]) & (forms["cell_norm"] == src_cell) & (forms["variant_idx"] == 0)]
    seglen = src["segments"].astype(str).str.split(" ").map(len)
    endings = src["form"].astype(str).str[-3:]
    rounds = json.load(open(sdir / "rounds.json"))
    return {"unit_id": uid, "repetition": rep, "outer_fold": k, "policy": policy, "pool_cap": cap,
            "budget": budget, "n_lemmas": b["n_lemmas"], "n_forms_panel": b["n_forms_panel_v0"],
            "n_training_examples_panel": b["n_training_examples_panel"],
            "n_forms_allcells_exported": b["n_rows_allforms"], "rounds_used": b["rounds_used"],
            "partial_last_round": b["partial_last_round"], "shortfall": b["shortfall"],
            "weights": b["weights"], "pool_size": summ["n_pool"], "seed_size": summ["n_seed"],
            "dev_size": summ["n_dev"], "n_panel_cells_covered": len(panel),
            "source_len_mean": float(seglen.mean()), "n_distinct_source_endings3": int(endings.nunique()),
            "top_source_endings3": "|".join(f"{e}:{c}" for e, c in endings.value_counts().head(5).items()),
            "n_acquisition_rounds_trained": len(rounds.get("rounds", [])),
            "selector_dev_acc_by_round": "|".join("na" if r.get("dev_acc") is None else f"{r['dev_acc']:.3f}"
                                          for r in rounds.get("rounds", []))}


def _truthy(col: pd.Series) -> pd.Series:
    return col.astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)


def stage_evaluate(cfg: dict, units=None, folds=None, policies=None) -> None:
    from morph_ldl.cv import bootstrap, evaluate
    out = output_dir(cfg) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with StageRecorder("evaluate", cfg, out) as rec:
        all_items, comp, mq, diag = [], [], [], []
        for unit in _units(cfg, units):
            uid = unit["unit_id"]
            forms = load_unit_forms(cfg, uid)
            for rep in cfg["cv"]["repetitions"]:
                rep = int(rep)
                man = splits.load_manifest(split_path(cfg, uid, rep))
                group_of = dict(zip(man["lemma_id"], man["group_id"]))
                for k in _folds(cfg, folds):
                    test_ids = splits.roles(man, rep, k)["test"]
                    gold = evaluate.gold_table(forms[forms["lemma_id"].isin(test_ids)])
                    for policy, cap, budgets in run_specs(cfg, policies):
                        for b in budgets:
                            meta = dict(unit_id=uid, repetition=rep, outer_fold=k, policy=policy,
                                        pool_cap=cap, budget=b)
                            ld = ldl_dir(cfg, uid, rep, k, policy, cap, b)
                            pred = pd.read_csv(ld / "predictions.csv", keep_default_na=False)
                            items = evaluate.score_items(pred, gold, {**meta, "model": "ldl"}, group_of)
                            if set(items["lemma_id"]) != set(test_ids):
                                raise RuntimeError(f"{ld}: predictions do not cover the test lemmas")
                            all_items.append(items)
                            m = pd.read_csv(ld / "mapping_quality.csv")
                            mq.append({**meta, "n_items": len(m),
                                       "chat_gold_cor_mean": m["chat_gold_cor"].mean(),
                                       "gold_reachable_rate": _truthy(m["gold_reachable"]).mean(),
                                       "gold_in_top_candidates_rate": _truthy(m["gold_in_top_candidates"]).mean(),
                                       "gold_cues_outside_inventory_mean": m["n_gold_cues_outside_inventory"].mean()})
                            d = json.load(open(ld / "diagnostics.json"))
                            diag.append({**meta, **{kk: v for kk, v in d.items() if not isinstance(v, (dict, list))}})
                            sp = (output_dir(cfg) / "selector_eval" / uid / f"rep{rep}" / f"fold{k}"
                                  / run_tag(policy, cap) / f"budget_{b}" / "predictions.csv")
                            if sp.exists():
                                sp_df = pd.read_csv(sp, keep_default_na=False)
                                sp_df["status"] = sp_df["status"].replace({"no_hypothesis": "no_candidate"})
                                all_items.append(evaluate.score_items(sp_df, gold, {**meta, "model": "selector"}, group_of))
                            comp.append(_sample_composition(cfg, uid, rep, k, policy, cap, b, forms))
        items = pd.concat(all_items, ignore_index=True)
        items.to_csv(out / "item_predictions.csv", index=False)
        design = ["unit_id", "policy", "pool_cap", "budget", "model"]
        evaluate.summarize(items, design).to_csv(out / "summary_point.csv", index=False)
        evaluate.summarize(items, design + ["target_cell"]).to_csv(out / "summary_by_cell.csv", index=False)
        per, spread = bootstrap.fold_variability(items, design)
        per.to_csv(out / "per_fold.csv", index=False)
        spread.to_csv(out / "fold_variability.csv", index=False)
        pd.DataFrame(comp).to_csv(out / "sample_composition.csv", index=False)
        pd.DataFrame(mq).to_csv(out / "ldl_mapping_quality.csv", index=False)
        pd.DataFrame(diag).to_csv(out / "ldl_diagnostics.csv", index=False)
        rec.extra["n_items"] = len(items)
        print(evaluate.summarize(items, design).to_string(index=False))


def stage_outcomes(cfg: dict, units=None, folds=None, policies=None) -> None:
    from morph_ldl.cv import outcomes
    from morph_ldl.data import build_population_links
    ev = output_dir(cfg) / "eval"
    out = output_dir(cfg) / "outcomes"
    out.mkdir(parents=True, exist_ok=True)
    with StageRecorder("outcomes", cfg, out, inputs=[ev / "item_predictions.csv"]) as rec:
        items = pd.read_csv(ev / "item_predictions.csv", keep_default_na=False)
        items["correct"] = items["correct"].astype(str).str.lower().eq("true")
        meta = {}
        for unit in _units(cfg, units):
            f = load_unit_forms(cfg, unit["unit_id"]).iloc[0]
            meta[unit["unit_id"]] = {k: f[k] for k in ("variety_id", "iso639_3", "glottocode", "pos",
                                                        "representation", "resource_version")}
        frozen = frozen_ldl_settings(cfg)
        for m in meta.values():
            m.update({f"ldl_{k}": v for k, v in frozen.items()})
            m.update({"ldl_source_binding": cfg["ldl"]["source_binding"], "ldl_decoder": cfg["ldl"]["decoder"],
                      "selector_arch": cfg["selector"]["arch"]})
        tab = outcomes.outcome_table(items, cfg, meta)
        tab.to_csv(out / "ldl_outcomes.csv", index=False)
        pt = outcomes.paired_table(items, cfg, comparisons(cfg))
        pt.to_csv(out / "paired_differences.csv", index=False)
        links = build_population_links([u["unit_id"] for u in _units(cfg, units)], cfg)
        links.to_csv(out / "population_links.csv", index=False)
        rec.extra.update({"n_outcome_rows": len(tab), "n_links": len(links)})
        cols = ["unit_id", "policy", "pool_cap", "budget", "model", "accuracy_micro",
                "accuracy_micro_ci_low", "accuracy_micro_ci_high", "edit_distance_micro"]
        print(tab[cols].to_string(index=False))
        if len(pt):
            print(pt[pt["statistic"].isin(["correct_micro", "edit_distance_micro"])][
                ["unit_id", "model", "comparison", "statistic", "difference", "ci_low", "ci_high"]].to_string(index=False))


# ----------------------------------------------------------------------------- artifact audit

def stage_audit(cfg: dict, units=None, folds=None, policies=None) -> None:
    """Check written artifacts against the split manifests (independent of component code)."""
    problems, checks = [], []
    for unit in _units(cfg, units):
        uid = unit["unit_id"]
        for rep in cfg["cv"]["repetitions"]:
            rep = int(rep)
            man = splits.load_manifest(split_path(cfg, uid, rep))
            for k in _folds(cfg, folds):
                full = splits.roles(man, rep, k)
                fm = man[(man["repetition"] == rep) & (man["outer_fold"] == k)]
                test_groups = set(fm.loc[fm["role"] == "test", "group_id"])
                group_of = dict(zip(fm["lemma_id"], fm["group_id"]))
                forbidden = set(full["test"]) | set(full["dev"])
                qpath = queries_path(cfg, uid, rep, k)
                if qpath.exists():
                    q = pd.read_csv(qpath, keep_default_na=False)
                    if set(q.columns) != set(["lemma_id", "source_cell", "source_form", "source_segments", "target_cell"]):
                        problems.append(f"{uid} r{rep} f{k}: query columns {list(q.columns)}")
                    if set(q["lemma_id"]) != set(full["test"]):
                        problems.append(f"{uid} r{rep} f{k}: queries != test lemmas")
                seeds_seen = {}
                for policy, cap, budgets in run_specs(cfg, policies):
                    sdir = selection_dir(cfg, uid, rep, k, policy, cap)
                    order = pd.read_csv(sdir / "order.csv")
                    seeds_seen[(policy, cap)] = tuple(sorted(order.loc[order["round"] == 0, "lemma_id"]))
                    pool_allowed = set(splits.roles(man, rep, k, pool_cap=cap)["pool"]) | set(full["seed"])
                    if not set(order["lemma_id"]) <= pool_allowed:
                        problems.append(f"{sdir}: selected lemmas outside seed+pool")
                    reveals = pd.read_csv(sdir / "oracle_reveals.csv")
                    if "lemma_id" in reveals and not set(reveals["lemma_id"]) <= set(order["lemma_id"]) | set(full["dev"]):
                        problems.append(f"{sdir}: oracle revealed unselected lemmas")
                    for b in budgets:
                        for name in (f"budget_{b}.csv", f"budget_{b}_allforms.csv"):
                            smp = pd.read_csv(sdir / "samples" / name, usecols=["lemma_id"])
                            ids = set(smp["lemma_id"])
                            if ids & forbidden or {group_of.get(i) for i in ids} & test_groups:
                                problems.append(f"{sdir}/{name}: contains test/dev lemma or test group")
                            if name == f"budget_{b}.csv" and len(ids) != b:
                                problems.append(f"{sdir}/{name}: {len(ids)} lemmas != budget {b}")
                        checks.append((uid, rep, k, policy, cap))
                if len(set(seeds_seen.values())) != 1:
                    problems.append(f"{uid} r{rep} f{k}: seed sets differ across policies")
        # Auxiliary lemmas (tuning, copy anchors) must lie outside every inventory group.
        inv_groups = set()
        for rep in cfg["cv"]["repetitions"]:
            inv_groups |= set(splits.load_manifest(split_path(cfg, uid, int(rep)))["group_id"])
        aux = pd.read_csv(aux_path(cfg, uid))
        if set(aux["group_id"]) & inv_groups:
            problems.append(f"{uid}: auxiliary lemmas share groups with the inventory")
        tdir = output_dir(cfg) / "ldl_tune" / uid
        for name in ("train.csv", "tune_heldout_queries.csv"):
            if (tdir / name).exists():
                ids = set(pd.read_csv(tdir / name, usecols=["lemma_id"])["lemma_id"])
                if not ids <= set(aux["lemma_id"]):
                    problems.append(f"{uid}: tuning file {name} uses non-auxiliary lemmas")
                checks.append((uid, "tune", name))
    out = output_dir(cfg) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "artifact_audit.json", "w") as fh:
        json.dump({"n_runs_checked": len(checks), "problems": problems}, fh, indent=2)
    print(f"audit: {len(checks)} sample sets checked, {len(problems)} problems")
    for p in problems:
        print("  PROBLEM", p)
    if problems:
        raise RuntimeError("artifact audit failed")
