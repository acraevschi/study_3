"""Stage orchestration for paradigm cell filling (main agent). Every stage reads what
earlier stages wrote. The pilot_v1 (source-known new-verb) orchestration is preserved at
commit 24390cf.

Layout under outputs/<experiment_id>/:
  data/ eligibility/ registry/ gelato/            data stage (morph_ldl.data)
  splits/<unit>/cell_inventory.csv, eligible_lemmas.csv, exposure_manifest.csv,
         auxiliary_manifest.csv, rep{r}/split_manifest.csv          splits stage
  ldl_tune/                                        LDL setting choice on auxiliary verbs
  selection/<unit>/rep{r}/fold{k}/<policy>@<pool>/ select stage (LDL selector)
  queries/<unit>/rep{r}/fold{k}/<policy>@<pool>/budget_{B}/queries.csv   gold-free queries
  ldl/<unit>/rep{r}/fold{k}/<policy>@<pool>/budget_{B}/                  ldl stage
  eval/       item_predictions.csv, summaries, selector checks, audit
  outcomes/   ldl_outcomes.csv, paired_differences.csv, population_links.csv
  typology/   Grambank inflection-extent outcome (independent of the stages above)
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from morph_ldl import seeds as seedlib
from morph_ldl.config import PIPELINE_ROOT, is_pcfp, output_dir, unit_cells
from morph_ldl.cv import pcfp, splits
from morph_ldl.provenance import StageRecorder


# ----------------------------------------------------------------------------- helpers

def _require_pcfp(cfg: dict) -> None:
    if not is_pcfp(cfg):
        raise RuntimeError("this pipeline runs the PCFP task (task.name: pcfp); the pilot_v1 "
                           "source-known task is preserved at commit 24390cf")


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
    """(policy, pool_cap, budgets): every policy at the primary pool cap, plus the
    pool-size sensitivity runs (declared policies, smallest budget only)."""
    primary = int(cfg["cv"]["pool_cap"])
    budgets = sorted(int(b) for b in cfg["selection"]["budgets"])
    specs = [(p, primary, budgets) for p in _policies(cfg, policies)]
    for cap in cfg["cv"].get("pool_cap_sensitivity", []):
        for p in _policies(cfg, policies):
            if p in cfg["cv"]["pool_cap_sensitivity_policies"]:
                specs.append((p, int(cap), budgets[:1]))
    return specs


def run_tag(policy: str, pool_cap: int) -> str:
    return f"{policy}@{pool_cap}"


def unit_dir(cfg: dict, unit_id: str) -> Path:
    return output_dir(cfg) / "splits" / unit_id


def split_path(cfg: dict, unit_id: str, rep: int) -> Path:
    return unit_dir(cfg, unit_id) / f"rep{rep}" / "split_manifest.csv"


def aux_path(cfg: dict, unit_id: str) -> Path:
    return unit_dir(cfg, unit_id) / "auxiliary_manifest.csv"


def exposure_path(cfg: dict, unit_id: str) -> Path:
    return unit_dir(cfg, unit_id) / "exposure_manifest.csv"


def selection_dir(cfg: dict, unit_id: str, rep: int, fold: int, policy: str, pool_cap: int) -> Path:
    return output_dir(cfg) / "selection" / unit_id / f"rep{rep}" / f"fold{fold}" / run_tag(policy, pool_cap)


def ldl_dir(cfg: dict, unit_id: str, rep: int, fold: int, policy: str, pool_cap: int, budget: int) -> Path:
    return (output_dir(cfg) / "ldl" / unit_id / f"rep{rep}" / f"fold{fold}" / run_tag(policy, pool_cap)
            / f"budget_{budget}")


def queries_path(cfg: dict, unit_id: str, rep: int, fold: int, policy: str, pool_cap: int, budget: int) -> Path:
    return (output_dir(cfg) / "queries" / unit_id / f"rep{rep}" / f"fold{fold}" / run_tag(policy, pool_cap)
            / f"budget_{budget}" / "queries.csv")


def gold_path(cfg: dict, unit_id: str, rep: int, fold: int, policy: str, pool_cap: int, budget: int) -> Path:
    return (output_dir(cfg) / "eval" / "gold" / unit_id / f"rep{rep}" / f"fold{fold}" / run_tag(policy, pool_cap)
            / f"budget_{budget}" / "gold.csv")


def load_aux(cfg: dict, unit_id: str) -> Dict[str, List[str]]:
    p = aux_path(cfg, unit_id)
    if not p.exists():
        raise FileNotFoundError(f"{p} (run stage splits first)")
    return splits.aux_roles(pd.read_csv(p))


def load_exposure(cfg: dict, unit_id: str) -> Dict[str, pcfp.Exposure]:
    p = exposure_path(cfg, unit_id)
    if not p.exists():
        raise FileNotFoundError(f"{p} (run stage splits first)")
    return pcfp.load_exposure(p)


def file_sha(path: Path) -> str:
    from morph_ldl.provenance import sha256_file
    return sha256_file(Path(path))[:16]


def load_unit_forms(cfg: dict, unit_id: str) -> pd.DataFrame:
    from morph_ldl.data import load_forms
    return load_forms(cfg, unit_id)


_FORMS_CACHE: Dict[str, pd.DataFrame] = {}


def _cached_forms(cfg: dict, unit_id: str) -> pd.DataFrame:
    key = f"{output_dir(cfg)}|{unit_id}"
    if key not in _FORMS_CACHE:
        _FORMS_CACHE[key] = load_unit_forms(cfg, unit_id)
    return _FORMS_CACHE[key]


def unit_cfg(cfg: dict, unit_id: str) -> dict:
    return next(u for u in cfg["units"] if u["unit_id"] == unit_id)


def derived_exclusions(cfg: dict, unit_id: str) -> set:
    if not cfg["task"].get("eligibility", {}).get("exclude_derived_paradigms"):
        return set()
    p = output_dir(cfg) / "eligibility" / f"{unit_id}_derived_paradigms.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} (run the data stage first)")
    return set(pd.read_csv(p)["lemma_id"])


def code_hash(subdirs: Iterable[str]) -> str:
    h = hashlib.sha256()
    for sub in subdirs:
        for p in sorted((PIPELINE_ROOT / sub).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".jl", ".toml"} and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(PIPELINE_ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _relevant_hash(cfg: dict, keys: Iterable[str], extra: object = None) -> str:
    blob = json.dumps({k: cfg.get(k) for k in keys} | {"extra": extra, "master": cfg["experiment"]["master_seed"]},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


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


def frozen_ldl_settings(cfg: dict) -> dict:
    """Settings chosen by ldl_tune; required before selection and any outer-test fit."""
    p = output_dir(cfg) / "ldl_tune" / "chosen_settings.json"
    if not p.exists():
        raise FileNotFoundError("run stage ldl_tune first: the LDL selector and outer-test fits need frozen settings")
    with open(p) as fh:
        d = json.load(fh)
    return {k: v for k, v in d.items() if k in cfg["ldl"]["tune"]["grid"]}


def _citation(cfg: dict, unit: dict) -> str:
    return unit_cells(unit, cfg)["citation"]


def _write_gold(forms: pd.DataFrame, queries: pd.DataFrame, path: Path) -> Path:
    from morph_ldl.ldl.runner import write_gold_csv
    return write_gold_csv(forms[forms["lemma_id"].isin(set(queries["lemma_id"]))], queries, path)


# ----------------------------------------------------------------------------- data / splits

def stage_data(cfg: dict, units=None, folds=None, policies=None) -> None:
    from morph_ldl.data import run_data_stage
    res = run_data_stage(cfg)
    print(json.dumps({k: v for k, v in res.items() if k != "summary"}, default=str)[:2000])


def stage_splits(cfg: dict, units=None, folds=None, policies=None) -> None:
    """Cell inventory check, eligible lemmas, exposure manifest, split and auxiliary manifests."""
    _require_pcfp(cfg)
    out = output_dir(cfg) / "splits"
    t = cfg["task"]
    master = int(cfg["experiment"]["master_seed"])
    with StageRecorder("splits", cfg, out) as rec:
        summary, expo = [], {}
        for unit in _units(cfg, units):
            uid = unit["unit_id"]
            udir = unit_dir(cfg, uid)
            udir.mkdir(parents=True, exist_ok=True)
            forms = load_unit_forms(cfg, uid)
            excl = derived_exclusions(cfg, uid)
            inv_tab = pcfp.cell_inventory_table(forms, excl, float(t["cell_rule"]["max_multiword_share"]))
            data_cells = sorted(inv_tab.loc[inv_tab["eligible"], "cell_norm"])
            uc = unit_cells(unit, cfg)
            inv_tab["declared"] = inv_tab["cell_norm"].isin(uc["eligible"])
            inv_tab.to_csv(udir / "cell_inventory.csv", index=False)
            if data_cells != uc["eligible"]:
                raise RuntimeError(f"{uid}: declared cells differ from the cell rule: "
                                   f"missing {sorted(set(data_cells) - set(uc['eligible']))}, "
                                   f"extra {sorted(set(uc['eligible']) - set(data_cells))}")
            lem = pcfp.eligible_lemmas(forms, uc["eligible"], uc["citation"], excl,
                                       bool(t["eligibility"].get("require_complete_paradigm", True)))
            lem.to_csv(udir / "eligible_lemmas.csv", index=False)
            em = pcfp.build_exposure_manifest(lem, uc["eligible"], uc["citation"], master, uid,
                                              int(t["exposure"]["max_shown"]))
            em.to_csv(exposure_path(cfg, uid), index=False)
            expo[uid] = pcfp.validate_exposure_manifest(pd.read_csv(exposure_path(cfg, uid)), uc["eligible"],
                                                        uc["citation"], master, uid, int(t["exposure"]["max_shown"]))
            rec.inputs += [exposure_path(cfg, uid), udir / "cell_inventory.csv"]
            for rep in cfg["cv"]["repetitions"]:
                man = splits.build_pcfp_manifest(lem, cfg, uid, int(rep))
                path = splits.write_manifest(man, split_path(cfg, uid, int(rep)).parent)
                rec.inputs.append(path)
                if rep == cfg["cv"]["repetitions"][0]:
                    tune = cfg["ldl"]["tune"]
                    aux = splits.build_auxiliary_manifest(
                        lem, man["lemma_id"].unique(),
                        {"tune_core": int(tune["core_size"]), "tune_extra": int(tune["extra_size"])}, master, uid)
                    aux.to_csv(aux_path(cfg, uid), index=False)
                    rec.inputs.append(aux_path(cfg, uid))
                counts = man.groupby(["outer_fold", "role"]).size().unstack(fill_value=0)
                emi = em.set_index("lemma_id")
                for k, row in counts.iterrows():
                    core = man[(man["outer_fold"] == k) & (man["role"] == "core")]["lemma_id"]
                    summary.append({"unit_id": uid, "repetition": rep, "outer_fold": k,
                                    "n_eligible_cells": len(uc["eligible"]), "n_eligible_lemmas": len(lem),
                                    **row.to_dict(),
                                    "core_k_mean": float(emi.loc[core, "k"].mean()),
                                    "core_test_items": int(emi.loc[core, "n_test_cells"].sum()),
                                    "core_test_cells_per_verb_mean": float(emi.loc[core, "n_test_cells"].mean())})
        df = pd.DataFrame(summary)
        df.to_csv(out / "split_summary.csv", index=False)
        with open(out / "exposure_summary.json", "w") as fh:
            json.dump(expo, fh, indent=1)
        rec.extra.update({"summary": df.to_dict("records"), "exposure": expo})
        print(df.to_string(index=False))
        print(json.dumps(expo, indent=1))


# ----------------------------------------------------------------------------- LDL tuning

def _outer_outputs_exist(cfg: dict) -> List[str]:
    found = []
    sel = output_dir(cfg) / "selection"
    if sel.exists() and any(sel.rglob("order.csv")):
        found.append("selection")
    ld = output_dir(cfg) / "ldl"
    if ld.exists() and any(ld.rglob("predictions.csv")):
        found.append("ldl")
    return found


def _grid_combos(grid: dict) -> List[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _extend_grid(grid: dict, chosen: dict) -> Optional[dict]:
    """Declared boundary rule: one step outward for an edge choice (sd 0.1 / 10.0; n-gram +1)."""
    new = {k: list(v) for k, v in grid.items()}
    changed = False
    sd = sorted(float(x) for x in grid["sem_sd_inflection"])
    if len(sd) > 1 and float(chosen["sem_sd_inflection"]) == sd[0] and sd[0] > 0.1:
        new["sem_sd_inflection"] = [0.1] + sd; changed = True
    elif len(sd) > 1 and float(chosen["sem_sd_inflection"]) == sd[-1] and sd[-1] < 10.0:
        new["sem_sd_inflection"] = sd + [10.0]; changed = True
    ng = sorted(int(x) for x in grid["cue_ngram"])
    if len(ng) > 1 and int(chosen["cue_ngram"]) == ng[-1] and ng[-1] < 4:
        new["cue_ngram"] = ng + [ng[-1] + 1]; changed = True
    return new if changed else None


def stage_ldl_tune(cfg: dict, units=None, folds=None, policies=None) -> None:
    """Choose cue_ngram x sem_sd_inflection on auxiliary verbs under core-like partial
    exposure (PROTOCOL.md §6). Runs before selection; refused once selection or outer-test
    LDL outputs exist."""
    _require_pcfp(cfg)
    from morph_ldl.cv import evaluate
    from morph_ldl.ldl.runner import run_ldl_jobs

    tune = cfg["ldl"]["tune"]
    rep, fold = int(tune["repetition"]), int(tune["fold"])
    out = output_dir(cfg) / "ldl_tune"
    found = _outer_outputs_exist(cfg)
    if found:
        raise RuntimeError(f"{found} outputs exist: LDL settings are frozen; re-tuning is refused")
    if units and set(units) != {u["unit_id"] for u in cfg["units"]}:
        raise RuntimeError("ldl_tune chooses one setting over all units; a unit filter is refused")
    with StageRecorder("ldl_tune", cfg, out) as rec:
        prep = {}
        for unit in _units(cfg, units):
            uid = unit["unit_id"]
            forms = load_unit_forms(cfg, uid)
            aux = load_aux(cfg, uid)
            expo = load_exposure(cfg, uid)
            core, extra = aux["tune_core"], aux["tune_extra"]
            udir = out / uid
            udir.mkdir(parents=True, exist_ok=True)
            pcfp.shown_rows(forms, expo, core + extra).to_csv(udir / "train.csv", index=False)
            q = pcfp.hidden_queries(expo, core, _citation(cfg, unit))
            q.to_csv(udir / "tune_queries.csv", index=False)
            prep[uid] = (forms, q, dict(zip(forms["lemma_id"], forms["group_id"])), expo)
        grid = {k: list(v) for k, v in tune["grid"].items()}
        history = []
        while True:
            combos = _grid_combos(grid)
            keys = sorted(grid)
            jobs, meta = [], []
            for uid in prep:
                udir = out / uid
                for combo in combos:
                    tag = "_".join(f"{k}-{combo[k]}" for k in keys)
                    jobs.append(dict(train_csv=str(udir / "train.csv"), queries_csv=str(udir / "tune_queries.csv"),
                                     out_dir=str(udir / tag), unit_id=uid, repetition=rep, fold=fold,
                                     overrides=dict(combo)))
                    meta.append((uid, tag, combo))
            run_ldl_jobs(jobs, cfg)
            rows = []
            for (uid, tag, combo), job in zip(meta, jobs):
                forms, q, group_of, expo = prep[uid]
                pred = pd.read_csv(Path(job["out_dir"]) / "predictions.csv", keep_default_na=False)
                gold = evaluate.gold_table(forms[forms["lemma_id"].isin(set(q["lemma_id"]))])
                items = evaluate.score_items(pred, gold, dict(unit_id=uid, repetition=rep, outer_fold=fold,
                                                              policy="tune", pool_cap=0, budget=0, model="ldl"),
                                             group_of)
                shown = _shown_segments(forms, expo, q["lemma_id"].unique())
                copy = [p in shown.get(l, set()) for l, p in zip(items["lemma_id"], items["prediction"])]
                rows.append({"unit_id": uid, "setting": tag, **combo, "heldout_accuracy": items["correct"].mean(),
                             "heldout_edit_distance": items["edit_distance"].mean(), "n_items": len(items),
                             "n_failed": int((items["status"] != "ok").sum()),
                             "copy_shown_form_rate": float(np.mean(copy))})
            res = pd.DataFrame(rows)
            agg = (res.groupby(["setting", *keys]).agg(mean_heldout_accuracy=("heldout_accuracy", "mean"),
                                                       mean_heldout_edit_distance=("heldout_edit_distance", "mean"))
                   .reset_index().sort_values(["mean_heldout_accuracy", "mean_heldout_edit_distance"],
                                              ascending=[False, True]))
            chosen = {k: agg.iloc[0][k] for k in keys}
            chosen = {k: (int(v) if k == "cue_ngram" else float(v)) for k, v in chosen.items()}
            history.append({"grid": grid, "chosen": chosen})
            nxt = _extend_grid(grid, chosen)
            if nxt is None or len(history) > 1:
                break
            print(f"ldl_tune: choice {chosen} is at a grid edge; extending the grid to {nxt} (declared rule)")
            grid = nxt
        res.to_csv(out / "tune_by_unit.csv", index=False)
        agg.to_csv(out / "tune_summary.csv", index=False)
        at_edge = _extend_grid(grid, chosen) is not None
        with open(out / "chosen_settings.json", "w") as fh:
            json.dump({**chosen, "_grid_final": grid, "_history": history, "_still_at_edge": at_edge}, fh, indent=2)
        rec.extra.update({"chosen": chosen, "history": history, "still_at_edge": at_edge})
        print(res.to_string(index=False)); print(agg.to_string(index=False)); print("chosen:", chosen)


def _shown_segments(forms: pd.DataFrame, expo: Dict[str, pcfp.Exposure], lemma_ids: Iterable[str]) -> Dict[str, set]:
    rows = pcfp.shown_rows(forms, expo, lemma_ids)
    out: Dict[str, set] = {}
    for l, s in zip(rows["lemma_id"], rows["segments"].astype(str)):
        out.setdefault(l, set()).add(s)
    return out


# ----------------------------------------------------------------------------- selection

def _selection_job(cfg: dict, unit_id: str, rep: int, fold: int, policy: str, cap: int, budgets: List[int]) -> dict:
    from morph_ldl.data.stage import forms_path
    from morph_ldl.ldl.selector import SelectorServer, selector_configs
    from morph_ldl.selection import SelectionSeeds
    from morph_ldl.selection.ldl_acquisition import run_ldl_acquisition
    out = selection_dir(cfg, unit_id, rep, fold, policy, cap)
    frozen = frozen_ldl_settings(cfg)
    h = _relevant_hash(cfg, ["selection", "cv", "task", "ldl"],
                       [unit_id, rep, fold, policy, cap, budgets, unit_cfg(cfg, unit_id), frozen,
                        file_sha(split_path(cfg, unit_id, rep)), file_sha(exposure_path(cfg, unit_id)),
                        file_sha(forms_path(cfg, unit_id)),
                        code_hash(["morph_ldl/selection", "morph_ldl/ldl", "morph_ldl/cv/pcfp.py", "julia/src",
                                   "julia/bin"])])
    done = out / "_done.json"
    if done.exists() and json.load(open(done)).get("hash") == h:
        return {"out": str(out), "skipped": True}
    forms = _cached_forms(cfg, unit_id)
    man = splits.load_manifest(split_path(cfg, unit_id, rep))
    r = splits.roles(man, rep, fold, pool_cap=cap)
    expo = load_exposure(cfg, unit_id)
    unit = unit_cfg(cfg, unit_id)
    labels = pcfp.lemma_labels(forms, r["pool"])
    representation = str(forms["representation"].iloc[0])
    sub_cfg = json.loads(json.dumps(cfg))
    sub_cfg["selection"]["budgets"] = budgets
    seeds = SelectionSeeds.derive(int(cfg["experiment"]["master_seed"]), unit_id, rep, fold)
    sel = cfg["selection"]
    configs = selector_configs(cfg, unit_id, rep, fold, frozen, int(sel["semantic_seeds"]))
    threads = int(sel.get("julia_threads", 1))
    sub_cfg["selection"]["_ldl_settings"] = frozen
    res = run_ldl_acquisition(policy, r["core"], r["seed"], r["pool"], forms, expo, labels, _citation(cfg, unit),
                              representation, sub_cfg, seeds, out, selector_configs=configs,
                              server_factory=lambda: SelectorServer(out / "selector_server.log", threads),
                              excluded_ids=r["dev"])
    with open(done, "w") as fh:
        json.dump({"hash": h, "summary": res.summary}, fh, default=str)
    return {"out": str(out), "skipped": False}


def stage_select(cfg: dict, units=None, folds=None, policies=None) -> None:
    _require_pcfp(cfg)
    frozen_ldl_settings(cfg)                      # ldl_tune must have run (the selector uses its settings)
    argsets = []
    for unit in _units(cfg, units):
        for rep in cfg["cv"]["repetitions"]:
            for k in _folds(cfg, folds):
                for policy, cap, budgets in run_specs(cfg, policies):
                    argsets.append((cfg, unit["unit_id"], int(rep), k, policy, cap, budgets))
    # active jobs first: they take longest
    argsets.sort(key=lambda a: a[4] == "random")
    with StageRecorder("select", cfg, output_dir(cfg) / "selection") as rec:
        rec.extra["jobs"] = _parallel(_selection_job, argsets, int(cfg["selection"].get("n_procs", 1)))
        rec.extra["frozen_settings"] = frozen_ldl_settings(cfg)


# ----------------------------------------------------------------------------- LDL

def _job_queries(cfg: dict, unit: dict, expo: Dict[str, pcfp.Exposure], core_ids: List[str],
                 lemmas_csv: Path) -> pd.DataFrame:
    lem = pd.read_csv(lemmas_csv)
    selected = lem.loc[lem["role"] != "core", "lemma_id"].tolist()
    if set(lem.loc[lem["role"] == "core", "lemma_id"]) != set(core_ids):
        raise RuntimeError(f"{lemmas_csv}: core verbs differ from the split manifest")
    cit = _citation(cfg, unit)
    qc = pcfp.hidden_queries(expo, sorted(core_ids), cit).assign(item_set="core")
    qs = pcfp.hidden_queries(expo, selected, cit).assign(item_set="selected")
    return pd.concat([qc, qs], ignore_index=True)


def _ldl_jobs(cfg: dict, units=None, folds=None, policies=None) -> List[dict]:
    frozen = frozen_ldl_settings(cfg)
    jobs = []
    for unit in _units(cfg, units):
        uid = unit["unit_id"]
        forms = None
        expo = load_exposure(cfg, uid)
        for rep in cfg["cv"]["repetitions"]:
            rep = int(rep)
            man = splits.load_manifest(split_path(cfg, uid, rep))
            for k in _folds(cfg, folds):
                core = splits.roles(man, rep, k)["core"]
                for policy, cap, budgets in run_specs(cfg, policies):
                    sdir = selection_dir(cfg, uid, rep, k, policy, cap)
                    for b in budgets:
                        sample = sdir / "samples" / f"budget_{b}.csv"
                        if not sample.exists():
                            raise FileNotFoundError(f"{sample} (run stage select first)")
                        qp = queries_path(cfg, uid, rep, k, policy, cap, b)
                        gp = gold_path(cfg, uid, rep, k, policy, cap, b)
                        q = _job_queries(cfg, unit, expo, core, sdir / "samples" / f"budget_{b}_lemmas.csv")
                        qp.parent.mkdir(parents=True, exist_ok=True)
                        changed = not qp.exists() or not pd.read_csv(qp).equals(q)
                        if changed:
                            q.to_csv(qp, index=False)
                        if changed or not gp.exists():
                            forms = forms if forms is not None else load_unit_forms(cfg, uid)
                            _write_gold(forms, q, gp)
                        jobs.append(dict(train_csv=str(sample), queries_csv=str(qp),
                                         out_dir=str(ldl_dir(cfg, uid, rep, k, policy, cap, b)),
                                         unit_id=uid, repetition=rep, fold=k, overrides=dict(frozen),
                                         _gold=str(gp)))
    return jobs


def stage_ldl(cfg: dict, units=None, folds=None, policies=None) -> None:
    _require_pcfp(cfg)
    from morph_ldl.ldl.runner import run_ldl_jobs, score_mapping_jobs
    jobs = _ldl_jobs(cfg, units, folds, policies)
    clean = [{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs]
    with StageRecorder("ldl", cfg, output_dir(cfg) / "ldl") as rec:
        run_ldl_jobs(clean, cfg)                                   # gold-free prediction
        score_mapping_jobs(clean, [j["_gold"] for j in jobs], cfg)  # gold-side diagnostics, afterwards
        rec.extra["n_jobs"] = len(jobs)
        rec.extra["frozen_settings"] = frozen_ldl_settings(cfg)


# ----------------------------------------------------------------------------- evaluation

def comparisons(cfg: dict) -> List[tuple]:
    primary = int(cfg["cv"]["pool_cap"])
    out = [(p, primary, "random", primary) for p in cfg["selection"]["policies"] if p != "random"]
    for cap in cfg["cv"].get("pool_cap_sensitivity", []):
        for p in cfg["cv"]["pool_cap_sensitivity_policies"]:
            if p in cfg["selection"]["policies"]:
                out.append((p, primary, p, int(cap)))
                out.append((p, int(cap), "random", primary))
    return out


def _truthy(col: pd.Series) -> pd.Series:
    return col.astype(str).str.lower().isin(["true", "1", "1.0"])


def _composition(cfg: dict, uid: str, rep: int, k: int, policy: str, cap: int, budget: int,
                 forms: pd.DataFrame) -> dict:
    sdir = selection_dir(cfg, uid, rep, k, policy, cap)
    summ = json.load(open(sdir / "selection_summary.json"))
    b = summ["budgets"][str(budget)]
    lem = pd.read_csv(sdir / "samples" / f"budget_{budget}_lemmas.csv")
    acq = lem[lem["role"] == "acquired"]["lemma_id"]
    sel = lem[lem["role"] != "core"]["lemma_id"]
    labels = pcfp.lemma_labels(forms, sel)
    cls = pd.Series([pcfp.inflection_class(uid, labels[l]) for l in acq], dtype=str)
    lens = pd.Series([len(labels[l]) for l in acq], dtype=float)
    return {"unit_id": uid, "repetition": rep, "outer_fold": k, "policy": policy, "pool_cap": cap, "budget": budget,
            "n_core_verbs": b["n_core_verbs"], "n_selected_verbs": b["n_selected_verbs"],
            "n_forms_total": b["n_forms_total"], "n_forms_core": b["n_forms_core"],
            "n_forms_selected": b["n_forms_selected"], "k_mean_selected": b["k_mean_selected"],
            "shortfall": b["shortfall"], "rounds_used": b["rounds_used"],
            "acquired_citation_len_mean": float(lens.mean()) if len(lens) else float("nan"),
            "acquired_n_distinct_endings3": int(pd.Series([labels[l][-3:] for l in acq]).nunique()),
            **{f"acquired_class_{c}": int((cls == c).sum()) for c in sorted(set(cls))}}


def _selector_checks(cfg: dict, uid: str, rep: int, k: int, policy: str, cap: int, forms: pd.DataFrame) -> List[dict]:
    """Comprehension-side check and degeneracy diagnostics per acquisition round."""
    def spearman(a, b) -> float:       # Pearson correlation of average ranks, pairwise complete
        d = pd.DataFrame({"a": np.asarray(a, float), "b": np.asarray(b, float)}).dropna()
        if len(d) < 3:
            return float("nan")
        return float(d["a"].rank().corr(d["b"].rank()))
    sdir = selection_dir(cfg, uid, rep, k, policy, cap)
    comp_p, cells_p = sdir / "comprehension_check.csv", sdir / "cell_scores.csv"
    if not comp_p.exists():
        return []
    comp = pd.read_csv(comp_p)
    cells = pd.read_csv(cells_p, keep_default_na=False, dtype={"supports": str, "top_prediction_segments": str})
    labels = pcfp.lemma_labels(forms, comp["lemma_id"].unique())
    rows = []
    for r, sub in comp.groupby("round"):
        s0 = sub[sub["semantic_seed_idx"] == 0].set_index("lemma_id")
        avg = sub.groupby("lemma_id")[["comp_cor", "comp_rel_dist", "share_citation_cues_unseen"]].mean()
        score = s0["lemma_score"]
        lab = pd.Series({l: labels[l] for l in score.index})
        feats = {"citation_len": lab.str.len(), "comp_cor": avg["comp_cor"], "comp_rel_dist": avg["comp_rel_dist"],
                 "share_cues_unseen": avg["share_citation_cues_unseen"]}
        rec = {"unit_id": uid, "repetition": rep, "outer_fold": k, "policy": policy, "pool_cap": cap, "round": r,
               "n_candidates": len(score), "score_mean": float(score.mean()), "score_sd": float(score.std()),
               "score_cv": float(score.std() / abs(score.mean())) if score.mean() else float("nan")}
        for name, v in feats.items():
            rec[f"spearman_score_{name}"] = spearman(score.values, v.reindex(score.index).values)
        for name in ("comp_cor", "comp_rel_dist", "share_cues_unseen"):
            rec[f"spearman_{name}_citation_len"] = spearman(feats[name].reindex(score.index).values,
                                                            feats["citation_len"].values)
        # share of score variance explained by the final letters / class proxy (eta^2)
        for name, grp in (("final2", lab.str[-2:]), ("final3", lab.str[-3:]),
                          ("class", lab.map(lambda x: pcfp.inflection_class(uid, x)))):
            g = pd.DataFrame({"s": score, "g": grp})
            ss_tot = ((g["s"] - g["s"].mean()) ** 2).sum()
            ss_b = g.groupby("g")["s"].agg(lambda x: len(x) * (x.mean() - g["s"].mean()) ** 2).sum()
            rec[f"eta2_score_{name}"] = float(ss_b / ss_tot) if ss_tot > 0 else float("nan")
        # citation length + final letters jointly (OLS R^2 and adjusted R^2; the adjustment
        # matters because small ending groups inflate R^2 with few candidates)
        tot = ((score.values - score.values.mean()) ** 2).sum()
        for nf in (2, 3):
            X = pd.get_dummies(lab.str[-nf:], drop_first=True).astype(float)
            X["len"] = lab.str.len().astype(float)
            X.insert(0, "const", 1.0)
            beta, *_ = np.linalg.lstsq(X.values, score.values, rcond=None)
            resid = score.values - X.values @ beta
            r2 = float(1 - (resid ** 2).sum() / tot) if tot > 0 else float("nan")
            n, p = len(score), X.shape[1] - 1
            rec[f"r2_score_len_final{nf}"] = r2
            rec[f"adj_r2_score_len_final{nf}"] = float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if n - p - 1 > 0 else float("nan")
        cr = cells[cells["round"] == r]
        rec["top_equals_citation_rate"] = float(_truthy(cr["top_equals_citation"]).mean())
        rec["no_candidate_rate"] = float((~_truthy(cr["scored"])).mean())
        rows.append(rec)
    return rows


def stage_evaluate(cfg: dict, units=None, folds=None, policies=None) -> None:
    _require_pcfp(cfg)
    from morph_ldl.cv import bootstrap, evaluate
    out = output_dir(cfg) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with StageRecorder("evaluate", cfg, out) as rec:
        all_items, comp, mq, diag, checks = [], [], [], [], []
        for unit in _units(cfg, units):
            uid = unit["unit_id"]
            forms = load_unit_forms(cfg, uid)
            expo = load_exposure(cfg, uid)
            cit = _citation(cfg, unit)
            em = pd.read_csv(exposure_path(cfg, uid)).set_index("lemma_id")
            for rep in cfg["cv"]["repetitions"]:
                rep = int(rep)
                man = splits.load_manifest(split_path(cfg, uid, rep))
                group_of = dict(zip(man["lemma_id"], man["group_id"]))
                for k in _folds(cfg, folds):
                    core = set(splits.roles(man, rep, k)["core"])
                    for policy, cap, budgets in run_specs(cfg, policies):
                        for b in budgets:
                            meta = dict(unit_id=uid, repetition=rep, outer_fold=k, policy=policy,
                                        pool_cap=cap, budget=b)
                            ld = ldl_dir(cfg, uid, rep, k, policy, cap, b)
                            q = pd.read_csv(queries_path(cfg, uid, rep, k, policy, cap, b))
                            pred = pd.read_csv(ld / "predictions.csv", keep_default_na=False)
                            if not pred[["lemma_id", "target_cell"]].equals(q[["lemma_id", "target_cell"]]):
                                raise RuntimeError(f"{ld}: predictions do not match the queries")
                            if set(q.loc[q["item_set"] == "core", "lemma_id"]) != core:
                                raise RuntimeError(f"{ld}: core items do not cover the core verbs")
                            lids = set(q["lemma_id"])
                            gold = evaluate.gold_table(forms[forms["lemma_id"].isin(lids)])
                            items = evaluate.score_items(pred, gold, {**meta, "model": "ldl"}, group_of)
                            items["item_set"] = q["item_set"].values
                            items["k_shown"] = items["lemma_id"].map(em["k"]).astype(int)
                            items["citation_shown"] = items["lemma_id"].map(em["citation_shown"]).astype(bool)
                            items["n_test_cells"] = items["lemma_id"].map(em["n_test_cells"]).astype(int)
                            shown = _shown_segments(forms, expo, lids)
                            labs = pcfp.lemma_labels(forms, lids)
                            cit_seg = {l: pcfp.citation_segments(labs[l], str(forms["representation"].iloc[0]))
                                       for l in lids}
                            items["pred_equals_shown_form"] = [p != "" and p in shown.get(l, set())
                                                               for l, p in zip(items["lemma_id"], items["prediction"])]
                            items["pred_equals_citation"] = [p != "" and p == cit_seg[l]
                                                             for l, p in zip(items["lemma_id"], items["prediction"])]
                            all_items.append(items)
                            m = pd.read_csv(ld / "mapping_quality.csv")
                            m = m.merge(q, on=["lemma_id", "target_cell"], how="left")
                            for iset, mm in m.groupby("item_set"):
                                mq.append({**meta, "item_set": iset, "n_items": len(mm),
                                           "chat_gold_cor_mean": mm["chat_gold_cor"].mean(),
                                           "gold_reachable_rate": _truthy(mm["gold_reachable"]).mean(),
                                           "gold_in_top_candidates_rate": _truthy(mm["gold_in_top_candidates"]).mean(),
                                           "gold_cues_outside_inventory_mean": mm["n_gold_cues_outside_inventory"].mean()})
                            d = json.load(open(ld / "diagnostics.json"))
                            diag.append({**meta, **{kk: v for kk, v in d.items() if not isinstance(v, (dict, list))},
                                         "unseen_target_features": "|".join(d.get("unseen_target_features", []))})
                            comp.append(_composition(cfg, uid, rep, k, policy, cap, b, forms))
                        if policy != "random":
                            checks += _selector_checks(cfg, uid, rep, k, policy, cap, forms)
        items = pd.concat(all_items, ignore_index=True)
        items.to_csv(out / "item_predictions.csv", index=False)
        design = ["unit_id", "item_set", "policy", "pool_cap", "budget", "model"]
        evaluate.summarize(items, design).to_csv(out / "summary_point.csv", index=False)
        evaluate.summarize(items, design + ["target_cell"]).to_csv(out / "summary_by_cell.csv", index=False)
        evaluate.summarize(items, design + ["k_shown"]).to_csv(out / "summary_by_k.csv", index=False)
        copy = (items.groupby(design).agg(copy_shown_form_rate=("pred_equals_shown_form", "mean"),
                                          copy_citation_rate=("pred_equals_citation", "mean"),
                                          n_items=("correct", "size")).reset_index())
        copy.to_csv(out / "copy_rates.csv", index=False)
        per_verb = (items[items["item_set"] == "core"].drop_duplicates(["unit_id", "repetition", "outer_fold", "lemma_id"])
                    .groupby(["unit_id", "n_test_cells"]).size().rename("n_core_verbs").reset_index())
        per_verb.to_csv(out / "test_cells_per_verb.csv", index=False)
        per, spread = bootstrap.fold_variability(items, design)
        per.to_csv(out / "per_fold.csv", index=False)
        spread.to_csv(out / "fold_variability.csv", index=False)
        pd.DataFrame(comp).to_csv(out / "sample_composition.csv", index=False)
        pd.DataFrame(mq).to_csv(out / "ldl_mapping_quality.csv", index=False)
        pd.DataFrame(diag).to_csv(out / "ldl_diagnostics.csv", index=False)
        pd.DataFrame(checks).to_csv(out / "selector_checks.csv", index=False)
        rec.extra["n_items"] = len(items)
        print(evaluate.summarize(items, design).to_string(index=False))
        print(copy.to_string(index=False))


def stage_outcomes(cfg: dict, units=None, folds=None, policies=None) -> None:
    _require_pcfp(cfg)
    from morph_ldl.cv import outcomes
    from morph_ldl.data import build_population_links
    ev = output_dir(cfg) / "eval"
    out = output_dir(cfg) / "outcomes"
    out.mkdir(parents=True, exist_ok=True)
    with StageRecorder("outcomes", cfg, out, inputs=[ev / "item_predictions.csv"]) as rec:
        items = pd.read_csv(ev / "item_predictions.csv", keep_default_na=False)
        for c in ("correct", "pred_equals_shown_form", "pred_equals_citation"):
            items[c] = items[c].astype(str).str.lower().eq("true")
        meta = {}
        frozen = frozen_ldl_settings(cfg)
        for unit in _units(cfg, units):
            uid = unit["unit_id"]
            f = load_unit_forms(cfg, uid).iloc[0]
            m = {k: f[k] for k in ("variety_id", "iso639_3", "glottocode", "pos", "representation", "resource_version")}
            em = pd.read_csv(exposure_path(cfg, uid))
            m.update({f"ldl_{k}": v for k, v in frozen.items()})
            m.update({"ldl_decoder": cfg["ldl"]["decoder"], "selector": cfg["selection"]["selector"],
                      "selector_candidate_scoring": cfg["selection"]["candidate_scoring"],
                      "selector_semantic_seeds": int(cfg["selection"]["semantic_seeds"]),
                      "core_size": int(cfg["cv"]["core_size"]),
                      "eligible_cells": "|".join(unit_cells(unit, cfg)["eligible"]),
                      "k_distribution_observed": json.dumps({int(k): int(v) for k, v in
                                                             em["k"].value_counts().sort_index().items()})})
            meta[uid] = m
        tab = outcomes.outcome_table(items, cfg, meta)
        tab.to_csv(out / "ldl_outcomes.csv", index=False)
        pt = outcomes.paired_table(items, cfg, comparisons(cfg))
        pt.to_csv(out / "paired_differences.csv", index=False)
        links = build_population_links([u["unit_id"] for u in _units(cfg, units)], cfg)
        links.to_csv(out / "population_links.csv", index=False)
        rec.extra.update({"n_outcome_rows": len(tab), "n_links": len(links)})
        cols = ["unit_id", "item_set", "policy", "pool_cap", "budget", "accuracy_micro",
                "accuracy_micro_ci_low", "accuracy_micro_ci_high", "edit_distance_micro"]
        print(tab[cols].to_string(index=False))
        if len(pt):
            print(pt[pt["statistic"].isin(["correct_micro", "edit_distance_micro"])][
                ["unit_id", "model", "comparison", "statistic", "difference", "ci_low", "ci_high"]].to_string(index=False))


def stage_typology(cfg: dict, units=None, folds=None, policies=None) -> None:
    from morph_ldl.typology.stage import stage_typology as run
    run(cfg, units=units, folds=folds, policies=policies)


# ----------------------------------------------------------------------------- artifact audit

def _pairs(df: pd.DataFrame) -> set:
    v0 = df[df["variant_idx"].astype(int) == 0] if "variant_idx" in df else df
    return set(zip(v0["lemma_id"], v0["cell_norm"]))


def audit_unit(cfg: dict, unit: dict, folds=None, policies=None) -> tuple:
    """Independent checks of every written PCFP artifact of one unit. Returns (checks, problems)."""
    from morph_ldl.selection.ldl_acquisition import CANDIDATE_COLUMNS
    problems, checks = [], []
    uid = unit["unit_id"]
    uc = unit_cells(unit, cfg)
    cit = uc["citation"]
    master = int(cfg["experiment"]["master_seed"])
    em = pd.read_csv(exposure_path(cfg, uid))
    try:
        pcfp.validate_exposure_manifest(em, uc["eligible"], cit, master, uid, int(cfg["task"]["exposure"]["max_shown"]))
    except pcfp.ExposureError as err:
        problems.append(f"{uid}: exposure manifest: {err}")
    expo = pcfp.exposure_from_frame(em)
    chosen = output_dir(cfg) / "ldl_tune" / "chosen_settings.json"
    tune_mtime = chosen.stat().st_mtime if chosen.exists() else None
    frozen = frozen_ldl_settings(cfg) if chosen.exists() else None
    shown_pairs = {(l, c) for l, e in expo.items() for c in e.shown}
    forms = None
    for rep in cfg["cv"]["repetitions"]:
        rep = int(rep)
        man = splits.load_manifest(split_path(cfg, uid, rep))
        for k in _folds(cfg, folds):
            full = splits.roles(man, rep, k)
            core = set(full["core"])
            core_hidden = {(l, c) for l in core for c in expo[l].hidden}
            fm = man[(man["repetition"] == rep) & (man["outer_fold"] == k)]
            group_of = dict(zip(fm["lemma_id"], fm["group_id"]))
            core_groups = {group_of[l] for l in core}
            seeds_seen = {}
            for policy, cap, budgets in run_specs(cfg, policies):
                sdir = selection_dir(cfg, uid, rep, k, policy, cap)
                tag = f"{uid} r{rep} f{k} {run_tag(policy, cap)}"
                if not (sdir / "order.csv").exists():
                    checks.append((uid, rep, k, policy, cap, "not_run"))
                    continue
                order = pd.read_csv(sdir / "order.csv")
                seeds_seen[(policy, cap)] = tuple(sorted(order.loc[order["round"] == 0, "lemma_id"]))
                pool_allowed = set(splits.roles(man, rep, k, pool_cap=cap)["pool"])
                if set(order.loc[order["round"] == 0, "lemma_id"]) != set(full["seed"]):
                    problems.append(f"{tag}: seed verbs differ from the manifest")
                if not set(order.loc[order["round"] > 0, "lemma_id"]) <= pool_allowed:
                    problems.append(f"{tag}: acquired verbs outside the pool")
                if {group_of.get(l) for l in order["lemma_id"]} & core_groups:
                    problems.append(f"{tag}: a selected verb shares a group with a core verb")
                rev = pd.read_csv(sdir / "oracle_reveals.csv")
                if not set(rev["lemma_id"]) <= set(order["lemma_id"]) | core:
                    problems.append(f"{tag}: oracle revealed a verb that was never acquired")
                # selector rounds: training = core + acquired-before-round, shown cells only;
                # candidates = remaining pool, citation label only
                for rdir in sorted((sdir / "rounds").glob("r*")) if (sdir / "rounds").exists() else []:
                    r = int(rdir.name[1:])
                    before = set(order.loc[order["round"] < r, "lemma_id"])
                    tr = pd.read_csv(rdir / "train.csv", usecols=["lemma_id", "cell_norm", "variant_idx"])
                    if set(tr["lemma_id"]) != core | before:
                        problems.append(f"{tag} round {r}: selector training verbs != core + acquired before the round")
                    if not _pairs(tr) <= shown_pairs or _pairs(tr) & core_hidden:
                        problems.append(f"{tag} round {r}: selector training contains a hidden cell")
                    cd = pd.read_csv(rdir / "candidates.csv", dtype=str, keep_default_na=False)
                    if list(cd.columns) != CANDIDATE_COLUMNS:
                        problems.append(f"{tag} round {r}: candidate columns {list(cd.columns)}")
                    if set(cd["lemma_id"]) != pool_allowed - before:
                        problems.append(f"{tag} round {r}: candidates != pool minus acquired")
                    forms = forms if forms is not None else load_unit_forms(cfg, uid)
                    labs = pcfp.lemma_labels(forms, cd["lemma_id"])
                    rep_ = str(forms["representation"].iloc[0])
                    if any(s != pcfp.citation_segments(labs[l], rep_) for l, s in zip(cd["lemma_id"], cd["citation_segments"])):
                        problems.append(f"{tag} round {r}: candidate citation segments are not the lemma label")
                    if any(tuple(sc.split(pcfp.SEP)) != expo[l].shown for l, sc in zip(cd["lemma_id"], cd["shown_cells"])):
                        problems.append(f"{tag} round {r}: candidate shown cells differ from the exposure manifest")
                    checks.append((uid, rep, k, policy, cap, f"round{r}"))
                summ = json.load(open(sdir / "selection_summary.json"))
                if policy != "random":
                    if summ.get("ldl_settings") != frozen:
                        problems.append(f"{tag}: selector settings {summ.get('ldl_settings')} != frozen {frozen}")
                    if tune_mtime is None or (sdir / "order.csv").stat().st_mtime < tune_mtime:
                        problems.append(f"{tag}: selection ran before ldl_tune finished")
                for b in budgets:
                    smp = pd.read_csv(sdir / "samples" / f"budget_{b}.csv",
                                      usecols=["lemma_id", "cell_norm", "variant_idx", "is_missing"])
                    lem = pd.read_csv(sdir / "samples" / f"budget_{b}_lemmas.csv")
                    sel_ids = order["lemma_id"].iloc[:b].tolist()
                    if set(smp["lemma_id"]) != core | set(sel_ids):
                        problems.append(f"{tag} budget {b}: sample verbs != core + first {b} selected")
                    if len(sel_ids) != b or (lem["role"] != "core").sum() != b:
                        problems.append(f"{tag} budget {b}: {len(sel_ids)} selected verbs != budget")
                    pairs = _pairs(smp)
                    if pairs & core_hidden:
                        problems.append(f"{tag} budget {b}: a hidden cell of a core verb is in the training sample")
                    want = {(l, c) for l in core | set(sel_ids) for c in expo[l].shown}
                    if pairs != want:
                        problems.append(f"{tag} budget {b}: sample cells differ from the exposure draw")
                    if {l for l, _ in pairs} & core != core:
                        problems.append(f"{tag} budget {b}: a core verb has no shown form in the sample")
                    if summ["budgets"][str(b)]["n_forms_total"] != len(pairs):
                        problems.append(f"{tag} budget {b}: form count != selection summary")
                    qp = queries_path(cfg, uid, rep, k, policy, cap, b)
                    if not qp.exists() and (ldl_dir(cfg, uid, rep, k, policy, cap, b) / "predictions.csv").exists():
                        problems.append(f"{tag} budget {b}: LDL predictions exist without a query file")
                    if qp.exists():
                        q = pd.read_csv(qp)
                        if not set(q.columns) <= {"lemma_id", "target_cell", "item_set"}:
                            problems.append(f"{tag} budget {b}: query columns {list(q.columns)}")
                        qc = q[q["item_set"] == "core"]
                        if set(zip(qc["lemma_id"], qc["target_cell"])) != {p for p in core_hidden if p[1] != cit}:
                            problems.append(f"{tag} budget {b}: core queries != hidden non-citation cells of core verbs")
                        qs = q[q["item_set"] == "selected"]
                        want_sel = {(l, c) for l in sel_ids for c in expo[l].hidden if c != cit}
                        if set(zip(qs["lemma_id"], qs["target_cell"])) != want_sel:
                            problems.append(f"{tag} budget {b}: selected queries != hidden non-citation cells of selected verbs")
                        if set(zip(q["lemma_id"], q["target_cell"])) & shown_pairs:
                            problems.append(f"{tag} budget {b}: a query asks for a shown cell")
                    checks.append((uid, rep, k, policy, cap, b))
            if len(set(seeds_seen.values())) > 1:
                problems.append(f"{uid} r{rep} f{k}: seed sets differ across policies")
    # auxiliary (tuning) verbs: outside every inventory group; tuning files shown-only
    inv_groups = set()
    for rep in cfg["cv"]["repetitions"]:
        inv_groups |= set(splits.load_manifest(split_path(cfg, uid, int(rep)))["group_id"])
    aux = pd.read_csv(aux_path(cfg, uid))
    if set(aux["group_id"]) & inv_groups:
        problems.append(f"{uid}: auxiliary verbs share groups with the inventory")
    tdir = output_dir(cfg) / "ldl_tune" / uid
    if (tdir / "train.csv").exists():
        tr = pd.read_csv(tdir / "train.csv", usecols=["lemma_id", "cell_norm", "variant_idx"])
        tune_ids = set(aux.loc[aux["aux_role"].isin(["tune_core", "tune_extra"]), "lemma_id"])
        if not set(tr["lemma_id"]) <= tune_ids or not _pairs(tr) <= shown_pairs:
            problems.append(f"{uid}: tuning training file uses non-auxiliary verbs or hidden cells")
        tq = pd.read_csv(tdir / "tune_queries.csv")
        if set(zip(tq["lemma_id"], tq["target_cell"])) & shown_pairs:
            problems.append(f"{uid}: tuning queries ask for shown cells")
        checks.append((uid, "tune"))
    return checks, problems


def stage_audit(cfg: dict, units=None, folds=None, policies=None) -> None:
    """Check written artifacts against the split and exposure manifests (independent of component code)."""
    _require_pcfp(cfg)
    problems, checks = [], []
    for unit in _units(cfg, units):
        c, p = audit_unit(cfg, unit, folds, policies)
        checks += c; problems += p
    typ = output_dir(cfg) / "typology"
    if typ.exists():
        from morph_ldl.typology.stage import audit_typology
        c, p = audit_typology(cfg)
        checks += [("typology", x) for x in c]; problems += [f"typology: {x}" for x in p]
    out = output_dir(cfg) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "artifact_audit.json", "w") as fh:
        json.dump({"n_checks": len(checks), "problems": problems}, fh, indent=2, default=str)
    print(f"audit: {len(checks)} artifact sets checked, {len(problems)} problems")
    for p in problems:
        print("  PROBLEM", p)
    if problems:
        raise RuntimeError("artifact audit failed")
