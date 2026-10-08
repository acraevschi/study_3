"""Bounded real-data smoke run on a TEMPORARY Italian-verb fixture (not a pilot run).

    .venv/bin/python -m morph_ldl.selection.smoke [--policies low_confidence ...] [--device cpu]

Reads mgn_data/data/ita-v.csv read-only via ``fixtures.mgn_wide_to_forms`` (fixture
reader, single-lemma groups), deals a seeded fixture split (seed 20 / dev 40 / pool 200),
and runs acquisition to budget 60 in batches of 20 with the pilot selector settings.
Outputs go to outputs/scratch_selection/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from morph_ldl.config import PIPELINE_ROOT, load_config
from morph_ldl.selection.acquisition import SelectionSeeds, SelectionTask, policy_dir, run_acquisition
from morph_ldl.selection.fixtures import eligible, fixture_roles, mgn_wide_to_forms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PIPELINE_ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--unit", default="ita.V.orth.mgn")
    ap.add_argument("--file", default="ita-v.csv")
    ap.add_argument("--policies", nargs="+", default=["low_confidence", "high_entropy", "random"])
    ap.add_argument("--seed-size", type=int, default=20)
    ap.add_argument("--dev-size", type=int, default=40)
    ap.add_argument("--pool-size", type=int, default=200)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--selector", default="{}", help="JSON overrides for the selector section")
    ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()
    over = {"selection": {"budgets": [a.budget], "batch_size": a.batch}, "selector": json.loads(a.selector)}
    if a.device:
        over["selector"]["device"] = a.device
    if a.threads:
        over["selector"]["num_threads"] = a.threads
    cfg = load_config(a.config, over)
    task = SelectionTask.from_config(cfg, a.unit)
    unit = next(u for u in cfg["units"] if u["unit_id"] == a.unit)
    t0 = time.perf_counter()
    forms = mgn_wide_to_forms(PIPELINE_ROOT / cfg["paths"]["mgn_data"] / a.file, a.unit, unit["resource_id"],
                              a.unit.split(".")[0], cells=[task.source_cell, *task.panel_cells])
    ids = eligible(forms, task.source_cell, task.panel_cells)
    roles = fixture_roles(ids, {"dev": a.dev_size, "seed": a.seed_size, "pool": a.pool_size}, seed=12345)
    print(f"fixture: {len(ids)} eligible lemmas, read in {time.perf_counter() - t0:.1f}s")
    seeds = SelectionSeeds.derive(cfg["experiment"]["master_seed"], a.unit, 0, 0)
    root = PIPELINE_ROOT / "outputs" / "scratch_selection" / a.tag
    for pol in a.policies:
        res = run_acquisition(pol, roles["seed"], roles["pool"], roles["dev"], forms, task, cfg, seeds,
                              policy_dir(root, a.unit, 0, 0, pol), log=print, final_fit=(pol != "random"))
        print(json.dumps({k: res.summary[k] for k in ("policy", "runtime_s", "train_runtime_total_s",
                                                       "score_runtime_total_s")}))


if __name__ == "__main__":
    main()
