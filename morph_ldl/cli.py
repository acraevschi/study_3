"""Single entry point: ``python -m morph_ldl.cli <stage> --config configs/pilot.yaml``.

Stages (each independently runnable; ``all`` runs them in order):

  data       ingest resources, registry, identifier crosswalk, eligibility, GeLaTo crosswalk
  splits     inventory, grouped outer folds and inner roles -> split manifests
  ldl_tune   choose LDL cue n-gram and inflection SD on inner-dev lemmas, then freeze
  select     active / random acquisition inside every outer fold -> samples and logs
  ldl        fit JudiLing on every exported sample; predict outer-test queries
  selector   (optional) selector accuracy on the same outer-test queries
  evaluate   score items, bootstrap intervals, paired differences
  outcomes   analysis-ready outcome table and population linkage table
  audit      check written samples, anchors and queries against the split manifests
  all        every stage above, in order
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from morph_ldl.config import load_config

STAGES = ["data", "splits", "ldl_tune", "select", "ldl", "selector", "evaluate", "outcomes", "audit"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--units", nargs="*", help="restrict to these unit_ids")
    ap.add_argument("--folds", nargs="*", type=int, help="restrict to these outer folds")
    ap.add_argument("--policies", nargs="*", help="restrict to these policies")
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=json",
                    help="override config values, e.g. cv.n_folds=2")
    args = ap.parse_args(argv)

    overrides: dict = {}
    for item in args.set:
        key, val = item.split("=", 1)
        cur = overrides
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = json.loads(val)
    cfg = load_config(args.config, overrides)

    from morph_ldl.cv import pipeline  # imported late: heavy deps
    sel = dict(units=args.units, folds=args.folds, policies=args.policies)
    stages = STAGES if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"== stage {st} ({cfg['experiment']['id']}, config {cfg['_config_hash']})", flush=True)
        getattr(pipeline, f"stage_{st}")(cfg, **sel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
