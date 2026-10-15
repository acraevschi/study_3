"""Per-language summary of the pilot mappings, read from the check reports.

Run from the repository root after `gridtools.py check <iso>` for every language:
    .venv/bin/python analyses/cell_grid_pilot_2026_10_14/summarize.py
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = []
for iso in ["ita", "fin", "eng", "hye", "hbs"]:
    txt = (HERE / "mappings" / f"{iso}_check.md").read_text()
    s = json.loads(re.search(r"```json\n(.*?)\n```", txt, re.S).group(1))
    rc = s["rule_counts"]
    rows.append({
        "iso": iso, "lemmas_in_data": s["lemmas_in_data"], "lemmas_in_scope": s["lemmas_in_scope"],
        "complete_grids": s["complete_grids"],
        "data_gap_cells": s["empty_cells"].get("data_gap", 0),
        "defective_cells": s["empty_cells"].get("defective", 0),
        "n_classes_declared": s["n_classes"],
        "formally_distinct_cells_declared": s["n_formally_distinct_cells"],
        "aux_carried_cells": s["aux_carried_cells"],
        "rule_source": rc.get("source", 0), "rule_syncretism": rc.get("syncretism", 0),
        "rule_main_verb": rc.get("main_verb", 0), "rule_restored": rc.get("restored", 0),
        "empirical_mean_distinct_forms": s["empirical_mean_distinct_forms"],
        "emp_mean_classes": s["emp_mean_classes"],
    })
with open(HERE / "pilot_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
