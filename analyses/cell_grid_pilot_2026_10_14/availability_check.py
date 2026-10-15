"""Phase-1 availability check: which verbs have every source cell the draft grid needs.

This is NOT a mapping. For each candidate dataset it lists the source cells that a
plausible mapping would draw on (present, past, future-or-substitute, imperative, and
the non-finite forms the main-verb rule needs) and counts verbs with all of them
non-missing. The final source cells are fixed only in Phase 2.

Run from the repository root:
    .venv/bin/python analyses/cell_grid_pilot_2026_10_14/availability_check.py
"""
import csv
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).with_name("availability.csv")
P = ["1;SG", "2;SG", "3;SG", "1;PL", "2;PL", "3;PL"]


def wide(path):
    rows = list(csv.reader(open(path)))
    h = rows[0]
    out = {}
    for r in rows[1:]:
        v = {}
        for c, x in zip(h[1:], r[1:]):  # some MGN headers repeat a cell; keep a non-NA value
            if v.get(c, "NA") == "NA":
                v[c] = x
        out[r[0]] = v
    return h, out


def unimorph(path):
    d = defaultdict(dict)
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            d[p[0]][p[2]] = p[1]
    return d


def english(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        d[r["lexeme"]][r["cell"]] = r["form"]
    return d


SETS = {
    "ita (MGN ita-v)": ("mgn", "mgn_data/data/ita-v.csv",
        [f"IND;PRS;{p}" for p in P] + [f"IND;PST;{p};PFV" for p in P]
        + [f"IND;FUT;{p}" for p in P] + ["POS;IMP;2;SG", "POS;IMP;2;PL"]),
    "ita (UniMorph ita, + past participle)": ("um", "external/unimorph-ita/ita",
        [f"V;IND;PRS;{p}" for p in P] + ["V.PTCP;PST"]
        + [f"V;IND;FUT;{p}" for p in P] + ["V;POS;IMP;2;SG", "V;POS;IMP;2;PL"]),
    "fin (MGN fin-v)": ("mgn", "mgn_data/data/fin-v.csv",
        [f"ACT;PRS;POS;IND;{p}" for p in P] + [f"ACT;PST;POS;IND;{p}" for p in P]
        + ["ACT;PRS;POS;IMP;2;SG", "ACT;PRS;POS;IMP;2;PL"]),
    "eng (MGN data-custom english-v)": ("eng", "mgn_data/data-custom/english-v.csv",
        ["inf", "pres1s", "pres3s", "presothers", "past13", "pastnot13"]),
    "hin (MGN hin-v)": ("mgn", "mgn_data/data/hin-v.csv",
        ["1;SG;HAB;PRS;MASC", "2;SG;HAB;PRS;MASC", "1;PL;HAB;PRS;MASC", "2;PL;HAB;PRS;MASC",
         "V.PTCP;MASC;SG;PFV", "V.PTCP;MASC;PL;PFV", "V.PTCP;MASC;SG;IPFV", "V.PTCP;MASC;PL;IPFV",
         "2;SG;IMP;INFM", "2;PL;IMP;FORM"]),
    "hye (MGN hye-v)": ("mgn", "mgn_data/data/hye-v.csv",
        [f"{p};PRS;IND" for p in P] + [f"{p};PST;IND" for p in P]
        + [f"{p};FUT;IND" for p in P] + ["2;SG;IMP", "2;PL;IMP", "V.CONV;IPFV", "V.CONV;FUT1"]),
    "hbs (MGN hbs-v)": ("mgn", "mgn_data/data/hbs-v.csv",
        [f"PRS;{p}" for p in P] + [f"PST;{p}" for p in P]
        + [f"FUT;{p}" for p in P if p != "3;PL"] + ["IMP;2;SG", "IMP;2;PL"]),
    "bul (MGN bul-v; 1SG PRS = lexeme)": ("mgn", "mgn_data/data/bul-v.csv",
        [f"IND;PRS;{p}" for p in P if p != "1;SG"]
        + [f"IND;PST;{p}" for p in P if p != "3;SG"] + ["IMP;2;SG", "IMP;2;PL"]),
    "ces (MGN ces-v)": ("mgn", "mgn_data/data/ces-v.csv",
        [f"IND;PRS;{p}" for p in P] + ["IND;PST;3;SG;MASC", "IND;PST;3;PL;MASC;ANIM"]
        + [f"IND;FUT;{p}" for p in P] + ["IMP;2;SG", "IMP;2;PL"]),
    "slv (MGN slv-v)": ("mgn", "mgn_data/data/slv-v.csv",
        [f"IND;PRS;{p}" for p in P] + ["IND;PST", "IND;FUT", "IMP;2;SG", "IMP;2;PL"]),
}

rows = []
for name, (kind, path, need) in SETS.items():
    if kind == "mgn":
        _, d = wide(path)
    elif kind == "um":
        d = unimorph(path)
    else:
        d = english(path)
    present = lambda v, c: v.get(c, "NA") not in ("NA", "", None)
    complete = [l for l, v in d.items() if all(present(v, c) for c in need)]
    single = [l for l in complete if " " not in l]
    missing = [c for c in need if not any(c in v for v in d.values())]
    # english-v forms are space-separated phoneme strings, so its share is not meaningful
    mw = 0 if kind == "eng" else sum(" " in d[l][c] for l in complete for c in need)
    rows.append({"dataset": name, "lemmas": len(d), "complete": len(complete),
                 "complete_single_word_lemma": len(single),
                 "multiword_share_of_needed_forms": round(mw / max(1, len(complete) * len(need)), 2),
                 "absent_source_cells": ";".join(missing), "n_needed_cells": len(need)})
    print(rows[-1])

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
