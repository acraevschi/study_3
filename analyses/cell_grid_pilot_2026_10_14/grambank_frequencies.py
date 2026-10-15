"""Cross-linguistic frequency of the verbal categories behind the cell grid.

Reads pinned Grambank v1.0.3 (external/grambank/cldf). Language-level rows only.
For each feature: n coded (0/1), raw share of 1, and family-weighted share (mean of
per-family shares; isolates count as their own family). No ancestry data is read.

Run from the repository root:
    .venv/bin/python analyses/cell_grid_pilot_2026_10_14/grambank_frequencies.py
"""
import csv
from collections import defaultdict
from pathlib import Path

GB = Path("external/grambank/cldf")
OUT = Path(__file__).with_name("grambank_frequencies.csv")

FEATURES = {
    "GB082": "present tense marked on verb",
    "GB083": "past tense marked on verb",
    "GB084": "future tense marked on verb",
    "GB086": "perfective/imperfective on verb",
    "GB312": "mood marked on verb",
    "GB089": "S indexed by suffix/enclitic",
    "GB090": "S indexed by prefix/proclitic",
    "GB091": "A indexed by suffix/enclitic",
    "GB092": "A indexed by prefix/proclitic",
    "GB093": "P indexed by suffix/enclitic",
    "GB094": "P indexed by prefix/proclitic",
    "GB028": "inclusive/exclusive (pronouns or indexing)",
    "GB031": "dual for all persons (pronouns or indexing)",
    "GB121": "tense by inflecting auxiliary",
    "GB120": "aspect by inflecting auxiliary",
    "GB119": "mood by inflecting auxiliary",
    "GB521": "tense by non-inflecting particle",
    "GB400": "person neutralised in some TAM/negation",
}
# derived ORs (GBI-style): any S indexing; any A or P indexing
DERIVED = {
    "S_index_any": ["GB089", "GB090"],
    "P_index_any": ["GB093", "GB094"],
    "tense_any": ["GB082", "GB083", "GB084"],
}

langs = {r["ID"]: r for r in csv.DictReader(open(GB / "languages.csv"))}
vals = defaultdict(dict)
for r in csv.DictReader(open(GB / "values.csv")):
    if r["Parameter_ID"] in FEATURES and r["Value"] in ("0", "1"):
        vals[r["Language_ID"]][r["Parameter_ID"]] = int(r["Value"])

def family(lid):
    fam = langs.get(lid, {}).get("Family_level_ID") or ""
    return fam or f"isolate:{lid}"

def summarise(name, getter):
    per_fam = defaultdict(list)
    for lid, v in vals.items():
        x = getter(v)
        if x is not None:
            per_fam[family(lid)].append(x)
    xs = [x for f in per_fam.values() for x in f]
    raw = sum(xs) / len(xs)
    fam = sum(sum(f) / len(f) for f in per_fam.values()) / len(per_fam)
    return {"feature": name, "n_coded": len(xs), "n_families": len(per_fam),
            "share_present": round(raw, 3), "family_weighted_share": round(fam, 3)}

rows = []
for fid, label in FEATURES.items():
    r = summarise(fid, lambda v, f=fid: v.get(f))
    r["label"] = label
    rows.append(r)

def any_of(fs):
    def g(v):
        got = [v[f] for f in fs if f in v]
        if any(got):
            return 1
        return 0 if len(got) == len(fs) else None
    return g

for name, fs in DERIVED.items():
    r = summarise(name, any_of(fs))
    r["label"] = "OR of " + "+".join(fs)
    rows.append(r)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["feature", "label", "n_coded", "n_families",
                                       "share_present", "family_weighted_share"])
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f'{r["feature"]:12s} {r["share_present"]:.3f} {r["family_weighted_share"]:.3f} n={r["n_coded"]:4d}  {r["label"]}')
