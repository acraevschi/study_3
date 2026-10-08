"""Are the GeLaTo languages "missing from Grambank" really absent, or coded under another
Glottocode or level? (reads outputs/pcfp_v1/typology and the pinned external sources;
writes grambank_missing_check.csv here)

For every language in outputs/pcfp_v1/typology/missing_from_grambank.csv, search Grambank
v1.0.3 for:
  * iso          a Grambank language with the same ISO 639-3 code;
  * descendant   a Grambank entry below the language in Glottolog (dialects; Grambank's
                 own lineage column or Language_level_ID);
  * ancestor     a Grambank entry that is an ancestor of the language (a group or family);
  * name         a Grambank entry whose name contains the language's Glottolog name.

Run: .venv/bin/python analyses/pcfp_v1_2026_10_08/grambank_missing_check.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    gb = pd.read_csv(ROOT / "external/grambank/cldf/languages.csv", keep_default_na=False, dtype=str)
    gl = pd.read_csv(ROOT / "external/glottolog-cldf/cldf/languages.csv", keep_default_na=False,
                     dtype=str).set_index("ID")
    vals = pd.read_csv(ROOT / "external/glottolog-cldf/cldf/values.csv", keep_default_na=False, dtype=str)
    lineage = vals[vals["Parameter_ID"] == "classification"].set_index("Language_ID")["Value"].to_dict()
    missing = pd.read_csv(ROOT / "outputs/pcfp_v1/typology/missing_from_grambank.csv", keep_default_na=False,
                          dtype=str)
    gb_ids = set(gb["ID"])
    gb_iso = {r.ISO639P3code: r.ID for r in gb.itertuples() if r.ISO639P3code}
    gb_lineage = gb["lineage"].str.split("/")
    rows = []
    for r in missing.itertuples():
        g = r.glottocode
        iso = gl.loc[g, "ISO639P3code"] if g in gl.index else ""
        hits = {"iso": [], "descendant": [], "ancestor": [], "name": []}
        if iso and iso in gb_iso:
            hits["iso"].append(gb_iso[iso])
        hits["ancestor"] = [f"{a} ({gb.set_index('ID').loc[a, 'Name']}, {gb.set_index('ID').loc[a, 'level']})"
                            for a in lineage.get(g, "").split("/") if a in gb_ids]
        desc = gb[gb_lineage.map(lambda l: g in l) | (gb["Language_level_ID"] == g)]
        hits["descendant"] = [f"{d.ID} ({d.Name}, {d.level})" for d in desc.itertuples()]
        name = r.name.lower().split(" (")[0]
        hits["name"] = [f"{d.ID} ({d.Name})" for d in gb[gb["Name"].str.lower().str.contains(re.escape(name))].itertuples()
                        if d.ID != g]
        rows.append({"glottocode": g, "name": r.name, "iso639_3": iso, "n_individuals_total": r.n_individuals_total,
                     **{f"grambank_{k}": "; ".join(v) for k, v in hits.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "grambank_missing_check.csv", index=False)
    any_hit = df[[c for c in df.columns if c.startswith("grambank_")]].ne("").any(axis=1)
    print(f"{len(df)} languages checked; {int(any_hit.sum())} with any candidate")
    print(df[any_hit].to_string(index=False))


if __name__ == "__main__":
    main()
