"""Write the first version of morph_ldl/data/gelato_link_review.csv (run once, 2026-10-20).

Rows:
  * the 12 entries of morph_ldl/data/gelato_review.yaml, with their decisions and
    confirmations unchanged (the YAML becomes an archive);
  * the Yukaghir swap: two relink proposals, for a person to confirm;
  * doubt labels for the sensitivity analyses (user decision 2026-10-20):
      diaspora    Graff et al. mapping Location "migrants_collected_in_<place>"
      wrong_code  shift_status "wrong_code" in the source check of 2026-10-16
                  (analyses/gelato_review_2026_10_16/provenance_check.csv)
    Only populations with a main-analysis link to a main-set language get a row (with
    --prune-with); the others would not change any analysis.
Evidence columns are left empty: `python -m morph_ldl.typology.link_review` fills them and
adds the group cases.

  .venv/bin/python analyses/gelato_review_2026_10_20/seed_sheet.py --links <links.csv> [--prune-with <analysis_links.csv>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from morph_ldl.data import gelato
from morph_ldl.typology import link_review as lr

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / ("analyses/gelato_feasibility_2026_10_01/sources/zenodo_15263706/"
                  "geneticAdmixture-linguisticDiffusion/input/GeLaTo-population-glottocode-mapping.csv")
PROVENANCE = ROOT / "analyses/gelato_review_2026_10_16/provenance_check.csv"
YUKAGHIR = (
    "GeLaTo and Table S1 of Graff et al. 2025 give the two Yukaghir samples (Lazaridis et al. "
    "2014) each other's code. Yukagir_Forest was sampled at 65.5 N 151 E (upper Kolyma, "
    "Nelemnoye), about 200 km from Glottolog's point for Southern (Forest/Kolyma) Yukaghir "
    "sout2750 and 650 km from Northern (Tundra) Yukaghir nort2745; Yukagir_Tundra at 70.6 N "
    "148 E, near nort2745. Found by the source check of 2026-10-16; coordinates checked 2026-10-20.")
ST2DEC = {"accepted": "accept", "excluded": "exclude", "ambiguous": "ambiguous"}


def row(**kw) -> dict:
    return {**{c: "" for c in lr.SHEET_COLUMNS}, **kw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", required=True, help="a grambank_population_links.csv (own link of each population)")
    ap.add_argument("--prune-with", help="analysis_links.csv: keep doubt rows only for main-set populations")
    ap.add_argument("--out", default=str(lr.SHEET_FILE))
    a = ap.parse_args()
    links = pd.read_csv(a.links, dtype=str, keep_default_na=False)
    own = (links[(links["is_proxy"] == "False") & (links["link_basis"] != "manual")]
           .groupby("population")["glottocode"].first().to_dict())
    rows = {}
    for rv in gelato.load_review(ROOT / "morph_ldl/data/gelato_review.yaml"):
        k = (rv["population"], rv["glottocode"])
        note = gelato._field(rv, "reviewer_note") or gelato._field(rv, "evidence")
        rows[k] = row(population=k[0], link_glottocode=k[1], why_in_sheet="decision of 2026-10-16 (gelato_review.yaml)",
                      decision=ST2DEC[rv["status"]], reason=" ".join(note.split()),
                      confirmed_by=gelato._field(rv, "confirmed_by"), confirmed_date=gelato._field(rv, "confirmed_date"))
    for pop, frm, to in (("Yukagir_Forest", "nort2745", "sout2750"), ("Yukagir_Tundra", "sout2750", "nort2745")):
        rows[(pop, frm)] = row(population=pop, link_glottocode=frm, why_in_sheet="known GeLaTo error (Yukaghir codes swapped)",
                               decision="relink", decision_glottocode=to, reason=YUKAGHIR)
    keep = None
    if a.prune_with:
        al = pd.read_csv(a.prune_with, dtype=str, keep_default_na=False)
        al = al[(al["analysis"] == "main") & (al["in_grambank"] == "True") & (al["meets_coverage_main"] == "True")]
        keep = set(al["population"])
    m = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
    prov = pd.read_csv(PROVENANCE, dtype=str, keep_default_na=False)
    labels = [(p, "diaspora", f"sampled as migrants ({loc})")
              for p, loc in zip(m["PopName"], m["Location"]) if loc.startswith("migrants_collected_in_")]
    labels += [(p, "wrong_code", "source check 2026-10-16: code points to another language")
               for p in prov.loc[prov["shift_status"] == "wrong_code", "population"]]
    for pop, lab, why in labels:
        if keep is not None and pop not in keep:
            continue
        k = (pop, own.get(pop, ""))
        r = rows.setdefault(k, row(population=pop, link_glottocode=k[1]))
        r["doubt"] = ";".join(dict.fromkeys([x for x in r["doubt"].split(";") if x] + [lab]))
        r["why_in_sheet"] = "; ".join(x for x in (r["why_in_sheet"], why) if x)
    df = pd.DataFrame(rows.values(), columns=lr.SHEET_COLUMNS).sort_values(["population", "link_glottocode"])
    lr.write_sheet(df, Path(a.out))
    print(f"{len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
