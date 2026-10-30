"""Two GeLaTo link review sheets (2026-10-16).

  main_set_links.csv              the 294 non-proxy links of the 175 languages in the main
                                  Grambank analysis (plus links outside it named in
                                  manual_flags.csv), with a review mark and a comment
  group_populations_to_review.csv the 29 group_ambiguous populations of the 2026-10-09 sheet
                                  (user decision 2026-10-16: only these are reviewed)

Reads the typology link and outcome tables of outputs/pcfp_v2_review_2026_10_16 (review rule
of 2026-10-16), GeLaTo c625fdc populations.csv, the Graff et al. 2025 population mapping and
the 2026-10-09 unresolved sheet, manual_flags.csv and Glottolog (names, endangerment).
No ancestry or genetic-value file is opened. Refuses to
overwrite a sheet with a filled FILL_* cell.

Run: .venv/bin/python analyses/gelato_review_2026_10_16/make_sheets.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TYP = ROOT / "outputs/pcfp_v2_review_2026_10_16/typology"
SRC = ROOT / "analyses/gelato_feasibility_2026_10_01/sources"
OLD_SHEET = ROOT / "analyses/gelato_review_2026_10_09/unresolved_populations.csv"
FILL = ["FILL_decision", "FILL_glottocode", "FILL_link_type", "FILL_reason", "FILL_sources_consulted",
        "FILL_confirmed_by", "FILL_confirmed_date"]


def read(p, **kw) -> pd.DataFrame:
    return pd.read_csv(p, keep_default_na=False, dtype=str, **kw)


def guard(path: Path) -> None:
    if path.exists() and read(path).reindex(columns=FILL, fill_value="").ne("").any().any():
        raise SystemExit(f"{path} has filled FILL_* cells; not overwriting")


def main() -> None:
    links = read(TYP / "grambank_population_links.csv")
    oc = read(TYP / "grambank_inflection.csv")
    main_langs = set(oc[(oc.gelato_linked == "True") & (oc.link_basis_proxy_only == "False")
                        & (oc.in_grambank == "True") & (oc.meets_coverage_main == "True")].glottocode)
    pops = read(SRC / "gelato_c625fdc/cldf/populations.csv",
                usecols=["Name", "geographicRegion", "country", "Latitude", "Longitude",
                         "curation_notes_linguistics", "Source"]).drop_duplicates("Name").set_index("Name")
    gm = read(SRC / "zenodo_15263706/geneticAdmixture-linguisticDiffusion/input/GeLaTo-population-glottocode-mapping.csv",
              usecols=["PopName", "Reference", "Location", "country", "lat", "lon"]).drop_duplicates("PopName").set_index("PopName")

    gl = read(ROOT / "external/glottolog-cldf/cldf/languages.csv", usecols=["ID", "Name", "Level"]).set_index("ID")
    gv = read(ROOT / "external/glottolog-cldf/cldf/values.csv", usecols=["Language_ID", "Parameter_ID", "Value"])
    aes = gv[gv.Parameter_ID == "aes"].set_index("Language_ID")["Value"].astype(int)
    flags = read(HERE / "manual_flags.csv").set_index("population")["reason"]
    name = lambda c: f"{c} ({gl.at[c, 'Name']})" if c in gl.index else c
    AES = {3: "shifting", 4: "moribund", 5: "nearly extinct", 6: "extinct"}

    live = links[(links.is_proxy == "False") & (links.match_status != "excluded")]
    ms = live[live.glottocode.isin(main_langs)]
    extra = live[live.population.isin(flags.index) & ~live.population.isin(ms.population)]   # e.g. Scottish
    unknown = sorted(set(flags.index) - set(ms.population) - set(extra.population))
    if unknown:
        raise SystemExit(f"manual_flags.csv names populations without a link: {unknown}")
    rows = []
    for r in pd.concat([ms, extra]).itertuples():
        p = pops.loc[r.population] if r.population in pops.index else None
        m = gm.loc[r.population] if r.population in gm.index else None
        pick = lambda mk, pk: (m[mk] if m is not None else "") or (p[pk] if p is not None else "")
        comment, review = [], ""
        if r.population in flags.index:
            review = "requires review"
            comment.append(flags[r.population])
        a = int(aes.get(r.glottocode, 0))
        if a >= 4:
            review = "requires review"
            comment.append(f"Glottolog: {r.language_name} is {AES[a]}; most of the sampled community probably "
                           "speak another language.")
        elif a == 3:
            comment.append(f"Glottolog: {r.language_name} is shifting; some of the sampled people can speak "
                           "another language.")
        if r.link_basis == "dialect_rollup":
            comment.append(f"Dialect code {name(r.source_glottocode)} rolled up to its language in Glottolog 5.3.")
        if r.link_basis == "group_map_down":
            comment.append(f"Group code {name(r.source_glottocode)} mapped down to {r.language_name}, the only "
                           "language of the group with Grambank data.")
        if r.review_ref:
            comment.append(f"Agent review ({r.review_ref}) proposes {r.proposed_status}, not confirmed: "
                           f"{r.decision_reason}")
        skip = ("rolled up to its language", "mapped to its only Grambank-coded", "awaiting human confirmation",
                "variety not reviewed", "review proposes")
        comment += [i for i in r.unresolved_issues.split("; ") if i and not any(k in i for k in skip)]
        rows.append({
            "population": r.population,
            "n_individuals": max(int(r.n_individuals_main or 0), int(r.n_individuals_expanded or 0)),
            "language_name": r.language_name, "glottocode": r.glottocode,
            "own_glottocode": name(r.source_glottocode), "own_glottocode_level": r.source_glottocode_level,
            "link_basis": r.link_basis, "in_main_set": r.glottocode in main_langs,
            "review": review, "comment": " ".join(c if c.endswith(".") else c + "." for c in comment),
            "decision": "",
            "location": pick("Location", "geographicRegion"), "country": pick("country", "country"),
            "lat": pick("lat", "Latitude"), "lon": pick("lon", "Longitude"),
            "reference": pick("Reference", "Source"),
            "gelato_curation_note": p["curation_notes_linguistics"] if p is not None else "",
        })
    out = pd.DataFrame(rows).sort_values(["review", "language_name", "population"], ascending=[False, True, True],
                                         kind="mergesort")
    dst = HERE / "main_set_links.csv"
    if dst.exists() and "decision" in read(dst).columns and read(dst)["decision"].ne("").any():
        raise SystemExit(f"{dst} has filled decision cells; not overwriting")
    out.to_csv(dst, index=False)
    print(f"main_set_links.csv: {len(out)} links ({int(out.in_main_set.sum())} in the main set, "
          f"{out.glottocode[out.in_main_set].nunique()} languages); "
          f"{(out.review != '').sum()} require review ({out[out.review != ''].in_main_set.sum()} in the main set)")

    dst = HERE / "group_populations_to_review.csv"
    guard(dst)
    old = read(OLD_SHEET)
    grp = old[old.issue == "group_ambiguous"]
    grp.to_csv(dst, index=False)
    print(f"group_populations_to_review.csv: {len(grp)} populations "
          f"(of {len(old)} in the 2026-10-09 sheet; {len(old) - len(grp)} not reviewed)")


if __name__ == "__main__":
    main()
