"""Fill-in sheet for GeLaTo populations that the typology stage could not link to a language.
(reads outputs/pcfp_v2/typology and population metadata only; writes
unresolved_populations.csv here)

One row per population link without a language-level Glottocode:
  * unresolved                    the population has no Glottocode (NA) or one not in Glottolog 5.3;
  * group_ambiguous               its Glottocode is a group with several Grambank-coded languages;
  * group_no_grambank_descendant  its Glottocode is a group with no Grambank-coded language.

Context columns come from the typology link table, GeLaTo c625fdc populations.csv and the
Graff et al. 2025 population mapping (place, reference, curator comments). No ancestry or
genetic-value file is opened. The FILL_* columns are for the reviewer.

The script refuses to overwrite a sheet in which any FILL_* cell is filled.

Run: .venv/bin/python analyses/gelato_review_2026_10_09/make_unresolved_sheet.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "unresolved_populations.csv"
SRC = ROOT / "analyses/gelato_feasibility_2026_10_01/sources"
FILL = ["FILL_decision", "FILL_glottocode", "FILL_link_type", "FILL_reason", "FILL_sources_consulted",
        "FILL_confirmed_by", "FILL_confirmed_date"]


def read(p, **kw) -> pd.DataFrame:
    return pd.read_csv(p, keep_default_na=False, dtype=str, **kw)


def main() -> None:
    if OUT.exists() and read(OUT).reindex(columns=FILL, fill_value="").ne("").any().any():
        raise SystemExit(f"{OUT} has filled FILL_* cells; not overwriting")
    links = read(ROOT / "outputs/pcfp_v2/typology/grambank_population_links.csv")
    gl = read(ROOT / "external/glottolog-cldf/cldf/languages.csv", usecols=["ID", "Name", "Level"]).set_index("ID")
    gb = set(read(ROOT / "external/grambank/cldf/languages.csv", usecols=["ID"]).ID)
    pops = read(SRC / "gelato_c625fdc/cldf/populations.csv",
                usecols=["Name", "Language_Name", "geographicRegion", "country", "Latitude", "Longitude",
                         "curation_notes_linguistics", "Source"]).drop_duplicates("Name").set_index("Name")
    gm = read(SRC / "zenodo_15263706/geneticAdmixture-linguisticDiffusion/input/GeLaTo-population-glottocode-mapping.csv",
              usecols=["PopName", "Reference", "Location", "country", "lat", "lon",
                       "comment.gbi.proxies", "comment.tli.proxies"]).drop_duplicates("PopName").set_index("PopName")

    def label(codes: str) -> str:
        out = []
        for c in filter(None, codes.split(";")):
            name, level = (gl.loc[c, "Name"], gl.loc[c, "Level"]) if c in gl.index else ("?", "not in Glottolog 5.3")
            out.append(f"{c} ({name}, {level}{', in Grambank' if c in gb else ''})")
        return "; ".join(out)

    linked = links[links.glottocode != ""]
    rows = []
    for r in links[links.glottocode == ""].itertuples():
        p = pops.loc[r.population] if r.population in pops.index else None
        m = gm.loc[r.population] if r.population in gm.index else None
        other = linked[linked.population == r.population]
        n = max(int(r.n_individuals_main or 0), int(r.n_individuals_expanded or 0))
        rows.append({
            "population": r.population,
            "issue": r.link_basis,
            "n_individuals": n,
            "gelato_language_name": p["Language_Name"] if p is not None else "",
            "own_glottocode": label(r.source_glottocode),
            "candidates": label(r.candidate_glottocodes),
            "graff_gbi_proxy": label(r.tableS1_gbi_glottocode),
            "graff_tli_proxy": label(r.tableS1_tli_glottocode),
            "other_links_of_population": "; ".join(f"{o.glottocode} ({o.language_name}, {o.link_basis})"
                                                   for o in other.itertuples()),
            "location": (m["Location"] if m is not None else "") or (p["geographicRegion"] if p is not None else ""),
            "country": (m["country"] if m is not None else "") or (p["country"] if p is not None else ""),
            "lat": (m["lat"] if m is not None else "") or (p["Latitude"] if p is not None else ""),
            "lon": (m["lon"] if m is not None else "") or (p["Longitude"] if p is not None else ""),
            "reference": (m["Reference"] if m is not None else "") or (p["Source"] if p is not None else ""),
            "gelato_curation_note": p["curation_notes_linguistics"] if p is not None else "",
            "graff_comment_gbi": m["comment.gbi.proxies"] if m is not None else "",
            "graff_comment_tli": m["comment.tli.proxies"] if m is not None else "",
            "pipeline_issues": r.unresolved_issues,
            **{c: "" for c in FILL},
        })
    df = pd.DataFrame(rows)
    order = {"group_ambiguous": 0, "group_no_grambank_descendant": 1, "unresolved": 2}
    df = df.sort_values(["issue", "n_individuals", "population"], ascending=[True, False, True],
                        key=lambda s: s.map(order) if s.name == "issue" else s)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows -> {OUT.relative_to(ROOT)}")
    print(df.issue.value_counts().to_string())


if __name__ == "__main__":
    main()
