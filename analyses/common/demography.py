"""Audited L1 speaker population and geographic range area lookup for any Glottocode.

Reads from `analyses/common/demography_sources/`:
  - `global_demographic_registry.csv`
  - `bromham_extracted.csv`
  - `ethnologue_population_data.csv`
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

COMMON_DIR = Path(__file__).resolve().parent
DEFAULT_DEMOG_DIR = COMMON_DIR / "demography_sources"

# Group-level / retired-ISO mappings for the two group_map_down languages whose
# parent group ISO carries measured L1 population and polygon area in Bromham/Koplenig.
GROUP_ISO_MAP: dict[str, str] = {
    "minh1238": "mjg",  # Mangghuer <- Tu (tuuu1240, ISO mjg)
    "tanz1241": "ngo",  # Tanzanian Ngoni <- Ngoni (ngon1269, retired ISO ngo)
}
PLURICENTRIC_AREA_MEMBERS: dict[str, list[str]] = {
    "hbs": ["hrv", "srp"],  # Serbian-Croatian-Bosnian: Koplenig Rangesize on hrv + srp
}

DEMOG_COLS: list[str] = [
    "iso_639_3",
    "population_l1",
    "population_source",
    "area_km2",
    "area_source",
    "log_pop",
    "log_area",
    "has_pop",
    "has_area",
]


def resolve_demography_rows(
    glottocodes: Iterable[str],
    demog_dir: Path = DEFAULT_DEMOG_DIR,
) -> list[dict[str, object]]:
    """Resolve audited L1 speaker population and measured range area (km^2) per Glottocode.

    Invariants:
      1. No population is ever invented or proxy-substituted from another language.
         Only direct Glottocode matches in global_demographic_registry.csv and the two
         group_map_down parent ISOs (minh1238 -> mjg, tanz1241 -> ngo) are accepted.
      2. No family/macroarea-imputed range area ever enters area_km2. Only measured polygon
         areas from Bromham (lang_polygon.area_lang) or Koplenig (Rangesize, including the
         hrv + srp sum for pluricentric hbs) are accepted; the 3 Ethnologue-multiISO rows
         without a polygon in Bromham or Koplenig (avl, ayl, zlm) get area_km2 = None.
    """
    with open(demog_dir / "global_demographic_registry.csv", encoding="utf-8-sig") as f:
        reg_by_gc = {r["glottocode"]: r for r in csv.DictReader(f) if r.get("glottocode")}
    with open(demog_dir / "bromham_extracted.csv", encoding="utf-8-sig") as f:
        brom_by_iso = {r["iso_639_3"]: r for r in csv.DictReader(f) if r.get("iso_639_3")}
    with open(demog_dir / "ethnologue_population_data.csv", encoding="utf-8-sig") as f:
        kop_by_iso = {r["ISO"]: r for r in csv.DictReader(f) if r.get("ISO")}

    out: list[dict[str, object]] = []
    for gc in glottocodes:
        iso = ""
        pop_l1: float | None = None
        pop_src = ""
        area_km2: float | None = None
        area_src = ""

        if gc in GROUP_ISO_MAP:
            iso = GROUP_ISO_MAP[gc]
            b = brom_by_iso[iso]
            pop_l1 = float(b["population_l1"])
            pop_src = "Bromham_group_ISO"
            area_km2 = float(b["area_km2"])
            area_src = "Bromham_group_ISO"
        elif gc in reg_by_gc:
            rr = reg_by_gc[gc]
            iso = rr["iso_639_3"]
            if rr.get("population_l1") not in (None, "", "nan", "NaN"):
                pop_l1 = float(rr["population_l1"])
                pop_src = rr["population_source"]
            b_area = brom_by_iso.get(iso, {}).get("area_km2")
            k_area = kop_by_iso.get(iso, {}).get("Rangesize")
            if b_area not in (None, "", "NA", "0"):
                area_km2 = float(b_area)
                area_src = "Bromham"
            elif k_area not in (None, "", "NA", "0"):
                area_km2 = float(k_area)
                area_src = "Koplenig"
            elif iso in PLURICENTRIC_AREA_MEMBERS:
                members = PLURICENTRIC_AREA_MEMBERS[iso]
                area_km2 = sum(float(kop_by_iso[m]["Rangesize"]) for m in members)
                area_src = "Aggregate_Koplenig"

        has_pop = 1 if (pop_l1 is not None and pop_l1 > 0) else 0
        has_area = 1 if (has_pop == 1 and area_km2 is not None and area_km2 > 0) else 0
        out.append({
            "glottocode": gc,
            "iso_639_3": iso,
            "population_l1": round(pop_l1, 6) if pop_l1 is not None else "",
            "population_source": pop_src,
            "area_km2": round(area_km2, 6) if area_km2 is not None else "",
            "area_source": area_src,
            "log_pop": round(math.log(pop_l1), 6) if pop_l1 is not None and pop_l1 > 0 else "",
            "log_area": round(math.log(area_km2), 6) if area_km2 is not None and area_km2 > 0 else "",
            "has_pop": has_pop,
            "has_area": has_area,
        })
    return out


def resolve_demography_df(
    glottocodes: Iterable[str],
    demog_dir: Path = DEFAULT_DEMOG_DIR,
):
    """Return a DataFrame (or list of row dicts if pandas is unavailable) of audited demography."""
    rows = resolve_demography_rows(glottocodes, demog_dir=demog_dir)
    if pd is None:
        return rows
    df = pd.DataFrame(rows)
    for col in ("population_l1", "area_km2", "log_pop", "log_area"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("has_pop", "has_area"):
        df[col] = df[col].astype(int)
    return df
