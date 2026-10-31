"""Glottolog classification path lookup and topology Newick tree builder.

Works for any list of Glottocodes (Grambank or LDL/Wug), both with pandas installed
and under a stdlib-only Python interpreter.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterable
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]


def resolve_glottolog_paths(
    glottocodes: Iterable[str],
    root_dir: Path = ROOT,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Resolve classification ancestor paths and tree source for each Glottocode.

    Uses `languages-of-the-world` (`low`) first and falls back to Glottolog 5.3 CLDF
    `values.csv` (`Parameter_ID == 'classification'`).
    """
    low_src = str(root_dir / "external/languages-of-the-world/src")
    if low_src not in sys.path:
        sys.path.insert(0, low_src)
    import low  # noqa: E402

    by_code = {l.glottocode: l for l in low.LanguagesOfTheWorld().languages if l.glottocode}
    lineage_gl: dict[str, str] = {}
    with open(root_dir / "external/glottolog-cldf/cldf/values.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Parameter_ID"] == "classification":
                lineage_gl[r["Language_ID"]] = r["Value"]

    paths: dict[str, list[str]] = {}
    src: dict[str, str] = {}
    for code in glottocodes:
        lang = by_code.get(code)
        if lang is not None and lang.family is not None:
            fam = lang.family
            paths[code] = [a.glottocode for a in reversed(fam.ancestors)] + [fam.glottocode]
            src[code] = "low"
        else:
            paths[code] = [x for x in lineage_gl.get(code, "").split("/") if x]
            src[code] = "glottolog_5.3"
    return paths, src


def resolve_glottolog_coords_rows(
    glottocodes: Iterable[str],
    root_dir: Path = ROOT,
) -> list[dict[str, object]]:
    """Resolve Glottolog 5.3 coordinates, family, macroarea, and name as a list of dicts."""
    with open(root_dir / "external/glottolog-cldf/cldf/languages.csv", encoding="utf-8") as f:
        gl_by_id = {r["ID"]: r for r in csv.DictReader(f)}
    rows: list[dict[str, object]] = []
    for gc in glottocodes:
        r = gl_by_id.get(gc, {})
        lat_s = r.get("Latitude", "")
        lon_s = r.get("Longitude", "")
        rows.append({
            "glottocode": gc,
            "name": r.get("Name", ""),
            "family": r.get("Family_ID") or "isolate",
            "macroarea": r.get("Macroarea", ""),
            "lat": float(lat_s) if lat_s not in ("", None) else float("nan"),
            "lon": float(lon_s) if lon_s not in ("", None) else float("nan"),
        })
    return rows


def resolve_glottolog_coords(
    glottocodes: Iterable[str],
    root_dir: Path = ROOT,
):
    """Resolve Glottolog 5.3 coordinates, family, macroarea, and name (DataFrame or list[dict])."""
    rows = resolve_glottolog_coords_rows(glottocodes, root_dir=root_dir)
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def build_newick(paths: dict[str, list[str]]) -> str:
    """Topology-only Newick: internal nodes are Glottolog groups, tips are languages.

    Families hang from a common root; branch lengths are set later (e.g. Grafen in R).
    """
    tree: dict = {}
    for tip, path in paths.items():
        node = tree
        for a in path:
            node = node.setdefault(a, {})
        node[tip] = None

    def emit(name: str, children: dict | None) -> str:
        if children is None:
            return name
        kids = [emit(k, v) for k, v in children.items()]
        return kids[0] if len(kids) == 1 else "(" + ",".join(kids) + ")" + name

    return emit("root", tree) + ";"


newick = build_newick
