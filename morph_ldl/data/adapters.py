"""Resource adapters -> raw long records -> contract `forms.csv` rows.

Every adapter returns a *raw* DataFrame with one row per (lemma, cell, variant):

    lemma_label, lemma_occurrence, cell_orig, form_orig, variant_idx, n_variants,
    variant_raw, is_missing, source_file, source_row, pos_hint

* `form_orig` is the source's cell content unchanged. Wide MGN files store all
  variants in one cell joined with ';' (form_orig = the whole cell). Long formats
  (data-custom, UniMorph, Paralex) store each variant on its own row; there
  form_orig = that row's form and `variant_idx` is the order of appearance among
  rows with the same (lemma, cell).
* `source_row` is the 1-based line number in the source file (header = line 1).
* `lemma_occurrence` numbers repeated rows of the same lemma label in a wide file
  (distinct paradigms, contract `#k` suffix); long formats cannot distinguish a
  repeated paradigm from a variant, so it is always 1 there.

`to_forms()` adds the contract columns (identifiers, cell_norm, NFC form, segments).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from morph_ldl.schemas import FORMS_COLUMNS

from .cells import normalize_label
from .segments import segment
from .util import MISSING_MARKERS, nfc, read_text_csv

RAW_COLUMNS = ["lemma_label", "lemma_occurrence", "cell_orig", "form_orig", "variant_idx",
               "n_variants", "variant_raw", "is_missing", "source_file", "source_row", "pos_hint"]

PARALEX_DEFECTIVE = {"#DEF#"}


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    for c in RAW_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[RAW_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------
# mgn_wide: lexeme + one column per cell; variants ';'-joined; missing = 'NA'
# --------------------------------------------------------------------------

def read_mgn_wide(path: Path, variant_sep: str = ";") -> pd.DataFrame:
    df = read_text_csv(path)
    if "lexeme" not in df.columns:
        raise ValueError(f"{path}: no 'lexeme' column")
    cells = [c for c in df.columns if c != "lexeme"]
    df.insert(0, "_line", range(2, len(df) + 2))
    df["_occ"] = df.groupby("lexeme").cumcount() + 1
    long = df.melt(id_vars=["_line", "lexeme", "_occ"], value_vars=cells,
                   var_name="cell_orig", value_name="form_orig")
    long["is_missing"] = long["form_orig"].str.strip().isin(MISSING_MARKERS)
    parts = long["form_orig"].where(~long["is_missing"], "").str.split(variant_sep, regex=False)
    long["_variants"] = parts
    long["n_variants"] = parts.map(len).where(~long["is_missing"], 0)
    exploded = long.explode("_variants")
    exploded["variant_idx"] = exploded.groupby(level=0).cumcount()
    exploded = exploded.reset_index(drop=True)
    out = pd.DataFrame({
        "lemma_label": exploded["lexeme"],
        "lemma_occurrence": exploded["_occ"],
        "cell_orig": exploded["cell_orig"],
        "form_orig": exploded["form_orig"],
        "variant_idx": exploded["variant_idx"].astype(int),
        "n_variants": exploded["n_variants"].astype(int),
        "variant_raw": exploded["_variants"].fillna(""),
        "is_missing": exploded["is_missing"].astype(bool),
        "source_file": str(path),
        "source_row": exploded["_line"].astype(int),
        "pos_hint": "",
    })
    out.loc[out["is_missing"], "variant_raw"] = ""
    return _finish(out)


# --------------------------------------------------------------------------
# long formats
# --------------------------------------------------------------------------

def _long_from_rows(df: pd.DataFrame, lemma_col: str, cell_col: str, form_col: str,
                    path: Path, pos_col: Optional[str], missing=MISSING_MARKERS) -> pd.DataFrame:
    df = df.copy()
    df["_line"] = range(2, len(df) + 2) if "_line" not in df.columns else df["_line"]
    df["is_missing"] = df[form_col].str.strip().isin(missing)
    key = [lemma_col, cell_col]
    df["variant_idx"] = df.groupby(key).cumcount()
    df["n_variants"] = df.groupby(key)[form_col].transform("size")
    # a cell explicitly marked missing alongside real variants keeps its marker row
    df.loc[df["is_missing"], "n_variants"] = 0
    out = pd.DataFrame({
        "lemma_label": df[lemma_col],
        "lemma_occurrence": 1,
        "cell_orig": df[cell_col],
        "form_orig": df[form_col],
        "variant_idx": df["variant_idx"].astype(int),
        "n_variants": df["n_variants"].astype(int),
        "variant_raw": df[form_col].where(~df["is_missing"], ""),
        "is_missing": df["is_missing"].astype(bool),
        "source_file": str(path),
        "source_row": df["_line"].astype(int),
        "pos_hint": df[pos_col] if pos_col else "",
    })
    return _finish(out)


def read_mgn_long(path: Path) -> pd.DataFrame:
    """data-custom: lexeme,cell,form,language_ID,POS (extra/index columns tolerated)."""
    df = read_text_csv(path)
    for c in ("lexeme", "cell", "form"):
        if c not in df.columns:
            raise ValueError(f"{path}: missing column {c!r}")
    return _long_from_rows(df, "lexeme", "cell", "form", path, "POS" if "POS" in df.columns else None)


def read_unimorph(path: Path, pos: Optional[str] = None) -> pd.DataFrame:
    """Standard UniMorph TSV: lemma <TAB> form <TAB> features [<TAB> extra ...].

    Blank lines are skipped. If `pos` is given, rows whose feature bundle lacks that
    POS tag are dropped; the POS tag stays in cell_orig and is removed by cell_norm.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"{path}:{i}: expected >=3 tab-separated columns")
            lemma, form, feats = parts[0], parts[1], parts[2]
            tags = {t.strip() for t in feats.split(";")}
            if pos and pos not in tags:
                continue
            rows.append({"lemma": lemma, "cell": feats, "form": form, "_line": i,
                         "pos": pos or next((t for t in ("V", "N", "ADJ") if t in tags), "")})
    df = pd.DataFrame(rows, columns=["lemma", "cell", "form", "_line", "pos"])
    return _long_from_rows(df, "lemma", "cell", "form", path, "pos")


def read_paralex(dataset_dir: Path, representation: str = "phon_custom",
                 forms_file: str = "forms.csv", cells_file: str = "cells.csv",
                 features_file: str = "features-values.csv") -> pd.DataFrame:
    """Paralex package (https://www.paralex-standard.org).

    forms table: form_id, lexeme, cell, phon_form and/or orth_form.
    `representation` 'phon_custom' uses phon_form (space-separated segments by Paralex
    convention), 'orth' uses orth_form. '#DEF#' (defective) counts as missing.
    Cells are resolved to UniMorph features via, in order: a `unimorph` column in
    cells.csv; the `unimorph` column of features-values.csv applied to the cell id's
    '.'-separated value ids; else the raw cell id (normaliser fallback).
    The resolved feature string is placed in `cell_orig` only as `cell_unimorph`
    metadata; `cell_orig` stays the Paralex cell id.
    """
    dataset_dir = Path(dataset_dir)
    forms = read_text_csv(dataset_dir / forms_file)
    col = {"phon_custom": "phon_form", "orth": "orth_form"}.get(representation)
    if col not in forms.columns:
        raise ValueError(f"{dataset_dir}: forms table has no {col!r} column for {representation}")
    raw = _long_from_rows(forms, "lexeme", "cell", col, dataset_dir / forms_file, None,
                          missing=MISSING_MARKERS | PARALEX_DEFECTIVE)
    raw.attrs["paralex_cell_map"] = paralex_cell_map(dataset_dir, cells_file, features_file)
    return raw


def paralex_cell_map(dataset_dir: Path, cells_file="cells.csv",
                     features_file="features-values.csv") -> Dict[str, str]:
    cmap: Dict[str, str] = {}
    cpath = dataset_dir / cells_file
    fpath = dataset_dir / features_file
    cells = read_text_csv(cpath) if cpath.exists() else pd.DataFrame(columns=["cell_id"])
    vals = {}
    if fpath.exists():
        fv = read_text_csv(fpath)
        if "unimorph" in fv.columns:
            vals = {r.value_id: r.unimorph for r in fv.itertuples() if r.unimorph}
    for r in cells.itertuples():
        cid = r.cell_id
        if "unimorph" in cells.columns and getattr(r, "unimorph"):
            cmap[cid] = r.unimorph
        elif vals and all(v in vals for v in cid.split(".")):
            cmap[cid] = ";".join(vals[v] for v in cid.split("."))
    return cmap


ADAPTERS = {"mgn_wide": read_mgn_wide, "mgn_long": read_mgn_long, "unimorph": read_unimorph,
            "paralex": read_paralex}


# --------------------------------------------------------------------------
# raw -> contract rows
# --------------------------------------------------------------------------

@dataclass
class ResourceMeta:
    unit_id: str
    resource_id: str
    resource_version: str
    variety_id: str
    iso639_3: str
    glottocode: str
    pos: str
    representation: str
    file_stem: str


def lemma_ids(raw: pd.DataFrame, resource_id: str) -> pd.Series:
    """`{resource_id}::{label}` with `#k` only when a label has several paradigms."""
    n_occ = raw.groupby("lemma_label")["lemma_occurrence"].transform("max")
    base = resource_id + "::" + raw["lemma_label"]
    return base.where(n_occ <= 1, base + "#" + raw["lemma_occurrence"].astype(str))


def to_forms(raw: pd.DataFrame, meta: ResourceMeta) -> pd.DataFrame:
    labels = raw["cell_orig"].unique()
    cell_map = raw.attrs.get("paralex_cell_map", {})
    norm = {c: normalize_label(cell_map.get(c, c), meta.file_stem)[0] for c in labels}
    forms = [nfc(v).strip() if not m else "" for v, m in zip(raw["variant_raw"], raw["is_missing"])]
    seg_cache: Dict[str, str] = {}

    def seg(f: str) -> str:
        if f not in seg_cache:
            seg_cache[f] = segment(f, meta.representation)
        return seg_cache[f]

    out = pd.DataFrame({
        "unit_id": meta.unit_id,
        "resource_id": meta.resource_id,
        "resource_version": meta.resource_version,
        "variety_id": meta.variety_id,
        "iso639_3": meta.iso639_3,
        "glottocode": meta.glottocode,
        "pos": meta.pos,
        "representation": meta.representation,
        "lemma_id": lemma_ids(raw, meta.resource_id),
        "lemma_label": raw["lemma_label"],
        "group_id": "",
        "cell_orig": raw["cell_orig"],
        "cell_norm": raw["cell_orig"].map(norm),
        "form_orig": raw["form_orig"],
        "variant_idx": raw["variant_idx"].astype(int),
        "n_variants": raw["n_variants"].astype(int),
        "form": forms,
        "segments": [seg(f) for f in forms],
        "is_missing": raw["is_missing"].astype(bool),
        "source_file": raw["source_file"],
        "source_row": raw["source_row"].astype(int),
    })
    return out[FORMS_COLUMNS]


def unparseable_cells(forms: pd.DataFrame) -> pd.DataFrame:
    bad = forms[forms["cell_norm"] == ""]
    if bad.empty:
        return pd.DataFrame(columns=["resource_id", "cell_orig", "n_rows"])
    return (bad.groupby(["resource_id", "cell_orig"]).size().rename("n_rows").reset_index())
