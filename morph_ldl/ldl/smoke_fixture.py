"""Build small labelled LDL fixtures from an MGN wide CSV (development / smoke tests only).

The real pipeline reads ``forms.csv`` and selection samples (CONTRACT §2, §5). Until those
exist, this module produces the same *shape* of inputs for the LDL runner:

* ``train_<name>.csv``  - background rows: lemma_id, cell_norm, form, segments, variant_idx
* ``test_queries.csv``  - CONTRACT §6 query table (no gold column)
* ``gold.csv``          - lemma_id, target_cell, gold_variants (read only by scoring)

Nothing is written into ``mgn_data/``.
"""

from __future__ import annotations

import argparse
import csv
import random
import unicodedata
from pathlib import Path

DEFAULT_CELLS = {
    # slot -> cell_norm (configs/pilot.yaml, unit ita.V.orth.mgn)
    "NFIN": "NFIN",
    "PRS.1SG": "1;IND;PRS;SG",
    "PRS.3SG": "3;IND;PRS;SG",
    "PRS.3PL": "3;IND;PL;PRS",
    "PST.1SG": "1;IND;PFV;PST;SG",
    "PST.3SG": "3;IND;PFV;PST;SG",
    "PST.3PL": "3;IND;PFV;PL;PST",
    "COND.3SG": "3;COND;SG",
    "IMP.2SG": "2;IMP;POS;SG",
}


def norm_cell(label: str, pos: str = "V") -> str:
    feats = sorted({f.strip().upper() for f in label.split(";") if f.strip()} - {pos})
    return ";".join(feats)


def segments(form: str) -> str:
    form = unicodedata.normalize("NFC", form.strip())
    out: list[str] = []
    for ch in form:
        if out and unicodedata.combining(ch):
            out[-1] += ch
        else:
            out.append("_" if ch == " " else ch)
    if any("#" in s for s in out):
        raise ValueError(f"reserved boundary symbol in {form!r}")
    return " ".join(out)


def read_wide(path: Path, resource_id: str, cells: dict[str, str]):
    """Return {lemma_id: {cell_norm: [variants]}} for lemmas with all requested cells."""
    wanted = set(cells.values())
    out: dict[str, dict[str, list[str]]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        col_by_norm = {norm_cell(c): c for c in reader.fieldnames if c != "lexeme"}
        missing = wanted - set(col_by_norm)
        if missing:
            raise KeyError(f"cells not in {path}: {sorted(missing)}")
        seen: dict[str, int] = {}
        for row in reader:
            label = row["lexeme"]
            seen[label] = seen.get(label, 0) + 1
            lemma_id = f"{resource_id}::{label}" + (f"#{seen[label]}" if seen[label] > 1 else "")
            paradigm = {}
            ok = True
            for cn in wanted:
                raw = row[col_by_norm[cn]]
                if raw in ("", "NA"):
                    ok = False
                    break
                paradigm[cn] = [unicodedata.normalize("NFC", v.strip()) for v in raw.split(";")]
            if ok:
                out[lemma_id] = paradigm
    return out


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def build(wide: Path, out_dir: Path, n_background: int, n_heldout: int, seed: int,
          resource_id: str = "mgn_data:ita-v", cells: dict[str, str] | None = None,
          n_background2_overlap: int | None = None) -> dict:
    cells = cells or DEFAULT_CELLS
    source_cell = cells["NFIN"]
    panel = [cn for slot, cn in cells.items() if slot != "NFIN"]
    data = read_wide(wide, resource_id, cells)
    ids = sorted(data)
    rng = random.Random(seed)
    rng.shuffle(ids)
    held = ids[:n_heldout]
    bg1 = ids[n_heldout:n_heldout + n_background]
    # second background: shares the first `overlap` lemmas of bg1, rest new
    ov = n_background // 2 if n_background2_overlap is None else n_background2_overlap
    bg2 = bg1[:ov] + ids[n_heldout + n_background:n_heldout + n_background + (n_background - ov)]

    def train_rows(lemmas):
        rows = []
        for lid in lemmas:
            for cn in [source_cell] + panel:
                f = data[lid][cn][0]
                rows.append([lid, cn, f, segments(f), 0])
        return rows

    hdr = ["lemma_id", "cell_norm", "form", "segments", "variant_idx"]
    write_csv(out_dir / "train_bg1.csv", hdr, train_rows(bg1))
    write_csv(out_dir / "train_bg2.csv", hdr, train_rows(bg2))
    q, g = [], []
    for lid in held:
        src = data[lid][source_cell][0]
        for cn in panel:
            q.append([lid, source_cell, src, segments(src), cn])
            g.append([lid, cn, "|".join(data[lid][cn])])
    write_csv(out_dir / "test_queries.csv",
              ["lemma_id", "source_cell", "source_form", "source_segments", "target_cell"], q)
    write_csv(out_dir / "gold.csv", ["lemma_id", "target_cell", "gold_variants"], g)
    return {"heldout": held, "bg1": bg1, "bg2": bg2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", default="mgn_data/data/ita-v.csv")
    ap.add_argument("--out", default="outputs/scratch_ldl/fixture_ita")
    ap.add_argument("--n-background", type=int, default=60)
    ap.add_argument("--n-heldout", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    info = build(Path(a.wide), Path(a.out), a.n_background, a.n_heldout, a.seed)
    print({k: len(v) for k, v in info.items()})


if __name__ == "__main__":
    main()
