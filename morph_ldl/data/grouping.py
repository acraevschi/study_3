"""Leakage groups (docs/CONTRACT.md §3).

Within one variety_id + pos, across all resources/representations ingested in a run,
lemmas are joined if they share (a) the normalised lemma label (NFC, case-folded,
whitespace-collapsed) or (b) an identical source-cell form in the same representation.
For (b) every non-missing variant of the source cell is used (conservative: more joins).
Groups are connected components; ids are `{variety_id}.{pos}::g{n:06d}` numbered by
the sorted smallest lemma_id of each component (deterministic for a given input set).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping

import pandas as pd

from .util import nfc

_WS = re.compile(r"\s+")


def norm_label(label: str) -> str:
    return _WS.sub(" ", nfc(label).casefold()).strip()


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.p = {x: x for x in items}

    def find(self, x: str) -> str:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def assign_groups(forms: pd.DataFrame, source_cells: Mapping[str, str]) -> pd.DataFrame:
    """Return a lemma table: lemma_id, variety_id, pos, group_id, join reasons.

    `source_cells` maps resource_id -> source cell_norm (resources without an entry
    are joined on labels only).
    """
    lem = forms[["lemma_id", "lemma_label", "variety_id", "pos", "resource_id"]].drop_duplicates("lemma_id")
    out = []
    for (variety, pos), sub in lem.groupby(["variety_id", "pos"], sort=True):
        uf = UnionFind(sub["lemma_id"])
        reasons: Dict[str, set] = {x: set() for x in sub["lemma_id"]}
        # (a) normalised label
        by_label: Dict[str, list] = {}
        for lid, lab in zip(sub["lemma_id"], sub["lemma_label"]):
            by_label.setdefault(norm_label(lab), []).append(lid)
        for ids in by_label.values():
            for other in ids[1:]:
                uf.union(ids[0], other)
            if len(ids) > 1:
                for x in ids:
                    reasons[x].add("label")
        # (b) identical source-cell form within a representation
        f = forms[(forms["variety_id"] == variety) & (forms["pos"] == pos) & (~forms["is_missing"])]
        f = f[f["resource_id"].map(lambda r: source_cells.get(r, None)) == f["cell_norm"]]
        if len(f):
            for (_, _), g in f.groupby(["representation", "form"]):
                ids = sorted(set(g["lemma_id"]))
                for other in ids[1:]:
                    uf.union(ids[0], other)
                if len(ids) > 1:
                    for x in ids:
                        reasons[x].add("source_form")
        comps: Dict[str, list] = {}
        for lid in sub["lemma_id"]:
            comps.setdefault(uf.find(lid), []).append(lid)
        ordered = sorted(comps.values(), key=lambda ids: min(ids))
        for n, ids in enumerate(ordered, start=1):
            gid = f"{variety}.{pos}::g{n:06d}"
            for lid in ids:
                out.append({"lemma_id": lid, "variety_id": variety, "pos": pos, "group_id": gid,
                            "group_size": len(ids), "join_reasons": ";".join(sorted(reasons[lid]))})
    return pd.DataFrame(out, columns=["lemma_id", "variety_id", "pos", "group_id", "group_size", "join_reasons"])


def quick_group_count(lemma_labels: pd.Series, source_forms: pd.DataFrame) -> int:
    """Number of groups among a resource's lemmas (label + source-form joins).

    `source_forms`: columns lemma_label, form (all non-missing source variants).
    Used by the broad audit, where full forms tables are not materialised.
    """
    labels = list(dict.fromkeys(lemma_labels))
    uf = UnionFind(labels)
    by_norm: Dict[str, str] = {}
    for lab in labels:
        k = norm_label(lab)
        if k in by_norm:
            uf.union(by_norm[k], lab)
        else:
            by_norm[k] = lab
    first: Dict[str, str] = {}
    for lab, form in zip(source_forms["lemma_label"], source_forms["form"]):
        if lab not in uf.p:
            continue
        if form in first:
            uf.union(first[form], lab)
        else:
            first[form] = lab
    return len({uf.find(x) for x in labels})
