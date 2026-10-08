"""Resource registry over every MGN paradigm file (data/ and data-custom/)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .adapters import read_mgn_long, read_mgn_wide
from .eligibility import profile_raw
from .identifiers import original_id, resolve_identifier
from .provenance import (BuildProvenance, build_provenance, representation_check,
                         representation_from_provenance)
from .util import resource_version

POS_FROM_SUFFIX = {"v": "V", "n": "N", "adj": "ADJ"}
COLLECTIONS = {"mgn_data": ("data", "mgn_data"), "mgn_data-custom": ("data-custom", "mgn_custom")}


def resource_id_for(collection: str, stem: str) -> str:
    return f"{COLLECTIONS[collection][1]}:{stem}"


def prospective_unit_id(variety_id: str, pos: str, representation: str, slug: str = "mgn") -> str:
    return f"{variety_id}.{pos}.{representation}.{slug}"


def list_mgn_files(mgn_root: Path) -> List[Tuple[str, Path]]:
    out = []
    for coll, (sub, _) in COLLECTIONS.items():
        out += [(coll, f) for f in sorted((mgn_root / sub).glob("*.csv"))]
    return out


def suitability(row: Dict[str, Any], prof: Dict[str, Any], inventory_size: int) -> Tuple[str, str]:
    if row["extinct"]:
        return "out_of_scope", "extinct/historical language (synchronic study)"
    if row["pos"] == "ADJ":
        return "not_assessed", "adjectives not part of the declared task"
    spec = prof["spec"]
    if not spec.source:
        return "insufficient", spec.method
    groups = prof["n_eligible_groups"]
    caveats = []
    if row["representation"] == "ipa_epitran":
        caveats.append("automatic epitran G2P (orthography-driven, unverified)")
    if prof["panel_multiword_rate"] > 0.05:
        caveats.append(f"multiword panel forms {prof['panel_multiword_rate']:.1%}")
    if spec.fallbacks:
        caveats.append("task cells by fallback: " + "; ".join(spec.fallbacks))
    if prof["variant_rate"] > 0.05:
        caveats.append(f"variant rate {prof['variant_rate']:.1%}")
    if not str(row["representation_check"]).startswith("consistent"):
        caveats.append(str(row["representation_check"]))
    if row["n_cells_unparseable"]:
        caveats.append(f"{row['n_cells_unparseable']} unparseable cells")
    if groups < inventory_size // 2:
        return "insufficient", f"{groups} eligible groups < {inventory_size // 2}"
    if groups < inventory_size:
        caveats.insert(0, f"{groups} eligible groups < inventory_size {inventory_size}")
        return "limited", "; ".join(caveats)
    return ("suitable", "") if not caveats else ("suitable_with_caveats", "; ".join(caveats))


def build_registry(mgn_root: Path, repo_root: Path, inventory_size: int,
                   only: List[str] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, BuildProvenance]]:
    prov = build_provenance(mgn_root)
    reg_rows, audit_rows = [], []
    for coll, path in list_mgn_files(mgn_root):
        stem = path.stem
        if only and stem not in only:
            continue
        p = prov[stem]
        pos = POS_FROM_SUFFIX[stem.rsplit("-", 1)[1]]
        rep = representation_from_provenance(p, coll)
        ident = resolve_identifier(original_id(stem, coll))
        raw = read_mgn_wide(path) if coll == "mgn_data" else read_mgn_long(path)
        prof = profile_raw(raw, stem, pos, rep, inventory_size, lexeme_recovery=(coll == "mgn_data"))
        rc = representation_check(prof["sample_forms"], rep)
        notes = list(p.notes)
        if prof["multiword_rate"] > 0.01:
            notes.append(f"multiword forms {prof['multiword_rate']:.1%}")
        if prof["n_forms_with_hash"]:
            notes.append(f"{prof['n_forms_with_hash']} forms contain reserved '#'")
        if prof["n_forms_with_underscore"]:
            notes.append(f"{prof['n_forms_with_underscore']} forms contain '_' (clashes with word separator)")
        if rep == "ipa_epitran" and rc["upper_rate"]:
            notes.append(f"upper-case letters in {rc['upper_rate']:.1%} of sampled epitran outputs")
        if coll == "mgn_data-custom" and pos == "N" and stem == "russian-n":
            notes.append("lemma labels are transcriptions with stress marks, not orthography")
        rid = resource_id_for(coll, stem)
        row = {
            "resource_id": rid, "collection": coll, "file": str(path.relative_to(repo_root)),
            "resource_version": resource_version(path), "language": ident["language_label_low"],
            "original_id": ident["original_id"], "iso639_3": ident["canonical_iso639_3"],
            "variety_id": ident["variety_id"], "glottocode": ident["final_glottocode"], "pos": pos,
            "representation": rep, "representation_check": rc["status"],
            "derivation": p.derivation, "upstream_code": p.upstream_code or "",
            "epitran_code": p.epitran_code or "", "build_script": p.script or "",
            "build_script_lines": ";".join(map(str, p.script_lines)),
            "cell_min_forms_filter": p.cell_min_forms or "",
            "extinct": ident["extinct"], "in_mgn_living_64": ident["in_mgn_living_64"],
            "unit_id_prospective": prospective_unit_id(ident["variety_id"], pos, rep),
            "n_lemmas": prof["n_lemmas"], "n_duplicate_labels": prof["n_duplicate_labels"],
            "n_cells": prof["n_cells"], "n_cells_unparseable": len(prof["unparseable_cells"]),
            "unparseable_cells": " | ".join(prof["unparseable_cells"]),
            "n_records": prof["n_records"], "missing_rate": prof["missing_rate"],
            "variant_rate": prof["variant_rate"], "multiword_rate": prof["multiword_rate"],
            "ipa_char_rate_sample": rc["ipa_char_rate"], "quality_notes": "; ".join(notes),
        }
        spec = prof["spec"]
        status, reason = suitability(row, prof, inventory_size)
        row.update({"task_source_cell": spec.source or "", "task_n_eligible_lemmas": prof["n_eligible_lemmas"],
                    "task_n_eligible_groups": prof["n_eligible_groups"],
                    "suitability": status, "suitability_reason": reason})
        reg_rows.append(row)
        if pos in ("V", "N"):
            audit_rows.append({
                "resource_id": rid, "unit_id_prospective": row["unit_id_prospective"],
                "language": row["language"], "iso639_3": row["iso639_3"], "glottocode": row["glottocode"],
                "pos": pos, "representation": rep, "extinct": row["extinct"], "n_lemmas": row["n_lemmas"],
                "source_cell": spec.source or "", "source_coverage": round(prof["coverage"].get(spec.source, 0.0), 4) if spec.source else 0.0,
                "panel_method": spec.method, "panel": " | ".join(f"{s}={c}" for s, c in spec.panel),
                "panel_min_coverage": round(min((prof["coverage"].get(c, 0.0) for _, c in spec.panel), default=0.0), 4),
                "panel_fallbacks": "; ".join(spec.fallbacks),
                "n_eligible_lemmas": prof["n_eligible_lemmas"], "n_eligible_groups": prof["n_eligible_groups"],
                "meets_inventory_size": prof["n_eligible_groups"] >= inventory_size,
                "panel_multiword_rate": prof["panel_multiword_rate"], "variant_rate": prof["variant_rate"],
                "suitability": status, "suitability_reason": reason,
            })
    return pd.DataFrame(reg_rows), pd.DataFrame(audit_rows), prov
