"""Data stage entry points: `run_data_stage(cfg)`, `load_forms(cfg, unit_id)`,
`build_population_links(outcome_units, cfg)`.

Outputs (all under outputs/<experiment_id>/):
  data/forms/<unit_id>.csv           contract forms.csv rows (group_id filled)
  data/lemmas/<unit_id>.csv          lemma_id, group_id, group_size, join_reasons, eligible
  data/unparseable_cells.csv         cell labels with empty cell_norm
  data/stage_manifest.json
  registry/registry.csv, registry/identifier_crosswalk.csv, registry/build_provenance.csv
  eligibility/<unit_id>_cells.csv, <unit_id>_duplicates.csv, summary.csv,
  eligibility/config_cell_check.csv, eligibility/broad_audit.csv
  gelato/crosswalk.csv, gelato/ancestry_sources.csv, gelato/population_links.csv
"""

from __future__ import annotations

import datetime as _dt
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from morph_ldl.config import PIPELINE_ROOT, output_dir, resolve, unit_cells
from morph_ldl.schemas import FORMS_COLUMNS

from . import adapters, eligibility, gelato, grouping, identifiers, registry
from .provenance import build_provenance, representation_from_provenance
from .util import (guard_path, read_text_csv, repo_root, resource_version, sha256_file, write_csv,
                   write_json, git_state)

STAGE = "data"


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------

def _ingest_entry(entry: Dict[str, Any], cfg: Dict[str, Any], prov) -> tuple[adapters.ResourceMeta, pd.DataFrame]:
    ad = entry["adapter"]
    if ad == "mgn_wide":
        path, coll = resolve(cfg, "mgn_data") / entry["file"], "mgn_data"
    elif ad == "mgn_long":
        path, coll = resolve(cfg, "mgn_custom") / entry["file"], "mgn_data-custom"
    else:
        path, coll = (PIPELINE_ROOT / entry["file"]).resolve(), ad
    stem = path.stem if path.is_file() else path.name
    declared = entry["representation"]
    if coll in ("mgn_data", "mgn_data-custom"):
        detected = representation_from_provenance(prov[stem], coll)
        if detected != declared:
            raise ValueError(f"{stem}: configured representation {declared!r} but build provenance says "
                             f"{detected!r}; representations may not be mixed or relabelled")
        rid = registry.resource_id_for(coll, stem)
    else:
        rid = f"{ad}:{stem}"
    ident = identifiers.resolve_identifier(entry["iso"])
    pos = entry["pos"]
    unit_id = next((u["unit_id"] for u in cfg.get("units", []) if u["resource_id"] == rid),
                   registry.prospective_unit_id(ident["variety_id"], pos, declared,
                                                "mgn" if coll.startswith("mgn") else ad))
    if ad == "mgn_wide":
        raw = adapters.read_mgn_wide(path)
    elif ad == "mgn_long":
        raw = adapters.read_mgn_long(path)
    elif ad == "unimorph":
        raw = adapters.read_unimorph(path, pos=pos)
    elif ad == "paralex":
        raw = adapters.read_paralex(path, representation=declared)
    else:
        raise ValueError(f"unknown adapter {ad!r}")
    version = resource_version(path) if path.is_file() else resource_version(path / "forms.csv")
    meta = adapters.ResourceMeta(unit_id, rid, entry.get("version", version), ident["variety_id"],
                                 ident["canonical_iso639_3"], ident["final_glottocode"], pos, declared, stem)
    return meta, raw


def _rel(path: str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return path


def ingest(cfg: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, List[adapters.ResourceMeta]]:
    mgn_root = resolve(cfg, "mgn_data").parent
    prov = build_provenance(mgn_root)
    root = repo_root(cfg)
    metas, frames = [], []
    for entry in cfg["resources"]["ingest"]:
        meta, raw = _ingest_entry(entry, cfg, prov)
        f = adapters.to_forms(raw, meta)
        f["source_file"] = f["source_file"].map(lambda p: _rel(p, root))
        metas.append(meta)
        frames.append(f)
    forms = pd.concat(frames, ignore_index=True)
    # one representation per unit
    reps = forms.groupby("unit_id")["representation"].nunique()
    if (reps > 1).any():
        raise ValueError(f"mixed representations within units: {list(reps[reps > 1].index)}")
    source_cells = {}
    for u in cfg.get("units", []):
        source_cells[u["resource_id"]] = unit_cells(u, cfg)["source"]
    groups = grouping.assign_groups(forms, source_cells)
    forms = forms.drop(columns="group_id").merge(groups[["lemma_id", "group_id"]], on="lemma_id", how="left")
    return forms[FORMS_COLUMNS], groups, metas


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def forms_path(cfg: Dict[str, Any], unit_id: str) -> Path:
    return output_dir(cfg) / "data" / "forms" / f"{unit_id}.csv"


def load_forms(cfg: Dict[str, Any], unit_id: str) -> pd.DataFrame:
    """Read a unit's forms.csv written by the data stage, with contract dtypes."""
    path = forms_path(cfg, unit_id)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the data stage first")
    df = read_text_csv(path)
    for c in ("variant_idx", "n_variants", "source_row"):
        df[c] = df[c].astype(int)
    df["is_missing"] = df["is_missing"].map({"True": True, "False": False}).astype(bool)
    return df[FORMS_COLUMNS]


def build_population_links(outcome_units: List[str], cfg: Dict[str, Any]) -> pd.DataFrame:
    """One row per (unit_id, population candidate): status + sample/ancestry-source ids.

    No morphology values and no aggregation over populations. Uses the crosswalk
    written by the data stage (gelato/crosswalk.csv); if absent, builds it for the
    configured units only.
    """
    path = output_dir(cfg) / "gelato" / "crosswalk.csv"
    if path.exists():
        xw = read_text_csv(path)
    else:
        xw = gelato.build_crosswalk(_unit_resource_table(cfg), cfg)
    return gelato.links_from_crosswalk(xw, list(outcome_units))


def _unit_resource_table(cfg: Dict[str, Any], reg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resource rows for configured units (from the registry if available)."""
    rows = []
    mgn_root = resolve(cfg, "mgn_data").parent
    prov = build_provenance(mgn_root)
    for u in cfg.get("units", []):
        rid = u["resource_id"]
        entry = next((e for e in cfg["resources"]["ingest"]
                      if rid.endswith(":" + Path(e["file"]).stem)), None)
        if entry is None:
            continue
        ident = identifiers.resolve_identifier(entry["iso"])
        coll = "mgn_data" if entry["adapter"] == "mgn_wide" else "mgn_data-custom"
        path = resolve(cfg, "mgn_data" if coll == "mgn_data" else "mgn_custom") / entry["file"]
        rows.append({"unit_id": u["unit_id"], "resource_id": rid, "resource_version": resource_version(path),
                     "language": ident["language_label_low"], "variety_id": ident["variety_id"],
                     "pos": entry["pos"], "representation": entry["representation"],
                     "original_id": ident["original_id"], "iso639_3": ident["canonical_iso639_3"],
                     "glottocode": ident["final_glottocode"], "extinct": ident["extinct"]})
    return pd.DataFrame(rows)


def run_data_stage(cfg: Dict[str, Any], with_registry: bool = True) -> Dict[str, Any]:
    t0 = _dt.datetime.now(_dt.timezone.utc)
    out = output_dir(cfg)
    root = repo_root(cfg)
    inv = int(cfg.get("cv", {}).get("inventory_size", 1200))
    written: List[str] = []

    def w(df, rel):
        written.append(str(write_csv(df, out / rel, cfg).relative_to(out)))

    # 1. ingest configured resources
    forms, groups, metas = ingest(cfg)
    unparse = adapters.unparseable_cells(forms)
    w(unparse, "data/unparseable_cells.csv")

    # 2. per-unit forms, lemma tables and eligibility
    summary, cfg_check = [], []
    for u in cfg.get("units", []):
        uid = u["unit_id"]
        f = forms[forms["unit_id"] == uid]
        if f.empty:
            raise ValueError(f"unit {uid} has no ingested forms (resource {u['resource_id']})")
        w(f, f"data/forms/{uid}.csv")
        uc = unit_cells(u, cfg)
        src, panel = uc["source"], uc["panel"]
        elig = eligibility.eligible_lemmas(f, src, [c for _, c in panel])
        g = groups[groups["lemma_id"].isin(f["lemma_id"].unique())].copy()
        g["eligible"] = g["lemma_id"].isin(elig)
        w(g, f"data/lemmas/{uid}.csv")
        cells_tab = eligibility.unit_cell_table(f, src, panel)
        w(cells_tab, f"eligibility/{uid}_cells.csv")
        w(eligibility.duplicate_table(f, groups), f"eligibility/{uid}_duplicates.csv")
        derived = eligibility.derived_paradigms(f, f["iso639_3"].iloc[0], src)
        derived["eligible"] = derived["lemma_id"].isin(elig)
        w(derived, f"eligibility/{uid}_derived_paradigms.csv")
        elig_set = set(elig)
        d_el = derived[derived["eligible"]]
        present = set(f["cell_norm"]) - {""}
        cov = dict(zip(cells_tab["cell_norm"], cells_tab["coverage"]))
        slots = u["cells"] if isinstance(u["cells"], dict) else {c: c for c in u["cells"]}  # PCFP: cell list
        cfg_check += eligibility.config_cell_check(uid, slots, present, cov)
        nm = f[~f["is_missing"]].drop_duplicates(["lemma_id", "cell_orig"])
        taskc = [src] + [c for _, c in panel]
        tnm = nm[nm["cell_norm"].isin(taskc)]
        n_slots = f.drop_duplicates(["lemma_id", "cell_orig"])
        eg = g[g["eligible"]]
        src_forms = f[(f["cell_norm"] == src) & (~f["is_missing"]) & (f["variant_idx"] == 0)]
        summary.append({
            "unit_id": uid, "resource_id": u["resource_id"], "role": u.get("role", ""),
            "resource_version": f["resource_version"].iloc[0], "representation": f["representation"].iloc[0],
            "n_lemmas": f["lemma_id"].nunique(), "n_groups": g["group_id"].nunique(),
            "n_cells": f["cell_orig"].nunique(), "n_cells_unparseable": int((cells_tab["cell_norm"] == "").sum()),
            "source_cell": src, "source_present": src in present,
            "panel_cells_missing_from_resource": " | ".join(c for _, c in panel if c not in present),
            "n_eligible_lemmas": len(elig), "n_eligible_groups": eg["group_id"].nunique(),
            "inventory_size": inv, "meets_inventory_size": eg["group_id"].nunique() >= inv,
            "missing_rate_all_cells": round(1 - len(nm) / max(1, len(n_slots)), 4),
            "missing_rate_task_cells": round(1 - len(tnm) / max(1, f["lemma_id"].nunique() * len(taskc)), 4),
            "variant_rate_all_cells": round(float((nm["n_variants"] > 1).mean()), 4),
            "variant_rate_task_cells": round(float((tnm["n_variants"] > 1).mean()) if len(tnm) else 0.0, 4),
            "multiword_rate_task_cells": round(float(f[f["cell_norm"].isin(taskc) & ~f["is_missing"]]["form"].str.contains(" ", regex=False).mean()), 4),
            "n_duplicate_label_lemmas": int(f.loc[f["lemma_id"].str.contains("#", regex=False), "lemma_id"].nunique()),
            "n_multi_lemma_groups": int((g.drop_duplicates("group_id")["group_size"] > 1).sum()),
            "n_lemmas_sharing_source_form": int(src_forms.duplicated("form", keep=False).sum()),
            "max_group_size": int(g["group_size"].max()),
            "n_eligible_pronominal_with_base": int((d_el["kind"] == "pronominal_of").sum()),
            "n_eligible_pronominal_no_base": int((d_el["kind"] == "pronominal_no_base").sum()),
            "n_eligible_multiword_source": int((d_el["kind"] == "multiword_source").sum()),
            "n_eligible_multiword_label_only": int((d_el["kind"] == "multiword_label").sum()),
            "n_eligible_groups_if_pronominal_joined": eligibility.groups_if_joined(g, derived, elig_set),
            "n_eligible_lemmas_excluding_derived": len(elig_set - set(d_el["lemma_id"])),
        })
    w(pd.DataFrame(summary), "eligibility/summary.csv")
    w(pd.DataFrame(cfg_check), "eligibility/config_cell_check.csv")

    # 3. identifier crosswalk, registry, broad audit
    mgn_root = resolve(cfg, "mgn_data").parent
    stems = {coll: [f.stem for f in sorted((mgn_root / sub).glob("*.csv"))]
             for coll, (sub, _) in registry.COLLECTIONS.items()}
    idx = identifiers.identifier_crosswalk(stems)
    w(idx, "registry/identifier_crosswalk.csv")
    reg = None
    if with_registry:
        reg, audit, prov = registry.build_registry(mgn_root, root, inv)
        w(reg, "registry/registry.csv")
        w(audit, "eligibility/broad_audit.csv")
        w(pd.DataFrame([{**vars(p), "script_lines": ";".join(map(str, p.script_lines)),
                         "notes": "; ".join(p.notes)} for p in prov.values()]), "registry/build_provenance.csv")

    # 3b. optional provenance verification against pinned UniMorph clones
    ext = PIPELINE_ROOT / "external"
    ver = []
    for code, files in (("ita", ["ita"]), ("fin", ["fin.1", "fin.2"])):
        d = ext / f"unimorph-{code}"
        mf = resolve(cfg, "mgn_data") / f"{code}-v.csv"
        if d.exists() and mf.exists() and with_registry:
            from .verify_unimorph import verify
            ver.append(verify(mf, [d / x for x in files], "V", git_state(d).get("commit")))
    if ver:
        w(pd.DataFrame(ver), "registry/unimorph_verification.csv")

    # 4. GeLaTo crosswalk: configured units + every registry resource
    res = _unit_resource_table(cfg)
    if reg is not None:
        r2 = reg.rename(columns={"unit_id_prospective": "unit_id"})
        r2 = r2[~r2["resource_id"].isin(res["resource_id"])] if len(res) else r2
        res = pd.concat([res, r2[res.columns]], ignore_index=True)
    xw = gelato.build_crosswalk(res, cfg)
    w(xw, "gelato/crosswalk.csv")
    w(gelato.ancestry_sources(cfg), "gelato/ancestry_sources.csv")
    unit_ids = [u["unit_id"] for u in cfg.get("units", [])]
    links = gelato.links_from_crosswalk(xw, unit_ids)
    w(links, "gelato/population_links.csv")

    # 5. manifest
    inputs = [{"path": str(p.relative_to(root)), "sha256": sha256_file(p)}
              for p in sorted({(resolve(cfg, "mgn_data") / e["file"]) if e["adapter"] == "mgn_wide"
                               else (resolve(cfg, "mgn_custom") / e["file"]) if e["adapter"] == "mgn_long"
                               else (PIPELINE_ROOT / e["file"]).resolve()
                               for e in cfg["resources"]["ingest"]}) if p.is_file()]
    inputs += [{"path": str(gelato.REVIEW_FILE.relative_to(root)), "sha256": sha256_file(gelato.REVIEW_FILE)}]
    pkgs = {}
    for name in ("pandas", "numpy", "pyyaml", "languages-of-the-world"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    ext = PIPELINE_ROOT / "external"
    manifest = {
        "stage": STAGE, "experiment_id": cfg["experiment"]["id"], "config_hash": cfg.get("_config_hash"),
        "config_path": cfg.get("_config_path"), "inputs": inputs,
        "external_revisions": {
            "languages-of-the-world": git_state(ext / "languages-of-the-world").get("commit"),
            "unimorph-ita": git_state(ext / "unimorph-ita").get("commit") if (ext / "unimorph-ita").exists() else None,
            "unimorph-fin": git_state(ext / "unimorph-fin").get("commit") if (ext / "unimorph-fin").exists() else None,
            "gelato": gelato.GELATO_COMMIT, "zenodo": gelato.ZENODO_RECORD,
        },
        "package_versions": {"python": platform.python_version(), **pkgs},
        "seeds": {}, "start_time": t0.isoformat(), "end_time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git": git_state(root), "outputs": sorted(written), "with_registry": with_registry,
    }
    write_json(manifest, out / "data" / "stage_manifest.json", cfg)
    return {"output_dir": str(out), "units": unit_ids, "summary": summary, "n_forms_rows": len(forms),
            "crosswalk_status_counts": xw["match_status"].value_counts().to_dict(), "outputs": sorted(written)}
