"""Stage-2 Bayesian data preparation for Track 2 (Paradigm Predictability).

Integrates paradigm learner outcomes (`unit_outcomes_{regime}.csv`, `lemma_outcomes_{regime}.csv`),
lexicon provenance (`unit_summary.csv`, `provenance_manifest.csv`), corpus token frequencies
(`frequencies/{unit_id}_freq.csv`), GeLaTo population-to-language links across all 4 sensitivity
sets (`main`, `no_doubtful_language`, `no_other_language_stand_ins`, `strict`), and shared
`analyses.common` (`genetics`, `demography`, `phylogeny`) + Track 1 `irreg13` overlap into
strictly separate Verb (`V`) and Noun (`N`) 1-row-per-`glottocode` Stage-2 inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyses.common.genetics as _genetics_mod  # noqa: E402
from analyses.common.demography import DEMOG_COLS, resolve_demography_rows  # noqa: E402
from analyses.common.genetics import (  # noqa: E402
    ADM_DIR,
    ARCH_DIR,
    DEFAULT_MIN_N_IND,
    GELATO_DATA_CSV,
    KS,
    N_BOOT,
    POP_BOOT_MEASURES,
    SEED,
    compute_individual_measures,
    compute_language_f3_maxp,
    compute_population_measures,
    load_admixture_q,
    load_f3_fixed_scores,
    load_fst_neighbor_regimes,
    write_rows_csv,
)
from analyses.common.phylogeny import build_newick, resolve_glottolog_paths  # noqa: E402

SETS: list[str] = ["main", "no_doubtful_language", "no_other_language_stand_ins", "strict"]

DEFAULT_PARADIGM_DATA_DIR = ROOT / "analyses/paradigm/data"
DEFAULT_MGL_RESULTS_DIR = ROOT / "analyses/paradigm/results/mgl"
DEFAULT_NEURAL_RESULTS_DIR = ROOT / "analyses/paradigm/results/neural"
DEFAULT_OUT_DIR = ROOT / "analyses/paradigm/results/stage2/prepared"
COMMON_DATA_DIR = ROOT / "analyses/common/data"
TRACKED_LINKS_CSV = COMMON_DATA_DIR / "analysis_links.csv"
LEGACY_LINKS_CSV = ROOT / "outputs/pcfp_v2_links_2026_10_20/typology/analysis_links.csv"
DEFAULT_LINKS_CSV = TRACKED_LINKS_CSV if TRACKED_LINKS_CSV.is_file() else LEGACY_LINKS_CSV
GRAMBANK_DATA_DIR = ROOT / "analyses/grambank/data"
GRAMBANK_SEM_RESULTS_DIR = ROOT / "analyses/grambank/latent_sem/results"

TRACK1_DOMAIN_GATES: tuple[tuple[str, int, int], ...] = (
    ("v_breadth", 6, 9),
    ("n_breadth", 3, 5),
    ("v_irreg", 3, 5),
    ("n_irreg", 2, 3),
    ("np_agr", 5, 7),
)

_IND_COLS: list[str] = ["population", "ind", "ent", "nonmax", "ent_K12", "ent_K20", "ent_K30", "glottocode"]
_POP_COLS: list[str] = [
    "population", "n_individuals", "pop_ent", "het", "support", "amount", "sqrt_het",
    "pop_ent_se", "het_se", "support_se", "amount_se", "sqrt_het_se",
    "graff_target", "neigh_flow", "n_B", "n_C", "n_triples", "n_z_lt_m3",
    "min_pop2", "min_pop3", "f3_z", "f3_est", "f3b3_z", "f3b3_est",
    "f3k_z", "f3k_est", "f3k_est_pos",
    "f3_fixed", "f3_fixed_iso", "f3_fixed_raw", "f3_fixed_se", "f3_all_fixed",
    "iso_fallback_used", "f4_est", "f4_se", "f4_z",
    "neigh_flow_out", "median_fst_all", "fst_mismatch_ratio",
    "barbieri_mismatch", "regime_a_enclave", "regime_b_shift",
    "glottocode", "link_basis", "accepted_by", "sensitivity_flags",
    "in_main", "in_no_doubtful_language", "in_no_other_language_stand_ins", "in_strict",
]
_POP_INT_RAW_COLS: tuple[str, ...] = (
    "n_individuals", "graff_target", "n_B", "n_C", "n_triples", "n_z_lt_m3",
    "iso_fallback_used", "barbieri_mismatch", "regime_a_enclave", "regime_b_shift",
)
_POP_FLT_RAW_COLS: tuple[str, ...] = (
    "pop_ent", "het", "support", "amount", "sqrt_het",
    "pop_ent_se", "het_se", "support_se", "amount_se", "sqrt_het_se",
    "neigh_flow", "f3_z", "f3_est", "f3b3_z", "f3b3_est",
    "f3k_z", "f3k_est", "f3k_est_pos",
    "f3_fixed", "f3_fixed_iso", "f3_fixed_raw", "f3_fixed_se", "f3_all_fixed",
    "f4_est", "f4_se", "f4_z",
    "neigh_flow_out", "median_fst_all", "fst_mismatch_ratio",
)


def _resolve_links_csv(links_csv: Path) -> Path:
    if links_csv.is_file():
        return links_csv
    if links_csv == TRACKED_LINKS_CSV and LEGACY_LINKS_CSV.is_file():
        return LEGACY_LINKS_CSV
    if links_csv == LEGACY_LINKS_CSV and TRACKED_LINKS_CSV.is_file():
        return TRACKED_LINKS_CSV
    raise FileNotFoundError(f"GeLaTo links CSV not found: {links_csv}")


def classify_source_provenance(source_id: str, priority_tier: str = "") -> str:
    """Collapse fine-grained `source_id` into 3 canonical provenance classes for regression controls."""
    sid = (source_id or "").strip().lower()
    pt = (priority_tier or "").strip().lower()
    if pt.startswith("1_") or sid.startswith(("paralex_", "celex_", "qumin_", "surrey_")):
        return "paralex"
    if pt.startswith("2_") or sid.startswith(("national_", "morphit", "polimorf", "vesum", "opencorpora", "morfflex")):
        return "national"
    return "unimorph"


def source_priority_rank(source_prov: str) -> int:
    """Priority order if multiple orthographic units ever exist for one (glottocode, pos)."""
    return {"paralex": 0, "national": 1, "unimorph": 2}.get(source_prov, 3)


def load_glottolog_metadata(root_dir: Path = ROOT) -> dict[str, dict[str, object]]:
    """Load Glottolog 5.3 language names, coordinates, macroareas, and human-readable family names."""
    lang_csv = root_dir / "external/glottolog-cldf/cldf/languages.csv"
    by_id: dict[str, dict[str, str]] = {}
    with open(lang_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_id[r["ID"]] = r

    out: dict[str, dict[str, object]] = {}
    for gc, r in by_id.items():
        fam_id = (r.get("Family_ID") or "").strip()
        if fam_id and fam_id in by_id:
            fam_name = by_id[fam_id].get("Name", fam_id) or "isolate"
        elif fam_id:
            fam_name = fam_id
        else:
            fam_name = "isolate"
        lat_s = r.get("Latitude", "")
        lon_s = r.get("Longitude", "")
        out[gc] = {
            "glottocode": gc,
            "name": r.get("Name", ""),
            "family": fam_name,
            "family_id": fam_id or "isolate",
            "macroarea": r.get("Macroarea", ""),
            "lat": float(lat_s) if lat_s not in ("", None) else float("nan"),
            "lon": float(lon_s) if lon_s not in ("", None) else float("nan"),
        }
    return out


def load_track1_overlap(
    gb_data_dir: Path = GRAMBANK_DATA_DIR,
    gb_sem_dir: Path = GRAMBANK_SEM_RESULTS_DIR,
) -> dict[str, dict[str, object]]:
    """Load Track 1 5-domain proportions (`>= 60%` gates) plus graceful legacy fallbacks."""
    ld_path = gb_data_dir / "language_data.csv"
    if not ld_path.exists():
        return {}

    with open(ld_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ld_cols = set(reader.fieldnames or [])
        ld_rows = {r["glottocode"]: r for r in reader}

    has_5dom = all(f"{dom}_coded" in ld_cols and f"{dom}_present" in ld_cols for dom, _, _ in TRACK1_DOMAIN_GATES)

    # Legacy item-level fallback (only used if legacy columns/artifacts exist)
    obs_path = gb_data_dir / "item_observations.csv"
    meta_path = gb_data_dir / "items_metadata.csv"
    ipar_path = gb_sem_dir / "item_parameters_irreg13_mb2pl.csv"

    irreg_domains: dict[str, str] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("in_irreg13") == "1":
                    irreg_domains[r["item_id"]] = r["domain"]

    obs_by_gc: dict[str, dict[str, int]] = defaultdict(dict)
    if obs_path.exists() and irreg_domains:
        with open(obs_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                iid = r["item_id"]
                if iid in irreg_domains:
                    obs_by_gc[r["glottocode"]][iid] = int(r["value"])

    irt_params: dict[str, tuple[float, float]] = {}
    if ipar_path.exists():
        with open(ipar_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("measure") == "ent":
                    irt_params[r["item_id"]] = (float(r["alpha_mean"]), float(r["lambda_mean"]))

    theta_grid = np.linspace(-4.0, 4.0, 161)
    log_prior = -0.5 * (theta_grid ** 2) - 0.5 * math.log(2.0 * math.pi)

    out: dict[str, dict[str, object]] = {}
    for gc, r in ld_rows.items():
        rec: dict[str, object] = {}

        # Canonical 5-domain proportions gated on >= 60% rule (>=6/9, >=3/5, >=3/5, >=2/3, >=5/7)
        any_5dom_elig = False
        for dom, min_cod, _k_tot in TRACK1_DOMAIN_GATES:
            pres = _to_int_or_none(r.get(f"{dom}_present"))
            cod = _to_int_or_none(r.get(f"{dom}_coded"))
            if pres is not None and cod is not None and cod >= min_cod:
                prop = pres / cod
                any_5dom_elig = True
            else:
                prop = float("nan")
            rec[f"{dom}_present"] = pres
            rec[f"{dom}_coded"] = cod
            rec[f"{dom}_prop"] = prop

        # Legacy columns (gracefully optional)
        c12_pres = _to_int_or_none(r.get("n_present"))
        c12_cod = _to_int_or_none(r.get("n_coded"))
        ir_pres = _to_int_or_none(r.get("irreg13_present"))
        ir_cod = _to_int_or_none(r.get("irreg13_coded"))
        ir_prop = (ir_pres / ir_cod) if (ir_pres is not None and ir_cod is not None and ir_cod >= 8) else float("nan")

        g_obs = obs_by_gc.get(gc, {})
        v_vals = [
            val for iid, val in g_obs.items()
            if irreg_domains.get(iid) in ("verb_suppletion_stem", "arg_split_classes")
        ]
        n_vals = [
            val for iid, val in g_obs.items()
            if irreg_domains.get(iid) == "noun_irreg"
        ]
        v_prop = (sum(v_vals) / len(v_vals)) if len(v_vals) >= 4 else float("nan")
        n_prop = (sum(n_vals) / len(n_vals)) if len(n_vals) >= 2 else float("nan")

        theta_eap = float("nan")
        if ir_cod is not None and ir_cod >= 8 and irt_params and g_obs:
            ll = log_prior.copy()
            for iid, y_val in g_obs.items():
                if iid in irt_params:
                    alpha_i, lam_i = irt_params[iid]
                    eta = alpha_i + lam_i * theta_grid
                    sign_eta = eta if y_val == 1 else -eta
                    ll += -np.logaddexp(0.0, -sign_eta)
            ll -= np.max(ll)
            w = np.exp(ll)
            theta_eap = float(np.sum(w * theta_grid) / np.sum(w))

        in_main_flag = r.get("in_main") == "1"
        if has_5dom:
            in_t1 = 1 if (in_main_flag and any_5dom_elig) else 0
        else:
            in_t1 = 1 if (in_main_flag and c12_cod is not None and c12_cod >= 8) else 0

        rec.update({
            "in_track1": in_t1,
            "cat12_present": c12_pres,
            "cat12_coded": c12_cod,
            "irreg13_present": ir_pres,
            "irreg13_coded": ir_cod,
            "irreg13_prop": ir_prop,
            "irreg_verb_prop": v_prop,
            "irreg_noun_prop": n_prop,
            "theta_irreg13_eap": theta_eap,
        })
        out[gc] = rec
    return out


def resolve_links_for_glottocodes(
    links_csv: Path,
    target_glottocodes: set[str],
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    """Resolve 1-to-1 `(population -> glottocode)` links across all 4 analysis link sets.

    In `analysis_links.csv`, 16 populations carry BOTH an own-language row (`is_proxy=False`)
    AND a Grambank stand-in row (`is_proxy=True`, `stand_in_other_language`). Within our active
    `target_glottocodes`, if a population has a non-proxy (`is_proxy=False`) link to a language
    in `target_glottocodes`, that own-language link strictly takes precedence over any proxy
    stand-in, guaranteeing a 1-to-1 `population -> glottocode` map in `populations_{pos}.csv`
    and `individuals_{pos}.csv`.
    """
    by_set_raw: dict[str, list[dict[str, str]]] = defaultdict(list)
    resolved_links_csv = _resolve_links_csv(links_csv)
    with open(resolved_links_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["analysis"] in SETS and r["glottocode"] in target_glottocodes:
                by_set_raw[r["analysis"]].append(r)

    def deduplicate_pop_links(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        by_pop: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in rows:
            by_pop[r["population"]].append(r)
        kept: list[dict[str, str]] = []
        for pop in sorted(by_pop.keys()):
            cands = by_pop[pop]
            non_proxy = [c for c in cands if str(c.get("is_proxy", "")).lower() != "true"]
            pool = non_proxy if non_proxy else cands
            # Deterministic tie-break if ever needed: exact > dialect_rollup > manual > lexicographic
            basis_rank = {"exact": 0, "manual": 1, "dialect_rollup": 2, "group_map_down": 3}
            pool_sorted = sorted(
                pool,
                key=lambda x: (basis_rank.get(x["link_basis"], 9), x["glottocode"]),
            )
            kept.append(pool_sorted[0])
        return kept

    main_links = deduplicate_pop_links(by_set_raw["main"])
    main_pop_to_gc = {r["population"]: r["glottocode"] for r in main_links}

    in_set: dict[str, set[str]] = {"main": set(main_pop_to_gc.keys())}
    for s in SETS:
        if s == "main":
            continue
        s_dedup = deduplicate_pop_links(by_set_raw[s])
        # Sensitivity sets only drop links from `main` (never switch a population to another language)
        in_set[s] = {
            r["population"]
            for r in s_dedup
            if main_pop_to_gc.get(r["population"]) == r["glottocode"]
        }
        assert in_set[s] <= in_set["main"]

    return main_links, in_set


def _read_csv_optional(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_missing_scalar(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        return val in ("", "NA", "<NA>", "NaN", "nan", "None")
    if isinstance(val, float):
        return math.isnan(val)
    return str(val) in ("", "NA", "<NA>", "NaN", "nan", "None")


def _to_float_or_nan(val: object) -> float:
    if _is_missing_scalar(val):
        return float("nan")
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _to_int_or_none(val: object) -> int | None:
    if _is_missing_scalar(val):
        return None
    try:
        return int(float(val))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _load_cached_genetics_for_pos(
    *,
    pos: str,
    main_link_by_pop: dict[str, dict[str, str]],
    in_set: dict[str, set[str]],
    ref_cat12_pops: set[str],
    out_dir: Path,
    fallback_prepared_dir: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float], dict[str, tuple[int, float, float]]]:
    """Load precomputed tracked `individuals_{pos}.csv` & `populations_{pos}.csv` when raw ADMIXTURE is absent."""
    search_dirs: list[Path] = [out_dir]
    if fallback_prepared_dir is not None and fallback_prepared_dir.resolve() != out_dir.resolve():
        search_dirs.append(fallback_prepared_dir)

    cached_dir: Path | None = None
    for cand_dir in search_dirs:
        if (cand_dir / f"individuals_{pos}.csv").is_file() and (cand_dir / f"populations_{pos}.csv").is_file():
            cached_dir = cand_dir
            break
    if cached_dir is None:
        raise FileNotFoundError(
            f"Raw ADMIXTURE files are unavailable and precomputed individuals_{pos}.csv / populations_{pos}.csv "
            f"were not found in {[str(d) for d in search_dirs]}"
        )

    cached_ind_raw = _read_csv_optional(cached_dir / f"individuals_{pos}.csv")
    cached_pop_raw = _read_csv_optional(cached_dir / f"populations_{pos}.csv")

    expected_pops = set(main_link_by_pop.keys())
    ind_pops = {r["population"] for r in cached_ind_raw}
    pop_pops = {r["population"] for r in cached_pop_raw}
    missing_pops = sorted(expected_pops - (ind_pops & pop_pops))
    if missing_pops:
        raise ValueError(
            f"Cached Stage-2 genetics in {cached_dir} for POS={pos} missing {len(missing_pops)} "
            f"resolved population(s): {missing_pops}"
        )

    ind_rows: list[dict[str, object]] = []
    for r in cached_ind_raw:
        pname = r["population"]
        if pname not in main_link_by_pop:
            continue
        ind_rows.append({
            "population": pname,
            "ind": int(r["ind"]),
            "ent": float(r["ent"]),
            "nonmax": float(r["nonmax"]),
            "ent_K12": float(r["ent_K12"]),
            "ent_K20": float(r["ent_K20"]) if "ent_K20" in r and r["ent_K20"] not in ("", None) else float("nan"),
            "ent_K30": float(r["ent_K30"]),
            "glottocode": main_link_by_pop[pname]["glottocode"],
        })
    ind_rows.sort(key=lambda x: int(x["ind"]))  # type: ignore[arg-type]
    write_rows_csv(ind_rows, out_dir / f"individuals_{pos}.csv", _IND_COLS, ndigits=6)

    f3f_raw = load_f3_fixed_scores()
    f3f_list = f3f_raw.to_dict("records") if hasattr(f3f_raw, "to_dict") else list(f3f_raw)
    f3f_by_pop = {str(r["population"]): r for r in f3f_list}
    fstr_raw = load_fst_neighbor_regimes()
    fstr_list = fstr_raw.to_dict("records") if hasattr(fstr_raw, "to_dict") else list(fstr_raw)
    fstr_by_pop = {str(r["population"]): r for r in fstr_list}

    pop_by_name = {r["population"]: r for r in cached_pop_raw if r["population"] in main_link_by_pop}
    pop_rows: list[dict[str, object]] = []
    for pname in sorted(main_link_by_pop.keys()):
        pr = pop_by_name[pname]
        ml = main_link_by_pop[pname]
        f3f_r = f3f_by_pop.get(pname, {})
        fstr_r = fstr_by_pop.get(pname, {})
        parsed_p: dict[str, object] = {
            "population": pname,
            "min_pop2": pr.get("min_pop2", ""),
            "min_pop3": pr.get("min_pop3", ""),
            "glottocode": ml["glottocode"],
            "link_basis": ml["link_basis"],
            "accepted_by": ml["accepted_by"],
            "sensitivity_flags": ml.get("sensitivity_flags", ""),
        }
        for ic in _POP_INT_RAW_COLS:
            raw_v = pr.get(ic) if ic in pr else (f3f_r.get(ic) if ic in f3f_r else fstr_r.get(ic))
            parsed_p[ic] = _to_int_or_none(raw_v)
        for fc in _POP_FLT_RAW_COLS:
            raw_v = pr.get(fc) if fc in pr else (f3f_r.get(fc) if fc in f3f_r else fstr_r.get(fc))
            parsed_p[fc] = _to_float_or_nan(raw_v)
        for s_name in SETS:
            parsed_p[f"in_{s_name}"] = 1 if pname in in_set[s_name] else 0
        pop_rows.append(parsed_p)
    write_rows_csv(pop_rows, out_dir / f"populations_{pos}.csv", _POP_COLS, ndigits=7)

    # Preserve exact pre-rounding 6-digit language `ent`/`nonmax` when population membership is unchanged
    cached_lang_ent: dict[str, tuple[int, float, float]] = {}
    cached_ld_path = cached_dir / f"language_data_{pos}.csv"
    if cached_ld_path.is_file():
        for lr in _read_csv_optional(cached_ld_path):
            gc_k = lr.get("glottocode", "")
            n_ind_k = _to_int_or_none(lr.get("n_individuals"))
            ent_k = _to_float_or_nan(lr.get("ent"))
            nmx_k = _to_float_or_nan(lr.get("nonmax"))
            if gc_k and n_ind_k is not None and not (math.isnan(ent_k) or math.isnan(nmx_k)):
                cached_lang_ent[gc_k] = (n_ind_k, ent_k, nmx_k)

    pooled_sd: dict[str, float] = {}
    summary_json_path = cached_dir / "stage2_prep_summary.json"
    if summary_json_path.is_file() and expected_pops == pop_pops:
        try:
            cached_sum = json.loads(summary_json_path.read_text(encoding="utf-8"))
            psd = cached_sum.get("pos", {}).get(pos, {}).get("pooled_per_individual_sd", {})
            if isinstance(psd, dict) and all(m in psd for m in POP_BOOT_MEASURES):
                pooled_sd = {m: float(psd[m]) for m in POP_BOOT_MEASURES}
        except (OSError, ValueError, TypeError):
            pooled_sd = {}
    if not pooled_sd:
        ref_active = ref_cat12_pops & expected_pops if ref_cat12_pops else expected_pops
        for m in POP_BOOT_MEASURES:
            big_vals = [
                float(p[f"{m}_se"]) * math.sqrt(int(p["n_individuals"]))  # type: ignore[arg-type]
                for p in pop_rows
                if str(p["population"]) in ref_active
                and int(p["n_individuals"]) >= 3  # type: ignore[arg-type]
                and not math.isnan(float(p[f"{m}_se"]))  # type: ignore[arg-type]
            ]
            if not big_vals:
                big_vals = [
                    float(p[f"{m}_se"]) * math.sqrt(int(p["n_individuals"]))  # type: ignore[arg-type]
                    for p in pop_rows
                    if int(p["n_individuals"]) >= 3  # type: ignore[arg-type]
                    and not math.isnan(float(p[f"{m}_se"]))  # type: ignore[arg-type]
                ]
            pooled_sd[m] = float(statistics.median(big_vals)) if big_vals else float("nan")

    return ind_rows, pop_rows, pooled_sd, cached_lang_ent


def prepare_stage2_inputs(
    *,
    paradigm_data_dir: Path = DEFAULT_PARADIGM_DATA_DIR,
    mgl_results_dir: Path = DEFAULT_MGL_RESULTS_DIR,
    neural_results_dir: Path = DEFAULT_NEURAL_RESULTS_DIR,
    links_csv: Path = DEFAULT_LINKS_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    pos_list: Sequence[str] = ("V", "N"),
    adm_dir: Path | None = None,
    gelato_data_csv: Path | None = None,
    arch_dir: Path | None = None,
    fallback_prepared_dir: Path | None = DEFAULT_OUT_DIR,
) -> dict[str, object]:
    """Build separate `V` and `N` 1-row-per-`glottocode` Stage-2 and entrenchment CSV/NWK files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_links_csv = _resolve_links_csv(links_csv)

    # 1. Load provenance & unit_summary metadata
    prov_rows = _read_csv_optional(paradigm_data_dir / "provenance_manifest.csv")
    tier_by_source = {r["source_id"]: r.get("priority_tier", "") for r in prov_rows if r.get("source_id")}

    summary_rows = _read_csv_optional(paradigm_data_dir / "unit_summary.csv")
    unit_meta_by_id: dict[str, dict[str, str]] = {
        r["unit_id"]: r for r in summary_rows if r.get("unit_id")
    }

    # 2. Load primary MGL outcomes (`uniform120`) + secondary regimes (`uniform480`, `topfreq120`) + neural
    mgl_120 = _read_csv_optional(mgl_results_dir / "unit_outcomes_uniform120.csv")
    if not mgl_120:
        raise FileNotFoundError(
            f"Primary outcome file not found or empty: {mgl_results_dir / 'unit_outcomes_uniform120.csv'}"
        )
    mgl_480_by_uid = {
        r["unit_id"]: r for r in _read_csv_optional(mgl_results_dir / "unit_outcomes_uniform480.csv")
    }
    mgl_topf_by_uid = {
        r["unit_id"]: r for r in _read_csv_optional(mgl_results_dir / "unit_outcomes_topfreq120.csv")
    }
    neu_120_by_uid = {
        r["unit_id"]: r for r in _read_csv_optional(neural_results_dir / "unit_outcomes_uniform120.csv")
    }
    neu_480_by_uid = {
        r["unit_id"]: r for r in _read_csv_optional(neural_results_dir / "unit_outcomes_uniform480.csv")
    }
    mgl_lemma_120 = _read_csv_optional(mgl_results_dir / "lemma_outcomes_uniform120.csv")

    # Index paired phon / malouf MGL outcomes by unit_id and (glottocode, pos),
    # filtering out `phon` rows from `malouf_by_gc_pos` so `*_phon` never overwrites `*_orth`.
    phon_by_gc_pos: dict[tuple[str, str], dict[str, str]] = {}
    malouf_by_uid: dict[str, dict[str, str]] = {}
    malouf_by_gc_pos: dict[tuple[str, str], dict[str, str]] = {}
    for r in mgl_120:
        rep = r.get("representation", "")
        mdl_name = r.get("model", "")
        if rep == "phon" and mdl_name != "malouf":
            phon_by_gc_pos[(r["glottocode"], r["pos"])] = r
        elif rep == "malouf" or mdl_name == "malouf":
            malouf_by_uid[r["unit_id"]] = r
            if rep != "phon":
                malouf_by_gc_pos[(r["glottocode"], r["pos"])] = r
    for mal_fname in ("unit_outcomes_uniform120_malouf.csv", "unit_outcomes_malouf.csv"):
        for r in _read_csv_optional(mgl_results_dir / mal_fname):
            malouf_by_uid[r["unit_id"]] = r
            if r.get("representation", "orth") != "phon":
                malouf_by_gc_pos[(r["glottocode"], r["pos"])] = r

    glotto_meta = load_glottolog_metadata(ROOT)
    track1_map = load_track1_overlap()

    eff_adm_dir = adm_dir if adm_dir is not None else (
        ADM_DIR if ADM_DIR != _genetics_mod.ADM_DIR else _genetics_mod.ADM_DIR
    )
    eff_gelato_csv = gelato_data_csv if gelato_data_csv is not None else (
        GELATO_DATA_CSV if GELATO_DATA_CSV != _genetics_mod.GELATO_DATA_CSV else _genetics_mod.GELATO_DATA_CSV
    )
    eff_arch_dir = arch_dir if arch_dir is not None else (
        ARCH_DIR if ARCH_DIR != _genetics_mod.ARCH_DIR else _genetics_mod.ARCH_DIR
    )
    has_raw_admixture = (
        (eff_adm_dir / "GeneticInfoID.csv").is_file()
        and eff_gelato_csv.is_file()
        and (eff_arch_dir / "tables/tableS2.csv").is_file()
    )
    info_adm, q_adm = load_admixture_q(adm_dir=eff_adm_dir, ks=KS) if has_raw_admixture else (None, None)

    # Reference populations from Track 1 cat12 for identical small-sample pooled bootstrap SE scaling
    all_al_rows = _read_csv_optional(resolved_links_csv)
    ref_cat12_pops = {
        r["population"]
        for r in all_al_rows
        if r.get("analysis") == "main"
        and r.get("in_grambank") == "True"
        and r.get("meets_coverage_main") == "True"
    }

    prep_summary: dict[str, object] = {"pos": {}}

    for pos in pos_list:
        # Select canonical 1-row-per-glottocode `orth` unit for this POS
        cands_by_gc: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in mgl_120:
            if r.get("pos") == pos and r.get("representation", "orth") == "orth":
                cands_by_gc[r["glottocode"]].append(r)

        chosen_by_gc: dict[str, dict[str, str]] = {}
        for gc, clist in sorted(cands_by_gc.items()):
            def _rank_unit(u_row: dict[str, str]) -> tuple[int, str]:
                um = unit_meta_by_id.get(u_row["unit_id"], {})
                sid = um.get("source_id", u_row.get("source_id", ""))
                sprov = classify_source_provenance(sid, tier_by_source.get(sid, ""))
                return (source_priority_rank(sprov), u_row["unit_id"])

            chosen_by_gc[gc] = sorted(clist, key=_rank_unit)[0]

        if not chosen_by_gc:
            prep_summary["pos"][pos] = {"n_languages_main": 0}
            continue

        main_links, in_set = resolve_links_for_glottocodes(resolved_links_csv, set(chosen_by_gc.keys()))
        linked_gcs = sorted({r["glottocode"] for r in main_links})
        if not linked_gcs:
            prep_summary["pos"][pos] = {"n_languages_main": 0}
            continue

        main_link_by_pop = {r["population"]: r for r in main_links}
        assert len(main_link_by_pop) == len(main_links)

        cached_lang_ent: dict[str, tuple[int, float, float]] = {}
        if has_raw_admixture:
            # Compute individual & population genetic measures via `analyses.common.genetics`
            ind_df = compute_individual_measures(info_adm, q_adm, links_df=main_links, ks=KS)
            if hasattr(ind_df, "to_dict"):
                ind_df.round(6).to_csv(out_dir / f"individuals_{pos}.csv", index=False)
                ind_rows = ind_df.to_dict("records")
            else:
                ind_rows = ind_df
                write_rows_csv(ind_rows, out_dir / f"individuals_{pos}.csv", _IND_COLS, ndigits=6)

            pop_res, pooled_sd = compute_population_measures(
                info_adm,
                q_adm,
                sorted(main_link_by_pop.keys()),
                priority_populations=ref_cat12_pops,
                pooled_ref_populations=ref_cat12_pops if ref_cat12_pops else None,
                n_boot=N_BOOT,
                seed=SEED,
                ks=KS,
                f3_path=COMMON_DATA_DIR / "f3_scores.csv",
                k5_path=COMMON_DATA_DIR / "f3_scores_k5.csv",
                include_extended_metadata=True,
                include_fixed_f3=True,
                include_fst_regimes=True,
                warn_missing_f3=True,
            )
            pop_raw_rows = pop_res.to_dict("records") if hasattr(pop_res, "to_dict") else list(pop_res)

            pop_rows = []
            for pr in pop_raw_rows:
                pname = str(pr["population"])
                ml = main_link_by_pop[pname]
                merged_p: dict[str, object] = {
                    "population": pname,
                    "min_pop2": "" if _is_missing_scalar(pr.get("min_pop2")) else str(pr.get("min_pop2")),
                    "min_pop3": "" if _is_missing_scalar(pr.get("min_pop3")) else str(pr.get("min_pop3")),
                    "glottocode": ml["glottocode"],
                    "link_basis": ml["link_basis"],
                    "accepted_by": ml["accepted_by"],
                    "sensitivity_flags": ml.get("sensitivity_flags", ""),
                }
                for ic in _POP_INT_RAW_COLS:
                    merged_p[ic] = _to_int_or_none(pr.get(ic))
                for fc in _POP_FLT_RAW_COLS:
                    fv = _to_float_or_nan(pr.get(fc))
                    merged_p[fc] = fv if math.isnan(fv) else round(fv, 7)
                for s_name in SETS:
                    merged_p[f"in_{s_name}"] = 1 if pname in in_set[s_name] else 0
                pop_rows.append(merged_p)
            write_rows_csv(pop_rows, out_dir / f"populations_{pos}.csv", _POP_COLS, ndigits=7)
        else:
            ind_rows, pop_rows, pooled_sd, cached_lang_ent = _load_cached_genetics_for_pos(
                pos=pos,
                main_link_by_pop=main_link_by_pop,
                in_set=in_set,
                ref_cat12_pops=ref_cat12_pops,
                out_dir=out_dir,
                fallback_prepared_dir=fallback_prepared_dir,
            )

        # Build Glottolog topology Newick tree
        paths_gl, tree_src = resolve_glottolog_paths(linked_gcs, root_dir=ROOT)
        (out_dir / f"phylo_tree_{pos}.nwk").write_text(build_newick(paths_gl) + "\n", encoding="utf-8")

        # Resolve audited L1 speaker population & range area
        demog_by_gc = {
            str(r["glottocode"]): r for r in resolve_demography_rows(linked_gcs)
        }

        pops_by_gc: dict[str, list[dict[str, object]]] = defaultdict(list)
        for pr in pop_rows:
            pops_by_gc[str(pr["glottocode"])].append(pr)
        inds_by_gc: dict[str, list[dict[str, object]]] = defaultdict(list)
        for ir in ind_rows:
            inds_by_gc[str(ir["glottocode"])].append(ir)

        f3_maxp_raw = compute_language_f3_maxp(
            [p for p in pop_rows if int(p["in_main"]) == 1],
            col="f3_fixed",
            min_n_ind=DEFAULT_MIN_N_IND,
        )
        f3_maxp_list = f3_maxp_raw.to_dict("records") if hasattr(f3_maxp_raw, "to_dict") else list(f3_maxp_raw)
        f3_maxp_by_gc = {str(r["glottocode"]): _to_float_or_nan(r.get("f3_fixed_maxp")) for r in f3_maxp_list}

        lang_cols = [
            "glottocode", "unit_id", "pos", "name", "family", "family_id", "macroarea", "lat", "lon",
            "n_populations", "n_individuals",
            "in_main", "n_individuals_main",
            "in_no_doubtful_language", "n_individuals_no_doubtful_language",
            "in_no_other_language_stand_ins", "n_individuals_no_other_language_stand_ins",
            "in_strict", "n_individuals_strict",
            "ent", "nonmax",
            "f3_fixed_maxp", "median_fst_all", "barbieri_mismatch", "regime_a_enclave", "regime_b_shift",
            "tree_source",
            *DEMOG_COLS,
            "source_id", "source_prov", "n_avail", "log_navail",
            "n_cells_distinct", "log_C_distinct", "n_test_lemmas", "base_cell_modal",
            "h_tree_bits", "h_tree_se", "rel_tree_mean", "rel_tree_se",
            "acc_tree_mean", "acc_tree_se",
            "h_k1_bits", "h_k1_se", "acc_k1_mean",
            "h_k3_bits", "h_k3_se", "acc_k3_mean",
            "has_uniform480", "h_tree_480_bits", "h_tree_480_se",
            "has_topfreq120", "h_tree_topfreq_bits", "h_tree_topfreq_se",
            "has_phon", "h_tree_phon_bits", "h_tree_phon_se",
            "has_malouf", "h_tree_malouf_bits", "h_tree_malouf_se",
            "has_neural", "h_tree_neural_bits", "h_tree_neural_se",
            "h_tree_neural_480_bits", "h_tree_neural_480_se",
            "in_track1",
            "v_breadth_present", "v_breadth_coded", "v_breadth_prop",
            "n_breadth_present", "n_breadth_coded", "n_breadth_prop",
            "v_irreg_present", "v_irreg_coded", "v_irreg_prop",
            "n_irreg_present", "n_irreg_coded", "n_irreg_prop",
            "np_agr_present", "np_agr_coded", "np_agr_prop",
            "cat12_present", "cat12_coded",
            "irreg13_present", "irreg13_coded", "irreg13_prop",
            "irreg_verb_prop", "irreg_noun_prop", "theta_irreg13_eap",
        ]

        lang_rows: list[dict[str, object]] = []
        active_uid_to_gc: dict[str, str] = {}

        for gc in linked_gcs:
            u_row = chosen_by_gc[gc]
            uid = u_row["unit_id"]
            active_uid_to_gc[uid] = gc
            um = unit_meta_by_id.get(uid, {})
            gm = glotto_meta.get(gc, {})
            dm = demog_by_gc[gc]
            p_list = pops_by_gc[gc]
            i_list = inds_by_gc[gc]

            sid = (
                um.get("source_id")
                or u_row.get("source_id")
                or ""
            )
            sprov = classify_source_provenance(sid, tier_by_source.get(sid, ""))

            # Available root-grouped lemma count `n_avail` from unit_summary (fallback n_lemmas / n_test_lemmas)
            n_avail = (
                _to_int_or_none(um.get("n_leakage_groups"))
                or _to_int_or_none(um.get("n_groups"))
                or _to_int_or_none(um.get("n_lemmas_complete"))
                or _to_int_or_none(um.get("n_lemmas"))
                or _to_int_or_none(u_row.get("n_test_lemmas"))
                or 150
            )
            n_cells = int(float(u_row["n_cells_distinct"]))

            r480 = mgl_480_by_uid.get(uid)
            rtop = mgl_topf_by_uid.get(uid)
            rphon = phon_by_gc_pos.get((gc, pos))
            rmal = malouf_by_uid.get(uid) or malouf_by_gc_pos.get((gc, pos))
            rneu = neu_120_by_uid.get(uid)
            rneu480 = neu_480_by_uid.get(uid)
            t1 = track1_map.get(gc, {})

            row_l: dict[str, object] = {
                "glottocode": gc,
                "unit_id": uid,
                "pos": pos,
                "name": gm.get("name", ""),
                "family": gm.get("family", "isolate"),
                "family_id": gm.get("family_id", "isolate"),
                "macroarea": gm.get("macroarea", ""),
                "lat": gm.get("lat", float("nan")),
                "lon": gm.get("lon", float("nan")),
                "n_populations": len(p_list),
                "n_individuals": sum(int(p["n_individuals"]) for p in p_list),
            }
            for s_name in SETS:
                sub_p = [p for p in p_list if int(p[f"in_{s_name}"]) == 1]
                row_l[f"in_{s_name}"] = 1 if sub_p else 0
                row_l[f"n_individuals_{s_name}"] = sum(int(p["n_individuals"]) for p in sub_p)

            mean_ent = sum(float(x["ent"]) for x in i_list) / len(i_list)
            mean_nmx = sum(float(x["nonmax"]) for x in i_list) / len(i_list)
            if gc in cached_lang_ent:
                c_n, c_ent, c_nmx = cached_lang_ent[gc]
                if c_n == len(i_list) and abs(c_ent - mean_ent) <= 2e-6 and abs(c_nmx - mean_nmx) <= 2e-6:
                    mean_ent, mean_nmx = c_ent, c_nmx
            row_l["ent"] = mean_ent
            row_l["nonmax"] = mean_nmx
            p_n3 = [
                p for p in p_list
                if int(p["in_main"]) == 1 and (_to_int_or_none(p.get("n_individuals")) or 0) >= DEFAULT_MIN_N_IND
            ]
            med_fst_vals = [
                _to_float_or_nan(p.get("median_fst_all"))
                for p in p_n3
                if not math.isnan(_to_float_or_nan(p.get("median_fst_all")))
            ]
            row_l["f3_fixed_maxp"] = f3_maxp_by_gc.get(gc, float("nan"))
            row_l["median_fst_all"] = (sum(med_fst_vals) / len(med_fst_vals)) if med_fst_vals else float("nan")
            for flag_col in ("barbieri_mismatch", "regime_a_enclave", "regime_b_shift"):
                f_vals = [_to_int_or_none(p.get(flag_col)) for p in p_n3]
                f_valid = [v for v in f_vals if v is not None]
                row_l[flag_col] = (1 if any(v == 1 for v in f_valid) else 0) if f_valid else None
            row_l["tree_source"] = tree_src[gc]
            for dc in DEMOG_COLS:
                row_l[dc] = dm[dc]

            row_l.update({
                "source_id": sid,
                "source_prov": sprov,
                "n_avail": n_avail,
                "log_navail": math.log(max(1, n_avail)),
                "n_cells_distinct": n_cells,
                "log_C_distinct": math.log(max(2, n_cells)),
                "n_test_lemmas": int(float(u_row["n_test_lemmas"])),
                "base_cell_modal": u_row.get("base_cell_modal", ""),
                "h_tree_bits": float(u_row["h_tree_bits"]),
                "h_tree_se": float(u_row["h_tree_se"]),
                "rel_tree_mean": float(u_row["rel_tree_mean"]),
                "rel_tree_se": float(u_row["rel_tree_se"]),
                "acc_tree_mean": float(u_row["acc_tree_mean"]),
                "acc_tree_se": float(u_row["acc_tree_se"]),
                "h_k1_bits": float(u_row["h_k1_bits"]),
                "h_k1_se": float(u_row["h_k1_se"]),
                "acc_k1_mean": float(u_row["acc_k1_mean"]),
                "h_k3_bits": float(u_row["h_k3_bits"]),
                "h_k3_se": float(u_row["h_k3_se"]),
                "acc_k3_mean": float(u_row["acc_k3_mean"]),
                "has_uniform480": 1 if r480 else 0,
                "h_tree_480_bits": float(r480["h_tree_bits"]) if r480 else float("nan"),
                "h_tree_480_se": float(r480["h_tree_se"]) if r480 else float("nan"),
                "has_topfreq120": 1 if rtop else 0,
                "h_tree_topfreq_bits": float(rtop["h_tree_bits"]) if rtop else float("nan"),
                "h_tree_topfreq_se": float(rtop["h_tree_se"]) if rtop else float("nan"),
                "has_phon": 1 if rphon else 0,
                "h_tree_phon_bits": float(rphon["h_tree_bits"]) if rphon else float("nan"),
                "h_tree_phon_se": float(rphon["h_tree_se"]) if rphon else float("nan"),
                "has_malouf": 1 if rmal else 0,
                "h_tree_malouf_bits": float(rmal["h_tree_bits"]) if rmal else float("nan"),
                "h_tree_malouf_se": float(rmal["h_tree_se"]) if rmal else float("nan"),
                "has_neural": 1 if rneu else 0,
                "h_tree_neural_bits": float(rneu["h_tree_bits"]) if rneu else float("nan"),
                "h_tree_neural_se": float(rneu["h_tree_se"]) if rneu else float("nan"),
                "h_tree_neural_480_bits": float(rneu480["h_tree_bits"]) if rneu480 else float("nan"),
                "h_tree_neural_480_se": float(rneu480["h_tree_se"]) if rneu480 else float("nan"),
                "in_track1": t1.get("in_track1", 0),
                "v_breadth_present": t1.get("v_breadth_present", None),
                "v_breadth_coded": t1.get("v_breadth_coded", None),
                "v_breadth_prop": t1.get("v_breadth_prop", float("nan")),
                "n_breadth_present": t1.get("n_breadth_present", None),
                "n_breadth_coded": t1.get("n_breadth_coded", None),
                "n_breadth_prop": t1.get("n_breadth_prop", float("nan")),
                "v_irreg_present": t1.get("v_irreg_present", None),
                "v_irreg_coded": t1.get("v_irreg_coded", None),
                "v_irreg_prop": t1.get("v_irreg_prop", float("nan")),
                "n_irreg_present": t1.get("n_irreg_present", None),
                "n_irreg_coded": t1.get("n_irreg_coded", None),
                "n_irreg_prop": t1.get("n_irreg_prop", float("nan")),
                "np_agr_present": t1.get("np_agr_present", None),
                "np_agr_coded": t1.get("np_agr_coded", None),
                "np_agr_prop": t1.get("np_agr_prop", float("nan")),
                "cat12_present": t1.get("cat12_present", None),
                "cat12_coded": t1.get("cat12_coded", None),
                "irreg13_present": t1.get("irreg13_present", None),
                "irreg13_coded": t1.get("irreg13_coded", None),
                "irreg13_prop": t1.get("irreg13_prop", float("nan")),
                "irreg_verb_prop": t1.get("irreg_verb_prop", float("nan")),
                "irreg_noun_prop": t1.get("irreg_noun_prop", float("nan")),
                "theta_irreg13_eap": t1.get("theta_irreg13_eap", float("nan")),
            })
            lang_rows.append(row_l)

        # Strict 1-row-per-glottocode invariant check
        assert len({r["glottocode"] for r in lang_rows}) == len(lang_rows)
        write_rows_csv(lang_rows, out_dir / f"language_data_{pos}.csv", lang_cols, ndigits=6)

        # Build two-level entrenchment table `lemma_freq_data_{pos}.csv`
        lemma_by_uid: dict[str, list[dict[str, str]]] = defaultdict(list)
        for lr in mgl_lemma_120:
            if lr["unit_id"] in active_uid_to_gc:
                lemma_by_uid[lr["unit_id"]].append(lr)

        lf_cols = [
            "glottocode", "unit_id", "pos", "lemma_id", "leakage_group_id",
            "n_distinct_forms", "h_tree_lemma", "rel_tree_lemma", "acc_tree_lemma",
            "h_k1_lemma", "h_k3_lemma",
            "freq_lemma_tokens", "freq_lemma_pmw", "log10_freq_lemma",
            "is_attested", "n_cells_attested", "primary_freq_source",
            "c_log10_freq_within", "z_freq_within",
        ]
        lf_rows: list[dict[str, object]] = []
        n_langs_with_freq = 0

        for uid, gc in sorted(active_uid_to_gc.items(), key=lambda kv: kv[1]):
            freq_file = paradigm_data_dir / "frequencies" / f"{uid}_freq.csv"
            if not freq_file.exists():
                continue
            freq_by_lid = {r["lemma_id"]: r for r in _read_csv_optional(freq_file)}
            matched_unit_rows: list[dict[str, object]] = []
            for lr in lemma_by_uid.get(uid, []):
                fr = freq_by_lid.get(lr["lemma_id"])
                if fr is None:
                    continue
                tok = _to_float_or_nan(fr.get("freq_lemma_tokens"))
                l10 = _to_float_or_nan(fr.get("log10_freq_lemma"))
                if math.isnan(l10):
                    continue
                matched_unit_rows.append({
                    "glottocode": gc,
                    "unit_id": uid,
                    "pos": pos,
                    "lemma_id": lr["lemma_id"],
                    "leakage_group_id": lr["leakage_group_id"],
                    "n_distinct_forms": int(float(lr["n_distinct_forms"])),
                    "h_tree_lemma": float(lr["h_tree_lemma"]),
                    "rel_tree_lemma": float(lr["rel_tree_lemma"]),
                    "acc_tree_lemma": float(lr["acc_tree_lemma"]),
                    "h_k1_lemma": float(lr["h_k1_lemma"]),
                    "h_k3_lemma": float(lr["h_k3_lemma"]),
                    "freq_lemma_tokens": 0.0 if math.isnan(tok) else tok,
                    "freq_lemma_pmw": _to_float_or_nan(fr.get("freq_lemma_pmw")),
                    "log10_freq_lemma": l10,
                    "is_attested": 1 if (not math.isnan(tok) and tok > 0.0) else 0,
                    "n_cells_attested": _to_int_or_none(fr.get("n_cells_attested")) or 0,
                    "primary_freq_source": fr.get("primary_freq_source", ""),
                })

            # Require >= 10 frequency-matched lemmas and non-zero frequency variance within language
            n_att = sum(int(r["is_attested"]) for r in matched_unit_rows)
            if len(matched_unit_rows) >= 10 and n_att >= 5:
                vals = np.array([float(r["log10_freq_lemma"]) for r in matched_unit_rows], dtype=np.float64)
                sd_v = float(np.std(vals, ddof=1))
                if sd_v > 1e-8:
                    mu_v = float(np.mean(vals))
                    for r in matched_unit_rows:
                        fv = float(r["log10_freq_lemma"])
                        r["c_log10_freq_within"] = fv - mu_v
                        r["z_freq_within"] = (fv - mu_v) / sd_v
                    lf_rows.extend(matched_unit_rows)
                    n_langs_with_freq += 1

        write_rows_csv(lf_rows, out_dir / f"lemma_freq_data_{pos}.csv", lf_cols, ndigits=6)

        prep_summary["pos"][pos] = {
            "n_languages_main": len(lang_rows),
            "n_families_main": len({str(r["family"]) for r in lang_rows}),
            "sets": {
                s_name: {
                    "languages": sum(int(r[f"in_{s_name}"]) for r in lang_rows),
                    "populations": sum(int(p[f"in_{s_name}"]) for p in pop_rows),
                    "individuals": sum(
                        int(p["n_individuals"]) for p in pop_rows if int(p[f"in_{s_name}"]) == 1
                    ),
                }
                for s_name in SETS
            },
            "n_has_pop": sum(int(r["has_pop"]) for r in lang_rows),
            "n_has_area": sum(int(r["has_area"]) for r in lang_rows),
            "n_uniform480": sum(int(r["has_uniform480"]) for r in lang_rows),
            "n_topfreq120": sum(int(r["has_topfreq120"]) for r in lang_rows),
            "n_neural": sum(int(r["has_neural"]) for r in lang_rows),
            "n_track1_overlap": sum(int(r["in_track1"]) for r in lang_rows),
            "n_track1_domains": {
                dom: sum(
                    1
                    for r in lang_rows
                    if not math.isnan(_to_float_or_nan(r.get(f"{dom}_prop")))
                )
                for dom, _, _ in TRACK1_DOMAIN_GATES
            },
            "n_languages_entrenchment": n_langs_with_freq,
            "n_lemmas_entrenchment": len(lf_rows),
            "pooled_per_individual_sd": pooled_sd,
        }

    (out_dir / "stage2_prep_summary.json").write_text(
        json.dumps(prep_summary, indent=2) + "\n", encoding="utf-8"
    )
    return prep_summary


def build_smoke_test_inputs(workspace_dir: Path) -> Path:
    """Create a synthetic-outcome smoke fixture on real GeLaTo Glottocodes inside `workspace_dir`.

    Used ONLY for unit testing (`tests/test_paradigm_stage2.py`) and `PARA_DRY=1` Stan compilation
    validation when `workspace_dir` is a temporary directory; never writes to production folders.
    """
    data_dir = workspace_dir / "paradigm_data"
    mgl_dir = workspace_dir / "mgl_results"
    neu_dir = workspace_dir / "neu_results"
    out_dir = workspace_dir / "prepared"
    (data_dir / "frequencies").mkdir(parents=True, exist_ok=True)
    mgl_dir.mkdir(parents=True, exist_ok=True)
    neu_dir.mkdir(parents=True, exist_ok=True)

    # 18 real GeLaTo Glottocodes spanning multiple families (IE, Uralic, Turkic, Mongolic, Kartvelian, AA)
    fixture_langs = [
        ("ital1282", "paralex_leffi", "1_paralex", 800, 18),
        ("stan1290", "paralex_vlexique", "1_paralex", 750, 16),
        ("stan1288", "paralex_velespa", "1_paralex", 900, 20),
        ("russ1263", "paralex_zalilex", "1_paralex", 1200, 12),
        ("poli1260", "national_polimorf", "2_national", 1100, 14),
        ("ukra1253", "national_vesum", "2_national", 950, 14),
        ("finn1318", "paralex_parafin", "1_paralex", 700, 24),
        ("esto1258", "paralex_eesthetic", "1_paralex", 680, 22),
        ("hung1274", "paralex_parahungarian", "1_paralex", 620, 18),
        ("udmu1245", "sigmorphon2020_uralic", "3_sigmorphon", 260, 15),
        ("nucl1301", "unimorph_tur", "3_unimorph", 650, 28),
        ("nort2697", "unimorph_azj", "3_unimorph", 320, 20),
        ("bash1264", "unimorph_bak", "3_unimorph", 290, 18),
        ("halh1238", "unimorph_khk", "3_unimorph", 240, 12),
        ("russ1264", "unimorph_bua", "3_unimorph", 210, 10),
        ("nucl1302", "unimorph_kat", "3_unimorph", 640, 16),
        ("nucl1235", "unimorph_hye", "3_unimorph", 610, 16),
        ("malt1254", "unimorph_mlt", "3_unimorph", 280, 10),
    ]

    rng = np.random.default_rng(20261112)
    prov_rows = []
    sum_rows = []
    u120_rows = []
    u480_rows = []
    topf_rows = []
    neu_rows = []
    lemma_rows = []

    for pos in ("V", "N"):
        for gc, sid, tier, n_avail, n_cells in fixture_langs:
            uid = f"{gc}_{pos}_orth"
            prov_rows.append({"source_id": sid, "priority_tier": tier})
            sum_rows.append({
                "unit_id": uid,
                "glottocode": gc,
                "pos": pos,
                "representation": "orth",
                "source_id": sid,
                "n_leakage_groups": n_avail,
                "n_cells_distinct": n_cells,
            })
            h_val = float(rng.uniform(0.3, 2.4))
            se_val = float(rng.uniform(0.04, 0.18))
            base_u = {
                "unit_id": uid,
                "glottocode": gc,
                "pos": pos,
                "representation": "orth",
                "model": "mgl",
                "regime": "uniform120",
                "n_draws": 5,
                "n_test_lemmas": 150,
                "n_cells_distinct": n_cells,
                "h_tree_bits": round(h_val, 5),
                "h_tree_se": round(se_val, 5),
                "rel_tree_mean": round(float(rng.uniform(0.55, 0.95)), 5),
                "rel_tree_se": 0.03,
                "acc_tree_mean": round(float(rng.uniform(0.60, 0.96)), 5),
                "acc_tree_se": 0.03,
                "h_k1_bits": round(h_val * 1.15, 5),
                "h_k1_se": round(se_val * 1.05, 5),
                "acc_k1_mean": 0.78,
                "h_k3_bits": round(h_val * 0.85, 5),
                "h_k3_se": round(se_val * 0.95, 5),
                "acc_k3_mean": 0.88,
                "base_cell_modal": "C0",
                "runtime_sec": 1.2,
            }
            u120_rows.append(base_u)
            if n_avail >= 600:
                u480_rows.append({**base_u, "regime": "uniform480", "h_tree_bits": round(h_val * 0.88, 5)})
            topf_rows.append({**base_u, "regime": "topfreq120", "h_tree_bits": round(h_val * 1.06, 5)})
            neu_rows.append({**base_u, "model": "neural_monotonic", "h_tree_bits": round(h_val * 0.97, 5)})

            f_rows = []
            for l_idx in range(25):
                lid = f"{uid}_lem{l_idx:02d}"
                grp = f"{uid}_grp{l_idx:02d}"
                tok = float(rng.lognormal(mean=2.5, sigma=1.8)) if l_idx < 22 else 0.0
                l10 = math.log10((tok + 1.0))
                h_lem = max(0.01, h_val + 0.25 * (l10 - 1.2) + float(rng.normal(0.0, 0.25)))
                lemma_rows.append({
                    "unit_id": uid,
                    "glottocode": gc,
                    "pos": pos,
                    "representation": "orth",
                    "model": "mgl",
                    "regime": "uniform120",
                    "lemma_id": lid,
                    "leakage_group_id": grp,
                    "n_distinct_forms": n_cells,
                    "h_tree_lemma": round(h_lem, 5),
                    "rel_tree_lemma": 0.8,
                    "acc_tree_lemma": 0.85,
                    "h_k1_lemma": round(h_lem * 1.1, 5),
                    "h_k3_lemma": round(h_lem * 0.85, 5),
                })
                f_rows.append({
                    "unit_id": uid,
                    "lemma_id": lid,
                    "leakage_group_id": grp,
                    "freq_lemma_tokens": round(tok, 4),
                    "freq_lemma_pmw": round(tok * 10.0, 4),
                    "log10_freq_lemma": round(l10, 5),
                    "n_cells_attested": 4 if tok > 0 else 0,
                    "primary_freq_source": "lcc" if tok > 0 else "unattested",
                })
            write_rows_csv(
                f_rows,
                data_dir / "frequencies" / f"{uid}_freq.csv",
                list(f_rows[0].keys()),
            )

    dedup_prov = list({r["source_id"]: r for r in prov_rows}.values())
    write_rows_csv(dedup_prov, data_dir / "provenance_manifest.csv", ["source_id", "priority_tier"])
    write_rows_csv(sum_rows, data_dir / "unit_summary.csv", list(sum_rows[0].keys()))
    write_rows_csv(u120_rows, mgl_dir / "unit_outcomes_uniform120.csv", list(u120_rows[0].keys()))
    write_rows_csv(u480_rows, mgl_dir / "unit_outcomes_uniform480.csv", list(u480_rows[0].keys()))
    write_rows_csv(topf_rows, mgl_dir / "unit_outcomes_topfreq120.csv", list(topf_rows[0].keys()))
    write_rows_csv(lemma_rows, mgl_dir / "lemma_outcomes_uniform120.csv", list(lemma_rows[0].keys()))
    write_rows_csv(neu_rows, neu_dir / "unit_outcomes_uniform120.csv", list(neu_rows[0].keys()))

    prepare_stage2_inputs(
        paradigm_data_dir=data_dir,
        mgl_results_dir=mgl_dir,
        neural_results_dir=neu_dir,
        out_dir=out_dir,
    )
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage-2 Bayesian inputs for Track 2.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_PARADIGM_DATA_DIR)
    parser.add_argument("--mgl-results-dir", type=Path, default=DEFAULT_MGL_RESULTS_DIR)
    parser.add_argument("--neural-results-dir", type=Path, default=DEFAULT_NEURAL_RESULTS_DIR)
    parser.add_argument("--links-csv", type=Path, default=DEFAULT_LINKS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--adm-dir", type=Path, default=None, help="Optional ADMIXTURE directory override.")
    parser.add_argument("--pos", nargs="+", default=["V", "N"], choices=["V", "N"])
    parser.add_argument(
        "--build-smoke-fixture",
        type=Path,
        default=None,
        help="Optional temporary directory in which to build synthetic smoke-test Stage-2 inputs.",
    )
    args = parser.parse_args(argv)

    if args.build_smoke_fixture is not None:
        out_p = build_smoke_test_inputs(args.build_smoke_fixture)
        print("Built smoke-test Stage-2 fixture inputs in:", out_p)
        return

    summ = prepare_stage2_inputs(
        paradigm_data_dir=args.data_dir,
        mgl_results_dir=args.mgl_results_dir,
        neural_results_dir=args.neural_results_dir,
        links_csv=args.links_csv,
        out_dir=args.out_dir,
        pos_list=args.pos,
        adm_dir=args.adm_dir,
    )
    print("Prepared Track 2 Stage-2 inputs in:", args.out_dir)
    for p_key, p_info in summ.get("pos", {}).items():  # type: ignore[union-attr]
        print(f"  POS={p_key}: {p_info}")


if __name__ == "__main__":
    main()
