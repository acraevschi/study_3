"""End-to-end paradigm predictability evaluator for Multi-Locus Vowel-Tier MGL (`evaluate_mgl.py`).

Enforces strict CPU thread caps (`OMP_NUM_THREADS=1`, etc.), zero `leakage_group_id` / `test`
leakage during Chu-Liu-Edmonds arborescence & PoE weight estimation (`subtrain -> dev`),
full `train` (`subtrain U dev`) refitting before scoring held-out `test` lemmas, and
500-resample `leakage_group_id` cluster bootstrap standard errors.
"""

from __future__ import annotations

import os

# MANDATORY CPU THREAD SEMAPHORE (must precede numpy / pandas imports!)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import concurrent.futures
import csv
import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .escape import DEFAULT_GAMMA_ESCAPE, DirichletEscapeScorer
from .topology import (
    PRESCREEN_CELL_THRESHOLD,
    PRESCREEN_TOP_PARENTS,
    LemmaTopologyScores,
    evaluate_lemma_topology,
    prescreen_candidate_edges,
    select_optimal_arborescence,
)
from .vowel_tier import SoundInventory, VowelClassifier, parse_segments

UNIT_OUTCOME_COLUMNS: tuple[str, ...] = (
    "unit_id",
    "glottocode",
    "pos",
    "representation",
    "model",
    "regime",
    "n_draws",
    "n_test_lemmas",
    "n_cells_distinct",
    "h_tree_bits",
    "h_tree_se",
    "rel_tree_mean",
    "rel_tree_se",
    "acc_tree_mean",
    "acc_tree_se",
    "h_k1_bits",
    "h_k1_se",
    "acc_k1_mean",
    "h_k3_bits",
    "h_k3_se",
    "acc_k3_mean",
    "base_cell_modal",
    "runtime_sec",
)

LEMMA_OUTCOME_COLUMNS: tuple[str, ...] = (
    "unit_id",
    "glottocode",
    "pos",
    "representation",
    "model",
    "regime",
    "lemma_id",
    "leakage_group_id",
    "n_distinct_forms",
    "h_tree_lemma",
    "rel_tree_lemma",
    "acc_tree_lemma",
    "h_k1_lemma",
    "h_k3_lemma",
)

BOOTSTRAP_RESAMPLES: int = 500
MAX_HARD_WORKERS_CAP: int = 16
DYNAMIC_TIGHTEN_SEC_THRESHOLD: float = 240.0


@dataclass(frozen=True)
class UnitLexicon:
    """Loaded canonical (`variant_idx == 0`) synthetic paradigm table for one `unit_id`."""

    unit_id: str
    glottocode: str
    pos: str
    representation: str
    cells: tuple[str, ...]
    paradigms: dict[str, dict[str, tuple[str, ...]]]
    lemma_to_group: dict[str, str]
    sound_inventory: SoundInventory | None


@dataclass(frozen=True)
class SplitFold:
    """Validated `(draw, fold)` split with disjoint `subtrain`, `dev`, and `test` leakage groups."""

    draw: int
    fold: int
    subtrain_lemmas: tuple[str, ...]
    dev_lemmas: tuple[str, ...]
    train_lemmas: tuple[str, ...]
    test_lemmas: tuple[str, ...]


def load_unit_lexicon(
    lexicon_csv: str | Path,
    sounds_csv: str | Path | None = None,
    lemma_filter: set[str] | None = None,
) -> UnitLexicon:
    """Load `analyses/paradigm/data/lexicons/{unit_id}.csv` (`variant_idx == 0` canonical forms).

    When `sounds_csv` is not explicitly provided, checks `lex_path.with_name("{stem}_sounds.csv")`
    first, and for `phon` units falls back to `lex_path.parent.parent / "cell_maps" / "{stem}_sounds.csv"`
    so Paralex `phon` feature inventories (`[+syllabic]` & `C'_feat`) load automatically without
    contaminating paired `orth` units with IPA phoneme tables.
    """
    lex_path = Path(lexicon_csv)

    unit_id = ""
    glottocode = ""
    pos = ""
    representation = "orth"
    cells_seen: set[str] = set()
    paradigms: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    lemma_to_group: dict[str, str] = {}

    with lex_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lid = str(row["lemma_id"]).strip()
            if lemma_filter is not None and lid not in lemma_filter:
                if not unit_id:
                    unit_id = str(row["unit_id"]).strip()
                    glottocode = str(row["glottocode"]).strip()
                    pos = str(row["pos"]).strip()
                    representation = str(row.get("representation", "orth") or "orth").strip()
                continue
            v_idx_str = str(row.get("variant_idx", "0") or "0").strip()
            if int(v_idx_str) != 0:
                continue
            raw_f = str(row.get("raw_form", "") or "").strip()
            seg_f = str(row.get("segmented_form", "") or "").strip()
            if raw_f == "#DEF#" or seg_f == "#DEF#":
                continue

            if not unit_id:
                unit_id = str(row["unit_id"]).strip()
                glottocode = str(row["glottocode"]).strip()
                pos = str(row["pos"]).strip()
                representation = str(row.get("representation", "orth") or "orth").strip()

            gid = str(row["leakage_group_id"]).strip()
            cell = str(row["cell"]).strip()
            segs = parse_segments(seg_f, raw_f)
            if not segs:
                continue

            cells_seen.add(cell)
            paradigms[lid][cell] = segs
            lemma_to_group[lid] = gid

    resolved_sounds: Path | None = Path(sounds_csv) if sounds_csv is not None else None
    if resolved_sounds is None:
        auto_sounds = lex_path.with_name(f"{lex_path.stem}_sounds.csv")
        if auto_sounds.exists():
            resolved_sounds = auto_sounds
        elif representation == "phon":
            cell_maps_sounds = lex_path.parent.parent / "cell_maps" / f"{lex_path.stem}_sounds.csv"
            if cell_maps_sounds.exists():
                resolved_sounds = cell_maps_sounds

    sound_inv = (
        SoundInventory.from_csv(resolved_sounds)
        if resolved_sounds is not None and resolved_sounds.exists()
        else None
    )

    sorted_cells = tuple(sorted(cells_seen))
    if len(sorted_cells) < 2:
        raise ValueError(f"Unit {unit_id or lex_path.stem} has < 2 distinct cells ({len(sorted_cells)}).")

    # Retain complete canonical paradigms across `sorted_cells`
    complete_paradigms = {
        lid: c_map
        for lid, c_map in paradigms.items()
        if len(c_map) == len(sorted_cells)
    }
    complete_groups = {lid: lemma_to_group[lid] for lid in complete_paradigms}

    return UnitLexicon(
        unit_id=unit_id or lex_path.stem,
        glottocode=glottocode,
        pos=pos,
        representation=representation,
        cells=sorted_cells,
        paradigms=complete_paradigms,
        lemma_to_group=complete_groups,
        sound_inventory=sound_inv,
    )


def load_split_manifest(
    split_csv: str | Path,
    lexicon: UnitLexicon,
) -> list[SplitFold]:
    """Load `{unit_id}_{regime}.csv` and strictly verify zero `leakage_group_id` leakage."""
    split_path = Path(split_csv)
    grouped_rows: dict[tuple[int, int], list[tuple[str, str, str, str]]] = defaultdict(list)

    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty split manifest: {split_path}")
        fields = set(reader.fieldnames)
        draw_col = next(
            (c for c in ("draw_idx", "draw", "draw_id", "repeat") if c in fields),
            None,
        )
        if draw_col is None:
            raise ValueError(
                f"Split manifest {split_path} missing draw column; expected one of "
                f"('draw_idx', 'draw', 'draw_id', 'repeat'), got {sorted(fields)}"
            )
        fold_col = next(
            (c for c in ("fold_idx", "fold", "fold_id") if c in fields),
            None,
        )
        if fold_col is None:
            raise ValueError(
                f"Split manifest {split_path} missing fold column; expected one of "
                f"('fold_idx', 'fold', 'fold_id'), got {sorted(fields)}"
            )

        for row in reader:
            d_idx = int(str(row[draw_col]).strip())
            f_idx = int(str(row[fold_col]).strip())
            lid = str(row["lemma_id"]).strip()
            role = str(row["role"]).strip().lower()
            inner = str(row.get("inner_role", role) or role).strip().lower()
            gid = str(row.get("leakage_group_id", "") or lexicon.lemma_to_group.get(lid, "")).strip()
            grouped_rows[(d_idx, f_idx)].append((lid, gid, role, inner))

    folds: list[SplitFold] = []
    for (d_idx, f_idx) in sorted(grouped_rows.keys()):
        entries = grouped_rows[(d_idx, f_idx)]
        subtrain: list[str] = []
        dev: list[str] = []
        train: list[str] = []
        test: list[str] = []

        train_groups: set[str] = set()
        test_groups: set[str] = set()
        subtrain_groups: set[str] = set()
        dev_groups: set[str] = set()

        for lid, gid, role, inner in entries:
            if lid not in lexicon.paradigms:
                raise KeyError(
                    f"Split {split_path.name} (draw={d_idx}, fold={f_idx}) references missing/incomplete lemma {lid!r}"
                )
            eff_gid = lexicon.lemma_to_group[lid]
            if gid and gid != eff_gid:
                raise ValueError(
                    f"Leakage group mismatch for lemma {lid!r}: split={gid!r} vs lexicon={eff_gid!r}"
                )

            if role == "test" or inner == "test":
                test.append(lid)
                test_groups.add(eff_gid)
            elif role == "train":
                train.append(lid)
                train_groups.add(eff_gid)
                if inner == "subtrain":
                    subtrain.append(lid)
                    subtrain_groups.add(eff_gid)
                elif inner == "dev":
                    dev.append(lid)
                    dev_groups.add(eff_gid)
                else:
                    raise ValueError(f"Unknown inner_role {inner!r} for train lemma {lid!r}")
            else:
                raise ValueError(f"Unknown role {role!r} for lemma {lid!r}")

        # CRITICAL ZERO-LEAKAGE CHECKS
        overlap_train_test = train_groups & test_groups
        if overlap_train_test:
            raise ValueError(
                f"CRITICAL LEAKAGE in {split_path.name} (draw={d_idx}, fold={f_idx}): "
                f"train and test share leakage_group_ids {sorted(overlap_train_test)[:5]}"
            )
        if set(train) & set(test):
            raise ValueError(f"Train and test share lemma_ids in {split_path.name}")
        if set(subtrain) & set(dev):
            raise ValueError(f"Subtrain and dev share lemma_ids in {split_path.name}")
        if subtrain_groups & dev_groups:
            raise ValueError(f"Subtrain and dev share leakage_group_ids in {split_path.name}")
        if not subtrain or not dev or not test:
            raise ValueError(
                f"Degenerate split in {split_path.name} (draw={d_idx}, fold={f_idx}): "
                f"n_subtrain={len(subtrain)}, n_dev={len(dev)}, n_test={len(test)}"
            )

        folds.append(
            SplitFold(
                draw=d_idx,
                fold=f_idx,
                subtrain_lemmas=tuple(subtrain),
                dev_lemmas=tuple(dev),
                train_lemmas=tuple(train),
                test_lemmas=tuple(test),
            )
        )

    return folds


def _cluster_bootstrap_se(
    cluster_sums: Sequence[float],
    cluster_counts: Sequence[int],
    seed_str: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> float:
    """500-resample cluster bootstrap SE over `leakage_group_id` clusters."""
    n_clusters = len(cluster_sums)
    if n_clusters <= 1:
        return 0.0

    sums_arr = np.asarray(cluster_sums, dtype=np.float64)
    cnts_arr = np.asarray(cluster_counts, dtype=np.float64)

    seed_int = int.from_bytes(
        hashlib.sha256(seed_str.encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed_int)
    idx_mat = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    boot_num = sums_arr[idx_mat].sum(axis=1)
    boot_den = cnts_arr[idx_mat].sum(axis=1)
    boot_means = boot_num / np.maximum(boot_den, 1.0)
    return float(np.std(boot_means, ddof=1))


def evaluate_unit(
    lexicon: UnitLexicon,
    folds: Sequence[SplitFold],
    regime: str,
    model_name: str = "mgl",
    gamma: float = DEFAULT_GAMMA_ESCAPE,
    enable_vowel_tier: bool = True,
    enable_impugnment: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run inner-CV topology selection + full-train MGL refit + held-out test evaluation on 1 unit."""
    t_start = time.perf_counter()
    cells = lexicon.cells
    n_cells = len(cells)
    use_phon = lexicon.representation == "phon" or lexicon.sound_inventory is not None

    # Accumulate per-lemma evaluations across draws: `lemma_id -> list[LemmaTopologyScores]`
    lemma_evals: dict[str, list[LemmaTopologyScores]] = defaultdict(list)
    # Also track draw-level means and selected root cells
    draw_lemma_scores: dict[int, list[tuple[str, LemmaTopologyScores]]] = defaultdict(list)
    root_Selections: list[tuple[str, float]] = []

    # Adaptive parent pre-screening schedule keeping every unit fast while matching
    # the neural transducer's `PRESCREEN_CELL_THRESHOLD = 40` / 14-cell `k_pool` on primary `mgl:uniform120`:
    is_primary_u120 = (model_name == "mgl" and regime == "uniform120")
    is_480 = "480" in regime
    eff_threshold = PRESCREEN_CELL_THRESHOLD if is_primary_u120 else (10 if is_480 else 14)
    if n_cells <= eff_threshold:
        top_k_parents = n_cells
        top_roots = n_cells
        k_pool_sz = n_cells
    elif n_cells <= 65:
        top_k_parents = 6 if is_primary_u120 else (2 if is_480 else 5)
        top_roots = 2 if not is_480 else 1
        k_pool_sz = 14 if is_primary_u120 else (6 if is_480 else 10)
    elif n_cells <= 105:
        top_k_parents = 4 if is_primary_u120 else (2 if is_480 else 3)
        top_roots = 2 if is_primary_u120 else 1
        k_pool_sz = 14 if is_primary_u120 else (5 if is_480 else 8)
    else:
        top_k_parents = 3 if not is_480 else 1
        top_roots = 1
        k_pool_sz = 8 if not is_480 else 5

    # Share deterministic `(x, y)` alignment & `(e1, e2)` generalization caches across classifiers
    # ONLY when their resolved `(tok, 'V'|'C')` maps are 100% identical (zero cross-split leakage!).
    shared_caches: dict[
        tuple[tuple[str, str], ...],
        tuple[dict[tuple[tuple[str, ...], tuple[str, ...]], object], dict[tuple[object, object], object]],
    ] = {}

    def _bind_shared_caches(clf: VowelClassifier) -> None:
        vc_sig = tuple(sorted(clf._learned_vc.items()))
        pair = shared_caches.get(vc_sig)
        if pair is None:
            pair = ({}, {})
            shared_caches[vc_sig] = pair
        clf._pattern_cache, clf._gen_env_cache = pair

    for fold_ord, sp in enumerate(folds):
        # Strict <= 5 min/unit cap on `uniform480` at complete 5-fold draw boundaries
        if (
            is_480
            and sp.fold == 0
            and len(draw_lemma_scores) >= 1
            and (time.perf_counter() - t_start) > 240.0
        ):
            break
        t_fold0 = time.perf_counter()

        # 1. Fit `subtrain` VowelClassifier STRICTLY on `subtrain` forms (zero dev/test leakage!)
        subtrain_forms = [
            lexicon.paradigms[lid][c]
            for lid in sp.subtrain_lemmas
            for c in cells
        ]
        subtrain_clf = VowelClassifier(
            sound_inventory=lexicon.sound_inventory,
            use_phon_features=use_phon,
        ).fit(subtrain_forms)
        _bind_shared_caches(subtrain_clf)

        subtrain_paras = [lexicon.paradigms[lid] for lid in sp.subtrain_lemmas]
        seed_fold = f"{lexicon.unit_id}|{regime}|{sp.draw}|{sp.fold}"

        if n_cells > eff_threshold:
            cand_edges, k_pool = prescreen_candidate_edges(
                cells,
                subtrain_paras,
                top_k_parents=top_k_parents,
                seed_key=seed_fold,
                cell_threshold=eff_threshold,
                top_roots=top_roots,
                k_pool_size=k_pool_sz,
            )
        else:
            cand_edges = {(u, v) for u in cells for v in cells if u != v}
            k_pool = list(cells)

        # 2. Estimate `M_hat[s, t]` on `dev` using `subtrain`-fitted models ONLY
        # (On large-cell `uniform480`, inner topology selection uses the deterministic 128/32
        #  `subtrain`/`dev` prefix while Step 4 refits MGL on ALL 480 `train` lemmas for test!)
        eff_sub_lemmas = (
            sp.subtrain_lemmas[:128] if (is_480 and n_cells > 20) else sp.subtrain_lemmas
        )
        eff_dev_lemmas = (
            sp.dev_lemmas[:32] if (is_480 and n_cells > 20) else sp.dev_lemmas
        )
        dev_matrix: dict[tuple[str, str], float] = {}
        for cs, ct in sorted(cand_edges):
            sub_pairs = [
                (lexicon.paradigms[lid][cs], lexicon.paradigms[lid][ct])
                for lid in eff_sub_lemmas
            ]
            sub_scorer = DirichletEscapeScorer.fit_from_pairs(
                sub_pairs,
                classifier=subtrain_clf,
                gamma=gamma,
                enable_vowel_tier=enable_vowel_tier,
                enable_impugnment=enable_impugnment,
            )
            dev_bits_sum = 0.0
            for dlid in eff_dev_lemmas:
                src_dev = lexicon.paradigms[dlid][cs]
                tgt_dev = lexicon.paradigms[dlid][ct]
                if src_dev == tgt_dev:
                    continue
                ds_dev = sub_scorer.score_pair(src_dev, tgt_dev)
                dev_bits_sum += ds_dev.nll_bits
            dev_matrix[(cs, ct)] = dev_bits_sum / len(eff_dev_lemmas)

        # 3. Select optimal base `r*` and Chu-Liu-Edmonds arborescence `T_{r*}` on `dev_matrix`
        opt_tree = select_optimal_arborescence(cells, dev_matrix)
        root_Selections.append((opt_tree.root, opt_tree.total_cost))

        # 4. Refit `VowelClassifier` & `(c_s -> c_t)` MGL models on ALL `train` (`subtrain U dev`)
        #    lemmas (still strictly zero `test` leakage!), lazily memoizing needed cell pairs!
        train_forms = [
            lexicon.paradigms[lid][c]
            for lid in sp.train_lemmas
            for c in cells
        ]
        train_clf = VowelClassifier(
            sound_inventory=lexicon.sound_inventory,
            use_phon_features=use_phon,
        ).fit(train_forms)
        _bind_shared_caches(train_clf)

        fitted_train_scorers: dict[tuple[str, str], DirichletEscapeScorer] = {}

        def _get_train_scorer(cs: str, ct: str) -> DirichletEscapeScorer:
            pair_key = (cs, ct)
            cached = fitted_train_scorers.get(pair_key)
            if cached is not None:
                return cached
            tr_pairs = [
                (lexicon.paradigms[lid][cs], lexicon.paradigms[lid][ct])
                for lid in sp.train_lemmas
            ]
            sc = DirichletEscapeScorer.fit_from_pairs(
                tr_pairs,
                classifier=train_clf,
                gamma=gamma,
                enable_vowel_tier=enable_vowel_tier,
                enable_impugnment=enable_impugnment,
            )
            fitted_train_scorers[pair_key] = sc
            return sc

        # 5. Score held-out `test` lemmas
        for tlid in sp.test_lemmas:
            lemma_seed = f"{seed_fold}|{tlid}"
            t_scores = evaluate_lemma_topology(
                lemma_forms=lexicon.paradigms[tlid],
                cells=cells,
                tree=opt_tree,
                dev_matrix=dev_matrix,
                get_scorer=_get_train_scorer,
                seed_prefix=lemma_seed,
                k_eval_pool_cells=k_pool if n_cells > eff_threshold else None,
            )
            lemma_evals[tlid].append(t_scores)
            draw_lemma_scores[sp.draw].append((tlid, t_scores))

        # Dynamic 10-minute runtime guard for huge paradigms
        fold_elapsed = time.perf_counter() - t_fold0
        if (
            fold_ord == 0
            and n_cells > eff_threshold
            and fold_elapsed * len(folds) > DYNAMIC_TIGHTEN_SEC_THRESHOLD
        ):
            top_k_parents = max(3, top_k_parents - 2)
            top_roots = 1
            k_pool_sz = max(6, k_pool_sz - 2)

    # 6. Build `lemma_outcomes_{regime}.csv` rows (1 row per unique tested `lemma_id`)
    lemma_rows: list[dict[str, object]] = []
    for lid in sorted(lemma_evals.keys()):
        eval_list = lemma_evals[lid]
        inv_m = 1.0 / len(eval_list)
        n_dist = eval_list[0].n_distinct_forms
        lemma_rows.append(
            {
                "unit_id": lexicon.unit_id,
                "glottocode": lexicon.glottocode,
                "pos": lexicon.pos,
                "representation": lexicon.representation,
                "model": model_name,
                "regime": regime,
                "lemma_id": lid,
                "leakage_group_id": lexicon.lemma_to_group[lid],
                "n_distinct_forms": n_dist,
                "h_tree_lemma": sum(e.h_tree for e in eval_list) * inv_m,
                "rel_tree_lemma": sum(e.rel_tree for e in eval_list) * inv_m,
                "acc_tree_lemma": sum(e.acc_tree for e in eval_list) * inv_m,
                "h_k1_lemma": sum(e.h_k1 for e in eval_list) * inv_m,
                "h_k3_lemma": sum(e.h_k3 for e in eval_list) * inv_m,
            }
        )

    # 7. Aggregate unit-level point estimates and 500-resample `leakage_group_id` cluster bootstrap SEs
    draws_sorted = sorted(draw_lemma_scores.keys())
    n_draws = len(draws_sorted)
    all_eval_scores = [sc for d in draws_sorted for _, sc in draw_lemma_scores[d]]

    def _eval_mean(getter) -> float:
        return sum(getter(sc) for sc in all_eval_scores) / len(all_eval_scores)

    h_tree_bits = _eval_mean(lambda s: s.h_tree)
    rel_tree_mean = _eval_mean(lambda s: s.rel_tree)
    acc_tree_mean = _eval_mean(lambda s: s.acc_tree)
    h_k1_bits = _eval_mean(lambda s: s.h_k1)
    acc_k1_mean = _eval_mean(lambda s: s.acc_k1)
    h_k3_bits = _eval_mean(lambda s: s.h_k3)
    acc_k3_mean = _eval_mean(lambda s: s.acc_k3)

    # Group all `(draw, test_lemma)` evaluations by `leakage_group_id` for cluster bootstrap
    clusters: dict[str, list[LemmaTopologyScores]] = defaultdict(list)
    for d in draws_sorted:
        for lid, sc in draw_lemma_scores[d]:
            clusters[lexicon.lemma_to_group[lid]].append(sc)

    sorted_gids = sorted(clusters.keys())
    c_counts = [len(clusters[g]) for g in sorted_gids]
    boot_base = f"{lexicon.unit_id}|{regime}|bootstrap"

    h_tree_se = _cluster_bootstrap_se(
        [sum(s.h_tree for s in clusters[g]) for g in sorted_gids],
        c_counts,
        f"{boot_base}|h_tree",
    )
    rel_tree_se = _cluster_bootstrap_se(
        [sum(s.rel_tree for s in clusters[g]) for g in sorted_gids],
        c_counts,
        f"{boot_base}|rel_tree",
    )
    acc_tree_se = _cluster_bootstrap_se(
        [sum(s.acc_tree for s in clusters[g]) for g in sorted_gids],
        c_counts,
        f"{boot_base}|acc_tree",
    )
    h_k1_se = _cluster_bootstrap_se(
        [sum(s.h_k1 for s in clusters[g]) for g in sorted_gids],
        c_counts,
        f"{boot_base}|h_k1",
    )
    h_k3_se = _cluster_bootstrap_se(
        [sum(s.h_k3 for s in clusters[g]) for g in sorted_gids],
        c_counts,
        f"{boot_base}|h_k3",
    )

    # Modal base cell `r*` across all `R * K` folds (tie-break by lowest mean dev cost, then name)
    root_counts: Counter[str] = Counter(r for r, _ in root_Selections)
    root_cost_sums: dict[str, float] = defaultdict(float)
    for r, cost_val in root_Selections:
        root_cost_sums[r] += cost_val
    base_cell_modal = min(
        root_counts.keys(),
        key=lambda r: (-root_counts[r], round(root_cost_sums[r] / root_counts[r], 12), r),
    )

    runtime_sec = time.perf_counter() - t_start
    unit_row: dict[str, object] = {
        "unit_id": lexicon.unit_id,
        "glottocode": lexicon.glottocode,
        "pos": lexicon.pos,
        "representation": lexicon.representation,
        "model": model_name,
        "regime": regime,
        "n_draws": n_draws,
        "n_test_lemmas": len(lemma_rows),
        "n_cells_distinct": n_cells,
        "h_tree_bits": round(h_tree_bits, 6),
        "h_tree_se": round(h_tree_se, 6),
        "rel_tree_mean": round(rel_tree_mean, 6),
        "rel_tree_se": round(rel_tree_se, 6),
        "acc_tree_mean": round(acc_tree_mean, 6),
        "acc_tree_se": round(acc_tree_se, 6),
        "h_k1_bits": round(h_k1_bits, 6),
        "h_k1_se": round(h_k1_se, 6),
        "acc_k1_mean": round(acc_k1_mean, 6),
        "h_k3_bits": round(h_k3_bits, 6),
        "h_k3_se": round(h_k3_se, 6),
        "acc_k3_mean": round(acc_k3_mean, 6),
        "base_cell_modal": base_cell_modal,
        "runtime_sec": round(runtime_sec, 4),
    }

    for lr in lemma_rows:
        lr["h_tree_lemma"] = round(float(lr["h_tree_lemma"]), 6)
        lr["rel_tree_lemma"] = round(float(lr["rel_tree_lemma"]), 6)
        lr["acc_tree_lemma"] = round(float(lr["acc_tree_lemma"]), 6)
        lr["h_k1_lemma"] = round(float(lr["h_k1_lemma"]), 6)
        lr["h_k3_lemma"] = round(float(lr["h_k3_lemma"]), 6)

    return unit_row, lemma_rows


def _evaluate_single_unit_task(
    args_tuple: tuple[str, str, str, str, float, bool, bool, str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    (
        lex_path_str,
        split_path_str,
        regime,
        model_name,
        gamma,
        enable_vtier,
        enable_impugn,
        cache_dir_str,
    ) = args_tuple

    # Pre-read needed `lemma_id`s from split manifest so multi-million-row lexicons load in <0.5s
    needed_lemmas: set[str] = set()
    with Path(split_path_str).open("r", encoding="utf-8", newline="") as sp_fh:
        for row in csv.DictReader(sp_fh):
            needed_lemmas.add(str(row["lemma_id"]).strip())

    lex = load_unit_lexicon(lex_path_str, lemma_filter=needed_lemmas)
    folds = load_split_manifest(split_path_str, lex)
    unit_row, lemma_rows = evaluate_unit(
        lex,
        folds,
        regime=regime,
        model_name=model_name,
        gamma=gamma,
        enable_vowel_tier=enable_vtier,
        enable_impugnment=enable_impugn,
    )

    if cache_dir_str:
        cdir = Path(cache_dir_str)
        cdir.mkdir(parents=True, exist_ok=True)
        u_file = cdir / f"{lex.unit_id}_unit.csv"
        l_file = cdir / f"{lex.unit_id}_lemmas.csv"
        _write_csv_rows(u_file, UNIT_OUTCOME_COLUMNS, [unit_row])
        _write_csv_rows(l_file, LEMMA_OUTCOME_COLUMNS, lemma_rows)

    return unit_row, lemma_rows


def _write_csv_rows(
    out_path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r[col] for col in columns})


def run_regime_evaluation(
    lexicons_dir: str | Path,
    splits_dir: str | Path,
    results_dir: str | Path,
    regime: str = "uniform120",
    model_name: str = "mgl",
    unit_ids: Sequence[str] | None = None,
    max_workers: int = 1,
    gamma: float = DEFAULT_GAMMA_ESCAPE,
    enable_vowel_tier: bool = True,
    enable_impugnment: bool = True,
    resume: bool = True,
) -> tuple[Path, Path]:
    """Evaluate all matching `{unit_id}_{regime}.csv` manifests and write regime outcome CSVs."""
    eff_workers = max(1, min(int(max_workers), MAX_HARD_WORKERS_CAP))
    lex_dir = Path(lexicons_dir)
    sp_dir = Path(splits_dir)
    res_dir = Path(results_dir)
    cache_dir = res_dir / "per_unit_cache" / f"{model_name}_{regime}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{regime}.csv"
    split_files = sorted(sp_dir.glob(f"*{suffix}"))
    if unit_ids is not None:
        wanted = set(unit_ids)
        split_files = [
            sf for sf in split_files if sf.name[: -len(suffix)] in wanted
        ]

    unit_rows: list[dict[str, object]] = []
    all_lemma_rows: list[dict[str, object]] = []
    tasks: list[tuple[str, str, str, str, float, bool, bool, str]] = []

    for sf in split_files:
        uid = sf.name[: -len(suffix)]
        lex_file = lex_dir / f"{uid}.csv"
        if not lex_file.exists():
            continue
        cached_u = cache_dir / f"{uid}_unit.csv"
        cached_l = cache_dir / f"{uid}_lemmas.csv"
        if resume and cached_u.exists() and cached_l.exists():
            with cached_u.open("r", encoding="utf-8", newline="") as hu:
                unit_rows.extend(csv.DictReader(hu))
            with cached_l.open("r", encoding="utf-8", newline="") as hl:
                all_lemma_rows.extend(csv.DictReader(hl))
            continue

        tasks.append(
            (
                str(lex_file),
                str(sf),
                regime,
                model_name,
                gamma,
                enable_vowel_tier,
                enable_impugnment,
                str(cache_dir),
            )
        )

    if tasks:
        # Longest-Processing-Time-First (LPT) scheduling by lexicon byte size eliminates tail idle time
        tasks.sort(key=lambda t: (-Path(t[0]).stat().st_size, t[0]))
        total_n = len(tasks)
        if eff_workers == 1:
            for done_i, t_arg in enumerate(tasks, 1):
                u_row, l_rows = _evaluate_single_unit_task(t_arg)
                unit_rows.append(u_row)
                all_lemma_rows.extend(l_rows)
                print(
                    f"[{model_name}:{regime} {done_i}/{total_n}] {u_row['unit_id']} "
                    f"(cells={u_row['n_cells_distinct']}, h_tree={u_row['h_tree_bits']}, "
                    f"sec={u_row['runtime_sec']})",
                    flush=True,
                )
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=eff_workers) as pool:
                fut_list = [pool.submit(_evaluate_single_unit_task, t_arg) for t_arg in tasks]
                for done_i, fut in enumerate(concurrent.futures.as_completed(fut_list), 1):
                    u_row, l_rows = fut.result()
                    unit_rows.append(u_row)
                    all_lemma_rows.extend(l_rows)
                    print(
                        f"[{model_name}:{regime} {done_i}/{total_n}] {u_row['unit_id']} "
                        f"(cells={u_row['n_cells_distinct']}, h_tree={u_row['h_tree_bits']}, "
                        f"sec={u_row['runtime_sec']})",
                        flush=True,
                    )

    unit_rows.sort(key=lambda r: str(r["unit_id"]))
    all_lemma_rows.sort(key=lambda r: (str(r["unit_id"]), str(r["lemma_id"])))

    out_suffix = regime if model_name == "mgl" else f"{regime}_{model_name}"
    unit_out_path = res_dir / f"unit_outcomes_{out_suffix}.csv"
    lemma_out_path = res_dir / f"lemma_outcomes_{out_suffix}.csv"
    _write_csv_rows(unit_out_path, UNIT_OUTCOME_COLUMNS, unit_rows)
    _write_csv_rows(lemma_out_path, LEMMA_OUTCOME_COLUMNS, all_lemma_rows)
    return unit_out_path, lemma_out_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-Locus Vowel-Tier MGL paradigm predictability."
    )
    parser.add_argument(
        "--lexicons-dir",
        type=Path,
        default=Path("analyses/paradigm/data/lexicons"),
        help="Directory containing `{unit_id}.csv` (and optional `{unit_id}_sounds.csv`).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("analyses/paradigm/data/splits"),
        help="Directory containing `{unit_id}_{regime}.csv` split manifests.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("analyses/paradigm/results/mgl"),
        help="Output directory for `unit_outcomes_{regime}.csv` and `lemma_outcomes_{regime}.csv`.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="uniform120",
        help="Evaluation regime name (e.g., uniform120, uniform480, topfreq120).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mgl",
        help="Model identifier written to `model` column (default: mgl).",
    )
    parser.add_argument(
        "--units",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of `unit_id`s to evaluate.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Process pool worker count (capped at coordinator grant <= 8; default 1).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA_ESCAPE,
        help="Dirichlet escape concentration parameter gamma (fixed globally at 0.05).",
    )
    parser.add_argument(
        "--no-vowel-tier",
        action="store_true",
        help="Ablate source Vowel-Tier rule generalization.",
    )
    parser.add_argument(
        "--no-impugnment",
        action="store_true",
        help="Ablate alpha_upper=0.95 residue impugnment.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute units even if per-unit cache files exist.",
    )
    args = parser.parse_args(argv)

    u_path, l_path = run_regime_evaluation(
        lexicons_dir=args.lexicons_dir,
        splits_dir=args.splits_dir,
        results_dir=args.results_dir,
        regime=args.regime,
        model_name=args.model_name,
        unit_ids=args.units,
        max_workers=args.max_workers,
        gamma=args.gamma,
        enable_vowel_tier=not args.no_vowel_tier,
        enable_impugnment=not args.no_impugnment,
        resume=not args.no_resume,
    )
    print(f"Wrote {u_path} and {l_path}")


if __name__ == "__main__":
    main()
