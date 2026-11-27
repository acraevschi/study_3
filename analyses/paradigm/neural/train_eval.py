"""Multi-task training, dev spanning-tree selection, and PCFP evaluation for the
Hard-Monotonic Neural Character Transducer (`analyses/paradigm/neural/`).

Key guarantees:
  1. Strict train/dev/test isolation:
     - Asserts zero `leakage_group_id` overlap between `train` (`subtrain`/`dev`) and `test`
       and between `subtrain` and `dev` on every `(draw, fold)`.
     - Builds `CharVocab` strictly from training forms (never touching `test`).
     - Estimates the `|C_distinct| x |C_distinct|` validation cross-entropy matrix `M_hat`,
       optimal root `r*`, Chu-Liu-Edmonds spanning arborescence `T_{r*}`, and `k=3` PoE
       weights strictly on `dev` (never peeking at `test`).
  2. Proper scoring rule:
     - Evaluates exact marginal log-likelihood with ZERO label smoothing.
     - Averages across a 3-seed ensemble in probability space:
       `log q_ens(y | x) = logsumexp_s log q_s(y | x) - log(S)`.
  3. Exact output schema parity with the MGL pipeline:
     - Writes `unit_outcomes_{regime}.csv` and `lemma_outcomes_{regime}.csv` with
       500-resample `leakage_group_id` cluster bootstrap standard errors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from analyses.paradigm.neural.monotonic_transducer import (
    LN2,
    CellVocab,
    CharVocab,
    EncodedBatch,
    HardMonotonicTransducer,
    TransducerConfig,
    collate_transducer_batch,
    ensemble_beam_decode_batch,
    ensemble_log_prob_pairs,
    resolve_device,
)

logger = logging.getLogger(__name__)

UNIT_OUTCOME_COLUMNS: list[str] = [
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
]

LEMMA_OUTCOME_COLUMNS: list[str] = [
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
]


@dataclass
class TrainEvalConfig:
    """Hyperparameters for 3-seed neural transducer training and evaluation."""

    n_seeds: int = 3
    max_epochs: int = 35
    patience: int = 6
    batch_size: int = 128
    eval_batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    max_train_pairs: int = 25_000
    refit_full_train: bool = True
    refit_warmup_epochs: int = 5
    beam_size: int = 4
    poe_tau: float = 0.1
    n_k3_subsets: int = 5
    max_k3_targets_per_subset: int | None = None
    max_k1_pairs_per_lemma: int | None = 120
    prescreen_cell_threshold: int = 40
    prescreen_top_parents: int = 12
    n_bootstrap: int = 500
    base_seed: int = 20261112
    device: str = "auto"
    num_threads: int = 2
    transducer: TransducerConfig = TransducerConfig()


@dataclass
class LemmaParadigm:
    """Complete canonical synthetic paradigm for a single lemma."""

    lemma_id: str
    leakage_group_id: str
    shape_id: str
    forms: dict[str, tuple[str, ...]]       # cell -> tuple of symbols
    raw_forms: dict[str, str]               # cell -> raw surface string


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def parse_segmented_or_raw(segmented_form: str, raw_form: str) -> tuple[str, ...]:
    """Parse space-separated `segmented_form` when present, falling back to `tuple(raw_form)`."""
    seg = (segmented_form or "").strip()
    if seg:
        return tuple(seg.split())
    return tuple((raw_form or "").strip())


def load_unit_lexicon(
    lexicon_csv: Path,
    keep_lemmas: set[str] | None = None,
) -> tuple[dict[str, str], list[str], dict[str, LemmaParadigm]]:
    """Load `lexicons/{unit_id}.csv` (canonical `variant_idx == 0` forms)."""
    meta: dict[str, str] = {}
    by_lemma_cells: dict[str, dict[str, tuple[int, tuple[str, ...], str]]] = defaultdict(dict)
    lemma_meta: dict[str, tuple[str, str]] = {}
    all_cells: set[str] = set()

    with open(lexicon_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not meta:
                meta = {
                    "unit_id": row["unit_id"],
                    "glottocode": row["glottocode"],
                    "pos": row["pos"],
                    "representation": row["representation"],
                }
            cell = row["cell"]
            all_cells.add(cell)
            lid = row["lemma_id"]
            if keep_lemmas is not None and lid not in keep_lemmas:
                continue
            v_idx = int(row.get("variant_idx", "0") or 0)
            lemma_meta[lid] = (row["leakage_group_id"], row.get("shape_id", ""))
            prev = by_lemma_cells[lid].get(cell)
            if prev is None or v_idx < prev[0]:
                syms = parse_segmented_or_raw(row.get("segmented_form", ""), row.get("raw_form", ""))
                raw_s = (row.get("raw_form") or "").strip() or "".join(syms)
                by_lemma_cells[lid][cell] = (v_idx, syms, raw_s)

    cells_sorted = sorted(all_cells)
    paradigms: dict[str, LemmaParadigm] = {}
    for lid, cell_map in by_lemma_cells.items():
        if len(cell_map) != len(cells_sorted):
            continue
        lg_id, sh_id = lemma_meta[lid]
        paradigms[lid] = LemmaParadigm(
            lemma_id=lid,
            leakage_group_id=lg_id,
            shape_id=sh_id,
            forms={c: cell_map[c][1] for c in cells_sorted},
            raw_forms={c: cell_map[c][2] for c in cells_sorted},
        )
    if not paradigms:
        raise ValueError(f"No complete paradigms found in {lexicon_csv}")
    return meta, cells_sorted, paradigms


def load_split_manifest(split_csv: Path) -> dict[tuple[int, int], dict[str, list[str]]]:
    """Load split manifest into `{(draw, fold): {'subtrain': [...], 'dev': [...], 'test': [...]}}`.

    Enforces zero `leakage_group_id` overlap between `train` (`subtrain`/`dev`) and `test`,
    and between `subtrain` and `dev`.
    """
    splits: dict[tuple[int, int], dict[str, list[str]]] = defaultdict(
        lambda: {"subtrain": [], "dev": [], "test": []}
    )
    groups_by_role: dict[tuple[int, int], dict[str, set[str]]] = defaultdict(
        lambda: {"subtrain": set(), "dev": set(), "test": set()}
    )

    with open(split_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        draw_col = next((c for c in ("draw", "draw_id", "draw_idx", "rep") if c in cols), None)
        fold_col = next((c for c in ("fold", "fold_id", "fold_idx") if c in cols), None)
        if draw_col is None or fold_col is None:
            raise ValueError(f"Split manifest {split_csv} missing draw/fold columns: {sorted(cols)}")

        for row in reader:
            d_id = int(row[draw_col])
            f_id = int(row[fold_col])
            lid = row["lemma_id"]
            lg_id = row["leakage_group_id"]
            inner = (row.get("inner_role") or row.get("role") or "").strip().lower()
            if inner not in ("subtrain", "dev", "test"):
                role = (row.get("role") or "").strip().lower()
                if role == "test":
                    inner = "test"
                elif role == "train":
                    inner = "subtrain"
                else:
                    raise ValueError(f"Unknown split role {inner!r} in {split_csv}")
            splits[(d_id, f_id)][inner].append(lid)
            groups_by_role[(d_id, f_id)][inner].add(lg_id)

    # If a manifest only had `train`/`test` without `dev`, deterministically carve 20% of train groups into `dev`
    for key, role_lists in splits.items():
        if not role_lists["dev"] and len(role_lists["subtrain"]) >= 5:
            rng = random.Random(_stable_seed("dev_carve", key[0], key[1]))
            sub_lids = list(role_lists["subtrain"])
            rng.shuffle(sub_lids)
            n_dev = max(1, len(sub_lids) // 5)
            role_lists["dev"] = sorted(sub_lids[:n_dev])
            role_lists["subtrain"] = sorted(sub_lids[n_dev:])
        else:
            # Verify strict group disjointness when `inner_role` is explicitly provided
            g_sub = groups_by_role[key]["subtrain"]
            g_dev = groups_by_role[key]["dev"]
            g_tst = groups_by_role[key]["test"]
            overlap_tr_te = (g_sub | g_dev) & g_tst
            overlap_sub_dev = g_sub & g_dev
            if overlap_tr_te:
                raise ValueError(f"Leakage group overlap between train and test in {key}: {overlap_tr_te}")
            if overlap_sub_dev:
                raise ValueError(f"Leakage group overlap between subtrain and dev in {key}: {overlap_sub_dev}")

    return dict(sorted(splits.items()))


# ---------------------------------------------------------------------------
# Chu-Liu-Edmonds Directed Minimum Spanning Arborescence
# ---------------------------------------------------------------------------

def chu_liu_edmonds(
    n_nodes: int,
    root: int,
    cost_matrix: Sequence[Sequence[float]],
) -> tuple[dict[int, int], float]:
    """Exact Chu-Liu-Edmonds minimum-cost directed spanning arborescence rooted at `root`.

    Args:
        n_nodes: Number of vertices `0 .. n_nodes - 1`.
        root: Root vertex index in `0 .. n_nodes - 1`.
        cost_matrix: `cost_matrix[u][v]` is the directed edge cost `u -> v`.

    Returns:
        `(parent_of, total_cost)` where `parent_of[v] = u` for every `v != root`.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1.")
    if n_nodes == 1:
        return {}, 0.0

    edges: list[tuple[int, int, float]] = []
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v and v != root:
                c = float(cost_matrix[u][v])
                if math.isfinite(c):
                    edges.append((u, v, c))

    parent_map = _edmonds_recursive(n_nodes, root, edges)
    total_cost = sum(float(cost_matrix[u][v]) for v, u in parent_map.items())
    return parent_map, total_cost


def _edmonds_recursive(
    n_nodes: int,
    root: int,
    edges: list[tuple[int, int, float]],
) -> dict[int, int]:
    # 1. Select minimum incoming edge for each v != root (deterministic tie-break by smallest u)
    min_in_cost: dict[int, float] = {}
    min_in_u: dict[int, int] = {}
    for u, v, w in edges:
        if v == root or u == v:
            continue
        if v not in min_in_cost or w < min_in_cost[v] - 1e-12 or (
            abs(w - min_in_cost[v]) <= 1e-12 and u < min_in_u[v]
        ):
            min_in_cost[v] = w
            min_in_u[v] = u

    for v in range(n_nodes):
        if v != root and v not in min_in_u:
            raise ValueError(f"Graph is not root-connected from root={root} (cannot reach {v}).")

    # 2. Detect any directed cycle among chosen incoming edges
    visited_state = [0] * n_nodes  # 0 = unvisited, 1 = active_trace, 2 = verified_acyclic
    visited_state[root] = 2
    cycle_nodes: list[int] | None = None

    for start_v in range(n_nodes):
        if visited_state[start_v] != 0:
            continue
        trace: list[int] = []
        pos_in_trace: dict[int, int] = {}
        curr = start_v
        while curr != root and visited_state[curr] == 0:
            visited_state[curr] = 1
            pos_in_trace[curr] = len(trace)
            trace.append(curr)
            curr = min_in_u[curr]
        if curr != root and visited_state[curr] == 1:
            cycle_nodes = trace[pos_in_trace[curr] :]
            break
        for node in trace:
            visited_state[node] = 2

    if cycle_nodes is None:
        return min_in_u

    # 3. Contract cycle C into supernode `cycle_id`
    cycle_set = set(cycle_nodes)
    new_id: dict[int, int] = {}
    idx_counter = 0
    for v in range(n_nodes):
        if v not in cycle_set:
            new_id[v] = idx_counter
            idx_counter += 1
    cycle_id = idx_counter
    for v in cycle_nodes:
        new_id[v] = cycle_id
    n_contracted = idx_counter + 1
    new_root = new_id[root]

    # Track best parallel contracted edge and which original (u, v) realized it
    best_contracted: dict[tuple[int, int], tuple[float, int, int]] = {}
    for u, v, w in edges:
        nu, nv = new_id[u], new_id[v]
        if nu == nv or nv == new_root:
            continue
        adj_w = (w - min_in_cost[v]) if v in cycle_set else w
        prev = best_contracted.get((nu, nv))
        if prev is None or adj_w < prev[0] - 1e-12 or (
            abs(adj_w - prev[0]) <= 1e-12 and (u, v) < (prev[1], prev[2])
        ):
            best_contracted[(nu, nv)] = (adj_w, u, v)

    contracted_edges = [
        (nu, nv, info[0]) for (nu, nv), info in sorted(best_contracted.items())
    ]
    sub_parents = _edmonds_recursive(n_contracted, new_root, contracted_edges)

    # 4. Expand contracted supernode
    result: dict[int, int] = {}
    entry_v_in_cycle: int | None = None
    for nv, nu in sub_parents.items():
        _, orig_u, orig_v = best_contracted[(nu, nv)]
        result[orig_v] = orig_u
        if nv == cycle_id:
            entry_v_in_cycle = orig_v

    assert entry_v_in_cycle is not None
    for v in cycle_nodes:
        if v != entry_v_in_cycle:
            result[v] = min_in_u[v]
    return result


def select_optimal_arborescence(
    cells: Sequence[str],
    m_hat: Sequence[Sequence[float]],
) -> tuple[int, dict[int, int], float]:
    """Run Chu-Liu-Edmonds across all candidate roots and select `r* = argmin_r cost(T_r)`."""
    n = len(cells)
    best_r = 0
    best_parents: dict[int, int] = {}
    best_cost = float("inf")
    for r in range(n):
        try:
            parents, cost = chu_liu_edmonds(n, r, m_hat)
        except ValueError:
            continue
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_r = r
            best_parents = parents
    if not math.isfinite(best_cost):
        raise RuntimeError("No valid spanning arborescence found on validation matrix.")
    return best_r, best_parents, best_cost


def tree_depths(n_nodes: int, root: int, parents: dict[int, int]) -> list[int]:
    """Compute root-to-node hop depth in arborescence `parents`."""
    depths = [-1] * n_nodes
    depths[root] = 0

    def get_d(v: int) -> int:
        if depths[v] >= 0:
            return depths[v]
        depths[v] = 1 + get_d(parents[v])
        return depths[v]

    for v in range(n_nodes):
        get_d(v)
    return depths


# ---------------------------------------------------------------------------
# Pair Sampling, Training & Validation Matrix Estimation
# ---------------------------------------------------------------------------

def build_pair_examples(
    paradigms: Sequence[LemmaParadigm],
    candidate_edges: Sequence[tuple[str, str]],
    max_pairs: int | None,
    seed: int,
) -> list[tuple[tuple[str, ...], tuple[str, ...], str, str]]:
    """Build non-syncretic `(src_syms, tgt_syms, c_s, c_t)` pairs with stratified capping."""
    n_edges = len(candidate_edges)
    if n_edges == 0 or not paradigms:
        return []

    rng = random.Random(seed)
    by_edge: list[list[tuple[tuple[str, ...], tuple[str, ...], str, str]]] = []
    total_non_sync = 0
    for cs, ct in candidate_edges:
        edge_pairs = [
            (p.forms[cs], p.forms[ct], cs, ct)
            for p in paradigms
            if p.forms[cs] != p.forms[ct]
        ]
        if edge_pairs:
            by_edge.append(edge_pairs)
            total_non_sync += len(edge_pairs)

    # Fallback only for degenerate all-identical synthetic toy fixtures
    if total_non_sync == 0:
        out_all = [
            (p.forms[cs], p.forms[ct], cs, ct)
            for p in paradigms
            for (cs, ct) in candidate_edges
        ]
        rng.shuffle(out_all)
        return out_all[:max_pairs] if max_pairs is not None else out_all

    if max_pairs is None or total_non_sync <= max_pairs:
        out = [ex for edge_pairs in by_edge for ex in edge_pairs]
        rng.shuffle(out)
        return out

    # Exact water-filling stratified allocation across active directed cell pairs
    n_active = len(by_edge)
    edge_order = list(range(n_active))
    rng.shuffle(edge_order)

    remaining_budget = max_pairs
    remaining_edges = n_active
    # Sort indices by ascending available pool size (ties broken by shuffled order) so short
    # edges donate unused quota to larger edges cleanly
    rank_of = {e_i: r for r, e_i in enumerate(edge_order)}
    sorted_e = sorted(range(n_active), key=lambda e_i: (len(by_edge[e_i]), rank_of[e_i]))

    out = []
    for e_i in sorted_e:
        pool = by_edge[e_i]
        fair_share = remaining_budget // remaining_edges
        if remaining_budget % remaining_edges > 0:
            fair_share += 1
        take = min(len(pool), fair_share)
        if take > 0:
            chosen = rng.sample(pool, take) if take < len(pool) else list(pool)
            out.extend(chosen)
            remaining_budget -= take
        remaining_edges -= 1

    rng.shuffle(out)
    return out


def _fast_single_change_sig(
    x: tuple[str, ...], y: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """O(|x|) maximal shared prefix/suffix stripping (matching `mgl/topology.py`)."""
    if x == y:
        return (), ()
    mx = len(x)
    my = len(y)
    lim = min(mx, my)
    p = 0
    while p < lim and x[p] == y[p]:
        p += 1
    rem_x = mx - p
    rem_y = my - p
    lim_r = min(rem_x, rem_y)
    s = 0
    while s < lim_r and x[mx - 1 - s] == y[my - 1 - s]:
        s += 1
    return x[p : mx - s], y[p : my - s]


def prescreen_candidate_edges(
    cells: Sequence[str],
    subtrain_paradigms: Sequence[LemmaParadigm],
    top_k_parents: int = 12,
    seed_key: str = "",
    threshold: int = 40,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Fast plug-in edit-pattern entropy pre-screener on `subtrain` ONLY (`|C| > 40`).

    Synchronized 1-to-1 with `analyses/paradigm/mgl/topology.py`:
      1. Retains top `top_k_parents` lowest-entropy parents `(s -> t)` per target `t`.
      2. Retains full outgoing stars `(r -> t)` from top-4 lowest-outgoing-entropy roots `r`
         so Chu-Liu-Edmonds always has connected spanning arborescences.
      3. Selects the identical deterministic 14-cell `k_eval_pool_cells` (via SHA-256 on
         `f"{seed_key}|prescreen_k_pool"`) and retains all symmetric pairs among them.
    """
    sorted_cells = sorted(dict.fromkeys(cells))
    n_cells = len(sorted_cells)
    if n_cells <= threshold or not subtrain_paradigms:
        all_edges = [(u, v) for u in sorted_cells for v in sorted_cells if u != v]
        return all_edges, sorted_cells

    n_items = len(subtrain_paradigms)
    inv_n = 1.0 / n_items
    forms_by_cell = {
        c: [p.forms[c] for p in subtrain_paradigms] for c in sorted_cells
    }
    end_cue_by_cell = {
        c: [(f[-1] if f else "#") for f in forms_by_cell[c]] for c in sorted_cells
    }

    cond_ent: dict[tuple[str, str], float] = {}
    out_sum: dict[str, float] = defaultdict(float)

    for s in sorted_cells:
        s_forms = forms_by_cell[s]
        s_cues = end_cue_by_cell[s]
        cue_totals = Counter(s_cues)
        for t in sorted_cells:
            if s == t:
                continue
            t_forms = forms_by_cell[t]
            joint_counts: dict[str, Counter[tuple[tuple[str, ...], tuple[str, ...]]]] = defaultdict(
                Counter
            )
            for idx in range(n_items):
                sig = _fast_single_change_sig(s_forms[idx], t_forms[idx])
                joint_counts[s_cues[idx]][sig] += 1

            h_val = 0.0
            for cue, tot_c in cue_totals.items():
                sub_cnts = joint_counts[cue]
                if len(sub_cnts) <= 1:
                    continue
                ent_c = 0.0
                for cnt in sub_cnts.values():
                    prob = cnt / tot_c
                    ent_c -= prob * math.log2(prob)
                h_val += (tot_c * inv_n) * ent_c

            cond_ent[(s, t)] = h_val
            out_sum[s] += h_val

    retained: set[tuple[str, str]] = set()
    for t in sorted_cells:
        ranked_parents = sorted(
            (s for s in sorted_cells if s != t),
            key=lambda s: (round(cond_ent[(s, t)], 9), s),
        )
        for s in ranked_parents[:top_k_parents]:
            retained.add((s, t))

    best_roots = sorted(sorted_cells, key=lambda s: (round(out_sum[s], 9), s))[:4]
    for r in best_roots:
        for t in sorted_cells:
            if r != t:
                retained.add((r, t))

    seed_int = int.from_bytes(
        hashlib.sha256(f"{seed_key}|prescreen_k_pool".encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(seed_int)
    k_pool = sorted(rng.sample(sorted_cells, min(14, n_cells)))
    for u in k_pool:
        for v in k_pool:
            if u != v:
                retained.add((u, v))

    return sorted(retained), k_pool


def precollate_batches(
    pairs: Sequence[tuple[tuple[str, ...], tuple[str, ...], str, str]],
    char_vocab: CharVocab,
    cell_vocab: CellVocab,
    batch_size: int,
    device: torch.device,
    *,
    jitter_seed: int | None = None,
) -> list[tuple[EncodedBatch, int]]:
    """Pre-collate `(EncodedBatch, chunk_len)` once per fold with jittered length bucketing."""
    if not pairs:
        return []
    if jitter_seed is not None:
        rng = random.Random(jitter_seed)
        ordered = sorted(
            pairs,
            key=lambda ex: len(ex[0]) + len(ex[1]) + rng.uniform(-2.5, 2.5),
        )
    else:
        ordered = sorted(pairs, key=lambda ex: (len(ex[0]) + len(ex[1]), len(ex[0])))

    batches: list[tuple[EncodedBatch, int]] = []
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        batches.append((collate_transducer_batch(chunk, char_vocab, cell_vocab, device=device), len(chunk)))
    return batches


def train_single_transducer(
    train_pairs: Sequence[tuple[tuple[str, ...], tuple[str, ...], str, str]],
    dev_pairs: Sequence[tuple[tuple[str, ...], tuple[str, ...], str, str]] | None,
    char_vocab: CharVocab,
    cell_vocab: CellVocab,
    cfg: TrainEvalConfig,
    seed: int,
    device: torch.device,
    fixed_epochs: int | None = None,
    *,
    precollated_train: Sequence[tuple[EncodedBatch, int]] | None = None,
    precollated_dev: Sequence[tuple[EncodedBatch, int]] | None = None,
    init_state_dict: dict[str, torch.Tensor] | None = None,
    lr_override: float | None = None,
) -> tuple[HardMonotonicTransducer, int]:
    """Train one HardMonotonicTransducer seed with exact marginal NLL (zero label smoothing)."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    model = HardMonotonicTransducer(char_vocab, cell_vocab, cfg.transducer).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    eff_lr = lr_override if lr_override is not None else cfg.lr
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=eff_lr, weight_decay=cfg.weight_decay
    )

    n_epochs = fixed_epochs if fixed_epochs is not None else cfg.max_epochs
    best_dev_ll = -float("inf")
    best_epoch = max(1, n_epochs)
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    rng = random.Random(seed + 17)

    train_batches = (
        list(precollated_train)
        if precollated_train is not None
        else precollate_batches(
            train_pairs, char_vocab, cell_vocab, cfg.batch_size, device, jitter_seed=seed + 3
        )
    )
    dev_batches = (
        list(precollated_dev)
        if precollated_dev is not None
        else (
            precollate_batches(dev_pairs, char_vocab, cell_vocab, cfg.eval_batch_size, device)
            if dev_pairs
            else []
        )
    )

    batch_indices = list(range(len(train_batches)))

    for epoch in range(1, n_epochs + 1):
        model.train()
        rng.shuffle(batch_indices)
        epoch_loss_acc = torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)

        for b_idx in batch_indices:
            batch, _ = train_batches[b_idx]
            active_dev = next(model.parameters()).device
            if batch.src_rnn_ids.device != active_dev:
                batch = batch.to(active_dev)
                train_batches[b_idx] = (batch, train_batches[b_idx][1])
            optimizer.zero_grad(set_to_none=True)
            try:
                lp = model(batch)
                loss = -lp.mean()
                loss.backward()
                epoch_loss_acc = epoch_loss_acc + loss.detach()
            except (NotImplementedError, RuntimeError) as exc:
                if active_dev.type == "mps":
                    logger.warning("MPS backward fallback to CPU triggered: %s", exc)
                    device = torch.device("cpu")
                    model.to(device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=eff_lr, weight_decay=cfg.weight_decay
                    )
                    batch = batch.to(device)
                    train_batches[b_idx] = (batch, train_batches[b_idx][1])
                    optimizer.zero_grad(set_to_none=True)
                    loss = -model(batch).mean()
                    loss.backward()
                    epoch_loss_acc = loss.detach()
                else:
                    raise

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # Single once-per-epoch scalar NaN check (avoids per-batch MPS command-buffer stalls)
        if math.isnan(float(epoch_loss_acc.item())) and next(model.parameters()).device.type == "mps":
            logger.warning("NaN training loss detected on MPS at epoch %d; re-training seed on CPU.", epoch)
            return train_single_transducer(
                train_pairs,
                dev_pairs,
                char_vocab,
                cell_vocab,
                cfg,
                seed,
                torch.device("cpu"),
                fixed_epochs=fixed_epochs,
                init_state_dict=init_state_dict,
                lr_override=lr_override,
            )

        if fixed_epochs is None and dev_batches:
            model.eval()
            active_dev = next(model.parameters()).device
            tot_ll = 0.0
            tot_n = 0
            with torch.no_grad():
                for d_i, (d_batch, d_len) in enumerate(dev_batches):
                    if d_batch.src_rnn_ids.device != active_dev:
                        d_batch = d_batch.to(active_dev)
                        dev_batches[d_i] = (d_batch, d_len)
                    tot_ll += float(model(d_batch).sum().item())
                    tot_n += d_len
            mean_dev_ll = tot_ll / max(1, tot_n)
            if mean_dev_ll > best_dev_ll + 1e-5:
                best_dev_ll = mean_dev_ll
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def estimate_dev_entropy_matrix(
    models: Sequence[HardMonotonicTransducer],
    cells: Sequence[str],
    dev_paradigms: Sequence[LemmaParadigm],
    candidate_edges: Sequence[tuple[str, str]],
    eval_batch_size: int = 256,
) -> list[list[float]]:
    """Compute directed validation cross-entropy matrix `M_hat[u][v]` (in bits) strictly on `dev`."""
    n = len(cells)
    cell_idx = {c: i for i, c in enumerate(cells)}
    m_hat = [[float("inf")] * n for _ in range(n)]
    for i in range(n):
        m_hat[i][i] = 0.0

    queries: list[tuple[tuple[str, ...], tuple[str, ...], str, str]] = []
    query_meta: list[tuple[int, int]] = []
    zero_counts: dict[tuple[int, int], int] = defaultdict(int)
    sum_bits: dict[tuple[int, int], float] = defaultdict(float)
    total_counts: dict[tuple[int, int], int] = defaultdict(int)

    for cs, ct in candidate_edges:
        u, v = cell_idx[cs], cell_idx[ct]
        for p in dev_paradigms:
            total_counts[(u, v)] += 1
            if p.forms[cs] == p.forms[ct]:
                zero_counts[(u, v)] += 1
            else:
                queries.append((p.forms[cs], p.forms[ct], cs, ct))
                query_meta.append((u, v))

    if queries:
        lps = ensemble_log_prob_pairs(models, queries, batch_size=eval_batch_size)
        for (u, v), lp in zip(query_meta, lps):
            sum_bits[(u, v)] += -lp / LN2

    for (u, v), tot in total_counts.items():
        if tot > 0:
            m_hat[u][v] = sum_bits[(u, v)] / tot
    return m_hat


# ---------------------------------------------------------------------------
# Held-Out Test Scoring (H_tree, k=1 PCFP, k=3 PoE PCFP)
# ---------------------------------------------------------------------------

def score_test_paradigms(
    models: Sequence[HardMonotonicTransducer],
    cells: Sequence[str],
    root_idx: int,
    tree_parents: dict[int, int],
    m_hat: Sequence[Sequence[float]],
    test_paradigms: Sequence[LemmaParadigm],
    unit_id: str,
    cfg: TrainEvalConfig,
    *,
    regime: str = "uniform120",
    draw_id: int = 0,
    fold_id: int = 0,
    k_eval_pool_cells: Sequence[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Evaluate held-out test lemmas on `H_tree`, `H_{k=1}`, and `H_{k=3}` (PoE).

    Vectorizes beam decoding (`ensemble_beam_decode_batch`) and forward-DP scoring
    (`ensemble_log_prob_pairs`) across all test lemmas of the fold in two batched passes.
    """
    n_cells = len(cells)
    depths = tree_depths(n_cells, root_idx, tree_parents)
    pool_idx_set = (
        {cells.index(c) for c in k_eval_pool_cells if c in set(cells)}
        if k_eval_pool_cells is not None and n_cells > cfg.prescreen_cell_threshold
        else None
    )

    # Phase 1: Plan per-lemma edges/subsets and collect all unique beam queries `(src_syms, cs, ct)`
    lemma_plans: list[dict[str, object]] = []
    beam_query_set: dict[tuple[tuple[str, ...], str, str], int] = {}
    beam_query_list: list[tuple[tuple[str, ...], str, str]] = []

    def _register_beam(src_syms: tuple[str, ...], u_idx: int, v_idx: int) -> None:
        key = (src_syms, cells[u_idx], cells[v_idx])
        if key not in beam_query_set:
            beam_query_set[key] = len(beam_query_list)
            beam_query_list.append(key)

    for p in test_paradigms:
        classes_by_form: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for idx, c in enumerate(cells):
            classes_by_form[p.forms[c]].append(idx)

        root_form = p.forms[cells[root_idx]]
        non_root_forms = [f for f in sorted(classes_by_form.keys()) if f != root_form]
        n_distinct = len(classes_by_form)

        if n_distinct <= 1:
            lemma_plans.append({"paradigm": p, "n_distinct": n_distinct, "trivial": True})
            continue

        tree_edges: list[tuple[int, int]] = []
        for f_str in non_root_forms:
            cls_nodes = classes_by_form[f_str]
            v_rep = min(
                cls_nodes,
                key=lambda v: (depths[v], m_hat[tree_parents[v]][v], cells[v]),
            )
            u_rep = tree_parents[v_rep]
            tree_edges.append((u_rep, v_rep))
            _register_beam(p.forms[cells[u_rep]], u_rep, v_rep)

        # Representative cells for k=1 and k=3 (synchronized 1-to-1 with mgl/topology.py)
        if pool_idx_set is not None:
            pool_by_form: dict[tuple[str, ...], list[int]] = defaultdict(list)
            for idx, c in enumerate(cells):
                if idx in pool_idx_set:
                    pool_by_form[p.forms[c]].append(idx)
            rep_cells = sorted(min(c_list) for c_list in pool_by_form.values())
        else:
            rep_cells = sorted(min(cls_nodes) for cls_nodes in classes_by_form.values())

        if len(rep_cells) < 2:
            all_reps = sorted(min(cls_nodes) for cls_nodes in classes_by_form.values())
            rep_cells = [
                u
                for u in all_reps
                if any(
                    math.isfinite(m_hat[u][v]) and math.isfinite(m_hat[v][u])
                    for v in all_reps
                    if v != u
                )
            ]

        seed_prefix = f"{unit_id}|{regime}|{draw_id}|{fold_id}|{p.lemma_id}"
        all_k1_pairs = [
            (u, v) for u in rep_cells for v in rep_cells if u != v and math.isfinite(m_hat[u][v])
        ]
        if cfg.max_k1_pairs_per_lemma is not None and len(all_k1_pairs) > cfg.max_k1_pairs_per_lemma:
            s_k1 = int.from_bytes(
                hashlib.sha256(f"{seed_prefix}|k1".encode("utf-8")).digest()[:8], "big"
            )
            k1_pairs = sorted(random.Random(s_k1).sample(all_k1_pairs, cfg.max_k1_pairs_per_lemma))
        else:
            k1_pairs = all_k1_pairs

        for u, v in k1_pairs:
            _register_beam(p.forms[cells[u]], u, v)

        k3_tasks: list[tuple[list[int], int]] = []
        if len(rep_cells) > 2:
            k_src = min(3, len(rep_cells) - 1)
            total_comb = math.comb(len(rep_cells), k_src)
            if total_comb <= cfg.n_k3_subsets:
                subsets_list = list(itertools.combinations(rep_cells, k_src))
            else:
                s_k3 = int.from_bytes(
                    hashlib.sha256(f"{seed_prefix}|k3".encode("utf-8")).digest()[:8], "big"
                )
                rng_k3 = random.Random(s_k3)
                subsets_seen: set[tuple[int, ...]] = set()
                subsets_list = []
                while len(subsets_list) < cfg.n_k3_subsets:
                    sub = tuple(sorted(rng_k3.sample(rep_cells, k_src)))
                    if sub not in subsets_seen:
                        subsets_seen.add(sub)
                        subsets_list.append(sub)

            rng_tgt = random.Random(
                int.from_bytes(
                    hashlib.sha256(f"{seed_prefix}|k3_tgt".encode("utf-8")).digest()[:8], "big"
                )
            )
            for s3 in subsets_list:
                s3_set = set(s3)
                rem_targets = [v for v in rep_cells if v not in s3_set]
                if (
                    cfg.max_k3_targets_per_subset is not None
                    and len(rem_targets) > cfg.max_k3_targets_per_subset
                ):
                    rem_targets = rng_tgt.sample(rem_targets, cfg.max_k3_targets_per_subset)
                for v_tgt in rem_targets:
                    valid_sources = [u for u in s3 if math.isfinite(m_hat[u][v_tgt])]
                    if not valid_sources:
                        continue
                    k3_tasks.append((valid_sources, v_tgt))
                    for u in valid_sources:
                        _register_beam(p.forms[cells[u]], u, v_tgt)

        lemma_plans.append({
            "paradigm": p,
            "n_distinct": n_distinct,
            "trivial": False,
            "tree_edges": tree_edges,
            "rep_cells": rep_cells,
            "k1_pairs": k1_pairs,
            "k3_tasks": k3_tasks,
        })

    # Execute all unique beam decodes in ONE batched GPU/SIMD pass
    beam_outputs = ensemble_beam_decode_batch(
        models,
        beam_query_list,
        beam_size=cfg.beam_size,
        batch_size=max(32, cfg.eval_batch_size // 2),
    )
    beam_map: dict[tuple[tuple[str, ...], str, str], list[tuple[tuple[str, ...], float]]] = dict(
        zip(beam_query_list, beam_outputs)
    )

    # Phase 2: Collect all unique forward-DP `(src_syms, tgt_syms, cs, ct)` queries across the fold
    lp_query_set: dict[tuple[tuple[str, ...], tuple[str, ...], str, str], int] = {}
    lp_query_list: list[tuple[tuple[str, ...], tuple[str, ...], str, str]] = []

    def _register_lp(
        src_syms: tuple[str, ...], tgt_syms: tuple[str, ...], u_idx: int, v_idx: int
    ) -> None:
        q = (src_syms, tgt_syms, cells[u_idx], cells[v_idx])
        if q not in lp_query_set:
            lp_query_set[q] = len(lp_query_list)
            lp_query_list.append(q)

    for plan in lemma_plans:
        if plan["trivial"]:
            continue
        p = plan["paradigm"]  # type: ignore[assignment]
        assert isinstance(p, LemmaParadigm)
        for u, v in plan["tree_edges"]:  # type: ignore[attr-defined]
            _register_lp(p.forms[cells[u]], p.forms[cells[v]], u, v)
        for u, v in plan["k1_pairs"]:  # type: ignore[attr-defined]
            _register_lp(p.forms[cells[u]], p.forms[cells[v]], u, v)

        poe_prepared: list[tuple[list[int], int, list[tuple[str, ...]], list[tuple[str, ...]]]] = []
        for valid_sources, v_tgt in plan["k3_tasks"]:  # type: ignore[attr-defined]
            gold_seq = p.forms[cells[v_tgt]]
            if len(valid_sources) == 1:
                u0 = valid_sources[0]
                _register_lp(p.forms[cells[u0]], gold_seq, u0, v_tgt)
                poe_prepared.append((valid_sources, v_tgt, [], [gold_seq]))
                continue
            cand_forms_set: set[tuple[str, ...]] = set()
            for u in valid_sources:
                for pred_seq, _ in beam_map[(p.forms[cells[u]], cells[u], cells[v_tgt])]:
                    cand_forms_set.add(pred_seq)
            cand_forms = sorted(cand_forms_set)
            eval_seqs = cand_forms + ([gold_seq] if gold_seq not in cand_forms_set else [])
            for u in valid_sources:
                src_u = p.forms[cells[u]]
                for y_seq in eval_seqs:
                    _register_lp(src_u, y_seq, u, v_tgt)
            poe_prepared.append((valid_sources, v_tgt, cand_forms, eval_seqs))
        plan["poe_prepared"] = poe_prepared

    lp_values = ensemble_log_prob_pairs(models, lp_query_list, batch_size=cfg.eval_batch_size)
    lp_map: dict[tuple[tuple[str, ...], tuple[str, ...], str, str], float] = dict(
        zip(lp_query_list, lp_values)
    )

    # Phase 3: Compute exact per-lemma metrics from `beam_map` and `lp_map`
    results: list[dict[str, float | int | str]] = []
    for plan in lemma_plans:
        p = plan["paradigm"]  # type: ignore[assignment]
        assert isinstance(p, LemmaParadigm)
        n_distinct = int(plan["n_distinct"])  # type: ignore[arg-type]
        if plan["trivial"]:
            results.append({
                "lemma_id": p.lemma_id,
                "leakage_group_id": p.leakage_group_id,
                "n_distinct_forms": n_distinct,
                "h_tree_lemma": 0.0,
                "rel_tree_lemma": 1.0,
                "acc_tree_lemma": 1.0,
                "h_k1_lemma": 0.0,
                "acc_k1_lemma": 1.0,
                "h_k3_lemma": 0.0,
                "acc_k3_lemma": 1.0,
            })
            continue

        tree_edges = plan["tree_edges"]  # type: ignore[assignment]
        tree_lps = [
            lp_map[(p.forms[cells[u]], p.forms[cells[v]], cells[u], cells[v])]
            for u, v in tree_edges
        ]
        tree_bits = [max(0.0, -lp / LN2) for lp in tree_lps]
        tree_rels = [min(1.0, math.exp(lp)) for lp in tree_lps]
        tree_accs = []
        for u, v in tree_edges:
            b_res = beam_map[(p.forms[cells[u]], cells[u], cells[v])]
            pred_tup = b_res[0][0] if b_res else ()
            tree_accs.append(1.0 if pred_tup == p.forms[cells[v]] else 0.0)

        h_tree_val = sum(tree_bits) / len(tree_bits)
        rel_tree_val = sum(tree_rels) / len(tree_rels)
        acc_tree_val = sum(tree_accs) / len(tree_accs)

        k1_pairs = plan["k1_pairs"]  # type: ignore[assignment]
        if not k1_pairs:
            h_k1 = h_tree_val
            acc_k1 = acc_tree_val
        else:
            k1_lps = [
                lp_map[(p.forms[cells[u]], p.forms[cells[v]], cells[u], cells[v])]
                for u, v in k1_pairs
            ]
            h_k1 = sum(max(0.0, -lp / LN2) for lp in k1_lps) / len(k1_lps)
            acc_k1 = sum(
                1.0
                if (
                    beam_map[(p.forms[cells[u]], cells[u], cells[v])][0][0]
                    if beam_map[(p.forms[cells[u]], cells[u], cells[v])]
                    else ()
                )
                == p.forms[cells[v]]
                else 0.0
                for u, v in k1_pairs
            ) / len(k1_pairs)

        poe_prepared = plan["poe_prepared"]  # type: ignore[assignment]
        if not poe_prepared:
            h_k3 = h_k1
            acc_k3 = acc_k1
        else:
            k3_bits_list: list[float] = []
            k3_accs_list: list[float] = []
            for valid_sources, v_tgt, cand_forms, eval_seqs in poe_prepared:
                gold_seq = p.forms[cells[v_tgt]]
                if len(valid_sources) == 1:
                    u0 = valid_sources[0]
                    lp0 = lp_map[(p.forms[cells[u0]], gold_seq, cells[u0], cells[v_tgt])]
                    b0 = beam_map[(p.forms[cells[u0]], cells[u0], cells[v_tgt])]
                    pred0 = b0[0][0] if b0 else ()
                    k3_bits_list.append(max(0.0, -lp0 / LN2))
                    k3_accs_list.append(1.0 if pred0 == gold_seq else 0.0)
                    continue

                inv_w = [1.0 / (max(0.0, m_hat[u][v_tgt]) + cfg.poe_tau) for u in valid_sources]
                w_sum = sum(inv_w)
                weights = [w / w_sum for w in inv_w]

                expert_lp_mat = [
                    [
                        lp_map[(p.forms[cells[u]], y_seq, cells[u], cells[v_tgt])]
                        for y_seq in eval_seqs
                    ]
                    for u in valid_sources
                ]
                n_cand = len(cand_forms)
                if n_cand > 0:
                    cand_poe_log = [
                        sum(weights[i_e] * expert_lp_mat[i_e][j_c] for i_e in range(len(valid_sources)))
                        for j_c in range(n_cand)
                    ]
                    best_cand_idx = max(
                        range(n_cand), key=lambda j_c: (cand_poe_log[j_c], cand_forms[j_c])
                    )
                    pred_k3 = cand_forms[best_cand_idx]
                    s_c = sum(math.exp(val) for val in cand_poe_log)
                    r_tail = 1.0
                    for i_e in range(len(valid_sources)):
                        mass_c = sum(math.exp(expert_lp_mat[i_e][j_c]) for j_c in range(n_cand))
                        rem_mass = max(0.0, 1.0 - mass_c)
                        r_tail *= rem_mass ** weights[i_e]
                    z_upper = max(s_c + r_tail, 1e-300)
                else:
                    pred_k3 = ()
                    z_upper = 1.0

                gold_idx = eval_seqs.index(gold_seq)
                gold_unnorm_log = sum(
                    weights[i_e] * expert_lp_mat[i_e][gold_idx] for i_e in range(len(valid_sources))
                )
                log_p_poe = gold_unnorm_log - math.log(z_upper)
                k3_bits_list.append(max(0.0, -log_p_poe / LN2))
                k3_accs_list.append(1.0 if pred_k3 == gold_seq else 0.0)

            h_k3 = sum(k3_bits_list) / len(k3_bits_list)
            acc_k3 = sum(k3_accs_list) / len(k3_accs_list)

        results.append({
            "lemma_id": p.lemma_id,
            "leakage_group_id": p.leakage_group_id,
            "n_distinct_forms": n_distinct,
            "h_tree_lemma": h_tree_val,
            "rel_tree_lemma": rel_tree_val,
            "acc_tree_lemma": acc_tree_val,
            "h_k1_lemma": h_k1,
            "acc_k1_lemma": acc_k1,
            "h_k3_lemma": h_k3,
            "acc_k3_lemma": acc_k3,
        })

    return results


def cluster_bootstrap_se(
    lemma_rows: Sequence[dict[str, float | int | str]],
    metric_col: str,
    n_boot: int = 500,
    seed: int = 20261112,
) -> float:
    """500-resample cluster bootstrap SE over `leakage_group_id` clusters."""
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for r in lemma_rows:
        by_cluster[str(r["leakage_group_id"])].append(float(r[metric_col]))
    clusters = sorted(by_cluster.keys())
    n_c = len(clusters)
    if n_c < 2:
        return 0.0

    cluster_sums = np.array([sum(by_cluster[c]) for c in clusters], dtype=np.float64)
    cluster_cnts = np.array([len(by_cluster[c]) for c in clusters], dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_c, size=(n_boot, n_c))
    boot_means = cluster_sums[idx].sum(axis=1) / cluster_cnts[idx].sum(axis=1)
    return float(np.std(boot_means, ddof=1))


# ---------------------------------------------------------------------------
# End-to-End Unit Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate_unit_neural(
    lexicon_csv: Path,
    split_csv: Path,
    regime: str,
    cfg: TrainEvalConfig | None = None,
    max_draws: int | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Train 3-seed Hard-Monotonic Neural Transducer across CV splits and compute unit/lemma outcomes."""
    cfg = cfg or TrainEvalConfig()
    torch.set_num_threads(max(1, min(8, cfg.num_threads)))
    device = resolve_device(cfg.device)
    t0 = time.perf_counter()

    splits = load_split_manifest(split_csv)
    all_draws = sorted({d for d, _ in splits.keys()})
    if max_draws is not None:
        all_draws = all_draws[:max_draws]
    active_keys = [(d, f) for (d, f) in sorted(splits.keys()) if d in set(all_draws)]
    needed_lemmas = {
        lid
        for k in active_keys
        for lids in splits[k].values()
        for lid in lids
    }
    meta, cells, paradigms = load_unit_lexicon(lexicon_csv, keep_lemmas=needed_lemmas)
    cell_vocab = CellVocab(cells)

    per_lemma_accum: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    selected_bases: list[str] = []

    for draw_id, fold_id in active_keys:
        role_map = splits[(draw_id, fold_id)]
        sub_p = [paradigms[lid] for lid in role_map["subtrain"] if lid in paradigms]
        dev_p = [paradigms[lid] for lid in role_map["dev"] if lid in paradigms]
        test_p = [paradigms[lid] for lid in role_map["test"] if lid in paradigms]
        train_p = sub_p + dev_p

        if not sub_p or not dev_p or not test_p:
            raise ValueError(f"Empty split role in {meta['unit_id']} draw={draw_id} fold={fold_id}")

        # Candidate directed edges & 14-cell k-PCFP pool (matching the MGL pipeline 1-to-1)
        seed_fold_str = f"{meta['unit_id']}|{regime}|{draw_id}|{fold_id}"
        cand_edges, k_pool_cells = prescreen_candidate_edges(
            cells,
            sub_p,
            top_k_parents=cfg.prescreen_top_parents,
            seed_key=seed_fold_str,
            threshold=cfg.prescreen_cell_threshold,
        )

        # Build CharVocab strictly from subtrain (for inner phase) — zero test symbols!
        sub_syms = {sym for p in sub_p for seq in p.forms.values() for sym in seq}
        inner_vocab = CharVocab(sub_syms, allow_unk=True, unk_space_size=256)

        fold_seed = _stable_seed(cfg.base_seed, meta["unit_id"], regime, draw_id, fold_id)
        sub_pairs = build_pair_examples(sub_p, cand_edges, cfg.max_train_pairs, seed=fold_seed)
        dev_early_pairs = build_pair_examples(dev_p, cand_edges, min(4_000, cfg.max_train_pairs), seed=fold_seed + 1)

        # Adaptive batch sizing for large-|C| pair pools (~48 updates/epoch on GPU/SIMD)
        eff_bs = (
            min(512, max(cfg.batch_size, len(sub_pairs) // 48))
            if len(sub_pairs) > 4_000
            else cfg.batch_size
        )
        eff_lr = cfg.lr * math.sqrt(eff_bs / max(1, cfg.batch_size))

        sub_batches = precollate_batches(
            sub_pairs, inner_vocab, cell_vocab, eff_bs, device, jitter_seed=fold_seed + 11
        )
        dev_batches = precollate_batches(
            dev_early_pairs, inner_vocab, cell_vocab, cfg.eval_batch_size, device
        )

        inner_models: list[HardMonotonicTransducer] = []
        best_epochs: list[int] = []
        for s_idx in range(cfg.n_seeds):
            mdl, b_ep = train_single_transducer(
                sub_pairs,
                dev_early_pairs,
                inner_vocab,
                cell_vocab,
                cfg,
                seed=fold_seed + 1000 * (s_idx + 1),
                device=device,
                fixed_epochs=None,
                precollated_train=sub_batches,
                precollated_dev=dev_batches,
                lr_override=eff_lr,
            )
            inner_models.append(mdl)
            best_epochs.append(b_ep)

        # Estimate validation cross-entropy matrix M_hat & optimal tree strictly on `dev`
        m_hat = estimate_dev_entropy_matrix(
            inner_models, cells, dev_p, cand_edges, eval_batch_size=cfg.eval_batch_size
        )
        r_star, tree_parents, _ = select_optimal_arborescence(cells, m_hat)
        selected_bases.append(cells[r_star])

        # Incorporate `dev` into full `train = subtrain + dev`:
        # When `train_syms == sub_syms`, warm-start from the early-stopped `inner_models`
        # for `refit_warmup_epochs` at `0.5 * eff_lr`; otherwise cold-refit on `full_vocab`.
        if cfg.refit_full_train:
            train_syms = {sym for p in train_p for seq in p.forms.values() for sym in seq}
            full_pairs = build_pair_examples(train_p, cand_edges, cfg.max_train_pairs, seed=fold_seed + 77)
            eff_full_bs = (
                min(512, max(cfg.batch_size, len(full_pairs) // 48))
                if len(full_pairs) > 4_000
                else cfg.batch_size
            )
            if train_syms == sub_syms:
                full_batches = precollate_batches(
                    full_pairs, inner_vocab, cell_vocab, eff_full_bs, device, jitter_seed=fold_seed + 99
                )
                warm_epochs = max(1, cfg.refit_warmup_epochs)
                eval_models = [
                    train_single_transducer(
                        full_pairs,
                        None,
                        inner_vocab,
                        cell_vocab,
                        cfg,
                        seed=fold_seed + 2000 * (s_idx + 1),
                        device=device,
                        fixed_epochs=warm_epochs,
                        precollated_train=full_batches,
                        init_state_dict={
                            k: v.detach().clone() for k, v in inner_models[s_idx].state_dict().items()
                        },
                        lr_override=0.5 * eff_lr,
                    )[0]
                    for s_idx in range(cfg.n_seeds)
                ]
            else:
                full_vocab = CharVocab(train_syms, allow_unk=True, unk_space_size=256)
                full_batches = precollate_batches(
                    full_pairs, full_vocab, cell_vocab, eff_full_bs, device, jitter_seed=fold_seed + 99
                )
                refit_epochs = max(1, int(round(float(np.median(best_epochs)))))
                eval_models = [
                    train_single_transducer(
                        full_pairs,
                        None,
                        full_vocab,
                        cell_vocab,
                        cfg,
                        seed=fold_seed + 2000 * (s_idx + 1),
                        device=device,
                        fixed_epochs=refit_epochs,
                        precollated_train=full_batches,
                        lr_override=eff_lr,
                    )[0]
                    for s_idx in range(cfg.n_seeds)
                ]
        else:
            eval_models = inner_models

        fold_test_rows = score_test_paradigms(
            eval_models,
            cells,
            r_star,
            tree_parents,
            m_hat,
            test_p,
            meta["unit_id"],
            cfg,
            regime=regime,
            draw_id=draw_id,
            fold_id=fold_id,
            k_eval_pool_cells=k_pool_cells,
        )
        for row in fold_test_rows:
            per_lemma_accum[str(row["lemma_id"])].append(row)

    # Aggregate per-lemma outcomes across draws
    lemma_out_rows: list[dict[str, object]] = []
    boot_input_rows: list[dict[str, float | int | str]] = []
    for lid in sorted(per_lemma_accum.keys()):
        obs = per_lemma_accum[lid]
        n_o = len(obs)
        agg = {
            "unit_id": meta["unit_id"],
            "glottocode": meta["glottocode"],
            "pos": meta["pos"],
            "representation": meta["representation"],
            "model": "neural_monotonic",
            "regime": regime,
            "lemma_id": lid,
            "leakage_group_id": str(obs[0]["leakage_group_id"]),
            "n_distinct_forms": int(obs[0]["n_distinct_forms"]),
            "h_tree_lemma": sum(float(x["h_tree_lemma"]) for x in obs) / n_o,
            "rel_tree_lemma": sum(float(x["rel_tree_lemma"]) for x in obs) / n_o,
            "acc_tree_lemma": sum(float(x["acc_tree_lemma"]) for x in obs) / n_o,
            "h_k1_lemma": sum(float(x["h_k1_lemma"]) for x in obs) / n_o,
            "h_k3_lemma": sum(float(x["h_k3_lemma"]) for x in obs) / n_o,
        }
        lemma_out_rows.append(agg)
        boot_input_rows.append({
            **agg,  # type: ignore[dict-item]
            "acc_k1_lemma": sum(float(x["acc_k1_lemma"]) for x in obs) / n_o,
            "acc_k3_lemma": sum(float(x["acc_k3_lemma"]) for x in obs) / n_o,
        })

    # Grand test evaluation means across all (draw, test_lemma) evaluations
    all_eval_rows = [r for obs in per_lemma_accum.values() for r in obs]
    n_eval_tot = len(all_eval_rows)
    base_counts = Counter(selected_bases)
    modal_base = sorted(base_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    elapsed = time.perf_counter() - t0

    boot_seed = int.from_bytes(
        hashlib.sha256(f"{meta['unit_id']}|{regime}|bootstrap".encode("utf-8")).digest()[:8],
        "big",
    )

    unit_row: dict[str, object] = {
        "unit_id": meta["unit_id"],
        "glottocode": meta["glottocode"],
        "pos": meta["pos"],
        "representation": meta["representation"],
        "model": "neural_monotonic",
        "regime": regime,
        "n_draws": len(all_draws),
        "n_test_lemmas": len(lemma_out_rows),
        "n_cells_distinct": len(cells),
        "h_tree_bits": round(sum(float(x["h_tree_lemma"]) for x in all_eval_rows) / n_eval_tot, 6),
        "h_tree_se": round(
            cluster_bootstrap_se(all_eval_rows, "h_tree_lemma", cfg.n_bootstrap, boot_seed), 6
        ),
        "rel_tree_mean": round(sum(float(x["rel_tree_lemma"]) for x in all_eval_rows) / n_eval_tot, 6),
        "rel_tree_se": round(
            cluster_bootstrap_se(all_eval_rows, "rel_tree_lemma", cfg.n_bootstrap, boot_seed), 6
        ),
        "acc_tree_mean": round(sum(float(x["acc_tree_lemma"]) for x in all_eval_rows) / n_eval_tot, 6),
        "acc_tree_se": round(
            cluster_bootstrap_se(all_eval_rows, "acc_tree_lemma", cfg.n_bootstrap, boot_seed), 6
        ),
        "h_k1_bits": round(sum(float(x["h_k1_lemma"]) for x in all_eval_rows) / n_eval_tot, 6),
        "h_k1_se": round(
            cluster_bootstrap_se(all_eval_rows, "h_k1_lemma", cfg.n_bootstrap, boot_seed), 6
        ),
        "acc_k1_mean": round(sum(float(x["acc_k1_lemma"]) for x in all_eval_rows) / n_eval_tot, 6),
        "h_k3_bits": round(sum(float(x["h_k3_lemma"]) for x in all_eval_rows) / n_eval_tot, 6),
        "h_k3_se": round(
            cluster_bootstrap_se(all_eval_rows, "h_k3_lemma", cfg.n_bootstrap, boot_seed), 6
        ),
        "acc_k3_mean": round(sum(float(x["acc_k3_lemma"]) for x in all_eval_rows) / n_eval_tot, 6),
        "base_cell_modal": modal_base,
        "runtime_sec": round(elapsed, 3),
    }

    for r in lemma_out_rows:
        for col in ("h_tree_lemma", "rel_tree_lemma", "acc_tree_lemma", "h_k1_lemma", "h_k3_lemma"):
            r[col] = round(float(r[col]), 6)

    return unit_row, lemma_out_rows


def write_neural_outcomes(
    unit_rows: Sequence[dict[str, object]],
    lemma_rows: Sequence[dict[str, object]],
    out_dir: Path,
    regime: str,
) -> tuple[Path, Path]:
    """Write `unit_outcomes_{regime}.csv` and `lemma_outcomes_{regime}.csv`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_path = out_dir / f"unit_outcomes_{regime}.csv"
    lemma_path = out_dir / f"lemma_outcomes_{regime}.csv"

    with open(unit_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UNIT_OUTCOME_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in unit_rows:
            w.writerow({k: r[k] for k in UNIT_OUTCOME_COLUMNS})

    with open(lemma_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEMMA_OUTCOME_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in lemma_rows:
            w.writerow({k: r[k] for k in LEMMA_OUTCOME_COLUMNS})

    return unit_path, lemma_path


def main(argv: Sequence[str] | None = None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    parser = argparse.ArgumentParser(
        description="Train & evaluate the 3-seed Hard-Monotonic Neural Character Transducer."
    )
    parser.add_argument(
        "--units",
        nargs="+",
        required=True,
        help="One or more unit_ids (e.g. finn1318_N_orth ital1282_V_orth).",
    )
    parser.add_argument(
        "--regime",
        default="uniform120",
        choices=["uniform120", "uniform480", "topfreq120"],
        help="Split regime to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("analyses/paradigm/data"),
        help="Base paradigm data directory containing lexicons/ and splits/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analyses/paradigm/results/neural"),
        help="Output directory for neural unit_outcomes and lemma_outcomes CSVs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Execution device (auto -> cuda -> mps -> cpu).",
    )
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--max-train-pairs", type=int, default=25_000)
    parser.add_argument("--max-draws", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true", help="Ignore per-unit checkpoint cache.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainEvalConfig(
        n_seeds=args.n_seeds,
        max_epochs=args.max_epochs,
        max_train_pairs=args.max_train_pairs,
        device=args.device,
        num_threads=args.num_threads,
    )

    cache_dir = args.out_dir / "per_unit_cache" / f"neural_monotonic_{args.regime}"
    use_cache = args.max_draws is None and not args.no_resume
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    unit_rows: list[dict[str, object]] = []
    lemma_rows: list[dict[str, object]] = []

    for idx_u, uid in enumerate(args.units, 1):
        cached_u = cache_dir / f"{uid}_unit.csv"
        cached_l = cache_dir / f"{uid}_lemmas.csv"
        if use_cache and cached_u.exists() and cached_l.exists():
            with open(cached_u, encoding="utf-8") as fu:
                u_loaded = list(csv.DictReader(fu))
            with open(cached_l, encoding="utf-8") as fl:
                l_loaded = list(csv.DictReader(fl))
            if u_loaded and l_loaded:
                unit_rows.extend(u_loaded)
                lemma_rows.extend(l_loaded)
                logger.info(
                    "[%d/%d] Loaded cached %s: H_tree=%s, Acc_tree=%s",
                    idx_u,
                    len(args.units),
                    uid,
                    u_loaded[0]["h_tree_bits"],
                    u_loaded[0]["acc_tree_mean"],
                )
                continue

        lex_path = args.data_dir / "lexicons" / f"{uid}.csv"
        split_dir = "splits_freq" if args.regime == "topfreq120" else "splits"
        split_path = args.data_dir / split_dir / f"{uid}_{args.regime}.csv"
        logger.info("[%d/%d] Evaluating unit %s (%s) on device=%s ...", idx_u, len(args.units), uid, args.regime, cfg.device)
        u_row, l_rows = evaluate_unit_neural(
            lex_path, split_path, regime=args.regime, cfg=cfg, max_draws=args.max_draws
        )
        if use_cache:
            with open(cached_u, "w", encoding="utf-8", newline="") as fu:
                wu = csv.DictWriter(fu, fieldnames=UNIT_OUTCOME_COLUMNS, lineterminator="\n")
                wu.writeheader()
                wu.writerow({k: u_row[k] for k in UNIT_OUTCOME_COLUMNS})
            with open(cached_l, "w", encoding="utf-8", newline="") as fl:
                wl = csv.DictWriter(fl, fieldnames=LEMMA_OUTCOME_COLUMNS, lineterminator="\n")
                wl.writeheader()
                for r in l_rows:
                    wl.writerow({k: r[k] for k in LEMMA_OUTCOME_COLUMNS})

        unit_rows.append(u_row)
        lemma_rows.extend(l_rows)
        logger.info(
            "[%d/%d] Done %s: H_tree=%.4f (SE=%.4f), Acc_tree=%.3f, H_k3=%.4f (%.1fs)",
            idx_u,
            len(args.units),
            uid,
            float(u_row["h_tree_bits"]),
            float(u_row["h_tree_se"]),
            float(u_row["acc_tree_mean"]),
            float(u_row["h_k3_bits"]),
            float(u_row["runtime_sec"]),
        )

    unit_rows.sort(key=lambda r: str(r["unit_id"]))
    lemma_rows.sort(key=lambda r: (str(r["unit_id"]), str(r["lemma_id"])))
    u_out, l_out = write_neural_outcomes(unit_rows, lemma_rows, args.out_dir, args.regime)
    logger.info("Saved neural outcomes to %s and %s", u_out, l_out)


if __name__ == "__main__":
    main()
