r"""Base-rooted Chu-Liu-Edmonds spanning arborescence (`H_tree`) and dynamic `k=1, 3` PCFP topology.

Implements Study 3 Track 2 §2.3:
1. Exact recursive Chu-Liu-Edmonds (1965/1967) minimum-cost directed spanning arborescence `T_r`
   rooted at each candidate base `r in C_distinct`, selecting `r* = argmin_r cost(T_r)` strictly
   on the inner validation cross-entropy matrix `M_hat[s, t]` (`subtrain -> dev`), NEVER touching
   held-out `test` data.
2. Fast plug-in edit-pattern conditional entropy pre-screening for huge paradigms (`|C_distinct| > 40`),
   retaining top-12 candidate parents per target cell + top-root stars + symmetric `k=1, 3` evaluation
   edges so 60-cell paradigms run in `<60s`.
3. Syncretism-aware test lemma aggregation (`shape_id` / surface equivalence partition `s_l`):
   - Edges between syncretic cells within `s_l` contribute `0` bits.
   - Each non-root distinct form class `sigma in s_l \ {[r*]}` is scored once via its minimal-depth
     entry vertex `v* in sigma` in `T_{r*}` (tie-broken by `M_hat[pa[v*], v*]`, then cell name),
     normalized by `|s_l| - 1`.
4. Dynamic `k=1` and `k=3` PCFP with `tau = 0.1` inverse-dev-entropy Product-of-Experts weights and
   exact Hölder upper partition normalization over `Sigma*` (`sum_{y in Sigma*} P_PoE(y) <= 1.0`).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Mapping, Sequence

from .escape import DirichletEscapeScorer, _LN2

PRESCREEN_CELL_THRESHOLD: int = 40
PRESCREEN_TOP_PARENTS: int = 12
POE_TAU: float = 0.1
K3_SUBSETS_PER_LEMMA: int = 5
K1_MAX_PAIRS_LARGE_PARADIGM: int = 120


@dataclass(frozen=True)
class ArborescenceResult:
    """Directed spanning arborescence `T_r` rooted at `root`."""

    root: str
    parent_of: dict[str, str]
    depth_of: dict[str, int]
    total_cost: float


@dataclass(frozen=True)
class LemmaTopologyScores:
    """Per-lemma held-out predictability outcomes across `H_tree`, `k=1`, and `k=3`."""

    n_distinct_forms: int
    h_tree: float
    rel_tree: float
    acc_tree: float
    h_k1: float
    acc_k1: float
    h_k3: float
    acc_k3: float


# ---------------------------------------------------------------------------
# Exact Recursive Chu-Liu-Edmonds Minimum Directed Spanning Arborescence
# ---------------------------------------------------------------------------


def _solve_cle_recursive(
    nodes: list[str],
    edges: dict[tuple[str, str], float],
    root: str,
) -> dict[str, str] | None:
    """Return `{child: parent}` spanning arborescence of minimum weight rooted at `root`, or `None`."""
    non_root = [v for v in nodes if v != root]
    if not non_root:
        return {}

    # 1. Select minimum-cost incoming edge `(u -> v)` for every `v != root`
    best_in: dict[str, tuple[float, str]] = {}
    for (u, v), w in edges.items():
        if v == root or u == v:
            continue
        cand = (round(w, 12), u)
        prev = best_in.get(v)
        if prev is None or cand < prev:
            best_in[v] = cand

    # If any non-root vertex has no incoming edge, no spanning arborescence exists
    if len(best_in) < len(non_root):
        return None

    chosen_parent: dict[str, str] = {v: best_in[v][1] for v in non_root}

    # 2. Detect any directed cycle in `chosen_parent`
    # state: 0 = unvisited, 1 = on current chain, 2 = verified acyclic to root
    visit_state: dict[str, int] = {root: 2}
    cycle_nodes: list[str] | None = None

    for start_v in sorted(non_root):
        if visit_state.get(start_v, 0) != 0:
            continue
        chain: list[str] = []
        pos_in_chain: dict[str, int] = {}
        curr = start_v
        while visit_state.get(curr, 0) == 0:
            visit_state[curr] = 1
            pos_in_chain[curr] = len(chain)
            chain.append(curr)
            curr = chosen_parent[curr]

        if visit_state.get(curr, 0) == 1:
            cycle_nodes = chain[pos_in_chain[curr] :]
            break
        for node in chain:
            visit_state[node] = 2

    if cycle_nodes is None:
        return chosen_parent

    # 3. Contract `cycle_nodes` into supernode `c_super`
    cycle_set = set(cycle_nodes)
    c_super = f"__cycle_{min(cycle_nodes)}_{len(nodes)}__"
    while c_super in nodes:
        c_super += "_"

    contracted_nodes = [v for v in nodes if v not in cycle_set] + [c_super]
    contracted_edges: dict[tuple[str, str], float] = {}
    # Track original `(u, v)` corresponding to contracted incoming/outgoing edges
    enter_orig: dict[str, tuple[str, str]] = {}
    exit_orig: dict[str, tuple[str, str]] = {}

    for (u, v), w in edges.items():
        if u == v or v == root:
            continue
        u_in = u in cycle_set
        v_in = v in cycle_set
        if not u_in and not v_in:
            contracted_edges[(u, v)] = w
        elif not u_in and v_in:
            # Adjusted incoming weight: w(u -> v) - w(chosen_parent[v] -> v)
            in_cycle_w = edges[(chosen_parent[v], v)]
            adj_w = w - in_cycle_w
            key = (u, c_super)
            prev_w = contracted_edges.get(key)
            if (
                prev_w is None
                or (round(adj_w, 12), u, v) < (round(prev_w, 12), *enter_orig[u])
            ):
                contracted_edges[key] = adj_w
                enter_orig[u] = (u, v)
        elif u_in and not v_in:
            key = (c_super, v)
            prev_w = contracted_edges.get(key)
            if (
                prev_w is None
                or (round(w, 12), u, v) < (round(prev_w, 12), *exit_orig[v])
            ):
                contracted_edges[key] = w
                exit_orig[v] = (u, v)

    sub_parents = _solve_cle_recursive(contracted_nodes, contracted_edges, root)
    if sub_parents is None:
        return None

    # 4. Expand supernode `c_super`
    parent_of_super = sub_parents[c_super]
    orig_u_enter, orig_v_enter = enter_orig[parent_of_super]

    expanded: dict[str, str] = {}
    for child, par in sub_parents.items():
        if child == c_super:
            continue
        if par == c_super:
            expanded[child] = exit_orig[child][0]
        else:
            expanded[child] = par

    for c_node in cycle_nodes:
        if c_node == orig_v_enter:
            expanded[c_node] = orig_u_enter
        else:
            expanded[c_node] = chosen_parent[c_node]

    return expanded


def chu_liu_edmonds(
    cells: Sequence[str],
    edge_weights: Mapping[tuple[str, str], float],
    root: str,
) -> ArborescenceResult:
    """Compute exact minimum-cost directed spanning arborescence rooted at `root`."""
    unique_cells = list(dict.fromkeys(cells))
    if root not in unique_cells:
        raise ValueError(f"Root cell {root!r} is not in cells {unique_cells!r}")
    if len(unique_cells) == 1:
        return ArborescenceResult(root=root, parent_of={}, depth_of={root: 0}, total_cost=0.0)

    clean_edges = {
        (u, v): float(w)
        for (u, v), w in edge_weights.items()
        if u in unique_cells and v in unique_cells and u != v and v != root
    }
    parent_of = _solve_cle_recursive(unique_cells, clean_edges, root)
    if parent_of is None:
        raise ValueError(f"Graph is not fully reachable from root {root!r}")

    # Compute exact tree depths from `root` and verify valid arborescence
    depth_of: dict[str, int] = {root: 0}

    def _get_depth(node: str) -> int:
        if node in depth_of:
            return depth_of[node]
        d = _get_depth(parent_of[node]) + 1
        depth_of[node] = d
        return d

    for c in unique_cells:
        _get_depth(c)

    total_cost = sum(clean_edges[(parent_of[v], v)] for v in unique_cells if v != root)
    return ArborescenceResult(
        root=root,
        parent_of=parent_of,
        depth_of=depth_of,
        total_cost=total_cost,
    )


def select_optimal_arborescence(
    cells: Sequence[str],
    dev_matrix: Mapping[tuple[str, str], float],
) -> ArborescenceResult:
    """Select `r* = argmin_r sum_{(u -> v) in T_r} M_hat[u, v]` strictly from dev matrix `M_hat`."""
    unique_cells = sorted(dict.fromkeys(cells))
    if len(unique_cells) == 1:
        r0 = unique_cells[0]
        return ArborescenceResult(root=r0, parent_of={}, depth_of={r0: 0}, total_cost=0.0)

    candidates: list[ArborescenceResult] = []
    for r in unique_cells:
        try:
            res = chu_liu_edmonds(unique_cells, dev_matrix, root=r)
            candidates.append(res)
        except ValueError:
            continue

    if not candidates:
        raise ValueError("No candidate root in `cells` can reach all paradigm cells in `dev_matrix`.")

    return min(candidates, key=lambda ar: (round(ar.total_cost, 12), ar.root))


# ---------------------------------------------------------------------------
# Fast Subtrain Pre-Screening for Large Paradigms (`|C_distinct| > 40`)
# ---------------------------------------------------------------------------


def _fast_single_change_sig(
    x: tuple[str, ...], y: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """O(|x|) maximal shared prefix/suffix stripping (Albright & Hayes 2002 §3.1) for pre-screening."""
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
    subtrain_paradigms: Sequence[Mapping[str, tuple[str, ...]]],
    top_k_parents: int = PRESCREEN_TOP_PARENTS,
    seed_key: str = "",
    cell_threshold: int = PRESCREEN_CELL_THRESHOLD,
    top_roots: int = 4,
    k_pool_size: int = 14,
) -> tuple[set[tuple[str, str]], list[str]]:
    """Fast plug-in conditional edit-pattern entropy pre-screener on `subtrain` ONLY.

    Returns `(candidate_directed_edges, k_eval_pool_cells)`:
    - Retains top `top_k_parents` lowest-entropy candidate parents `(s -> t)` for every target `t`.
    - Retains full outgoing star `(r -> t)` from the top-`top_roots` lowest-mean-outgoing-entropy roots `r`
      so Chu-Liu-Edmonds is guaranteed connected spanning arborescences.
    - Selects a deterministic stratified pool of `k_pool_size` cells (`k_eval_pool_cells`) and retains all
      symmetric pairs among them so `k=1` and `k=3` PoE PCFP are evaluated with exact dev weights.
    """
    sorted_cells = sorted(dict.fromkeys(cells))
    n_cells = len(sorted_cells)
    if n_cells <= cell_threshold or not subtrain_paradigms:
        all_edges = {(u, v) for u in sorted_cells for v in sorted_cells if u != v}
        return all_edges, sorted_cells

    n_items = len(subtrain_paradigms)
    inv_n = 1.0 / n_items

    # Pre-extract forms per cell
    forms_by_cell: dict[str, list[tuple[str, ...]]] = {
        c: [para[c] for para in subtrain_paradigms] for c in sorted_cells
    }
    # Conditioning cue for fast plug-in entropy: final segment of `w_s` (captures suffix classes)
    end_cue_by_cell: dict[str, list[str]] = {
        c: [(f[-1] if f else "#") for f in forms_by_cell[c]] for c in sorted_cells
    }

    cond_ent: dict[tuple[str, str], float] = {}
    out_sum: dict[str, float] = defaultdict(float)

    for s in sorted_cells:
        s_forms = forms_by_cell[s]
        s_cues = end_cue_by_cell[s]
        cue_totals: Counter[str] = Counter(s_cues)
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
                    p = cnt / tot_c
                    ent_c -= p * math.log2(p)
                h_val += (tot_c * inv_n) * ent_c

            cond_ent[(s, t)] = h_val
            out_sum[s] += h_val

    retained: set[tuple[str, str]] = set()

    # 1. Top-K parents per target cell `t`
    for t in sorted_cells:
        ranked_parents = sorted(
            (s for s in sorted_cells if s != t),
            key=lambda s: (round(cond_ent[(s, t)], 9), s),
        )
        for s in ranked_parents[:top_k_parents]:
            retained.add((s, t))

    # 2. Full outgoing stars from top candidate base cells
    best_roots = sorted(sorted_cells, key=lambda s: (round(out_sum[s], 9), s))[: max(1, top_roots)]
    for r in best_roots:
        for t in sorted_cells:
            if r != t:
                retained.add((r, t))

    # 3. Deterministic uniform cell subsample with full symmetric pairs for unbiased k=1, 3 PCFP
    seed_int = int.from_bytes(
        hashlib.sha256(f"{seed_key}|prescreen_k_pool".encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(seed_int)
    pool_size = min(max(4, k_pool_size), n_cells)
    k_pool = sorted(rng.sample(sorted_cells, pool_size))
    for u in k_pool:
        for v in k_pool:
            if u != v:
                retained.add((u, v))

    return retained, k_pool


# ---------------------------------------------------------------------------
# Held-Out Test Lemma Evaluation (`H_tree`, `k=1`, `k=3` PoE)
# ---------------------------------------------------------------------------


def _derive_k3_subsets(
    rep_cells: list[str],
    k_eff: int,
    seed_prefix: str,
) -> list[tuple[str, ...]]:
    """Return all subsets of size `k_eff` if `comb <= 5`, else `J = 5` deterministic subsets."""
    n_c = len(rep_cells)
    if k_eff <= 0 or n_c <= k_eff:
        return []
    total_comb = math.comb(n_c, k_eff)
    if total_comb <= K3_SUBSETS_PER_LEMMA:
        return list(combinations(rep_cells, k_eff))

    seed_int = int.from_bytes(
        hashlib.sha256(f"{seed_prefix}|k3".encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(seed_int)
    chosen: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    while len(chosen) < K3_SUBSETS_PER_LEMMA:
        sub = tuple(sorted(rng.sample(rep_cells, k_eff)))
        if sub not in seen:
            seen.add(sub)
            chosen.append(sub)
    return chosen


def score_poe_target(
    source_cells: Sequence[str],
    target_cell: str,
    lemma_forms: Mapping[str, tuple[str, ...]],
    get_scorer: Callable[[str, str], DirichletEscapeScorer],
    dev_matrix: Mapping[tuple[str, str], float],
    cand_cache: dict[
        tuple[str, str],
        tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float], tuple[str, ...], float],
    ],
    tau: float = POE_TAU,
) -> tuple[float, float]:
    """Compute exact Hölder-normalized Product-of-Experts `(nll_bits, acc)` for `S_3 -> target_cell`.

    Weights:
        `w_c = (M_hat[c, t] + tau)^(-1) / sum_{c' in S_3} (M_hat[c', t] + tau)^(-1)`
    Normalized over `Sigma*` via exact union of MGL candidates `C_cand` + Hölder tail bound:
        `Z_upper = sum_{y in C_cand} prod_c P_c(y)^{w_c} + prod_c max(0, 1 - sum_{y in C_cand} P_c(y))^{w_c}`
    guaranteeing `sum_{y in Sigma*} P_PoE(y) <= 1.0` strictly.
    """
    y_gold = lemma_forms[target_cell]
    if len(source_cells) == 1:
        c0 = source_cells[0]
        scorer0 = get_scorer(c0, target_cell)
        ds = scorer0.score_pair(lemma_forms[c0], y_gold)
        return ds.nll_bits, ds.exact_match

    raw_weights = [1.0 / (max(0.0, dev_matrix[(c, target_cell)]) + tau) for c in source_cells]
    w_sum = sum(raw_weights)
    weights = [rw / w_sum for rw in raw_weights]

    scorers: list[DirichletEscapeScorer] = []
    summaries: list[
        tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float], tuple[str, ...], float]
    ] = []
    c_cand_set: set[tuple[str, ...]] = set()

    for c in source_cells:
        sc_model = get_scorer(c, target_cell)
        scorers.append(sc_model)
        key = (c, target_cell)
        summ = cand_cache.get(key)
        if summ is None:
            summ = sc_model.candidate_summary(lemma_forms[c])
            cand_cache[key] = summ
        summaries.append(summ)
        c_cand_set.update(summ[1].keys())

    c_cand_list = sorted(c_cand_set)

    # Evaluate unnormalized log PoE score `sum_r w_r ln P_r(y)` on every `y in C_cand`
    # AND accumulate `sum_{y in C_cand} P_r(y)` for each expert `r` to form Hölder tail bound!
    cand_mass_per_expert = [0.0] * len(source_cells)
    log_unnorm_cands: dict[tuple[str, ...], float] = {}

    for y_c in c_cand_list:
        lp_weighted = 0.0
        for r_idx, (c, w_r, sc_model, summ) in enumerate(
            zip(source_cells, weights, scorers, summaries)
        ):
            lp_r = sc_model.log_prob_with_cache(
                lemma_forms[c], y_c, summ[0], summ[1], summ[3]
            )
            lp_weighted += w_r * lp_r
            cand_mass_per_expert[r_idx] += math.exp(lp_r)
        log_unnorm_cands[y_c] = lp_weighted

    # Hölder upper bound on remaining string space `Sigma* \ C_cand`
    log_tail = 0.0
    tail_zero = False
    for w_r, m_r in zip(weights, cand_mass_per_expert):
        rem_r = max(0.0, 1.0 - m_r)
        if rem_r <= 0.0:
            if w_r > 0.0:
                tail_zero = True
                break
        else:
            log_tail += w_r * math.log(rem_r)

    # Compute `ln Z_upper = logsumexp({log_unnorm_cands[y] for y in C_cand} U {log_tail})`
    log_terms: list[float] = list(log_unnorm_cands.values())
    if not tail_zero:
        log_terms.append(log_tail)

    if log_terms:
        m_max = max(log_terms)
        log_z_upper = m_max + math.log(sum(math.exp(lt - m_max) for lt in log_terms))
    else:
        log_z_upper = 0.0

    # Top-1 prediction under PoE
    if log_unnorm_cands:
        top1_y = min(log_unnorm_cands.keys(), key=lambda y: (-log_unnorm_cands[y], " ".join(y)))
    else:
        top1_y = scorers[0].edit_model.greedy_decode(lemma_forms[source_cells[0]])
    acc = 1.0 if top1_y == y_gold else 0.0

    # Gold target `y_gold` log probability
    if y_gold in log_unnorm_cands:
        log_unnorm_gold = log_unnorm_cands[y_gold]
    else:
        log_unnorm_gold = 0.0
        for c, w_r, sc_model, summ in zip(source_cells, weights, scorers, summaries):
            lp_r = sc_model.log_prob_with_cache(
                lemma_forms[c], y_gold, summ[0], summ[1], summ[3]
            )
            log_unnorm_gold += w_r * lp_r

    log_p_poe = log_unnorm_gold - log_z_upper
    nll_bits = max(0.0, -log_p_poe / _LN2)
    return nll_bits, acc


def evaluate_lemma_topology(
    lemma_forms: Mapping[str, tuple[str, ...]],
    cells: Sequence[str],
    tree: ArborescenceResult,
    dev_matrix: Mapping[tuple[str, str], float],
    get_scorer: Callable[[str, str], DirichletEscapeScorer],
    seed_prefix: str,
    k_eval_pool_cells: Sequence[str] | None = None,
) -> LemmaTopologyScores:
    """Evaluate `H_tree`, `h_k1`, and `h_k3` on a single held-out test lemma `l`."""
    # 1. Partition `cells` by identical `variant_idx == 0` surface form (`s_l`)
    by_form: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for c in cells:
        by_form[lemma_forms[c]].append(c)

    classes = [sorted(cls_cells) for cls_cells in by_form.values()]
    n_distinct = len(classes)

    if n_distinct <= 1:
        return LemmaTopologyScores(
            n_distinct_forms=n_distinct,
            h_tree=0.0,
            rel_tree=1.0,
            acc_tree=1.0,
            h_k1=0.0,
            acc_k1=1.0,
            h_k3=0.0,
            acc_k3=1.0,
        )

    # 2. `H_tree(l)`: for each non-root equivalence class `sigma in s_l \ {[r*]}`, select
    # `v* = min(sigma, key=lambda v: (tree.depth_of[v], dev_matrix[(pa[v], v)], v))`
    # and average over the `|s_l| - 1` non-root distinct form classes!
    h_tree_sum = 0.0
    rel_tree_sum = 0.0
    acc_tree_sum = 0.0
    n_non_root = 0

    for sigma in classes:
        if tree.root in sigma:
            continue
        v_star = min(
            sigma,
            key=lambda v: (
                tree.depth_of[v],
                round(dev_matrix[(tree.parent_of[v], v)], 12),
                v,
            ),
        )
        u_parent = tree.parent_of[v_star]
        assert u_parent not in sigma, "Minimal-depth vertex in non-root class cannot have parent in class"
        scorer_uv = get_scorer(u_parent, v_star)
        ds = scorer_uv.score_pair(lemma_forms[u_parent], lemma_forms[v_star])
        h_tree_sum += ds.nll_bits
        rel_tree_sum += ds.reliability
        acc_tree_sum += ds.exact_match
        n_non_root += 1

    assert n_non_root == n_distinct - 1
    h_tree = h_tree_sum / n_non_root
    rel_tree = rel_tree_sum / n_non_root
    acc_tree = acc_tree_sum / n_non_root

    # 3. Dynamic `k = 1` and `k = 3` PCFP over canonical class representatives `rep(sigma) = min(sigma)`
    if k_eval_pool_cells is not None:
        pool_set = set(k_eval_pool_cells)
        pool_by_form: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for c in cells:
            if c in pool_set:
                pool_by_form[lemma_forms[c]].append(c)
        rep_cells = sorted(min(c_list) for c_list in pool_by_form.values())
    else:
        rep_cells = sorted(min(sigma) for sigma in classes)

    # Fall back to at least 2 distinct representatives if pre-screened pool happened to collapse
    if len(rep_cells) < 2:
        all_reps = sorted(min(sigma) for sigma in classes)
        rep_cells = [
            c
            for c in all_reps
            if any((c, c2) in dev_matrix and (c2, c) in dev_matrix for c2 in all_reps if c2 != c)
        ]
        if len(rep_cells) < 2:
            return LemmaTopologyScores(
                n_distinct_forms=n_distinct,
                h_tree=h_tree,
                rel_tree=rel_tree,
                acc_tree=acc_tree,
                h_k1=h_tree,
                acc_k1=acc_tree,
                h_k3=h_tree,
                acc_k3=acc_tree,
            )

    cand_cache: dict[
        tuple[str, str],
        tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float], tuple[str, ...], float],
    ] = {}

    # `k = 1` pairwise evaluation across distinct representative cells
    ordered_pairs = [(cs, ct) for cs in rep_cells for ct in rep_cells if cs != ct and (cs, ct) in dev_matrix]
    if not ordered_pairs:
        return LemmaTopologyScores(
            n_distinct_forms=n_distinct,
            h_tree=h_tree,
            rel_tree=rel_tree,
            acc_tree=acc_tree,
            h_k1=h_tree,
            acc_k1=acc_tree,
            h_k3=h_tree,
            acc_k3=acc_tree,
        )
    if len(ordered_pairs) > K1_MAX_PAIRS_LARGE_PARADIGM:
        s_k1 = int.from_bytes(
            hashlib.sha256(f"{seed_prefix}|k1".encode("utf-8")).digest()[:8], "big"
        )
        ordered_pairs = sorted(random.Random(s_k1).sample(ordered_pairs, K1_MAX_PAIRS_LARGE_PARADIGM))

    k1_bits_sum = 0.0
    k1_acc_sum = 0.0
    for cs, ct in ordered_pairs:
        sc_model = get_scorer(cs, ct)
        key = (cs, ct)
        summ = cand_cache.get(key)
        if summ is None:
            summ = sc_model.candidate_summary(lemma_forms[cs])
            cand_cache[key] = summ
        lp = sc_model.log_prob_with_cache(
            lemma_forms[cs], lemma_forms[ct], summ[0], summ[1], summ[3]
        )
        k1_bits_sum += max(0.0, -lp / _LN2)
        k1_acc_sum += 1.0 if summ[2] == lemma_forms[ct] else 0.0

    h_k1 = k1_bits_sum / len(ordered_pairs)
    acc_k1 = k1_acc_sum / len(ordered_pairs)

    # `k = 3` Product-of-Experts evaluation
    k_eff = min(3, len(rep_cells) - 1)
    subsets_s3 = _derive_k3_subsets(rep_cells, k_eff, seed_prefix)
    k3_bits_sum = 0.0
    k3_acc_sum = 0.0
    k3_count = 0

    for s3 in subsets_s3:
        s3_set = set(s3)
        for ct in rep_cells:
            if ct in s3_set:
                continue
            valid_sources = [cs for cs in s3 if (cs, ct) in dev_matrix]
            if not valid_sources:
                continue
            nb, ac = score_poe_target(
                valid_sources,
                ct,
                lemma_forms,
                get_scorer,
                dev_matrix,
                cand_cache,
                tau=POE_TAU,
            )
            k3_bits_sum += nb
            k3_acc_sum += ac
            k3_count += 1

    h_k3 = (k3_bits_sum / k3_count) if k3_count > 0 else h_k1
    acc_k3 = (k3_acc_sum / k3_count) if k3_count > 0 else acc_k1

    return LemmaTopologyScores(
        n_distinct_forms=n_distinct,
        h_tree=h_tree,
        rel_tree=rel_tree,
        acc_tree=acc_tree,
        h_k1=h_k1,
        acc_k1=acc_k1,
        h_k3=h_k3,
        acc_k3=acc_k3,
    )
