"""Eligible inventory, grouped outer folds and inner roles (docs/CONTRACT.md §4).

The main agent owns this module. Every downstream component reads the manifests it
writes; nothing else may construct or change splits.

Procedure for one unit
----------------------
1. Eligible lemmas: variant-0, non-missing rows for the source cell and every panel cell.
2. Inventory (fixed across repetitions): eligible groups are shuffled with the
   ``inventory`` seed and taken whole, in that order, while the running lemma count stays
   within ``inventory_size`` (a group that would overshoot is skipped).
3. Outer folds (per repetition): inventory groups are shuffled with the ``split`` seed and
   dealt greedily to the fold with the fewest lemmas so far (ties -> lowest fold index).
4. Inner roles (per repetition and fold): the remaining groups are shuffled with the
   ``fold`` seed and filled, in order, into ``dev`` (dev_size), ``seed`` (seed_size) and
   ``pool`` (pool_cap) using the same take-while-it-fits rule; the rest is
   ``pool_overflow``. ``role_rank`` records the order inside each role, so a smaller
   pool cap is exactly the first lemmas of the larger pool (nested sensitivity pools).

Requested sizes are never silently reduced: if any role cannot be filled exactly,
``SplitError`` is raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from morph_ldl import seeds as seedlib
from morph_ldl.schemas import SPLIT_COLUMNS

SPLIT_COLUMNS_EXT = SPLIT_COLUMNS + ["role_rank", "fold_seed"]


class SplitError(RuntimeError):
    """A requested inventory, fold or role size cannot be met."""


@dataclass(frozen=True)
class SplitSpec:
    inventory_size: int
    n_folds: int
    dev_size: int
    seed_size: int
    pool_cap: int
    min_pool_for_budget: int  # max(budgets) - seed_size

    @classmethod
    def from_cfg(cls, cfg: dict) -> "SplitSpec":
        cv = cfg["cv"]
        max_budget = max(cfg["selection"]["budgets"])
        return cls(
            inventory_size=int(cv["inventory_size"]),
            n_folds=int(cv["n_folds"]),
            dev_size=int(cv["dev_size"]),
            seed_size=int(cv["seed_size"]),
            pool_cap=int(max([cv["pool_cap"], *cv.get("pool_cap_sensitivity", [])])),
            min_pool_for_budget=max_budget - int(cv["seed_size"]),
        )


def eligible_lemmas(forms: pd.DataFrame, source_cell: str, panel_cells: Sequence[str],
                    exclude_ids: Iterable[str] = (), exclude_multiword: bool = False) -> pd.DataFrame:
    """Return one row per eligible lemma: lemma_id, group_id.

    A lemma is eligible when its variant-0 row for the source cell and for every panel
    cell exists and is not missing, it is not in ``exclude_ids`` (declared derived
    paradigms), and, with ``exclude_multiword``, none of those task forms contains a
    word space (segment ``_``).
    """
    needed = [source_cell, *panel_cells]
    if len(set(needed)) != len(needed):
        raise SplitError(f"source/panel cells overlap or repeat: {needed}")
    ok = forms[(forms["variant_idx"] == 0) & (~forms["is_missing"].astype(bool))
               & (forms["cell_norm"].isin(needed)) & (forms["form"].astype(str) != "")]
    have = ok.groupby("lemma_id")["cell_norm"].nunique()
    keep = set(have[have == len(needed)].index) - set(exclude_ids)
    if exclude_multiword:
        multi = ok.loc[ok["segments"].astype(str).str.split(" ").map(lambda s: "_" in s), "lemma_id"]
        keep -= set(multi)
    lem = (forms.loc[forms["lemma_id"].isin(keep), ["lemma_id", "group_id"]]
           .drop_duplicates().sort_values("lemma_id").reset_index(drop=True))
    dup = lem["lemma_id"].duplicated()
    if dup.any():
        raise SplitError(f"lemmas with several group_ids: {lem.loc[dup, 'lemma_id'].tolist()[:5]}")
    return lem


def _group_table(lemmas: pd.DataFrame) -> pd.DataFrame:
    g = lemmas.groupby("group_id")["lemma_id"].agg(list).reset_index()
    g["size"] = g["lemma_id"].map(len)
    return g.sort_values("group_id").reset_index(drop=True)


def _shuffled(groups: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    return groups.iloc[order].reset_index(drop=True)


def _take(groups: pd.DataFrame, start_mask: np.ndarray, target: int) -> np.ndarray:
    """Take whole groups in table order while they fit into ``target`` lemmas.

    Returns a boolean mask over ``groups`` of the groups taken. ``start_mask`` marks
    groups still available.
    """
    taken = np.zeros(len(groups), dtype=bool)
    count = 0
    for i, size in enumerate(groups["size"].to_numpy()):
        if count == target:
            break
        if start_mask[i] and count + size <= target:
            taken[i] = True
            count += size
    if count != target:
        raise SplitError(f"could only fill {count} of {target} lemmas with whole groups")
    return taken


def build_inventory(lemmas: pd.DataFrame, spec: SplitSpec, master_seed: int, unit_id: str) -> Tuple[pd.DataFrame, int]:
    inv_seed = seedlib.derive(master_seed, "inventory", unit_id)
    groups = _shuffled(_group_table(lemmas), inv_seed)
    if groups["size"].sum() < spec.inventory_size:
        raise SplitError(
            f"{unit_id}: only {groups['size'].sum()} eligible lemmas; inventory_size="
            f"{spec.inventory_size} requested")
    taken = _take(groups, np.ones(len(groups), dtype=bool), spec.inventory_size)
    inv = groups[taken].reset_index(drop=True)
    return inv, inv_seed


def check_capacity(spec: SplitSpec, unit_id: str) -> None:
    """Fail early when the declared sizes cannot work for any split."""
    per_fold_test = spec.inventory_size // spec.n_folds
    rest = spec.inventory_size - per_fold_test - 1  # largest fold may be one bigger
    need = spec.dev_size + spec.seed_size + spec.pool_cap
    if spec.pool_cap < spec.min_pool_for_budget:
        raise SplitError(
            f"{unit_id}: pool_cap={spec.pool_cap} < max budget - seed = {spec.min_pool_for_budget}")
    if need > rest:
        raise SplitError(
            f"{unit_id}: dev+seed+pool = {need} lemmas needed but only about {rest} remain "
            f"outside a test fold (inventory {spec.inventory_size}, {spec.n_folds} folds)")


def build_split_manifest(lemmas: pd.DataFrame, cfg: dict, unit_id: str, repetition: int) -> pd.DataFrame:
    """Build the manifest for one unit and repetition (all folds)."""
    spec = SplitSpec.from_cfg(cfg)
    master = int(cfg["experiment"]["master_seed"])
    check_capacity(spec, unit_id)
    inv, inv_seed = build_inventory(lemmas, spec, master, unit_id)

    split_seed = seedlib.derive(master, "split", unit_id, repetition)
    inv = _shuffled(inv.sort_values("group_id").reset_index(drop=True), split_seed)
    fold_of = np.empty(len(inv), dtype=int)
    load = [0] * spec.n_folds
    for i, size in enumerate(inv["size"].to_numpy()):
        k = int(np.argmin(load))
        fold_of[i] = k
        load[k] += size
    inv["fold"] = fold_of

    rows: List[dict] = []
    for k in range(spec.n_folds):
        fold_seed = seedlib.derive(master, "fold", unit_id, repetition, k)
        test = inv[inv["fold"] == k]
        rest = _shuffled(inv[inv["fold"] != k].sort_values("group_id").reset_index(drop=True), fold_seed)
        avail = np.ones(len(rest), dtype=bool)
        roles = np.full(len(rest), "pool_overflow", dtype=object)
        for role, target in (("dev", spec.dev_size), ("seed", spec.seed_size), ("pool", spec.pool_cap)):
            try:
                taken = _take(rest, avail, target)
            except SplitError as err:
                raise SplitError(f"{unit_id} rep {repetition} fold {k} role {role}: {err}") from None
            roles[taken] = role
            avail &= ~taken
        rank = {r: 0 for r in ("dev", "seed", "pool", "pool_overflow")}
        for (_, g), role in zip(rest.iterrows(), roles):
            for lid in g["lemma_id"]:
                rows.append(dict(lemma_id=lid, group_id=g["group_id"], role=role,
                                 role_rank=rank[role], outer_fold=k, fold_seed=fold_seed))
                rank[role] += 1
        for t_rank, (_, g) in enumerate(test.sort_values("group_id").iterrows()):
            for lid in g["lemma_id"]:
                rows.append(dict(lemma_id=lid, group_id=g["group_id"], role="test",
                                 role_rank=-1, outer_fold=k, fold_seed=fold_seed))
    man = pd.DataFrame(rows)
    man["unit_id"] = unit_id
    man["repetition"] = repetition
    man["inventory_seed"] = inv_seed
    man["split_seed"] = split_seed
    man = man[SPLIT_COLUMNS_EXT].sort_values(["outer_fold", "role", "role_rank", "lemma_id"]).reset_index(drop=True)
    validate_manifest(man, spec)
    return man


def validate_manifest(man: pd.DataFrame, spec: SplitSpec | None = None) -> None:
    """Structural checks: one role per lemma per fold; groups never straddle roles;
    each inventory lemma is test in exactly one fold; requested sizes met exactly."""
    for (rep, k), fm in man.groupby(["repetition", "outer_fold"]):
        if fm["lemma_id"].duplicated().any():
            raise SplitError(f"rep {rep} fold {k}: lemma listed twice")
        roles_per_group = fm.groupby("group_id")["role"].nunique()
        if (roles_per_group > 1).any():
            raise SplitError(f"rep {rep} fold {k}: group straddles roles: "
                             f"{roles_per_group[roles_per_group > 1].index[:5].tolist()}")
        if spec is not None:
            counts = fm["role"].value_counts()
            for role, target in (("dev", spec.dev_size), ("seed", spec.seed_size), ("pool", spec.pool_cap)):
                if counts.get(role, 0) != target:
                    raise SplitError(f"rep {rep} fold {k}: {role} has {counts.get(role, 0)} != {target}")
    for rep, rm in man.groupby("repetition"):
        tests = rm[rm["role"] == "test"]
        if tests["lemma_id"].duplicated().any():
            raise SplitError(f"rep {rep}: lemma is test in more than one fold")
        if set(tests["lemma_id"]) != set(rm["lemma_id"]):
            raise SplitError(f"rep {rep}: some inventory lemmas are never tested")


def roles(man: pd.DataFrame, repetition: int, fold: int, pool_cap: int | None = None) -> Dict[str, List[str]]:
    """Lemma ids per role for one fold. ``pool_cap`` takes the first ``pool_cap`` pool
    lemmas (nested sensitivity pools)."""
    fm = man[(man["repetition"] == repetition) & (man["outer_fold"] == fold)]
    out: Dict[str, List[str]] = {}
    for role in ("test", "core", "dev", "seed", "pool", "pool_overflow"):
        sub = fm[fm["role"] == role].sort_values(["role_rank", "lemma_id"])
        out[role] = sub["lemma_id"].tolist()
    if pool_cap is not None:
        if pool_cap > len(out["pool"]):
            raise SplitError(f"pool_cap {pool_cap} > pool size {len(out['pool'])}")
        out["pool"] = out["pool"][:pool_cap]
    return out


def write_manifest(man: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "split_manifest.csv"
    man.to_csv(path, index=False)
    return path


def load_manifest(path: Path) -> pd.DataFrame:
    man = pd.read_csv(path, dtype={"lemma_id": str, "group_id": str, "role": str})
    if (man["role"] == "core").any():
        validate_pcfp_manifest(man)
    else:
        validate_manifest(man)
    return man


# ----------------------------------------------------------------------------- auxiliary set

AUX_ROLES = ("tune_background", "tune_heldout", "copy_anchor", "aux_unused")
AUX_ROLES_PCFP = ("tune_core", "tune_extra", "aux_unused")


def build_auxiliary_manifest(lemmas: pd.DataFrame, inventory_ids: Iterable[str], sizes: Dict[str, int],
                             master_seed: int, unit_id: str) -> pd.DataFrame:
    """Eligible lemmas OUTSIDE the inventory, by whole group, for uses that must never
    touch any outer-test lemma: LDL setting choice (``tune_background``, ``tune_heldout``)
    and selector copy anchors (``copy_anchor``, source forms only). Because no inventory
    lemma is ever auxiliary, these lemmas are never test, dev, seed or pool in any fold
    or repetition, and no pool candidate is ever a copy anchor.
    """
    inv = set(inventory_ids)
    inv_groups = set(lemmas.loc[lemmas["lemma_id"].isin(inv), "group_id"])
    rest = lemmas[~lemmas["group_id"].isin(inv_groups)]
    if set(rest["lemma_id"]) & inv:
        raise SplitError("inventory group bookkeeping error")
    seed = seedlib.derive(master_seed, "auxiliary", unit_id)
    groups = _shuffled(_group_table(rest), seed)
    avail = np.ones(len(groups), dtype=bool)
    role = np.full(len(groups), "aux_unused", dtype=object)
    for r in sizes:                       # roles filled in the order given (dict order)
        try:
            taken = _take(groups, avail, int(sizes[r]))
        except SplitError as err:
            raise SplitError(f"{unit_id} auxiliary {r}: {err}") from None
        role[taken] = r
        avail &= ~taken
    rows, rank = [], {r: 0 for r in [*sizes, "aux_unused"]}
    for (_, g), r in zip(groups.iterrows(), role):
        for lid in g["lemma_id"]:
            rows.append(dict(unit_id=unit_id, lemma_id=lid, group_id=g["group_id"], aux_role=r,
                             aux_rank=rank[r], auxiliary_seed=seed))
            rank[r] += 1
    return pd.DataFrame(rows)


def aux_roles(aux: pd.DataFrame) -> Dict[str, List[str]]:
    names = list(dict.fromkeys([*AUX_ROLES, *AUX_ROLES_PCFP, *aux["aux_role"].unique()]))
    return {r: aux[aux["aux_role"] == r].sort_values("aux_rank")["lemma_id"].tolist() for r in names}


# ----------------------------------------------------------------------------- PCFP design

@dataclass(frozen=True)
class PcfpSplitSpec:
    inventory_size: int
    n_folds: int
    core_size: int
    dev_size: int
    seed_size: int
    pool_cap: int
    min_pool_for_budget: int

    @classmethod
    def from_cfg(cls, cfg: dict) -> "PcfpSplitSpec":
        cv = cfg["cv"]
        return cls(inventory_size=int(cv["inventory_size"]), n_folds=int(cv["n_folds"]),
                   core_size=int(cv["core_size"]), dev_size=int(cv.get("dev_size", 0)),
                   seed_size=int(cv["seed_size"]),
                   pool_cap=int(max([cv["pool_cap"], *cv.get("pool_cap_sensitivity", [])])),
                   min_pool_for_budget=max(cfg["selection"]["budgets"]) - int(cv["seed_size"]))


def check_pcfp_capacity(spec: PcfpSplitSpec, unit_id: str, caps: Sequence[int] = ()) -> None:
    for cap in [spec.pool_cap, *caps]:
        if cap < spec.min_pool_for_budget:
            raise SplitError(f"{unit_id}: pool cap {cap} < max budget - seed = {spec.min_pool_for_budget}")
    if spec.n_folds * spec.core_size > spec.inventory_size:
        raise SplitError(f"{unit_id}: {spec.n_folds} disjoint core sets of {spec.core_size} exceed the "
                         f"inventory of {spec.inventory_size}")
    need = spec.core_size + spec.dev_size + spec.seed_size + spec.pool_cap
    if need > spec.inventory_size:
        raise SplitError(f"{unit_id}: core+dev+seed+pool = {need} > inventory {spec.inventory_size}")


def build_pcfp_manifest(lemmas: pd.DataFrame, cfg: dict, unit_id: str, repetition: int) -> pd.DataFrame:
    """PCFP split manifest for one unit and repetition (all folds).

    1. Inventory as in ``build_inventory`` (``inventory`` seed, fixed across repetitions).
    2. Core sets: inventory groups shuffled with the ``split`` seed; K disjoint core sets of
       exactly ``core_size`` lemmas are taken in that order (whole groups).
    3. Per fold k: every other inventory lemma (including other folds' core verbs) is
       shuffled with the ``fold`` seed and filled into dev (if any), seed and pool; the
       rest is pool_overflow. ``role_rank`` orders lemmas within a role (nested pools).
    """
    spec = PcfpSplitSpec.from_cfg(cfg)
    master = int(cfg["experiment"]["master_seed"])
    check_pcfp_capacity(spec, unit_id, [int(c) for c in cfg["cv"].get("pool_cap_sensitivity", [])])
    inv_spec = SplitSpec(spec.inventory_size, spec.n_folds, spec.dev_size, spec.seed_size, spec.pool_cap,
                         spec.min_pool_for_budget)
    inv, inv_seed = build_inventory(lemmas, inv_spec, master, unit_id)
    split_seed = seedlib.derive(master, "split", unit_id, repetition)
    inv = _shuffled(inv.sort_values("group_id").reset_index(drop=True), split_seed)
    avail = np.ones(len(inv), dtype=bool)
    core_of = np.full(len(inv), -1, dtype=int)
    for k in range(spec.n_folds):
        try:
            taken = _take(inv, avail, spec.core_size)
        except SplitError as err:
            raise SplitError(f"{unit_id} rep {repetition} core set {k}: {err}") from None
        core_of[taken] = k
        avail &= ~taken
    inv["core_fold"] = core_of
    rows: List[dict] = []
    for k in range(spec.n_folds):
        fold_seed = seedlib.derive(master, "fold", unit_id, repetition, k)
        core = inv[inv["core_fold"] == k].sort_values("group_id")
        rest = _shuffled(inv[inv["core_fold"] != k].sort_values("group_id").reset_index(drop=True), fold_seed)
        av = np.ones(len(rest), dtype=bool)
        roles_ = np.full(len(rest), "pool_overflow", dtype=object)
        for role, target in (("dev", spec.dev_size), ("seed", spec.seed_size), ("pool", spec.pool_cap)):
            if target == 0:
                continue
            try:
                taken = _take(rest, av, target)
            except SplitError as err:
                raise SplitError(f"{unit_id} rep {repetition} fold {k} role {role}: {err}") from None
            roles_[taken] = role
            av &= ~taken
        rank = {r: 0 for r in ("core", "dev", "seed", "pool", "pool_overflow")}
        for _, g in core.iterrows():
            for lid in g["lemma_id"]:
                rows.append(dict(lemma_id=lid, group_id=g["group_id"], role="core", role_rank=rank["core"],
                                 outer_fold=k, fold_seed=fold_seed))
                rank["core"] += 1
        for (_, g), role in zip(rest.iterrows(), roles_):
            for lid in g["lemma_id"]:
                rows.append(dict(lemma_id=lid, group_id=g["group_id"], role=role, role_rank=rank[role],
                                 outer_fold=k, fold_seed=fold_seed))
                rank[role] += 1
    man = pd.DataFrame(rows)
    man["unit_id"] = unit_id
    man["repetition"] = repetition
    man["inventory_seed"] = inv_seed
    man["split_seed"] = split_seed
    man = man[SPLIT_COLUMNS_EXT].sort_values(["outer_fold", "role", "role_rank", "lemma_id"]).reset_index(drop=True)
    validate_pcfp_manifest(man, spec)
    return man


def validate_pcfp_manifest(man: pd.DataFrame, spec: "PcfpSplitSpec | None" = None) -> None:
    """One role per lemma per fold; groups never straddle roles; exact sizes; core sets
    disjoint across the folds of a repetition; every fold lists the same inventory."""
    for (rep, k), fm in man.groupby(["repetition", "outer_fold"]):
        if fm["lemma_id"].duplicated().any():
            raise SplitError(f"rep {rep} fold {k}: lemma listed twice")
        rpg = fm.groupby("group_id")["role"].nunique()
        if (rpg > 1).any():
            raise SplitError(f"rep {rep} fold {k}: group straddles roles: {rpg[rpg > 1].index[:5].tolist()}")
        if spec is not None:
            counts = fm["role"].value_counts()
            for role, target in (("core", spec.core_size), ("dev", spec.dev_size), ("seed", spec.seed_size),
                                 ("pool", spec.pool_cap)):
                if counts.get(role, 0) != target:
                    raise SplitError(f"rep {rep} fold {k}: {role} has {counts.get(role, 0)} != {target}")
    for rep, rm in man.groupby("repetition"):
        core = rm[rm["role"] == "core"]
        if core["lemma_id"].duplicated().any() or core.groupby("group_id")["outer_fold"].nunique().gt(1).any():
            raise SplitError(f"rep {rep}: core sets overlap across folds")
        inv_sets = {k: frozenset(fm["lemma_id"]) for k, fm in rm.groupby("outer_fold")}
        if len(set(inv_sets.values())) != 1:
            raise SplitError(f"rep {rep}: folds list different inventories")
