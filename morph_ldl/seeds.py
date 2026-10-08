"""Named, order-independent seed derivation (docs/CONTRACT.md §8)."""

from __future__ import annotations

import hashlib

PURPOSES = {
    "inventory", "split", "fold", "selector_init", "random_policy", "tie",
    "semantic", "bootstrap", "pool_size", "auxiliary", "exposure", "selector_semantic",
    "core_split", "random_draw",
}


def derive(master: int, purpose: str, *keys: object) -> int:
    """Derive a reproducible seed in [0, 2**31-1) from a master seed, a purpose and keys."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown seed purpose {purpose!r}; add it to seeds.PURPOSES")
    text = "|".join([str(int(master)), purpose, *map(str, keys)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def tie_key(tie_seed: int, lemma_id: str) -> str:
    """Deterministic tie-break key for ranking lemmas with equal scores."""
    return hashlib.sha256(f"{tie_seed}:{lemma_id}".encode("utf-8")).hexdigest()
