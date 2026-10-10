"""Python port of the identifier-keyed simulated semantics of julia/src/LDLRunner.jl.

s(lemma, cell) = L(lemma) + sum_f V(f) [+ V(cell)] + N(lemma, cell), each a Gaussian vector generated
from SplitMix64 seeded by the first 8 bytes of sha256("|".join(parts)), Box-Muller.
Used for parity tests and for inspecting vectors without Julia (LDL_PROTOCOL.md §3.2).
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List

import numpy as np

MASK = (1 << 64) - 1


def seed_of(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class SplitMix64:
    def __init__(self, seed: int):
        self.s = seed & MASK

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def uniform01(self) -> float:
        return ((self.next() >> 11) + 1) * 2.0 ** -53


def gaussian_vector(seed: int, n: int, sd: float) -> np.ndarray:
    r = SplitMix64(seed)
    v = np.empty(n)
    i = 0
    while i < n:
        u1, u2 = r.uniform01(), r.uniform01()
        rad, th = math.sqrt(-2.0 * math.log(u1)), 2.0 * math.pi * u2
        v[i] = rad * math.cos(th)
        i += 1
        if i < n:
            v[i] = rad * math.sin(th)
            i += 1
    return v * sd


def cell_features(cell: str) -> List[str]:
    return cell.split(";")


def lexeme_vec(c: Dict, lemma_id: str) -> np.ndarray:
    return gaussian_vector(seed_of(c["semantic_seed"], "lexeme", lemma_id), c["sem_dim"], c["sem_sd_lexeme"])


def feature_vec(c: Dict, feature: str) -> np.ndarray:
    return gaussian_vector(seed_of(c["semantic_seed"], "feature", feature), c["sem_dim"], c["sem_sd_inflection"])


def noise_vec(c: Dict, lemma_id: str, cell: str) -> np.ndarray:
    return gaussian_vector(seed_of(c["semantic_seed"], "noise", lemma_id, cell), c["sem_dim"], c["sem_sd_noise"])


def cell_vec(c: Dict, cell: str) -> np.ndarray:
    return gaussian_vector(seed_of(c["semantic_seed"], "cell", cell), c["sem_dim"], c["sem_sd_cell"])


def form_semantics(c: Dict, lemma_id: str, cell: str) -> np.ndarray:
    s = lexeme_vec(c, lemma_id) + sum(feature_vec(c, f) for f in cell_features(cell))
    if c.get("sem_sd_cell", 0) > 0:
        s = s + cell_vec(c, cell)
    if c["sem_sd_noise"] > 0:
        s = s + noise_vec(c, lemma_id, cell)
    return s
