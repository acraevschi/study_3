"""Multi-Locus Vowel-Tier Minimal Generalization Learner (MGL) with C-speed Python `int` bitmasks.

Implements Study 3 Track 2 §2.1 steps 2–3:
- Fixed global hyperparameters across all languages (zero per-language tuning):
    * Mikheev lower confidence bound `alpha = 0.75`:
        `p* = (x + 0.5) / (n + 1.0)`
        `pi_L = max(0.0, p* - t_{0.75, df=n-1} * sqrt(p*(1 - p*) / n))` for `n >= 2`
        (matching Albright 2002 / Albright & Hayes 2003 benchmarks:
         `5/5 -> 0.825`, `4/4 -> 0.785`, `9/9 -> 0.899`, `65/65 -> 0.985`, `19/20 -> 0.889`;
         `n == 1` uses calibrated normal quantile `z_{alpha}` so `1/1 -> 0.458 < 2/2 -> 0.570`).
    * Residue impugnment upper confidence bound `alpha_upper = 0.95` (Albright & Hayes 2002 §3.7):
        For every strictly narrower sub-rule `C' subset C` (`0 < s_sub < s_sup` and
        `s_sub & ~s_sup == 0`), computes residue `(s_res = s_sup & ~s_sub, h_res = h_sup & ~s_sub)`
        and downgrades `Confidence(R) <- min(Confidence(R), pi_U(x_res, n_res, 0.95))`.
- Python `int` bitmask representation (`scope_mask`, `hit_mask`) over all `N_train` lemmas:
  `n = scope_mask.bit_count()`, `x = hit_mask.bit_count()`, subset tests in nanoseconds, and
  exact deduplication of `(scope_mask, hit_mask)` equivalence classes before impugnment.
- Parallel generalization over:
  1. Multi-locus local context `X C'_feat C'_shared ___ D'_shared D'_feat Y` around each locus, and
  2. Source-only Vowel-Tier contexts (`v_left_mask`, `v_right_mask`, `req_vowel_mask`,
     `allowed_vowel_mask`).
"""

from __future__ import annotations

import functools
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import mpmath as mp

from .align import AlignmentResult, LocusContext, LocusEdit, MultiLocusPattern, extract_pattern
from .vowel_tier import VowelClassifier

DEFAULT_ALPHA_LOWER: float = 0.75
DEFAULT_ALPHA_UPPER: float = 0.95
MAX_PAIRWISE_SEEDS_PER_CHANGE: int = 64
MAX_LOCAL_LATTICE_PER_CHANGE: int = 64
MAX_VTIER_LATTICE_PER_CHANGE: int = 64


# ---------------------------------------------------------------------------
# Exact Student-t Quantiles & Mikheev Confidence Limits
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4096)
def _normal_quantile(prob: float) -> float:
    """Exact standard normal quantile `Phi^{-1}(prob)` for `0 < prob < 1`."""
    return float(mp.sqrt(2.0) * mp.erfinv(2.0 * prob - 1.0))


@functools.lru_cache(maxsize=8192)
def student_t_quantile(prob: float, df: int) -> float:
    """Exact one-sided upper quantile `t_{prob, df}` of Student's t distribution (`P(T <= t) = prob`)."""
    if df <= 0:
        raise ValueError(f"degrees of freedom must be >= 1, got {df}")
    if prob == 0.5:
        return 0.0
    if prob < 0.5:
        return -student_t_quantile(1.0 - prob, df)

    # Closed forms for df = 1 (Cauchy) and df = 2
    if df == 1:
        return math.tan(math.pi * (prob - 0.5))
    if df == 2:
        return (2.0 * prob - 1.0) / math.sqrt(2.0 * prob * (1.0 - prob))

    # Cornish-Fisher asymptotic expansion is accurate to < 1e-11 for df > 500
    if df > 500:
        z = _normal_quantile(prob)
        z2 = z * z
        z3 = z2 * z
        z5 = z3 * z2
        g1 = (z3 + z) / 4.0
        g2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / 96.0
        g3 = (3.0 * z5 * z2 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / 384.0
        return z + g1 / df + g2 / (df * df) + g3 / (df * df * df)

    # Exact root-finding via regularized incomplete beta for 3 <= df <= 500
    def _cdf(t_val: mp.mpf) -> mp.mpf:
        if t_val <= 0:
            x_b = df / (df + t_val * t_val)
            return mp.mpf("0.5") * mp.betainc(df / 2.0, 0.5, 0, x_b, regularized=True)
        x_b = df / (df + t_val * t_val)
        return 1 - mp.mpf("0.5") * mp.betainc(df / 2.0, 0.5, 0, x_b, regularized=True)

    z0 = _normal_quantile(prob)
    guess = z0 * math.sqrt(df / max(1.0, df - 1.5))
    root = mp.findroot(lambda t: _cdf(t) - prob, guess)
    return float(root)


@functools.lru_cache(maxsize=65536)
def mikheev_lower(hits: int, scope: int, alpha: float = DEFAULT_ALPHA_LOWER) -> float:
    """Mikheev (1997) / Albright & Hayes (2002, 2003) lower confidence limit `pi_L`.

    Uses `p* = (hits + 0.5) / (scope + 1.0)` and one-sided `t_{alpha, df=scope-1}` for `scope >= 2`
    (`z_{alpha}` normal approximation for `scope == 1`).
    """
    if scope <= 0 or hits < 0:
        return 0.0
    x = min(hits, scope)
    p_star = (x + 0.5) / (scope + 1.0)
    sd = math.sqrt(p_star * (1.0 - p_star) / scope)
    crit = student_t_quantile(alpha, scope - 1) if scope >= 2 else _normal_quantile(alpha)
    return max(0.0, min(1.0, p_star - crit * sd))


@functools.lru_cache(maxsize=65536)
def mikheev_upper(hits: int, scope: int, alpha_upper: float = DEFAULT_ALPHA_UPPER) -> float:
    """Upper confidence limit `pi_U` used in Albright & Hayes (2002 §3.7) residue impugnment."""
    if scope <= 0:
        return 1.0
    x = max(0, min(hits, scope))
    p_star = (x + 0.5) / (scope + 1.0)
    sd = math.sqrt(p_star * (1.0 - p_star) / scope)
    crit = (
        student_t_quantile(alpha_upper, scope - 1)
        if scope >= 2
        else _normal_quantile(alpha_upper)
    )
    return max(0.0, min(1.0, p_star + crit * sd))


# ---------------------------------------------------------------------------
# Structural Environment Representations (Local + Source Vowel Tier)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalLocusEnv:
    """Albright & Hayes (2002 eq. 5) structural context `X C'_feat C' ___ D' D'_feat Y` at one locus.

    - `left_shared`: contiguous literal segments immediately to the left of `___` (excluding `#`).
    - `left_feat`: optional shared feature bundle immediately left of `left_shared` (`None` if none).
    - `left_anchored`: `True` iff `#` must immediately precede `(left_feat, left_shared)` with no
      intervening segments (`X` absent); `False` when free variable `X` precedes.
    - `right_shared`, `right_feat`, `right_anchored`: symmetric right-hand specification.
    """

    left_shared: tuple[str, ...] = ()
    left_feat: frozenset[str] | None = None
    left_anchored: bool = False
    right_shared: tuple[str, ...] = ()
    right_feat: frozenset[str] | None = None
    right_anchored: bool = False

    @classmethod
    def from_word_context(cls, ctx: LocusContext) -> LocalLocusEnv:
        # ctx.left_local starts with "#", ctx.right_local ends with "#"
        l_segs = tuple(s for s in ctx.left_local if s != "#")
        r_segs = tuple(s for s in ctx.right_local if s != "#")
        return cls(
            left_shared=l_segs,
            left_feat=None,
            left_anchored=True,
            right_shared=r_segs,
            right_feat=None,
            right_anchored=True,
        )

    @classmethod
    def unconstrained(cls, anchor_left: bool = False, anchor_right: bool = False) -> LocalLocusEnv:
        return cls(
            left_shared=(),
            left_feat=None,
            left_anchored=anchor_left,
            right_shared=(),
            right_feat=None,
            right_anchored=anchor_right,
        )


def generalize_local_env(
    e1: LocalLocusEnv,
    e2: LocalLocusEnv,
    classifier: VowelClassifier,
) -> LocalLocusEnv:
    """Minimal generalization of two local locus contexts following Albright & Hayes (2002 eq. 5)."""
    if e1 == e2:
        return e1
    gen_cache = classifier._gen_env_cache
    cached = gen_cache.get((e1, e2))
    if cached is not None:
        return cached  # type: ignore[return-value]

    # --- Left side (working outward right-to-left from `___`) ---
    s1 = e1.left_shared
    s2 = e2.left_shared
    len1 = len(s1)
    len2 = len(s2)
    max_common_l = min(len1, len2)
    k_l = 0
    while k_l < max_common_l and s1[len1 - 1 - k_l] == s2[len2 - 1 - k_l]:
        k_l += 1
    c_shared = s1[len1 - k_l :] if k_l > 0 else ()

    # Inspect the element immediately to the left of `c_shared` in e1 and e2
    rem1 = len1 - k_l
    rem2 = len2 - k_l

    def _left_next_term(
        env: LocalLocusEnv, rem: int
    ) -> tuple[frozenset[str] | None, bool, bool]:
        """Return `(feat_bundle_or_None, has_more_left_of_term, is_hash_boundary_here)`."""
        if rem > 0:
            seg = env.left_shared[rem - 1]
            has_more = (rem - 1 > 0) or (env.left_feat is not None) or (not env.left_anchored)
            return classifier.segment_features(seg), has_more, False
        if env.left_feat is not None:
            has_more = not env.left_anchored
            return env.left_feat, has_more, False
        # Reached left edge of specified material: either `#` (`left_anchored=True`) or `X` (`False`)
        return None, not env.left_anchored, env.left_anchored

    f1_l, more1_l, hash1_l = _left_next_term(e1, rem1)
    f2_l, more2_l, hash2_l = _left_next_term(e2, rem2)

    if hash1_l and hash2_l:
        c_feat: frozenset[str] | None = None
        l_anchored = True
    elif f1_l is not None and f2_l is not None:
        inter_l = f1_l & f2_l
        if inter_l:
            c_feat = inter_l
            l_anchored = (not more1_l) and (not more2_l)
        else:
            c_feat = None
            l_anchored = False
    else:
        c_feat = None
        l_anchored = False

    # --- Right side (working outward left-to-right from `___`) ---
    d1 = e1.right_shared
    d2 = e2.right_shared
    rlen1 = len(d1)
    rlen2 = len(d2)
    max_common_r = min(rlen1, rlen2)
    k_r = 0
    while k_r < max_common_r and d1[k_r] == d2[k_r]:
        k_r += 1
    d_shared = d1[:k_r] if k_r > 0 else ()

    rrem1 = rlen1 - k_r
    rrem2 = rlen2 - k_r

    def _right_next_term(
        env: LocalLocusEnv, rrem: int, offset: int
    ) -> tuple[frozenset[str] | None, bool, bool]:
        if rrem > 0:
            seg = env.right_shared[offset]
            has_more = (rrem - 1 > 0) or (env.right_feat is not None) or (not env.right_anchored)
            return classifier.segment_features(seg), has_more, False
        if env.right_feat is not None:
            has_more = not env.right_anchored
            return env.right_feat, has_more, False
        return None, not env.right_anchored, env.right_anchored

    f1_r, more1_r, hash1_r = _right_next_term(e1, rrem1, k_r)
    f2_r, more2_r, hash2_r = _right_next_term(e2, rrem2, k_r)

    if hash1_r and hash2_r:
        d_feat: frozenset[str] | None = None
        r_anchored = True
    elif f1_r is not None and f2_r is not None:
        inter_r = f1_r & f2_r
        if inter_r:
            d_feat = inter_r
            r_anchored = (not more1_r) and (not more2_r)
        else:
            d_feat = None
            r_anchored = False
    else:
        d_feat = None
        r_anchored = False

    res_env = LocalLocusEnv(
        left_shared=c_shared,
        left_feat=c_feat,
        left_anchored=l_anchored,
        right_shared=d_shared,
        right_feat=d_feat,
        right_anchored=r_anchored,
    )
    if len(gen_cache) < 250_000:
        gen_cache[(e1, e2)] = res_env
        gen_cache[(e2, e1)] = res_env
    return res_env


@dataclass(frozen=True)
class VowelTierEnv:
    """Source-only autosegmental Vowel-Tier environment over indexed vowel alphabet.

    - `v_left_mask`: bitmask of allowed closest source vowel to the left of the primary affix locus
      (including `no_vowel_bit` if vowelless stems are permitted); `None` if unconstrained.
    - `v_right_mask`: bitmask of allowed closest source vowel to the right of locus 0 (for prefix
      harmony / right-to-left umlaut); `None` if unconstrained.
    - `req_vowel_mask`: bitmask of vowels that MUST appear somewhere in `w_s` (`0` = none required).
    - `allowed_vowel_mask`: bitmask of vowels permitted in `w_s` (`vset & ~allowed == 0`);
      `None` if unconstrained.
    """

    v_left_mask: int | None = None
    v_right_mask: int | None = None
    req_vowel_mask: int = 0
    allowed_vowel_mask: int | None = None


@dataclass(frozen=True)
class MGLRule:
    """Compiled Multi-Locus Vowel-Tier MGL rule with training bitmasks and confidence."""

    pattern: MultiLocusPattern
    local_envs: tuple[LocalLocusEnv, ...]
    vtier_env: VowelTierEnv | None
    scope_mask: int
    hit_mask: int
    scope: int
    hits: int
    raw_reliability: float
    lower_confidence: float
    confidence: float


# ---------------------------------------------------------------------------
# Fast Rule Application & Matching Engine
# ---------------------------------------------------------------------------


def _local_env_matches_at(
    env: LocalLocusEnv,
    x: tuple[str, ...],
    feat_seq: tuple[frozenset[str], ...],
    p_j: int,
    q_j: int,
) -> bool:
    """Check whether `env` matches source word `x` (`len = m`) around span `[p_j, q_j)`."""
    m = len(x)

    # Left side check before index `p_j`
    l_sh = env.left_shared
    k_l = len(l_sh)
    if p_j < k_l:
        return False
    if k_l > 0 and x[p_j - k_l : p_j] != l_sh:
        return False
    pos_l = p_j - k_l
    if env.left_feat is not None:
        if pos_l <= 0:
            return False
        if not env.left_feat.issubset(feat_seq[pos_l - 1]):
            return False
        pos_l -= 1
    if env.left_anchored and pos_l != 0:
        return False

    # Right side check starting at index `q_j`
    r_sh = env.right_shared
    k_r = len(r_sh)
    if m - q_j < k_r:
        return False
    if k_r > 0 and x[q_j : q_j + k_r] != r_sh:
        return False
    pos_r = q_j + k_r
    if env.right_feat is not None:
        if pos_r >= m:
            return False
        if not env.right_feat.issubset(feat_seq[pos_r]):
            return False
        pos_r += 1
    if env.right_anchored and pos_r != m:
        return False

    return True


def _find_rule_outputs(
    pattern: MultiLocusPattern,
    local_envs: tuple[LocalLocusEnv, ...],
    vtier_env: VowelTierEnv | None,
    x: tuple[str, ...],
    feat_seq: tuple[frozenset[str], ...],
    vc_seq: tuple[str, ...],
    v_bit_seq: tuple[int, ...],
    word_vset_mask: int,
    no_vowel_bit: int,
) -> list[tuple[str, ...]]:
    """Return all target segment tuples `y` derivable from `x` by `(pattern, local_envs, vtier_env)`."""
    # 1. Word-level Vowel-Tier checks (O(1) bitwise operations)
    if vtier_env is not None:
        if vtier_env.req_vowel_mask and (
            (word_vset_mask & vtier_env.req_vowel_mask) != vtier_env.req_vowel_mask
        ):
            return []
        if vtier_env.allowed_vowel_mask is not None and (
            (word_vset_mask & ~vtier_env.allowed_vowel_mask) != 0
        ):
            return []

    m = len(x)
    loci = pattern.loci
    n_loci = len(loci)

    # Fast path for identity rule (`_ -> _` anchored at right boundary `p_0 = q_0 = m`)
    if pattern.is_identity:
        if not _local_env_matches_at(local_envs[0], x, feat_seq, m, m):
            return []
        if vtier_env is not None and vtier_env.v_left_mask is not None:
            last_v_bit = no_vowel_bit
            for idx in range(m - 1, -1, -1):
                if v_bit_seq[idx] != 0:
                    last_v_bit = v_bit_seq[idx]
                    break
            if not (last_v_bit & vtier_env.v_left_mask):
                return []
        return [x]

    # Pre-collect valid `[p_j, q_j)` spans in `x` for each locus `j`
    candidate_spans: list[list[tuple[int, int]]] = []
    for j, (loc, env) in enumerate(zip(loci, local_envs)):
        sub = loc.source_sub
        slen = len(sub)
        if slen > m:
            return []

        # Determine valid start range for `p_j`
        if loc.anchor_left or (env.left_anchored and len(env.left_shared) == 0 and env.left_feat is None):
            starts = [0]
        elif loc.anchor_right or (
            env.right_anchored and len(env.right_shared) == 0 and env.right_feat is None
        ):
            starts = [m - slen]
        elif env.left_anchored:
            exact_p = len(env.left_shared) + (1 if env.left_feat is not None else 0)
            starts = [exact_p] if 0 <= exact_p <= m - slen else []
        elif env.right_anchored:
            exact_q = m - len(env.right_shared) - (1 if env.right_feat is not None else 0)
            exact_p = exact_q - slen
            starts = [exact_p] if 0 <= exact_p <= m - slen else []
        else:
            min_p = len(env.left_shared) + (1 if env.left_feat is not None else 0)
            max_p = m - slen - len(env.right_shared) - (1 if env.right_feat is not None else 0)
            if slen == 0 and not loc.anchor_left and not loc.anchor_right:
                # Internal pure insertion (`_um_`): must be strictly internal (`0 < p < m`)
                min_p = max(1, min_p)
                max_p = min(m - 1, max_p)
            starts = list(range(min_p, max_p + 1)) if min_p <= max_p else []

        valid_j: list[tuple[int, int]] = []
        for p_j in starts:
            q_j = p_j + slen
            if slen > 0 and x[p_j:q_j] != sub:
                continue
            if _local_env_matches_at(env, x, feat_seq, p_j, q_j):
                valid_j.append((p_j, q_j))
        if not valid_j:
            return []
        candidate_spans.append(valid_j)

    # Helper to check positional Vowel-Tier constraints (`v_left_mask` at final locus,
    # `v_right_mask` at initial locus)
    def _vtier_positional_ok(chosen: list[tuple[int, int]]) -> bool:
        if vtier_env is None:
            return True
        if vtier_env.v_left_mask is not None:
            p_last = chosen[-1][0]
            last_v_bit = no_vowel_bit
            for idx in range(p_last - 1, -1, -1):
                if v_bit_seq[idx] != 0:
                    last_v_bit = v_bit_seq[idx]
                    break
            if not (last_v_bit & vtier_env.v_left_mask):
                return False
        if vtier_env.v_right_mask is not None:
            q_first = chosen[0][1]
            first_v_bit = no_vowel_bit
            for idx in range(q_first, m):
                if v_bit_seq[idx] != 0:
                    first_v_bit = v_bit_seq[idx]
                    break
            if not (first_v_bit & vtier_env.v_right_mask):
                return False
        return True

    # Fast single-locus path (`n_loci == 1`)
    if n_loci == 1:
        b0 = loci[0].target_sub
        outs: list[tuple[str, ...]] = []
        for p0, q0 in candidate_spans[0]:
            if vtier_env is not None and not _vtier_positional_ok([(p0, q0)]):
                continue
            y_cand = (*x[:p0], *b0, *x[q0:])
            if y_cand not in outs:
                outs.append(y_cand)
        return outs

    # General multi-locus backtracking (`n_loci >= 2`, non-overlapping monotonic spans)
    outs_multi: list[tuple[str, ...]] = []
    chosen_spans: list[tuple[int, int]] = []

    def _dfs(j_idx: int, min_start: int) -> None:
        if j_idx == n_loci:
            if not _vtier_positional_ok(chosen_spans):
                return
            parts: list[str] = []
            cursor = 0
            for (pj, qj), loc in zip(chosen_spans, loci):
                parts.extend(x[cursor:pj])
                parts.extend(loc.target_sub)
                cursor = qj
            parts.extend(x[cursor:])
            y_tup = tuple(parts)
            if y_tup not in outs_multi:
                outs_multi.append(y_tup)
            return

        for pj, qj in candidate_spans[j_idx]:
            if pj < min_start:
                continue
            # If two consecutive pure insertions occur (`e.g. ge_t`), require non-empty stem between
            # them when `pj == min_start` and `len(source_sub) == 0` for both
            if (
                j_idx > 0
                and pj == min_start
                and len(loci[j_idx - 1].source_sub) == 0
                and len(loci[j_idx].source_sub) == 0
            ):
                continue
            chosen_spans.append((pj, qj))
            _dfs(j_idx + 1, qj)
            chosen_spans.pop()

    _dfs(0, 0)
    return outs_multi


# ---------------------------------------------------------------------------
# Pre-indexed Training Set for Nanosecond Bitmask Evaluation
# ---------------------------------------------------------------------------


class _IndexedTrainingSet:
    """Bitmask-indexed representation of `N_train` `(w_s, w_t)` pairs."""

    def __init__(
        self,
        pairs: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
        classifier: VowelClassifier,
    ) -> None:
        self.n_items = len(pairs)
        self.all_mask = (1 << self.n_items) - 1
        self.classifier = classifier

        self.x_list: list[tuple[str, ...]] = [p[0] for p in pairs]
        self.y_list: list[tuple[str, ...]] = [p[1] for p in pairs]
        self.alignments: list[AlignmentResult] = [
            extract_pattern(x, y, classifier) for x, y in pairs
        ]

        # Build vowel index over all vowels attested in training `w_s` (strictly source-only!)
        train_vowels = sorted(
            {
                v
                for x in self.x_list
                for s in x
                if s != "#"
                for v in classifier.segment_vowels(s)
            }
        )
        self.vowel_to_bit: dict[str, int] = {v: (1 << idx) for idx, v in enumerate(train_vowels)}
        self.other_vowel_bit: int = 1 << len(train_vowels)
        self.no_vowel_bit: int = 1 << (len(train_vowels) + 1)
        self.all_vowel_bits_mask: int = (
            ((1 << len(train_vowels)) - 1) | self.other_vowel_bit | self.no_vowel_bit
        )

        self.feat_seqs: list[tuple[frozenset[str], ...]] = []
        self.vc_seqs: list[tuple[str, ...]] = []
        self.v_bit_seqs: list[tuple[int, ...]] = []
        self.word_vset_masks: list[int] = []

        # Segment presence mask (`seg -> bitmask of items whose x contains seg`)
        self.seg_present_mask: dict[str, int] = defaultdict(int)
        # Right-edge & left-edge positional indices for O(1) suffix/prefix bitmask intersections
        self.len_at_least_mask: list[int] = []
        self.len_exact_mask: dict[int, int] = defaultdict(int)
        self.right_seg_mask: dict[tuple[int, str], int] = defaultdict(int)
        self.left_seg_mask: dict[tuple[int, str], int] = defaultdict(int)
        self.last_vowel_bit_items: dict[int, int] = defaultdict(int)
        self.first_vowel_bit_items: dict[int, int] = defaultdict(int)
        self.has_vowel_bit_mask: dict[int, int] = defaultdict(int)

        max_len = max((len(x) for x in self.x_list), default=0)
        len_ge = [0] * (max_len + 2)

        for i, x in enumerate(self.x_list):
            ibit = 1 << i
            m = len(x)
            self.len_exact_mask[m] |= ibit
            for l_val in range(m + 1):
                len_ge[l_val] |= ibit

            f_seq = tuple(classifier.segment_features(s) for s in x)
            vc_seq = tuple(classifier.classify(s) for s in x)
            vb_list: list[int] = []
            vset_m = 0
            first_vb = self.no_vowel_bit
            last_vb = self.no_vowel_bit

            for pos, s in enumerate(x):
                self.seg_present_mask[s] |= ibit
                self.left_seg_mask[(pos, s)] |= ibit
                self.right_seg_mask[(m - 1 - pos, s)] |= ibit
                svs = classifier.segment_vowels(s)
                if svs:
                    seg_vb = 0
                    for v_sym in svs:
                        vb = self.vowel_to_bit.get(v_sym, self.other_vowel_bit)
                        vset_m |= vb
                        self.has_vowel_bit_mask[vb] |= ibit
                        seg_vb = vb
                    vb_list.append(seg_vb)
                    if first_vb == self.no_vowel_bit:
                        first_vb = seg_vb
                    last_vb = seg_vb
                else:
                    vb_list.append(0)

            self.feat_seqs.append(f_seq)
            self.vc_seqs.append(vc_seq)
            self.v_bit_seqs.append(tuple(vb_list))
            self.word_vset_masks.append(vset_m)
            self.last_vowel_bit_items[last_vb] |= ibit
            self.first_vowel_bit_items[first_vb] |= ibit

        self.len_at_least_mask = len_ge
        self._right_feat_cache: dict[tuple[int, frozenset[str]], int] = {}
        self._left_feat_cache: dict[tuple[int, frozenset[str]], int] = {}
        self._suffix_hit_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
        self._prefix_hit_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
        self._vleft_offset_cache: dict[int, dict[int, int]] = {0: self.last_vowel_bit_items}
        self.identity_hit_mask: int = 0
        for i, (x, y) in enumerate(zip(self.x_list, self.y_list)):
            if x == y:
                self.identity_hit_mask |= 1 << i

    def _vleft_items_at_right_offset(self, alen: int) -> dict[int, int]:
        cached = self._vleft_offset_cache.get(alen)
        if cached is not None:
            return cached
        by_vb: dict[int, int] = defaultdict(int)
        for idx_i, x_i in enumerate(self.x_list):
            m_i = len(x_i)
            if m_i < alen:
                continue
            lv_b = self.no_vowel_bit
            vb_seq_i = self.v_bit_seqs[idx_i]
            for pos_i in range(m_i - alen - 1, -1, -1):
                if vb_seq_i[pos_i] != 0:
                    lv_b = vb_seq_i[pos_i]
                    break
            by_vb[lv_b] |= 1 << idx_i
        self._vleft_offset_cache[alen] = by_vb
        return by_vb

    def _get_len_ge(self, min_len: int) -> int:
        if min_len <= 0:
            return self.all_mask
        if min_len >= len(self.len_at_least_mask):
            return 0
        return self.len_at_least_mask[min_len]

    def _right_feat_mask(self, r_offset: int, feat: frozenset[str]) -> int:
        key = (r_offset, feat)
        cached = self._right_feat_cache.get(key)
        if cached is not None:
            return cached
        mask = 0
        cand = self._get_len_ge(r_offset + 1)
        while cand:
            lsb = cand & -cand
            i = lsb.bit_length() - 1
            f_seq = self.feat_seqs[i]
            if feat.issubset(f_seq[len(f_seq) - 1 - r_offset]):
                mask |= lsb
            cand ^= lsb
        self._right_feat_cache[key] = mask
        return mask

    def _left_feat_mask(self, l_offset: int, feat: frozenset[str]) -> int:
        key = (l_offset, feat)
        cached = self._left_feat_cache.get(key)
        if cached is not None:
            return cached
        mask = 0
        cand = self._get_len_ge(l_offset + 1)
        while cand:
            lsb = cand & -cand
            i = lsb.bit_length() - 1
            if feat.issubset(self.feat_seqs[i][l_offset]):
                mask |= lsb
            cand ^= lsb
        self._left_feat_cache[key] = mask
        return mask

    def _suffix_hit_eligible_mask(self, a_sub: tuple[str, ...], b_sub: tuple[str, ...]) -> int:
        key = (a_sub, b_sub)
        cached = self._suffix_hit_cache.get(key)
        if cached is not None:
            return cached
        alen = len(a_sub)
        mask = self._get_len_ge(alen)
        for off, seg in enumerate(reversed(a_sub)):
            mask &= self.right_seg_mask.get((off, seg), 0)
        hit_m = 0
        cand = mask
        while cand:
            lsb = cand & -cand
            i = lsb.bit_length() - 1
            x = self.x_list[i]
            stem = x[: len(x) - alen] if alen > 0 else x
            if (*stem, *b_sub) == self.y_list[i]:
                hit_m |= lsb
            cand ^= lsb
        self._suffix_hit_cache[key] = hit_m
        return hit_m

    def _prefix_hit_eligible_mask(self, a_sub: tuple[str, ...], b_sub: tuple[str, ...]) -> int:
        key = (a_sub, b_sub)
        cached = self._prefix_hit_cache.get(key)
        if cached is not None:
            return cached
        alen = len(a_sub)
        mask = self._get_len_ge(alen)
        for off, seg in enumerate(a_sub):
            mask &= self.left_seg_mask.get((off, seg), 0)
        hit_m = 0
        cand = mask
        while cand:
            lsb = cand & -cand
            i = lsb.bit_length() - 1
            x = self.x_list[i]
            if (*b_sub, *x[alen:]) == self.y_list[i]:
                hit_m |= lsb
            cand ^= lsb
        self._prefix_hit_cache[key] = hit_m
        return hit_m

    def _word_vtier_mask(self, vtier_env: VowelTierEnv) -> int:
        mask = self.all_mask
        req = vtier_env.req_vowel_mask
        while req:
            vbit = req & -req
            mask &= self.has_vowel_bit_mask.get(vbit, 0)
            if not mask:
                return 0
            req ^= vbit
        if vtier_env.allowed_vowel_mask is not None:
            forbidden = (
                self.all_vowel_bits_mask
                & ~self.no_vowel_bit
                & ~vtier_env.allowed_vowel_mask
            )
            while forbidden:
                vbit = forbidden & -forbidden
                mask &= ~self.has_vowel_bit_mask.get(vbit, 0)
                if not mask:
                    return 0
                forbidden ^= vbit
        return mask

    def evaluate_rule_masks(
        self,
        pattern: MultiLocusPattern,
        local_envs: tuple[LocalLocusEnv, ...],
        vtier_env: VowelTierEnv | None,
    ) -> tuple[int, int]:
        """Compute exact `(scope_mask, hit_mask)` bitmasks across all `N_train` items."""
        # Fast O(1) pure-bitmask path for right-anchored single-locus suffix / identity rules!
        if len(pattern.loci) == 1:
            loc0 = pattern.loci[0]
            env0 = local_envs[0]
            is_right_edge = pattern.is_identity or (
                env0.right_anchored and len(env0.right_shared) == 0 and env0.right_feat is None
            )
            if is_right_edge:
                a_sub = loc0.source_sub
                alen = len(a_sub)
                k_l = len(env0.left_shared)
                need_feat = 1 if env0.left_feat is not None else 0
                min_len = alen + k_l + need_feat

                scope = (
                    self.len_exact_mask.get(min_len, 0)
                    if env0.left_anchored
                    else self._get_len_ge(min_len)
                )
                if not scope:
                    return 0, 0

                for idx_from_end, seg in enumerate(reversed(a_sub)):
                    scope &= self.right_seg_mask.get((idx_from_end, seg), 0)
                    if not scope:
                        return 0, 0

                for idx_c, seg in enumerate(reversed(env0.left_shared)):
                    scope &= self.right_seg_mask.get((alen + idx_c, seg), 0)
                    if not scope:
                        return 0, 0

                if env0.left_feat is not None:
                    scope &= self._right_feat_mask(alen + k_l, env0.left_feat)
                    if not scope:
                        return 0, 0

                if vtier_env is not None:
                    scope &= self._word_vtier_mask(vtier_env)
                    if not scope:
                        return 0, 0
                    if vtier_env.v_left_mask is not None and vtier_env.v_right_mask is None:
                        # O(1) bitmask OR for ANY `alen >= 0` (`___ a_sub #`)!
                        vleft_map = self._vleft_items_at_right_offset(alen)
                        allowed_lv = 0
                        rem_lv = vtier_env.v_left_mask
                        while rem_lv:
                            vb = rem_lv & -rem_lv
                            allowed_lv |= vleft_map.get(vb, 0)
                            rem_lv ^= vb
                        scope &= allowed_lv
                        if not scope:
                            return 0, 0
                    elif vtier_env.v_left_mask is not None or vtier_env.v_right_mask is not None:
                        # Stem-final vowel/segment change (`alen > 0`): verify positional vowel on surviving items
                        filtered = 0
                        cand = scope
                        while cand:
                            lsb = cand & -cand
                            i = lsb.bit_length() - 1
                            outs = _find_rule_outputs(
                                pattern,
                                local_envs,
                                vtier_env,
                                self.x_list[i],
                                self.feat_seqs[i],
                                self.vc_seqs[i],
                                self.v_bit_seqs[i],
                                self.word_vset_masks[i],
                                self.no_vowel_bit,
                            )
                            if outs:
                                filtered |= lsb
                            cand ^= lsb
                        scope = filtered
                        if not scope:
                            return 0, 0

                hit_elig = (
                    self.identity_hit_mask
                    if pattern.is_identity
                    else self._suffix_hit_eligible_mask(a_sub, loc0.target_sub)
                )
                return scope, (scope & hit_elig)

            # Fast O(1) pure-bitmask path for left-anchored single-locus prefix rules (`# ___`)!
            is_left_edge = (
                env0.left_anchored and len(env0.left_shared) == 0 and env0.left_feat is None
            )
            if is_left_edge and (vtier_env is None or vtier_env.v_left_mask is None):
                a_sub = loc0.source_sub
                alen = len(a_sub)
                k_r = len(env0.right_shared)
                need_feat = 1 if env0.right_feat is not None else 0
                min_len = alen + k_r + need_feat

                scope = (
                    self.len_exact_mask.get(min_len, 0)
                    if env0.right_anchored
                    else self._get_len_ge(min_len)
                )
                if not scope:
                    return 0, 0
                for idx_l, seg in enumerate(a_sub):
                    scope &= self.left_seg_mask.get((idx_l, seg), 0)
                    if not scope:
                        return 0, 0
                for idx_d, seg in enumerate(env0.right_shared):
                    scope &= self.left_seg_mask.get((alen + idx_d, seg), 0)
                    if not scope:
                        return 0, 0
                if env0.right_feat is not None:
                    scope &= self._left_feat_mask(alen + k_r, env0.right_feat)
                    if not scope:
                        return 0, 0
                if vtier_env is not None:
                    scope &= self._word_vtier_mask(vtier_env)
                    if not scope:
                        return 0, 0
                    if vtier_env.v_right_mask is not None and alen == 0:
                        allowed_rv = 0
                        rem_rv = vtier_env.v_right_mask
                        while rem_rv:
                            vb = rem_rv & -rem_rv
                            allowed_rv |= self.first_vowel_bit_items.get(vb, 0)
                            rem_rv ^= vb
                        scope &= allowed_rv
                        if not scope:
                            return 0, 0
                    elif vtier_env.v_right_mask is not None:
                        filtered = 0
                        cand = scope
                        while cand:
                            lsb = cand & -cand
                            i = lsb.bit_length() - 1
                            if _find_rule_outputs(
                                pattern,
                                local_envs,
                                vtier_env,
                                self.x_list[i],
                                self.feat_seqs[i],
                                self.vc_seqs[i],
                                self.v_bit_seqs[i],
                                self.word_vset_masks[i],
                                self.no_vowel_bit,
                            ):
                                filtered |= lsb
                            cand ^= lsb
                        scope = filtered
                        if not scope:
                            return 0, 0

                hit_elig = self._prefix_hit_eligible_mask(a_sub, loc0.target_sub)
                return scope, (scope & hit_elig)

        # General multi-locus / internal-locus path with segment-presence bitmask prefilter
        cand = self.all_mask
        if vtier_env is not None:
            cand &= self._word_vtier_mask(vtier_env)
            if not cand:
                return 0, 0

        min_req_len = 0
        for loc, env in zip(pattern.loci, local_envs):
            min_req_len += len(loc.source_sub)
            for s in loc.source_sub:
                cand &= self.seg_present_mask.get(s, 0)
            for s in env.left_shared:
                cand &= self.seg_present_mask.get(s, 0)
            for s in env.right_shared:
                cand &= self.seg_present_mask.get(s, 0)
            if not cand:
                return 0, 0
        cand &= self._get_len_ge(min_req_len)

        scope_m = 0
        hit_m = 0
        while cand:
            lsb = cand & -cand
            i = lsb.bit_length() - 1
            outs = _find_rule_outputs(
                pattern,
                local_envs,
                vtier_env,
                self.x_list[i],
                self.feat_seqs[i],
                self.vc_seqs[i],
                self.v_bit_seqs[i],
                self.word_vset_masks[i],
                self.no_vowel_bit,
            )
            if outs:
                scope_m |= lsb
                if self.y_list[i] in outs:
                    hit_m |= lsb
            cand ^= lsb
        return scope_m, hit_m


# ---------------------------------------------------------------------------
# Multi-Locus Vowel-Tier MGL Learner
# ---------------------------------------------------------------------------


def _specificity_rank(local_envs: tuple[LocalLocusEnv, ...]) -> int:
    score = 0
    for env in local_envs:
        score += 4 * (len(env.left_shared) + len(env.right_shared))
        score += (2 if env.left_feat is not None else 0) + (2 if env.right_feat is not None else 0)
        score += (1 if env.left_anchored else 0) + (1 if env.right_anchored else 0)
    return score


class MGLLearner:
    """Multi-Locus Vowel-Tier Minimal Generalization Learner for a single `(c_s -> c_t)` pair."""

    def __init__(
        self,
        classifier: VowelClassifier | None = None,
        alpha_lower: float = DEFAULT_ALPHA_LOWER,
        alpha_upper: float = DEFAULT_ALPHA_UPPER,
        enable_vowel_tier: bool = True,
        enable_impugnment: bool = True,
    ) -> None:
        self.classifier = classifier if classifier is not None else VowelClassifier()
        self.alpha_lower = alpha_lower
        self.alpha_upper = alpha_upper
        self.enable_vowel_tier = enable_vowel_tier
        self.enable_impugnment = enable_impugnment

        self.rules: list[MGLRule] = []
        self._rules_by_pattern: dict[MultiLocusPattern, list[MGLRule]] = {}
        self._pattern_unique_locus: dict[MultiLocusPattern, bool] = {}
        self._vowel_to_bit: dict[str, int] = {}
        self._other_vowel_bit: int = 1
        self._no_vowel_bit: int = 2
        self._last_alignments: list[AlignmentResult] = []

    def fit(
        self,
        train_pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
    ) -> MGLLearner:
        """Induce, evaluate, and impugn Multi-Locus + Vowel-Tier MGL rules on `train_pairs`."""
        cleaned = [
            (
                tuple(s for s in src if s != "#"),
                tuple(s for s in tgt if s != "#"),
            )
            for src, tgt in train_pairs
        ]
        if not cleaned:
            self.rules = []
            self._rules_by_pattern = {}
            self._last_alignments = []
            return self

        indexed = _IndexedTrainingSet(cleaned, self.classifier)
        self._last_alignments = indexed.alignments
        self._vowel_to_bit = dict(indexed.vowel_to_bit)
        self._other_vowel_bit = indexed.other_vowel_bit
        self._no_vowel_bit = indexed.no_vowel_bit

        # Group training item indices by multi-locus alternation pattern `f`
        by_pattern: dict[MultiLocusPattern, list[int]] = defaultdict(list)
        for idx, aln in enumerate(indexed.alignments):
            by_pattern[aln.pattern].append(idx)

        all_compiled_rules: list[MGLRule] = []
        rules_by_pat: dict[MultiLocusPattern, list[MGLRule]] = {}

        for pattern, item_indices in by_pattern.items():
            pat_rules = self._induce_rules_for_pattern(pattern, item_indices, indexed)
            if pat_rules:
                rules_by_pat[pattern] = pat_rules
                all_compiled_rules.extend(pat_rules)

        # Sort globally by decreasing confidence for fast pruning at prediction time
        all_compiled_rules.sort(
            key=lambda r: (-r.confidence, -r.hits, -r.scope, r.pattern.signature())
        )
        self.rules = all_compiled_rules
        self._rules_by_pattern = rules_by_pat
        self._pattern_unique_locus = {
            pat: (
                pat.is_identity
                or (
                    len(pat.loci) == 1
                    and (
                        pat.loci[0].anchor_left
                        or pat.loci[0].anchor_right
                        or all(
                            r.local_envs[0].right_anchored
                            and not r.local_envs[0].right_shared
                            and r.local_envs[0].right_feat is None
                            for r in p_rules
                        )
                        or all(
                            r.local_envs[0].left_anchored
                            and not r.local_envs[0].left_shared
                            and r.local_envs[0].left_feat is None
                            for r in p_rules
                        )
                    )
                )
            )
            for pat, p_rules in rules_by_pat.items()
        }
        return self

    def _induce_rules_for_pattern(
        self,
        pattern: MultiLocusPattern,
        item_indices: list[int],
        indexed: _IndexedTrainingSet,
    ) -> list[MGLRule]:
        n_loci = len(pattern.loci)

        # 1. Construct word-specific local environments for all items instantiating `pattern`
        word_local_envs: list[tuple[LocalLocusEnv, ...]] = []
        for i in item_indices:
            aln = indexed.alignments[i]
            envs_i = tuple(LocalLocusEnv.from_word_context(c) for c in aln.locus_contexts)
            word_local_envs.append(envs_i)

        # Cap seed set deterministically if `> MAX_PAIRWISE_SEEDS_PER_CHANGE` (e.g. uniform480)
        unique_word_envs = list(dict.fromkeys(word_local_envs))
        seed_envs = unique_word_envs
        seed_indices = item_indices
        if len(seed_envs) > MAX_PAIRWISE_SEEDS_PER_CHANGE:
            step = len(seed_envs) / MAX_PAIRWISE_SEEDS_PER_CHANGE
            seed_envs = [seed_envs[int(k * step)] for k in range(MAX_PAIRWISE_SEEDS_PER_CHANGE)]
        if len(seed_indices) > MAX_PAIRWISE_SEEDS_PER_CHANGE:
            step_i = len(seed_indices) / MAX_PAIRWISE_SEEDS_PER_CHANGE
            seed_indices = [seed_indices[int(k * step_i)] for k in range(MAX_PAIRWISE_SEEDS_PER_CHANGE)]

        # 2. Iterative Minimal Generalization on Local Contexts (`vtier_env = None`)
        local_pool: set[tuple[LocalLocusEnv, ...]] = set(unique_word_envs)
        active_gen: list[tuple[LocalLocusEnv, ...]] = []

        for idx_s, w_env in enumerate(seed_envs):
            new_specs: set[tuple[LocalLocusEnv, ...]] = set()
            # Compare against prior generalized rules AND earlier deduplicated word-specific seeds
            for prev_env in active_gen:
                g = tuple(
                    generalize_local_env(e_a, e_b, self.classifier)
                    for e_a, e_b in zip(w_env, prev_env)
                )
                if g not in local_pool:
                    new_specs.add(g)
            for other_w in seed_envs[:idx_s]:
                g = tuple(
                    generalize_local_env(e_a, e_b, self.classifier)
                    for e_a, e_b in zip(w_env, other_w)
                )
                if g not in local_pool:
                    new_specs.add(g)

            if new_specs:
                local_pool.update(new_specs)
                active_gen.extend(new_specs)
                if len(active_gen) > MAX_LOCAL_LATTICE_PER_CHANGE:
                    # Retain broader generalizations (lower specificity) + diverse shared contexts
                    active_gen.sort(key=_specificity_rank)
                    active_gen = active_gen[:MAX_LOCAL_LATTICE_PER_CHANGE]

        # Guarantee the maximally general rule across ALL items of `pattern` is always present!
        if len(word_local_envs) >= 2:
            top_gen = word_local_envs[0]
            for nxt_env in word_local_envs[1:]:
                top_gen = tuple(
                    generalize_local_env(e_a, e_b, self.classifier)
                    for e_a, e_b in zip(top_gen, nxt_env)
                )
            local_pool.add(top_gen)

        candidate_specs: set[tuple[tuple[LocalLocusEnv, ...], VowelTierEnv | None]] = {
            (l_env, None) for l_env in local_pool
        }

        # 3. Parallel Source-Only Vowel-Tier Generalization (`enable_vowel_tier=True`)
        if self.enable_vowel_tier and len(seed_indices) >= 2 and len(indexed.vowel_to_bit) >= 2:
            # Base local frames for Vowel-Tier rules: maximally general local context for `pattern`,
            # plus major-class stem-final context (`X C ___ #` vs `X V ___ #`) if applicable.
            base_local_frames: set[tuple[LocalLocusEnv, ...]] = set()
            top_local = word_local_envs[0]
            for nxt_env in word_local_envs[1:]:
                top_local = tuple(
                    generalize_local_env(e_a, e_b, self.classifier)
                    for e_a, e_b in zip(top_local, nxt_env)
                )
            base_local_frames.add(top_local)
            # Also include up to 6 broad generalized local frames (<= 2 shared segments)
            broad_locals = sorted(
                (le for le in local_pool if _specificity_rank(le) <= 8 * n_loci),
                key=_specificity_rank,
            )[:6]
            base_local_frames.update(broad_locals)

            # Extract word-specific Vowel-Tier properties strictly from source `w_s`!
            word_vleft: list[int] = []
            word_vright: list[int | None] = []
            word_vset: list[int] = []

            for i in seed_indices:
                aln = indexed.alignments[i]
                vset_m = indexed.word_vset_masks[i]
                word_vset.append(vset_m)

                lv_tuple = aln.locus_contexts[-1].left_vowels
                if lv_tuple:
                    lv_bit = indexed.vowel_to_bit.get(lv_tuple[-1], indexed.other_vowel_bit)
                else:
                    lv_bit = indexed.no_vowel_bit
                word_vleft.append(lv_bit)

                rv_tuple = aln.locus_contexts[0].right_vowels
                if rv_tuple:
                    rv_bit: int | None = indexed.vowel_to_bit.get(
                        rv_tuple[0], indexed.other_vowel_bit
                    )
                else:
                    rv_bit = None
                word_vright.append(rv_bit)

            vtier_specs: set[VowelTierEnv] = set()
            pat_item_mask = 0
            for idx_item in item_indices:
                pat_item_mask |= 1 << idx_item

            # (a) Adjacent left-vowel tier (`v_left_mask`): singletons, whole-pattern union, and
            # purity-filtered harmonic class union (`hit_rate >= 0.75`, preventing spurious vowel
            # dredging on consonant-conditioned non-harmonic languages while uniting `{a, ı, o, u}`!).
            # When the final locus strips a non-empty suffix (`alen > 0`, e.g. Korean `-다` -> `-아/-어`),
            # measure vowel purity left of `m - alen` rather than at the word-final suffix vowel!
            distinct_vleft = sorted(set(word_vleft))
            if (
                not pattern.is_identity
                and not pattern.has_right_slot
                and len(pattern.loci[-1].source_sub) > 0
            ):
                suf_sub = pattern.loci[-1].source_sub
                suf_len = len(suf_sub)
                vleft_pool_items: dict[int, int] = defaultdict(int)
                for idx_all, x_all in enumerate(indexed.x_list):
                    if len(x_all) >= suf_len and x_all[len(x_all) - suf_len :] == suf_sub:
                        lv_all = indexed.no_vowel_bit
                        vb_all = indexed.v_bit_seqs[idx_all]
                        for p_scan in range(len(x_all) - suf_len - 1, -1, -1):
                            if vb_all[p_scan] != 0:
                                lv_all = vb_all[p_scan]
                                break
                        vleft_pool_items[lv_all] |= 1 << idx_all
            else:
                vleft_pool_items = indexed.last_vowel_bit_items

            union_all_vleft = 0
            high_purity_vleft = 0
            pure_vleft_list: list[int] = []
            for b in distinct_vleft:
                union_all_vleft |= b
                vtier_specs.add(VowelTierEnv(v_left_mask=b))
                tot_b = vleft_pool_items.get(b, 0).bit_count()
                hits_b = (vleft_pool_items.get(b, 0) & pat_item_mask).bit_count()
                if tot_b > 0 and (hits_b / tot_b) >= 0.75:
                    high_purity_vleft |= b
                    pure_vleft_list.append(b)
            if union_all_vleft and union_all_vleft != indexed.all_vowel_bits_mask:
                vtier_specs.add(VowelTierEnv(v_left_mask=union_all_vleft))
            if high_purity_vleft and high_purity_vleft != indexed.all_vowel_bits_mask:
                vtier_specs.add(VowelTierEnv(v_left_mask=high_purity_vleft))
            for idx_a, ba in enumerate(pure_vleft_list):
                for bb in pure_vleft_list[idx_a + 1 :]:
                    vtier_specs.add(VowelTierEnv(v_left_mask=ba | bb))

            # (b) Adjacent right-vowel tier (`v_right_mask`) for prefix harmony / umlaut
            distinct_vright = sorted({b for b in word_vright if b is not None})
            if distinct_vright:
                union_vr = 0
                high_purity_vr = 0
                for b in distinct_vright:
                    union_vr |= b
                    vtier_specs.add(VowelTierEnv(v_right_mask=b))
                    tot_r = indexed.first_vowel_bit_items.get(b, 0).bit_count()
                    hits_r = (indexed.first_vowel_bit_items.get(b, 0) & pat_item_mask).bit_count()
                    if tot_r > 0 and (hits_r / tot_r) >= 0.75:
                        high_purity_vr |= b
                if union_vr != indexed.all_vowel_bits_mask:
                    vtier_specs.add(VowelTierEnv(v_right_mask=union_vr))
                if high_purity_vr and high_purity_vr != indexed.all_vowel_bits_mask:
                    vtier_specs.add(VowelTierEnv(v_right_mask=high_purity_vr))

            # (c) Stem vowel-set intersection (`req_vowel_mask`: e.g. `talo & paperi -> {'a'}`)
            # and whole-pattern / purity-filtered allowed-vowel envelopes (`{i, e, y, ä, ö}`)!
            distinct_vsets = sorted({vs for vs in word_vset if vs != 0})
            union_all_vset = 0
            for vs in distinct_vsets:
                union_all_vset |= vs
                vtier_specs.add(VowelTierEnv(req_vowel_mask=vs))
                vtier_specs.add(VowelTierEnv(allowed_vowel_mask=vs))
            if union_all_vset and union_all_vset != (
                indexed.all_vowel_bits_mask & ~indexed.no_vowel_bit
            ):
                vtier_specs.add(VowelTierEnv(allowed_vowel_mask=union_all_vset))

            # Pairwise intersections (`&`) of stem vowel sets (strictly narrower/shared trigger vowels)
            # plus union of vowel sets whose individual stem-presence hit rate is high
            high_purity_vset = 0
            for vbit in indexed.vowel_to_bit.values():
                tot_v = indexed.has_vowel_bit_mask.get(vbit, 0).bit_count()
                hits_v = (indexed.has_vowel_bit_mask.get(vbit, 0) & pat_item_mask).bit_count()
                if tot_v > 0 and (hits_v / tot_v) >= 0.75:
                    high_purity_vset |= vbit
            if high_purity_vset and high_purity_vset != (
                indexed.all_vowel_bits_mask & ~indexed.no_vowel_bit
            ):
                vtier_specs.add(VowelTierEnv(allowed_vowel_mask=high_purity_vset))

            for idx_a, vs_a in enumerate(distinct_vsets):
                for vs_b in distinct_vsets[idx_a + 1 :]:
                    inter_vs = vs_a & vs_b
                    if inter_vs:
                        vtier_specs.add(VowelTierEnv(req_vowel_mask=inter_vs))
                        if union_all_vset:
                            vtier_specs.add(
                                VowelTierEnv(
                                    req_vowel_mask=inter_vs,
                                    allowed_vowel_mask=union_all_vset,
                                )
                            )

            # Also combine `v_left_mask` with `union_all_vset` so transparent-vowel rules like
            # `v_left='i'` have their harmonic-domain specialization (`v_left='i' & allowed<=front`)
            # materialized in the grammar for exact Albright & Hayes (§3.7) impugnment!
            if union_all_vset and union_all_vset != (
                indexed.all_vowel_bits_mask & ~indexed.no_vowel_bit
            ):
                for vt_vl in list(vtier_specs):
                    if vt_vl.v_left_mask is not None and vt_vl.allowed_vowel_mask is None:
                        vtier_specs.add(
                            VowelTierEnv(
                                v_left_mask=vt_vl.v_left_mask,
                                allowed_vowel_mask=union_all_vset,
                            )
                        )

            mandatory_vtier: set[VowelTierEnv] = set()
            if union_all_vleft and union_all_vleft != indexed.all_vowel_bits_mask:
                mandatory_vtier.add(VowelTierEnv(v_left_mask=union_all_vleft))
            if union_all_vset and union_all_vset != (
                indexed.all_vowel_bits_mask & ~indexed.no_vowel_bit
            ):
                mandatory_vtier.add(VowelTierEnv(allowed_vowel_mask=union_all_vset))
            # Pair mandatory whole-pattern Vowel-Tier envelopes with all generalized local contexts
            # so local accidental contexts (e.g. `X t V ___ #`) are cleanly specialized & impugned
            for l_env in local_pool:
                for m_vt in mandatory_vtier:
                    candidate_specs.add((l_env, m_vt))

            if len(vtier_specs) > MAX_VTIER_LATTICE_PER_CHANGE:
                rem_specs = sorted(
                    vtier_specs - mandatory_vtier,
                    key=lambda vt: (
                        0 if vt.req_vowel_mask == 0 else 1,
                        -((vt.allowed_vowel_mask or 0).bit_count()),
                        -((vt.v_left_mask or 0).bit_count()),
                        vt.req_vowel_mask.bit_count(),
                    ),
                )[: max(0, MAX_VTIER_LATTICE_PER_CHANGE - len(mandatory_vtier))]
                vtier_list = list(mandatory_vtier) + rem_specs
            else:
                vtier_list = list(vtier_specs | mandatory_vtier)

            for lf in base_local_frames:
                for vt in vtier_list:
                    candidate_specs.add((lf, vt))

        # 4. Evaluate exact `(scope_mask, hit_mask)` bitmasks across ALL `N_train` items
        evaluated: list[
            tuple[tuple[LocalLocusEnv, ...], VowelTierEnv | None, int, int]
        ] = []
        unique_mask_pairs: set[tuple[int, int]] = set()

        for l_envs, vt_env in candidate_specs:
            s_mask, h_mask = indexed.evaluate_rule_masks(pattern, l_envs, vt_env)
            if h_mask == 0 or s_mask == 0:
                continue
            evaluated.append((l_envs, vt_env, s_mask, h_mask))
            unique_mask_pairs.add((s_mask, h_mask))

        if not evaluated:
            return []

        # 5. Compute Mikheev `pi_L(alpha=0.75)` & Exact Residue Impugnment `pi_U(alpha_upper=0.95)`
        # over deduplicated `(scope_mask, hit_mask)` equivalence classes!
        mask_list = sorted(unique_mask_pairs, key=lambda pair: (pair[0].bit_count(), pair[0]))
        lower_by_pair: dict[tuple[int, int], float] = {}
        final_conf_by_pair: dict[tuple[int, int], float] = {}

        for s_m, h_m in mask_list:
            n_val = s_m.bit_count()
            x_val = h_m.bit_count()
            pi_l = mikheev_lower(x_val, n_val, self.alpha_lower)
            lower_by_pair[(s_m, h_m)] = pi_l
            final_conf_by_pair[(s_m, h_m)] = pi_l

        if self.enable_impugnment and len(mask_list) >= 2:
            # Sort unique scope bitmasks by `(bit_count, mask)` so `n_sub >= n_sup` is a valid break!
            unique_scopes = sorted(
                {(s_m.bit_count(), s_m) for s_m, _ in mask_list}
            )
            for s_sup, h_sup in mask_list:
                conf = lower_by_pair[(s_sup, h_sup)]
                if conf <= 0.0:
                    continue
                n_sup = s_sup.bit_count()
                for n_sub, s_sub in unique_scopes:
                    if n_sub >= n_sup:
                        break
                    # Strict subset check in O(1) Python int bit-ops: `s_sub & ~s_sup == 0`
                    if s_sub < s_sup and (s_sub & ~s_sup) == 0:
                        s_res = s_sup & ~s_sub
                        n_res = n_sup - n_sub
                        x_res = (h_sup & s_res).bit_count()
                        pi_u = mikheev_upper(x_res, n_res, self.alpha_upper)
                        if pi_u < conf:
                            conf = pi_u
                            if conf <= 0.0:
                                break
                final_conf_by_pair[(s_sup, h_sup)] = conf

        # 6. Assemble `MGLRule` objects
        compiled: list[MGLRule] = []
        for l_envs, vt_env, s_m, h_m in evaluated:
            n_val = s_m.bit_count()
            x_val = h_m.bit_count()
            pi_l = lower_by_pair[(s_m, h_m)]
            conf = final_conf_by_pair[(s_m, h_m)]
            compiled.append(
                MGLRule(
                    pattern=pattern,
                    local_envs=l_envs,
                    vtier_env=vt_env,
                    scope_mask=s_m,
                    hit_mask=h_m,
                    scope=n_val,
                    hits=x_val,
                    raw_reliability=x_val / n_val,
                    lower_confidence=pi_l,
                    confidence=conf,
                )
            )

        compiled.sort(key=lambda r: (-r.confidence, -r.hits, -r.scope, _specificity_rank(r.local_envs)))
        return compiled

    def score_candidates(self, source_segs: Sequence[str]) -> dict[tuple[str, ...], float]:
        """Apply learned grammar to novel `source_segs` (`w_s`) and compute `Score(y | w_s)` via `max`.

        Within-candidate maximization (NO summation across nested rules!):
            `Score(y | w_s) = max_{R: w_s => y} Confidence(R) in [0, 1]`.
        """
        x = tuple(s for s in source_segs if s != "#")
        if not self.rules:
            return {}

        feat_seq = tuple(self.classifier.segment_features(s) for s in x)
        vc_seq = tuple(self.classifier.classify(s) for s in x)
        v_bit_list: list[int] = []
        word_vset = 0
        for s in x:
            svs = self.classifier.segment_vowels(s)
            if svs:
                seg_vb = 0
                for v_sym in svs:
                    vb = self._vowel_to_bit.get(v_sym, self._other_vowel_bit)
                    word_vset |= vb
                    seg_vb = vb
                v_bit_list.append(seg_vb)
            else:
                v_bit_list.append(0)
        v_bit_seq = tuple(v_bit_list)

        best_scores: dict[tuple[str, ...], float] = {}

        # Evaluate per alternation pattern `f` in descending order of `r.confidence`:
        # For single-output patterns (suffix, prefix, identity — where every rule of `f` produces
        # the SAME output string `y` on `x`), the FIRST matching rule immediately achieves the
        # exact maximum `max_R Confidence(R)` for `f`, skipping all lower-confidence rules of `f`!
        m_x = len(x)
        x_seg_set = frozenset(x)
        for pattern, pat_rules in self._rules_by_pattern.items():
            # O(1) pattern-level structural precheck before iterating over rules of `pattern`!
            if not pattern.is_identity:
                loci_p = pattern.loci
                if len(loci_p) == 1:
                    sub0 = loci_p[0].source_sub
                    slen0 = len(sub0)
                    if slen0 > m_x:
                        continue
                    if slen0 > 0:
                        if not pattern.has_right_slot and x[m_x - slen0 :] != sub0:
                            continue
                        if not pattern.has_left_slot and x[:slen0] != sub0:
                            continue
                        if slen0 == 1 and sub0[0] not in x_seg_set:
                            continue
                else:
                    if any(s_req not in x_seg_set for loc_p in loci_p for s_req in loc_p.source_sub):
                        continue

            is_unique_locus = self._pattern_unique_locus.get(pattern, False)
            for rule in pat_rules:
                if rule.confidence <= 0.0:
                    break
                outs = _find_rule_outputs(
                    pattern,
                    rule.local_envs,
                    rule.vtier_env,
                    x,
                    feat_seq,
                    vc_seq,
                    v_bit_seq,
                    word_vset,
                    self._no_vowel_bit,
                )
                if not outs:
                    continue
                for y_cand in outs:
                    prev = best_scores.get(y_cand, 0.0)
                    if rule.confidence > prev:
                        best_scores[y_cand] = rule.confidence
                if is_unique_locus:
                    break

        return best_scores
