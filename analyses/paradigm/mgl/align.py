"""Global Needleman-Wunsch V/C-weighted sequence aligner and multi-locus pattern extractor.

Implements Study 3 Track 2 §2.1 step 1:
- Global Needleman-Wunsch alignment between `# x_1 ... x_m #` and `# y_1 ... y_n #`
  with exact costs:
    * `cost(x == y) = 0.0`
    * `cost(V, V') = cost(C, C') = 1.0`  (same major class substitution)
    * `cost(V, C) = cost(C, V) = 2.5`   (vowel-consonant mismatch penalty)
    * `indel = 1.1`                     (insertion or deletion)
- Deterministic tie-breaking among equal-cost alignments via a tiny locus-contiguity
  tie-breaker (`_LOCUS_OPEN_EPS = 1e-6`, strictly smaller than `0.1` even across 10,000
  segments) so duplicate stem-edge segments (e.g., `need -> needed`, `talo -> taloon`,
  `essen -> gegessen`) keep the stem contiguous rather than splitting an infix out of the stem.
- Extracts multi-slot alternation pattern `f` (e.g. `_ -> _t`, `_ -> ge_t`, `_e_o -> _ie_e`)
  together with invariant slot segments and source-only Vowel-Tier contexts around each locus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .vowel_tier import VowelClassifier

COST_MATCH: float = 0.0
COST_SAME_CLASS_SUB: float = 1.0
COST_CROSS_CLASS_SUB: float = 2.5
COST_INDEL: float = 1.1

# Strictly infinitesimally small tie-breakers (< 1e-4 << 0.1 minimum cost quantum)
# ensuring contiguous stem/affix spans and canonical edge placement on equal-cost ties.
_LOCUS_OPEN_EPS: float = 1e-6
_INTERNAL_INDEL_EPS: float = 1e-8


@dataclass(frozen=True)
class LocusEdit:
    """Single alternation locus `source_sub -> target_sub` inside a multi-locus pattern `f`.

    `anchor_left`: True iff `source_sub == ()` and the insertion occurs immediately at `# ___`
                   (prefix insertion, `S_0 == ()`).
    `anchor_right`: True iff `source_sub == ()` and the insertion occurs immediately at `___ #`
                    (suffix insertion, `S_J == ()`, or identity `_ -> _` at `___ #`).
    """

    source_sub: tuple[str, ...]
    target_sub: tuple[str, ...]
    anchor_left: bool = False
    anchor_right: bool = False


@dataclass(frozen=True)
class MultiLocusPattern:
    """Canonical hashable representation of a multi-slot alternation pattern `f`."""

    loci: tuple[LocusEdit, ...]
    is_identity: bool = False
    has_left_slot: bool = True
    has_right_slot: bool = False

    def signature(self) -> str:
        """Human-readable signature such as `'_ -> _t'`, `'_ -> ge_t'`, `'_ɛ_o -> _ie_e'`."""
        if self.is_identity:
            return "_ -> _"
        src_parts: list[str] = []
        tgt_parts: list[str] = []
        n_loci = len(self.loci)
        for idx, loc in enumerate(self.loci):
            if idx == 0:
                if self.has_left_slot and not loc.anchor_left:
                    src_parts.append("_")
                    tgt_parts.append("_")
            else:
                src_parts.append("_")
                tgt_parts.append("_")
            src_parts.append("".join(loc.source_sub))
            tgt_parts.append("".join(loc.target_sub))
            if idx == n_loci - 1 and self.has_right_slot and not loc.anchor_right:
                src_parts.append("_")
                tgt_parts.append("_")
        src_s = "".join(src_parts) or "_"
        tgt_s = "".join(tgt_parts) or "_"
        return f"{src_s} -> {tgt_s}"


@dataclass(frozen=True)
class LocusContext:
    """Source-only local and Vowel-Tier context around one change locus in `# x_1 ... x_m #`."""

    # Full left/right segment contexts in `# x_1 ... x_m #` (including leading/trailing `#`)
    left_local: tuple[str, ...]
    right_local: tuple[str, ...]
    # Source-only Vowel-Tier projection to the left and right of this locus in `w_s`
    left_vowels: tuple[str, ...]
    right_vowels: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentResult:
    """Result of aligning `# x_1 ... x_m #` and `# y_1 ... y_n #`."""

    columns: tuple[tuple[str, str], ...]
    distance: float
    pattern: MultiLocusPattern
    invariant_slots: tuple[tuple[str, ...], ...]
    locus_contexts: tuple[LocusContext, ...]
    source_vowel_tier: tuple[str, ...]


# Exact 1e8-scaled integer DP costs (`0.0, 1.0, 2.5, 1.1, 1e-6, 1e-8`) eliminating float round()!
_COST_SAME_SUB_I: int = 100_000_000
_COST_CROSS_SUB_I: int = 250_000_000
_COST_INDEL_I: int = 110_000_000
_LOCUS_OPEN_I: int = 100
_INTERNAL_INDEL_I: int = 1
_INF_I: int = 10**18


def align_pair(
    source_segs: Sequence[str],
    target_segs: Sequence[str],
    classifier: VowelClassifier,
) -> tuple[tuple[tuple[str, str], ...], float]:
    """Align `# x_1 ... x_m #` and `# y_1 ... y_n #` via global V/C-penalized Needleman-Wunsch.

    Returns `(aligned_columns_including_boundaries, exact_base_distance)`.
    `aligned_columns_including_boundaries` starts with `("#", "#")` and ends with `("#", "#")`;
    gaps are represented by `""`.
    """
    x = tuple(s for s in source_segs if s != "#")
    y = tuple(s for s in target_segs if s != "#")
    m = len(x)
    n = len(y)

    if m == 0 and n == 0:
        return (("#", "#"), ("#", "#")), 0.0
    if x == y:
        return (("#", "#"), *((s, s) for s in x), ("#", "#")), 0.0

    vc_x = [classifier.classify(s) for s in x]
    vc_y = [classifier.classify(s) for s in y]

    # 2-state exact integer affine-tie-broken DP:
    # state 0 = currently in invariant MATCH (or start `#`)
    # state 1 = currently inside an EDIT locus (SUB / DEL / INS)
    # Backpointer codes:
    #   bp0: 0 = diag from state 0, 1 = diag from state 1
    #   bp1: 0 = sub from 0, 1 = sub from 1, 2 = del from 0, 3 = del from 1, 4 = ins from 0, 5 = ins from 1
    dp0 = [[_INF_I] * (n + 1) for _ in range(m + 1)]
    dp1 = [[_INF_I] * (n + 1) for _ in range(m + 1)]
    bp0 = [[-1] * (n + 1) for _ in range(m + 1)]
    bp1 = [[-1] * (n + 1) for _ in range(m + 1)]

    dp0[0][0] = 0

    for i in range(m + 1):
        row0 = dp0[i]
        row1 = dp1[i]
        has_next_i = i < m
        if has_next_i:
            xi = x[i]
            vxi = vc_x[i]
            next_row0 = dp0[i + 1]
            next_row1 = dp1[i + 1]
            next_bp0 = bp0[i + 1]
            next_bp1 = bp1[i + 1]
            del_edge_pen = 0 if (i == 0 or i + 1 == m) else _INTERNAL_INDEL_I
            del_step = _COST_INDEL_I + del_edge_pen

        cur_bp1 = bp1[i]
        for j in range(n + 1):
            c0 = row0[j]
            c1 = row1[j]

            # 1. Diagonal step (i -> i+1, j -> j+1)
            if has_next_i and j < n:
                if xi == y[j]:
                    if c0 <= c1:
                        best_match = c0
                        code0 = 0
                    else:
                        best_match = c1
                        code0 = 1
                    if best_match < next_row0[j + 1]:
                        next_row0[j + 1] = best_match
                        next_bp0[j + 1] = code0
                else:
                    sub_i = _COST_SAME_SUB_I if vxi == vc_y[j] else _COST_CROSS_SUB_I
                    via_0 = c0 + sub_i + _LOCUS_OPEN_I
                    via_1 = c1 + sub_i
                    if via_1 <= via_0:
                        best_sub = via_1
                        code1 = 1
                    else:
                        best_sub = via_0
                        code1 = 0
                    if best_sub < next_row1[j + 1]:
                        next_row1[j + 1] = best_sub
                        next_bp1[j + 1] = code1

            # 2. Deletion step (x[i] -> "", i -> i+1, j -> j)
            if has_next_i:
                via_0 = c0 + del_step + _LOCUS_OPEN_I
                via_1 = c1 + del_step
                if via_1 <= via_0:
                    best_del = via_1
                    dcode = 3
                else:
                    best_del = via_0
                    dcode = 2
                if best_del < next_row1[j]:
                    next_row1[j] = best_del
                    next_bp1[j] = dcode

            # 3. Insertion step ("" -> y[j], i -> i, j -> j+1)
            if j < n:
                ins_edge_pen = 0 if (j == 0 or j + 1 == n) else _INTERNAL_INDEL_I
                ins_step = _COST_INDEL_I + ins_edge_pen
                via_0 = c0 + ins_step + _LOCUS_OPEN_I
                via_1 = c1 + ins_step
                if via_1 <= via_0:
                    best_ins = via_1
                    icode = 5
                else:
                    best_ins = via_0
                    icode = 4
                if best_ins < row1[j + 1]:
                    row1[j + 1] = best_ins
                    cur_bp1[j + 1] = icode

    end_state = 0 if dp0[m][n] <= dp1[m][n] else 1

    rev_cols: list[tuple[str, str]] = []
    total_base_i = 0
    ci, cj, cs = m, n, end_state
    while ci > 0 or cj > 0:
        if cs == 0:
            code = bp0[ci][cj]
            cs = code
            ci -= 1
            cj -= 1
            rev_cols.append((x[ci], y[cj]))
        else:
            code = bp1[ci][cj]
            cs = code & 1
            op = code >> 1
            if op == 0:
                ci -= 1
                cj -= 1
                rev_cols.append((x[ci], y[cj]))
                total_base_i += _COST_SAME_SUB_I if vc_x[ci] == vc_y[cj] else _COST_CROSS_SUB_I
            elif op == 1:
                ci -= 1
                rev_cols.append((x[ci], ""))
                total_base_i += _COST_INDEL_I
            else:
                cj -= 1
                rev_cols.append(("", y[cj]))
                total_base_i += _COST_INDEL_I

    rev_cols.reverse()
    full_cols = (("#", "#"), *rev_cols, ("#", "#"))
    return full_cols, round(total_base_i * 1e-8, 6)


def extract_pattern(
    source_segs: Sequence[str],
    target_segs: Sequence[str],
    classifier: VowelClassifier,
) -> AlignmentResult:
    """Align `(source_segs, target_segs)` and extract `f`, invariant slots, and source Vowel Tier.

    CRITICAL: `locus_contexts` and `source_vowel_tier` are computed strictly from `source_segs`
    (`w_s`) and never include target vowels from `target_segs` (`w_t`).
    """
    x = tuple(s for s in source_segs if s != "#")
    y = tuple(s for s in target_segs if s != "#")
    cache_key = (x, y)
    cached_aln = classifier._pattern_cache.get(cache_key)
    if cached_aln is not None:
        return cached_aln  # type: ignore[return-value]

    cols, dist = align_pair(x, y, classifier)
    inner_cols = cols[1:-1]

    seg_v_x = [classifier.segment_vowels(s) for s in x]
    source_vtier = tuple(v for vs in seg_v_x for v in vs)

    # Identity mapping (`w_s == w_t`): represent as single zero-change locus at right boundary `___ #`
    if x == y:
        ident_loc = LocusEdit(source_sub=(), target_sub=(), anchor_left=False, anchor_right=True)
        pattern = MultiLocusPattern(loci=(ident_loc,), is_identity=True)
        ctx = LocusContext(
            left_local=("#", *x),
            right_local=("#",),
            left_vowels=source_vtier,
            right_vowels=(),
        )
        res_ident = AlignmentResult(
            columns=cols,
            distance=0.0,
            pattern=pattern,
            invariant_slots=(x, ()),
            locus_contexts=(ctx,),
            source_vowel_tier=source_vtier,
        )
        if len(classifier._pattern_cache) < 250_000:
            classifier._pattern_cache[cache_key] = res_ident
        return res_ident

    # Partition `inner_cols` into alternating invariant spans and contiguous edit loci,
    # tracking exact source-segment start/end indices `[src_start, src_end)` in `x`.
    invariant_slots: list[tuple[str, ...]] = []
    loci: list[LocusEdit] = []
    locus_spans_in_x: list[tuple[int, int]] = []

    cur_inv: list[str] = []
    cur_edit_src: list[str] = []
    cur_edit_tgt: list[str] = []
    in_edit = False
    src_pos = 0
    edit_start_x = 0

    for a, b in inner_cols:
        if a == b:
            if in_edit:
                invariant_slots.append(tuple(cur_inv))
                cur_inv = []
                loci.append(
                    LocusEdit(
                        source_sub=tuple(cur_edit_src),
                        target_sub=tuple(cur_edit_tgt),
                    )
                )
                locus_spans_in_x.append((edit_start_x, src_pos))
                cur_edit_src = []
                cur_edit_tgt = []
                in_edit = False
            cur_inv.append(a)
            src_pos += 1
        else:
            if not in_edit:
                edit_start_x = src_pos
                in_edit = True
            if a != "":
                cur_edit_src.append(a)
                src_pos += 1
            if b != "":
                cur_edit_tgt.append(b)

    if in_edit:
        invariant_slots.append(tuple(cur_inv))
        cur_inv = []
        loci.append(
            LocusEdit(
                source_sub=tuple(cur_edit_src),
                target_sub=tuple(cur_edit_tgt),
            )
        )
        locus_spans_in_x.append((edit_start_x, src_pos))

    invariant_slots.append(tuple(cur_inv))
    assert len(invariant_slots) == len(loci) + 1

    # Annotate boundary anchors on pure insertions (`source_sub == ()`) so prefixes `ge_`,
    # suffixes `_t`, and infixes `_um_` remain structurally distinct in `f`.
    anchored_loci: list[LocusEdit] = []
    n_loci = len(loci)
    for idx, loc in enumerate(loci):
        is_pure_ins = len(loc.source_sub) == 0
        anc_l = bool(is_pure_ins and idx == 0 and len(invariant_slots[0]) == 0)
        anc_r = bool(is_pure_ins and idx == n_loci - 1 and len(invariant_slots[-1]) == 0)
        anchored_loci.append(
            LocusEdit(
                source_sub=loc.source_sub,
                target_sub=loc.target_sub,
                anchor_left=anc_l,
                anchor_right=anc_r,
            )
        )

    # Extract local & source-only Vowel-Tier contexts around each locus `[p_j, q_j)` in `x`
    contexts: list[LocusContext] = []
    for p_j, q_j in locus_spans_in_x:
        left_loc = ("#", *x[:p_j])
        right_loc = (*x[q_j:], "#")
        left_v = tuple(v for vs in seg_v_x[:p_j] for v in vs)
        right_v = tuple(v for vs in seg_v_x[q_j:] for v in vs)
        contexts.append(
            LocusContext(
                left_local=left_loc,
                right_local=right_loc,
                left_vowels=left_v,
                right_vowels=right_v,
            )
        )

    pattern = MultiLocusPattern(
        loci=tuple(anchored_loci),
        is_identity=False,
        has_left_slot=(len(invariant_slots[0]) > 0),
        has_right_slot=(len(invariant_slots[-1]) > 0),
    )
    res_aln = AlignmentResult(
        columns=cols,
        distance=dist,
        pattern=pattern,
        invariant_slots=tuple(invariant_slots),
        locus_contexts=tuple(contexts),
        source_vowel_tier=source_vtier,
    )
    if len(classifier._pattern_cache) < 250_000:
        classifier._pattern_cache[cache_key] = res_aln
    return res_aln
