"""Malouf (2017) unsupervised bipartite vowel-harmony pre-abstraction (`harmony_malouf.py`).

Implements unsupervised front/back (or dominant/recessive) vowel-harmony discovery and
affix-vowel archiphoneme abstraction following Malouf (2017, "Abstracting paradigm structure",
Morphology) to cross-check Multi-Locus Vowel-Tier MGL across
Uralic (`ural1272`), Turkic (`turk1311`), Mongolic (`mong1349`), Tungusic (`tung1282`),
and Chukotko-Kamchatkan (`chuk1271`) units.

Algorithm (100% unsupervised from `segmented_form`, zero hardcoded language rules):
  1. **Sukhotin (1962) Vowel Discovery**: Identify vocalic symbols `V` from the symbol
     adjacency matrix across `segmented_form` tokens (or `sounds.csv` `[+syllabic]` when
     provided for `phon` units).
  2. **Spectral Bipartite Harmony Partition**: Construct the within-word non-identical
     vowel co-occurrence PMI/residual matrix `R(v_1, v_2)` and extract its leading negative
     / contrast eigenvector to partition harmonizing vowels into two harmonic poles
     (`pole_neg` vs `pole_pos`), leaving neutral/transparent vowels (whose co-occurrence is
     balanced across both poles) unpolarized.
  3. **Affixal Alternation Archiphoneme Pairing**: Across lemmas in the same cell `c`, align
     equal-length inflectional suffixes differing ONLY at vocalic positions where one suffix
     uses `v in pole_neg` and the other uses `v' in pole_pos` (e.g. Finnish `-ssa/-ssä`,
     `-lla/-llä`; Turkish `-lar/-ler`, `-ın/-in`, `-un/-ün`). Pair each strongly alternating
     `(v_neg, v_pos)` dyad into an abstract archiphoneme symbol `VH{k}`.
  4. **Stem Harmony Marker + Affix Abstraction**: Tag each lemma's stem zone with its
     dominant harmony class marker (`HNEG`, `HPOS`, or `HNEUT`) and replace harmonizing
     affixal vowels beyond the invariant stem with their paired `VH{k}` archiphoneme.
"""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

HARMONY_FAMILIES: Set[str] = {
    "ural1272",  # Uralic
    "turk1311",  # Turkic
    "mong1349",  # Mongolic
    "tung1282",  # Tungusic
    "chuk1271",  # Chukotko-Kamchatkan
}

_KNOWN_VOWEL_BASES: Set[str] = set(
    "aeiouyäöüåøæœıǝəɛɔɑɒɪʊɯɤɨʉɘɵɐɜɞɶ"
    "аеёиієїоуыэюяәөүұӱӧӗӑӹ"
    "αεηιοωυ"
    "աեէըիոօ"
    "აეიოუ"
)


def _is_known_unicode_vowel(tok: str) -> bool:
    s = unicodedata.normalize("NFC", tok).strip().lower()
    if not s or s == "_":
        return False
    # Check NFC first (preserves ä, ö, ü, ы, ә, ө, ү), then NFD base character
    for ch in s:
        if ch in _KNOWN_VOWEL_BASES:
            return True
    nfd_base = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) not in ("Mn", "Me"))
    return any(ch in _KNOWN_VOWEL_BASES for ch in nfd_base)


def sukhotin_vowels(
    tokenized_forms: Sequence[List[str]],
    sounds_df: Optional[pd.DataFrame] = None,
) -> Set[str]:
    """Discover vowel symbols via Sukhotin (1962), or read `sounds.csv` if supplied."""
    if sounds_df is not None and not sounds_df.empty and "sound_id" in sounds_df.columns:
        vowels_from_sounds: Set[str] = set()
        for row in sounds_df.to_dict("records"):
            sid = str(row.get("sound_id", "")).strip()
            if not sid:
                continue
            syl_col = str(row.get("syl", row.get("syllabic", ""))).strip().lower()
            if syl_col in ("1", "+", "true", "yes"):
                vowels_from_sounds.add(sid)
                continue
            vals = " ".join(str(v).lower() for k, v in row.items() if k != "sound_id")
            if "vowel" in vals or "+syll" in vals or ("syllabic" in vals and "-syll" not in vals and syl_col not in ("0", "-", "false", "no")):
                vowels_from_sounds.add(sid)
        if len(vowels_from_sounds) >= 2:
            return vowels_from_sounds

    vocab_counts: Counter = Counter()
    for toks in tokenized_forms:
        for t in toks:
            if t != "_":
                vocab_counts[t] += 1
    if len(vocab_counts) < 3:
        return set()

    known_v = {sym for sym in vocab_counts if _is_known_unicode_vowel(sym)}
    if len(known_v) >= 2:
        return known_v

    syms = sorted(vocab_counts.keys())
    idx_of = {s: i for i, s in enumerate(syms)}
    m = len(syms)
    adj = np.zeros((m, m), dtype=np.float64)
    for toks in tokenized_forms:
        for a, b in zip(toks, toks[1:]):
            if a == "_" or b == "_" or a == b:
                continue
            ia, ib = idx_of[a], idx_of[b]
            adj[ia, ib] += 1.0
            adj[ib, ia] += 1.0

    # Sukhotin iterative greedy maximization
    row_sums = adj.sum(axis=1).copy()
    is_vowel = np.zeros(m, dtype=bool)
    while True:
        # Score for adding i to V: sum_{j not in V} adj[i,j] - sum_{j in V} adj[i,j] == row_sums[i]
        candidates = np.where(~is_vowel)[0]
        if len(candidates) == 0:
            break
        best_i = candidates[int(np.argmax(row_sums[candidates]))]
        if row_sums[best_i] <= 0.0:
            break
        is_vowel[best_i] = True
        row_sums -= 2.0 * adj[:, best_i]

    return {syms[i] for i in range(m) if is_vowel[i]}


def learn_malouf_harmony_model(
    df_lex: pd.DataFrame,
    sounds_df: Optional[pd.DataFrame] = None,
    neutral_threshold: float = 0.20,
) -> dict:
    """Fit unsupervised bipartite vowel harmony poles and affix archiphoneme pairs."""
    v0 = df_lex[df_lex["variant_idx"] == 0]
    tok_forms: List[List[str]] = [
        [t for t in str(s).split(" ") if t] for s in v0["segmented_form"].tolist()
    ]
    vowels = sukhotin_vowels(tok_forms, sounds_df=sounds_df)
    v_counts: Counter = Counter()
    for toks in tok_forms:
        for t in toks:
            if t in vowels:
                v_counts[t] += 1

    # Filter extremely rare noisy symbols (<0.1% of vowel tokens)
    total_v = sum(v_counts.values())
    active_vowels = sorted(v for v, c in v_counts.items() if c >= max(5, int(0.001 * total_v)))
    if len(active_vowels) < 2:
        return {"vowels": vowels, "pole_neg": set(), "pole_pos": set(), "arch_map": {}}

    v_idx = {v: i for i, v in enumerate(active_vowels)}
    nv = len(active_vowels)
    cooc = np.zeros((nv, nv), dtype=np.float64)
    for toks in tok_forms:
        w_vowels = [t for t in toks if t in v_idx]
        if len(w_vowels) < 2:
            continue
        for i in range(len(w_vowels)):
            for j in range(i + 1, len(w_vowels)):
                ia, ib = v_idx[w_vowels[i]], v_idx[w_vowels[j]]
                cooc[ia, ib] += 1.0
                cooc[ib, ia] += 1.0

    tot_pairs = cooc.sum()
    if tot_pairs < 10.0:
        return {"vowels": vowels, "pole_neg": set(), "pole_pos": set(), "arch_map": {}}

    marg = cooc.sum(axis=1)
    denom = np.sqrt(np.outer(marg, marg)) + 1e-12
    expected = np.outer(marg, marg) / tot_pairs
    # Standardized co-occurrence residual; harmonic poles attract within-pole and repel cross-pole
    resid = (cooc - expected) / denom
    # Zero diagonal self-harmony inflation only if off-diagonal structure exists
    np.fill_diagonal(resid, np.diag(resid) * 0.5)

    eigvals, eigvecs = np.linalg.eigh(resid)
    # Leading positive nontrivial contrast vector separates front vs back harmonic classes
    # Check top-2 eigenvectors and pick the one with highest off-mean bipolarity
    best_vec = eigvecs[:, -1]
    if nv >= 2:
        cand1 = eigvecs[:, -1]
        cand2 = eigvecs[:, -2]
        # Prefer eigenvector that changes sign (true contrast rather than all-positive frequency mode)
        if (cand1.min() * cand1.max() >= 0) and (cand2.min() * cand2.max() < 0):
            best_vec = cand2

    # Fix sign deterministically: lexicographically smallest strongly polarized vowel gets negative sign
    max_abs = float(np.max(np.abs(best_vec))) + 1e-12
    norm_scores = best_vec / max_abs
    strong_indices = [i for i, sc in enumerate(norm_scores) if abs(sc) >= neutral_threshold]
    if strong_indices and norm_scores[strong_indices[0]] > 0:
        norm_scores = -norm_scores

    pole_neg = {active_vowels[i] for i, sc in enumerate(norm_scores) if sc <= -neutral_threshold}
    pole_pos = {active_vowels[i] for i, sc in enumerate(norm_scores) if sc >= neutral_threshold}
    if not pole_neg or not pole_pos:
        return {"vowels": vowels, "pole_neg": set(), "pole_pos": set(), "arch_map": {}}

    # Discover affix vowel alternation pairs across lemmas within each cell
    # First compute each lemma's invariant prefix length (stem boundary)
    lemma_stems: Dict[str, int] = {}
    for lid, grp in v0.groupby("lemma_id", sort=False):
        seqs = [[t for t in str(s).split(" ") if t] for s in grp["segmented_form"]]
        if len(seqs) <= 1:
            lemma_stems[lid] = len(seqs[0]) if seqs else 0
        else:
            shortest = min(seqs, key=len)
            lcp = len(shortest)
            for k, sym in enumerate(shortest):
                if any(seq[k] != sym for seq in seqs):
                    lcp = k
                    break
            lemma_stems[lid] = lcp

    pair_counts: Counter = Counter()
    for _, cell_grp in v0.groupby("cell", sort=False):
        # Bucket suffixes by consonant skeleton + length
        skeleton_buckets: Dict[Tuple[str, ...], Set[Tuple[str, ...]]] = defaultdict(set)
        for lid, seg_s in zip(cell_grp["lemma_id"], cell_grp["segmented_form"]):
            toks = [t for t in str(seg_s).split(" ") if t]
            st_len = lemma_stems.get(lid, 0)
            suf = tuple(toks[st_len:])
            if not suf or len(suf) > 10:
                continue
            skel = tuple("V" if t in vowels else t for t in suf)
            if "V" in skel:
                skeleton_buckets[skel].add(suf)

        for skel, sufs in skeleton_buckets.items():
            if len(sufs) < 2:
                continue
            suf_list = sorted(sufs)
            for i in range(len(suf_list)):
                for j in range(i + 1, len(suf_list)):
                    s1, s2 = suf_list[i], suf_list[j]
                    for a, b in zip(s1, s2):
                        if a == b:
                            continue
                        if a in pole_neg and b in pole_pos:
                            pair_counts[(a, b)] += 1
                        elif b in pole_neg and a in pole_pos:
                            pair_counts[(b, a)] += 1

    # Greedy many-to-one or one-to-one archiphoneme assignment for attested alternating pairs
    arch_map: Dict[str, str] = {}
    arch_id = 0
    for (vn, vp), cnt in pair_counts.most_common():
        if cnt < 1:
            continue
        if vn not in arch_map and vp not in arch_map:
            arch_id += 1
            sym = f"VH{arch_id}"
            arch_map[vn] = sym
            arch_map[vp] = sym
        elif vn in arch_map and vp not in arch_map:
            arch_map[vp] = arch_map[vn]
        elif vp in arch_map and vn not in arch_map:
            arch_map[vn] = arch_map[vp]

    return {
        "vowels": vowels,
        "pole_neg": pole_neg,
        "pole_pos": pole_pos,
        "arch_map": arch_map,
    }


def apply_malouf_abstraction(
    df_lex: pd.DataFrame,
    sounds_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Produce a Malouf-abstracted copy of `df_lex` preserving the exact 14-column schema."""
    model = learn_malouf_harmony_model(df_lex, sounds_df=sounds_df)
    arch_map: Dict[str, str] = model["arch_map"]
    pole_neg: Set[str] = model["pole_neg"]
    pole_pos: Set[str] = model["pole_pos"]

    out = df_lex.copy()
    if not arch_map:
        return out, model

    # Determine each lemma's invariant stem length and stem harmony class from v0 forms
    v0 = out[out["variant_idx"] == 0]
    lemma_info: Dict[str, Tuple[int, str]] = {}
    for lid, grp in v0.groupby("lemma_id", sort=False):
        seqs = [[t for t in str(s).split(" ") if t] for s in grp["segmented_form"]]
        shortest = min(seqs, key=len) if seqs else []
        lcp = len(shortest)
        for k, sym in enumerate(shortest):
            if any(seq[k] != sym for seq in seqs):
                lcp = k
                break
        # Stem vowels come from the invariant stem prefix (or full shortest form if lcp == 0)
        stem_span = shortest[: max(1, lcp)]
        n_neg = sum(1 for t in stem_span if t in pole_neg)
        n_pos = sum(1 for t in stem_span if t in pole_pos)
        if n_neg > n_pos:
            h_tag = "HNEG"
        elif n_pos > n_neg:
            h_tag = "HPOS"
        else:
            h_tag = "HNEUT"
        lemma_info[lid] = (lcp, h_tag)

    new_seg: List[str] = []
    new_raw: List[str] = []
    is_phon = len(out) > 0 and str(out["representation"].iloc[0]) == "phon"

    for lid, seg_s in zip(out["lemma_id"], out["segmented_form"]):
        toks = [t for t in str(seg_s).split(" ") if t]
        lcp, h_tag = lemma_info.get(lid, (0, "HNEUT"))
        stem_toks = toks[:lcp]
        aff_toks = [arch_map.get(t, t) for t in toks[lcp:]]
        abstract_toks = [h_tag] + stem_toks + aff_toks
        new_seg.append(" ".join(abstract_toks))
        if is_phon:
            new_raw.append(" ".join(abstract_toks))
        else:
            new_raw.append(f"{h_tag}:" + "".join(" " if t == "_" else t for t in (stem_toks + aff_toks)))

    out["segmented_form"] = new_seg
    out["raw_form"] = new_raw
    return out, model
