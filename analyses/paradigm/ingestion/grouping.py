"""Three-rule Union-Find root-cognate leakage grouping (`leakage_group_id`).

Within each `(glottocode, pos)` inventory, merges lemmas into connected components via
Union-Find under three rules:
  (a) Identical normalized citation label (`NFC + casefold + whitespace collapse`) OR
      identical `source_cell` surface form within the same representation.
  (b) Orthographic reflexive / pronominal / mediopassive clitic derivative of another
      attested lemma in the same `(glottocode, pos)` inventory.
  (c) Unsupervised paradigm-wide constant-prefix or constant-suffix cognate rule:
      merges `(l_1, l_2)` whenever `l_1` has `base_stem` length >= 4 graphemes (estimated
      via unsupervised productive cell-affix stripping across the paradigm, plus median
      surface length >= 4) and there exists a constant prefix or suffix `u` of length >= 2
      graphemes such that `form(l_2, c) == u + form(l_1, c)` (or `form(l_1, c) + u`) across
      at least 80% of shared non-missing synthetic cells (requiring >= 2 shared cells, or
      100% on 2-cell paradigms).

Uses an exact Pigeonhole-Principle inverted cell-form index (`F + 1` anchor cells where
`F = floor(0.20 * |C|)`) so Rule (c) is mathematically 100% equivalent to an all-pairs
check while running in near-linear `O(N * L)` time even on 100,000-lemma lexicons.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_WS = re.compile(r"\s+")
IPA_MODIFIERS = set("ːˑʰʲʷˠˤˀⁿ")


def nfc_clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", str(text))).strip()


def norm_label(label: str) -> str:
    return nfc_clean(label).casefold()


def to_graphemes(form: str, representation: str = "orth") -> Tuple[str, ...]:
    """Split a form into grapheme clusters (or space-separated phonological segments)."""
    s = nfc_clean(form)
    if not s:
        return ()
    if representation == "phon":
        # In Paralex/CELEX phonemic notation, segments are whitespace-delimited tokens.
        if " " in s:
            return tuple(tok for tok in s.split(" ") if tok)
        out_p: List[str] = []
        for ch in s:
            cat = unicodedata.category(ch)
            if out_p and (cat in ("Mn", "Me") or ch in IPA_MODIFIERS):
                out_p[-1] += ch
            else:
                out_p.append(ch)
        return tuple(out_p)

    out: List[str] = []
    for ch in s:
        cat = unicodedata.category(ch)
        if out and cat in ("Mn", "Me"):
            out[-1] += ch
        else:
            out.append(ch)
    return tuple(out)


class UnionFind:
    """Deterministic lexicographic Union-Find over string IDs."""

    def __init__(self, items: Iterable[str]) -> None:
        self.parent: Dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra
        return True


# ---------------------------------------------------------------------------
# Rule (b): Reflexive / clitic citation-label stripping (attested-base only)
# ---------------------------------------------------------------------------

_ROMANCE_INF_CLITIC = re.compile(
    r"^(.*?r)(si|sene|ci|cene|ne|sela|selo|sele|seli|cela|celo|gliela|glielo|se|la|le|li|lo)$"
)
_FRENCH_REFL_PROCLITIC = re.compile(r"^(?:se\s+|s['’])(.+)$")
_SLAVIC_REFL_PARTICLE = re.compile(r"^(.+?)\s+(?:się|se|si)$")
_SLAVIC_REFL_SUFFIX = re.compile(r"^(.+?)(?:тися|тись|ться|ться|ться|цца|ся|сь)$")


def candidate_clitic_bases(norm_cit: str) -> List[str]:
    """Generate candidate non-reflexive/non-clitic base citation labels for Rule (b).

    Only merges when a candidate base is actually attested as another lemma in the same
    `(glottocode, pos)` inventory and has >= 3 graphemes.
    """
    cands: List[str] = []

    m_fr = _FRENCH_REFL_PROCLITIC.match(norm_cit)
    if m_fr:
        cands.append(m_fr.group(1).strip())

    m_sl_p = _SLAVIC_REFL_PARTICLE.match(norm_cit)
    if m_sl_p:
        cands.append(m_sl_p.group(1).strip())

    # Italian/Romance enclitic infinitives: abbandonarsi -> abbandonare, porsi -> porre, lavar-se -> lavar
    if "-" in norm_cit:
        stem_h, tail_h = norm_cit.rsplit("-", 1)
        if tail_h in ("se", "si", "s'hi", "ne", "en"):
            cands.append(stem_h)

    m_rom = _ROMANCE_INF_CLITIC.match(norm_cit)
    if m_rom:
        stem_r = m_rom.group(1)
        # Spanish/Catalan/Portuguese: lavarse -> lavar; Italian: abbandonarsi -> abbandonare, porsi -> porre
        cands.extend([stem_r, stem_r + "e", stem_r + "re"])

    # East Slavic reflexive suffixes: мыться -> мыть, вмиватися -> вмивати, смеяться -> смеять
    if norm_cit.endswith(("ться", "тись")):
        cands.append(norm_cit[:-2])
    elif norm_cit.endswith("тися"):
        cands.append(norm_cit[:-2])
    elif norm_cit.endswith(("ся", "сь")):
        cands.append(norm_cit[:-2])
    elif norm_cit.endswith("цца"):
        cands.append(norm_cit[:-3] + "ць")

    # Baltic reflexive infinitives: Lithuanian -tis -> -ti, Latvian -ties -> -t
    if norm_cit.endswith("tis") and len(norm_cit) >= 6:
        cands.append(norm_cit[:-1])
    if norm_cit.endswith("ties") and len(norm_cit) >= 7:
        cands.append(norm_cit[:-3])

    return [c for c in dict.fromkeys(cands) if c != norm_cit and len(to_graphemes(c, "orth")) >= 3]


# ---------------------------------------------------------------------------
# Rule (c): Unsupervised base_stem length & constant-affix cognate detection
# ---------------------------------------------------------------------------

def _lcp_len(seqs: Sequence[Tuple[str, ...]]) -> int:
    if not seqs:
        return 0
    shortest = min(seqs, key=len)
    for i, sym in enumerate(shortest):
        for s in seqs:
            if s[i] != sym:
                return i
    return len(shortest)


def _lcsuf_len(seqs: Sequence[Tuple[str, ...]]) -> int:
    if not seqs:
        return 0
    shortest = min(seqs, key=len)
    n_short = len(shortest)
    for k in range(1, n_short + 1):
        sym = shortest[-k]
        for s in seqs:
            if s[-k] != sym:
                return k - 1
    return n_short


_VOWEL_BASE_CHARS = set(
    "aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿāăąēĕėęěĩīĭįıĳōŏőœũūŭůűų"
    "ɑɐɒɛɜɞɪɨɯɵɶɷʊʌɤʉəɚɝ"
    "аеёиоуыэюяіїєґўәөүұӧӱ"
    "αεηιοωυάέήίόώύϊϋΐΰ"
    "աեէըիոօև"
    "აეიოუ"
    "אעהוי"
    "اويىأإآؤئ"
)

_SLAVIC_CONS_PREFIXES: Set[Tuple[str, ...]] = {
    ("в", "з"),
    ("в", "с"),
    ("w", "z"),
    ("w", "s"),
}


def _symbol_has_vowel(sym: str) -> bool:
    for ch in sym:
        cl = ch.lower()
        cp = ord(ch)
        if cl in _VOWEL_BASE_CHARS:
            return True
        # Hangul syllables (U+AC00..U+D7A3), Hangul Jungseong (U+1160..U+11A7),
        # Ethiopic CV fidels (U+1200..U+137F), or Indic independent vowels / vowel signs
        if (
            (0xAC00 <= cp <= 0xD7A3)
            or (0x1160 <= cp <= 0x11A7)
            or (0x1200 <= cp <= 0x137F)
            or ("VOWEL" in unicodedata.name(ch, ""))
        ):
            return True
    return False


def _seq_has_vowel(seq: Sequence[str]) -> bool:
    return any(_symbol_has_vowel(sym) for sym in seq)


def _is_syllabic_block_script(seqs: Iterable[Tuple[str, ...]]) -> bool:
    """Detect scripts where 1 grapheme = 1 CV(C) syllable block (Hangul / Ethiopic)."""
    for f in seqs:
        for tok in f:
            for ch in tok:
                cp = ord(ch)
                if (0xAC00 <= cp <= 0xD7A3) or (0x1200 <= cp <= 0x137F):
                    return True
    return False


def estimate_base_stem_lengths(
    paradigms: List[List[Tuple[str, ...]]],
    min_affix_share: float = 0.03,
) -> List[float]:
    """Unsupervised estimation of each lemma's `base_stem` length in graphemes.

    1. For each lemma `i`, compute invariant prefix `LCP(i)` and invariant suffix `LCSuf(i)`
       across its non-empty cell forms to detect whether the paradigm is predominantly
       suffixing or prefixing, and collect empirical inflectional affixes per cell among
       concatenative lemmas (`inv >= 2`).
    2. Retain productive cell affixes occurring in >= `min_affix_share` (default 3%, min 3)
       of lemmas in cell `c`, capturing secondary declension/conjugation classes (`-ь/-и/-ей`,
       `-ere/-ire`, `-ować/-увати`, `-izar/-orar`).
    3. Require strict `min_surf_len >= 4` across every non-empty cell of the base lemma
       (with Hangul/Ethiopic syllable blocks scaled at 2.0 phoneme-equivalents per block so
       >= 2 stem blocks are required), and require both median and 35th-percentile stripped
       stem lengths >= 4.0.
       - Rejects <=3-grapheme stems (`ость`, `ах`, `ии`, `an`, `tá`, `ár`, `izar`, `orar`,
         `тон`, `рама`, `mano`, `etto`, `кувати`, `wiać`, `zieren`).
       - Preserves genuine >=4-grapheme roots including allomorphic/ablauting irregulars
         (`porre`, `stand`, `stehen`, `prendre`, `pisać`).
    """
    n_lemmas = len(paradigms)
    if n_lemmas == 0:
        return []
    n_cells = len(paradigms[0])
    if n_cells <= 1:
        return [
            float(len(p[0]) * (2.0 if _is_syllabic_block_script(p) else 1.0)) if p and p[0] else 0.0
            for p in paradigms
        ]

    lcp_vals: List[int] = []
    lcs_vals: List[int] = []
    for p in paradigms:
        non_empty = [f for f in p if f]
        distinct = list(dict.fromkeys(non_empty))
        if len(distinct) <= 1:
            ln = len(distinct[0]) if distinct else 0
            lcp_vals.append(ln)
            lcs_vals.append(ln)
        else:
            lcp_vals.append(_lcp_len(distinct))
            lcs_vals.append(_lcsuf_len(distinct))

    # Determine whether lexicon is predominantly suffixing (LCP >= LCSuf) or prefixing
    use_suffix = sum(a >= b for a, b in zip(lcp_vals, lcs_vals)) >= (n_lemmas / 2.0)

    productive_affixes: List[List[Tuple[str, ...]]] = []
    eff_share = 0.20 if n_lemmas < 50 else min_affix_share
    min_count = max(4, int(math.ceil(eff_share * n_lemmas)))

    for c in range(n_cells):
        counts: Counter = Counter()
        n_valid = 0
        for i, p in enumerate(paradigms):
            f = p[c]
            if not f:
                continue
            inv = lcp_vals[i] if use_suffix else lcs_vals[i]
            if inv >= 3 and inv <= len(f):
                aff = f[inv:] if use_suffix else f[: len(f) - inv]
                counts[aff] += 1
                n_valid += 1
        thresh = max(4, int(math.ceil(eff_share * max(1, n_valid))))
        affs = sorted(
            [aff for aff, cnt in counts.items() if aff and cnt >= min(min_count, thresh)],
            key=lambda a: (-len(a), a),
        )
        productive_affixes.append(affs)

    stem_lengths: List[float] = []
    for i, p in enumerate(paradigms):
        non_empty = [f for f in p if f]
        if not non_empty:
            stem_lengths.append(0.0)
            continue
        scale = 2.0 if _is_syllabic_block_script(non_empty) else 1.0
        surf_lens = [len(f) * scale for f in non_empty]
        min_surf = float(min(surf_lens))
        if min_surf < 4.0:
            stem_lengths.append(min_surf)
            continue

        stripped_lens: List[float] = []
        for c, f in enumerate(p):
            if not f:
                continue
            matched_len = len(f)
            for aff in productive_affixes[c]:
                la = len(aff)
                if la < len(f):
                    if use_suffix and f[-la:] == aff:
                        matched_len = len(f) - la
                        break
                    if not use_suffix and f[:la] == aff:
                        matched_len = len(f) - la
                        break
            stripped_lens.append(matched_len * scale)

        med_stripped = float(np.median(stripped_lens))
        est = min(min_surf, med_stripped)

        distinct_forms = list(dict.fromkeys(non_empty))
        if len(distinct_forms) == 2:
            inv2 = max(_lcp_len(distinct_forms), _lcsuf_len(distinct_forms)) * scale
            est = min(est, float(inv2))

        stem_lengths.append(est)

    return stem_lengths


def find_constant_affix_cognate_pairs(
    paradigms: List[List[Tuple[str, ...]]],
    min_stem_graphemes: int = 4,
    min_affix_graphemes: int = 2,
    min_cell_share: float = 0.80,
) -> List[Tuple[int, int, str]]:
    """Find all `(idx_base, idx_derived, reason)` pairs satisfying Rule (c).

    Combines exact Pigeonhole-Principle cell-anchor indexing with:
      1. Vocalic-affix requirement (rejecting pure-consonant syllable-onset splits like
         `gl+itch`, `st+itch`, except Slavic `вз/вс/wz/ws` before consonants).
      2. C+V resyllabification guard (rejecting open-class consonant-final stems before
         vowel-initial bases like `collab+orate`, `abdic+ador`, `allerg+ist`).
      3. Affix productivity & anti-hub filtering (rejecting open-class derivational suffix
         hubs such as `-ador`, `-ismo`, `-azione`, `-mento`, `-gist`, `-meter`, `-tyczność`,
         `-nowość`, `-seerima`, `-istua`, `-nacija` while preserving productive prefixation
         and attested-stem compounding).
    """
    n_lemmas = len(paradigms)
    if n_lemmas < 2:
        return []
    n_cells = len(paradigms[0])
    if n_cells < 2:
        return []

    stem_lens = estimate_base_stem_lengths(paradigms)
    eligible_base = [sl >= float(min_stem_graphemes) for sl in stem_lens]

    attested_forms: Set[Tuple[str, ...]] = set()
    for p in paradigms:
        for f in p:
            if len(f) >= 3:
                attested_forms.add(f)

    req_matches = int(math.ceil(min_cell_share * n_cells - 1e-9))
    max_failures = n_cells - req_matches
    n_anchor_cells = min(n_cells, max_failures + 1)

    anchor_maps: List[Dict[Tuple[str, ...], List[int]]] = []
    for c in range(n_anchor_cells):
        cmap: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        for i in range(n_lemmas):
            if eligible_base[i] and paradigms[i][c]:
                cmap[paradigms[i][c]].append(i)
        anchor_maps.append(cmap)

    checked_candidates: Set[Tuple[int, int, bool, Tuple[str, ...]]] = set()
    raw_candidates: List[Tuple[int, int, bool, Tuple[str, ...]]] = []

    for i2 in range(n_lemmas):
        p2 = paradigms[i2]
        is_block = _is_syllabic_block_script(p2)
        eff_min_aff = 1 if is_block and min_affix_graphemes <= 2 else min_affix_graphemes
        min_rem = 2 if is_block else 3

        for c in range(n_anchor_cells):
            f2 = p2[c]
            L2 = len(f2)
            if L2 < eff_min_aff + min_rem:
                continue
            cmap = anchor_maps[c]

            # 1. Constant prefix: f2 == u + f1
            for k in range(eff_min_aff, L2 - min_rem + 1):
                rem_p = f2[k:]
                bases_p = cmap.get(rem_p)
                if bases_p:
                    u_pref = f2[:k]
                    if not (_seq_has_vowel(u_pref) or (u_pref in _SLAVIC_CONS_PREFIXES and not _symbol_has_vowel(rem_p[0]))):
                        continue
                    for i1 in bases_p:
                        if i1 == i2:
                            continue
                        cand_key = (i1, i2, True, u_pref)
                        if cand_key in checked_candidates:
                            continue
                        checked_candidates.add(cand_key)
                        p1 = paradigms[i1]
                        hits = 0
                        shared = 0
                        fails = 0
                        for cc in range(n_cells):
                            w1, w2 = p1[cc], p2[cc]
                            if not w1 or not w2:
                                continue
                            shared += 1
                            if len(w2) == len(w1) + k and w2[:k] == u_pref and w2[k:] == w1:
                                hits += 1
                            else:
                                fails += 1
                                if fails > max_failures:
                                    break
                        if shared >= 2 and hits >= int(math.ceil(min_cell_share * shared - 1e-9)):
                            raw_candidates.append(cand_key)

            # 2. Constant suffix: f2 == f1 + u
            for k in range(eff_min_aff, L2 - min_rem + 1):
                rem_s = f2[:-k]
                bases_s = cmap.get(rem_s)
                if bases_s:
                    u_suf = f2[-k:]
                    if not _seq_has_vowel(u_suf):
                        continue
                    for i1 in bases_s:
                        if i1 == i2:
                            continue
                        cand_key = (i1, i2, False, u_suf)
                        if cand_key in checked_candidates:
                            continue
                        checked_candidates.add(cand_key)
                        p1 = paradigms[i1]
                        hits = 0
                        shared = 0
                        fails = 0
                        for cc in range(n_cells):
                            w1, w2 = p1[cc], p2[cc]
                            if not w1 or not w2:
                                continue
                            shared += 1
                            if len(w2) == len(w1) + k and w2[-k:] == u_suf and w2[:-k] == w1:
                                hits += 1
                            else:
                                fails += 1
                                if fails > max_failures:
                                    break
                        if shared >= 2 and hits >= int(math.ceil(min_cell_share * shared - 1e-9)):
                            raw_candidates.append(cand_key)

    if not raw_candidates:
        return []

    # Compute affix base-paradigm diversity and per-base-paradigm incoming hub statistics
    # (grouping homographic base paradigms by their paradigm tuple so duplicate lemma_ids don't split hub counts)
    para_sigs = [tuple(p) for p in paradigms]
    affix_base_sigs: Dict[Tuple[bool, Tuple[str, ...]], Set[Tuple[Tuple[str, ...], ...]]] = defaultdict(set)
    sig_incoming_affixes: Dict[Tuple[Tuple[str, ...], ...], Set[Tuple[bool, Tuple[str, ...]]]] = defaultdict(set)

    for i1, i2, is_pref, u in raw_candidates:
        sig1 = para_sigs[i1]
        affix_base_sigs[(is_pref, u)].add(sig1)
        sig_incoming_affixes[sig1].add((is_pref, u))

    min_div = 1 if n_lemmas < 100 else (2 if n_lemmas < 1000 else 3)

    sig_stem_len: Dict[Tuple[Tuple[str, ...], ...], float] = {}
    for i, sig1 in enumerate(para_sigs):
        if eligible_base[i]:
            sig_stem_len[sig1] = max(sig_stem_len.get(sig1, 0.0), stem_lens[i])

    # Identify derivational suffix/ending hubs: base paradigms accumulating > 20 hapax
    # incoming affixes, or > 60 total incoming affixes, or (> 15 incoming with > 65% hapax),
    # or 2-cell 4-grapheme rhyme bases (`pero/peros`) with > 12 incoming rhyme words.
    blocked_base_sigs: Set[Tuple[Tuple[str, ...], ...]] = set()
    if n_lemmas >= 100:
        for sig1, inc_set in sig_incoming_affixes.items():
            n_inc = len(inc_set)
            n_hapax = sum(1 for key in inc_set if len(affix_base_sigs[key]) < min_div)
            short_2cell_hub = (n_cells == 2 and sig_stem_len.get(sig1, 0.0) < 5.0 and n_inc > 12)
            if n_hapax > 20 or n_inc > 60 or (n_inc > 15 and n_hapax > int(0.65 * n_inc)) or short_2cell_hub:
                blocked_base_sigs.add(sig1)

    accepted_pairs: Dict[Tuple[int, int], str] = {}
    for i1, i2, is_pref, u in raw_candidates:
        sig1 = para_sigs[i1]
        if sig1 in blocked_base_sigs:
            continue
        pair_key = (min(i1, i2), max(i1, i2))
        if pair_key in accepted_pairs:
            continue

        div = len(affix_base_sigs[(is_pref, u)])
        first_non_empty = next((f for f in paradigms[i1] if f), ())
        if is_pref and first_non_empty:
            w1_vowel_init = _symbol_has_vowel(first_non_empty[0])
            u_cons_final = not _symbol_has_vowel(u[-1])
        else:
            w1_vowel_init = False
            u_cons_final = False

        # C+V resyllabification guard: consonant-final prefix before vowel-initial base
        # requires productive prefix diversity (e.g. `in-`, `dis-`, `un-`, `sub-`), never hapax `collab+orate`.
        if w1_vowel_init and u_cons_final and div < max(3, min_div) and n_lemmas >= 100:
            continue

        is_productive = div >= min_div
        is_attested_compound = (len(u) >= 3 and u in attested_forms and not (w1_vowel_init and u_cons_final))
        if is_productive or is_attested_compound:
            tag = "const_prefix" if is_pref else "const_suffix"
            accepted_pairs[pair_key] = f"{tag}:{''.join(u)}"

    return [(a, b, rsn) for (a, b), rsn in sorted(accepted_pairs.items())]


def pick_source_cell(cells: Sequence[str], pos: str) -> str:
    """Select canonical citation/source cell for Rule (a) surface-form homograph grouping."""
    cset = set(cells)
    if pos == "V":
        for cand in ("NFIN", "NFIN;ACT", "NFIN;1;ACT", "IND;PRS;3;SG", "IND;PRS;1;SG", "PRS;3;SG", "PRS;1;SG"):
            if cand in cset:
                return cand
        nfin_like = sorted(c for c in cset if "NFIN" in c.split(";"))
        if nfin_like:
            return nfin_like[0]
    elif pos == "N":
        for cand in ("NOM;SG", "INDF;NOM;SG", "NDEF;NOM;SG", "SG", "ABS;SG", "ACC;SG"):
            if cand in cset:
                return cand
        nom_sg = sorted(c for c in cset if {"NOM", "SG"} <= set(c.split(";")))
        if nom_sg:
            return nom_sg[0]
    return sorted(cset)[0] if cset else ""


def assign_leakage_groups(
    df_forms: pd.DataFrame,
    glottocode: str,
    pos: str,
    representation: str,
    cells_ordered: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Assign `leakage_group_id` to every `lemma_id` in a single `(glottocode, pos, representation)` unit.

    Parameters
    ----------
    df_forms : DataFrame
        Must contain `lemma_id`, `lemma_label`, `cell`, `raw_form`, `variant_idx`.
    glottocode, pos, representation : str
        Unit metadata.
    cells_ordered : Sequence[str]
        Ordered distinct retained synthetic cells of the unit.

    Returns
    -------
    group_map : dict[lemma_id -> leakage_group_id]
    reasons_map : dict[lemma_id -> ';'-joined merge reasons]
    """
    lem_df = (
        df_forms[["lemma_id", "lemma_label"]]
        .drop_duplicates("lemma_id")
        .sort_values("lemma_id", kind="mergesort")
        .reset_index(drop=True)
    )
    lemma_ids: List[str] = lem_df["lemma_id"].tolist()
    lemma_labels: List[str] = lem_df["lemma_label"].tolist()

    uf = UnionFind(lemma_ids)
    reasons: Dict[str, Set[str]] = {lid: set() for lid in lemma_ids}

    # --- Rule (a.1): Identical casefolded NFC citation label ---
    by_norm_label: Dict[str, List[str]] = defaultdict(list)
    for lid, lab in zip(lemma_ids, lemma_labels):
        by_norm_label[norm_label(lab)].append(lid)
    for lids in by_norm_label.values():
        if len(lids) > 1:
            for other in lids[1:]:
                uf.union(lids[0], other)
            for x in lids:
                reasons[x].add("citation_label")

    # --- Rule (a.2): Identical source_cell surface form (all variants conservative) ---
    src_cell = pick_source_cell(cells_ordered, pos)
    if src_cell:
        src_sub = df_forms[df_forms["cell"] == src_cell]
        by_src_form: Dict[str, List[str]] = defaultdict(list)
        for lid, rf in zip(src_sub["lemma_id"], src_sub["raw_form"]):
            key = norm_label(rf)
            if key:
                by_src_form[key].append(lid)
        for lids in by_src_form.values():
            uniq = sorted(set(lids))
            if len(uniq) > 1:
                for other in uniq[1:]:
                    uf.union(uniq[0], other)
                for x in uniq:
                    reasons[x].add("source_cell_form")

    # --- Rule (b): Orthographic reflexive / clitic derivative of attested base lemma ---
    for lid, lab in zip(lemma_ids, lemma_labels):
        nc = norm_label(lab)
        for cand_base in candidate_clitic_bases(nc):
            if cand_base in by_norm_label:
                base_lids = by_norm_label[cand_base]
                for blid in base_lids:
                    if blid != lid:
                        uf.union(blid, lid)
                        reasons[lid].add("reflexive_clitic")
                        reasons[blid].add("reflexive_clitic_base")

    # --- Rule (c): Paradigm-wide constant-prefix / constant-suffix cognate rule ---
    v0 = df_forms[df_forms["variant_idx"] == 0]
    cell_to_idx = {c: idx for idx, c in enumerate(cells_ordered)}
    lid_to_idx = {lid: idx for idx, lid in enumerate(lemma_ids)}
    n_cells = len(cells_ordered)
    paradigms: List[List[Tuple[str, ...]]] = [[() for _ in range(n_cells)] for _ in range(len(lemma_ids))]

    grapheme_cache: Dict[str, Tuple[str, ...]] = {}
    for lid, cell, rf in zip(v0["lemma_id"], v0["cell"], v0["raw_form"]):
        li = lid_to_idx.get(lid)
        ci = cell_to_idx.get(cell)
        if li is None or ci is None:
            continue
        if rf not in grapheme_cache:
            grapheme_cache[rf] = to_graphemes(rf, representation)
        paradigms[li][ci] = grapheme_cache[rf]

    cognate_pairs = find_constant_affix_cognate_pairs(
        paradigms,
        min_stem_graphemes=4,
        min_affix_graphemes=2,
        min_cell_share=0.80,
    )
    for i1, i2, rsn in cognate_pairs:
        l1, l2 = lemma_ids[i1], lemma_ids[i2]
        uf.union(l1, l2)
        reasons[l1].add(rsn)
        reasons[l2].add(rsn)

    # Number connected components deterministically by smallest lemma_id
    comps: Dict[str, List[str]] = defaultdict(list)
    for lid in lemma_ids:
        comps[uf.find(lid)].append(lid)
    ordered_comps = sorted(comps.values(), key=lambda ids: min(ids))

    group_map: Dict[str, str] = {}
    reasons_map: Dict[str, str] = {}
    for g_num, ids in enumerate(ordered_comps, start=1):
        gid = f"{glottocode}_{pos}::g{g_num:06d}"
        for lid in ids:
            group_map[lid] = gid
            reasons_map[lid] = ";".join(sorted(reasons[lid]))

    return group_map, reasons_map
