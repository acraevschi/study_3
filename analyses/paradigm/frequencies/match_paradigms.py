#!/usr/bin/env python3
"""Match open-corpus wordform frequencies onto Track 2 morphological paradigms.

Reads extracted paradigm tables (`analyses/paradigm/data/lexicons/{unit_id}.csv`), matches
NFC-normalized casefolded surface wordforms across:
  - Tier A: Universal Dependencies v2.15 `(lemma, upos, form)` token counts
  - Tier B: Leipzig Corpora Collection (`*-words.txt`) surface wordform token counts
  - Tier C: `wordfreq` v3.1.1 multi-register smoothed probabilities (supplementing unattested
    cells in finite news/wiki samples where attested in Speer 2022)

Cross-lemma homographs (where surface string `w` belongs to >= 2 lemmas in `{unit_id}.csv`)
are disambiguated in two stages:
  1. Exact UD v2.15 `(lemma, target_upos, form)` empirical ratios when attested in UD;
  2. 5 iterations of proportional Expectation-Maximization (`p(l | w) proportional to F(l)`
     initialized uniformly across matching lemmas) for all remaining homographs, guaranteeing
     exact conservation of matched token mass with zero NaN/Inf values.

For every orthographic unit with `>= 200` attested leakage groups (`freq_lemma_tokens > 0`),
selects the top `min(500, n_attested_groups)` leakage groups and generates deterministic
SHA-256-seeded CV split manifests `analyses/paradigm/data/splits_freq/{unit_id}_topfreq120.csv`
(`R=5` draws x `K=5` folds of 150 groups: 120 train [96 subtrain + 24 dev] / 30 test),
mirroring `data-ingest`'s split column schema 1-to-1.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import unicodedata

import numpy as np
import pandas as pd

from analyses.paradigm.frequencies.fetch_frequencies import (
    FREQ_SOURCES_DIR,
    ISO_TO_WORDFREQ,
    PROVENANCE_CSV,
    REPO_ROOT,
    corpus_aliases_for_iso,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

LEXICONS_DIR = REPO_ROOT / "analyses" / "paradigm" / "data" / "lexicons"
SPLITS_DIR = REPO_ROOT / "analyses" / "paradigm" / "data" / "splits"
FREQ_OUT_DIR = REPO_ROOT / "analyses" / "paradigm" / "data" / "frequencies"
SPLITS_FREQ_DIR = REPO_ROOT / "analyses" / "paradigm" / "data" / "splits_freq"
UNIT_SUMMARY_CSV = REPO_ROOT / "analyses" / "paradigm" / "data" / "unit_summary.csv"

FREQ_UNIT_COLUMNS = [
    "unit_id",
    "lemma_id",
    "leakage_group_id",
    "freq_lemma_tokens",
    "freq_lemma_pmw",
    "log10_freq_lemma",
    "n_cells_attested",
    "primary_freq_source",
]

DEFAULT_SPLIT_COLUMNS = [
    "unit_id",
    "regime",
    "draw_idx",
    "fold_idx",
    "lemma_id",
    "leakage_group_id",
    "role",
    "inner_role",
]

TURKIC_DOTTED_ISOS = {"tur", "azj", "aze", "crh", "gag", "tat", "kaz"}
CYRILLIC_STRESS_ISOS = {"bel", "rus", "ukr", "bul"}
HBS_PITCH_ISOS = {"hbs", "hrv", "srp", "bos"}
ABJAD_VOCALIZATION_ISOS = {
    "arb",
    "ara",
    "heb",
    "pes",
    "fas",
    "urd",
    "pus",
    "pbt",
    "pst",
    "ckb",
    "arz",
    "afb",
}

_PUNCT_ONLY_RE = re.compile(r"^[\W\d_]+$", flags=re.UNICODE)
_ABJAD_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u0591-\u05c7]")
_APOSTROPHES_RE = re.compile(r"[\u2019\u2018\u02bc\u02bb`]")
_CYRILLIC_STRESS_RE = re.compile(r"[\u0300\u0301]")
_LIT_PITCH_RE = re.compile(r"[\u0300\u0301\u0303]")
_HBS_PITCH_RE = re.compile(r"[\u0300\u0302\u0304\u030f\u0311]")
_HBS_VOWEL_ACUTE_RE = re.compile(r"([aeiouAEIOUrR])\u0301")


def normalize_surface(text: str, iso: str = "", strip_abjad_marks: bool = False) -> str:
    """NFC-normalize, strip non-orthographic dictionary stress/pitch accents, and casefold."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text).strip())
    # Remove invisible formatting characters (Arabic tatweel, soft hyphen, zero-width space)
    s = s.replace("\u0640", "").replace("\u00ad", "").replace("\u200b", "")
    s = _APOSTROPHES_RE.sub("'", s)
    if iso in TURKIC_DOTTED_ISOS:
        s = s.replace("İ", "i").replace("I", "ı")
    s = s.casefold()
    if iso in CYRILLIC_STRESS_ISOS:
        # Strip phonological stress (U+0300/U+0301) while keeping й/ў (U+0306) and ё/ї (U+0308)
        s = _CYRILLIC_STRESS_RE.sub("", unicodedata.normalize("NFD", s))
    elif iso == "lit":
        # Strip Baltic pitch accents (U+0300/U+0301/U+0303) while keeping ąęįų (U+0328), ė (U+0307), ū (U+0304), čšž (U+030C)
        s = _LIT_PITCH_RE.sub("", unicodedata.normalize("NFD", s))
    elif iso in HBS_PITCH_ISOS:
        # Strip Gajica dictionary pitch/length accents while strictly preserving ć (c+U+0301) and čšž (U+030C)
        s_nfd = _HBS_PITCH_RE.sub("", unicodedata.normalize("NFD", s))
        s = _HBS_VOWEL_ACUTE_RE.sub(r"\1", s_nfd)
    # Clean up Python's 'İ'.casefold() -> 'i\u0307' and Lithuanian 'i\u0307\u0303' -> 'i\u0307'
    s = unicodedata.normalize("NFC", s).replace("i\u0307", "i")
    s = unicodedata.normalize("NFC", s)
    if strip_abjad_marks:
        s = unicodedata.normalize("NFC", _ABJAD_DIACRITICS_RE.sub("", s))
    return s


def map_ud_upos_to_pos(upos: str) -> Optional[str]:
    """Map UD UPOS tag to Track 2 POS ('V' includes AUX so copulas/auxiliaries keep full counts)."""
    if upos in {"VERB", "AUX"}:
        return "V"
    if upos == "NOUN":
        return "N"
    return None


def dominant_alpha_script(texts: Iterable[str]) -> str:
    """Return majority Unicode alphabetic script prefix (e.g. LATIN, CYRILLIC, ARABIC)."""
    counts: Counter = Counter()
    for txt in texts:
        for ch in str(txt):
            if ch.isalpha():
                nm = unicodedata.name(ch, "")
                if nm:
                    counts[nm.split()[0]] += 1
    return counts.most_common(1)[0][0] if counts else "UNKNOWN"


def bundle_supported_scripts(bundle: "IsoFrequencyBundle", min_share: float = 0.05) -> Set[str]:
    """Return Unicode alphabetic scripts accounting for >= min_share of corpus token mass."""
    sc_mass: Counter = Counter()
    for w, cnt in bundle.lcc_surface.most_common(5000):
        sc = dominant_alpha_script([w])
        if sc != "UNKNOWN":
            sc_mass[sc] += cnt
    for w, cnt in bundle.ud_surface.most_common(5000):
        sc = dominant_alpha_script([w])
        if sc != "UNKNOWN":
            sc_mass[sc] += cnt
    if not sc_mass and bundle.wf_probs:
        for w, prob in list(bundle.wf_probs.items())[:5000]:
            sc = dominant_alpha_script([w])
            if sc != "UNKNOWN":
                sc_mass[sc] += prob * 1_000_000.0
    total = sum(sc_mass.values())
    if total <= 0:
        return set()
    return {sc for sc, val in sc_mass.items() if (val / total) >= min_share}


class IsoFrequencyBundle:
    """Pre-loaded corpus token tables for one language ISO."""

    def __init__(
        self,
        iso: str,
        lcc_surface: Counter,
        lcc_total_tokens: int,
        ud_surface: Counter,
        ud_lemma_pos: Counter,
        ud_total_tokens: int,
        wf_lang: Optional[str],
        source_ids: List[str],
        wf_probs: Optional[Dict[str, float]] = None,
        ud_pos_surface: Optional[Counter] = None,
        n_script_filtered_sources: int = 0,
    ) -> None:
        self.iso = iso
        self.lcc_surface = lcc_surface
        self.lcc_total_tokens = int(lcc_total_tokens)
        self.ud_surface = ud_surface
        self.ud_lemma_pos = ud_lemma_pos  # keyed by (pos, lemma_norm, form_norm)
        self.ud_total_tokens = int(ud_total_tokens)
        self.wf_lang = wf_lang
        self.source_ids = source_ids
        self.wf_probs: Dict[str, float] = wf_probs or {}
        self.n_script_filtered_sources = int(n_script_filtered_sources)
        if ud_pos_surface is not None:
            self.ud_pos_surface = ud_pos_surface
        else:
            derived: Counter = Counter()
            for (p_tag, _lem, f_norm), cnt in ud_lemma_pos.items():
                derived[(p_tag, f_norm)] += cnt
            self.ud_pos_surface = derived

    @property
    def has_any_source(self) -> bool:
        return (self.lcc_total_tokens > 0) or (self.ud_total_tokens > 0) or bool(self.wf_probs)


def _prov_iso_matches(val: object, aliases: Set[str]) -> bool:
    tags = {tok.strip().lower() for tok in str(val).split(";") if tok.strip()}
    return bool(tags & aliases)


def _counter_matches_script(counter_obj: Counter, expected_script: Optional[str]) -> bool:
    if not expected_script or expected_script == "UNKNOWN":
        return True
    sample_words = [w for w, _ in counter_obj.most_common(500)]
    src_script = dominant_alpha_script(sample_words)
    return src_script in {"UNKNOWN", expected_script}


def load_iso_frequency_bundle(
    iso: str,
    freq_dir: Path = FREQ_SOURCES_DIR,
    strip_abjad_marks: bool = False,
    provenance_csv: Optional[Path] = None,
    expected_script: Optional[str] = None,
) -> IsoFrequencyBundle:
    """Load script-compatible UD v2.15, LCC *-words.txt, and wordfreq tables for `iso`."""
    aliases = set(corpus_aliases_for_iso(iso))
    source_ids: List[str] = []
    n_script_filtered = 0

    if provenance_csv is not None:
        prov_path = provenance_csv
    elif freq_dir.resolve() == FREQ_SOURCES_DIR.resolve():
        prov_path = PROVENANCE_CSV
    else:
        prov_path = freq_dir / "freq_provenance.csv"
    prov_df = pd.read_csv(prov_path) if prov_path.exists() else pd.DataFrame()

    # 1. Load UD v2.15 count tables (enforcing per-treebank dominant-script compatibility)
    ud_surface: Counter = Counter()
    ud_pos_surface: Counter = Counter()
    ud_lemma_pos: Counter = Counter()
    ud_total_tokens = 0

    ud_cnt_dir = freq_dir / "ud" / "counts"
    ud_repo_entries: List[Tuple[str, str]] = []
    if not prov_df.empty and "source_tier" in prov_df.columns:
        sub_ud = prov_df[
            (prov_df["source_tier"] == "tier_a_ud_v2.15")
            & (prov_df["iso639_3"].map(lambda v: _prov_iso_matches(v, aliases)))
        ]
        seen_repos: Set[str] = set()
        for _, r in sub_ud.iterrows():
            cid = str(r["corpus_id"]).split("@")[0]
            if cid not in seen_repos:
                seen_repos.add(cid)
                ud_repo_entries.append((cid, str(r["source_id"])))

    for repo, sid in ud_repo_entries:
        cnt_file = ud_cnt_dir / f"{repo}-r2.15_counts.tsv.gz"
        if not cnt_file.exists():
            continue
        loc_surf: Counter = Counter()
        loc_pos_surf: Counter = Counter()
        loc_lem_pos: Counter = Counter()
        loc_tokens = 0
        with gzip.open(cnt_file, "rt", encoding="utf-8", errors="replace") as gz:
            header = gz.readline()
            if not header:
                continue
            for line in gz:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                lem_raw, upos, form_raw, cnt_str = parts[0], parts[1], parts[2], parts[3]
                try:
                    cnt = int(cnt_str)
                except ValueError:
                    continue
                if cnt <= 0:
                    continue
                w_norm = normalize_surface(form_raw, iso=iso, strip_abjad_marks=strip_abjad_marks)
                if not w_norm:
                    continue
                loc_tokens += cnt
                loc_surf[w_norm] += cnt
                mapped_pos = map_ud_upos_to_pos(upos)
                if mapped_pos is not None:
                    loc_pos_surf[(mapped_pos, w_norm)] += cnt
                    l_norm = normalize_surface(lem_raw, iso=iso, strip_abjad_marks=strip_abjad_marks)
                    loc_lem_pos[(mapped_pos, l_norm, w_norm)] += cnt

        if loc_tokens == 0:
            continue
        if not _counter_matches_script(loc_surf, expected_script):
            n_script_filtered += 1
            continue
        ud_total_tokens += loc_tokens
        ud_surface.update(loc_surf)
        ud_pos_surface.update(loc_pos_surf)
        ud_lemma_pos.update(loc_lem_pos)
        source_ids.append(sid)

    # 2. Load LCC *-words.txt files (enforcing per-corpus dominant-script compatibility)
    lcc_surface: Counter = Counter()
    lcc_total_tokens = 0
    lcc_entries: List[Tuple[Path, str]] = []
    if not prov_df.empty and "source_tier" in prov_df.columns:
        sub_lcc = prov_df[
            (prov_df["source_tier"] == "tier_b_leipzig_corpora")
            & (prov_df["iso639_3"].map(lambda v: _prov_iso_matches(v, aliases)))
        ]
        seen_paths: Set[Path] = set()
        for _, r in sub_lcc.iterrows():
            p = REPO_ROOT / str(r["local_path"])
            if p.exists() and p not in seen_paths:
                seen_paths.add(p)
                lcc_entries.append((p, str(r["source_id"])))
    else:
        lcc_root = freq_dir / "lcc"
        if lcc_root.exists():
            for d in sorted(lcc_root.iterdir()):
                if not d.is_dir():
                    continue
                base = d.name.split("_")[0].split("-")[0]
                if base in aliases:
                    wf = d / f"{d.name}-words.txt"
                    if wf.exists():
                        lcc_entries.append((wf, f"lcc:{d.name}"))

    for wpath, sid in lcc_entries:
        loc_lcc: Counter = Counter()
        loc_tokens = 0
        with open(wpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 3:
                    continue
                word_raw = parts[1].strip()
                if not word_raw or _PUNCT_ONLY_RE.match(word_raw):
                    continue
                try:
                    cnt = int(parts[-1])
                except ValueError:
                    continue
                if cnt <= 0:
                    continue
                w_norm = normalize_surface(word_raw, iso=iso, strip_abjad_marks=strip_abjad_marks)
                if not w_norm:
                    continue
                loc_lcc[w_norm] += cnt
                loc_tokens += cnt

        if loc_tokens == 0:
            continue
        if not _counter_matches_script(loc_lcc, expected_script):
            n_script_filtered += 1
            continue
        lcc_surface.update(loc_lcc)
        lcc_total_tokens += loc_tokens
        source_ids.append(sid)

    # 3. Check wordfreq v3.1.1 availability (enforcing dominant-script compatibility)
    wf_lang: Optional[str] = None
    wf_probs: Dict[str, float] = {}
    try:
        import wordfreq

        cand = ISO_TO_WORDFREQ.get(iso)
        if cand and cand in wordfreq.available_languages("best"):
            raw_wf = wordfreq.get_frequency_dict(cand, wordlist="best")
            sample_wf_words = list(raw_wf.keys())[:500]
            wf_script = dominant_alpha_script(sample_wf_words)
            if not expected_script or expected_script == "UNKNOWN" or wf_script in {"UNKNOWN", expected_script}:
                wf_lang = cand
                for k_raw, prob in raw_wf.items():
                    k_norm = normalize_surface(k_raw, iso=iso, strip_abjad_marks=strip_abjad_marks)
                    if k_norm and prob > 0.0:
                        wf_probs[k_norm] = wf_probs.get(k_norm, 0.0) + float(prob)
                source_ids.append(f"wordfreq:{cand}")
            else:
                n_script_filtered += 1
    except ImportError:
        wf_lang = None

    return IsoFrequencyBundle(
        iso=iso,
        lcc_surface=lcc_surface,
        lcc_total_tokens=lcc_total_tokens,
        ud_surface=ud_surface,
        ud_lemma_pos=ud_lemma_pos,
        ud_total_tokens=ud_total_tokens,
        wf_lang=wf_lang,
        source_ids=sorted(set(source_ids)),
        wf_probs=wf_probs,
        ud_pos_surface=ud_pos_surface,
        n_script_filtered_sources=n_script_filtered,
    )


def disambiguate_and_aggregate(
    lex_df: pd.DataFrame,
    bundle: IsoFrequencyBundle,
    pos: str,
    iso: str,
    em_iterations: int = 5,
    strip_abjad_marks: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Match surface forms onto paradigm cells and disambiguate homographs via UD + 5-step EM."""
    unit_id = str(lex_df["unit_id"].iloc[0])

    # Filter out defective / missing cells
    valid_mask = (
        lex_df["raw_form"].notna()
        & (lex_df["raw_form"].astype(str).str.strip() != "")
        & (lex_df["raw_form"].astype(str).str.strip() != "#DEF#")
    )
    df = lex_df[valid_mask].copy()

    lemma_to_group: Dict[str, str] = {}
    lemma_to_label_norm: Dict[str, str] = {}
    lemma_to_raw_label: Dict[str, str] = {}
    forms_by_lemma_cell: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    forms_by_lemma: Dict[str, Set[str]] = defaultdict(set)

    for row in df[["lemma_id", "lemma_label", "leakage_group_id", "cell", "raw_form"]].itertuples(index=False):
        lid = str(row.lemma_id)
        lab = str(row.lemma_label)
        gid = str(row.leakage_group_id)
        cell = str(row.cell)
        f_norm = normalize_surface(str(row.raw_form), iso=iso, strip_abjad_marks=strip_abjad_marks)
        if not f_norm:
            continue
        if lid not in lemma_to_group:
            lemma_to_group[lid] = gid
            lemma_to_raw_label[lid] = lab
            lemma_to_label_norm[lid] = normalize_surface(lab, iso=iso, strip_abjad_marks=strip_abjad_marks)
        forms_by_lemma_cell[lid][cell].add(f_norm)
        forms_by_lemma[lid].add(f_norm)

    all_lemmas = sorted(lemma_to_group.keys())
    form_to_lemmas: Dict[str, List[str]] = defaultdict(list)
    for lid in all_lemmas:
        for w in forms_by_lemma[lid]:
            form_to_lemmas[w].append(lid)

    # Surface form counts C(w) across Tier B (LCC), Tier A (UD), and Tier C (wordfreq supplement)
    n_raw = bundle.lcc_total_tokens + bundle.ud_total_tokens
    n_corpus = float(n_raw) if n_raw > 0 else 1_000_000.0
    has_wf = bool(bundle.wf_probs)

    form_counts: Dict[str, float] = {}
    form_tier_mass: Dict[str, Tuple[float, float, float]] = {}  # (lcc, ud, wf)

    for w in form_to_lemmas:
        c_lcc = float(bundle.lcc_surface.get(w, 0))
        c_ud_all = int(bundle.ud_surface.get(w, 0))
        c_ud_pos = int(bundle.ud_pos_surface.get((pos, w), 0))
        # Empirical-Bayes target-POS share discounts function-word collisions (e.g. 'and', 'a', 'do')
        # while keeping 1.0 whenever w is 100% target-POS or unseen in UD.
        if c_ud_all > 0 and c_ud_pos < c_ud_all:
            pos_share = (float(c_ud_pos) + 0.5) / (float(c_ud_all) + 1.0)
        else:
            pos_share = 1.0
        c_lcc_adj = pos_share * c_lcc
        c_ud_adj = float(c_ud_pos) if c_ud_all > 0 else 0.0

        if n_raw > 0:
            if c_lcc_adj + c_ud_adj > 0.0:
                form_counts[w] = c_lcc_adj + c_ud_adj
                form_tier_mass[w] = (c_lcc_adj, c_ud_adj, 0.0)
            elif has_wf:
                p_wf = float(bundle.wf_probs.get(w, 0.0))
                c_wf = pos_share * p_wf * n_corpus if p_wf > 0.0 else 0.0
                form_counts[w] = c_wf
                form_tier_mass[w] = (0.0, 0.0, c_wf)
            else:
                form_counts[w] = 0.0
                form_tier_mass[w] = (0.0, 0.0, 0.0)
        elif has_wf:
            p_wf = float(bundle.wf_probs.get(w, 0.0))
            c_wf = pos_share * p_wf * n_corpus if p_wf > 0.0 else 0.0
            form_counts[w] = c_wf
            form_tier_mass[w] = (0.0, 0.0, c_wf)
        else:
            form_counts[w] = 0.0
            form_tier_mass[w] = (0.0, 0.0, 0.0)

    # Partition surface forms into fixed (unambiguous or UD-resolved) vs EM homographs
    gamma_fixed: Dict[str, Dict[str, float]] = {}
    em_forms: List[str] = []
    n_homographs_total = 0
    n_homographs_ud = 0
    n_homographs_em = 0

    for w, llist in form_to_lemmas.items():
        cw = form_counts[w]
        if cw <= 0.0:
            continue
        if len(llist) == 1:
            gamma_fixed[w] = {llist[0]: 1.0}
            continue

        n_homographs_total += 1
        # Step 1: Check UD (pos, lemma_norm, w) counts
        label_mult = Counter(lemma_to_label_norm[l] for l in llist)
        ud_weights: Dict[str, float] = {}
        for l in llist:
            lnorm = lemma_to_label_norm[l]
            u_cnt = float(bundle.ud_lemma_pos.get((pos, lnorm, w), 0))
            ud_weights[l] = u_cnt / float(label_mult[lnorm])

        u_sum = sum(ud_weights.values())
        # Use UD ratios when UD resolves matching lemmas cleanly (and doesn't leave identical-label ties)
        has_same_label_tie = any(
            label_mult[lemma_to_label_norm[l]] > 1 and ud_weights[l] > 0.0 for l in llist
        )
        if u_sum > 0.0 and not has_same_label_tie:
            gamma_fixed[w] = {l: ud_weights[l] / u_sum for l in llist}
            n_homographs_ud += 1
        else:
            em_forms.append(w)
            n_homographs_em += 1

    # Base fixed lemma token totals
    f_fixed: Dict[str, float] = {l: 0.0 for l in all_lemmas}
    for w, gmap in gamma_fixed.items():
        cw = form_counts[w]
        for l, weight in gmap.items():
            f_fixed[l] += weight * cw

    # Step 2: 5 iterations of proportional EM initialized uniformly across matching lemmas
    gamma_em: Dict[str, Dict[str, float]] = {
        w: {l: 1.0 / len(form_to_lemmas[w]) for l in form_to_lemmas[w]} for w in em_forms
    }

    f_curr: Dict[str, float] = dict(f_fixed)
    for w in em_forms:
        cw = form_counts[w]
        for l, weight in gamma_em[w].items():
            f_curr[l] += weight * cw

    for _ in range(max(0, int(em_iterations))):
        if not em_forms:
            break
        # E-step
        for w in em_forms:
            llist = form_to_lemmas[w]
            denom = sum(f_curr[l] for l in llist)
            if denom > 0.0:
                gamma_em[w] = {l: f_curr[l] / denom for l in llist}
            else:
                inv = 1.0 / len(llist)
                gamma_em[w] = {l: inv for l in llist}
        # M-step
        f_curr = dict(f_fixed)
        for w in em_forms:
            cw = form_counts[w]
            for l, weight in gamma_em[w].items():
                f_curr[l] += weight * cw

    # Verify exact conservation of token mass & zero NaN/Inf
    total_matched_surface = sum(v for v in form_counts.values() if v > 0.0)
    total_allocated_lemma = sum(f_curr.values())
    if not math.isfinite(total_allocated_lemma):
        raise RuntimeError(f"{unit_id}: non-finite EM frequency sum detected!")
    if abs(total_allocated_lemma - total_matched_surface) > 1e-5 * max(1.0, total_matched_surface):
        raise RuntimeError(
            f"{unit_id}: EM mass mismatch {total_allocated_lemma} vs {total_matched_surface}"
        )

    # Track source-tier breakdown per lemma
    lemma_tier_mass: Dict[str, List[float]] = {l: [0.0, 0.0, 0.0] for l in all_lemmas}
    lemma_used_ud_disambig: Set[str] = set()
    for w, gmap in gamma_fixed.items():
        m_lcc, m_ud, m_wf = form_tier_mass[w]
        is_ud_disambig = len(gmap) > 1
        for l, weight in gmap.items():
            if weight > 0.0:
                lemma_tier_mass[l][0] += weight * m_lcc
                lemma_tier_mass[l][1] += weight * m_ud
                lemma_tier_mass[l][2] += weight * m_wf
                if is_ud_disambig:
                    lemma_used_ud_disambig.add(l)
    for w, gmap in gamma_em.items():
        m_lcc, m_ud, m_wf = form_tier_mass[w]
        for l, weight in gmap.items():
            if weight > 0.0:
                lemma_tier_mass[l][0] += weight * m_lcc
                lemma_tier_mass[l][1] += weight * m_ud
                lemma_tier_mass[l][2] += weight * m_wf

    out_rows: List[dict] = []
    for l in all_lemmas:
        tok = max(0.0, float(f_curr[l]))
        pmw = (tok * 1_000_000.0) / n_corpus
        # Laplace +1 token smoothed PMW on log10 scale -> strictly finite, scale-invariant
        log10_f = math.log10(((tok + 1.0) * 1_000_000.0) / n_corpus)

        n_cells_att = sum(
            1
            for cell_forms in forms_by_lemma_cell[l].values()
            if any(form_counts.get(w, 0.0) > 0.0 for w in cell_forms)
        )

        if tok <= 0.0:
            prim_src = "unattested"
        else:
            m_lcc, m_ud, m_wf = lemma_tier_mass[l]
            if m_lcc > 0.0 and (m_ud > 0.0 or l in lemma_used_ud_disambig):
                prim_src = "lcc+ud"
            elif m_lcc >= m_ud and m_lcc >= m_wf and m_lcc > 0.0:
                prim_src = "lcc"
            elif m_ud >= m_wf and m_ud > 0.0:
                prim_src = "ud_v2.15"
            else:
                prim_src = "wordfreq"

        out_rows.append(
            {
                "unit_id": unit_id,
                "lemma_id": l,
                "leakage_group_id": lemma_to_group[l],
                "freq_lemma_tokens": round(tok, 6),
                "freq_lemma_pmw": round(pmw, 6),
                "log10_freq_lemma": round(log10_f, 6),
                "n_cells_attested": int(n_cells_att),
                "primary_freq_source": prim_src,
            }
        )

    out_df = pd.DataFrame(out_rows, columns=FREQ_UNIT_COLUMNS)
    n_lemmas = len(out_df)
    n_lemmas_att = int((out_df["freq_lemma_tokens"] > 0).sum())
    grp_totals = out_df.groupby("leakage_group_id")["freq_lemma_tokens"].sum()
    n_groups = int(len(grp_totals))
    n_groups_att = int((grp_totals > 0).sum())

    meta = {
        "n_lemmas": n_lemmas,
        "n_lemmas_attested": n_lemmas_att,
        "lemma_match_rate": round(n_lemmas_att / max(1, n_lemmas), 4),
        "n_groups": n_groups,
        "n_groups_attested": n_groups_att,
        "group_match_rate": round(n_groups_att / max(1, n_groups), 4),
        "corpus_tokens_denominator": int(round(n_corpus)),
        "matched_paradigm_tokens": round(total_matched_surface, 2),
        "n_homographs_total": n_homographs_total,
        "n_homographs_ud": n_homographs_ud,
        "n_homographs_em": n_homographs_em,
        "lemma_raw_labels": lemma_to_raw_label,
    }
    return out_df, meta


def _derive_seed(*parts: object) -> int:
    key = "::".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def detect_reference_split_schema(
    unit_id: str,
    splits_dir: Path = SPLITS_DIR,
) -> Tuple[List[str], str, str, int]:
    """Inspect data-ingest's `{unit_id}_uniform120.csv` (or any `*_uniform120.csv`) to match schema."""
    candidates: List[Path] = []
    exact = splits_dir / f"{unit_id}_uniform120.csv"
    if exact.exists():
        candidates.append(exact)
    elif splits_dir.exists():
        candidates.extend(sorted(splits_dir.glob("*_uniform120.csv"))[:1])

    if not candidates:
        return list(DEFAULT_SPLIT_COLUMNS), "draw_idx", "fold_idx", 0

    ref_df = pd.read_csv(candidates[0], nrows=200)
    cols = list(ref_df.columns)
    draw_col = next((c for c in cols if c in {"draw_idx", "draw", "rep", "repetition", "draw_id"}), "draw_idx")
    fold_col = next((c for c in cols if c in {"fold_idx", "fold", "fold_id"}), "fold_idx")
    base_idx = int(ref_df[fold_col].min()) if fold_col in ref_df.columns and len(ref_df) > 0 else 0
    return cols, draw_col, fold_col, base_idx


def generate_topfreq120_split(
    freq_df: pd.DataFrame,
    lemma_raw_labels: Dict[str, str],
    splits_dir: Path = SPLITS_DIR,
    n_draws: int = 5,
    n_folds: int = 5,
    sample_groups: int = 150,
    top_pool_cap: int = 500,
    min_attested_groups: int = 200,
) -> Optional[pd.DataFrame]:
    """Generate `{unit_id}_topfreq120.csv` if `>= 200` distinct leakage groups have frequency > 0."""
    unit_id = str(freq_df["unit_id"].iloc[0])

    # Rank lemmas within each group: highest freq_lemma_tokens -> shortest label -> lemma_id
    df = freq_df.copy()
    df["_label_len"] = df["lemma_id"].map(lambda lid: len(lemma_raw_labels.get(str(lid), str(lid))))
    df = df.sort_values(
        by=["freq_lemma_tokens", "_label_len", "lemma_id"],
        ascending=[False, True, True],
    )

    grp_sum = df.groupby("leakage_group_id", sort=False)["freq_lemma_tokens"].sum()
    attested_groups = grp_sum[grp_sum > 0.0]
    if len(attested_groups) < min_attested_groups:
        return None

    # Canonical representative lemma per group = highest-frequency lemma in that group
    canon_per_group = df.drop_duplicates(subset=["leakage_group_id"], keep="first").set_index("leakage_group_id")

    ranked_gids = sorted(
        attested_groups.index.tolist(),
        key=lambda g: (-float(attested_groups[g]), -float(canon_per_group.loc[g, "freq_lemma_tokens"]), str(g)),
    )
    pool_size = min(int(top_pool_cap), len(ranked_gids))
    top_pool = ranked_gids[:pool_size]

    ref_cols, draw_col, fold_col, idx_base = detect_reference_split_schema(unit_id, splits_dir=splits_dir)

    rows: List[dict] = []
    per_fold_test = sample_groups // n_folds  # 30
    subtrain_size = 96
    dev_size = 24
    assert per_fold_test * n_folds == sample_groups
    assert subtrain_size + dev_size == sample_groups - per_fold_test

    for r in range(n_draws):
        rng_draw = np.random.default_rng(_derive_seed("topfreq120_draw", unit_id, r))
        chosen_idx = rng_draw.choice(len(top_pool), size=sample_groups, replace=False)
        chosen_gids = [top_pool[int(i)] for i in chosen_idx]
        # Shuffle chosen groups into K=5 equal folds of 30 groups
        fold_perm = rng_draw.permutation(sample_groups)
        fold_of_pos = {int(fold_perm[pos_i]): (pos_i // per_fold_test) for pos_i in range(sample_groups)}

        for k in range(n_folds):
            test_gids = [chosen_gids[idx] for idx in range(sample_groups) if fold_of_pos[idx] == k]
            train_gids = [chosen_gids[idx] for idx in range(sample_groups) if fold_of_pos[idx] != k]
            assert len(test_gids) == per_fold_test
            assert len(train_gids) == subtrain_size + dev_size
            assert set(test_gids).isdisjoint(set(train_gids))

            rng_inner = np.random.default_rng(_derive_seed("topfreq120_inner", unit_id, r, k))
            inner_perm = rng_inner.permutation(len(train_gids))
            subtrain_set = {train_gids[int(i)] for i in inner_perm[:subtrain_size]}
            dev_set = {train_gids[int(i)] for i in inner_perm[subtrain_size:]}
            assert len(subtrain_set) == subtrain_size and len(dev_set) == dev_size
            assert subtrain_set.isdisjoint(dev_set)

            ordered_fold_gids = sorted(train_gids) + sorted(test_gids)
            for rank_i, gid in enumerate(ordered_fold_gids):
                lid = str(canon_per_group.loc[gid, "lemma_id"])
                if gid in subtrain_set:
                    role, inner_role = "train", "subtrain"
                elif gid in dev_set:
                    role, inner_role = "train", "dev"
                else:
                    role, inner_role = "test", "test"

                rec = {
                    "unit_id": unit_id,
                    draw_col: r + idx_base,
                    fold_col: k + idx_base,
                    "lemma_id": lid,
                    "leakage_group_id": gid,
                    "role": role,
                    "inner_role": inner_role,
                    "regime": "topfreq120",
                    "rank_in_fold": rank_i,
                }
                rows.append({c: rec[c] for c in ref_cols if c in rec} if set(ref_cols) <= set(rec) else rec)

    out_cols = ref_cols if all(c in rows[0] for c in ref_cols) else DEFAULT_SPLIT_COLUMNS
    split_df = pd.DataFrame(rows)[out_cols]

    # Final zero-leakage invariant assertion
    for (d_val, f_val), grp in split_df.groupby([draw_col, fold_col]):
        tr_g = set(grp.loc[grp["role"] == "train", "leakage_group_id"])
        te_g = set(grp.loc[grp["role"] == "test", "leakage_group_id"])
        if tr_g & te_g:
            raise AssertionError(f"Group leakage in {unit_id} draw={d_val} fold={f_val}!")
        if len(tr_g) != 120 or len(te_g) != 30:
            raise AssertionError(f"Unexpected split counts in {unit_id}: {len(tr_g)}/{len(te_g)}")

    return split_df


def match_all_units(
    lexicons_dir: Path = LEXICONS_DIR,
    freq_dir: Path = FREQ_SOURCES_DIR,
    freq_out_dir: Path = FREQ_OUT_DIR,
    splits_dir: Path = SPLITS_DIR,
    splits_freq_dir: Path = SPLITS_FREQ_DIR,
    em_iterations: int = 5,
) -> pd.DataFrame:
    """Match frequencies across all `orth` lexicon units and build `splits_freq/` manifests."""
    freq_out_dir.mkdir(parents=True, exist_ok=True)
    splits_freq_dir.mkdir(parents=True, exist_ok=True)

    lex_files = sorted(lexicons_dir.glob("*.csv"))
    if not lex_files:
        raise FileNotFoundError(f"No lexicon CSV files found in {lexicons_dir}")

    bundle_cache: Dict[Tuple[str, bool, str], IsoFrequencyBundle] = {}
    summary_rows: List[dict] = []

    for lex_path in lex_files:
        lex_df = pd.read_csv(lex_path, keep_default_na=False)
        if lex_df.empty:
            continue
        unit_id = str(lex_df["unit_id"].iloc[0])
        glottocode = str(lex_df["glottocode"].iloc[0])
        iso = str(lex_df["iso639_3"].iloc[0]).strip().lower()
        pos = str(lex_df["pos"].iloc[0]).strip()
        rep = str(lex_df["representation"].iloc[0]).strip()

        # Surface wordform matching targets orthographic units per specification
        if rep != "orth":
            continue

        lex_script = dominant_alpha_script(lex_df["raw_form"].head(500).tolist())
        cache_key = (iso, False, lex_script)
        if cache_key not in bundle_cache:
            bundle_cache[cache_key] = load_iso_frequency_bundle(
                iso,
                freq_dir=freq_dir,
                strip_abjad_marks=False,
                expected_script=lex_script,
            )
        bundle = bundle_cache[cache_key]

        if not bundle.has_any_source:
            (freq_out_dir / f"{unit_id}_freq.csv").unlink(missing_ok=True)
            (splits_freq_dir / f"{unit_id}_topfreq120.csv").unlink(missing_ok=True)
            n_lemmas = int(lex_df["lemma_id"].nunique())
            n_groups = int(lex_df["leakage_group_id"].nunique())
            summary_rows.append(
                {
                    "unit_id": unit_id,
                    "glottocode": glottocode,
                    "iso639_3": iso,
                    "pos": pos,
                    "representation": rep,
                    "has_freq_source": False,
                    "sources_used": "",
                    "abjad_unvocalized_fallback": False,
                    "script_mismatch_excluded": bool(bundle.n_script_filtered_sources > 0),
                    "n_lemmas": n_lemmas,
                    "n_lemmas_attested": 0,
                    "lemma_match_rate": 0.0,
                    "n_groups": n_groups,
                    "n_groups_attested": 0,
                    "group_match_rate": 0.0,
                    "corpus_tokens_denominator": 0,
                    "matched_paradigm_tokens": 0.0,
                    "n_homographs_total": 0,
                    "n_homographs_ud": 0,
                    "n_homographs_em": 0,
                    "has_topfreq120_split": False,
                }
            )
            continue

        freq_df, meta = disambiguate_and_aggregate(
            lex_df=lex_df,
            bundle=bundle,
            pos=pos,
            iso=iso,
            em_iterations=em_iterations,
            strip_abjad_marks=False,
        )
        used_abjad_fallback = False

        # Automatic Abjad vocalization fallback when paradigm forms carry diacritics absent in news/wiki
        if iso in ABJAD_VOCALIZATION_ISOS and meta["group_match_rate"] < 0.10:
            alt_key = (iso, True, lex_script)
            if alt_key not in bundle_cache:
                bundle_cache[alt_key] = load_iso_frequency_bundle(
                    iso,
                    freq_dir=freq_dir,
                    strip_abjad_marks=True,
                    expected_script=lex_script,
                )
            alt_df, alt_meta = disambiguate_and_aggregate(
                lex_df=lex_df,
                bundle=bundle_cache[alt_key],
                pos=pos,
                iso=iso,
                em_iterations=em_iterations,
                strip_abjad_marks=True,
            )
            if (
                alt_meta["group_match_rate"] >= 0.15
                and alt_meta["n_groups_attested"] >= 2 * max(1, meta["n_groups_attested"])
            ):
                freq_df, meta = alt_df, alt_meta
                used_abjad_fallback = True

        # Secondary guard against cross-script mismatches if a corpus had mixed scripts
        corp_scripts = bundle_supported_scripts(
            bundle_cache[(iso, True, lex_script)] if used_abjad_fallback else bundle
        )
        if (
            lex_script != "UNKNOWN"
            and corp_scripts
            and lex_script not in corp_scripts
            and meta["group_match_rate"] < 0.05
        ):
            (freq_out_dir / f"{unit_id}_freq.csv").unlink(missing_ok=True)
            (splits_freq_dir / f"{unit_id}_topfreq120.csv").unlink(missing_ok=True)
            summary_rows.append(
                {
                    "unit_id": unit_id,
                    "glottocode": glottocode,
                    "iso639_3": iso,
                    "pos": pos,
                    "representation": rep,
                    "has_freq_source": False,
                    "sources_used": "",
                    "abjad_unvocalized_fallback": False,
                    "script_mismatch_excluded": True,
                    "n_lemmas": meta["n_lemmas"],
                    "n_lemmas_attested": 0,
                    "lemma_match_rate": 0.0,
                    "n_groups": meta["n_groups"],
                    "n_groups_attested": 0,
                    "group_match_rate": 0.0,
                    "corpus_tokens_denominator": 0,
                    "matched_paradigm_tokens": 0.0,
                    "n_homographs_total": 0,
                    "n_homographs_ud": 0,
                    "n_homographs_em": 0,
                    "has_topfreq120_split": False,
                }
            )
            continue

        out_unit_csv = freq_out_dir / f"{unit_id}_freq.csv"
        freq_df.to_csv(out_unit_csv, index=False)

        split_df = generate_topfreq120_split(
            freq_df=freq_df,
            lemma_raw_labels=meta["lemma_raw_labels"],
            splits_dir=splits_dir,
        )
        has_split = split_df is not None
        if split_df is not None:
            split_path = splits_freq_dir / f"{unit_id}_topfreq120.csv"
            split_df.to_csv(split_path, index=False)
        else:
            (splits_freq_dir / f"{unit_id}_topfreq120.csv").unlink(missing_ok=True)

        summary_rows.append(
            {
                "unit_id": unit_id,
                "glottocode": glottocode,
                "iso639_3": iso,
                "pos": pos,
                "representation": rep,
                "has_freq_source": True,
                "sources_used": ";".join(bundle.source_ids),
                "abjad_unvocalized_fallback": used_abjad_fallback,
                "script_mismatch_excluded": False,
                "n_lemmas": meta["n_lemmas"],
                "n_lemmas_attested": meta["n_lemmas_attested"],
                "lemma_match_rate": meta["lemma_match_rate"],
                "n_groups": meta["n_groups"],
                "n_groups_attested": meta["n_groups_attested"],
                "group_match_rate": meta["group_match_rate"],
                "corpus_tokens_denominator": meta["corpus_tokens_denominator"],
                "matched_paradigm_tokens": meta["matched_paradigm_tokens"],
                "n_homographs_total": meta["n_homographs_total"],
                "n_homographs_ud": meta["n_homographs_ud"],
                "n_homographs_em": meta["n_homographs_em"],
                "has_topfreq120_split": has_split,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("unit_id").reset_index(drop=True)
        valid_freq_uids = set(summary_df.loc[summary_df["has_freq_source"], "unit_id"])
        valid_split_uids = set(summary_df.loc[summary_df["has_topfreq120_split"], "unit_id"])
        for fp in freq_out_dir.glob("*_freq.csv"):
            uid_cand = fp.name[: -len("_freq.csv")]
            if uid_cand not in valid_freq_uids:
                fp.unlink(missing_ok=True)
        for sp in splits_freq_dir.glob("*_topfreq120.csv"):
            uid_cand = sp.name[: -len("_topfreq120.csv")]
            if uid_cand not in valid_split_uids:
                sp.unlink(missing_ok=True)
    summary_path = freq_out_dir / "freq_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(
        f"[match_paradigms] Processed {len(summary_df)} orth units -> {summary_path} "
        f"(with_freq={int(summary_df['has_freq_source'].sum()) if len(summary_df) else 0}, "
        f"topfreq120={int(summary_df['has_topfreq120_split'].sum()) if len(summary_df) else 0})"
    )
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Match frequencies onto Track 2 lexicon paradigms.")
    parser.add_argument("--lexicons-dir", type=Path, default=LEXICONS_DIR)
    parser.add_argument("--freq-dir", type=Path, default=FREQ_SOURCES_DIR)
    parser.add_argument("--out-dir", type=Path, default=FREQ_OUT_DIR)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--splits-freq-dir", type=Path, default=SPLITS_FREQ_DIR)
    parser.add_argument("--em-iters", type=int, default=5)
    args = parser.parse_args()

    match_all_units(
        lexicons_dir=args.lexicons_dir,
        freq_dir=args.freq_dir,
        freq_out_dir=args.out_dir,
        splits_dir=args.splits_dir,
        splits_freq_dir=args.splits_freq_dir,
        em_iterations=args.em_iters,
    )


if __name__ == "__main__":
    main()
