"""Comprehensive mathematical, linguistic, and statistical unit tests for Multi-Locus Vowel-Tier MGL."""

from __future__ import annotations

import csv
import itertools
import math
import random
from pathlib import Path

import networkx as nx
import pytest

from analyses.paradigm.mgl.align import align_pair, extract_pattern
from analyses.paradigm.mgl.escape import DirichletEscapeScorer, StochasticEditModel
from analyses.paradigm.mgl.evaluate_mgl import (
    LEMMA_OUTCOME_COLUMNS,
    UNIT_OUTCOME_COLUMNS,
    evaluate_unit,
    load_split_manifest,
    load_unit_lexicon,
    run_regime_evaluation,
)
from analyses.paradigm.mgl.mgl_learner import (
    MGLLearner,
    _find_rule_outputs,
    _IndexedTrainingSet,
    mikheev_lower,
    mikheev_upper,
)
from analyses.paradigm.mgl.topology import (
    chu_liu_edmonds,
    evaluate_lemma_topology,
    prescreen_candidate_edges,
    score_poe_target,
    select_optimal_arborescence,
)
from analyses.paradigm.mgl.vowel_tier import (
    SoundInventory,
    VowelClassifier,
    parse_segments,
    sukhotin_discover_vowels,
)


# ---------------------------------------------------------------------------
# 1. Mikheev Confidence Benchmarks (Albright 2002 / Albright & Hayes 2003)
# ---------------------------------------------------------------------------


def test_mikheev_published_benchmarks() -> None:
    """Verify exact Student-t Mikheev `pi_L(alpha=0.75)` against Albright (2002) & A&H (2003)."""
    # Primary targets explicitly cited in Albright & Hayes (2003 p. 127) and Albright (2002 Table B.1)
    assert abs(mikheev_lower(5, 5, 0.75) - 0.8251) < 5e-4
    assert abs(mikheev_lower(4, 4, 0.75) - 0.7853) < 5e-4
    assert abs(mikheev_lower(9, 9, 0.75) - 0.8987) < 5e-4

    published_table_b1 = [
        (65, 65, 0.985),
        (104, 105, 0.978),
        (43, 43, 0.978),
        (19, 19, 0.950),
        (11, 11, 0.916),
        (10, 10, 0.908),
        (19, 20, 0.889),
    ]
    for hits, scope, expected in published_table_b1:
        val = mikheev_lower(hits, scope, 0.75)
        assert abs(val - expected) < 1e-3, f"{hits}/{scope}: got {val:.4f}, expected {expected}"

    # Strict monotonicity of k/k for k = 1 .. 30 and clean n=1 / n=0 handling
    assert mikheev_lower(0, 0, 0.75) == 0.0
    vals_kk = [mikheev_lower(k, k, 0.75) for k in range(1, 31)]
    for k_idx in range(len(vals_kk) - 1):
        assert 0.0 < vals_kk[k_idx] < vals_kk[k_idx + 1] < 1.0

    # Upper bound properties for residue impugnment (alpha_upper = 0.95)
    assert 0.0 < mikheev_upper(0, 20, 0.95) < 0.10
    assert mikheev_upper(1, 151, 0.95) < 0.03  # A&H (2002 §3.7) burnt residue 1/151


# ---------------------------------------------------------------------------
# 2. Multi-Script Vowel Tier, Sukhotin Fallback, `sounds.csv` & Zero Target Leakage
# ---------------------------------------------------------------------------


def test_vowel_tier_scripts_sukhotin_and_zero_target_leakage(tmp_path: Path) -> None:
    clf = VowelClassifier()

    # Latin + Turkic/Uralic Cyrillic + Greek + Armenian + Georgian + Hebrew/Arabic + Devanagari + Hangul
    assert clf.classify("a") == "V" and clf.classify("k") == "C"
    for cyr_v in ("ә", "ө", "ү", "і", "ӱ", "ӧ", "ы", "э", "я", "ю", "е", "ё", "а", "у"):
        assert clf.classify(cyr_v) == "V", f"Cyrillic vowel {cyr_v} misclassified"
    for cyr_c in ("к", "т", "й", "ь", "ъ", "ш"):
        assert clf.classify(cyr_c) == "C", f"Cyrillic non-vowel {cyr_c} misclassified"

    assert clf.classify("ω") == "V" and clf.classify("π") == "C"  # Greek
    assert clf.classify("ա") == "V" and clf.classify("մ") == "C"  # Armenian
    assert clf.classify("ა") == "V" and clf.classify("მ") == "C"  # Georgian
    assert clf.classify("\u05b4") == "V" and clf.classify("ב") == "C"  # Hebrew
    assert clf.classify("ۆ") == "V" and clf.classify("ک") == "C"  # Kurdish/Uyghur Arabic
    assert clf.classify("ा") == "V" and clf.classify("क") == "C"  # Devanagari matra vs C
    assert clf.classify("ㅏ") == "V" and clf.classify("ㄱ") == "C"  # Hangul jamo
    assert clf.classify("tʲ") == "C" and clf.classify("aː") == "V"  # IPA modifiers

    # Ambiguous glide resolution via Sukhotin neighbor statistics:
    # Finnish 'y' between consonants -> 'V'; Turkish 'y' between vowels -> 'C'
    fin_clf = VowelClassifier().fit([("k", "y", "l", "ä"), ("t", "y", "t", "t", "ö"), ("s", "y", "y")])
    tur_clf = VowelClassifier().fit([("o", "k", "u", "y", "o", "r"), ("g", "e", "l", "i", "y", "o", "r")])
    assert fin_clf.classify("y") == "V"
    assert tur_clf.classify("y") == "C"

    # Pure cipher / unknown script fallback via Sukhotin's (1962) algorithm
    cipher_map = {"t": "S1", "k": "S2", "l": "S3", "m": "S4", "s": "S5", "a": "Q1", "o": "Q2", "i": "Q3"}
    raw_words = ["t a l o", "k a l a", "m a t k a", "s i l m a", "k o t i", "t a l o s a", "m o l i"]
    cipher_forms = [[cipher_map[ch] for ch in w.split()] for w in raw_words]
    discovered_v = sukhotin_discover_vowels(cipher_forms)
    assert discovered_v == {"Q1", "Q2", "Q3"}

    unk_clf = VowelClassifier().fit(cipher_forms)
    for v_sym in ("Q1", "Q2", "Q3"):
        assert unk_clf.classify(v_sym) == "V"
    for c_sym in ("S1", "S2", "S3", "S4", "S5"):
        assert unk_clf.classify(c_sym) == "C"

    # Paralex-style `{unit_id}_sounds.csv` feature loading
    sounds_file = tmp_path / "test_sounds.csv"
    sounds_file.write_text(
        "sound_id,description\n"
        "p,voiceless bilabial plosive\n"
        "f,voiceless labiodental fricative\n"
        "x,voiceless velar fricative\n"
        "a,open front unrounded vowel\n",
        encoding="utf-8",
    )
    inv = SoundInventory.from_csv(sounds_file)
    phon_clf = VowelClassifier(sound_inventory=inv, use_phon_features=True)
    assert phon_clf.classify("a") == "V"
    assert phon_clf.classify("x") == "C"
    shared_fx = phon_clf.segment_features("f") & phon_clf.segment_features("x")
    assert {"-syllabic", "+continuant", "-voice"}.issubset(shared_fx)

    # CRITICAL ZERO TARGET LEAKAGE CHECK: changing `w_t` vowels NEVER alters `source_vowel_tier`
    # or `left_vowels` / `right_vowels` in `extract_pattern`!
    aln_front_tgt = extract_pattern(("t", "a", "l", "o"), ("t", "a", "l", "o", "s", "s", "ä"), clf)
    aln_back_tgt = extract_pattern(("t", "a", "l", "o"), ("t", "a", "l", "o", "s", "s", "a"), clf)
    assert aln_front_tgt.source_vowel_tier == ("a", "o")
    assert aln_front_tgt.source_vowel_tier == aln_back_tgt.source_vowel_tier
    assert aln_front_tgt.locus_contexts == aln_back_tgt.locus_contexts


# ---------------------------------------------------------------------------
# 3. V/C-Weighted Alignment Symmetry & Multi-Slot Pattern Extraction
# ---------------------------------------------------------------------------


def test_alignment_symmetry_and_multilocus_patterns() -> None:
    clf = VowelClassifier()

    # Exact cost checks: same-class V/V = 1.0 < 2*1.1; cross-class V/C = 2.5 > 2*1.1 -> indel (2.2)
    _, d_vv = align_pair(("s", "i", "n", "g"), ("s", "a", "n", "g"), clf)
    assert abs(d_vv - 1.0) < 1e-6
    _, d_vc = align_pair(("a",), ("t",), clf)
    assert abs(d_vc - 2.2) < 1e-6

    # Exact symmetry `dist(x, y) == dist(y, x)` across diverse word pairs
    sample_forms = [
        ("m", "i", "s", "s"),
        ("n", "i", "d"),
        ("n", "i", "d", "ə", "d"),
        ("s", "c", "h", "l", "e", "p", "p"),
        ("g", "e", "s", "c", "h", "l", "e", "p", "p", "t"),
        ("p", "e", "n", "s", "o"),
        ("p", "i", "e", "n", "s", "e"),
        ("t", "a", "l", "o"),
        ("t", "a", "l", "o", "o", "n"),
    ]
    for fx in sample_forms:
        for fy in sample_forms:
            _, d1 = align_pair(fx, fy, clf)
            _, d2 = align_pair(fy, fx, clf)
            assert abs(d1 - d2) < 1e-6, f"Asymmetric distance on {fx} vs {fy}: {d1} != {d2}"

    # Canonical multi-locus pattern signatures matching brief examples!
    assert extract_pattern(("m", "i", "s"), ("m", "i", "s", "t"), clf).pattern.signature() == "_ -> _t"
    # Contiguous stem preservation on duplicate stem-final segment (`need -> needed`, `talo -> taloon`)
    assert extract_pattern(("n", "i", "d"), ("n", "i", "d", "ə", "d"), clf).pattern.signature() == "_ -> _əd"
    assert extract_pattern(("t", "a", "l", "o"), ("t", "a", "l", "o", "o", "n"), clf).pattern.signature() == "_ -> _on"
    # German circumfix `ge_t`
    assert (
        extract_pattern(
            ("s", "c", "h", "l", "e", "p", "p"),
            ("g", "e", "s", "c", "h", "l", "e", "p", "p", "t"),
            clf,
        ).pattern.signature()
        == "_ -> ge_t"
    )
    # Simultaneous stem diphthongization + suffix replacement (`_ɛ_o -> _ie_e` and `_o_o -> _ue_e`)
    assert (
        extract_pattern(("p", "ɛ", "n", "s", "o"), ("p", "i", "e", "n", "s", "e"), clf).pattern.signature()
        == "_ɛ_o -> _ie_e"
    )
    assert (
        extract_pattern(("c", "o", "n", "t", "o"), ("c", "u", "e", "n", "t", "e"), clf).pattern.signature()
        == "_o_o -> _ue_e"
    )


# ---------------------------------------------------------------------------
# 4. English `-t / -d / -əd`, `out-Bach`, Bitmask Exactness, `max` vs `sum`, & Impugnment
# ---------------------------------------------------------------------------


def test_english_past_tense_and_burnt_impugnment() -> None:
    raw_pairs = [
        # Voiceless non-alveolar-stop -> -t
        ("m ɪ s", "m ɪ s t"),
        ("p r ɛ s", "p r ɛ s t"),
        ("k ɪ s", "k ɪ s t"),
        ("l æ f", "l æ f t"),
        ("s t ʌ f", "s t ʌ f t"),
        ("w ɪ ʃ", "w ɪ ʃ t"),
        ("p ʊ ʃ", "p ʊ ʃ t"),
        ("b e k", "b e k t"),
        ("l ʊ k", "l ʊ k t"),
        ("dʒ ʌ m p", "dʒ ʌ m p t"),
        ("h o p", "h o p t"),
        # Voiced non-alveolar-stop -> -d
        ("h ʌ g", "h ʌ g d"),
        ("r ʌ b", "r ʌ b d"),
        ("g r æ b", "g r æ b d"),
        ("l ʌ v", "l ʌ v d"),
        ("m u v", "m u v d"),
        ("b ʌ z", "b ʌ z d"),
        ("p l æ n", "p l æ n d"),
        ("r e n", "r e n d"),
        ("s m aɪ l", "s m aɪ l d"),
        ("k ɔ l", "k ɔ l d"),
        ("p l e", "p l e d"),
        ("ʃ o", "ʃ o d"),
        # Alveolar stops t/d -> -əd
        ("n i d", "n i d ə d"),
        ("w e d", "w e d ə d"),
        ("f e d", "f e d ə d"),
        ("v o t", "v o t ə d"),
        ("w e t", "w e t ə d"),
        ("s t ɑ r t", "s t ɑ r t ə d"),
    ]
    pairs = [(parse_segments(s), parse_segments(t)) for s, t in raw_pairs]
    clf = VowelClassifier(use_phon_features=True).fit([x for x, _ in pairs])
    scorer = DirichletEscapeScorer.fit_from_pairs(pairs, classifier=clf)

    # Verify all three regular allomorphs + unseen velar fricative [x] in `out-Bach`!
    assert scorer.score_pair(parse_segments("b l e f"), parse_segments("b l e f t")).exact_match == 1.0
    assert scorer.score_pair(parse_segments("s m ɛ r g"), parse_segments("s m ɛ r g d")).exact_match == 1.0
    assert scorer.score_pair(parse_segments("g l i d"), parse_segments("g l i d ə d")).exact_match == 1.0
    assert scorer.score_pair(parse_segments("b ɑ x"), parse_segments("b ɑ x t")).exact_match == 1.0

    # Verify within-candidate `max` (NEVER `sum`!): even with dozens of nested rules matching `b l e f`,
    # `Score(bleft | blef)` MUST be strictly in (0, 1]!
    cands_blef = scorer.mgl.score_candidates(parse_segments("b l e f"))
    for y_c, sc in cands_blef.items():
        assert 0.0 <= sc <= 1.0, f"Score exceeded 1.0 (summed instead of max?): {y_c} -> {sc}"

    # Verify fast bitmask scope/hit evaluator matches brute-force `_find_rule_outputs` bit-for-bit!
    indexed = _IndexedTrainingSet(pairs, clf)
    for r in scorer.mgl.rules[:80]:
        brute_s = 0
        brute_h = 0
        for idx, (x_i, y_i) in enumerate(pairs):
            outs = _find_rule_outputs(
                r.pattern,
                r.local_envs,
                r.vtier_env,
                x_i,
                indexed.feat_seqs[idx],
                indexed.vc_seqs[idx],
                indexed.v_bit_seqs[idx],
                indexed.word_vset_masks[idx],
                indexed.no_vowel_bit,
            )
            if outs:
                brute_s |= 1 << idx
                if y_i in outs:
                    brute_h |= 1 << idx
        assert r.scope_mask == brute_s
        assert r.hit_mask == brute_h

    # Albright & Hayes (2002 §3.7) `burnt` distributional encroachment check!
    encroach_pairs = list(pairs) + [
        (parse_segments("t ɜ r n"), parse_segments("t ɜ r n d")),
        (parse_segments("l ɜ r n"), parse_segments("l ɜ r n d")),
        (parse_segments("y ɜ r n"), parse_segments("y ɜ r n d")),
        (parse_segments("w ɔ r n"), parse_segments("w ɔ r n d")),
        (parse_segments("b ɑ r n"), parse_segments("b ɑ r n d")),
        (parse_segments("f i l"), parse_segments("f i l d")),
        (parse_segments("h i l"), parse_segments("h i l d")),
        (parse_segments("b ɜ r n"), parse_segments("b ɜ r n t")),  # single encroaching exception!
    ]
    mgl_no_imp = MGLLearner(classifier=clf, enable_impugnment=False).fit(encroach_pairs)
    mgl_with_imp = MGLLearner(classifier=clf, enable_impugnment=True).fit(encroach_pairs)

    wug_son = parse_segments("p r æ n")
    bad_t = parse_segments("p r æ n t")
    good_d = parse_segments("p r æ n d")
    sc_no_imp = mgl_no_imp.score_candidates(wug_son).get(bad_t, 0.0)
    sc_with_imp = mgl_with_imp.score_candidates(wug_son).get(bad_t, 0.0)
    sc_good_d = mgl_with_imp.score_candidates(wug_son).get(good_d, 0.0)
    assert sc_with_imp < sc_no_imp
    assert sc_good_d > sc_with_imp + 0.4


# ---------------------------------------------------------------------------
# 5. Finnish & Hungarian Vowel Harmony (`~0` bits Harmonic vs. `>0` bits Disharmonic/Neutral)
# ---------------------------------------------------------------------------


def test_finnish_and_hungarian_vowel_harmony() -> None:
    # Finnish inessive `-ssa` (back) vs `-ssä` (front + pure neutral `i, e`) across varied codas
    fin_raw = [
        # Back stems (including transparent neutral `i, e` after back vowels!)
        ("t a l o", "t a l o s s a"),
        ("k a l a", "k a l a s s a"),
        ("k a t u", "k a t u s s a"),
        ("l i n n a", "l i n n a s s a"),
        ("k i r k k o", "k i r k k o s s a"),
        ("p u i s t o", "p u i s t o s s a"),
        ("m a t k a", "m a t k a s s a"),
        ("j u n a", "j u n a s s a"),
        ("p a p e r i", "p a p e r i s s a"),
        ("t u o l i", "t u o l i s s a"),
        ("k a u p u n k i", "k a u p u n k i s s a"),
        ("a s k e l", "a s k e l s s a"),
        ("t a i p a l", "t a i p a l s s a"),
        ("s a m m a l", "s a m m a l s s a"),
        ("p a i m e n", "p a i m e n s s a"),
        ("a s k a r", "a s k a r s s a"),
        ("m a n n e r", "m a n n e r s s a"),
        ("u i s t i n", "u i s t i n s s a"),
        ("k a i m a n", "k a i m a n s s a"),
        ("o p e t t a j a", "o p e t t a j a s s a"),
        ("s a t a m a", "s a t a m a s s a"),
        ("s a n o m a", "s a n o m a s s a"),
        ("v a l o", "v a l o s s a"),
        ("t a l l i", "t a l l i s s a"),
        ("k u l m a", "k u l m a s s a"),
        # Front + pure neutral stems
        ("k y l ä", "k y l ä s s ä"),
        ("m e t s ä", "m e t s ä s s ä"),
        ("t y t t ö", "t y t t ö s s ä"),
        ("p ö y t ä", "p ö y t ä s s ä"),
        ("s ä v e l", "s ä v e l s s ä"),
        ("k y t k i n", "k y t k i n s s ä"),
        ("j ä s e n", "j ä s e n s s ä"),
        ("s i e m e n", "s i e m e n s s ä"),
        ("k y y n e l", "k y y n e l s s ä"),
        ("y t i m e n", "y t i m e n s s ä"),
        ("e l ä i n", "e l ä i n s s ä"),
        ("s y d ä n", "s y d ä n s s ä"),
        ("h ä m ä h ä k k i", "h ä m ä h ä k k i s s ä"),
        ("l ä ä k ä r i", "l ä ä k ä r i s s ä"),
        ("y s t ä v ä", "y s t ä v ä s s ä"),
        ("e l ä m ä", "e l ä m ä s s ä"),
        ("k e n k ä", "k e n k ä s s ä"),
        ("p ä i v ä", "p ä i v ä s s ä"),
        ("s y k s y", "s y k s y s s ä"),
        ("h y l l y", "h y l l y s s ä"),
        ("m ö k k i", "m ö k k i s s ä"),
        ("r i s t i", "r i s t i s s ä"),
        ("k i v i", "k i v i s s ä"),
        ("n i m i", "n i m i s s ä"),
        ("m e r i", "m e r i s s ä"),
    ]
    fin_pairs = [(parse_segments(s), parse_segments(t)) for s, t in fin_raw]
    fin_clf = VowelClassifier().fit([x for x, _ in fin_pairs])

    scorer_vtier = DirichletEscapeScorer.fit_from_pairs(
        fin_pairs, classifier=fin_clf, enable_vowel_tier=True
    )
    scorer_novtier = DirichletEscapeScorer.fit_from_pairs(
        fin_pairs, classifier=fin_clf, enable_vowel_tier=False
    )

    # Held-out harmonic stems (`p a t o`, `k y n ä`, plus consonant-final `p a t o s`, `k y n ä s`
    # where an intervening consonant completely blocks local MGL!) achieve near-0 bits:
    for s_str, t_str in [
        ("p a t o", "p a t o s s a"),
        ("k y n ä", "k y n ä s s ä"),
        ("p a t o s", "p a t o s s s a"),
        ("k y n ä s", "k y n ä s s s ä"),
    ]:
        ds_vt = scorer_vtier.score_pair(parse_segments(s_str), parse_segments(t_str))
        ds_no = scorer_novtier.score_pair(parse_segments(s_str), parse_segments(t_str))
        assert ds_vt.exact_match == 1.0
        assert ds_vt.reliability > 0.85
        max_bits = 0.25 if s_str in {"p a t o", "k y n ä"} else 0.40
        assert ds_vt.nll_bits < max_bits, f"Expected near-0 bits on {s_str}, got {ds_vt.nll_bits:.4f}"
        assert ds_vt.nll_bits < ds_no.nll_bits

    # Across intervening consonants (`p a t o s`, `k y n ä s`), local MGL collapses to ~1 bit while
    # Vowel-Tier MGL cuts cross-entropy by > 0.60 bits!
    for s_cons, t_cons in [("p a t o s", "p a t o s s s a"), ("k y n ä s", "k y n ä s s s ä")]:
        ds_vt_c = scorer_vtier.score_pair(parse_segments(s_cons), parse_segments(t_cons))
        ds_no_c = scorer_novtier.score_pair(parse_segments(s_cons), parse_segments(t_cons))
        assert ds_vt_c.nll_bits < ds_no_c.nll_bits - 0.60

    for s_str, t_str in [("k a m e l i", "k a m e l i s s a"), ("v ä r i", "v ä r i s s ä")]:
        ds_vt = scorer_vtier.score_pair(parse_segments(s_str), parse_segments(t_str))
        ds_no = scorer_novtier.score_pair(parse_segments(s_str), parse_segments(t_str))
        assert ds_vt.exact_match == 1.0
        assert ds_vt.reliability > 0.90
        assert ds_vt.reliability > ds_no.reliability + 0.35
        assert ds_vt.nll_bits < ds_no.nll_bits - 0.20
    # Specifically verify `p a r t t i` (ending in `-ti` like front `risti`) fails WITHOUT
    # Vowel Tier (`exact_match == 0.0`) and succeeds WITH Vowel Tier (`exact_match == 1.0`)!
    assert (
        scorer_novtier.score_pair(
            parse_segments("p a r t t i"), parse_segments("p a r t t i s s a")
        ).exact_match
        == 0.0
    )
    assert (
        scorer_vtier.score_pair(
            parse_segments("p a r t t i"), parse_segments("p a r t t i s s a")
        ).exact_match
        == 1.0
    )

    # Disharmonic stem (`t a l ö`) triggers competing back & front Vowel-Tier rules -> `> 0.5` bits!
    ds_disharm = scorer_vtier.score_pair(parse_segments("t a l ö"), parse_segments("t a l ö s s a"))
    assert ds_disharm.nll_bits > 0.5

    # Hungarian ambiguous neutral-vowel stems (`híd -> hídnak` vs `víz -> víznek`) retain `> 0` bits!
    hun_raw = [
        ("h á z", "h á z n a k"),
        ("a b l a k", "a b l a k n a k"),
        ("a s z t a l", "a s z t a l n a k"),
        ("p a p í r", "p a p í r n a k"),
        ("h í d", "h í d n a k"),
        ("ny í l", "ny í l n a k"),
        ("s í r", "s í r n a k"),
        ("k e r t", "k e r t n e k"),
        ("f ü l", "f ü l n e k"),
        ("t ü k ö r", "t ü k ö r n e k"),
        ("v í z", "v í z n e k"),
        ("sz í v", "sz í v n e k"),
        ("h í r", "h í r n e k"),
    ]
    hun_pairs = [(parse_segments(s), parse_segments(t)) for s, t in hun_raw]
    hun_clf = VowelClassifier().fit([x for x, _ in hun_pairs])
    hun_scorer = DirichletEscapeScorer.fit_from_pairs(hun_pairs, classifier=hun_clf)

    # Harmonic back stem `k a p u` vs genuinely ambiguous neutral `í` stem `c í m`
    ds_hun_harm = hun_scorer.score_pair(parse_segments("k a r"), parse_segments("k a r n a k"))
    ds_hun_ambig = hun_scorer.score_pair(parse_segments("c í m"), parse_segments("c í m n e k"))
    assert ds_hun_harm.exact_match == 1.0
    assert ds_hun_ambig.nll_bits > 0.6
    assert ds_hun_ambig.nll_bits > ds_hun_harm.nll_bits + 0.3


# ---------------------------------------------------------------------------
# 6. German Circumfix `ge_t`
# ---------------------------------------------------------------------------


def test_german_circumfix_ge_t() -> None:
    deu_raw = [
        ("m a c h", "g e m a c h t"),
        ("s a g", "g e s a g t"),
        ("l a c h", "g e l a c h t"),
        ("k o c h", "g e k o c h t"),
        ("s p i e l", "g e s p i e l t"),
        ("l i e b", "g e l i e b t"),
        ("s c h i c k", "g e s c h i c k t"),
        ("f ü h r", "g e f ü h r t"),
        ("h ö r", "g e h ö r t"),
        ("w o h n", "g e w o h n t"),
        ("l e b", "g e l e b t"),
        ("z a h l", "g e z a h l t"),
    ]
    pairs = [(parse_segments(s), parse_segments(t)) for s, t in deu_raw]
    clf = VowelClassifier().fit([x for x, _ in pairs])
    scorer = DirichletEscapeScorer.fit_from_pairs(pairs, classifier=clf)

    for wug_s, wug_t in [("k a u f", "g e k a u f t"), ("l e r n", "g e l e r n t"), ("f r a g", "g e f r a g t")]:
        ds = scorer.score_pair(parse_segments(wug_s), parse_segments(wug_t))
        assert ds.exact_match == 1.0
        assert ds.reliability > 0.85
        assert ds.nll_bits < 0.25


# ---------------------------------------------------------------------------
# 7. Proper Probability Distribution over `Sigma*` (`sum == 1.0`) & Suppletion Escape
# ---------------------------------------------------------------------------


def test_proper_distribution_sum_to_one_and_suppletion() -> None:
    """Verify `sum_{y in Sigma^{<=L}} P(y | w_s) -> 1.0` strictly from below, and finite suppletion."""
    alphabet = ("a", "b")
    train_pairs = [
        (("a",), ("a", "b")),
        (("b", "a"), ("b", "a", "b")),
        (("a", "a"), ("a", "a", "b")),
        (("b",), ("b", "a")),
        (("a", "b"), ("a", "b", "a")),
    ]
    clf = VowelClassifier().fit([x for x, _ in train_pairs])
    scorer = DirichletEscapeScorer.fit_from_pairs(
        train_pairs, classifier=clf, alphabet=alphabet, gamma=0.05
    )

    for query_x in [("a",), ("b", "a"), ("b", "b")]:
        sum_edit = 0.0
        sum_full = 0.0
        for length in range(16):
            for y_tup in itertools.product(alphabet, repeat=length):
                sum_edit += scorer.edit_model.prob(query_x, y_tup)
                sum_full += scorer.prob(query_x, y_tup)
        # Must NEVER exceed 1.0 and must converge to 1.0 within < 1e-4 by L = 15!
        assert sum_edit <= 1.0 + 1e-12
        assert sum_full <= 1.0 + 1e-12
        assert abs(1.0 - sum_edit) < 1e-4, f"P_edit sum={sum_edit}"
        assert abs(1.0 - sum_full) < 1e-4, f"P_full sum={sum_full}"

    # Verify full suppletion & unseen test characters receive finite bits and `Rel == 0.0`!
    open_scorer = DirichletEscapeScorer.fit_from_pairs(train_pairs, classifier=clf, gamma=0.05)
    ds_supp = open_scorer.score_pair(("a", "b"), ("b", "b", "b", "a", "a"))
    assert ds_supp.reliability == 0.0
    assert math.isfinite(ds_supp.nll_bits) and 5.0 < ds_supp.nll_bits < 60.0

    ds_oov = open_scorer.score_pair(("ž", "a"), ("ž", "x", "q"))
    assert ds_oov.reliability == 0.0
    assert math.isfinite(ds_oov.nll_bits) and ds_oov.nll_bits > 10.0

    # Verify `k=3` PoE properness (`sum_{y in Sigma^{<=12}} P_PoE(y) <= 1.0`)!
    s2 = DirichletEscapeScorer.fit_from_pairs(
        [(("a",), ("b", "b")), (("b",), ("b", "b"))], classifier=clf, alphabet=alphabet
    )
    s3 = DirichletEscapeScorer.fit_from_pairs(
        [(("a",), ("a", "b")), (("b",), ("b", "b"))], classifier=clf, alphabet=alphabet
    )
    scorers_map = {("c1", "ct"): scorer, ("c2", "ct"): s2, ("c3", "ct"): s3}
    dev_mat = {("c1", "ct"): 0.2, ("c2", "ct"): 0.8, ("c3", "ct"): 0.4}
    poe_mass = 0.0
    cand_cache: dict = {}
    for length in range(13):
        for y_tup in itertools.product(alphabet, repeat=length):
            lemma_f = {"c1": ("a",), "c2": ("a",), "c3": ("a",), "ct": y_tup}
            nll_b, _ = score_poe_target(
                ["c1", "c2", "c3"],
                "ct",
                lemma_f,
                lambda u, v: scorers_map[(u, v)],
                dev_mat,
                cand_cache,
            )
            poe_mass += 2.0 ** (-nll_b)
    assert poe_mass <= 1.0 + 1e-9, f"PoE violated properness: total mass = {poe_mass}"


# ---------------------------------------------------------------------------
# 8. Chu-Liu-Edmonds Spanning Arborescence Exactness (Cycle Contraction vs Brute Force & NX)
# ---------------------------------------------------------------------------


def test_chu_liu_edmonds_exact_optimality() -> None:
    rng = random.Random(2026)
    cells = ["C0", "C1", "C2", "C3", "C4"]

    for trial in range(15):
        weights: dict[tuple[str, str], float] = {}
        G = nx.DiGraph()
        for u in cells:
            for v in cells:
                if u == v:
                    continue
                # Create deliberate low-cost cycles among non-root nodes
                w = rng.uniform(0.05, 3.0)
                if trial == 0 and (u, v) in {("C1", "C2"), ("C2", "C3"), ("C3", "C1")}:
                    w = 0.01
                weights[(u, v)] = w
                G.add_edge(u, v, weight=w)

        for root in cells:
            cle = chu_liu_edmonds(cells, weights, root=root)
            # Compare against networkx Chu-Liu-Edmonds on subgraph with incoming-to-root stripped
            H = nx.DiGraph()
            for (u, v), w in weights.items():
                if v != root:
                    H.add_edge(u, v, weight=w)
            nx_arb = nx.minimum_spanning_arborescence(H, attr="weight")
            nx_cost = sum(H[u][v]["weight"] for u, v in nx_arb.edges())
            assert abs(cle.total_cost - nx_cost) < 1e-9

            # Brute-force check over ALL directed trees rooted at `root`
            non_root = [c for c in cells if c != root]
            best_bf = float("inf")
            for parents in itertools.product(cells, repeat=len(non_root)):
                p_map = dict(zip(non_root, parents))
                if any(k == v for k, v in p_map.items()):
                    continue
                # Check acyclicity to `root`
                valid = True
                for node in non_root:
                    cur = node
                    steps = 0
                    while cur != root and steps <= len(cells):
                        cur = p_map[cur]
                        steps += 1
                    if cur != root:
                        valid = False
                        break
                if valid:
                    cost_bf = sum(weights[(p_map[v], v)] for v in non_root)
                    if cost_bf < best_bf:
                        best_bf = cost_bf
            assert abs(cle.total_cost - best_bf) < 1e-9


# ---------------------------------------------------------------------------
# 9. End-to-End `evaluate_mgl`, Syncretism Normalization, Pre-Screening & Zero Leakage Guard
# ---------------------------------------------------------------------------


def test_end_to_end_evaluate_mgl_syncretism_and_leakage_guard(tmp_path: Path) -> None:
    lex_dir = tmp_path / "lexicons"
    sp_dir = tmp_path / "splits"
    res_dir = tmp_path / "results"
    lex_dir.mkdir()
    sp_dir.mkdir()

    unit_id = "finn1318_N_orth"
    lex_csv = lex_dir / f"{unit_id}.csv"
    sp_csv = sp_dir / f"{unit_id}_uniform120.csv"

    # Create 25 synthetic lemmas across 4 cells (`NOM`, `GEN`, `INE`, `ELA`), where:
    # - `GEN` reveals the stem vowel while `NOM` has a syncretic consonant truncation on some stems
    #   so `GEN` is strictly the optimal Chu-Liu-Edmonds base `r*`!
    # - Test lemmas include a sibling-syncretic lemma (`INE == ELA`, `|s_l| = 3`) and a completely
    #   indeclinable syncretic lemma (`NOM == GEN == INE == ELA`, `|s_l| = 1 -> 0.0` bits)!
    rng = random.Random(7)
    cons = ["t", "k", "p", "m", "n", "l", "r", "s", "v", "h"]
    lex_rows: list[dict[str, object]] = []

    for idx in range(25):
        lid = f"L{idx:03d}"
        gid = f"G{idx:03d}"
        is_back = idx % 2 == 0
        c1 = cons[idx % len(cons)]
        c2 = cons[(idx * 3 + 1) % len(cons)]
        v1 = "a" if is_back else "ä"
        v2 = "o" if is_back else "ö"
        stem_segs = (c1, v1, c2, v2)

        if idx == 24:
            # Fully syncretic indeclinable test lemma (`|s_l| == 1`)
            forms = {c: stem_segs for c in ("NOM", "GEN", "INE", "ELA")}
            shape = "0_0_0_0"
        elif idx == 23:
            # Sibling-syncretic test lemma (`INE == ELA`, `|s_l| == 3`)
            ine = (*stem_segs, "s", "s", "a" if is_back else "ä")
            forms = {
                "NOM": stem_segs[:-1],
                "GEN": (*stem_segs, "n"),
                "INE": ine,
                "ELA": ine,
            }
            shape = "0_1_2_2"
        else:
            forms = {
                "NOM": stem_segs[:-1],  # stem-final vowel deleted in NOM -> ambiguous!
                "GEN": (*stem_segs, "n"),  # preserves stem-final vowel -> optimal base!
                "INE": (*stem_segs, "s", "s", "a" if is_back else "ä"),
                "ELA": (*stem_segs, "s", "t", "a" if is_back else "ä"),
            }
            shape = "0_1_2_3"

        for cell_name, segs in forms.items():
            lex_rows.append(
                {
                    "unit_id": unit_id,
                    "glottocode": "finn1318",
                    "iso639_3": "fin",
                    "pos": "N",
                    "representation": "orth",
                    "source_id": "synthetic_test",
                    "lemma_id": lid,
                    "lemma_label": "".join(forms["NOM"]),
                    "leakage_group_id": gid,
                    "cell": cell_name,
                    "raw_form": "".join(segs),
                    "segmented_form": " ".join(segs),
                    "variant_idx": 0,
                    "shape_id": shape,
                }
            )

    with lex_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(lex_rows[0].keys()))
        writer.writeheader()
        writer.writerows(lex_rows)

    # Write clean 1-draw, 2-fold split manifest (`L000..L019` train = 16 subtrain + 4 dev, `L020..L024` test)
    sp_rows: list[dict[str, object]] = []
    for fold_id in (0, 1):
        for idx in range(25):
            lid = f"L{idx:03d}"
            gid = f"G{idx:03d}"
            if idx >= 20:
                role, inner = "test", "test"
            elif (idx + fold_id) % 5 == 0:
                role, inner = "train", "dev"
            else:
                role, inner = "train", "subtrain"
            sp_rows.append(
                {
                    "draw": 0,
                    "fold": fold_id,
                    "lemma_id": lid,
                    "leakage_group_id": gid,
                    "role": role,
                    "inner_role": inner,
                }
            )

    with sp_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sp_rows)

    u_out, l_out = run_regime_evaluation(
        lexicons_dir=lex_dir,
        splits_dir=sp_dir,
        results_dir=res_dir,
        regime="uniform120",
        max_workers=1,
        resume=False,
    )
    assert u_out.exists() and l_out.exists()

    with u_out.open("r", encoding="utf-8") as fh:
        u_rows = list(csv.DictReader(fh))
    with l_out.open("r", encoding="utf-8") as fh:
        l_rows = list(csv.DictReader(fh))

    assert tuple(u_rows[0].keys()) == UNIT_OUTCOME_COLUMNS
    assert tuple(l_rows[0].keys()) == LEMMA_OUTCOME_COLUMNS
    assert len(u_rows) == 1
    assert len(l_rows) == 5

    by_lid = {r["lemma_id"]: r for r in l_rows}
    # Indeclinable 1-form lemma `L024` MUST have n_distinct_forms == 1 and 0.0 bits!
    assert int(by_lid["L024"]["n_distinct_forms"]) == 1
    assert float(by_lid["L024"]["h_tree_lemma"]) == 0.0
    assert float(by_lid["L024"]["acc_tree_lemma"]) == 1.0
    # Sibling-syncretic lemma `L023` MUST have n_distinct_forms == 3 (`|s_l| - 1 == 2`)!
    assert int(by_lid["L023"]["n_distinct_forms"]) == 3

    # ZERO TEST PEEK CHECK: corrupting all `test` target forms into random suppletive gibberish
    # MUST NOT change `base_cell_modal` (selected strictly on `subtrain -> dev`!)
    modal_clean = u_rows[0]["base_cell_modal"]
    assert modal_clean in {"GEN", "INE", "ELA"}  # informative vowel-preserving base, NOT truncated NOM!

    lex_obj = load_unit_lexicon(lex_csv)
    folds_obj = load_split_manifest(sp_csv, lex_obj)
    corrupted_paras = dict(lex_obj.paradigms)
    for idx in range(20, 25):
        tlid = f"L{idx:03d}"
        corrupted_paras[tlid] = {
            "NOM": ("z", "q"),
            "GEN": ("x", "w", "v"),
            "INE": ("m", "p", "r"),
            "ELA": ("k", "t", "f"),
        }
    corrupted_lex = type(lex_obj)(
        unit_id=lex_obj.unit_id,
        glottocode=lex_obj.glottocode,
        pos=lex_obj.pos,
        representation=lex_obj.representation,
        cells=lex_obj.cells,
        paradigms=corrupted_paras,
        lemma_to_group=lex_obj.lemma_to_group,
        sound_inventory=lex_obj.sound_inventory,
    )
    u_corrupt, _ = evaluate_unit(corrupted_lex, folds_obj, regime="uniform120")
    assert u_corrupt["base_cell_modal"] == modal_clean

    # LEAKAGE GUARD CHECK: overlapping `leakage_group_id` between `train` and `test` MUST raise ValueError!
    leaky_sp_csv = sp_dir / f"{unit_id}_leaky.csv"
    leaky_lex_rows = list(lex_rows)
    # Make L020 share leakage_group_id `G000` with train lemma L000!
    for r in leaky_lex_rows:
        if r["lemma_id"] == "L020":
            r["leakage_group_id"] = "G000"
    leaky_lex_csv = lex_dir / "leaky_unit.csv"
    with leaky_lex_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(leaky_lex_rows[0].keys()))
        writer.writeheader()
        writer.writerows(leaky_lex_rows)
    leaky_sp_rows = [dict(r, leakage_group_id=("G000" if r["lemma_id"] == "L020" else r["leakage_group_id"])) for r in sp_rows]
    with leaky_sp_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(leaky_sp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(leaky_sp_rows)

    leaky_lex_loaded = load_unit_lexicon(leaky_lex_csv)
    with pytest.raises(ValueError, match="LEAKAGE"):
        load_split_manifest(leaky_sp_csv, leaky_lex_loaded)

    # LARGE PARADIGM (`|C_distinct| = 45 > 40`) PRE-SCREENING CHECK:
    big_cells = [f"CELL_{i:02d}" for i in range(45)]
    big_subtrain = [
        {c: ("t", "a", "l", "o", f"s{i % 5}") for i, c in enumerate(big_cells)}
        for _ in range(12)
    ]
    cand_edges, k_pool = prescreen_candidate_edges(big_cells, big_subtrain, top_k_parents=12, seed_key="big45")
    assert len(cand_edges) < 45 * 44
    assert len(k_pool) == 14
    # Verify every root among the top candidate roots can reach all 45 cells
    dummy_dev = {e: 0.5 for e in cand_edges}
    arb45 = select_optimal_arborescence(big_cells, dummy_dev)
    assert len(arb45.parent_of) == 44


# ---------------------------------------------------------------------------
# 10. Canonical `SPLIT_COLUMNS`, `cell_maps/{unit}_sounds.csv`, Malouf & NFC Hangul Harmony
# ---------------------------------------------------------------------------


def test_split_columns_cell_maps_sounds_malouf_and_hangul_harmony(tmp_path: Path) -> None:
    """Verify `SPLIT_COLUMNS` (`draw_idx`, `fold_idx`), `cell_maps/` sounds auto-discovery,
    Malouf (`VH1/VH2`, `HPOS/HNEG/HNEUT`) token handling, and NFC Hangul `-아/-어` vowel harmony.
    """
    clf = VowelClassifier()

    # 1. Malouf archiphonemes (`VH1`, `VH2`, ...) & stem harmony tags (`HNEG`, `HPOS`, `HNEUT`)
    for vh_tok in ("VH1", "VH2", "VH12"):
        assert clf.classify(vh_tok) == "V"
        assert clf.segment_vowels(vh_tok) == (vh_tok,)
    for h_tag in ("HNEG", "HPOS", "HNEUT"):
        assert clf.classify(h_tag) == "C"
        assert clf.segment_vowels(h_tag) == ()

    assert clf.project_vowel_tier(["HPOS", "k", "y", "l", "ä", "s", "s", "VH1"]) == (
        "y",
        "ä",
        "VH1",
    )
    # Ensure `HPOS` before initial `y` (`HPOS y s t ä v ä`) does NOT corrupt Sukhotin `y -> V`!
    malouf_fin_clf = VowelClassifier().fit(
        [
            ("HPOS", "y", "s", "t", "ä", "v", "ä"),
            ("HPOS", "k", "y", "l", "ä", "s", "s", "VH1"),
            ("HNEG", "t", "a", "l", "o", "s", "s", "VH1"),
        ]
    )
    assert malouf_fin_clf.classify("y") == "V"
    assert malouf_fin_clf.classify("VH1") == "V"
    assert malouf_fin_clf.classify("HPOS") == "C"

    # 2. Precomposed NFC Hangul syllable blocks: keep full syllables on local tier while projecting
    #    medial Jungseong vowel `chr(0x1161 + ((ord(ch) - 0xAC00) // 28) % 21)` on Vowel Tier!
    assert clf.classify("막") == "C"
    assert clf.project_vowel_tier(["막", "다"]) == ("\u1161", "\u1161")  # ㅏ, ㅏ
    assert clf.project_vowel_tier(["먹", "다"]) == ("\u1165", "\u1161")  # ㅓ, ㅏ
    assert clf.project_vowel_tier(["집", "다"]) == ("\u1175", "\u1161")  # ㅣ, ㅏ

    # Korean `-다 -> -아/-어` vowel harmony across distinct NFC Hangul stem blocks (`막/밥/보` vs `먹/집/울`)
    kor_raw = [
        ("막 다", "막 아"),
        ("밥 다", "밥 아"),
        ("깎 다", "깎 아"),
        ("받 다", "받 아"),
        ("앉 다", "앉 아"),
        ("보 다", "보 아"),
        ("노 다", "노 아"),
        ("좁 다", "좁 아"),
        ("먹 다", "먹 어"),
        ("집 다", "집 어"),
        ("울 다", "울 어"),
        ("읽 다", "읽 어"),
        ("입 다", "입 어"),
        ("웃 다", "웃 어"),
        ("죽 다", "죽 어"),
        ("믿 다", "믿 어"),
    ]
    kor_pairs = [(parse_segments(s), parse_segments(t)) for s, t in kor_raw]
    kor_clf = VowelClassifier().fit([x for x, _ in kor_pairs])
    kor_vtier = DirichletEscapeScorer.fit_from_pairs(
        kor_pairs, classifier=kor_clf, enable_vowel_tier=True
    )
    kor_novtier = DirichletEscapeScorer.fit_from_pairs(
        kor_pairs, classifier=kor_clf, enable_vowel_tier=False
    )

    # Held-out stems `잡 다 -> 잡 아` (ㅏ) and `신 다 -> 신 어` (ㅣ) have novel stem syllable blocks
    # unseen in training (`잡`, `신`), so local-only MGL mixes `-아/-어` while Vowel-Tier MGL gets 100%!
    ds_jap_vt = kor_vtier.score_pair(parse_segments("잡 다"), parse_segments("잡 아"))
    ds_jap_no = kor_novtier.score_pair(parse_segments("잡 다"), parse_segments("잡 아"))
    ds_sin_vt = kor_vtier.score_pair(parse_segments("신 다"), parse_segments("신 어"))
    ds_sin_no = kor_novtier.score_pair(parse_segments("신 다"), parse_segments("신 어"))
    assert ds_jap_vt.exact_match == 1.0 and ds_sin_vt.exact_match == 1.0
    assert ds_jap_vt.nll_bits < ds_jap_no.nll_bits - 0.50
    assert ds_sin_vt.nll_bits < ds_sin_no.nll_bits - 0.50

    # Verify fast bitmask evaluator matches brute-force `_find_rule_outputs` on Hangul too!
    kor_indexed = _IndexedTrainingSet(kor_pairs, kor_clf)
    for r in kor_vtier.mgl.rules:
        b_s, b_h = 0, 0
        for idx, (x_i, y_i) in enumerate(kor_pairs):
            outs = _find_rule_outputs(
                r.pattern,
                r.local_envs,
                r.vtier_env,
                x_i,
                kor_indexed.feat_seqs[idx],
                kor_indexed.vc_seqs[idx],
                kor_indexed.v_bit_seqs[idx],
                kor_indexed.word_vset_masks[idx],
                kor_indexed.no_vowel_bit,
            )
            if outs:
                b_s |= 1 << idx
                if y_i in outs:
                    b_h |= 1 << idx
        assert r.scope_mask == b_s and r.hit_mask == b_h

    # 3. `cell_maps/{unit_id}_sounds.csv` auto-discovery on `phon` units + exact `SPLIT_COLUMNS`
    data_root = tmp_path / "data"
    lex_dir = data_root / "lexicons"
    cell_maps_dir = data_root / "cell_maps"
    splits_dir = data_root / "splits"
    lex_dir.mkdir(parents=True)
    cell_maps_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    sounds_text = (
        "sound_id,CLTS_id,label,syllabic,consonantal,voice,continuant\n"
        "ɑ,ɑ,unrounded open back vowel,1,0,1,1\n"
        "p,p,voiceless bilabial stop consonant,0,1,0,0\n"
        "f,f,voiceless labio-dental fricative consonant,0,1,0,1\n"
        "x,x,voiceless velar fricative consonant,0,1,0,1\n"
        "t,t,voiceless alveolar stop consonant,0,1,0,0\n"
    )
    for uid, rep in [("test1234_N_phon", "phon"), ("test1234_N_orth", "orth")]:
        (cell_maps_dir / f"{uid}_sounds.csv").write_text(sounds_text, encoding="utf-8")
        rows = []
        for idx in range(10):
            lid = f"L{idx:02d}"
            gid = f"G{idx:02d}"
            c_end = "f" if idx % 2 == 0 else "p"
            for cell_name, segs in [("NOM", ("p", "ɑ", c_end)), ("ACC", ("p", "ɑ", c_end, "t"))]:
                rows.append(
                    {
                        "unit_id": uid,
                        "glottocode": "test1234",
                        "iso639_3": "tst",
                        "pos": "N",
                        "representation": rep,
                        "source_id": "paralex_test",
                        "lemma_id": lid,
                        "lemma_label": "".join(segs),
                        "leakage_group_id": gid,
                        "cell": cell_name,
                        "raw_form": " ".join(segs) if rep == "phon" else "".join(segs),
                        "segmented_form": " ".join(segs),
                        "variant_idx": 0,
                        "shape_id": "0_1",
                    }
                )
        lex_file = lex_dir / f"{uid}.csv"
        with lex_file.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    phon_lex = load_unit_lexicon(lex_dir / "test1234_N_phon.csv")
    orth_lex = load_unit_lexicon(lex_dir / "test1234_N_orth.csv")
    assert phon_lex.sound_inventory is not None
    assert phon_lex.sound_inventory.vc_of("ɑ") == "V"
    assert {"-syllabic", "+continuant", "-voice"}.issubset(
        phon_lex.sound_inventory.features_of("x")
    )
    # Paired `orth` unit must NOT auto-attach IPA phoneme table from `cell_maps/`
    assert orth_lex.sound_inventory is None

    # Exact `SPLIT_COLUMNS` (`unit_id,regime,draw_idx,fold_idx,lemma_id,leakage_group_id,role,inner_role`)
    split_cols = [
        "unit_id",
        "regime",
        "draw_idx",
        "fold_idx",
        "lemma_id",
        "leakage_group_id",
        "role",
        "inner_role",
    ]
    sp_file = splits_dir / "test1234_N_phon_uniform120.csv"
    sp_rows = []
    for idx in range(10):
        lid = f"L{idx:02d}"
        gid = f"G{idx:02d}"
        role = "test" if idx >= 8 else "train"
        inner = "test" if idx >= 8 else ("dev" if idx >= 6 else "subtrain")
        sp_rows.append(
            {
                "unit_id": "test1234_N_phon",
                "regime": "uniform120",
                "draw_idx": 0,
                "fold_idx": 0,
                "lemma_id": lid,
                "leakage_group_id": gid,
                "role": role,
                "inner_role": inner,
            }
        )
    with sp_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=split_cols)
        writer.writeheader()
        writer.writerows(sp_rows)

    folds = load_split_manifest(sp_file, phon_lex)
    assert len(folds) == 1
    assert folds[0].draw == 0 and folds[0].fold == 0
    assert len(folds[0].subtrain_lemmas) == 6
    assert len(folds[0].dev_lemmas) == 2
    assert len(folds[0].test_lemmas) == 2

    # Missing draw/fold column must raise a clear ValueError rather than KeyError
    bad_sp = splits_dir / "bad_split.csv"
    bad_sp.write_text("unit_id,lemma_id,role,inner_role\ntest1234_N_phon,L00,train,subtrain\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing draw column"):
        load_split_manifest(bad_sp, phon_lex)
