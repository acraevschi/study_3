"""Unit tests for Track 2 open-corpus token frequency & paradigm matching pipeline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyses.paradigm.frequencies.fetch_frequencies import (
    corpus_aliases_for_iso,
    parse_conllu_stream,
    parse_lcc_words_bytes,
)
from analyses.paradigm.frequencies.match_paradigms import (
    FREQ_UNIT_COLUMNS,
    IsoFrequencyBundle,
    disambiguate_and_aggregate,
    generate_topfreq120_split,
    load_iso_frequency_bundle,
    match_all_units,
    normalize_surface,
)


def test_normalize_surface_multiscript_and_nfc() -> None:
    # NFD vs NFC equivalence
    nfd = "cafe\u0301"
    nfc = "caf\u00e9"
    assert normalize_surface(nfd, "fra") == normalize_surface(nfc, "fra") == "café"

    # Turkic dotted/dotless I handling (and Python İ.casefold() -> i\u0307 cleanup)
    assert normalize_surface("İstanbul", "tur") == "istanbul"
    assert normalize_surface("IŞIK", "tur") == "ışık"
    assert normalize_surface("İstanbul", "eng") == "istanbul"

    # Greek final sigma unification
    assert normalize_surface("ΛΌΓΟΣ", "ell") == normalize_surface("λόγος", "ell")

    # Cyrillic & Armenian casefolding
    assert normalize_surface("Читают", "rus") == "читают"
    assert normalize_surface("ԿԱՐԴԱԼ", "hye") == "կարդալ"

    # Abjad tatweel removal & optional vocalization stripping
    assert normalize_surface("كِتَـابٌ", "arb", strip_abjad_marks=False) == "كِتَابٌ"
    assert normalize_surface("كِتَـابٌ", "arb", strip_abjad_marks=True) == "كتاب"


def test_iso_aliases_never_conflate_arabic_vernaculars() -> None:
    # Standard Arabic maps to arb/ara, whereas Egyptian/Gulf vernaculars stay distinct
    assert set(corpus_aliases_for_iso("arb")) == {"arb", "ara"}
    assert corpus_aliases_for_iso("arz") == ["arz"]
    assert corpus_aliases_for_iso("afb") == ["afb"]
    # Standard Estonian & Western Persian aliases
    assert "est" in corpus_aliases_for_iso("ekk")
    assert "fas" in corpus_aliases_for_iso("pes")


def test_parse_conllu_stream_handles_mwt_empty_and_notext() -> None:
    sample_conllu = [
        "# sent_id = 1",
        "1-2\tdello\t_\t_\t_\t_\t_\t_\t_",
        "1\tdi\tdi\tADP\t_\t_\t2\tcase\t_",
        "2\til\til\tDET\t_\t_\t3\tdet\t_",
        "3\tlibro\tlibro\tNOUN\t_\t_\t0\troot\t_",
        "3.1\tè\tessere\tAUX\t_\t_\t_\t_\t_",
        "4\tè\tessere\tAUX\t_\t_\t3\tcop\t_",
        "5\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_",
    ]
    counts, total_words = parse_conllu_stream(sample_conllu)
    # 4 syntactic word tokens (di, il, libro, è); 1-2 MWT, 3.1 ellipsis, and PUNCT excluded
    assert total_words == 4
    assert counts[("libro", "NOUN", "libro")] == 1
    assert counts[("essere", "AUX", "è")] == 1

    # No-text stub treebank returns 0
    notext = ["1\t_\t_\tVERB\t_\t_\t0\troot\t_"]
    c_empty, n_empty = parse_conllu_stream(notext)
    assert n_empty == 0 and len(c_empty) == 0


def test_parse_lcc_words_bytes_handles_quotes_and_punct() -> None:
    # Both 3-column standard LCC and 4-column community_2017 LCC formats
    raw = '1\t!\t130\n2\t"\t1246\n3\t1234\t50\n4\tcammina\t420\n5\tessere\tessere\t1580\n'.encode("utf-8")
    n_tok, n_typ = parse_lcc_words_bytes(raw)
    assert n_tok == 2000
    assert n_typ == 2


def test_syncretism_no_double_counting_and_ud_plus_em_disambiguation() -> None:
    # Build synthetic Verb lexicon with:
    # - L1 ('essere'): high-freq, unambiguous 'è' (5000), 'siamo' (1000), UD-resolved homograph 'sono' (2000)
    # - L2 ('sonare'): rare, unambiguous 'suona' (10), shares 'sono' (2000) -> UD ratio 99:1
    # - L3 ('walk'): syncretic 'walked' in both PST and PTCP (400), unambiguous 'walks' (600),
    #                and EM-resolved homograph 'walk' (1000) shared with rare L4 ('walk_rare')
    # - L4 ('walk_rare'): unambiguous 'walketh' (2), shares 'walk' (1000) with L3 (no UD entry)
    # - L5 ('zero_a') & L6 ('zero_b'): share ONLY homograph 'twin' (100), other cells unattested (0)
    # - L7 ('unseen'): completely unattested in corpus (0 tokens)
    rows = [
        # L1 essere
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L1", "essere", "G1", "PRS;3;SG", "è", "è", 0, "sh1"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L1", "essere", "G1", "PRS;1;PL", "siamo", "s i a m o", 0, "sh1"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L1", "essere", "G1", "PRS;1;SG", "sono", "s o n o", 0, "sh1"),
        # L2 sonare
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L2", "sonare", "G2", "PRS;3;SG", "suona", "s u o n a", 0, "sh2"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L2", "sonare", "G2", "PRS;1;SG", "sono", "s o n o", 0, "sh2"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L2", "sonare", "G2", "PRS;1;PL", "soniamo", "s o n i a m o", 0, "sh2"),
        # L3 walk (syncretic walked across 2 cells!)
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L3", "walk", "G3", "NFIN", "walk", "w a l k", 0, "sh3"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L3", "walk", "G3", "PRS;3;SG", "walks", "w a l k s", 0, "sh3"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L3", "walk", "G3", "PST", "walked", "w a l k e d", 0, "sh3"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L3", "walk", "G3", "PTCP", "walked", "w a l k e d", 0, "sh3"),
        # L4 walk_rare
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L4", "walk_rare", "G4", "NFIN", "walk", "w a l k", 0, "sh4"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L4", "walk_rare", "G4", "PRS;3;SG", "walketh", "w a l k e t h", 0, "sh4"),
        # L5 & L6 pure tie
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L5", "zero_a", "G5", "C1", "twin", "t w i n", 0, "sh5"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L5", "zero_a", "G5", "C2", "twina", "t w i n a", 0, "sh5"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L6", "zero_b", "G6", "C1", "twin", "t w i n", 0, "sh6"),
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L6", "zero_b", "G6", "C2", "twinb", "t w i n b", 0, "sh6"),
        # L7 unattested
        ("u_v", "ital1282", "ita", "V", "orth", "s", "L7", "unseen", "G7", "C1", "xyzzy", "x y z z y", 0, "sh7"),
    ]
    cols = [
        "unit_id", "glottocode", "iso639_3", "pos", "representation", "source_id",
        "lemma_id", "lemma_label", "leakage_group_id", "cell", "raw_form",
        "segmented_form", "variant_idx", "shape_id",
    ]
    lex_df = pd.DataFrame(rows, columns=cols)

    lcc_counts = Counter({
        "è": 4900,
        "siamo": 1000,
        "sono": 1900,
        "suona": 10,
        "walk": 1000,
        "walks": 600,
        "walked": 400,
        "walketh": 2,
        "twin": 100,
    })
    ud_surface = Counter({"è": 100, "sono": 100})
    ud_lemma_pos = Counter({
        ("V", "essere", "è"): 100,
        ("V", "essere", "sono"): 99,
        ("V", "sonare", "sono"): 1,
    })

    bundle = IsoFrequencyBundle(
        iso="ita",
        lcc_surface=lcc_counts,
        lcc_total_tokens=990_000,
        ud_surface=ud_surface,
        ud_lemma_pos=ud_lemma_pos,
        ud_total_tokens=10_000,
        wf_lang=None,
        source_ids=["lcc:test", "ud_v2.15:test"],
    )

    out_df, meta = disambiguate_and_aggregate(lex_df, bundle, pos="V", iso="ita", em_iterations=5)
    assert list(out_df.columns) == FREQ_UNIT_COLUMNS
    assert not out_df.isna().any().any()
    assert np.all(np.isfinite(out_df["log10_freq_lemma"]))

    by_id = out_df.set_index("lemma_id")

    # 1. Exact mass conservation across all 7 lemmas (syncretic 'walked'=400 counted ONCE, not 800!)
    expected_total = 5000 + 1000 + 2000 + 10 + 1000 + 600 + 400 + 2 + 100
    assert pytest.approx(out_df["freq_lemma_tokens"].sum(), rel=1e-6) == expected_total

    # 2. UD ratio disambiguation on 'sono' (99% essere = 1980, 1% sonare = 20)
    assert pytest.approx(by_id.loc["L1", "freq_lemma_tokens"], abs=1e-4) == 5000 + 1000 + 1980
    assert pytest.approx(by_id.loc["L2", "freq_lemma_tokens"], abs=1e-4) == 10 + 20

    # 3. Syncretic cell attestation count vs single token count on L3 ('walk')
    assert by_id.loc["L3", "n_cells_attested"] == 4  # NFIN, PRS;3;SG, PST, PTCP all attested!
    # Exact 5-step EM recurrence: L4_0=502 -> 252.749 -> 128.249 -> 66.061 -> 35.000 -> 19.481053
    f3, f4 = 1500.0, 502.0
    for _ in range(5):
        g3 = f3 / (f3 + f4)
        f3, f4 = 1000.0 + 1000.0 * g3, 2.0 + 1000.0 * (1.0 - g3)
    assert pytest.approx(by_id.loc["L3", "freq_lemma_tokens"], abs=1e-5) == f3
    assert pytest.approx(by_id.loc["L4", "freq_lemma_tokens"], abs=1e-5) == f4

    # 4. Symmetric tie ('twin' = 100) stays 50/50 without NaN
    assert pytest.approx(by_id.loc["L5", "freq_lemma_tokens"], abs=1e-6) == 50.0
    assert pytest.approx(by_id.loc["L6", "freq_lemma_tokens"], abs=1e-6) == 50.0

    # 5. Unattested lemma has 0 tokens, finite log10_freq_lemma, and primary_freq_source == 'unattested'
    assert by_id.loc["L7", "freq_lemma_tokens"] == 0.0
    assert by_id.loc["L7", "primary_freq_source"] == "unattested"
    assert by_id.loc["L7", "log10_freq_lemma"] == pytest.approx(0.0, abs=1e-6)  # (0+1)*1e6 / 1e6 = 1 -> log10=0


def test_topfreq120_split_threshold_cap_and_zero_leakage(tmp_path: Path) -> None:
    # Case 1: Only 180 attested groups (< 200 threshold) -> None
    small_rows = [
        {
            "unit_id": "test_u",
            "lemma_id": f"L_{i}",
            "leakage_group_id": f"G_{i}",
            "freq_lemma_tokens": float(1000 - i) if i < 180 else 0.0,
            "freq_lemma_pmw": float(1000 - i) if i < 180 else 0.0,
            "log10_freq_lemma": 1.0,
            "n_cells_attested": 2,
            "primary_freq_source": "lcc",
        }
        for i in range(220)
    ]
    small_df = pd.DataFrame(small_rows)
    assert generate_topfreq120_split(small_df, {}, splits_dir=tmp_path) is None

    # Case 2: 650 attested groups (with multi-lemma group G_0 containing L_0a=99999 and L_0b=5)
    # Also create a reference uniform120 split file with custom column names to test schema mirroring!
    ref_split = pd.DataFrame(
        [
            {
                "unit_id": "test_u",
                "draw": 0,
                "fold": 0,
                "lemma_id": "L_0a",
                "leakage_group_id": "G_0",
                "role": "train",
                "inner_role": "subtrain",
            }
        ]
    )
    ref_split.to_csv(tmp_path / "test_u_uniform120.csv", index=False)

    big_rows = [
        {
            "unit_id": "test_u",
            "lemma_id": "L_0_rare_cognate",
            "leakage_group_id": "G_0",
            "freq_lemma_tokens": 5.0,
            "freq_lemma_pmw": 5.0,
            "log10_freq_lemma": 0.8,
            "n_cells_attested": 1,
            "primary_freq_source": "lcc",
        },
        {
            "unit_id": "test_u",
            "lemma_id": "L_0_core",
            "leakage_group_id": "G_0",
            "freq_lemma_tokens": 99999.0,
            "freq_lemma_pmw": 99999.0,
            "log10_freq_lemma": 5.0,
            "n_cells_attested": 4,
            "primary_freq_source": "lcc",
        },
    ]
    for i in range(1, 650):
        freq_val = float(10_000 - i)  # G_0..G_499 are top 500; G_500..G_649 are bottom 150
        big_rows.append(
            {
                "unit_id": "test_u",
                "lemma_id": f"L_{i}",
                "leakage_group_id": f"G_{i}",
                "freq_lemma_tokens": freq_val,
                "freq_lemma_pmw": freq_val,
                "log10_freq_lemma": np.log10(freq_val + 1.0),
                "n_cells_attested": 3,
                "primary_freq_source": "lcc",
            }
        )
    big_df = pd.DataFrame(big_rows)
    labels = {r["lemma_id"]: r["lemma_id"] for r in big_rows}

    split_df = generate_topfreq120_split(big_df, labels, splits_dir=tmp_path)
    assert split_df is not None
    # Mirrors reference split columns exactly
    assert list(split_df.columns) == [
        "unit_id", "draw", "fold", "lemma_id", "leakage_group_id", "role", "inner_role"
    ]
    assert len(split_df) == 5 * 5 * 150

    # Bottom 150 groups (G_500..G_649) must NEVER appear in top-500 pool
    low_groups = {f"G_{i}" for i in range(500, 650)}
    assert set(split_df["leakage_group_id"]).isdisjoint(low_groups)

    # Whenever G_0 is sampled, its canonical lemma must be L_0_core (never L_0_rare_cognate)
    g0_rows = split_df[split_df["leakage_group_id"] == "G_0"]
    if not g0_rows.empty:
        assert set(g0_rows["lemma_id"]) == {"L_0_core"}

    # Verify exact 96 subtrain / 24 dev / 30 test and ZERO group leakage per (draw, fold)
    for (d, f), grp in split_df.groupby(["draw", "fold"]):
        assert len(grp) == 150
        assert grp["leakage_group_id"].nunique() == 150
        counts = grp["inner_role"].value_counts().to_dict()
        assert counts == {"subtrain": 96, "dev": 24, "test": 30}
        tr = set(grp.loc[grp["role"] == "train", "leakage_group_id"])
        te = set(grp.loc[grp["role"] == "test", "leakage_group_id"])
        assert tr.isdisjoint(te)


def test_end_to_end_match_all_units_on_fixture_dirs(tmp_path: Path) -> None:
    lex_dir = tmp_path / "lexicons"
    freq_dir = tmp_path / "freq_sources"
    out_dir = tmp_path / "frequencies"
    splits_dir = tmp_path / "splits"
    splits_freq_dir = tmp_path / "splits_freq"
    lex_dir.mkdir()
    (freq_dir / "lcc" / "xyz_news_2020_100K").mkdir(parents=True)
    splits_dir.mkdir()

    # Write synthetic LCC words file for fictional ISO 'qzz' (no wordfreq interference)
    words_lines = [f"{i+1}\tform_{i}_a\t{500 - i}" for i in range(230)]
    (freq_dir / "lcc" / "qzz_news_2020_100K").mkdir(parents=True)
    (freq_dir / "lcc" / "qzz_news_2020_100K" / "qzz_news_2020_100K-words.txt").write_text(
        "\n".join(words_lines) + "\n", encoding="utf-8"
    )

    lex_rows = []
    for i in range(250):
        for cell, sfx in [("C1", "a"), ("C2", "b")]:
            lex_rows.append(
                {
                    "unit_id": "qzzz1234_N_orth",
                    "glottocode": "qzzz1234",
                    "iso639_3": "qzz",
                    "pos": "N",
                    "representation": "orth",
                    "source_id": "synthetic",
                    "lemma_id": f"lem_{i}",
                    "lemma_label": f"form_{i}_a",
                    "leakage_group_id": f"grp_{i}",
                    "cell": cell,
                    "raw_form": f"form_{i}_{sfx}",
                    "segmented_form": f"f o r m {i} {sfx}",
                    "variant_idx": 0,
                    "shape_id": "s0",
                }
            )
    pd.DataFrame(lex_rows).to_csv(lex_dir / "qzzz1234_N_orth.csv", index=False)

    summary = match_all_units(
        lexicons_dir=lex_dir,
        freq_dir=freq_dir,
        freq_out_dir=out_dir,
        splits_dir=splits_dir,
        splits_freq_dir=splits_freq_dir,
    )
    assert len(summary) == 1
    assert bool(summary.loc[0, "has_freq_source"]) is True
    assert int(summary.loc[0, "n_groups_attested"]) == 230
    assert bool(summary.loc[0, "has_topfreq120_split"]) is True
    assert (out_dir / "qzzz1234_N_orth_freq.csv").exists()
    assert (splits_freq_dir / "qzzz1234_N_orth_topfreq120.csv").exists()


def test_ud_pos_share_discounts_function_word_collisions_and_script_guard(tmp_path: Path) -> None:
    # 1. Verify function-word collision ('and' = 5000 CCONJ, 0 NOUN in UD) does NOT beat real noun ('time' = 800 NOUN)
    lex_df = pd.DataFrame(
        [
            ("u_n", "stan1293", "eng", "N", "orth", "s", "L_and", "and", "G_and", "SG", "and", "a n d", 0, "s1"),
            ("u_n", "stan1293", "eng", "N", "orth", "s", "L_and", "and", "G_and", "PL", "ands", "a n d s", 0, "s1"),
            ("u_n", "stan1293", "eng", "N", "orth", "s", "L_time", "time", "G_time", "SG", "time", "t i m e", 0, "s2"),
            ("u_n", "stan1293", "eng", "N", "orth", "s", "L_time", "time", "G_time", "PL", "times", "t i m e s", 0, "s2"),
        ],
        columns=[
            "unit_id", "glottocode", "iso639_3", "pos", "representation", "source_id",
            "lemma_id", "lemma_label", "leakage_group_id", "cell", "raw_form",
            "segmented_form", "variant_idx", "shape_id",
        ],
    )
    bundle = IsoFrequencyBundle(
        iso="eng",
        lcc_surface=Counter({"and": 300_000, "time": 20_000, "times": 5_000}),
        lcc_total_tokens=1_000_000,
        ud_surface=Counter({"and": 5_000, "time": 800, "times": 200}),
        ud_lemma_pos=Counter({("N", "time", "time"): 800, ("N", "time", "times"): 200}),
        ud_total_tokens=100_000,
        wf_lang=None,
        source_ids=["lcc:eng", "ud_v2.15:eng"],
        ud_pos_surface=Counter({("N", "time"): 800, ("N", "times"): 200}),
    )
    out_df, _ = disambiguate_and_aggregate(lex_df, bundle, pos="N", iso="eng")
    by_id = out_df.set_index("lemma_id")
    assert by_id.loc["L_time", "freq_lemma_tokens"] == pytest.approx(26_000.0)
    assert by_id.loc["L_and", "freq_lemma_tokens"] < 50.0  # discounted by 0.5 / 5001!

    # 2. Verify Latin lexicon vs Cyrillic corpus (<5% loan collision) is flagged script_mismatch_excluded
    lex_dir = tmp_path / "lex"
    freq_dir = tmp_path / "freq"
    out_dir = tmp_path / "out"
    sp_dir = tmp_path / "sp"
    spf_dir = tmp_path / "spf"
    lex_dir.mkdir()
    (freq_dir / "lcc" / "qcy_web_2020_100K").mkdir(parents=True)
    sp_dir.mkdir()

    cyr_words = [f"{i+1}\tслово{i}\t{1000 - i}" for i in range(100)] + ["101\tbank\t2"]
    (freq_dir / "lcc" / "qcy_web_2020_100K" / "qcy_web_2020_100K-words.txt").write_text(
        "\n".join(cyr_words) + "\n", encoding="utf-8"
    )
    lat_rows = [
        {
            "unit_id": "qcyr1234_N_orth",
            "glottocode": "qcyr1234",
            "iso639_3": "qcy",
            "pos": "N",
            "representation": "orth",
            "source_id": "synthetic",
            "lemma_id": f"L_{i}",
            "lemma_label": "bank" if i == 0 else f"kitap{i}",
            "leakage_group_id": f"G_{i}",
            "cell": "SG",
            "raw_form": "bank" if i == 0 else f"kitap{i}",
            "segmented_form": "b a n k",
            "variant_idx": 0,
            "shape_id": "s0",
        }
        for i in range(100)
    ]
    pd.DataFrame(lat_rows).to_csv(lex_dir / "qcyr1234_N_orth.csv", index=False)

    summary = match_all_units(
        lexicons_dir=lex_dir,
        freq_dir=freq_dir,
        freq_out_dir=out_dir,
        splits_dir=sp_dir,
        splits_freq_dir=spf_dir,
    )
    assert bool(summary.loc[0, "has_freq_source"]) is False
    assert bool(summary.loc[0, "script_mismatch_excluded"]) is True
    assert not (out_dir / "qcyr1234_N_orth_freq.csv").exists()


def test_dictionary_accent_stripping_and_per_source_script_filtering(tmp_path: Path) -> None:
    # 1. Belarusian / Russian / Ukrainian stress stripping while preserving й/ў/ё/ї
    assert normalize_surface("абавя\u0301зак", "bel") == "абавязак"
    assert normalize_surface("аб’яўля\u0301ем", "bel") == "аб'яўляем"
    assert normalize_surface("ёжик і їжак", "ukr") == "ёжик і їжак"

    # 2. Lithuanian Baltic pitch accents stripped while preserving ąęįų, ė, ū, čšž
    assert normalize_surface("abãtą", "lit") == "abatą"
    assert normalize_surface("formalýbės", "lit") == "formalybės"
    assert normalize_surface("abatùs", "lit") == "abatus"
    assert normalize_surface("i\u0307\u0303rašas", "lit") == "įrašas".replace("į", "i")
    assert normalize_surface("ūkininkų čempionė", "lit") == "ūkininkų čempionė"

    # 3. Serbo-Croatian Gajica pitch/length accents stripped while preserving ć, č, š, ž, đ
    assert normalize_surface("dȃm", "hbs") == "dam"
    assert normalize_surface("dȁde", "hbs") == "dade"
    assert normalize_surface("dojàvljujēm", "hbs") == "dojavljujem"
    assert normalize_surface("gláva", "hbs") == "glava"
    assert normalize_surface("četrnaestoro ću đaka šetati", "hbs") == "četrnaestoro ću đaka šetati"

    # 4. kmr never aliases to Sorani 'kur'
    assert corpus_aliases_for_iso("kmr") == ["kmr"]
    assert "kur" in corpus_aliases_for_iso("ckb")

    # 5. Per-source script filtering prevents Cyrillic/Arabic sub-corpora from inflating Latin denominator
    fdir = tmp_path / "fsrc"
    lat_dir = fdir / "lcc" / "hbs_mixed_2014_300K"
    cyr_dir = fdir / "lcc" / "srp_news_2019_30K"
    lat_dir.mkdir(parents=True)
    cyr_dir.mkdir(parents=True)
    lat_w = lat_dir / "hbs_mixed_2014_300K-words.txt"
    cyr_w = cyr_dir / "srp_news_2019_30K-words.txt"
    lat_w.write_text("1\tdam\t400\n2\tbiti\t600\n", encoding="utf-8")
    cyr_w.write_text("1\tбити\t90000\n2\tданас\t10000\n", encoding="utf-8")

    prov = pd.DataFrame(
        [
            {
                "source_id": "lcc:hbs_mixed_2014_300K",
                "source_tier": "tier_b_leipzig_corpora",
                "iso639_3": "hbs;hrv",
                "corpus_id": "hbs_mixed_2014_300K",
                "url": "https://example.org/hbs.tar.gz",
                "version_or_commit": "2014",
                "license": "CC-BY-4.0",
                "citation": "Goldhahn 2012",
                "local_path": str(lat_w.relative_to(Path(__file__).resolve().parents[1]))
                if str(lat_w).startswith(str(Path(__file__).resolve().parents[1]))
                else "",
                "sha256": "abc",
                "archive_sha256": "abc",
                "n_tokens": 1000,
                "n_types": 2,
            }
        ]
    )
    # Test fallback directory scan with expected_script='LATIN' when both hbs_ and srp_ are in aliases
    b_lat = load_iso_frequency_bundle("hbs", freq_dir=fdir, expected_script="LATIN")
    assert b_lat.lcc_total_tokens == 1000  # 100,000 Cyrillic tokens from srp_news completely excluded!
    assert b_lat.n_script_filtered_sources >= 1
    assert "lcc:srp_news_2019_30K" not in b_lat.source_ids
