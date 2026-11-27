"""Comprehensive unit and end-to-end integrity tests for Track 2 paradigm ingestion."""

from __future__ import annotations

import hashlib
import http.server
import io
import shutil
import threading
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import analyses.paradigm.ingestion.fetch_sources as fetch_sources_mod
from analyses.paradigm.ingestion.build_lexicons import (
    _repair_romanian_noun_triples,
    is_polluted_entry,
    normalize_orth_surface,
    read_celex_english_v,
    read_morphit_records,
)
from analyses.paradigm.ingestion.fetch_sources import (
    fetch_celex_english,
    fetch_morphit,
    find_unimorph_data_files,
    resolve_analysis_links_csv,
    resolve_celex_english_path,
    resolve_morphit_path,
)
from analyses.paradigm.ingestion.grouping import (
    assign_leakage_groups,
    find_constant_affix_cognate_pairs,
    to_graphemes,
)
from analyses.paradigm.ingestion.harmony_malouf import (
    apply_malouf_abstraction,
    learn_malouf_harmony_model,
)
from analyses.paradigm.ingestion.splits import (
    SPLIT_COLUMNS,
    build_regime_splits,
    select_canonical_lemma_per_group,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "analyses" / "paradigm" / "data"

EXPECTED_LEXICON_COLS = [
    "unit_id",
    "glottocode",
    "iso639_3",
    "pos",
    "representation",
    "source_id",
    "lemma_id",
    "lemma_label",
    "leakage_group_id",
    "cell",
    "raw_form",
    "segmented_form",
    "variant_idx",
    "shape_id",
]


def _make_paradigm_df(unit_id: str, glottocode: str, pos: str, rep: str, entries: dict, cells: list[str]) -> pd.DataFrame:
    rows = []
    for lemma, forms in entries.items():
        for cell, form in zip(cells, forms):
            seg = " ".join(to_graphemes(form, rep))
            rows.append({
                "unit_id": unit_id,
                "glottocode": glottocode,
                "iso639_3": "tst",
                "pos": pos,
                "representation": rep,
                "source_id": "fixture",
                "lemma_id": f"fixture::{lemma}",
                "lemma_label": lemma,
                "cell": cell,
                "raw_form": form,
                "segmented_form": seg,
                "variant_idx": 0,
            })
    return pd.DataFrame(rows)


def test_grouping_merges_italian_porre_cognates_and_clitics():
    cells = ["NFIN", "IND;PRS;1;SG", "IND;PRS;3;SG", "IND;PST;1;SG;PFV", "IND;FUT;1;SG", "V.PTCP;PST"]
    entries = {
        "porre": ["porre", "pongo", "pone", "posi", "porrò", "posto"],
        "comporre": ["comporre", "compongo", "compone", "composi", "comporrò", "composto"],
        "imporre": ["imporre", "impongo", "impone", "imposi", "imporrò", "imposto"],
        "porsi": ["porsi", "mi pongo", "si pone", "mi posi", "mi porrò", "postosi"],
        "parlare": ["parlare", "parlo", "parla", "parlai", "parlerò", "parlato"],
        "riparlare": ["riparlare", "riparlo", "riparla", "riparlai", "riparlerò", "riparlato"],
        "parlarsi": ["parlarsi", "mi parlo", "si parla", "mi parlai", "mi parlerò", "parlatosi"],
        "dormire": ["dormire", "dormo", "dorme", "dormii", "dormirò", "dormito"],
    }
    # Add background regular verbs so productive cell affixes (-o, -a/-e, -ò, -to) are well-attested
    for stem in ("cant", "port", "mand", "torn", "pass", "rest", "bast", "cost", "ferm", "guid"):
        entries[f"{stem}are"] = [f"{stem}are", f"{stem}o", f"{stem}a", f"{stem}ai", f"{stem}erò", f"{stem}ato"]

    df = _make_paradigm_df("ital1282_V_orth", "ital1282", "V", "orth", entries, cells)
    gmap, _ = assign_leakage_groups(df, "ital1282", "V", "orth", cells)

    assert gmap["fixture::porre"] == gmap["fixture::comporre"]
    assert gmap["fixture::porre"] == gmap["fixture::imporre"]
    assert gmap["fixture::porre"] == gmap["fixture::porsi"]
    assert gmap["fixture::parlare"] == gmap["fixture::riparlare"]
    assert gmap["fixture::parlare"] == gmap["fixture::parlarsi"]
    assert gmap["fixture::porre"] != gmap["fixture::parlare"]
    assert gmap["fixture::porre"] != gmap["fixture::dormire"]


def test_grouping_merges_english_stand_understand():
    cells = ["NFIN", "PRS;3;SG", "PST", "V.PTCP;PRS", "V.PTCP;PST"]
    entries = {
        "stand": ["stand", "stands", "stood", "standing", "stood"],
        "understand": ["understand", "understands", "understood", "understanding", "understood"],
        "withstand": ["withstand", "withstands", "withstood", "withstanding", "withstood"],
    }
    for stem in ("walk", "talk", "work", "jump", "look", "help", "call", "play", "turn", "open"):
        entries[stem] = [stem, f"{stem}s", f"{stem}ed", f"{stem}ing", f"{stem}ed"]

    df = _make_paradigm_df("stan1293_V_orth", "stan1293", "V", "orth", entries, cells)
    gmap, _ = assign_leakage_groups(df, "stan1293", "V", "orth", cells)

    assert gmap["fixture::stand"] == gmap["fixture::understand"]
    assert gmap["fixture::stand"] == gmap["fixture::withstand"]
    assert gmap["fixture::stand"] != gmap["fixture::walk"]


def test_grouping_blocks_short_accidental_rhymes_across_scripts():
    # 1. Italian 2-cell nouns: 3-grapheme stems `man-` (mano) and `ett-` (etto) must NOT merge rhymes
    it_cells = ["SG", "PL"]
    it_entries = {
        "mano": ["mano", "mani"],
        "romano": ["romano", "romani"],
        "umano": ["umano", "umani"],
        "etto": ["etto", "etti"],
        "oggetto": ["oggetto", "oggetti"],
        "aspetto": ["aspetto", "aspetti"],
        "porta": ["porta", "porte"],
        "antiporta": ["antiporta", "antiporte"],  # genuine 4-grapheme stem `port-` + `anti-`
        "arco": ["arco", "archi"],
        "parco": ["parco", "parchi"],  # 1-grapheme prefix `p-` (< 2) must NOT merge
    }
    for stem in ("libr", "tavol", "strad", "piant", "finestr", "scuol"):
        if stem.endswith(("l", "r")):
            it_entries[f"{stem}o"] = [f"{stem}o", f"{stem}i"]
        else:
            it_entries[f"{stem}a"] = [f"{stem}a", f"{stem}e"]

    df_it = _make_paradigm_df("ital1282_N_orth", "ital1282", "N", "orth", it_entries, it_cells)
    gmap_it, _ = assign_leakage_groups(df_it, "ital1282", "N", "orth", it_cells)

    assert gmap_it["fixture::mano"] != gmap_it["fixture::romano"]
    assert gmap_it["fixture::mano"] != gmap_it["fixture::umano"]
    assert gmap_it["fixture::etto"] != gmap_it["fixture::oggetto"]
    assert gmap_it["fixture::etto"] != gmap_it["fixture::aspetto"]
    assert gmap_it["fixture::arco"] != gmap_it["fixture::parco"]
    assert gmap_it["fixture::porta"] == gmap_it["fixture::antiporta"]

    # 2. Cyrillic 6-cell nouns: 3-grapheme stems `тон` and `рам-` must NOT merge `питон`/`батон` or `панорама`
    ru_cells = ["NOM;SG", "GEN;SG", "DAT;SG", "INS;SG", "NOM;PL", "INS;PL"]
    ru_entries = {
        "тон": ["тон", "тона", "тону", "тоном", "тоны", "тонами"],
        "питон": ["питон", "питона", "питону", "питоном", "питоны", "питонами"],
        "батон": ["батон", "батона", "батону", "батоном", "батоны", "батонами"],
        "рама": ["рама", "рамы", "раме", "рамой", "рамы", "рамами"],
        "панорама": ["панорама", "панорамы", "панораме", "панорамой", "панорамы", "панорамами"],
        "книга": ["книга", "книги", "книге", "книгой", "книги", "книгами"],
        "аудиокнига": ["аудиокнига", "аудиокниги", "аудиокниге", "аудиокнигой", "аудиокниги", "аудиокнигами"],
    }
    for stem in ("завод", "народ", "город", "вагон", "закон", "лимон"):
        ru_entries[stem] = [stem, f"{stem}а", f"{stem}у", f"{stem}ом", f"{stem}ы", f"{stem}ами"]

    df_ru = _make_paradigm_df("russ1263_N_orth", "russ1263", "N", "orth", ru_entries, ru_cells)
    gmap_ru, _ = assign_leakage_groups(df_ru, "russ1263", "N", "orth", ru_cells)

    assert gmap_ru["fixture::тон"] != gmap_ru["fixture::питон"]
    assert gmap_ru["fixture::тон"] != gmap_ru["fixture::батон"]
    assert gmap_ru["fixture::рама"] != gmap_ru["fixture::панорама"]
    assert gmap_ru["fixture::книга"] == gmap_ru["fixture::аудиокнига"]


def test_grouping_blocks_short_stems_onset_splits_and_suffix_hubs():
    # 1. Russian 3-grapheme stem `ост-` (`ость`, min_surf=4, stripped=3) & Spanish `izar` (min_surf=3 `iza`)
    ru_cells = ["NOM;SG", "GEN;SG", "INS;SG", "GEN;PL", "DAT;PL", "INS;PL"]
    ru_entries = {
        "ость": ["ость", "ости", "остью", "остей", "остям", "остями"],
        "радость": ["радость", "радости", "радостью", "радостей", "радостям", "радостями"],
        "скорость": ["скорость", "скорости", "скоростью", "скоростей", "скоростям", "скоростями"],
    }
    for s in ("новост", "смелост", "слабост", "глупост", "гордост"):
        ru_entries[f"{s}ь"] = [f"{s}ь", f"{s}и", f"{s}ью", f"{s}ей", f"{s}ям", f"{s}ями"]
    df_ru = _make_paradigm_df("russ1263_N_orth", "russ1263", "N", "orth", ru_entries, ru_cells)
    gmap_ru, _ = assign_leakage_groups(df_ru, "russ1263", "N", "orth", ru_cells)
    assert gmap_ru["fixture::ость"] != gmap_ru["fixture::радость"]
    assert gmap_ru["fixture::ость"] != gmap_ru["fixture::скорость"]

    # 2. Pure-consonant onset splits (`gl+itch`, `st+itch`) must NOT merge into `itch`
    en_cells = ["NFIN", "PRS;3;SG", "PST", "V.PTCP;PRS"]
    en_entries = {
        "itch": ["itch", "itches", "itched", "itching"],
        "glitch": ["glitch", "glitches", "glitched", "glitching"],
        "stitch": ["stitch", "stitches", "stitched", "stitching"],
        "switch": ["switch", "switches", "switched", "switching"],
    }
    for s in ("walk", "talk", "work", "jump", "look", "help", "call", "play"):
        en_entries[s] = [s, f"{s}s", f"{s}ed", f"{s}ing"]
    df_en = _make_paradigm_df("stan1293_V_orth", "stan1293", "V", "orth", en_entries, en_cells)
    gmap_en, _ = assign_leakage_groups(df_en, "stan1293", "V", "orth", en_cells)
    assert gmap_en["fixture::itch"] != gmap_en["fixture::glitch"]
    assert gmap_en["fixture::itch"] != gmap_en["fixture::stitch"]
    assert gmap_en["fixture::itch"] != gmap_en["fixture::switch"]

    # 3. Open-class derivational suffix hubs (`gist`, `azione`, `ador`) & C+V `collab+orate` in >=100-lemma lexicon
    hub_cells = ["SG", "PL"]
    hub_entries = {
        "gist": ["gist", "gists"],
        "allergist": ["allergist", "allergists"],
    }
    for i in range(25):
        stem_p = f"biolo{chr(97 + (i % 20))}{chr(97 + (i // 5))}"
        hub_entries[f"{stem_p}gist"] = [f"{stem_p}gist", f"{stem_p}gists"]
    for i in range(90):
        st = f"planet{i:02d}"
        hub_entries[st] = [st, f"{st}s"]
    df_hub = _make_paradigm_df("stan1293_N_orth", "stan1293", "N", "orth", hub_entries, hub_cells)
    gmap_hub, _ = assign_leakage_groups(df_hub, "stan1293", "N", "orth", hub_cells)
    assert gmap_hub["fixture::gist"] != gmap_hub["fixture::allergist"]
    assert sum(1 for v in gmap_hub.values() if v == gmap_hub["fixture::gist"]) == 1


def test_unimorph_file_whitelist_and_romanian_noun_repair(tmp_path: Path):
    # Verify canonical UniMorph file whitelist
    ces_dir = tmp_path / "ces"
    ces_dir.mkdir()
    for fname in ("ces", "ces.derivations", "ces.segmentations", "ces.xz", "ces.args"):
        (ces_dir / fname).write_text("x\ty\tN;NOM;SG\n", encoding="utf-8")
    assert [p.name for p in find_unimorph_data_files(ces_dir, "ces")] == ["ces"]

    aym_dir = tmp_path / "aym"
    aym_dir.mkdir()
    (aym_dir / "Muylaq'AymaraUnimorphConversion.tsv").write_text("doc\n", encoding="utf-8")
    (aym_dir / "aym").write_text("uta\tutana\tN;LOC;SG\n", encoding="utf-8")
    assert [p.name for p in find_unimorph_data_files(aym_dir, "aym")] == ["aym"]

    krl_dir = tmp_path / "krl"
    krl_dir.mkdir()
    for fname in ("krl", "krl-new-written-karelian", "krl-tikhvin", "krl-valdai"):
        (krl_dir / fname).write_text("x\ty\tN;NOM;SG\n", encoding="utf-8")
    assert [p.name for p in find_unimorph_data_files(krl_dir, "krl")] == ["krl-new-written-karelian"]

    # Verify Romanian N SG/PL definite tag collision repair
    raw_ron = [
        ("greblă", "greblei", {"DEF", "GEN/DAT", "PL"}),
        ("greblă", "greblă", {"SG", "VOC"}),
        ("greblă", "grebla", {"DEF", "NOM/ACC", "PL"}),
        ("greblă", "greblelor", {"PL", "VOC"}),
        ("greblă", "greblelor", {"DEF", "GEN/DAT", "PL"}),
        ("greblă", "greblele", {"DEF", "NOM/ACC", "PL"}),
    ]
    rep = _repair_romanian_noun_triples(raw_ron)
    tag_by_form = {(f, ";".join(sorted(ct))) for _, f, ct in rep}
    assert ("greblei", "DEF;GEN/DAT;SG") in tag_by_form
    assert ("grebla", "DEF;NOM/ACC;SG") in tag_by_form
    assert ("greblelor", "DEF;GEN/DAT;PL") in tag_by_form
    assert ("greblele", "DEF;NOM/ACC;PL") in tag_by_form


def test_orth_normalization_and_wiktionary_pollution_filters():
    # Cyrillic acute stress stripped while preserving ё/й/ў
    assert normalize_orth_surface("абавя\u0301зак", "bel") == "абавязак"
    assert normalize_orth_surface("ё\u0301лкай", "rus") == "ёлкай"
    assert normalize_orth_surface("аб’яўля\u0301ем", "bel") == "аб’яўляем"

    # Lithuanian pitch accents (grave/acute/tilde) stripped while preserving ąęįųėūčšž
    assert normalize_orth_surface("abãtą", "lit") == "abatą"
    assert normalize_orth_surface("abatùs", "lit") == "abatus"
    assert normalize_orth_surface("formalýbės", "lit") == "formalybės"
    assert normalize_orth_surface("žūklė\u0303", "lit") == "žūklė"

    # Serbo-Croatian pitch/length marks stripped while preserving ć and čšž
    assert normalize_orth_surface("dojàvljujēm", "hbs") == "dojavljujem"
    assert normalize_orth_surface("dȃm", "hbs") == "dam"
    assert normalize_orth_surface("kuća", "hbs") == "kuća"

    # Wiktionary pollution, bound affixes, Hanja+Romanization, mixed-script typos, multiword idioms
    assert is_polluted_entry("auga", "indefinite", "orth", "isl")
    assert is_polluted_entry("fear", "Singular", "orth", "gle")
    assert not is_polluted_entry("indefinite", "indefinites", "orth", "eng")
    assert is_polluted_entry("-csál", "-csálok", "orth", "hun")
    assert is_polluted_entry("rêze-", "rêzeyan", "orth", "kmr")
    assert is_polluted_entry("基礎하자", "基礎하자基礎haja", "orth", "kor")
    assert not is_polluted_entry("공부하다", "공부해", "orth", "kor")
    assert is_polluted_entry("ποδηλάτης", "ποδηλatών", "orth", "ell")
    assert is_polluted_entry("Aa machen", "mache Aa", "orth", "deu")


def test_pigeonhole_index_matches_bruteforce_all_pairs():
    paradigms = [
        [tuple("parlare"), tuple("parlo"), tuple("parla"), tuple("parlai"), tuple("parlato")],
        [tuple("riparlare"), tuple("riparlo"), tuple("riparla"), tuple("riparlai"), tuple("parlato")],  # 4/5 = 80% match!
        [tuple("straparlare"), tuple("straparlo"), tuple("parla"), tuple("parlai"), tuple("parlato")],  # 2/5 = 40% (<80%)
        [tuple("cantare"), tuple("canto"), tuple("canta"), tuple("cantai"), tuple("cantato")],
        [tuple("cantarebis"), tuple("cantobis"), tuple("cantabis"), tuple("cantaibis"), tuple("cantatobis")],  # suffix!
    ]
    for s in ("port", "mand", "torn", "pass", "rest", "bast", "cost", "ferm"):
        paradigms.append([tuple(f"{s}are"), tuple(f"{s}o"), tuple(f"{s}a"), tuple(f"{s}ai"), tuple(f"{s}ato")])

    pairs = {(a, b) for a, b, _ in find_constant_affix_cognate_pairs(paradigms)}
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
    assert (3, 4) in pairs


def test_splits_zero_leakage_and_exact_sizes():
    rows = []
    for i in range(620):
        gid = f"tst_V::g{i+1:06d}"
        rows.append({"lemma_id": f"src::stem{i:04d}", "lemma_label": f"stem{i:04d}", "leakage_group_id": gid})
        if i % 20 == 0:
            # Add a derived compound in the same group; canonical selector must pick shorter simplex!
            rows.append({
                "lemma_id": f"src::understem{i:04d}",
                "lemma_label": f"understem{i:04d}",
                "leakage_group_id": gid,
            })
    lex_df = pd.DataFrame(rows)
    canon = select_canonical_lemma_per_group(lex_df, "orth")
    assert len(canon) == 620
    assert all(not lab.startswith("under") for lab in canon["lemma_label"])

    s120 = build_regime_splits("tst_V_orth", canon, "uniform120", 150, n_draws=5, n_folds=5)
    assert list(s120.columns) == SPLIT_COLUMNS
    assert len(s120) == 5 * 5 * 150
    for (r, k), grp in s120.groupby(["draw_idx", "fold_idx"]):
        assert grp["leakage_group_id"].nunique() == 150
        tr = set(grp[grp["role"] == "train"]["leakage_group_id"])
        te = set(grp[grp["role"] == "test"]["leakage_group_id"])
        sub = set(grp[grp["inner_role"] == "subtrain"]["leakage_group_id"])
        dev = set(grp[grp["inner_role"] == "dev"]["leakage_group_id"])
        assert len(tr) == 120 and len(te) == 30
        assert len(sub) == 96 and len(dev) == 24
        assert tr.isdisjoint(te)
        assert sub.isdisjoint(dev)
        assert sub | dev == tr

    s480 = build_regime_splits("tst_V_orth", canon, "uniform480", 600, n_draws=3, n_folds=5)
    assert len(s480) == 3 * 5 * 600
    for (r, k), grp in s480.groupby(["draw_idx", "fold_idx"]):
        assert grp["leakage_group_id"].nunique() == 600
        assert (grp["inner_role"] == "subtrain").sum() == 384
        assert (grp["inner_role"] == "dev").sum() == 96
        assert (grp["inner_role"] == "test").sum() == 120

    # Determinism check
    s120_again = build_regime_splits("tst_V_orth", canon, "uniform120", 150, n_draws=5, n_folds=5)
    pd.testing.assert_frame_equal(s120, s120_again)


def test_malouf_unsupervised_vowel_harmony_abstraction():
    cells = ["NOM;SG", "IN+ESS;SG", "AT+ESS;SG"]
    entries = {}
    back_stems = ["talo", "pata", "kala", "sota", "koko", "puku", "suku", "latu", "matto", "katu", "laulu", "taulu"]
    front_stems = ["kylä", "pesä", "kesä", "metsä", "tyttö", "pöytä", "köyhä", "myrsky", "hylly", "käpy", "sänky", "lämpö"]
    for st in back_stems:
        entries[st] = [st, f"{st}ssa", f"{st}lla"]
    for st in front_stems:
        entries[st] = [st, f"{st}ssä", f"{st}llä"]

    df = _make_paradigm_df("finn1318_N_orth", "finn1318", "N", "orth", entries, cells)
    df["leakage_group_id"] = [f"g{i:04d}" for i in range(len(df))]
    df["shape_id"] = "s0"
    df = df[EXPECTED_LEXICON_COLS]

    abs_df, model = apply_malouf_abstraction(df)
    assert list(abs_df.columns) == EXPECTED_LEXICON_COLS
    assert "a" in model["arch_map"] and "ä" in model["arch_map"]
    assert model["arch_map"]["a"] == model["arch_map"]["ä"]

    # After Malouf abstraction, the IN+ESS;SG suffix of `talo` and `kylä` must be identical!
    talo_iness = abs_df[(abs_df["lemma_label"] == "talo") & (abs_df["cell"] == "IN+ESS;SG")]["segmented_form"].iloc[0]
    kyla_iness = abs_df[(abs_df["lemma_label"] == "kylä") & (abs_df["cell"] == "IN+ESS;SG")]["segmented_form"].iloc[0]
    assert talo_iness.split()[-3:] == kyla_iness.split()[-3:]
    assert talo_iness.split()[0] != kyla_iness.split()[0]  # opposite stem harmony class markers


@pytest.mark.skipif(not (DATA_DIR / "unit_summary.csv").exists(), reason="Run build_lexicons.py first")
def test_extracted_lexicons_provenance_and_zero_leakage_audit():
    manifest = pd.read_csv(DATA_DIR / "provenance_manifest.csv", keep_default_na=False)
    assert len(manifest) >= 80
    active = manifest[manifest["status"].isin(["downloaded", "reused_local"])]
    assert len(active) >= 75
    for row in active.to_dict("records"):
        assert row["url"] and row["version_or_commit"] and row["license"] and row["citation"]
        target = PROJECT_ROOT / row["sha256_target"]
        assert target.exists(), f"Missing sha256_target {target}"
        h = hashlib.sha256(target.read_bytes()).hexdigest()
        assert h == row["sha256"], f"SHA-256 mismatch for {row['source_id']}"

    summary = pd.read_csv(DATA_DIR / "unit_summary.csv", keep_default_na=False)
    assert len(summary) >= 80
    lex_dir = DATA_DIR / "lexicons"
    splits_dir = DATA_DIR / "splits"

    lex_files = sorted(lex_dir.glob("*.csv"))
    assert len(lex_files) == len(summary)

    for row in summary.to_dict("records"):
        uid = row["unit_id"]
        lpath = lex_dir / f"{uid}.csv"
        assert lpath.exists()
        df = pd.read_csv(lpath, keep_default_na=False)
        assert list(df.columns) == EXPECTED_LEXICON_COLS
        assert int(row["n_leakage_groups"]) >= 150
        assert int(row["n_cells_distinct"]) >= 2

        # Check NFC normalization, absence of #DEF#, no bound-affix lemmas, and zero multiword `orth` entries
        assert not df["lemma_label"].str.startswith("-").any(), f"Bound prefix lemma in {uid}"
        assert not df["lemma_label"].str.endswith("-").any(), f"Bound suffix lemma in {uid}"
        assert not df["lemma_label"].str.contains(r"\s", regex=True).any(), f"Multiword lemma_label in {uid}"
        if row["representation"] == "orth":
            assert not df["raw_form"].str.contains(r"\s", regex=True).any(), f"Multiword orth raw_form in {uid}"
        for rf in df["raw_form"].head(500):
            assert rf != "#DEF#"
            assert unicodedata.is_normalized("NFC", rf)

        # Check zero group leakage in uniform120 (and uniform480 if tier2)
        s120_path = splits_dir / f"{uid}_uniform120.csv"
        assert s120_path.exists()
        s120 = pd.read_csv(s120_path, keep_default_na=False)
        for (_, _), grp in s120.groupby(["draw_idx", "fold_idx"]):
            tr = set(grp[grp["role"] == "train"]["leakage_group_id"])
            te = set(grp[grp["role"] == "test"]["leakage_group_id"])
            assert tr.isdisjoint(te), f"Group leakage detected in {uid}_uniform120.csv!"

    # Verify canonical prefixed irregular cognates share one `leakage_group_id` on disk across both orth & phon
    for it_uid in ("ital1282_V_orth", "ital1282_V_phon"):
        df_it = pd.read_csv(lex_dir / f"{it_uid}.csv", keep_default_na=False)
        lbl_g = dict(zip(df_it["lemma_label"], df_it["leakage_group_id"]))
        assert lbl_g["porre"] == lbl_g["comporre"] == lbl_g["imporre"], f"porre cognates unmerged in {it_uid}"

    for en_uid in ("stan1293_V_orth", "stan1293_V_phon"):
        df_en = pd.read_csv(lex_dir / f"{en_uid}.csv", keep_default_na=False)
        lbl_g = dict(zip(df_en["lemma_label"], df_en["leakage_group_id"]))
        assert lbl_g["stand"] == lbl_g["understand"], f"stand/understand unmerged in {en_uid}"


def test_cold_start_source_resolvers_and_archive_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Tracked analysis_links.csv resolution without outputs/ directory
    tracked_links = resolve_analysis_links_csv(PROJECT_ROOT)
    assert tracked_links == PROJECT_ROOT / "analyses" / "common" / "data" / "analysis_links.csv"
    assert tracked_links.is_file()

    fake_root = tmp_path / "fresh_clone"
    fake_ext = fake_root / "external" / "paradigm_sources"
    fake_links = fake_root / "analyses" / "common" / "data" / "analysis_links.csv"
    fake_links.parent.mkdir(parents=True)
    shutil.copy2(tracked_links, fake_links)
    assert resolve_analysis_links_csv(fake_root) == fake_links

    # 2. CELEX English verbs (`celex_eng_v`): HTTP 206 Range ZIP extraction when mgn_data/ is absent
    celex_csv_bytes = (
        "lexeme,cell,form,language_ID,POS\n"
        "sing,inf,sIN,eng,V\n"
        "sing,past13,s{N,eng,V\n"
        "sing,ppart,sVN,eng,V\n"
    ).encode("utf-8")
    celex_sha = hashlib.sha256(celex_csv_bytes).hexdigest()

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("code/README.txt", b"padding " * 512)
        zf.writestr("code/data-custom/english-v.csv", celex_csv_bytes)
    zip_payload = zip_buf.getvalue()

    class _RangeHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                s_str, e_str = rng[6:].split("-", 1)
                start = int(s_str)
                end = min(len(zip_payload) - 1, int(e_str))
                chunk = zip_payload[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(zip_payload)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(zip_payload)))
                self.end_headers()
                self.wfile.write(zip_payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        monkeypatch.setattr(fetch_sources_mod, "CELEX_ENG_V_SHA256", celex_sha)
        monkeypatch.setattr(
            fetch_sources_mod,
            "fetch_json",
            lambda _url, timeout=45: {
                "files": [
                    {
                        "key": "code.zip",
                        "size": len(zip_payload),
                        "links": {"self": f"http://127.0.0.1:{port}/code.zip"},
                    }
                ]
            },
        )
        celex_row = fetch_celex_english(force=False, project_root=fake_root, external_root=fake_ext)
    finally:
        server.shutdown()
        server.server_close()

    assert celex_row["status"] == "downloaded"
    assert celex_row["sha256"] == celex_sha
    resolved_celex = resolve_celex_english_path(project_root=fake_root, external_root=fake_ext)
    assert resolved_celex == fake_ext / "celex" / "english-v.csv"
    celex_recs = read_celex_english_v(resolved_celex)
    assert len(celex_recs) == 3
    assert ("sing", "sing", "LGSPEC1;PST", "s{N", "", 0) in celex_recs

    # 3. Morph-it! v0.48 (`national_morphit`): ISO-8859-1 archive -> UTF-8 transcoding & resolver fallback
    morphit_text = (
        "città\tcittà\tNOUN-F:s\n"
        "città\tcittà\tNOUN-F:p\n"
        "parlò\tparlare\tVER:ind+past+3+s\n"
        "parlare\tparlare\tVER:inf+pres\n"
    )
    morphit_latin1 = morphit_text.encode("iso-8859-1")
    morphit_utf8_sha = hashlib.sha256(morphit_text.encode("utf-8")).hexdigest()

    morphit_zip = tmp_path / "hfst-it-morph-morph-it.zip"
    with zipfile.ZipFile(morphit_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hfst-it-morph-morph-it/morph-it/morph-it_048.txt", morphit_latin1)

    monkeypatch.setattr(fetch_sources_mod, "MORPHIT_ARCHIVE_URL", morphit_zip.as_uri())
    monkeypatch.setattr(fetch_sources_mod, "MORPHIT_UTF8_SHA256", morphit_utf8_sha)

    morph_row = fetch_morphit(force=False, project_root=fake_root, external_root=fake_ext)
    assert morph_row["status"] == "downloaded"
    assert morph_row["sha256"] == morphit_utf8_sha
    resolved_morph = resolve_morphit_path(project_root=fake_root, external_root=fake_ext)
    assert resolved_morph == fake_ext / "national" / "morphit" / "morph-it_048.utf8.txt"
    assert len(read_morphit_records("N", resolved_morph)) == 2
    assert len(read_morphit_records("V", resolved_morph)) == 2
