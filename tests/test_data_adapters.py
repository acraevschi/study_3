"""Adapter round-trips on small fixtures (variants, NA, multiword, custom tokens)."""

from pathlib import Path

import pandas as pd

from morph_ldl.data import adapters
from morph_ldl.data.adapters import ResourceMeta, to_forms
from morph_ldl.data.segments import has_reserved, segment
from morph_ldl.schemas import FORMS_COLUMNS

FIX = Path(__file__).parent / "fixtures" / "data"


def _meta(stem, rep, rid=None):
    return ResourceMeta("toy.V.%s.mgn" % rep, rid or f"mgn_data:{stem}", "sha256:0", "ita", "ita",
                        "ital1282", "V", rep, stem)


def test_mgn_wide_roundtrip():
    raw = adapters.read_mgn_wide(FIX / "mgn/data/toy-v.csv")
    f = to_forms(raw, _meta("toy-v", "orth"))
    assert list(f.columns) == FORMS_COLUMNS
    # every source cell is represented: 6 rows x 4 cells + 1 extra variant row
    assert len(f) == 6 * 4 + 1
    # variants: 'dice;dici' -> two rows, form_orig unchanged on both
    v = f[(f.lemma_label == "dire") & (f.cell_orig == "IND;PRS;3;SG") & (f.form_orig == "dice;dici")]
    assert v["form"].tolist() == ["dice", "dici"]
    assert v["variant_idx"].tolist() == [0, 1] and set(v["n_variants"]) == {2}
    # NA -> is_missing, empty form/segments, marker kept in form_orig
    na = f[f.is_missing]
    assert set(na["form_orig"]) == {"NA"} and set(na["form"]) == {""} and set(na["segments"]) == {""}
    assert (na["n_variants"] == 0).all()
    # multiword -> '_' word separator
    mw = f[(f.lemma_label == "lavarsi") & (f.cell_norm == "1;IND;PRS;SG")].iloc[0]
    assert mw["segments"] == "m i _ l a v o"
    # duplicated label -> #k lemma ids, provenance lines
    assert {"mgn_data:toy-v::dire#1", "mgn_data:toy-v::dire#2"} <= set(f["lemma_id"])
    assert "mgn_data:toy-v::lavare" in set(f["lemma_id"])
    assert f.loc[f.lemma_id == "mgn_data:toy-v::dire#2", "source_row"].unique().tolist() == [5]
    # cell_norm sorted features
    assert set(f["cell_norm"]) == {"NFIN", "1;IND;PRS;SG", "3;IND;PRS;SG", "3;IND;PST;SG"}


def test_mgn_long_custom_tokens_and_variants():
    raw = adapters.read_mgn_long(FIX / "mgn/data-custom/toyish-v.csv")
    f = to_forms(raw, _meta("toyish-v", "phon_custom", "mgn_custom:toyish-v"))
    v = f[(f.lemma_label == "kataba") & (f.cell_orig == "prs.1sg")]
    assert v["variant_idx"].tolist() == [0, 1]
    assert v["segments"].tolist() == ["ʔ a k t u b u", "ʔ a k t u b"]
    assert v["form_orig"].tolist() == ["ʔ a k t u b u", "ʔ a k t u b"]
    assert f.loc[(f.lemma_label == "darasa") & (f.cell_orig == "prs.1sg"), "is_missing"].item()
    assert set(f.loc[f.cell_orig == "inf", "cell_norm"]) == {"NFIN"}


def test_unimorph_extra_columns_blank_lines_and_pos_filter():
    raw = adapters.read_unimorph(FIX / "toy_unimorph.tsv", pos="V")
    assert len(raw) == 3
    f = to_forms(raw, _meta("toy_unimorph", "orth", "unimorph:toy"))
    one = f[f.cell_norm == "1;IND;PRS;SG"]
    assert one["variant_idx"].tolist() == [0, 1]  # repeated row = variant
    assert set(f["cell_norm"]) == {"1;IND;PRS;SG", "NFIN"}   # POS tag removed
    assert set(f["cell_orig"]) == {"V;IND;PRS;1;SG", "V;NFIN"}  # original kept


def test_paralex_interface():
    raw = adapters.read_paralex(FIX / "paralex_toy", representation="phon_custom")
    f = to_forms(raw, _meta("paralex_toy", "phon_custom", "paralex:paralex_toy"))
    assert set(f["cell_norm"]) == {"NFIN", "1;PRS;SG"}
    d = f[(f.lemma_label == "falloir") & (f.cell_orig == "prs.1sg")].iloc[0]
    assert d["is_missing"] and d["form_orig"] == "#DEF#" and d["form"] == ""
    assert f.loc[(f.lemma_label == "chanter") & (f.cell_orig == "inf"), "segments"].item() == "ʃ ɑ̃ t e"
    o = to_forms(adapters.read_paralex(FIX / "paralex_toy", representation="orth"),
                 _meta("paralex_toy", "orth", "paralex:paralex_toy"))
    assert o.loc[(o.lemma_label == "chanter") & (o.cell_orig == "inf"), "segments"].item() == "c h a n t e r"


def test_segmentation_rules():
    assert segment("città", "orth") == "c i t t à"
    assert segment("città", "orth") == "c i t t à"  # NFC first
    assert segment("ẹ́x", "orth") == "ẹ́ x"  # combining marks attached
    assert segment("aːlən", "ipa_epitran") == "aː l ə n"
    assert segment("aːlən", "orth") == "a ː l ə n"
    assert segment("  ʃ  ɑ̃ t ", "phon_custom") == "ʃ ɑ̃ t"
    assert segment("a  b", "orth") == "a _ b"
    assert has_reserved("ab#c") and not has_reserved("abc")
