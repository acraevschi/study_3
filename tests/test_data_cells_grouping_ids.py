"""Cell normalisation, grouping, identifier overrides vs `low`, build provenance."""

from pathlib import Path

import pandas as pd
import pytest

from morph_ldl.config import PIPELINE_ROOT, load_config, unit_cells
from morph_ldl.data.cells import normalize_label
from morph_ldl.data.grouping import assign_groups
from morph_ldl.data.identifiers import low_glottocode, resolve_identifier
from morph_ldl.data.provenance import build_provenance, representation_from_provenance

FIX = Path(__file__).parent / "fixtures" / "data"


@pytest.mark.parametrize("label,stem,expected", [
    ("IND;PST;1;SG;PFV", "ita-v", "1;IND;PFV;PST;SG"),
    ("V;NFIN", None, "NFIN"),
    ("ACT;PRS;POS;IND;3;SG", "fin-v", "3;ACT;IND;POS;PRS;SG"),
    ("pl:acc:m1.p1", "pol-n", "ACC;MASC;PL"),
    ("ind.prs.act.m/f.1.s", "arabic-v", "1;ACT;IND;PRS;SG"),
    ("prs.1sg", "french-v", "1;PRS;SG"),
    ("pst.ptcp.f.pl", "french-v", "FEM;PL;PST;V.PTCP"),
    ("nominative singular", "russian-n", "NOM;SG"),
    ("prepositional plural", "russian-n", "ESS;PL"),
    ("gen pl", "latvian-n", "GEN;PL"),
    ("in+ess;sg", "hungarian-n", "IN+ESS;SG"),
    ("inst;pl", "hungarian-n", "INS;PL"),
    ("NOUN:Abl+Plur", "latin-n", "ABL;PL"),
    ("PresIndic4", "portuguese-v", "1;IND;PL;PRS"),
    ("FUT.3apl:IPA", "navajo-v", "4;FUT;PL"),
    ("pres3s", "english-v", "3;PRS;SG"),
])
def test_normalize(label, stem, expected):
    assert normalize_label(label, stem)[0] == expected


def test_unparseable_is_empty():
    assert normalize_label("weird label 7", None) == ("", "unparseable")
    assert normalize_label("VERB:Fin+Imp+Fut+-+Act+2+Plur+-+-", "latin-v")[0] == ""


def test_pilot_config_cells_match_normaliser():
    cfg = load_config(PIPELINE_ROOT / "configs" / "pilot.yaml")
    root = PIPELINE_ROOT / "mgn_data" / "data"
    for u in cfg["units"]:
        stem = u["resource_id"].split(":")[1]
        header = (root / f"{stem}.csv").read_text(encoding="utf-8").split("\n", 1)[0].split(",")[1:]
        present = {normalize_label(c, stem)[0] for c in header}
        uc = unit_cells(u, cfg)
        for cn in [uc["source"]] + [c for _, c in uc["panel"]]:
            assert cn in present, (u["unit_id"], cn)


def _forms(rows):
    cols = ["lemma_id", "lemma_label", "variety_id", "pos", "resource_id", "representation",
            "cell_norm", "form", "is_missing"]
    return pd.DataFrame(rows, columns=cols)


def test_grouping_shared_label_across_resources_and_source_form():
    f = _forms([
        ("A::Lavare", "Lavare", "ita", "V", "A", "orth", "NFIN", "lavare", False),
        ("B::lavare", "lavare ", "ita", "V", "B", "phon_custom", "NFIN", "l a v a r e", False),
        ("A::x1", "x1", "ita", "V", "A", "orth", "NFIN", "same", False),
        ("A::x2", "x2", "ita", "V", "A", "orth", "NFIN", "same", False),
        ("A::x3", "x3", "ita", "V", "A", "orth", "1;IND;PRS;SG", "same", False),   # not source cell
        ("C::x4", "x4", "ita", "V", "C", "phon_custom", "NFIN", "same", False),    # other representation
        ("A::lavare_n", "lavare", "ita", "N", "A", "orth", "NFIN", "lavare", False),  # other POS
    ])
    g = assign_groups(f, {"A": "NFIN", "B": "NFIN", "C": "NFIN"}).set_index("lemma_id")
    assert g.loc["A::Lavare", "group_id"] == g.loc["B::lavare", "group_id"]       # label rule, cross-resource
    assert g.loc["A::x1", "group_id"] == g.loc["A::x2", "group_id"]              # source-form rule
    assert g.loc["A::x3", "group_id"] != g.loc["A::x1", "group_id"]
    assert g.loc["C::x4", "group_id"] != g.loc["A::x1", "group_id"]
    assert g.loc["A::lavare_n", "group_id"].startswith("ita.N::")
    assert g.loc["A::Lavare", "group_id"].startswith("ita.V::g")
    # deterministic
    g2 = assign_groups(f.sample(frac=1, random_state=3), {"A": "NFIN", "B": "NFIN", "C": "NFIN"}).set_index("lemma_id")
    assert (g2.loc[g.index, "group_id"] == g["group_id"]).all()


def test_identifier_overrides_vs_low():
    gal = resolve_identifier("gal")
    assert low_glottocode("gal") == "galo1243"           # low: Galoli
    assert gal["canonical_iso639_3"] == "glg" and gal["final_glottocode"] == "gali1258"
    assert "original_code_is_false_friend_in_low" in gal["unresolved_flags"]
    yai = resolve_identifier("yai")
    assert yai["final_glottocode"] == "west2644" and yai["low_glottocode_for_original"] == "yagn1238"
    nob = resolve_identifier("nob")
    assert nob["variety_id"] == "nor-bokmal" and nob["final_glottocode"] == "norw1258"
    ita = resolve_identifier("ita")
    assert ita["project_override"] == "no" and ita["final_glottocode"] == "ital1282"
    assert resolve_identifier("fre")["final_glottocode"] == "stan1290"
    assert resolve_identifier("lat")["extinct"]


def test_build_provenance_and_representation():
    prov = build_provenance(FIX / "mgn")
    assert prov["toy-v"].upstream_code == "toy" and prov["toy-v"].epitran_code is None
    assert prov["toy-n"].epitran_code == "epi-Latn"
    assert representation_from_provenance(prov["toy-v"], "mgn_data") == "orth"
    assert representation_from_provenance(prov["toy-n"], "mgn_data") == "ipa_epitran"
    assert representation_from_provenance(prov["toyish-v"], "mgn_data-custom") == "phon_custom"


def test_real_mgn_provenance_pilot_languages():
    prov = build_provenance(PIPELINE_ROOT / "mgn_data")
    assert prov["ita-v"].epitran_code is None and prov["fin-v"].epitran_code is None
    assert prov["deu-v"].epitran_code == "deu-Latn" and prov["spa-v"].epitran_code == "spa-Latn"
    assert prov["pol-n"].derivation == "polimorf"
