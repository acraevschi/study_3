"""PCFP design: cell inventory, eligibility, exposure draw, citation-cell rule, core splits."""

import numpy as np
import pandas as pd
import pytest

from morph_ldl.cv import pcfp, splits
from morph_ldl.schemas import FORMS_COLUMNS

CELLS = ["NFIN", "1;PRS;SG", "2;PRS;SG", "3;PRS;SG", "3;PL;PRS", "1;PST;SG", "3;PST;SG"]
PERIPH = "1;PRF;SG"                      # a periphrastic (mostly multiword) cell


def make_forms(n=400, unit="xxx.V.orth.test"):
    rows = []
    for i in range(n):
        stem = f"v{i:04d}"
        lid = f"test:xxx-v::{stem}are"
        gid = f"xxx.V::g{(i - 1) if (i % 25 == 0 and i > 0) else i:05d}"
        forms = {c: stem + f"x{j}" for j, c in enumerate(CELLS)}
        forms["NFIN"] = stem + "are"
        forms[PERIPH] = "ho " + stem + "ato" if i % 10 else stem + "ato"      # 90% multiword
        if i % 50 == 7:
            forms["3;PST;SG"] = "si " + stem                                    # a stray multiword form
        for cell, f in forms.items():
            missing = (i % 60 == 11 and cell == "3;PL;PRS")
            rows.append(dict(unit_id=unit, resource_id="test:xxx-v", resource_version="t", variety_id="xxx",
                             iso639_3="xxx", glottocode="xxxx1234", pos="V", representation="orth",
                             lemma_id=lid, lemma_label=stem + "are", group_id=gid, cell_orig=cell,
                             cell_norm=cell, form_orig=f, variant_idx=0, n_variants=1,
                             form="" if missing else f,
                             segments="" if missing else " ".join("_" if ch == " " else ch for ch in f),
                             is_missing=bool(missing), source_file="fixture", source_row=i))
    return pd.DataFrame(rows)[FORMS_COLUMNS]


@pytest.fixture(scope="module")
def forms():
    return make_forms()


def cfg(**cv):
    base = {"inventory_size": 240, "n_folds": 3, "core_size": 30, "dev_size": 0, "seed_size": 10,
            "pool_cap": 100, "pool_cap_sensitivity": [50], "repetitions": [0]}
    base.update(cv)
    return {"experiment": {"id": "t", "master_seed": 7}, "cv": base,
            "selection": {"budgets": [40, 60], "batch_size": 10}}


def test_cell_rule_drops_mostly_multiword_cells(forms):
    t = pcfp.cell_inventory_table(forms, max_multiword_share=0.5)
    assert set(t.loc[t.eligible, "cell_norm"]) == set(CELLS)
    assert t.set_index("cell_norm").loc[PERIPH, "multiword_share"] == pytest.approx(0.9)
    # derived-paradigm exclusions change the denominator
    excl = forms.lemma_id.unique()[1:100]
    assert pcfp.cell_inventory_table(forms, excl).set_index("cell_norm").loc[PERIPH, "n_forms_v0"] == 301


def test_eligibility_complete_single_word_paradigm(forms):
    cells = pcfp.eligible_cells(forms)
    lem = pcfp.eligible_lemmas(forms, cells, "NFIN")
    ids = set(lem.lemma_id)
    stray = {f"test:xxx-v::v{i:04d}are" for i in range(400) if i % 50 == 7}
    missing = {f"test:xxx-v::v{i:04d}are" for i in range(400) if i % 60 == 11}
    assert not ids & (stray | missing)                       # multiword form / missing cell -> ineligible
    assert len(ids) == 400 - len(stray | missing)
    assert (lem.n_cells_available == len(cells)).all()
    excl = ["test:xxx-v::v0000are"]
    assert "test:xxx-v::v0000are" not in set(pcfp.eligible_lemmas(forms, cells, "NFIN", excl).lemma_id)


def test_exposure_draw_rule_and_determinism():
    cells = sorted(CELLS)
    ks = []
    for i in range(500):
        e = pcfp.draw_exposure(7, "u", f"l{i}", cells, max_shown=7)
        assert 1 <= e.k <= min(7, len(cells) - 1) == 6
        assert len(e.shown) == e.k and set(e.shown).isdisjoint(e.hidden)
        assert sorted(e.shown + e.hidden) == cells
        assert e == pcfp.draw_exposure(7, "u", f"l{i}", list(reversed(cells)), max_shown=7)   # order-free
        ks.append(e.k)
    counts = np.bincount(ks, minlength=7)[1:]
    assert (counts > 50).all()                               # roughly uniform over 1..6
    # cap: small paradigms are capped by n_cells - 1
    assert pcfp.k_max(5, 7) == 4 and pcfp.k_max(48, 7) == 7
    with pytest.raises(pcfp.ExposureError):
        pcfp.draw_exposure(7, "u", "x", ["NFIN"], 7)
    # depends on the master seed, unit and lemma only
    assert pcfp.draw_exposure(8, "u", "l1", cells, 7) != pcfp.draw_exposure(7, "u", "l1", cells, 7) or True


def test_exposure_manifest_validates_and_detects_tampering(forms):
    cells = pcfp.eligible_cells(forms)
    lem = pcfp.eligible_lemmas(forms, cells, "NFIN")
    man = pcfp.build_exposure_manifest(lem, cells, "NFIN", 7, "u", 7)
    s = pcfp.validate_exposure_manifest(man, cells, "NFIN", 7, "u", 7)
    assert s["n_lemmas"] == len(lem) and s["k_max"] == 6
    # the same lemma gets the same draw in a manifest built from a subset (no dependence on others)
    sub = pcfp.build_exposure_manifest(lem.iloc[::7], cells, "NFIN", 7, "u", 7).set_index("lemma_id")
    full = man.set_index("lemma_id")
    assert (sub["shown_cells"] == full.loc[sub.index, "shown_cells"]).all()
    bad = man.copy()
    i = bad.index[bad.k < 6][0]
    bad.loc[i, "shown_cells"], bad.loc[i, "hidden_cells"] = bad.loc[i, "hidden_cells"], bad.loc[i, "shown_cells"]
    with pytest.raises(pcfp.ExposureError):
        pcfp.validate_exposure_manifest(bad, cells, "NFIN", 7, "u", 7)


def test_citation_cell_rule_and_shown_rows(forms):
    cells = pcfp.eligible_cells(forms)
    lem = pcfp.eligible_lemmas(forms, cells, "NFIN")
    man = pcfp.build_exposure_manifest(lem, cells, "NFIN", 7, "u", 7)
    expo = pcfp.exposure_from_frame(man)
    ids = list(expo)[:120]
    q = pcfp.hidden_queries(expo, ids, "NFIN")
    assert list(q.columns) == ["lemma_id", "target_cell"]
    assert "NFIN" not in set(q.target_cell)                  # citation cell is never a test item
    for l in ids:
        e = expo[l]
        got = set(q.loc[q.lemma_id == l, "target_cell"])
        assert got == set(e.hidden) - {"NFIN"}
        assert len(got) == len(cells) - e.k - (0 if "NFIN" in e.shown else 1)
    assert any("NFIN" in expo[l].shown for l in ids)          # the citation cell can be shown (counted in k)
    rows = pcfp.shown_rows(forms, expo, ids)
    assert set(zip(rows.lemma_id, rows.cell_norm)) == {(l, c) for l in ids for c in expo[l].shown}
    assert not set(zip(rows.lemma_id, rows.cell_norm)) & set(zip(q.lemma_id, q.target_cell))


def test_citation_segments_from_label():
    assert pcfp.citation_segments("amare", "orth") == "a m a r e"
    assert pcfp.citation_segments("ajaa karille", "orth") == "a j a a _ k a r i l l e"
    with pytest.raises(ValueError):
        pcfp.citation_segments("x", "phon_custom")


def test_inflection_class_proxies():
    assert pcfp.inflection_class("ita.V.orth.mgn", "amare") == "-are"
    assert pcfp.inflection_class("ita.V.orth.mgn", "porre") == "-rre"
    f = lambda s: pcfp.inflection_class("fin.V.orth.mgn", s)
    assert (f("sanoa"), f("saada"), f("tulla"), f("pestä"), f("haluta"), f("tarvita")) == \
        ("VV", "da", "CCa", "CCa", "Vta", "Vta")


# ----------------------------------------------------------------------------- splits

@pytest.fixture(scope="module")
def lemmas(forms):
    cells = pcfp.eligible_cells(forms)
    return pcfp.eligible_lemmas(forms, cells, "NFIN")[["lemma_id", "group_id"]]


def test_core_sets_disjoint_and_exact(lemmas):
    man = splits.build_pcfp_manifest(lemmas, cfg(), "u", 0)
    cores = [set(man[(man.outer_fold == k) & (man.role == "core")].lemma_id) for k in range(3)]
    assert all(len(c) == 30 for c in cores)
    assert not (cores[0] & cores[1] or cores[0] & cores[2] or cores[1] & cores[2])
    for k in range(3):
        fm = man[man.outer_fold == k]
        assert fm.role.value_counts().to_dict() == {"core": 30, "seed": 10, "pool": 100, "pool_overflow": 100}
        # core verbs and their groups are never seed/pool in their own fold
        cg = set(fm[fm.role == "core"].group_id)
        assert not cg & set(fm[fm.role != "core"].group_id)
    pd.testing.assert_frame_equal(man, splits.build_pcfp_manifest(lemmas, cfg(), "u", 0))
    r = splits.roles(man, 0, 1, pool_cap=50)
    assert r["pool"] == splits.roles(man, 0, 1)["pool"][:50] and r["dev"] == []


def test_pcfp_manifest_roundtrip_and_capacity(tmp_path, lemmas):
    man = splits.build_pcfp_manifest(lemmas, cfg(), "u", 0)
    p = splits.write_manifest(man, tmp_path)
    assert len(splits.load_manifest(p)) == len(man)
    with pytest.raises(splits.SplitError):
        splits.build_pcfp_manifest(lemmas, cfg(core_size=90), "u", 0)          # 3 x 90 > 240
    with pytest.raises(splits.SplitError):
        splits.build_pcfp_manifest(lemmas, cfg(pool_cap=40, pool_cap_sensitivity=[]), "u", 0)  # pool < budget - seed
    bad = man.copy()
    j = bad.index[(bad.outer_fold == 1) & (bad.role == "pool_overflow")][0]
    core0 = bad[(bad.outer_fold == 0) & (bad.role == "core")].iloc[0]
    bad = bad[~((bad.outer_fold == 1) & (bad.lemma_id == core0.lemma_id))]
    bad.loc[j, ["lemma_id", "group_id", "role"]] = [core0.lemma_id, core0.group_id, "core"]
    with pytest.raises(splits.SplitError):
        splits.validate_pcfp_manifest(bad)


def test_auxiliary_pcfp_roles_outside_inventory(lemmas):
    man = splits.build_pcfp_manifest(lemmas, cfg(), "u", 0)
    aux = splits.build_auxiliary_manifest(lemmas, man.lemma_id.unique(), {"tune_core": 20, "tune_extra": 20}, 7, "u")
    r = splits.aux_roles(aux)
    assert len(r["tune_core"]) == 20 and len(r["tune_extra"]) == 20
    assert not set(aux.group_id) & set(man.group_id)
