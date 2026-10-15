"""Cell-grid input (pcfp_grid_v1): the `grid` adapter, multiword eligibility, citation-in-test."""

import pandas as pd
import pytest

from morph_ldl.cv import pcfp
from morph_ldl.cv.pipeline import _citation
from morph_ldl.data import adapters

from test_pcfp import CELLS, PERIPH, make_forms

GRID_HEADER = ["lemma", "grid_cell", "form", "variants", "form_b", "variants_b", "rule", "na_reason"]


def write_grid(tmp_path, rows):
    p = tmp_path / "xxx_grid.csv"
    pd.DataFrame(rows, columns=GRID_HEADER).to_csv(p, index=False)
    return p


def test_read_grid_forms_variants_and_gaps(tmp_path):
    p = write_grid(tmp_path, [
        ["parlare", "PST.1SG", "parlato", "", "ho parlato", "", "main_verb", ""],
        ["parlare", "PRS.3PL", "parlano", "parlan", "parlano", "parlan", "source", ""],
        ["parlare", "IMP.2SG", "", "", "", "", "source", "data_gap"],
        ["parlare", "NFIN", "parlare", "", "parlare", "", "citation", ""],
    ])
    raw = adapters.read_grid(p, form="b")
    by = {(r.cell_orig, r.variant_idx): r for r in raw.itertuples()}
    assert by[("IND;PST;1;SG", 0)].variant_raw == "ho parlato"          # option B form
    assert by[("IND;PRS;3;PL", 1)].variant_raw == "parlan"              # variants kept in order
    assert by[("IMP;2;SG", 0)].is_missing and by[("IMP;2;SG", 0)].n_variants == 0
    assert by[("NFIN", 0)].variant_raw == "parlare"
    raw_c = adapters.read_grid(p, form="c")
    assert raw_c.loc[raw_c["cell_orig"] == "IND;PST;1;SG", "variant_raw"].tolist() == ["parlato"]


def test_read_grid_rejects_unknown_cells_and_forms(tmp_path):
    p = write_grid(tmp_path, [["x", "PRF.1SG", "a", "", "a", "", "source", ""]])
    with pytest.raises(ValueError, match="not in grid.csv"):
        adapters.read_grid(p)
    with pytest.raises(ValueError, match="'b' or 'c'"):
        adapters.read_grid(p, form="a")


def test_allow_multiword_keeps_periphrastic_cells():
    forms = make_forms(100)
    cells = sorted(CELLS + [PERIPH])
    strict = pcfp.eligible_lemmas(forms, cells, "NFIN")
    loose = pcfp.eligible_lemmas(forms, cells, "NFIN", allow_multiword=True)
    assert len(strict) < len(loose)                    # 90% of PERIPH forms are multiword
    assert set(strict["lemma_id"]) <= set(loose["lemma_id"])


def test_citation_rule_include_in_test():
    unit = {"unit_id": "u", "citation_cell": "NFIN", "cells": CELLS}
    cfg = {"task": {"name": "pcfp", "citation_cell_rule": "include_in_test"}}
    assert _citation(cfg, unit) is None
    cfg["task"]["citation_cell_rule"] = "exclude_from_test"
    assert _citation(cfg, unit) == "NFIN"
    cfg["task"]["citation_cell_rule"] = "sometimes"
    with pytest.raises(ValueError):
        _citation(cfg, unit)
    e = pcfp.Exposure("l", 1, ("1;PRS;SG",), ("NFIN", "2;PRS;SG"))
    assert e.test_cells(None) == ("NFIN", "2;PRS;SG")
    assert e.test_cells("NFIN") == ("2;PRS;SG",)


def test_exposure_manifest_counts_citation_when_included():
    forms = make_forms(60)
    lem = pcfp.eligible_lemmas(forms, CELLS, "NFIN")
    a = pcfp.build_exposure_manifest(lem, CELLS, "NFIN", 7, "u", 4)
    b = pcfp.build_exposure_manifest(lem, CELLS, "NFIN", 7, "u", 4, test_exclude=None)
    hidden_cit = ~a["citation_shown"]
    assert (b["n_test_cells"] - a["n_test_cells"])[hidden_cit].eq(1).all()
    assert (b["n_test_cells"] == a["n_test_cells"])[~hidden_cit].all()
    pcfp.validate_exposure_manifest(b, CELLS, "NFIN", 7, "u", 4, test_exclude=None)
    with pytest.raises(pcfp.ExposureError):
        pcfp.validate_exposure_manifest(b, CELLS, "NFIN", 7, "u", 4)
