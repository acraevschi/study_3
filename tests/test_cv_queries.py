"""The query gate exposes only the source anchor and the requested target cell."""

import pytest

from cv_fixtures import PANEL, SOURCE, make_forms
from morph_ldl.cv import queries
from morph_ldl.schemas import TEST_QUERY_COLUMNS


def test_queries_carry_no_target_forms():
    forms = make_forms(30, missing_every=0)
    ids = forms["lemma_id"].unique()[:5]
    q = queries.build_queries(forms, ids, SOURCE, PANEL)
    assert list(q.columns) == TEST_QUERY_COLUMNS
    assert len(q) == 5 * len(PANEL)
    targets = set(forms[(forms.lemma_id.isin(ids)) & (forms.cell_norm.isin(PANEL))]["form"])
    assert not targets & set(q["source_form"])
    budget = queries.source_information_budget(q)
    assert budget["n_source_forms_supplied"] == 5


def test_source_cell_cannot_be_target():
    forms = make_forms(5, missing_every=0)
    with pytest.raises(ValueError):
        queries.build_queries(forms, forms["lemma_id"].unique(), SOURCE, [SOURCE])
