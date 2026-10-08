"""The PCFP artifact audit catches injected leaks (runs on a copy of the smoke outputs).

Skipped unless `outputs/pcfp_smoke` holds a finished smoke run (scripts/run_pcfp.sh
configs/pcfp_smoke.yaml)."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from morph_ldl.config import PIPELINE_ROOT, load_config
from morph_ldl.cv import pipeline

pytestmark = pytest.mark.slow
SMOKE = PIPELINE_ROOT / "outputs" / "pcfp_smoke"
UNIT = "ita.V.orth.mgn"


@pytest.fixture()
def tree(tmp_path):
    if not (SMOKE / "selection" / UNIT / "rep0" / "fold0").exists():
        pytest.skip("no finished smoke run")
    root = tmp_path / "out"
    dst = root / "pcfp_smoke"
    dst.mkdir(parents=True)
    for d in ("data", "eligibility", "registry", "gelato"):
        (dst / d).symlink_to((SMOKE / d).resolve())
    for d in ("splits", "selection", "queries", "ldl_tune"):
        shutil.copytree(SMOKE / d, dst / d, symlinks=False)
    cfg = load_config(PIPELINE_ROOT / "configs" / "pcfp_smoke.yaml", {"paths": {"outputs": str(root)}})
    unit = next(u for u in cfg["units"] if u["unit_id"] == UNIT)
    return cfg, unit, dst


def _problems(tree):
    cfg, unit, _ = tree
    return pipeline.audit_unit(cfg, unit)[1]


def test_clean_copy_passes(tree):
    assert _problems(tree) == []


def test_hidden_cell_in_training_sample_is_caught(tree):
    cfg, unit, dst = tree
    sdir = dst / "selection" / UNIT / "rep0" / "fold0" / "random@60" / "samples"
    smp = pd.read_csv(sdir / "budget_30.csv", keep_default_na=False, dtype=str)
    expo = pipeline.load_exposure(cfg, UNIT)
    core = pd.read_csv(sdir / "budget_30_lemmas.csv").query("role == 'core'").lemma_id.iloc[0]
    hidden = expo[core].hidden[0]
    row = smp.iloc[0].copy()
    row["lemma_id"], row["cell_norm"] = core, hidden
    pd.concat([smp, row.to_frame().T]).to_csv(sdir / "budget_30.csv", index=False)
    assert any("hidden cell of a core verb" in p for p in _problems(tree))


def test_candidate_in_selector_fit_and_extra_candidate_column_are_caught(tree):
    cfg, unit, dst = tree
    rdir = dst / "selection" / UNIT / "rep0" / "fold0" / "low_confidence@60" / "rounds" / "r1"
    tr = pd.read_csv(rdir / "train.csv", keep_default_na=False, dtype=str)
    cd = pd.read_csv(rdir / "candidates.csv", keep_default_na=False, dtype=str)
    row = tr.iloc[0].copy()
    row["lemma_id"] = cd.lemma_id.iloc[0]
    pd.concat([tr, row.to_frame().T]).to_csv(rdir / "train.csv", index=False)
    cd.assign(form="x").to_csv(rdir / "candidates.csv", index=False)
    probs = _problems(tree)
    assert any("selector training verbs" in p for p in probs)
    assert any("candidate columns" in p for p in probs)


def test_query_for_shown_cell_is_caught(tree):
    cfg, unit, dst = tree
    qp = dst / "queries" / UNIT / "rep0" / "fold0" / "random@60" / "budget_30" / "queries.csv"
    q = pd.read_csv(qp)
    expo = pipeline.load_exposure(cfg, UNIT)
    l = q.lemma_id.iloc[0]
    pd.concat([q, pd.DataFrame({"lemma_id": [l], "target_cell": [expo[l].shown[0]], "item_set": ["core"]})]).to_csv(qp, index=False)
    assert any("shown cell" in p for p in _problems(tree))
