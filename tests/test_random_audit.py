"""The repeated-random artifact audit catches injected leaks (runs on a copy of the smoke outputs).

Skipped unless `outputs/pcfp_v2_smoke` holds a finished smoke run (scripts/run_pcfp.sh
configs/pcfp_v2_smoke.yaml)."""

import shutil

import pandas as pd
import pytest

from morph_ldl.config import PIPELINE_ROOT, load_config
from morph_ldl.cv import pipeline

pytestmark = pytest.mark.slow
SMOKE = PIPELINE_ROOT / "outputs" / "pcfp_v2_smoke"
UNIT = "ita.V.orth.mgn"
RUN = "random@240"


@pytest.fixture()
def tree(tmp_path):
    if not (SMOKE / "selection" / UNIT / "rep1" / "fold0" / RUN / "samples").exists():
        pytest.skip("no finished smoke run")
    root = tmp_path / "out"
    dst = root / "pcfp_v2_smoke"
    dst.mkdir(parents=True)
    for d in ("data", "eligibility", "registry", "gelato"):
        (dst / d).symlink_to((SMOKE / d).resolve())
    for d in ("splits", "selection", "queries", "ldl_tune"):
        shutil.copytree(SMOKE / d, dst / d, symlinks=False)
    cfg = load_config(PIPELINE_ROOT / "configs" / "pcfp_v2_smoke.yaml", {"paths": {"outputs": str(root)}})
    unit = next(u for u in cfg["units"] if u["unit_id"] == UNIT)
    return cfg, unit, dst


def _problems(tree):
    cfg, unit, _ = tree
    return pipeline.audit_unit(cfg, unit)[1]


def test_clean_copy_passes(tree):
    assert _problems(tree) == []


def test_hidden_core_cell_in_sample_is_caught(tree):
    cfg, unit, dst = tree
    sdir = dst / "selection" / UNIT / "rep1" / "fold2" / RUN / "samples"
    smp = pd.read_csv(sdir / "budget_20.csv", keep_default_na=False, dtype=str)
    core = pd.read_csv(sdir / "budget_20_lemmas.csv").query("role == 'core'").lemma_id.iloc[0]
    row = smp.iloc[0].copy()
    row["lemma_id"], row["cell_norm"] = core, pipeline.load_exposure(cfg, UNIT)[core].hidden[0]
    pd.concat([smp, row.to_frame().T]).to_csv(sdir / "budget_20.csv", index=False)
    assert any("hidden cell of a core verb" in p for p in _problems(tree))


def test_training_verbs_not_from_the_draw_are_caught(tree):
    cfg, unit, dst = tree
    p = dst / "selection" / UNIT / "rep0" / "fold1" / RUN / "order.csv"
    o = pd.read_csv(p)
    other = pd.read_csv(dst / "selection" / UNIT / "rep1" / "fold1" / RUN / "order.csv")
    o.loc[0, "lemma_id"] = next(l for l in other.lemma_id if l not in set(o.lemma_id))
    o.to_csv(p, index=False)
    assert any("random draw" in x for x in _problems(tree))


def test_core_sets_changing_between_draws_are_caught(tree):
    cfg, unit, dst = tree
    p = dst / "splits" / UNIT / "rep1" / "split_manifest.csv"
    m = pd.read_csv(p, dtype={"lemma_id": str, "group_id": str, "role": str})
    # in draw 1, swap a fold-0 core verb with a singleton-group verb that is no fold's core
    # verb (the draw's manifest stays internally valid; only the cross-draw check can see it)
    singles = m.groupby("group_id").lemma_id.nunique().loc[lambda s: s == 1].index
    f0 = m[m.outer_fold == 0]
    a = f0[(f0.role == "core") & f0.group_id.isin(singles)].lemma_id.iloc[0]
    b = f0[(f0.role == "unused") & f0.group_id.isin(singles)
           & ~f0.lemma_id.isin(m.loc[m.role == "core", "lemma_id"])].lemma_id.iloc[0]
    m.loc[(m.outer_fold == 0) & (m.lemma_id == a), "role"] = "unused"
    m.loc[(m.outer_fold == 0) & (m.lemma_id == b), "role"] = "core"
    m.to_csv(p, index=False)
    probs = _problems(tree)
    assert any("core sets differ across random draws" in x for x in probs)
