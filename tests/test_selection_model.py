"""Selector model: input format, beam search scores, decode length, determinism."""

import math

import pytest
import torch

from morph_ldl.selection.acquisition import SelectionTask, build_examples
from morph_ldl.selection.fixtures import toy_forms
from morph_ldl.selection.model import (EOS, SelectorConfig, encode_input, train_selector)

TASK = SelectionTask("toy.V.orth.test", "NFIN", ("1;PRS;SG", "3;PRS;SG", "3;PL;PRS"))
TINY = dict(d_model=32, n_heads=2, n_enc_layers=1, n_dec_layers=1, d_ff=64, dropout=0.1, warmup_steps=20,
            max_steps=120, eval_every=40, early_stop_patience=2, batch_size=32, beam_size=4, num_threads=1,
            device="cpu")


@pytest.fixture(scope="module")
def data():
    forms = toy_forms(60, seed=3)
    ids = sorted(forms.lemma_id.unique())
    tr, _ = build_examples(forms[forms.lemma_id.isin(ids[:40])], TASK)
    dev, _ = build_examples(forms[forms.lemma_id.isin(ids[40:50])], TASK, mode="panel", with_gold_variants=True)
    return forms, tr, dev


@pytest.fixture(scope="module")
def selector(data):
    _, tr, dev = data
    return train_selector(tr, dev, SelectorConfig(**TINY), seed=11)


def test_input_format():
    toks = encode_input("NFIN", "a m a r e", "1;IND;PRS;SG")
    assert toks == ("<S:NFIN>", "a", "m", "a", "r", "e", "<T:1>", "<T:IND>", "<T:PRS>", "<T:SG>")
    with pytest.raises(ValueError):
        encode_input("NFIN", "<T:X> a", "1;SG")


def test_config_from_pipeline_cfg_rejects_unknown_keys():
    c = SelectorConfig.from_cfg({"selector": {"arch": "char_transformer", "d_model": 64}, "limits": {"max_threads": 3}})
    assert c.d_model == 64 and c.num_threads == 3
    with pytest.raises(ValueError):
        SelectorConfig.from_cfg({"selector": {"d_modle": 64}})
    assert c.max_decode_len(7) == int(2.0 * 7) + 10


def test_beam_logprob_matches_teacher_forced_score(selector, data):
    _, _, dev = data
    srcs = [e.src for e in dev[:6]]
    beams = selector.beam_search(srcs, [e.n_source_segments for e in dev[:6]], beam_size=4)
    for src, hyps in zip(srcs, beams):
        assert 1 <= len(hyps) <= 4
        lps = [h.logprob_sum for h in hyps]
        assert lps == sorted(lps, reverse=True)
        for h in hyps:
            assert h.hyp_len == len(h.segments)  # excludes EOS
            assert EOS not in h.segments
            assert h.logprob_sum == pytest.approx(selector.sequence_logprob(src, h.segments), abs=1e-4)
            assert h.logprob_sum < 0 and math.isfinite(h.logprob_sum)


def test_max_decode_length_comes_from_source_length(selector, data):
    _, _, dev = data
    cfg = selector.config
    old = (cfg.max_decode_len_factor, cfg.max_decode_len_offset)
    try:
        cfg.max_decode_len_factor, cfg.max_decode_len_offset = 0.0, 2
        beams = selector.beam_search([e.src for e in dev[:5]], [e.n_source_segments for e in dev[:5]])
        assert all(h.hyp_len <= 2 for hyps in beams for h in hyps)
        assert all(len(hyps) > 0 for hyps in beams)  # forced EOS still yields hypotheses
    finally:
        cfg.max_decode_len_factor, cfg.max_decode_len_offset = old


def test_training_is_deterministic_and_order_invariant(data):
    _, tr, dev = data
    a = train_selector(tr, dev, SelectorConfig(**{**TINY, "max_steps": 40}), seed=5)
    b = train_selector(list(reversed(tr)), dev, SelectorConfig(**{**TINY, "max_steps": 40}), seed=5)
    c = train_selector(tr, dev, SelectorConfig(**{**TINY, "max_steps": 40}), seed=6)
    assert a.info["model_hash"] == b.info["model_hash"]
    assert a.info["model_hash"] != c.info["model_hash"]
    srcs = [e.src for e in dev[:4]]
    n = [e.n_source_segments for e in dev[:4]]
    assert [[(h.segments, h.logprob_sum) for h in x] for x in a.beam_search(srcs, n)] == \
           [[(h.segments, h.logprob_sum) for h in x] for x in b.beam_search(srcs, n)]


def test_early_stopping_records_dev_curve(selector):
    info = selector.info
    assert info["best_step"] in [c["step"] for c in info["dev_curve"]]
    assert info["n_dev_examples"] > 0 and info["device"] == "cpu"
    assert 0.0 <= info["dev_acc"] <= 1.0


def test_shared_embeddings_never_emit_feature_tokens(data):
    _, tr, dev = data
    sel = train_selector(tr, dev, SelectorConfig(**{**TINY, "share_embeddings": True, "max_steps": 40}), seed=1)
    assert sel.src_vocab is sel.tgt_vocab
    assert sel.model.src_emb is sel.model.tgt_emb
    beams = sel.beam_search([e.src for e in dev], [e.n_source_segments for e in dev])
    for hyps in beams:
        for h in hyps:
            assert not any(s.startswith("<") and s.endswith(">") and len(s) > 2 for s in h.segments)


def test_global_torch_state_restored(data):
    _, tr, dev = data
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    train_selector(tr[:10], dev[:3], SelectorConfig(**{**TINY, "max_steps": 5}), seed=1)
    assert (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()) == before
