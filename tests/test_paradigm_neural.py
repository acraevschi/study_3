"""Unit and mathematical verification tests for the Hard-Monotonic Neural Character Transducer
(`analyses/paradigm/neural/`).
"""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import pytest
import torch

from analyses.paradigm.neural.monotonic_transducer import (
    CellVocab,
    CharVocab,
    HardMonotonicTransducer,
    TransducerConfig,
    collate_transducer_batch,
    ensemble_beam_decode,
    ensemble_beam_decode_batch,
    ensemble_log_prob_pairs,
    resolve_device,
)
from analyses.paradigm.neural.train_eval import (
    LEMMA_OUTCOME_COLUMNS,
    UNIT_OUTCOME_COLUMNS,
    LemmaParadigm,
    TrainEvalConfig,
    chu_liu_edmonds,
    evaluate_unit_neural,
    load_split_manifest,
    prescreen_candidate_edges,
    select_optimal_arborescence,
    write_neural_outcomes,
)


def _available_devices() -> list[str]:
    devs = ["cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("dev_name", _available_devices())
def test_exact_normalization_over_sigma_star_closed_alphabet(dev_name: str) -> None:
    """Verify sum_{y in Sigma^{<=L}} q_theta(y | x, c_s, c_t) == 1.0 to < 1e-4 on CPU & MPS."""
    torch.manual_seed(42)
    char_vocab = CharVocab(["a", "b"], allow_unk=False)
    cell_vocab = CellVocab(["NOM", "GEN"])
    cfg = TransducerConfig(
        char_emb_dim=16,
        cell_emb_dim=8,
        enc_hidden_dim=24,
        dec_hidden_dim=24,
        joint_hidden_dim=24,
        dropout=0.0,
        init_copy_bias=0.8,
    )
    device = resolve_device(dev_name)
    model = HardMonotonicTransducer(char_vocab, cell_vocab, cfg).to(device)
    # 1. On unmodified initialized weights, verify total mass <= 1.0 and strictly
    #    monotonically approaches 1.0 as L grows from 4 -> 8 -> 12.
    model.eval()
    masses_by_l: list[float] = []
    for l_check in (4, 8, 12):
        tgts_l = [
            tuple(seq)
            for length in range(l_check + 1)
            for seq in itertools.product(("a", "b"), repeat=length)
        ]
        lps_l = model.log_prob_pairs([(("a", "b"), t, "NOM", "GEN") for t in tgts_l], batch_size=1024)
        m_val = math.fsum(math.exp(lp) for lp in lps_l)
        assert m_val <= 1.0 + 1e-5
        masses_by_l.append(m_val)
    assert masses_by_l[0] < masses_by_l[1] < masses_by_l[2]
    assert abs(masses_by_l[2] - 1.0) < 1e-3

    # 2. Verify sum_{y in Sigma^{<=12}} q_theta(y | x, c_s, c_t) == 1.0 to < 1e-4 (and < 1e-5)!
    with torch.no_grad():
        model.out_proj.bias[3:5] = -0.6       # dampen infinite-insert geometric tail above L=12

    max_l = 12
    all_targets = [
        tuple(seq)
        for length in range(max_l + 1)
        for seq in itertools.product(("a", "b"), repeat=length)
    ]
    assert len(all_targets) == (2 ** (max_l + 1)) - 1

    for src in ((), ("a",), ("a", "b", "a")):
        examples = [(src, tgt, "NOM", "GEN") for tgt in all_targets]
        lps = model.log_prob_pairs(examples, batch_size=1024)
        total_prob = math.fsum(math.exp(lp) for lp in lps)
        assert abs(total_prob - 1.0) < 1e-4, (
            f"Device={dev_name}, src={src}: sum_{{y<=L}} q(y|x) = {total_prob:.8f} != 1.0"
        )


def test_exact_normalization_with_unk_and_oov_source_copy() -> None:
    """Verify allow_unk=True with unk_space_size=2 sums to 1.0 over (Sigma_known U {u1, u2})*."""
    torch.manual_seed(123)
    char_vocab = CharVocab(["a"], allow_unk=True, unk_space_size=2)
    cell_vocab = CellVocab(["C1", "C2"])
    cfg = TransducerConfig(
        char_emb_dim=12,
        cell_emb_dim=8,
        enc_hidden_dim=16,
        dec_hidden_dim=16,
        joint_hidden_dim=16,
        dropout=0.0,
        init_copy_bias=1.0,
    )
    model = HardMonotonicTransducer(char_vocab, cell_vocab, cfg)
    with torch.no_grad():
        model.out_proj.bias[0] = 1.8
        model.out_proj.bias[3:5] = -0.8  # dampen infinite-insert geometric tail above L=8
    model.eval()

    universe = ("a", "u1", "u2")  # 1 known symbol + 2 OOV symbols matching unk_space_size=2
    max_l = 8
    all_targets = [
        tuple(seq)
        for length in range(max_l + 1)
        for seq in itertools.product(universe, repeat=length)
    ]

    # Case 1: in-vocab source; Case 2: OOV symbol `u1` in source (tests OOV COPY exactness!)
    for src in (("a", "a"), ("u1", "a")):
        examples = [(src, tgt, "C1", "C2") for tgt in all_targets]
        lps = model.log_prob_pairs(examples, batch_size=1024)
        total_prob = math.fsum(math.exp(lp) for lp in lps)
        assert abs(total_prob - 1.0) < 1e-4, f"src={src}: total_prob={total_prob:.8f}"


def test_batch_padding_invariance_and_cpu_mps_parity() -> None:
    """Verify batched forward/backward is invariant to padding and matches across CPU & MPS."""
    torch.manual_seed(7)
    char_vocab = CharVocab(["k", "a", "t", "e", "o", "i"], allow_unk=True)
    cell_vocab = CellVocab(["SG", "PL", "INE"])
    cfg = TransducerConfig(
        char_emb_dim=16,
        cell_emb_dim=8,
        enc_hidden_dim=24,
        dec_hidden_dim=24,
        joint_hidden_dim=24,
        dropout=0.0,
    )
    model_cpu = HardMonotonicTransducer(char_vocab, cell_vocab, cfg)
    model_cpu.eval()

    examples = [
        ("", "a", "SG", "PL"),
        ("k", "", "SG", "PL"),
        ("kat", "katte", "SG", "INE"),
        ("talo", "talossa", "SG", "INE"),
        ("ki", "kissa", "PL", "INE"),
    ]
    batched_lps = model_cpu.log_prob_pairs(examples, batch_size=len(examples))
    single_lps = [model_cpu.log_prob_pairs([ex], batch_size=1)[0] for ex in examples]
    for b_lp, s_lp in zip(batched_lps, single_lps):
        assert abs(b_lp - s_lp) < 1e-5

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model_mps = HardMonotonicTransducer(char_vocab, cell_vocab, cfg).to(torch.device("mps"))
        model_mps.load_state_dict(model_cpu.state_dict())

        # Forward parity
        mps_lps = model_mps.log_prob_pairs(examples, batch_size=len(examples))
        for c_lp, m_lp in zip(batched_lps, mps_lps):
            assert abs(c_lp - m_lp) < 1e-4

        # Backward parity & NaN-free gradients
        model_cpu.train()
        model_mps.train()
        batch_cpu = collate_transducer_batch(examples, char_vocab, cell_vocab, torch.device("cpu"))
        batch_mps = batch_cpu.to(torch.device("mps"))
        loss_cpu = -model_cpu(batch_cpu).mean()
        loss_mps = -model_mps(batch_mps).mean()
        loss_cpu.backward()
        loss_mps.backward()
        for (n, p_c), (_, p_m) in zip(model_cpu.named_parameters(), model_mps.named_parameters()):
            assert p_c.grad is not None and p_m.grad is not None
            assert not torch.isnan(p_m.grad).any(), f"NaN grad on MPS for {n}"
            max_diff = float((p_c.grad - p_m.grad.cpu()).abs().max().item())
            assert max_diff < 1e-3, f"Grad mismatch on {n}: {max_diff}"


def test_ensemble_logsumexp_and_beam_decoder_exactness() -> None:
    """Verify 3-seed logsumexp averaging and exact string-marginal beam decoding."""
    char_vocab = CharVocab(["a", "b", "s"], allow_unk=False)
    cell_vocab = CellVocab(["SG", "PL"])
    cfg = TransducerConfig(
        char_emb_dim=12,
        cell_emb_dim=8,
        enc_hidden_dim=16,
        dec_hidden_dim=16,
        joint_hidden_dim=16,
        dropout=0.0,
    )
    models = []
    for seed in (10, 20, 30):
        torch.manual_seed(seed)
        m = HardMonotonicTransducer(char_vocab, cell_vocab, cfg)
        m.eval()
        models.append(m)

    pair = [("ab", "abs", "SG", "PL")]
    per_seed = [m.log_prob_pairs(pair)[0] for m in models]
    expected_ens = math.log(sum(math.exp(lp) for lp in per_seed) / 3.0)
    got_ens = ensemble_log_prob_pairs(models, pair)[0]
    assert abs(got_ens - expected_ens) < 1e-6

    # Beam decoder must assign the exact same marginal ensemble log-prob to every returned string
    decoded = ensemble_beam_decode(models, "ab", "SG", "PL", beam_size=6, max_len=6)
    assert len(decoded) == 6
    check_pairs = [("ab", seq, "SG", "PL") for seq, _ in decoded]
    dp_lps = ensemble_log_prob_pairs(models, check_pairs)
    for (seq, beam_lp), dp_lp in zip(decoded, dp_lps):
        assert abs(beam_lp - dp_lp) < 1e-5, f"Beam vs DP mismatch on {seq}: {beam_lp} vs {dp_lp}"


def test_oov_source_stem_beam_decode_matches_forward_dp() -> None:
    """Verify OOV stem symbol ('š' in 'špacero' -> 'špacera') decodes & matches forward DP."""
    # 'š' is intentionally absent from CharVocab known_symbols
    char_vocab = CharVocab(["p", "a", "c", "e", "r", "o"], allow_unk=True, unk_space_size=256)
    cell_vocab = CellVocab(["INF", "PRS3"])
    cfg = TransducerConfig(
        char_emb_dim=16,
        cell_emb_dim=8,
        enc_hidden_dim=24,
        dec_hidden_dim=24,
        joint_hidden_dim=24,
        dropout=0.0,
        init_copy_bias=4.5,  # favor stem copying so 'špacer-' stays at top of beam
    )
    models = []
    a_emit_id = char_vocab.sym_to_emit["a"]
    v_emit = char_vocab.num_emit
    for seed in (101, 202, 303):
        torch.manual_seed(seed)
        m = HardMonotonicTransducer(char_vocab, cell_vocab, cfg)
        with torch.no_grad():
            # Encourage SUBSTITUTE('a') alongside COPY so ('š','p','a','c','e','r','a') is top-ranked
            m.out_proj.bias[3 + v_emit + a_emit_id] = 3.2
        m.eval()
        models.append(m)

    src_seq = ("š", "p", "a", "c", "e", "r", "o")
    target_inflected = ("š", "p", "a", "c", "e", "r", "a")

    decoded = ensemble_beam_decode(models, src_seq, "INF", "PRS3", beam_size=8, max_len=9)
    decoded_dict = dict(decoded)
    assert target_inflected in decoded_dict, (
        f"Expected OOV-stem inflected form {target_inflected} in beam, got {list(decoded_dict.keys())}"
    )

    dp_lps = ensemble_log_prob_pairs(
        models, [(src_seq, seq, "INF", "PRS3") for seq, _ in decoded]
    )
    for (seq, beam_lp), dp_lp in zip(decoded, dp_lps):
        assert abs(beam_lp - dp_lp) < 1e-5, (
            f"OOV beam vs forward-DP mismatch on {seq}: beam={beam_lp:.6f}, dp={dp_lp:.6f}"
        )


def _brute_force_min_arborescence(n: int, root: int, cost: list[list[float]]) -> float:
    best = float("inf")
    non_root = [v for v in range(n) if v != root]
    for parents_tuple in itertools.product(range(n), repeat=len(non_root)):
        pmap = dict(zip(non_root, parents_tuple))
        if any(u == v for v, u in pmap.items()):
            continue
        # Verify acyclicity (every node reaches root)
        ok = True
        for v in non_root:
            curr = v
            steps = 0
            while curr != root and steps < n:
                curr = pmap[curr]
                steps += 1
            if curr != root:
                ok = False
                break
        if ok:
            c_tot = sum(cost[u][v] for v, u in pmap.items())
            if c_tot < best:
                best = c_tot
    return best


def test_chu_liu_edmonds_against_brute_force() -> None:
    """Verify Chu-Liu-Edmonds matches brute-force search across random cyclic digraphs."""
    gen = torch.Generator().manual_seed(2026)
    for trial in range(25):
        n = 5 if trial < 15 else 6
        mat = torch.rand((n, n), generator=gen).tolist()
        for i in range(n):
            mat[i][i] = 0.0
        # Inject cheap 2-cycles and 3-cycles to force recursive cycle contractions
        if trial % 2 == 0:
            mat[1][2] = 0.01
            mat[2][1] = 0.01
        else:
            mat[1][2] = 0.01
            mat[2][3] = 0.01
            mat[3][1] = 0.01

        for r in range(n):
            parents, edmonds_cost = chu_liu_edmonds(n, r, mat)
            bf_cost = _brute_force_min_arborescence(n, r, mat)
            assert len(parents) == n - 1
            assert abs(edmonds_cost - bf_cost) < 1e-9

        r_star, _, opt_cost = select_optimal_arborescence([f"C{i}" for i in range(n)], mat)
        all_root_costs = [_brute_force_min_arborescence(n, r, mat) for r in range(n)]
        assert abs(opt_cost - min(all_root_costs)) < 1e-9
        assert 0 <= r_star < n


def test_leakage_rejection_and_end_to_end_toy_paradigm(tmp_path: Path) -> None:
    """Verify split leakage guard and end-to-end neural evaluation + CSV schema parity."""
    # 1. Verify split leakage detection raises ValueError
    bad_split = tmp_path / "bad_split.csv"
    with open(bad_split, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draw", "fold", "lemma_id", "leakage_group_id", "role", "inner_role"])
        w.writerow([0, 0, "l1", "grp_shared", "train", "subtrain"])
        w.writerow([0, 0, "l2", "grp_dev", "train", "dev"])
        w.writerow([0, 0, "l3", "grp_shared", "test", "test"])
    with pytest.raises(ValueError, match="Leakage group overlap"):
        load_split_manifest(bad_split)

    # 2. Build a clean synthetic agglutinative paradigm dataset (4 cells, 30 lemmas, 1 irregular)
    lex_dir = tmp_path / "lexicons"
    spl_dir = tmp_path / "splits"
    out_dir = tmp_path / "results"
    lex_dir.mkdir()
    spl_dir.mkdir()

    unit_id = "toy1234_N_orth"
    stems = [
        f"{c1}{v1}{c2}"
        for c1 in ("k", "p", "t", "m", "n")
        for v1 in ("a", "o", "i")
        for c2 in ("l", "r")
    ]  # 30 distinct stems

    lex_path = lex_dir / f"{unit_id}.csv"
    with open(lex_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "unit_id", "glottocode", "iso639_3", "pos", "representation", "source_id",
            "lemma_id", "lemma_label", "leakage_group_id", "cell", "raw_form",
            "segmented_form", "variant_idx", "shape_id",
        ])
        for idx, stem in enumerate(stems):
            lid = f"lem_{idx:02d}"
            grp = f"grp_{idx:02d}"
            # Make lem_29 suppletive in PL; make ACC syncretic with NOM on half the stems
            is_sync = idx % 2 == 0
            nom = stem
            acc = stem if is_sync else f"{stem}n"
            gen = f"{stem}in"
            pl = "xqz" if idx == 29 else f"{stem}t"
            shape = "S_sync" if is_sync else "S_dist"
            for cell, form in (("NOM", nom), ("ACC", acc), ("GEN", gen), ("PL", pl)):
                w.writerow([
                    unit_id, "toy1234", "toy", "N", "orth", "synthetic_fixture",
                    lid, stem, grp, cell, form, " ".join(form), 0, shape,
                ])

    split_path = spl_dir / f"{unit_id}_uniform120.csv"
    with open(split_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draw", "fold", "lemma_id", "leakage_group_id", "role", "inner_role"])
        # Single fold smoke check: 18 subtrain, 6 dev, 6 test (including regular + irregular lem_29)
        for idx in range(30):
            lid = f"lem_{idx:02d}"
            grp = f"grp_{idx:02d}"
            if idx < 18:
                role, inner = "train", "subtrain"
            elif idx < 24:
                role, inner = "train", "dev"
            else:
                role, inner = "test", "test"
            w.writerow([0, 0, lid, grp, role, inner])

    cfg = TrainEvalConfig(
        n_seeds=3,
        max_epochs=18,
        patience=5,
        batch_size=32,
        eval_batch_size=64,
        lr=3e-3,
        refit_full_train=True,
        beam_size=3,
        n_bootstrap=50,
        device="cpu",
        num_threads=2,
        transducer=TransducerConfig(
            char_emb_dim=16,
            cell_emb_dim=8,
            enc_hidden_dim=32,
            dec_hidden_dim=32,
            joint_hidden_dim=32,
            dropout=0.0,
        ),
    )
    unit_row, lemma_rows = evaluate_unit_neural(
        lex_path, split_path, regime="uniform120", cfg=cfg
    )
    u_file, l_file = write_neural_outcomes([unit_row], lemma_rows, out_dir, "uniform120")

    with open(u_file, encoding="utf-8") as f:
        u_parsed = list(csv.DictReader(f))
        assert list(u_parsed[0].keys()) == UNIT_OUTCOME_COLUMNS
    with open(l_file, encoding="utf-8") as f:
        l_parsed = list(csv.DictReader(f))
        assert list(l_parsed[0].keys()) == LEMMA_OUTCOME_COLUMNS

    by_lid = {str(r["lemma_id"]): r for r in lemma_rows}
    # Suppletive lem_29 must have strictly higher unpredictability than regular test lemmas
    reg_h = [float(by_lid[f"lem_{i:02d}"]["h_tree_lemma"]) for i in range(24, 29)]
    sup_h = float(by_lid["lem_29"]["h_tree_lemma"])
    assert sup_h > max(reg_h) + 0.5
    # Syncretic test lemmas must report n_distinct_forms == 3, non-syncretic == 4
    assert int(by_lid["lem_24"]["n_distinct_forms"]) == 3
    assert int(by_lid["lem_25"]["n_distinct_forms"]) == 4


@pytest.mark.parametrize("dev_name", _available_devices())
def test_batched_beam_decode_matches_single_beam_on_cpu_and_mps(dev_name: str) -> None:
    """Verify `ensemble_beam_decode_batch` matches `ensemble_beam_decode` across lengths & OOV."""
    char_vocab = CharVocab(["p", "a", "c", "e", "r", "o", "s", "t"], allow_unk=True, unk_space_size=256)
    cell_vocab = CellVocab(["INF", "PRS1", "PRS3"])
    cfg = TransducerConfig(
        char_emb_dim=16,
        cell_emb_dim=8,
        enc_hidden_dim=24,
        dec_hidden_dim=24,
        joint_hidden_dim=24,
        dropout=0.0,
    )
    dev = resolve_device(dev_name)
    models = []
    for seed in (11, 22, 33):
        torch.manual_seed(seed)
        m = HardMonotonicTransducer(char_vocab, cell_vocab, cfg).to(dev)
        m.eval()
        models.append(m)

    queries = [
        ("pa", "INF", "PRS1"),
        ("pacero", "INF", "PRS3"),
        (("š", "p", "a", "c", "e", "r", "o"), "INF", "PRS3"),
        ("st", "PRS1", "INF"),
        ("", "INF", "PRS1"),
    ]
    batched = ensemble_beam_decode_batch(models, queries, beam_size=5, max_len=9, batch_size=2)
    assert len(batched) == len(queries)
    for q, b_res in zip(queries, batched):
        s_res = ensemble_beam_decode(models, q[0], q[1], q[2], beam_size=5, max_len=9)
        assert [seq for seq, _ in b_res] == [seq for seq, _ in s_res]
        for (_, lp_b), (_, lp_s) in zip(b_res, s_res):
            assert abs(lp_b - lp_s) < 1e-4


def test_prescreen_candidate_edges_matches_mgl_topology() -> None:
    """Verify `prescreen_candidate_edges` matches `mgl.topology.prescreen_candidate_edges` 1-to-1."""
    from analyses.paradigm.mgl.topology import prescreen_candidate_edges as mgl_prescreen

    cells = [f"C_{i:02d}" for i in range(45)]
    sub_p = []
    mgl_p = []
    for idx in range(25):
        stem = f"st{idx % 5}{'a' if idx % 2 == 0 else 'o'}"
        forms = {c: tuple(f"{stem}_{c[-2:]}") for c in cells}
        raw = {c: "".join(forms[c]) for c in cells}
        sub_p.append(LemmaParadigm(f"l{idx}", f"g{idx}", "S0", forms, raw))
        mgl_p.append(forms)

    seed_key = "test_unit|uniform120|0|0"
    n_edges, n_pool = prescreen_candidate_edges(cells, sub_p, top_k_parents=12, seed_key=seed_key, threshold=40)
    m_edges, m_pool = mgl_prescreen(cells, mgl_p, top_k_parents=12, seed_key=seed_key)
    assert n_pool == m_pool
    assert set(n_edges) == m_edges

