"""Hard-Monotonic Neural Character Transducer (Wu & Cotterell 2019 style).

Models the exact string conditional probability distribution q_theta(y | x, c_s, c_t)
over y in Sigma* via a character BiLSTM encoder, learned source/target cell embeddings,
a causal target-prefix LSTM decoder, and an exact O(|x| * |y|) anti-diagonal wavefront
forward dynamic program over monotonic edit operations:
  - COPY:          advance source pointer i -> i + 1 and emit x_{i+1}  (valid for i < |x|)
  - DELETE:        advance source pointer i -> i + 1 without emitting  (valid for i < |x|)
  - SUBSTITUTE(c): advance source pointer i -> i + 1 and emit c        (valid for i < |x|)
  - INSERT(c):     keep source pointer at i and emit c                 (valid for i <= |x|)
  - STOP:          terminate generation at source <eos>                (valid iff i == |x|)

Because the decoder hidden state depends strictly on the emitted target prefix y_{1:j}
(and not on alignment history), the (M+1) x (N+1) lattice satisfies the exact Markov
property:
  1. Teacher-forced target prefixes (<bos>, y_1, ..., y_N) are encoded in parallel
     across all j = 0..N, and all lattice log-probabilities are computed in a single
     fused batched projection before the wavefront DP.
  2. At every reachable lattice state (i, y_{1:j}), outgoing action probabilities sum
     strictly to 1.0, guaranteeing sum_{y in Sigma*} q_theta(y | x, c_s, c_t) == 1.0
     with zero label-smoothing distortion.
  3. Portable across cuda -> mps -> cpu with zero in-place tensor mutations and
     automatic CPU fallback if any backend runtime quirk is encountered.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

NEG_INF_SAFE: float = -1e9
LN2: float = math.log(2.0)


def resolve_device(device_spec: str = "auto") -> torch.device:
    """Resolve device specification ('auto', 'cuda', 'mps', 'cpu') to a torch.device."""
    spec = (device_spec or "auto").strip().lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda', but CUDA is not available.")
        return torch.device("cuda")
    if spec == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested device 'mps', but Apple MPS is not available.")
        return torch.device("mps")
    if spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device specification {device_spec!r}; expected auto, cuda, mps, or cpu.")


def to_symbol_tuple(seq: str | Sequence[str]) -> tuple[str, ...]:
    """Convert either a raw string or a sequence of grapheme/segment symbols into a tuple."""
    if isinstance(seq, str):
        return tuple(seq)
    return tuple(seq)


class CharVocab:
    """Symbol vocabulary separating RNN input tokens from generative emission symbols.

    RNN input IDs:
      0: <pad>, 1: <bos>, 2: <eos>, [3: <unk> if allow_unk], then known symbols.
    Emission IDs (0 .. num_emit - 1):
      - If allow_unk=False: exact closed alphabet Sigma = known_symbols (size |Sigma|).
      - If allow_unk=True:  index 0 is <unk> representing `unk_space_size` unseen symbols
        (each unseen symbol receives p(<unk>) / unk_space_size on INSERT/SUBSTITUTE),
        and indices 1..|Sigma_known| are known_symbols. This preserves exact normalization
        sum_{y in (Sigma_known U Sigma_unk)*} q(y | x) == 1.0 without ever peeking at test
        symbols! Meanwhile COPY matches raw symbol strings directly (`x[i] == y[j]`) with
        full p_copy mass and zero unk splitting.
    """

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        allow_unk: bool = True,
        unk_space_size: int = 256,
    ) -> None:
        unique_syms = sorted(set(symbols))
        if not unique_syms and not allow_unk:
            raise ValueError("Closed CharVocab (allow_unk=False) requires at least 1 symbol.")
        if allow_unk and unk_space_size < 1:
            raise ValueError("unk_space_size must be >= 1.")

        self.allow_unk = bool(allow_unk)
        self.unk_space_size = int(unk_space_size)
        self.known_symbols: tuple[str, ...] = tuple(unique_syms)

        if self.allow_unk:
            self.unk_rnn_id: int | None = 3
            self.unk_emit_id: int | None = 0
            rnn_offset = 4
            emit_offset = 1
        else:
            self.unk_rnn_id = None
            self.unk_emit_id = None
            rnn_offset = 3
            emit_offset = 0

        self.sym_to_rnn: dict[str, int] = {
            s: rnn_offset + idx for idx, s in enumerate(self.known_symbols)
        }
        self.sym_to_emit: dict[str, int] = {
            s: emit_offset + idx for idx, s in enumerate(self.known_symbols)
        }
        self.emit_to_sym: list[str] = (["<unk>"] if self.allow_unk else []) + list(self.known_symbols)
        self.emit_to_rnn: list[int] = (
            ([self.unk_rnn_id] if self.unk_rnn_id is not None else [])
            + [rnn_offset + idx for idx in range(len(self.known_symbols))]
        )
        self.num_rnn_tokens: int = rnn_offset + len(self.known_symbols)
        self.num_emit: int = emit_offset + len(self.known_symbols)
        self.log_unk_penalty: float = -math.log(self.unk_space_size) if self.allow_unk else 0.0

    def encode_source_rnn(self, seq: Sequence[str]) -> list[int]:
        """Encode source symbols plus trailing EOS_ID (length M + 1)."""
        out: list[int] = []
        for s in seq:
            rid = self.sym_to_rnn.get(s)
            if rid is None:
                if self.unk_rnn_id is None:
                    raise KeyError(f"Unseen source symbol {s!r} in closed CharVocab.")
                rid = self.unk_rnn_id
            out.append(rid)
        out.append(self.EOS_ID)
        return out

    def encode_target_prefix_rnn(self, seq: Sequence[str]) -> list[int]:
        """Encode BOS_ID followed by target symbols y_1..y_N (length N + 1)."""
        out: list[int] = [self.BOS_ID]
        for s in seq:
            rid = self.sym_to_rnn.get(s)
            if rid is None:
                if self.unk_rnn_id is None:
                    raise KeyError(f"Unseen target symbol {s!r} in closed CharVocab.")
                rid = self.unk_rnn_id
            out.append(rid)
        return out

    def encode_target_emit(self, seq: Sequence[str]) -> tuple[list[int], list[float]]:
        """Encode target emission IDs and per-position log-unk penalties (length N)."""
        eids: list[int] = []
        pens: list[float] = []
        for s in seq:
            eid = self.sym_to_emit.get(s)
            if eid is None:
                if self.unk_emit_id is None:
                    raise KeyError(f"Unseen target symbol {s!r} in closed CharVocab.")
                eids.append(self.unk_emit_id)
                pens.append(self.log_unk_penalty)
            else:
                eids.append(eid)
                pens.append(0.0)
        return eids, pens


class CellVocab:
    """Vocabulary of paradigm cell labels."""

    def __init__(self, cells: Iterable[str]) -> None:
        unique_cells = sorted(set(cells))
        if not unique_cells:
            raise ValueError("CellVocab requires at least 1 cell label.")
        self.cells: tuple[str, ...] = tuple(unique_cells)
        self.cell_to_id: dict[str, int] = {c: i for i, c in enumerate(self.cells)}

    def __len__(self) -> int:
        return len(self.cells)

    def encode(self, cell: str) -> int:
        return self.cell_to_id[cell]


@dataclass(frozen=True)
class TransducerConfig:
    """Architecture hyperparameters for HardMonotonicTransducer."""

    char_emb_dim: int = 64
    cell_emb_dim: int = 32
    enc_hidden_dim: int = 128
    dec_hidden_dim: int = 128
    joint_hidden_dim: int = 128
    dropout: float = 0.2
    init_copy_bias: float = 2.0
    init_stop_bias: float = 1.5


@dataclass
class EncodedBatch:
    """Collated tensor batch for parallel transducer training or scoring."""

    src_rnn_ids: torch.Tensor       # (B, M_max + 1) long
    src_lens: torch.Tensor          # (B,) long, values M_b >= 0
    tgt_prefix_rnn_ids: torch.Tensor  # (B, N_max + 1) long
    tgt_emit_ids: torch.Tensor      # (B, max(N_max, 1)) long
    tgt_unk_penalties: torch.Tensor  # (B, max(N_max, 1)) float
    tgt_lens: torch.Tensor          # (B,) long, values N_b >= 0
    copy_match_mask: torch.Tensor   # (B, max(M_max, 1), max(N_max, 1)) bool
    src_cell_ids: torch.Tensor      # (B,) long
    tgt_cell_ids: torch.Tensor      # (B,) long
    m_max: int
    n_max: int

    def to(self, device: torch.device) -> EncodedBatch:
        return EncodedBatch(
            src_rnn_ids=self.src_rnn_ids.to(device),
            src_lens=self.src_lens.to(device),
            tgt_prefix_rnn_ids=self.tgt_prefix_rnn_ids.to(device),
            tgt_emit_ids=self.tgt_emit_ids.to(device),
            tgt_unk_penalties=self.tgt_unk_penalties.to(device),
            tgt_lens=self.tgt_lens.to(device),
            copy_match_mask=self.copy_match_mask.to(device),
            src_cell_ids=self.src_cell_ids.to(device),
            tgt_cell_ids=self.tgt_cell_ids.to(device),
            m_max=self.m_max,
            n_max=self.n_max,
        )


def collate_transducer_batch(
    examples: Sequence[tuple[str | Sequence[str], str | Sequence[str], str, str]],
    char_vocab: CharVocab,
    cell_vocab: CellVocab,
    device: torch.device | None = None,
) -> EncodedBatch:
    """Collate (src_seq, tgt_seq, src_cell, tgt_cell) examples into an EncodedBatch."""
    if not examples:
        raise ValueError("Cannot collate an empty batch.")

    b_size = len(examples)
    src_tuples = [to_symbol_tuple(ex[0]) for ex in examples]
    tgt_tuples = [to_symbol_tuple(ex[1]) for ex in examples]
    m_lens = [len(s) for s in src_tuples]
    n_lens = [len(t) for t in tgt_tuples]
    m_max = max(m_lens)
    n_max = max(n_lens)
    m_alloc = max(m_max, 1)
    n_alloc = max(n_max, 1)

    src_rnn = torch.full((b_size, m_max + 1), char_vocab.PAD_ID, dtype=torch.long)
    tgt_pref = torch.full((b_size, n_max + 1), char_vocab.PAD_ID, dtype=torch.long)
    tgt_emit = torch.zeros((b_size, n_alloc), dtype=torch.long)
    tgt_pen = torch.zeros((b_size, n_alloc), dtype=torch.float32)
    copy_mask = torch.zeros((b_size, m_alloc, n_alloc), dtype=torch.bool)
    c_src = torch.empty(b_size, dtype=torch.long)
    c_tgt = torch.empty(b_size, dtype=torch.long)

    for b, (s_tup, t_tup, cs, ct) in enumerate(zip(src_tuples, tgt_tuples, (e[2] for e in examples), (e[3] for e in examples))):
        s_ids = char_vocab.encode_source_rnn(s_tup)
        src_rnn[b, : len(s_ids)] = torch.tensor(s_ids, dtype=torch.long)

        tp_ids = char_vocab.encode_target_prefix_rnn(t_tup)
        tgt_pref[b, : len(tp_ids)] = torch.tensor(tp_ids, dtype=torch.long)

        if t_tup:
            e_ids, pens = char_vocab.encode_target_emit(t_tup)
            tgt_emit[b, : len(e_ids)] = torch.tensor(e_ids, dtype=torch.long)
            tgt_pen[b, : len(pens)] = torch.tensor(pens, dtype=torch.float32)

        if s_tup and t_tup:
            for i_pos, s_sym in enumerate(s_tup):
                for j_pos, t_sym in enumerate(t_tup):
                    if s_sym == t_sym:
                        copy_mask[b, i_pos, j_pos] = True

        c_src[b] = cell_vocab.encode(cs)
        c_tgt[b] = cell_vocab.encode(ct)

    batch = EncodedBatch(
        src_rnn_ids=src_rnn,
        src_lens=torch.tensor(m_lens, dtype=torch.long),
        tgt_prefix_rnn_ids=tgt_pref,
        tgt_emit_ids=tgt_emit,
        tgt_unk_penalties=tgt_pen,
        tgt_lens=torch.tensor(n_lens, dtype=torch.long),
        copy_match_mask=copy_mask,
        src_cell_ids=c_src,
        tgt_cell_ids=c_tgt,
        m_max=m_max,
        n_max=n_max,
    )
    return batch.to(device) if device is not None else batch


class HardMonotonicTransducer(nn.Module):
    """Exact Hard-Monotonic Neural Character Transducer over Sigma*."""

    def __init__(
        self,
        char_vocab: CharVocab,
        cell_vocab: CellVocab,
        config: TransducerConfig | None = None,
    ) -> None:
        super().__init__()
        self.char_vocab = char_vocab
        self.cell_vocab = cell_vocab
        self.config = config or TransducerConfig()

        cfg = self.config
        v_emit = char_vocab.num_emit
        self.num_actions = 3 + 2 * v_emit  # [STOP, COPY, DELETE, INSERT(V), SUBSTITUTE(V)]

        self.char_emb = nn.Embedding(
            char_vocab.num_rnn_tokens, cfg.char_emb_dim, padding_idx=char_vocab.PAD_ID
        )
        self.src_cell_emb = nn.Embedding(len(cell_vocab), cfg.cell_emb_dim)
        self.tgt_cell_emb = nn.Embedding(len(cell_vocab), cfg.cell_emb_dim)

        # Separate forward & involution-reversed backward LSTMs avoid pack_padded_sequence
        # backend quirks on Apple MPS while guaranteeing exact batch-padding invariance.
        self.enc_fwd = nn.LSTM(cfg.char_emb_dim, cfg.enc_hidden_dim, batch_first=True)
        self.enc_bwd = nn.LSTM(cfg.char_emb_dim, cfg.enc_hidden_dim, batch_first=True)

        self.dec_cell_proj = nn.Linear(2 * cfg.cell_emb_dim, cfg.char_emb_dim)
        self.dec_lstm = nn.LSTM(2 * cfg.char_emb_dim, cfg.dec_hidden_dim, batch_first=True)

        self.cell_ctx_mlp = nn.Sequential(
            nn.Linear(2 * cfg.cell_emb_dim, cfg.joint_hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.joint_hidden_dim, cfg.joint_hidden_dim),
        )
        self.enc_proj = nn.Linear(2 * cfg.enc_hidden_dim, cfg.joint_hidden_dim, bias=False)
        self.dec_proj = nn.Linear(cfg.dec_hidden_dim, cfg.joint_hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.out_proj = nn.Linear(cfg.joint_hidden_dim, self.num_actions)
        with torch.no_grad():
            self.out_proj.bias[0] = float(cfg.init_stop_bias)
        self.copy_bias = nn.Parameter(torch.tensor(float(cfg.init_copy_bias)))
        self._mps_fallback_triggered: bool = False

    def encode_source(self, src_rnn_ids: torch.Tensor, src_lens: torch.Tensor) -> torch.Tensor:
        """Encode source sequences [x_1, ..., x_M, <eos>] with exact padding invariance.

        Returns tensor of shape (B, M_max + 1, 2 * enc_hidden_dim).
        """
        emb = self.dropout(self.char_emb(src_rnn_ids))
        h_fwd, _ = self.enc_fwd(emb)

        # Involution index rev_idx[b, k] = (M_b - k) for k <= M_b, else k.
        # Reversing only the valid prefix 0..M_b ensures enc_bwd never reads <pad>.
        b_size, seq_len, _ = emb.shape
        pos = torch.arange(seq_len, device=emb.device).unsqueeze(0).expand(b_size, seq_len)
        m_exp = src_lens.unsqueeze(1)
        rev_idx = torch.where(pos <= m_exp, m_exp - pos, pos)
        rev_idx_3d = rev_idx.unsqueeze(-1).expand(-1, -1, emb.shape[-1])
        emb_rev = torch.gather(emb, 1, rev_idx_3d)

        h_bwd_rev, _ = self.enc_bwd(emb_rev)
        rev_idx_hid = rev_idx.unsqueeze(-1).expand(-1, -1, h_bwd_rev.shape[-1])
        h_bwd = torch.gather(h_bwd_rev, 1, rev_idx_hid)
        return torch.cat([h_fwd, h_bwd], dim=-1)

    def compute_cell_representations(
        self, src_cell_ids: torch.Tensor, tgt_cell_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute decoder input cell context and joint lattice cell context."""
        cs = self.src_cell_emb(src_cell_ids)
        ct = self.tgt_cell_emb(tgt_cell_ids)
        cat_c = torch.cat([cs, ct], dim=-1)
        dec_c = torch.tanh(self.dec_cell_proj(cat_c))
        joint_c = self.cell_ctx_mlp(cat_c)
        return dec_c, joint_c

    def encode_target_prefix(
        self, tgt_prefix_rnn_ids: torch.Tensor, dec_cell_ctx: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-force target prefix [<bos>, y_1, ..., y_N] in parallel.

        Because LSTM is strictly causal left-to-right, h_dec[b, j] depends ONLY on
        (<bos>, y_1, ..., y_j) and is 100% invariant to right-padding beyond N_b.
        """
        emb = self.dropout(self.char_emb(tgt_prefix_rnn_ids))
        cell_rep = dec_cell_ctx.unsqueeze(1).expand(-1, emb.shape[1], -1)
        dec_in = torch.cat([emb, cell_rep], dim=-1)
        h_dec, _ = self.dec_lstm(dec_in)
        return h_dec

    def compute_lattice_log_probs(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        joint_cell_ctx: torch.Tensor,
        src_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalized action log-probabilities across the entire (B, M+1, N+1) lattice."""
        u_enc = self.enc_proj(self.dropout(enc_out))      # (B, M+1, H)
        u_dec = self.dec_proj(self.dropout(dec_out))      # (B, N+1, H)
        z = torch.tanh(
            u_enc.unsqueeze(2) + u_dec.unsqueeze(1) + joint_cell_ctx.unsqueeze(1).unsqueeze(2)
        )
        logits = self.out_proj(self.dropout(z))           # (B, M+1, N+1, 3 + 2*V)

        # Add dedicated scalar bias to COPY action (index 1) without in-place mutation
        bias_vec = torch.zeros(self.num_actions, dtype=logits.dtype, device=logits.device)
        bias_vec[1] = self.copy_bias
        logits = logits + bias_vec

        b_size, m_plus_1, _, _ = logits.shape
        v_emit = self.char_vocab.num_emit
        src_pos = torch.arange(m_plus_1, device=logits.device).unsqueeze(0)
        is_eos = src_pos >= src_lens.unsqueeze(1)         # True at i == M_b (and padded i > M_b)

        invalid = torch.zeros(
            (b_size, m_plus_1, 1, self.num_actions), dtype=torch.bool, device=logits.device
        )
        invalid[:, :, 0, 0] = ~is_eos                     # STOP valid iff i == M_b
        invalid[:, :, 0, 1] = is_eos                      # COPY valid iff i < M_b
        invalid[:, :, 0, 2] = is_eos                      # DELETE valid iff i < M_b
        invalid[:, :, 0, 3 + v_emit :] = is_eos.unsqueeze(-1)  # SUBSTITUTE valid iff i < M_b

        masked_logits = logits.masked_fill(invalid, NEG_INF_SAFE)
        return F.log_softmax(masked_logits, dim=-1)

    def _forward_wavefront_dp(self, batch: EncodedBatch) -> torch.Tensor:
        """Exact anti-diagonal wavefront marginalization returning log q_theta(y | x, c_s, c_t) in nats."""
        enc_out = self.encode_source(batch.src_rnn_ids, batch.src_lens)
        dec_c, joint_c = self.compute_cell_representations(batch.src_cell_ids, batch.tgt_cell_ids)
        dec_out = self.encode_target_prefix(batch.tgt_prefix_rnn_ids, dec_c)
        log_probs = self.compute_lattice_log_probs(enc_out, dec_out, joint_c, batch.src_lens)

        b_size = batch.src_rnn_ids.shape[0]
        m_max = batch.m_max
        n_max = batch.n_max
        v_emit = self.char_vocab.num_emit
        device = log_probs.device
        dtype = log_probs.dtype

        stop_lp = log_probs[:, :, :, 0]                   # (B, M+1, N+1)

        if m_max > 0:
            del_lp = log_probs[:, :m_max, :, 2]           # (B, M, N+1)
        else:
            del_lp = torch.empty((b_size, 0, n_max + 1), dtype=dtype, device=device)

        if n_max > 0:
            gather_idx = (
                batch.tgt_emit_ids[:, :n_max]
                .unsqueeze(1)
                .expand(b_size, m_max + 1, n_max)
                .unsqueeze(-1)
            )
            unk_pen = batch.tgt_unk_penalties[:, :n_max].unsqueeze(1)  # (B, 1, N)
            ins_all = log_probs[:, :, :n_max, 3 : 3 + v_emit]
            ins_lp = torch.gather(ins_all, -1, gather_idx).squeeze(-1) + unk_pen  # (B, M+1, N)
        else:
            ins_lp = torch.empty((b_size, m_max + 1, 0), dtype=dtype, device=device)

        if m_max > 0 and n_max > 0:
            copy_lp = log_probs[:, :m_max, :n_max, 1]     # (B, M, N)
            sub_all = log_probs[:, :m_max, :n_max, 3 + v_emit : 3 + 2 * v_emit]
            sub_lp = (
                torch.gather(sub_all, -1, gather_idx[:, :m_max, :, :]).squeeze(-1)
                + batch.tgt_unk_penalties[:, :n_max].unsqueeze(1)
            )
            match_mask = batch.copy_match_mask[:, :m_max, :n_max]
            diag_lp = torch.where(match_mask, torch.logaddexp(copy_lp, sub_lp), sub_lp)
        else:
            diag_lp = torch.empty((b_size, m_max, n_max), dtype=dtype, device=device)

        # Fixed-width (B, M_max + 1) zero-sync anti-diagonal wavefront DP (d = i + j = 0 .. M_max + N_max).
        # Zero `.item()` / `.tolist()` device syncs and single O(1) terminal gather for Apple MPS speed.
        b_idx = torch.arange(b_size, device=device)
        end_stop = stop_lp[b_idx, batch.src_lens, batch.tgt_lens]
        d_tot = m_max + n_max
        if d_tot == 0:
            return end_stop

        d_vals = torch.arange(1, d_tot + 1, device=device).unsqueeze(1)  # (D_tot, 1)
        m_3d = batch.src_lens.view(b_size, 1, 1)

        if n_max >= 1:
            i_full = torch.arange(m_max + 1, device=device).unsqueeze(0)  # (1, M_max+1)
            j_ins = d_vals - i_full                                       # (D_tot, M_max+1)
            valid_ins = ((j_ins >= 1) & (j_ins <= n_max)).unsqueeze(0) & (i_full.unsqueeze(0) <= m_3d)
            j_ins_safe = (j_ins - 1).clamp(0, n_max - 1)
            raw_ins = ins_lp[:, i_full.expand(d_tot, m_max + 1), j_ins_safe]
            ins_by_d = torch.where(valid_ins, raw_ins, torch.full_like(raw_ins, NEG_INF_SAFE))
        else:
            ins_by_d = torch.full((b_size, d_tot, m_max + 1), NEG_INF_SAFE, dtype=dtype, device=device)

        if m_max >= 1:
            i_pos = torch.arange(1, m_max + 1, device=device).unsqueeze(0)  # (1, M_max)
            j_del = d_vals - i_pos                                          # (D_tot, M_max)
            i_prev = (i_pos - 1).expand(d_tot, m_max)
            in_m = i_pos.unsqueeze(0) <= m_3d

            valid_del = ((j_del >= 0) & (j_del <= n_max)).unsqueeze(0) & in_m
            raw_del = del_lp[:, i_prev, j_del.clamp(0, n_max)]
            del_by_d = torch.where(valid_del, raw_del, torch.full_like(raw_del, NEG_INF_SAFE))

            if n_max >= 1:
                valid_diag = ((j_del >= 1) & (j_del <= n_max)).unsqueeze(0) & in_m
                raw_diag = diag_lp[:, i_prev, (j_del - 1).clamp(0, n_max - 1)]
                diag_by_d = torch.where(valid_diag, raw_diag, torch.full_like(raw_diag, NEG_INF_SAFE))
            else:
                diag_by_d = torch.empty((b_size, d_tot, 0), dtype=dtype, device=device)
        else:
            del_by_d = torch.empty((b_size, d_tot, 0), dtype=dtype, device=device)
            diag_by_d = torch.empty((b_size, d_tot, 0), dtype=dtype, device=device)

        neg_col = torch.full((b_size, 1), NEG_INF_SAFE, dtype=dtype, device=device)
        w0 = (
            torch.cat(
                [
                    torch.zeros((b_size, 1), dtype=dtype, device=device),
                    torch.full((b_size, m_max), NEG_INF_SAFE, dtype=dtype, device=device),
                ],
                dim=1,
            )
            if m_max >= 1
            else torch.zeros((b_size, 1), dtype=dtype, device=device)
        )
        waves: list[torch.Tensor] = [w0]

        for d in range(1, d_tot + 1):
            w_prev1 = waves[d - 1]
            c_ins = w_prev1 + ins_by_d[:, d - 1, :]
            if m_max >= 1:
                c_del = torch.cat([neg_col, w_prev1[:, :-1] + del_by_d[:, d - 1, :]], dim=1)
                if d >= 2 and n_max >= 1:
                    w_prev2 = waves[d - 2]
                    c_diag = torch.cat([neg_col, w_prev2[:, :-1] + diag_by_d[:, d - 1, :]], dim=1)
                    w_cur = torch.logsumexp(torch.stack((c_del, c_ins, c_diag), dim=0), dim=0)
                else:
                    w_cur = torch.logaddexp(c_del, c_ins)
            else:
                w_cur = c_ins
            waves.append(w_cur)

        all_waves = torch.stack(waves, dim=1)  # (B, D_tot + 1, M_max + 1)
        end_alpha = all_waves[b_idx, batch.src_lens + batch.tgt_lens, batch.src_lens]
        return end_alpha + end_stop

    def forward(self, batch: EncodedBatch) -> torch.Tensor:
        """Compute exact log q_theta(y | x, c_s, c_t) in natural log units (shape (B,)).

        Includes automatic fallback from MPS to CPU if an unexpected backend kernel issue arises,
        without forcing per-batch MPS command-buffer syncs during training.
        """
        dev = next(self.parameters()).device
        if dev.type == "mps":
            try:
                return self._forward_wavefront_dp(batch)
            except (NotImplementedError, RuntimeError) as exc:
                if not self._mps_fallback_triggered:
                    logger.warning("MPS fallback to CPU triggered in HardMonotonicTransducer: %s", exc)
                    self._mps_fallback_triggered = True
                self.to(torch.device("cpu"))
                return self._forward_wavefront_dp(batch.to(torch.device("cpu")))
        return self._forward_wavefront_dp(batch)

    @torch.no_grad()
    def log_prob_pairs(
        self,
        examples: Sequence[tuple[str | Sequence[str], str | Sequence[str], str, str]],
        batch_size: int = 256,
    ) -> list[float]:
        """Evaluate exact natural log-probabilities log q_theta(y | x, c_s, c_t) with dropout off.

        Sorts examples by sequence length internally before batching to minimize padding overhead,
        restoring exact original input order upon return.
        """
        if not examples:
            return []
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device

        indexed_order = sorted(
            range(len(examples)),
            key=lambda idx: (len(examples[idx][0]) + len(examples[idx][1]), len(examples[idx][0])),
        )
        out = [0.0] * len(examples)
        for start in range(0, len(indexed_order), batch_size):
            idx_chunk = indexed_order[start : start + batch_size]
            chunk = [examples[i] for i in idx_chunk]
            batch = collate_transducer_batch(chunk, self.char_vocab, self.cell_vocab, device=dev)
            lp = self.forward(batch)
            lp_cpu = lp.detach().cpu()
            if dev.type == "mps" and torch.isnan(lp_cpu).any():
                if not self._mps_fallback_triggered:
                    logger.warning("NaN detected on MPS eval pass; falling back to CPU.")
                    self._mps_fallback_triggered = True
                self.to(torch.device("cpu"))
                dev = torch.device("cpu")
                lp_cpu = self.forward(batch.to(dev)).detach().cpu()
            for orig_i, val in zip(idx_chunk, lp_cpu.tolist()):
                out[orig_i] = val

        if was_training:
            self.train()
        return out


@torch.no_grad()
def ensemble_log_prob_pairs(
    models: Sequence[HardMonotonicTransducer],
    examples: Sequence[tuple[str | Sequence[str], str | Sequence[str], str, str]],
    batch_size: int = 256,
) -> list[float]:
    """Exact probability-space ensemble log q_ens(y | x) = logsumexp_s log q_s(y | x) - log(S)."""
    if not models:
        raise ValueError("models must be non-empty.")
    if not examples:
        return []
    per_seed = [
        torch.tensor(m.log_prob_pairs(examples, batch_size=batch_size), dtype=torch.float64)
        for m in models
    ]
    stacked = torch.stack(per_seed, dim=0)  # (S, N_ex)
    ens_lp = torch.logsumexp(stacked, dim=0) - math.log(len(models))
    return ens_lp.tolist()


@torch.no_grad()
def ensemble_beam_decode(
    models: Sequence[HardMonotonicTransducer],
    src_seq: str | Sequence[str],
    src_cell: str,
    tgt_cell: str,
    *,
    beam_size: int = 4,
    max_len: int | None = None,
) -> list[tuple[tuple[str, ...], float]]:
    """Exact string-marginal beam decoder over a K-seed ensemble.

    Every beam hypothesis is a distinct target string y_{1:j} carrying, for each seed m,
    its exact forward alignment vector alpha^{(m)} in R^{M+1} and LSTM state (h^{(m)}, c^{(m)}).
    Closing over DELETE steps along i = 0..M yields the EXACT marginal log-probability
    log q_ens(y_{1:j} | x, c_s, c_t) with zero alignment-fragmentation error.

    Returns up to `beam_size` completed (target_tuple, log_prob_nats) pairs sorted by
    descending exact ensemble log-probability.
    """
    if not models:
        raise ValueError("models must be non-empty.")

    s_tup = to_symbol_tuple(src_seq)
    m_len = len(s_tup)
    max_steps = max_len if max_len is not None else max(m_len + 12, int(1.5 * m_len) + 6)
    char_vocab = models[0].char_vocab
    cell_vocab = models[0].cell_vocab
    v_emit = char_vocab.num_emit
    log_s = math.log(len(models))

    # Build local beam symbol inventory: all known symbols PLUS any OOV source symbols
    # so OOV stem characters (e.g. 'š' in 'špacero' -> 'špacera') are emitted with exact
    # COPY + (<unk> + log_unk_penalty) marginal probabilities at every decoding step.
    emit_offset = 1 if char_vocab.allow_unk else 0
    known_syms = list(char_vocab.known_symbols)
    oov_src_syms = (
        sorted({sym for sym in s_tup if sym not in char_vocab.sym_to_emit})
        if char_vocab.allow_unk
        else []
    )
    n_oov = len(oov_src_syms)
    cand_syms = known_syms + oov_src_syms
    cand_rnn_ids = [char_vocab.sym_to_rnn[s] for s in known_syms] + (
        [int(char_vocab.unk_rnn_id)] * n_oov if char_vocab.unk_rnn_id is not None else []
    )
    cand_sym_to_idx = {s: idx for idx, s in enumerate(cand_syms)}
    src_match_cand_idx = [cand_sym_to_idx.get(sym, -1) for sym in s_tup]

    # Pre-encode source & cell representations per seed model
    seed_states = []
    for mdl in models:
        mdl.eval()
        dev = next(mdl.parameters()).device
        s_ids = torch.tensor([char_vocab.encode_source_rnn(s_tup)], dtype=torch.long, device=dev)
        s_len_t = torch.tensor([m_len], dtype=torch.long, device=dev)
        c_s_t = torch.tensor([cell_vocab.encode(src_cell)], dtype=torch.long, device=dev)
        c_t_t = torch.tensor([cell_vocab.encode(tgt_cell)], dtype=torch.long, device=dev)

        enc_out = mdl.encode_source(s_ids, s_len_t)[0]             # (M+1, 2*D_enc)
        u_enc = mdl.enc_proj(enc_out)                              # (M+1, H)
        dec_c, joint_c = mdl.compute_cell_representations(c_s_t, c_t_t)
        dec_c_1 = dec_c[0]                                         # (D_emb,)
        joint_c_1 = joint_c[0]                                     # (H,)

        # Initial decoder step on <bos>
        bos_emb = mdl.char_emb(torch.tensor([[char_vocab.BOS_ID]], dtype=torch.long, device=dev))
        dec_in = torch.cat([bos_emb, dec_c_1.view(1, 1, -1)], dim=-1)
        h_out, (h_st, c_st) = mdl.dec_lstm(dec_in)

        alpha0 = torch.full((m_len + 1,), NEG_INF_SAFE, dtype=torch.float32, device=dev)
        alpha0[0] = 0.0
        seed_states.append({
            "mdl": mdl,
            "dev": dev,
            "u_enc": u_enc,
            "dec_c": dec_c_1,
            "joint_c": joint_c_1,
            "init_alpha": alpha0,
            "init_dec_out": h_out[0, 0],
            "init_hc": (h_st, c_st),
        })

    # Each active beam entry: (prefix_tuple, list_per_seed_of_(alpha_vec, dec_out_vec, (h, c)), prefix_score)
    active = [
        (
            (),
            [(st["init_alpha"], st["init_dec_out"], st["init_hc"]) for st in seed_states],
            0.0,
        )
    ]
    completed: dict[tuple[str, ...], float] = {}

    for step in range(max_steps + 1):
        next_candidates: list[tuple[float, int, int, list[torch.Tensor]]] = []
        closed_per_hyp: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

        for hyp_idx, (prefix, per_seed, _) in enumerate(active):
            stop_lps_seed: list[float] = []
            hyp_seed_trans: list[tuple[torch.Tensor, torch.Tensor]] = []

            for s_idx, st in enumerate(seed_states):
                mdl = st["mdl"]
                dev = st["dev"]
                alpha_vec, dec_out_vec, _ = per_seed[s_idx]

                # Action log-probs across i = 0..M for this single prefix state: shape (M+1, 3 + 2*V)
                u_dec = mdl.dec_proj(dec_out_vec)                  # (H,)
                z = torch.tanh(st["u_enc"] + u_dec.unsqueeze(0) + st["joint_c"].unsqueeze(0))
                logits = mdl.out_proj(z)
                logits[:, 1] = logits[:, 1] + mdl.copy_bias
                if m_len > 0:
                    logits[:m_len, 0] = NEG_INF_SAFE               # STOP invalid for i < M
                logits[m_len, 1] = NEG_INF_SAFE                    # COPY invalid at i == M
                logits[m_len, 2] = NEG_INF_SAFE                    # DELETE invalid at i == M
                logits[m_len, 3 + v_emit :] = NEG_INF_SAFE         # SUB invalid at i == M
                lp = F.log_softmax(logits, dim=-1)                 # (M+1, 3 + 2*V)

                # 1. Exact DELETE closure along i = 0 -> 1 -> ... -> M
                alpha_closed = alpha_vec.clone()
                for i_pos in range(1, m_len + 1):
                    alpha_closed[i_pos] = torch.logaddexp(
                        alpha_closed[i_pos], alpha_closed[i_pos - 1] + lp[i_pos - 1, 2]
                    )

                stop_lps_seed.append(float((alpha_closed[m_len] + lp[m_len, 0]).item()))
                hyp_seed_trans.append((alpha_closed, lp))

            # Exact ensemble completion log-probability for `prefix`
            comp_lp = float(torch.logsumexp(torch.tensor(stop_lps_seed, dtype=torch.float64), dim=0).item()) - log_s
            completed[prefix] = comp_lp
            closed_per_hyp.append(hyp_seed_trans)

            if step == max_steps or not cand_syms:
                continue

            # 2. Score all candidate emission symbols (known_syms + oov_src_syms)
            next_alphas_per_seed: list[torch.Tensor] = []
            marginal_per_seed: list[torch.Tensor] = []

            for s_idx, st in enumerate(seed_states):
                alpha_closed, lp = hyp_seed_trans[s_idx]
                ins_known = lp[:, 3 + emit_offset : 3 + v_emit]    # (M+1, n_known)
                if n_oov > 0:
                    ins_oov = (lp[:, 3:4] + char_vocab.log_unk_penalty).expand(m_len + 1, n_oov)
                    ins_cand = torch.cat([ins_known, ins_oov], dim=1)
                else:
                    ins_cand = ins_known
                next_alpha = alpha_closed.unsqueeze(1) + ins_cand  # (M+1, n_cand) via INSERT

                if m_len > 0:
                    sub_known = lp[:m_len, 3 + v_emit + emit_offset : 3 + 2 * v_emit]
                    if n_oov > 0:
                        sub_oov = (
                            lp[:m_len, 3 + v_emit : 3 + v_emit + 1] + char_vocab.log_unk_penalty
                        ).expand(m_len, n_oov)
                        sub_cand = torch.cat([sub_known, sub_oov], dim=1).clone()
                    else:
                        sub_cand = sub_known.clone()
                    copy_lp = lp[:m_len, 1]                        # (M,)
                    for i_pos, match_cidx in enumerate(src_match_cand_idx):
                        if match_cidx >= 0:
                            sub_cand[i_pos, match_cidx] = torch.logaddexp(
                                sub_cand[i_pos, match_cidx], copy_lp[i_pos]
                            )
                    from_diag = alpha_closed[:m_len].unsqueeze(1) + sub_cand  # (M, n_cand)
                    next_alpha[1:, :] = torch.logaddexp(next_alpha[1:, :], from_diag)

                next_alphas_per_seed.append(next_alpha)
                marginal_per_seed.append(torch.logsumexp(next_alpha, dim=0).cpu().to(torch.float64))

            ens_prefix_scores = torch.logsumexp(torch.stack(marginal_per_seed, dim=0), dim=0) - log_s
            k_top = min(beam_size, ens_prefix_scores.numel())
            top_vals, top_cidxs = torch.topk(ens_prefix_scores, k=k_top)

            for score_val, cidx_raw in zip(top_vals.tolist(), top_cidxs.tolist()):
                cidx = int(cidx_raw)
                per_s_alpha = [na[:, cidx] for na in next_alphas_per_seed]
                next_candidates.append((score_val, hyp_idx, cidx, per_s_alpha))

        if step == max_steps or not next_candidates:
            break

        next_candidates.sort(key=lambda t: t[0], reverse=True)
        selected = next_candidates[:beam_size]

        # Early exit if best possible active prefix is already worse than the K-th best completed string
        if len(completed) >= beam_size:
            kth_best_done = sorted(completed.values(), reverse=True)[beam_size - 1]
            if selected[0][0] <= kth_best_done - 1e-4:
                break

        new_active = []
        for score_val, hyp_idx, cidx, per_s_alpha in selected:
            old_prefix, old_per_seed, _ = active[hyp_idx]
            sym = cand_syms[cidx]
            rnn_id = cand_rnn_ids[cidx]
            new_per_seed = []
            for s_idx, st in enumerate(seed_states):
                mdl = st["mdl"]
                dev = st["dev"]
                _, _, old_hc = old_per_seed[s_idx]
                tok_emb = mdl.char_emb(torch.tensor([[rnn_id]], dtype=torch.long, device=dev))
                dec_in = torch.cat([tok_emb, st["dec_c"].view(1, 1, -1)], dim=-1)
                h_out, new_hc = mdl.dec_lstm(dec_in, old_hc)
                new_per_seed.append((per_s_alpha[s_idx], h_out[0, 0], new_hc))
            new_active.append((old_prefix + (sym,), new_per_seed, score_val))

        active = new_active

    ranked = sorted(completed.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:beam_size]


@torch.no_grad()
def ensemble_beam_decode_batch(
    models: Sequence[HardMonotonicTransducer],
    queries: Sequence[tuple[str | Sequence[str], str, str]],
    *,
    beam_size: int = 4,
    max_len: int | None = None,
    batch_size: int = 128,
) -> list[list[tuple[tuple[str, ...], float]]]:
    """Batched exact string-marginal beam decoder over a K-seed ensemble.

    Vectorizes `ensemble_beam_decode` across `B` `(src_seq, src_cell, tgt_cell)` queries
    in parallel with exact batch-padding invariance on `cuda`, `mps`, and `cpu`.
    Any query containing unseen OOV source symbols (or when `known_symbols` is empty)
    is routed transparently to `ensemble_beam_decode` so OOV `COPY` semantics remain exact.
    """
    if not models:
        raise ValueError("models must be non-empty.")
    if not queries:
        return []

    char_vocab = models[0].char_vocab
    cell_vocab = models[0].cell_vocab
    known_syms = list(char_vocab.known_symbols)
    c_cand = len(known_syms)

    results: list[list[tuple[tuple[str, ...], float]] | None] = [None] * len(queries)
    in_vocab_indices: list[int] = []
    for idx, (src_seq, cs, ct) in enumerate(queries):
        s_tup = to_symbol_tuple(src_seq)
        if c_cand == 0 or any(sym not in char_vocab.sym_to_emit for sym in s_tup):
            results[idx] = ensemble_beam_decode(
                models, s_tup, cs, ct, beam_size=beam_size, max_len=max_len
            )
        else:
            in_vocab_indices.append(idx)

    if not in_vocab_indices:
        return [r if r is not None else [] for r in results]

    v_emit = char_vocab.num_emit
    emit_offset = 1 if char_vocab.allow_unk else 0
    log_s = math.log(len(models))
    sym_to_cidx = {s: i for i, s in enumerate(known_syms)}
    cand_rnn_ids_list = [char_vocab.sym_to_rnn[s] for s in known_syms]

    for start in range(0, len(in_vocab_indices), batch_size):
        chunk_q_idxs = in_vocab_indices[start : start + batch_size]
        b_size = len(chunk_q_idxs)
        s_tups = [to_symbol_tuple(queries[q_i][0]) for q_i in chunk_q_idxs]
        cs_list = [queries[q_i][1] for q_i in chunk_q_idxs]
        ct_list = [queries[q_i][2] for q_i in chunk_q_idxs]

        m_lens_list = [len(s) for s in s_tups]
        m_max = max(m_lens_list)
        m_alloc = max(m_max, 1)
        max_steps_per_b = [
            max_len if max_len is not None else max(m + 12, int(1.5 * m) + 6)
            for m in m_lens_list
        ]
        max_steps_chunk = max(max_steps_per_b)

        src_rnn_cpu = torch.full((b_size, m_max + 1), char_vocab.PAD_ID, dtype=torch.long)
        copy_match_cpu = torch.zeros((b_size, m_alloc, c_cand), dtype=torch.bool)
        for b_i, s_tup in enumerate(s_tups):
            s_ids = char_vocab.encode_source_rnn(s_tup)
            src_rnn_cpu[b_i, : len(s_ids)] = torch.tensor(s_ids, dtype=torch.long)
            for pos_i, sym in enumerate(s_tup):
                copy_match_cpu[b_i, pos_i, sym_to_cidx[sym]] = True

        s_lens_cpu = torch.tensor(m_lens_list, dtype=torch.long)
        cs_cpu = torch.tensor([cell_vocab.encode(c) for c in cs_list], dtype=torch.long)
        ct_cpu = torch.tensor([cell_vocab.encode(c) for c in ct_list], dtype=torch.long)
        cand_rnn_cpu = torch.tensor(cand_rnn_ids_list, dtype=torch.long)

        seed_states = []
        for mdl in models:
            mdl.eval()
            dev = next(mdl.parameters()).device
            s_rnn = src_rnn_cpu.to(dev)
            s_lens = s_lens_cpu.to(dev)
            cs_t = cs_cpu.to(dev)
            ct_t = ct_cpu.to(dev)

            enc_out = mdl.encode_source(s_rnn, s_lens)           # (B, M_max+1, 2*D_enc)
            u_enc = mdl.enc_proj(enc_out)                        # (B, M_max+1, H)
            dec_c, joint_c = mdl.compute_cell_representations(cs_t, ct_t)

            bos_ids = torch.full((b_size, 1), char_vocab.BOS_ID, dtype=torch.long, device=dev)
            bos_emb = mdl.char_emb(bos_ids)
            dec_in0 = torch.cat([bos_emb, dec_c.unsqueeze(1)], dim=-1)
            h_out0, (h0, c0) = mdl.dec_lstm(dec_in0)

            alpha0 = torch.full(
                (b_size, 1, m_max + 1), NEG_INF_SAFE, dtype=torch.float32, device=dev
            )
            alpha0[:, 0, 0] = 0.0

            bias_vec = torch.zeros(mdl.num_actions, dtype=torch.float32, device=dev)
            bias_vec[1] = mdl.copy_bias

            pos_grid = torch.arange(m_max + 1, device=dev).view(1, 1, m_max + 1)
            m_3d = s_lens.view(b_size, 1, 1)
            not_at_m = pos_grid != m_3d
            ge_m = pos_grid >= m_3d
            invalid_mask = torch.zeros(
                (b_size, 1, m_max + 1, mdl.num_actions), dtype=torch.bool, device=dev
            )
            invalid_mask[:, :, :, 0] = not_at_m
            invalid_mask[:, :, :, 1] = ge_m
            invalid_mask[:, :, :, 2] = ge_m
            invalid_mask[:, :, :, 3 + v_emit :] = ge_m.unsqueeze(-1)

            valid_pos_mask = (pos_grid <= m_3d).unsqueeze(-1)    # (B, 1, M_max+1, 1)
            valid_diag_mask = (
                torch.arange(m_alloc, device=dev).view(1, 1, m_alloc, 1)
                < s_lens.view(b_size, 1, 1, 1)
            )

            seed_states.append({
                "mdl": mdl,
                "dev": dev,
                "s_lens": s_lens,
                "u_enc": u_enc,
                "dec_c": dec_c,
                "joint_c": joint_c,
                "bias_vec": bias_vec,
                "invalid_mask": invalid_mask,
                "valid_pos_mask": valid_pos_mask,
                "valid_diag_mask": valid_diag_mask,
                "copy_match": copy_match_cpu.to(dev),
                "cand_rnn": cand_rnn_cpu.to(dev),
                "alpha": alpha0,                                 # (B, K_cur, M_max+1)
                "dec_out": h_out0,                               # (B, K_cur, H_dec)
                "hc": (h0, c0),                                  # (1, B*K_cur, H_dec)
            })

        prefixes: list[list[tuple[str, ...]]] = [[()] for _ in range(b_size)]
        completed_per_b: list[dict[tuple[str, ...], float]] = [{} for _ in range(b_size)]
        done_b = [False] * b_size

        for step in range(max_steps_chunk + 1):
            k_cur = len(prefixes[0])
            stop_per_seed: list[torch.Tensor] = []
            closed_and_lp: list[tuple[torch.Tensor, torch.Tensor]] = []

            for st in seed_states:
                mdl = st["mdl"]
                s_lens = st["s_lens"]
                alpha = st["alpha"]
                u_dec = mdl.dec_proj(st["dec_out"])              # (B, K_cur, H)
                z = torch.tanh(
                    st["u_enc"].unsqueeze(1)
                    + u_dec.unsqueeze(2)
                    + st["joint_c"].view(b_size, 1, 1, -1)
                )
                logits = mdl.out_proj(z) + st["bias_vec"]
                masked_logits = logits.masked_fill(st["invalid_mask"], NEG_INF_SAFE)
                lp = F.log_softmax(masked_logits, dim=-1)        # (B, K_cur, M_max+1, A)

                cols = [alpha[:, :, 0]]
                for i_pos in range(1, m_max + 1):
                    valid_i = (i_pos <= s_lens).unsqueeze(1)
                    from_del = torch.where(
                        valid_i,
                        cols[-1] + lp[:, :, i_pos - 1, 2],
                        torch.full_like(cols[-1], NEG_INF_SAFE),
                    )
                    cols.append(torch.logaddexp(alpha[:, :, i_pos], from_del))
                alpha_closed = torch.stack(cols, dim=2)          # (B, K_cur, M_max+1)

                m_idx = s_lens.view(b_size, 1, 1).expand(b_size, k_cur, 1)
                stop_alpha = torch.gather(alpha_closed, 2, m_idx).squeeze(2)
                stop_act = torch.gather(lp[:, :, :, 0], 2, m_idx).squeeze(2)
                stop_per_seed.append((stop_alpha + stop_act).cpu().to(torch.float64))
                closed_and_lp.append((alpha_closed, lp))

            comp_lp_mat = (
                torch.logsumexp(torch.stack(stop_per_seed, dim=0), dim=0) - log_s
            ).tolist()

            for b_i in range(b_size):
                if not done_b[b_i] and step <= max_steps_per_b[b_i]:
                    comp_dict = completed_per_b[b_i]
                    row_lps = comp_lp_mat[b_i]
                    for k_i, pref in enumerate(prefixes[b_i]):
                        comp_dict[pref] = row_lps[k_i]
                    if step == max_steps_per_b[b_i]:
                        done_b[b_i] = True

            if all(done_b) or step == max_steps_chunk:
                break

            next_alphas_per_seed: list[torch.Tensor] = []
            marg_per_seed: list[torch.Tensor] = []

            for s_idx, st in enumerate(seed_states):
                alpha_closed, lp = closed_and_lp[s_idx]
                ins_known = lp[:, :, :, 3 + emit_offset : 3 + v_emit]  # (B, K_cur, M_max+1, C)
                next_alpha = torch.where(
                    st["valid_pos_mask"],
                    alpha_closed.unsqueeze(-1) + ins_known,
                    torch.full_like(ins_known, NEG_INF_SAFE),
                )
                if m_max > 0:
                    sub_known = lp[
                        :, :, :m_max, 3 + v_emit + emit_offset : 3 + 2 * v_emit
                    ]  # (B, K_cur, M_max, C)
                    copy_lp_exp = lp[:, :, :m_max, 1].unsqueeze(-1).expand_as(sub_known)
                    cm = st["copy_match"][:, :m_max, :].unsqueeze(1)
                    sub_cand = torch.where(
                        cm, torch.logaddexp(sub_known, copy_lp_exp), sub_known
                    )
                    from_diag = torch.where(
                        st["valid_diag_mask"][:, :, :m_max, :],
                        alpha_closed[:, :, :m_max].unsqueeze(-1) + sub_cand,
                        torch.full_like(sub_cand, NEG_INF_SAFE),
                    )
                    next_alpha = torch.cat(
                        [
                            next_alpha[:, :, :1, :],
                            torch.logaddexp(next_alpha[:, :, 1:, :], from_diag),
                        ],
                        dim=2,
                    )
                next_alphas_per_seed.append(next_alpha)
                marg_per_seed.append(torch.logsumexp(next_alpha, dim=2).cpu().to(torch.float64))

            ens_scores = (
                torch.logsumexp(torch.stack(marg_per_seed, dim=0), dim=0) - log_s
            )  # (B, K_cur, C)
            k_next = min(beam_size, k_cur * c_cand)
            top_vals, top_flat = torch.topk(ens_scores.view(b_size, -1), k=k_next, dim=1)
            parent_k = top_flat // c_cand                        # (B, K_next)
            sym_cidx = top_flat % c_cand                         # (B, K_next)

            for b_i in range(b_size):
                if not done_b[b_i] and len(completed_per_b[b_i]) >= beam_size:
                    kth_done = sorted(completed_per_b[b_i].values(), reverse=True)[beam_size - 1]
                    if float(top_vals[b_i, 0].item()) <= kth_done - 1e-4:
                        done_b[b_i] = True

            if all(done_b):
                break

            parent_k_list = parent_k.tolist()
            sym_cidx_list = sym_cidx.tolist()
            prefixes = [
                [
                    prefixes[b_i][pk] + (known_syms[sc],)
                    for pk, sc in zip(parent_k_list[b_i], sym_cidx_list[b_i])
                ]
                for b_i in range(b_size)
            ]

            for s_idx, st in enumerate(seed_states):
                mdl = st["mdl"]
                dev = st["dev"]
                na = next_alphas_per_seed[s_idx]                 # (B, K_cur, M_max+1, C)
                pk_dev = parent_k.to(dev)                        # (B, K_next)
                sc_dev = sym_cidx.to(dev)                        # (B, K_next)

                # Gather selected (parent_k, sym_cidx) alignment vectors -> (B, K_next, M_max+1)
                na_perm = na.permute(0, 1, 3, 2).reshape(b_size, k_cur * c_cand, m_max + 1)
                flat_dev = top_flat.to(dev).unsqueeze(-1).expand(b_size, k_next, m_max + 1)
                st["alpha"] = torch.gather(na_perm, 1, flat_dev)

                # Gather parent LSTM states (1, B*K_cur, H_dec) -> (1, B*K_next, H_dec)
                h_old, c_old = st["hc"]
                h_dim = h_old.shape[-1]
                base_offsets = (torch.arange(b_size, device=dev) * k_cur).unsqueeze(1)
                hc_gather_idx = (
                    (base_offsets + pk_dev).reshape(1, b_size * k_next, 1).expand(1, -1, h_dim)
                )
                h_sel = torch.gather(h_old, 1, hc_gather_idx).contiguous()
                c_sel = torch.gather(c_old, 1, hc_gather_idx).contiguous()

                tok_rnn_ids = st["cand_rnn"][sc_dev.reshape(-1)].unsqueeze(1)  # (B*K_next, 1)
                tok_emb = mdl.char_emb(tok_rnn_ids)
                dec_c_rep = (
                    st["dec_c"]
                    .unsqueeze(1)
                    .expand(b_size, k_next, -1)
                    .reshape(b_size * k_next, 1, -1)
                )
                dec_in = torch.cat([tok_emb, dec_c_rep], dim=-1)
                h_out_next, new_hc = mdl.dec_lstm(dec_in, (h_sel, c_sel))
                st["dec_out"] = h_out_next.reshape(b_size, k_next, h_dim)
                st["hc"] = new_hc

        for b_i, q_idx in enumerate(chunk_q_idxs):
            ranked = sorted(completed_per_b[b_i].items(), key=lambda kv: (-kv[1], kv[0]))
            results[q_idx] = ranked[:beam_size]

    return [r if r is not None else [] for r in results]
