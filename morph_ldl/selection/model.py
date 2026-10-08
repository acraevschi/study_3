"""Character-level Transformer encoder-decoder used only to *choose* lemmas.

The selector is a probabilistic morphological inflector, distinct from LDL. Its only
role in the pipeline is to produce beam hypotheses whose log-probabilities drive the
active-selection scores (docs/SELECTION.md). It is a from-scratch PyTorch
re-implementation of the fairseq character Transformer used by Muradoglu & Hulden (2022)
(hyper-parameters after Liu & Hulden 2020), scaled down for tiny training sets.

Input / output format
---------------------
Encoder input (one token per position, no separators needed because the namespaces are
disjoint)::

    <S:F1> <S:F2> ...   s1 s2 ... sn   <T:G1> <T:G2> ...

* ``<S:F>``: one token per feature of the supplied *source* cell (``cell_norm`` split on
  ``;``), e.g. ``<S:NFIN>``;
* ``s1 .. sn``: the source form's segments (``forms.csv`` ``segments``, variant 0);
* ``<T:G>``: one token per feature of the requested *target* cell, e.g. ``<T:1>
  <T:IND> <T:PRS> <T:SG>``.

Decoder output: the target form's segments followed by ``<eos>`` (``<bos>`` is the
first decoder input). Segments are atomic symbols; ``_`` (word space) is an ordinary
symbol. Symbols outside the training vocabulary map to ``<unk>`` on the encoder side;
the decoder can never emit ``<pad>``, ``<bos>`` or ``<unk>``.

Determinism
-----------
Everything random (parameter init, dropout, batch order) is seeded from the single
``selector_init`` seed, the training examples are sorted before use (so the fitted model
depends on the *set* of training lemmas, not on their acquisition order), and CPU runs
use ``torch.use_deterministic_algorithms`` with a fixed thread count. See
docs/SELECTION.md for the residual nondeterminism (thread count / BLAS build, MPS).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import math
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


# ----------------------------------------------------------------------------- format

def feature_tokens(cell: str, prefix: str) -> List[str]:
    return [f"<{prefix}:{f}>" for f in str(cell).split(";") if f]


def split_segments(segments: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(segments, str):
        return tuple(s for s in segments.split(" ") if s != "")
    return tuple(segments)


def encode_input(source_cell: str, source_segments: str | Sequence[str], target_cell: str) -> Tuple[str, ...]:
    """Encoder token sequence for one (source anchor, target cell) query."""
    segs = split_segments(source_segments)
    for s in segs:
        if len(s) > 2 and s.startswith("<") and s.endswith(">"):
            raise ValueError(f"segment {s!r} collides with the reserved <...> token namespace")
    return tuple(feature_tokens(source_cell, "S")) + segs + tuple(feature_tokens(target_cell, "T"))


@dataclass(frozen=True)
class Example:
    """One selector training / dev item. ``gold_variants`` is only filled for dev items."""

    src: Tuple[str, ...]
    tgt: Tuple[str, ...]
    lemma_id: str = ""
    target_cell: str = ""
    n_source_segments: int = 0
    gold_variants: Tuple[Tuple[str, ...], ...] = ()


class Vocab:
    def __init__(self, symbols: Iterable[str]):
        rest = sorted(set(symbols) - set(SPECIALS))
        self.itos: List[str] = list(SPECIALS) + rest
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, toks: Sequence[str]) -> List[int]:
        return [self.stoi.get(t, UNK_ID) for t in toks]

    def decode(self, ids: Sequence[int]) -> Tuple[str, ...]:
        return tuple(self.itos[i] for i in ids)


# ----------------------------------------------------------------------------- config

@dataclass
class SelectorConfig:
    d_model: int = 128
    n_heads: int = 4
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.3
    label_smoothing: float = 0.1
    lr: float = 1e-3
    warmup_steps: int = 400
    batch_size: int = 64
    max_steps: int = 3000
    eval_every: int = 250
    early_stop_patience: int = 4
    beam_size: int = 5
    max_decode_len_factor: float = 2.0
    max_decode_len_offset: int = 10
    adam_betas: Tuple[float, float] = (0.9, 0.98)
    clip_norm: float = 1.0
    share_embeddings: bool = False  # joint encoder/decoder/output symbol embeddings (fairseq --share-all-embeddings)
    # Auxiliary copy (autoencoding) items, source cell -> source cell: "none"; "train" (sources of the
    # training lemmas); "pool" (training lemmas + every seed/pool candidate source form, which the
    # scorer is allowed to see anyway). Never test or dev forms.
    aux_copy: str = "none"
    device: str = "auto"
    num_threads: int = 8
    decode_batch_size: int = 256

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "SelectorConfig":
        """Build from a full pipeline config (``selector:`` + ``limits.max_threads``)
        or directly from a ``selector:`` mapping."""
        sel = dict(cfg.get("selector", cfg))
        arch = sel.pop("arch", "char_transformer")
        if arch != "char_transformer":
            raise ValueError(f"unsupported selector arch {arch!r}")
        known = {f.name for f in fields(cls)}
        unknown = set(sel) - known
        if unknown:
            raise ValueError(f"unknown selector settings: {sorted(unknown)}")
        if "num_threads" not in sel and "limits" in cfg:
            sel["num_threads"] = int(cfg["limits"].get("max_threads", 8))
        if "adam_betas" in sel:
            sel["adam_betas"] = tuple(sel["adam_betas"])
        return cls(**sel)

    def __post_init__(self):
        if self.aux_copy not in ("none", "train", "pool"):
            raise ValueError(f"selector.aux_copy must be none|train|pool, not {self.aux_copy!r}")

    def resolved_device(self) -> str:
        if self.device in ("cpu", "mps"):
            if self.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("selector.device=mps but MPS is unavailable")
            return self.device
        if self.device == "auto":
            # CPU by default: deterministic and, for models this small, not slower than
            # MPS (measured, docs/SELECTION.md).
            return "cpu"
        raise ValueError(f"unknown selector.device {self.device!r}")

    def max_decode_len(self, n_source_segments: int) -> int:
        """Maximum number of output symbols (excluding EOS), from the *source* length."""
        return int(math.floor(self.max_decode_len_factor * n_source_segments)) + int(self.max_decode_len_offset)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["adam_betas"] = list(d["adam_betas"])
        return d


# ----------------------------------------------------------------------------- model

class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        assert d % h == 0
        self.h, self.dh, self.dropout = h, d // h, dropout
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)

    def project_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._split(self.k_proj(x)), self._split(self.v_proj(x))

    def attend(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               mask: Optional[torch.Tensor]) -> torch.Tensor:
        """``mask``: bool, broadcastable to [B, H, Tq, Tk]; True = blocked."""
        b, tq, d = x.shape
        q = self._split(self.q_proj(x))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else ~mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(b, tq, d))


class FeedForward(nn.Module):
    def __init__(self, d: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1, self.fc2, self.dropout = nn.Linear(d, d_ff), nn.Linear(d_ff, d), dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.dropout(F.relu(self.fc1(x)), p=self.dropout, training=self.training))


class EncoderLayer(nn.Module):
    def __init__(self, d: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, h, dropout)
        self.ff = FeedForward(d, d_ff, dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        k, v = self.attn.project_kv(h)
        x = x + F.dropout(self.attn.attend(h, k, v, pad_mask), p=self.dropout, training=self.training)
        return x + F.dropout(self.ff(self.ln2(x)), p=self.dropout, training=self.training)


class DecoderLayer(nn.Module):
    def __init__(self, d: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = MultiHeadAttention(d, h, dropout)
        self.cross_attn = MultiHeadAttention(d, h, dropout)
        self.ff = FeedForward(d, d_ff, dropout)
        self.dropout = dropout

    def forward(self, x, mem_kv, src_mask, self_mask=None, cache=None):
        """Full-sequence (cache None, causal ``self_mask``) or incremental step (cache =
        (k, v) of previous positions; x holds only the new position)."""
        h = self.ln1(x)
        k, v = self.self_attn.project_kv(h)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        x = x + F.dropout(self.self_attn.attend(h, k, v, self_mask), p=self.dropout, training=self.training)
        x = x + F.dropout(self.cross_attn.attend(self.ln2(x), mem_kv[0], mem_kv[1], src_mask),
                          p=self.dropout, training=self.training)
        x = x + F.dropout(self.ff(self.ln3(x)), p=self.dropout, training=self.training)
        return x, (k, v)


def sinusoidal_positions(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CharTransformer(nn.Module):
    """Pre-norm Transformer encoder-decoder with sinusoidal positions and a decoder
    output projection tied to the decoder input embedding."""

    MAX_POS = 512

    def __init__(self, n_src: int, n_tgt: int, c: SelectorConfig):
        super().__init__()
        d = c.d_model
        self.d = d
        self.tgt_emb = nn.Embedding(n_tgt, d, padding_idx=PAD_ID)
        if c.share_embeddings:
            if n_src != n_tgt:
                raise ValueError("shared embeddings need one joint vocabulary")
            self.src_emb = self.tgt_emb
        else:
            self.src_emb = nn.Embedding(n_src, d, padding_idx=PAD_ID)
        self.register_buffer("pe", sinusoidal_positions(self.MAX_POS, d), persistent=False)
        self.enc = nn.ModuleList([EncoderLayer(d, c.n_heads, c.d_ff, c.dropout) for _ in range(c.n_enc_layers)])
        self.dec = nn.ModuleList([DecoderLayer(d, c.n_heads, c.d_ff, c.dropout) for _ in range(c.n_dec_layers)])
        self.enc_ln, self.dec_ln = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = c.dropout
        self._reset()

    def _reset(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and "emb" not in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        for emb in {id(e): e for e in (self.src_emb, self.tgt_emb)}.values():
            nn.init.normal_(emb.weight, mean=0.0, std=self.d ** -0.5)
            with torch.no_grad():
                emb.weight[PAD_ID].zero_()

    def _embed(self, emb: nn.Embedding, ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = emb(ids) * math.sqrt(self.d) + self.pe[offset: offset + ids.shape[1]].unsqueeze(0)
        return F.dropout(x, p=self.dropout, training=self.training)

    def encode(self, src: torch.Tensor):
        src_mask = (src == PAD_ID)[:, None, None, :]  # [B,1,1,S]
        x = self._embed(self.src_emb, src)
        for layer in self.enc:
            x = layer(x, src_mask)
        mem = self.enc_ln(x)
        mem_kv = [layer.cross_attn.project_kv(mem) for layer in self.dec]
        return mem_kv, src_mask

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.tgt_emb.weight)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        mem_kv, src_mask = self.encode(src)
        t = tgt_in.shape[1]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=src.device), diagonal=1)[None, None]
        x = self._embed(self.tgt_emb, tgt_in)
        for layer, kv in zip(self.dec, mem_kv):
            x, _ = layer(x, kv, src_mask, self_mask=causal)
        return self.logits(self.dec_ln(x))

    def decode_step(self, tok: torch.Tensor, pos: int, mem_kv, src_mask, cache):
        """One incremental decoder step. ``tok``: [B,1]. Returns log-probs [B,V], cache."""
        x = self._embed(self.tgt_emb, tok, offset=pos)
        new_cache = []
        for i, layer in enumerate(self.dec):
            x, kv = layer(x, mem_kv[i], src_mask, self_mask=None, cache=None if cache is None else cache[i])
            new_cache.append(kv)
        return torch.log_softmax(self.logits(self.dec_ln(x))[:, -1].float(), dim=-1), new_cache


# ----------------------------------------------------------------------------- helpers

@contextlib.contextmanager
def torch_runtime(num_threads: int, device: str):
    """Fix threads and deterministic algorithms for the duration of a fit/predict call."""
    old_threads = torch.get_num_threads()
    old_det = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(max(1, int(num_threads)))
    torch.use_deterministic_algorithms(True, warn_only=(device != "cpu"))
    try:
        yield
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_det, warn_only=old_warn)


def _pad(seqs: Sequence[Sequence[int]], device: str) -> torch.Tensor:
    n = max(1, max((len(s) for s in seqs), default=1))
    out = torch.full((len(seqs), n), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out.to(device)


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


@dataclass
class Hypothesis:
    segments: Tuple[str, ...]
    logprob_sum: float  # natural log, includes the EOS step
    hyp_len: int  # number of output symbols excluding EOS

    @property
    def hyp(self) -> str:
        return " ".join(self.segments)


# ----------------------------------------------------------------------------- trained model

@dataclass
class TrainedSelector:
    model: CharTransformer
    src_vocab: Vocab
    tgt_vocab: Vocab
    config: SelectorConfig
    device: str
    info: Dict[str, Any] = field(default_factory=dict)

    # -- decoding -----------------------------------------------------------------
    @torch.no_grad()
    def beam_search(self, srcs: Sequence[Sequence[str]], n_source_segments: Sequence[int],
                    beam_size: Optional[int] = None) -> List[List[Hypothesis]]:
        """Top-k hypotheses per input, sorted by ``logprob_sum`` (descending; ties by
        symbol sequence). Max output length per input = config.max_decode_len(source len)."""
        k = int(beam_size or self.config.beam_size)
        max_lens = [self.config.max_decode_len(int(n)) for n in n_source_segments]
        ids = [self.src_vocab.encode(s) for s in srcs]
        order = sorted(range(len(ids)), key=lambda i: (len(ids[i]), max_lens[i], i))
        out: List[Optional[List[Hypothesis]]] = [None] * len(ids)
        bs = max(1, int(self.config.decode_batch_size))
        self.model.eval()
        with torch_runtime(self.config.num_threads, self.device):
            for start in range(0, len(order), bs):
                chunk = order[start: start + bs]
                res = self._beam_batch([ids[i] for i in chunk], [max_lens[i] for i in chunk], k)
                for i, r in zip(chunk, res):
                    out[i] = r
        return out  # type: ignore[return-value]

    def _beam_batch(self, src_ids: List[List[int]], max_lens: List[int], k: int) -> List[List[Hypothesis]]:
        dev = self.device
        n, v = len(src_ids), len(self.tgt_vocab)
        mem_kv, src_mask = self.model.encode(_pad(src_ids, dev))
        rep = torch.arange(n, device=dev).repeat_interleave(k)
        mem_kv = [(a[rep], b[rep]) for a, b in mem_kv]
        src_mask = src_mask[rep]
        scores = torch.full((n, k), float("-inf"), device=dev)
        scores[:, 0] = 0.0
        seqs: List[List[int]] = [[] for _ in range(n * k)]
        last = torch.full((n * k, 1), BOS_ID, dtype=torch.long, device=dev)
        cache = None
        finished: List[List[Tuple[float, Tuple[int, ...]]]] = [[] for _ in range(n)]
        done = [False] * n
        maxlen_rows = torch.tensor(max_lens, device=dev).repeat_interleave(k)
        blocked = torch.tensor(self.blocked_output_ids(), device=dev)
        n_cand = min(2 * k, k * v)
        for t in range(max(max_lens) + 1):
            lp, cache = self.model.decode_step(last, t, mem_kv, src_mask, cache)
            lp[:, blocked] = float("-inf")
            force = maxlen_rows <= t  # at max length only EOS is allowed
            if bool(force.any()):
                eos_col = lp[:, EOS_ID].clone()
                lp[force] = float("-inf")
                lp[force, EOS_ID] = eos_col[force]
            cand = (scores.view(n * k, 1) + lp).view(n, k * v)
            top_s, top_i = cand.topk(n_cand, dim=1)
            top_s_l, top_i_l = top_s.cpu().tolist(), top_i.cpu().tolist()
            new_scores = torch.full((n, k), float("-inf"))
            src_rows = list(range(n * k))
            new_tok = [PAD_ID] * (n * k)
            new_seqs: List[List[int]] = [[] for _ in range(n * k)]
            for i in range(n):
                if done[i]:
                    continue
                j = 0
                for s, idx in zip(top_s_l[i], top_i_l[i]):
                    if s == float("-inf") or j >= k:
                        break
                    b, w = divmod(idx, v)
                    row = i * k + b
                    if w == EOS_ID:
                        finished[i].append((s, tuple(seqs[row])))
                    else:
                        new_scores[i, j] = s
                        src_rows[i * k + j] = row
                        new_tok[i * k + j] = w
                        new_seqs[i * k + j] = seqs[row] + [w]
                        j += 1
                if t >= max_lens[i] or j == 0:
                    done[i] = True
                elif len(finished[i]) >= k:
                    kth = sorted((f[0] for f in finished[i]), reverse=True)[k - 1]
                    if kth >= float(new_scores[i, 0]):  # scores only decrease: exact stop
                        done[i] = True
            if all(done):
                break
            for i in range(n):
                if done[i]:
                    new_scores[i] = float("-inf")
            idx_t = torch.tensor(src_rows, device=dev)
            cache = [(a[idx_t], b[idx_t]) for a, b in cache]
            last = torch.tensor(new_tok, device=dev).view(n * k, 1)
            scores = new_scores.to(dev)
            seqs = new_seqs
        results = []
        for i in range(n):
            fin = sorted(finished[i], key=lambda f: (-f[0], f[1]))[:k]
            results.append([Hypothesis(self.tgt_vocab.decode(s), float(sc), len(s)) for sc, s in fin])
        return results

    def blocked_output_ids(self) -> List[int]:
        """Output ids the decoder may never emit: <pad>, <bos>, <unk> and (with a joint
        vocabulary) the <S:..>/<T:..> feature tokens."""
        return [i for i, s in enumerate(self.tgt_vocab.itos)
                if s in (PAD, BOS, UNK) or (len(s) > 2 and s[0] == "<" and s[-1] == ">" and s != EOS)]

    @torch.no_grad()
    def sequence_logprob(self, src: Sequence[str], tgt: Sequence[str]) -> float:
        """Teacher-forced natural-log probability of ``tgt`` + EOS (used in tests)."""
        self.model.eval()
        s = _pad([self.src_vocab.encode(src)], self.device)
        y = self.tgt_vocab.encode(tgt)
        tin = _pad([[BOS_ID] + y], self.device)
        lp = torch.log_softmax(self.model(s, tin).float(), dim=-1)[0]
        gold = y + [EOS_ID]
        return float(sum(lp[i, g] for i, g in enumerate(gold)))


# ----------------------------------------------------------------------------- training

def _batch_loss(model, src_ids, tgt_ids, idx, device, smoothing):
    src = _pad([src_ids[i] for i in idx], device)
    tin = _pad([[BOS_ID] + tgt_ids[i] for i in idx], device)
    tout = _pad([tgt_ids[i] + [EOS_ID] for i in idx], device)
    logits = model(src, tin)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tout.reshape(-1),
                           ignore_index=PAD_ID, label_smoothing=smoothing, reduction="sum"), int((tout != PAD_ID).sum())


@torch.no_grad()
def _evaluate_dev(sel: TrainedSelector, dev: Sequence[Example]) -> Tuple[float, float]:
    """Greedy (beam 1) exact-match accuracy against any gold variant, and the
    teacher-forced per-token NLL (no smoothing) of variant 0."""
    if not dev:
        return float("nan"), float("nan")
    hyps = sel.beam_search([e.src for e in dev], [e.n_source_segments for e in dev], beam_size=1)
    correct = 0
    for e, h in zip(dev, hyps):
        golds = e.gold_variants or (e.tgt,)
        correct += int(bool(h) and h[0].segments in golds)
    sel.model.eval()
    src_ids = [sel.src_vocab.encode(e.src) for e in dev]
    tgt_ids = [sel.tgt_vocab.encode(e.tgt) for e in dev]
    tot, ntok = 0.0, 0
    for s in range(0, len(dev), 512):
        l, nt = _batch_loss(sel.model, src_ids, tgt_ids, list(range(s, min(len(dev), s + 512))), sel.device, 0.0)
        tot += float(l)
        ntok += nt
    return correct / len(dev), tot / max(1, ntok)


def train_selector(train: Sequence[Example], dev: Sequence[Example], config: SelectorConfig, seed: int,
                   log: Optional[Callable[[str], None]] = None) -> TrainedSelector:
    """Train a fresh selector. Dev gold only guides early stopping / checkpoint choice."""
    if not train:
        raise ValueError("no training examples")
    t0 = time.perf_counter()
    device = config.resolved_device()
    train = sorted(train, key=lambda e: (e.lemma_id, e.target_cell, e.src, e.tgt))
    if config.share_embeddings:
        src_vocab = tgt_vocab = Vocab([s for e in train for s in e.src] + [s for e in train for s in e.tgt])
    else:
        src_vocab = Vocab(s for e in train for s in e.src)
        tgt_vocab = Vocab(s for e in train for s in e.tgt)
    with torch_runtime(config.num_threads, device):
        torch.manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))
        model = CharTransformer(len(src_vocab), len(tgt_vocab), config).to(device)
        sel = TrainedSelector(model, src_vocab, tgt_vocab, config, device)
        opt = torch.optim.Adam(model.parameters(), lr=config.lr, betas=tuple(config.adam_betas), eps=1e-8)
        src_ids = [src_vocab.encode(e.src) for e in train]
        tgt_ids = [tgt_vocab.encode(e.tgt) for e in train]
        n = len(train)
        best_key, best_state, best_step, bad = None, None, 0, 0
        curve: List[Dict[str, float]] = []
        step, stopped_early = 0, False
        while step < config.max_steps and not stopped_early:
            perm = rng.permutation(n)
            for s in range(0, n, config.batch_size):
                step += 1
                lr = config.lr * min(step / max(1, config.warmup_steps), math.sqrt(max(1, config.warmup_steps) / step))
                for g in opt.param_groups:
                    g["lr"] = lr
                model.train()
                loss, ntok = _batch_loss(model, src_ids, tgt_ids, perm[s: s + config.batch_size].tolist(),
                                         device, config.label_smoothing)
                opt.zero_grad(set_to_none=True)
                (loss / max(1, ntok)).backward()
                if config.clip_norm and config.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
                opt.step()
                if step % config.eval_every == 0 or step == config.max_steps:
                    acc, dloss = _evaluate_dev(sel, dev)
                    curve.append({"step": step, "dev_acc": acc, "dev_loss": dloss,
                                  "train_loss": loss.item() / max(1, ntok)})
                    key = (acc, -dloss) if dev else (step, 0.0)
                    if best_key is None or key > best_key:
                        best_key, best_step, bad = key, step, 0
                        best_state = copy.deepcopy(model.state_dict())
                    else:
                        bad += 1
                    if log:
                        log(f"  step {step}: dev_acc={acc:.3f} dev_loss={dloss:.3f} lr={lr:.2e}")
                    if dev and bad >= config.early_stop_patience:
                        stopped_early = True
                        break
                if step >= config.max_steps:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
    best = next((c for c in curve if c["step"] == best_step), {})
    sel.info = {
        "n_train_examples": n,
        "n_dev_examples": len(dev),
        "steps_run": step,
        "best_step": best_step,
        "stopped_early": stopped_early,
        "dev_acc": best.get("dev_acc"),
        "dev_loss": best.get("dev_loss"),
        "dev_curve": curve,
        "train_runtime_s": round(time.perf_counter() - t0, 3),
        "device": device,
        "num_threads": config.num_threads,
        "seed": int(seed),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "model_hash": state_hash(model),
    }
    return sel
