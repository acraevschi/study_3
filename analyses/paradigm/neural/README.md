# Hard-Monotonic Neural Character Transducer (`analyses/paradigm/neural/`)

Secondary convergent form-to-form ($w_{c_s} \to w_{c_t}$) morphological predictability engine for Study 3 Track 2, following Wu & Cotterell (2019, *ACL*).

## Architecture & Proper Scoring Guarantees
1. **Exact Generative String Probability over $\Sigma^*$ (`monotonic_transducer.py`):**
   - Encodes source symbol sequence $x_1 \dots x_M \langle\text{eos}\rangle$ via a padding-invariant BiLSTM (`enc_fwd` + involution-reversed `enc_bwd`, avoiding `pack_padded_sequence` MPS quirks).
   - Encodes causal target prefixes $\langle\text{bos}\rangle y_1 \dots y_N$ in parallel via a 1-layer LSTM conditioned on learned cell embeddings `emb(c_s), emb(c_t)`.
   - Projects `(enc_out[i], dec_out[j], cell_ctx)` across the entire `(B, M+1, N+1)` lattice in one fused matrix multiply to normalized log-probabilities over `[STOP, COPY, DELETE, INSERT(Sigma), SUBSTITUTE(Sigma)]`, where `STOP` is active iff $i = M$ and `COPY/DELETE/SUBSTITUTE` are active iff $i < M$.
   - Marginalizes over all monotonic alignments via an exact **anti-diagonal wavefront dynamic program** (`d = i + j` from `1` to `M + N`) with zero in-place tensor writes and zero label smoothing.
2. **3-Seed Ensemble & Dev-Only Topology (`train_eval.py`):**
   - Trains 3 independent random seeds on `subtrain` with `dev` exact marginal log-likelihood early stopping (stratified cap of 35,000 `(c_s, c_t)` pairs for large paradigms).
   - Computes probability-averaged ensemble predictions `logsumexp_s log q_s(y | x) - log(S)`.
   - Selects optimal root $r^*$, **Chu-Liu-Edmonds** minimum directed spanning arborescence $T_{r^*}$, and `k=3` inverse-entropy PoE weights strictly on `dev` (zero test peeking), refits on full `train`, and evaluates held-out `test` paradigms.
   - Emits `unit_outcomes_{regime}.csv` and `lemma_outcomes_{regime}.csv` matching the MGL pipeline's exact schema.

## Usage
```bash
OMP_NUM_THREADS=1 .venv/bin/python -m analyses.paradigm.neural.train_eval \
  --units finn1318_N_orth ital1282_V_orth \
  --regime uniform120 \
  --device auto
```
