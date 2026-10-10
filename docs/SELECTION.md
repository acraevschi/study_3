# Active verb selection (`morph_ldl/selection`)

Owner: main agent (pcfp_v1 LDL selector); the Transformer sections below are the
selection subagent's pilot_v1 record. Interface: docs/CONTRACT.md §5.

## Part A. pcfp_v1: LDL selects its own training verbs

In Muradoğlu & Hulden (2022) the model that chooses the data is the model that is
evaluated. In pilot_v1 it was not: a character Transformer chose verbs that LDL was then
trained on, and the choices helped the Transformer (Italian +0.039) but hurt LDL (Italian
−0.137 against random). In pcfp_v1 the selector **is** the evaluated learner: a JudiLing
LDL model with the frozen `ldl_tune` settings and the evaluated semantics.

### A.1 Files

| file | content |
|---|---|
| `selection/ldl_acquisition.py` | `ShownOracle`, `CitationPoolView`, `run_ldl_acquisition` (rounds, logs, samples) |
| `ldl/selector.py` | `SelectorServer` (persistent Julia process), `selector_configs`, score definitions |
| `julia/bin/selector_server.jl` | JSON-line server: fit per semantic seed, score every candidate |
| `julia/src/LDLRunner.jl` | `add_row` / `row_chat` (exact rank-one update), `score_candidate`, `score_round` |
| `selection/acquisition.py`, `model.py`, `scoring.py` | reused helpers (`round_plan`, `rank_lemmas`, seeds); the Transformer itself is kept only so pilot_v1 stays reproducible and is not used in pcfp_v1 |

### A.2 One acquisition round

1. Training rows = the shown forms of core + seed + acquired verbs (`ShownOracle.reveal`;
   the oracle never holds a hidden cell). Written to `rounds/r{n}/train.csv`.
2. For each semantic seed j ∈ {0, 1, 2} (seed 0 = the evaluated LDL's `semantic` seed,
   others `selector_semantic`), the server fits one background (C, S, F, G).
3. Each remaining pool candidate is scored **from that same immutable background**:
   * Its citation row is added: cues of the segmented lemma label (the only form the
     selector knows), and the simulated meaning L(v) + Σ V(NFIN features) + N(v, NFIN).
     Production and comprehension are updated by exact rank-one (Sherman–Morrison)
     updates (`add_row`). The citation's novel cues extend the inventory and the
     adjacency for this candidate only.
   * The candidate's pre-drawn shown cells (exposure manifest) are decoded with
     `learn_paths` from L(v) + Σ V(cell features). This is the k = 1 case of the
     evaluated task.
   * The row is discarded. No candidate affects another candidate's score, and the
     candidate order does not matter (tested).
   * The citation cell itself may be among the shown cells. It is decoded like any
     other cell, and the candidate is then (rightly) less uncertain.
4. Scores per cell from the top `max_can` = 10 supports s₁ ≥ s₂ ≥ … (synthesis-by-analysis
   correlations):
   `low_confidence` u = 1 − s₁; `high_entropy` H(softmax(s / 0.02)). T was 0.1 when first
   declared and was lowered before the real run; see A.5. A cell with no
   candidate or a decoder error counts as s₁ = −1 (u = 2) and H = log 10 (logged as
   `n_nonfinite`). Lemma score = mean over shown cells, then over the 3 seeds; the seed SD
   is logged (`score_seed_sd`).
5. Rank by score (descending), ties by `sha256(tie_seed:lemma_id)`; acquire `batch_size`;
   reveal their shown forms.

`random` draws one uniform number per pool verb (`random_policy` seed) and never runs the
selector. The seed verbs are identical for every policy.

### A.3 What the selector may know, and why

LDL cannot know the meaning of a verb it has never seen: the simulated lexeme vector
carries no information about the forms. The user chose the **citation form as one known
form** as the primary method. The lemma label is treated as a known lexical label for
the selector only, and the citation cell is therefore never a test item
(PROTOCOL §2). The alternative of estimating the meaning by comprehension and binding it
to the citation form (`wug_refit`, pilot_v1) is not used. Candidate forms other than the
label never reach the selector: the candidate table has exactly `lemma_id,
citation_cell, citation_segments, shown_cells`, and the audit re-derives every
citation segment from the label.

### A.4 Comprehension-side check (not used for selection)

Per candidate and round, from the same background: cor(c·F, s) and ‖c·F − s‖/‖s‖ between
the meaning read from the citation cues and the candidate's simulated citation meaning,
and the share of citation cues unseen in training (`comprehension_check.csv`).
`eval/selector_checks.csv` reports per round:
* their Spearman correlations with the selector score and with citation length;
* η² of the score by final 2/3 letters and by the inflection-class proxy (Italian
  -are/-ere/-ire/-rre; Finnish infinitive-ending proxy for the Kotus types);
* R² and adjusted R² of the score on citation length + final letters;
* the rate of top candidates that copy the citation form.

### A.5 Degeneracy stop condition (smoke run, before the real run)

Declared criteria: scores near-constant (round CV < 0.01), dominated by citation copying
(> 90% of scored cells), or explained by length and final letters alone (adjusted
R² > 0.9). In `pcfp_smoke` (40–60 candidates, 2 seeds) the round CV was 0.10–0.43, the
copy rate 7–29%, and the adjusted R² at most 0.54. All Italian infinitives end in *-re*,
so the final-2 model is empty there and the final-3 model is used. None of the criteria
was met, so the primary method was kept. The between-seed SD of a candidate's score is
about 1/3 (low_confidence) to 1/2 (high_entropy) of the between-candidate SD. That noise is
why scores are averaged over 3 seeds.

**Review finding and pre-run change (user decision).** The independent review found that
`high_entropy` at the declared T = 0.1 mostly counted decoded candidates. Its correlation
with log(n candidates) was 0.94, 60% of cells sat at the `max_can` cap, and its
correlation with 1 − s₁ was −0.12. The supports of competing candidates are close: the
top-two gap has quartiles 0.016 / 0.040 / 0.085. With T = 0.1 the softmax is therefore
nearly flat and H ≈ log n. The user chose T = 0.02 (r with log n = 0.54) before any real
selection. Only the smoke score distributions informed this; no accuracy was looked at.
The declared degeneracy criteria were not met either way.

**Known property, kept as declared.** A candidate whose citation cell (NFIN) is among its
shown cells looks almost certain on that cell: the citation row makes NFIN a training
item for the selector. Its mean score is therefore lower (low_confidence 0.219 vs 0.250
in the smoke run), and NFIN-only k = 1 verbs are rarely selected. This follows from the
selector's knowledge model (the label is known) and is reported, not corrected.

### A.6 Runtime

One Julia process per acquisition job (unit × fold × policy × pool cap), started once.
In the smoke run each round's scoring took 2–13 s for 30–60 candidates and 2 seeds.
Real-run timings are in the run's `rounds.json` and the report.

## Part B. pilot_v1: character-Transformer selector (archived record)

The text below describes the selector of pilot_v1 (code state 24390cf). It is kept
unchanged.

Source: Muradoğlu & Hulden (2022), *Eeny, meeny, miny, moe. How to choose data for
morphological inflection*, EMNLP. Upstream scripts: `external/ALmorphinfl`
(smuradoglu/ALmorphinfl), revision `3caf0d059846569ed0aff4b833492c104786c8e1` (checked
with `git rev-parse HEAD`).

## 1. Files

| file | content |
|---|---|
| `selection/model.py` | char-level Transformer selector (PyTorch), training, beam search |
| `selection/scoring.py` | per-hypothesis/cell scores, lemma aggregation, ranking |
| `selection/acquisition.py` | `CandidatePoolView`, `Oracle`, policies, `run_acquisition`, outputs, `predict_queries` |
| `selection/fixtures.py` | TEMPORARY fixture reader for MGN wide CSVs and a fixture splitter (tests and smoke runs only) |
| `selection/smoke.py` | bounded real-data smoke run (Italian verbs) to `outputs/scratch_selection/` |
| `tests/test_selection_*.py` | tests for scoring, model and acquisition (§5, §7) |

## 2. Reused vs rewritten

Nothing from the upstream scripts is imported or copied. They are one-off analysis scripts:
hard-coded `os.chdir` paths and language ids, sizes fixed in code (`sample_size = 250`,
`range(0, 1000)`), no seeds, positional beam records (5 lines per input, `log_like[0::5]`),
and fairseq output files as input. What is kept is the *method*: the three policies (lowest
model confidence, highest beam entropy with the 0.05 cut-off, random), and the
retrain-score-select cycle. Everything else is rewritten for lemma-level selection, inner
cross-validation and gold isolation.

Upstream problems and how they are handled:

| upstream issue | finding | resolution here |
|---|---|---|
| log-likelihood read as strings and sorted lexicographically (`resample_confidence.py`, `sort_values('loglikelihood', ascending=False)`) | On all 70 released `tst.*n.guesses` files every value is a plain decimal in (−10, 0), so the lexicographically "descending" order equals the *numerically ascending* order; the top 250 overlap is 100 % in every file. So upstream "low confidence" did select the lowest log-likelihoods, but only by accident; numeric `ascending=False` would have selected the most confident items, and values ≤ −10 or in scientific notation would break it. | Scores are floats from the model. One direction convention: every score is "higher = more uncertain = selected first" (`surprisal_norm = −logprob/(len+1)`, entropy). Tested. |
| ambiguous score direction / sign | The upstream values are negative log-probabilities (−0.017 … −2.2). | We store `logprob_sum` (natural log, ≤ 0) and derive the positive `surprisal_norm`. |
| no length normalisation | The released scores are about −0.03 … −1 for whole words, i.e. they are fairseq's *length-normalised* hypothesis scores (fairseq's default `normalize_scores`), so upstream was implicitly per-token. | Explicit: `surprisal_norm = −logprob_sum / (hyp_len + 1)` with the *predicted* length (+1 for EOS); gold length is never used. |
| paper threshold p_i ≥ 0.05 vs script's cumulative `while p_sum <= 0.95` | The two rules can differ in general. On the 211 released `*n5.guesses` files they give identical entropies, because upstream exponentiated the per-token normalised scores, which makes the 5 beam probabilities nearly flat (all ≥ 0.05 and all included by the loop). | Paper rule: `H = −Σ_{p_i ≥ 0.05} p_i log p_i`, with p_i renormalised over the beam from sequence-level `logprob_sum`. See §3 for the deliberate divergence from per-token renormalisation. |
| positional 5-line beam records, fixed beam count | | Beams are structured objects; any number of hypotheses (≤ beam size) is handled. |
| hard-coded paths, languages, sizes, no seeds | | Everything from config + `seeds.derive`; outputs per CONTRACT §5. |
| non-finite scores unhandled | | §3 non-finite rule; tested. |
| selection unit = individual (lemma, tag, form) triples drawn from the *test file* | | The selection unit is the lemma (all panel cells come together), drawn from the inner candidate pool. Outer-test lemmas are never visible. |

## 3. Scores, directions and aggregation

For each candidate lemma and each panel target cell (the source cell is supplied, never a
target) the selector returns up to `beam_size` hypotheses with `logprob_sum` (natural log
of P(symbols, EOS | input), EOS step included) and `hyp_len` (symbols, EOS excluded).

* Top hypothesis = highest `logprob_sum` (unnormalised sequence probability, i.e. the mode
  of the beam distribution; ties by symbol string).
* **low_confidence** cell score: `surprisal_norm` of the top hypothesis,
  `−logprob_sum / (hyp_len + 1)`. Higher = selected first.
* **high_entropy** cell score: `p_i = exp(lp_i − logsumexp(lp))` over the finite beam
  hypotheses, `H = −Σ_{p_i ≥ 0.05} p_i log p_i`. Probabilities are not renormalised again
  after the cut-off (the paper renormalises over the beam, then filters). Higher = selected
  first. Note: we renormalise *sequence* probabilities (CONTRACT §5); upstream in practice
  renormalised exp(per-token score), which gives much flatter distributions and makes the
  0.05 cut-off inactive. Main-agent decision if the per-token variant is preferred.
* **random**: one uniform draw per pool lemma from the `random_policy` seed (sorted lemma
  ids); each round takes the highest remaining draws. The resulting order is a seeded random
  permutation, independent of batch size and of any model (tested).
* **oracle_incorrect** (optional, labelled, never pooled): mean over panel cells of the
  normalised edit distance between the top hypothesis and the closest gold variant. It reads
  candidate gold through `Oracle.peek_for_oracle_policy`, logged as `oracle_policy_scoring`.

Lemma score (`aggregation: mean_cell`) = mean of the finite cell scores over the fixed
panel. Per-cell scores are kept (`cell_scores.csv` per hypothesis, `cell_summary.csv`
per cell with top hypothesis, surprisal, entropy and hypothesis counts).

Non-finite handling: hypotheses with a NaN/±inf `logprob_sum` are dropped before ranking
and renormalisation (kept in `cell_scores.csv` with `prob_renorm = NaN`). A cell with no
finite hypothesis is counted in `n_nonfinite`; the lemma score is the mean over its finite
cells; a lemma with no finite cell gets `+inf` (treated as most uncertain, listed first) and
is flagged (`all_nonfinite`, `rounds.json: n_lemmas_all_nonfinite`). With the current
model none occurred in any run.

Ranking: score descending, ties by `seeds.tie_key(tie_seed, lemma_id)` (CONTRACT §5).
Seed lemmas are ordered by the same tie key (`round = 0`, `score_name = seed`).

### How missing cells, paradigm size and aggregation affect selection

* Missing cells: eligibility (CONTRACT §4) requires all panel cells, so every candidate is
  scored on the same 8 cells and missing targets cannot change a lemma's denominator. In
  `all_cells` training mode, extra non-panel cells only enter *training* after a lemma is
  selected; selection scores are always over the panel.
* Paradigm size: because the score is a *mean* over a fixed panel, lemmas with larger
  paradigms or more variants gain no advantage. With a fixed panel, mean and sum give
  the same ranking. They would differ only with variable panels, which are not used here.
  In `all_cells` mode a selected lemma brings more training items when its paradigm is
  larger; that exposure is reported (`n_training_examples_all_cells`), but it never
  enters the score.
* Variants: candidate scoring uses only the source variant 0; training uses variant 0
  targets; dev accuracy accepts any variant.
* Length: `surprisal_norm` is per output step, so long forms are not automatically
  "uncertain"; entropy is sequence-level and is length-sensitive (longer outputs tend to
  have flatter beams). This is a property of the paper's definition, kept on purpose.
* Mean aggregation lets a lemma with one very uncertain cell (e.g. an irregular past) rank
  below a lemma that is moderately uncertain everywhere. `cell_summary.csv` allows a
  max-cell sensitivity analysis without re-running.

## 4. Selector model

Character-level pre-norm Transformer encoder–decoder written in PyTorch (no fairseq
dependency): sinusoidal positions, embeddings scaled by √d, decoder output projection tied
to the decoder input embedding, ReLU feed-forward, cross-entropy with label smoothing,
Adam (β = 0.9, 0.98, ε = 1e-8), linear warm-up then inverse-square-root decay, gradient
clipping 1.0, dropout on embeddings/attention/residuals.

Input format (encoder tokens): `<S:f>` per source-cell feature, the source segments,
`<T:g>` per target-cell feature, e.g.
`<S:NFIN> a m a r e <T:1> <T:IND> <T:PRS> <T:SG>`; decoder: target segments + `<eos>`.
Segments are `forms.csv` `segments` symbols (word space `_`).

Training protocol (identical for all policies, rounds, folds, budgets):
* trained from scratch every round with the fixed `selector_init` seed (shared by all
  policies of a fold, CONTRACT §8);
* training examples = (source variant 0 → each panel target cell, variant 0) of the
  revealed lemmas, sorted by (lemma, cell), so the model depends on the *set* of training
  lemmas, not their acquisition order;
* optional auxiliary copy items (`selector.aux_copy`, §6);
* dev lemmas (inner dev role): greedy exact-match accuracy (any gold variant) every
  `eval_every` steps; checkpoint = best (dev accuracy, then lower dev NLL); stop after
  `early_stop_patience` evaluations without improvement or at `max_steps`. Dev gold is
  used only for this; dev lemmas never become training data;
* beam search with exact termination for unnormalised scores (a lemma stops once its k-th
  finished hypothesis beats every live one), at most `beam_size` hypotheses; maximum
  output length `floor(max_decode_len_factor × source_len) + max_decode_len_offset`
  (2.0 × n + 10), from the *source* length only; at that length EOS is forced. The decoder
  cannot emit `<pad>`, `<bos>`, `<unk>` or feature tokens.
* Beam scores equal teacher-forced scores of the same strings (tested to 1e-4).

Adaptations from the paper (fairseq Transformer, hyper-parameters after Liu & Hulden 2020,
trained on 600–3,500 triples):
1. Much smaller training sets (seed 20 lemmas = 160 items), so a smaller model
   (`d_model` 128, 2+2 layers, 4 heads, FFN 512, from `pilot.yaml`) and a step budget with
   dev early stopping instead of a fixed epoch count.
2. Retraining from scratch each round (the paper also retrained per cycle).
3. Lemma-level batches of `batch_size` lemmas instead of 250 triples.
4. Sequence-level beam renormalisation for entropy (see §3).
5. Optional shared embeddings and auxiliary copy items (§6) — not in the paper.

## 5. Gold isolation (how it is enforced)

* `CandidatePoolView.from_forms` keeps only the variant-0 *source-cell* rows of pool
  lemmas and exposes `CandidateQuery(lemma_id, source_form, source_segments, source_cell,
  target_cells)`; `score_candidates` / `beam_candidates` receive only these objects.
* `Oracle` holds seed + pool rows. `select_and_reveal` registers the lemmas (round, rank)
  and only then returns rows; `reveal` of an unselected lemma raises `GoldAccessError`;
  every reveal (seed, selected, dev read, oracle-policy peek) is in `oracle_reveals.csv`.
* The selector is trained only on `oracle.reveal(selected)`; samples are written from
  the oracle as well. Rows of lemmas outside seed/pool/dev (e.g. outer test) are dropped at
  entry; overlapping role lists raise.
* Tests: permuting the gold forms of pool lemmas that were never selected leaves
  `order.csv`, `acquisition_log.csv` and `cell_scores.csv` byte-identical; permuting *all*
  pool gold leaves round 1 unchanged; `reveal` before selection raises; prediction ignores
  extra gold columns in a query table.

## 6. Smoke runs, runtime and recommended settings

Machine: Apple-silicon Mac, 18 cores, torch 2.14.1, CPU. Temporary Italian fixture
(`fixtures.mgn_wide_to_forms` on `mgn_data/data/ita-v.csv`, NFIN → the 8 pilot panel
cells, 9,983 eligible lemmas, one-lemma groups, seeded fixture split).

Timing per training step (batch 64, `pilot.yaml` model), one process: 1 thread 0.051 s,
2 threads 0.057 s, 4 threads 0.059 s, 8 threads 0.066 s, MPS 0.081 s. The model is too
small to profit from threads or MPS, so `device: auto` resolves to CPU and one thread per
process is fastest; parallelise across (unit, fold, policy) processes instead. With 12–13
concurrent one-thread processes a step takes about 0.13–0.15 s (contention), i.e. ~4.5×
total throughput.

Dev accuracy (80 dev lemmas × 8 cells, greedy) of one selector training, by number of
training lemmas (13 runs in parallel, one thread each; times are under contention):

| setting | 20 lemmas | 60 | 100 | wall time / training |
|---|---|---|---|---|
| A `pilot.yaml` as is | 0.00 (smoke) | 0.09 | 0.36 | 3000 steps ≈ 150 s alone, 445 s in parallel |
| B + `share_embeddings` | 0.00 | 0.19 | 0.45 | same |
| C B with d_model 64 / FFN 256 | 0.00 | 0.06 | 0.28 | ~0.6× |
| D B with dropout 0.1 | 0.00 | 0.17 | 0.32 | same |
| E B with max_steps 6000 | – | 0.20 | 0.51 | 2× |
| F B + `aux_copy: train` | 0.00 | 0.23 | 0.47 | same |
| **F B + `aux_copy: pool`** | **0.60** | **0.70** | **0.81** | same |
| G F-pool with max_steps 6000 | 0.62 | 0.73 | 0.83 | ~1.4× (stops ~4300) |

With the `pilot.yaml` model the seed-round selector (20 lemmas) never produces a correct
form on dev: it does not learn to copy stems from 160 items, so its round-1 scores mostly
reflect stem shape rather than inflectional difficulty. Shared embeddings plus auxiliary
copy items over the seed + pool *source* forms (which the scorer may already see) fix this.

Scoring (beam 5, one process): 200 candidates × 8 cells 1.4–3.4 s; 500 × 8 cells 3.7 s
(1 thread) / 3.0 s (4 threads). Scoring is negligible next to training.

### 6.1 Recommended-settings smoke run (`smoke.py`, tag `smoke_rec`)

Fixture split seed 20 / dev 40 / pool 200 lemmas, batch 20, budget 60, NFIN → 8 panel
cells, `pilot.yaml` selector + `share_embeddings: true`, `aux_copy: pool`,
`num_threads: 1`; low_confidence and high_entropy ran as two concurrent processes.
Outputs: `outputs/scratch_selection/smoke_rec/selection/ita.V.orth.mgn/rep0/fold0/`.

| policy | round | train lemmas / items (+copy) | dev acc | best step | train s | score s |
|---|---|---|---|---|---|---|
| low_confidence | 1 | 20 / 160 (+220) | 0.431 | 2500 | 152 | 1.5 |
| low_confidence | 2 | 40 / 320 (+220) | 0.556 | 2750 | 150 | 1.5 |
| low_confidence | final fit | 60 / 480 (+220) | 0.581 | 2500 | 158 | – |
| high_entropy | 1 | 20 / 160 (+220) | 0.431 | 2500 | 153 | 1.5 |
| high_entropy | 2 | 40 / 320 (+220) | 0.544 | 3000 | 150 | 1.4 |
| high_entropy | final fit | 60 / 480 (+220) | 0.600 | 3000 | 157 | – |
| random (selector fitted afterwards on `budget_60.csv`) | – | 60 / 480 (+220) | 0.478 | 3000 | 167 | – |

Round 1 has the same model hash under both active policies (same seed set and
`selector_init`), as intended. Round-1 lemma scores of the two active policies correlate
strongly (Spearman 0.96); 26 of 40 acquired lemmas are shared; overlap with random is 7–10.
The active policies first pick reflexive (-rsi) and 2nd/3rd-conjugation verbs
(*rinchiudersi, correggersi, trasmettersi, contorcersi, disperdersi*). Under the
sequence-level definition most cells have one hypothesis above 0.05 (1,010 of 1,600 cells in
round 1), so the 0.05 cut-off is active. One fold, 40 dev lemmas: anecdotal.

Runtime per acquisition job (one policy, one fold, budget 100 from seed 20 = 4 rounds):
≈ 4 × 2.5 min training + < 10 s scoring ≈ 10–11 min on one core when few jobs run
together; slower (up to ~7 min per training) when 12+ jobs share the machine.

## 7. Determinism

* All randomness comes from `selector_init` (init, dropout, batch order via a NumPy
  generator), `random_policy` and `tie` seeds (CONTRACT §8). Training items are sorted.
* CPU runs use `torch.use_deterministic_algorithms(True)`. Verified: two runs with the same
  data, seed and thread count give the same model hash; reversed input order gives the same
  hash.
* Residual nondeterminism: (1) the CPU thread count changes floating-point reduction
  order — 1, 4 and 8 threads gave three different hashes — so `num_threads` must be fixed
  (and recorded; it is in `selection_summary.json`); (2) BLAS/PyTorch build or hardware
  changes; (3) MPS is not deterministic (two identical MPS runs gave different hashes) and
  is slower here, so `device: auto` resolves to CPU. The hash of every round's model is in
  `rounds.json`.
* The global torch thread count and determinism flag are restored after each call.

## 8. Interface (for the CLI)

```python
from morph_ldl.selection import (SelectionTask, SelectionSeeds, run_acquisition, policy_dir,
                                 predict_queries, train_selector_on_sample)
task  = SelectionTask.from_config(cfg, unit_id)          # source cell, panel cells, training_mode
seeds = SelectionSeeds.derive(master_seed, unit_id, rep, fold)
res = run_acquisition(policy, seed_ids, pool_ids, dev_ids, forms_df, task, cfg, seeds,
                      policy_dir(outputs_root, unit_id, rep, fold, policy),
                      test_ids=test_ids,            # checked for disjointness only
                      train_random_rounds=None,     # default cfg.selection.random_trains_selector or False
                      final_fit=False, keep_selectors=False, log=print)
# res.order, res.rounds, res.sample_paths[B] = {"panel", "allforms", "lemmas"}, res.summary
pred = predict_queries(selector_or_sample_df, test_queries_df, task=task, dev_rows=dev_forms,
                       cfg=cfg, seed=seeds.selector_init, anchor_rows=seed_pool_forms)
# columns: lemma_id, source_cell, target_cell, prediction, prediction_segments,
#          logprob_sum, hyp_len, surprisal_norm, n_hyps, status (ok|no_hypothesis)
```

Outputs per policy directory (CONTRACT §5): `order.csv`, `acquisition_log.csv`,
`cell_scores.csv`, `rounds.json`, `oracle_reveals.csv`, `samples/budget_{B}.csv`,
`samples/budget_{B}_allforms.csv`; additional: `samples/budget_{B}_lemmas.csv`
(lemma_id, acquisition_rank, round, weight = 1.0), `cell_summary.csv`,
`selection_summary.json` (per-budget lemma/form/example counts, weights, rounds used,
partial last round), `stage_manifest.json` (CONTRACT §9).

`budget_{B}.csv` holds all variant rows of the source + panel cells of the first B lemmas
(FORMS_COLUMNS only, ordered by acquisition rank); `budget_{B}_allforms.csv` holds all
non-missing rows with a parseable `cell_norm` of the same lemmas.

## 9. Open points for the main agent

1. Adopt `share_embeddings: true` and `aux_copy: pool` (both outside the paper), and
   `num_threads: 1` with process-level parallelism. Without them the seed-round selector
   is at 0 % dev accuracy.
2. `aux_copy: pool` makes the selector's training set depend on the pool size (pool_cap 500
   vs the 250 sensitivity pool → 520 vs 270 copy items), which confounds the pool-cap
   sensitivity analysis with selector quality. Options: copy items from a fixed-size anchor
   set (e.g. seed + the first 250 pool lemmas by `role_rank`, identical for both caps), or
   accept and report.
3. Entropy basis: sequence-level (implemented, CONTRACT §5) vs upstream's effective
   per-token renormalisation (§2/§3).
4. Most runs pick the checkpoint at 2500–3000 of 3000 steps; more steps help a little
   (0.81 → 0.83 at 100 lemmas with 6000 steps) at ~1.4× cost. Keep 3000 for the pilot.
5. Random-policy rounds do not fit the selector by default (`random_trains_selector`);
   set it to true only if matched per-round dev curves are wanted (≈ +10 min per job).

## Main-agent integration decisions (2026-10-06)

* Adopted settings: `share_embeddings: true`, `aux_copy: pool`, `num_threads: 1`, and
  `max_steps: 3000`. Entropy is computed over renormalised sequence probabilities, as the
  contract specifies.
* **Copy anchors are fixed non-inventory lemmas.** `run_acquisition(..., anchor_ids=...)`
  takes them from the auxiliary manifest: 270 eligible lemmas outside every inventory
  group, and only their source forms are used. They are identical for every fold,
  policy and pool cap. No pool candidate is ever an anchor, so the pool-cap comparison is
  not confounded by copy items (review finding, minor 1). Anchors are looked up in the
  full forms table and only their source cell is read.
* Selector dev accuracy is used both to pick the checkpoint and as the reported dev
  figure, so that figure is optimistic. The bias is the same for every policy. Selector
  performance is compared on the outer-test items in `selector_eval/`.
