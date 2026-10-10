# LDL protocol: paradigm cell filling with JudiLing (pcfp_v1)

Owner: main agent (`julia/`, `morph_ldl/ldl/`). Decisions are in `configs/pcfp_v1.yaml`
(`ldl:`) and `CONTRACT.md` v5. Part A describes the current protocol. Part B is the
pilot_v1 record (source binding of new verbs; code state 24390cf), kept for reference.
Its audit table (B §2) still applies to every JudiLing entry point used here.

## Part A. Known-lexeme PCFP (runner `ldl-runner-3-pcfp`)

### A.1 Model
* Training rows: the **shown** forms (variant 0) of the training verbs: core + seed +
  acquired (`samples/budget_{B}.csv`), or `tune_core` + `tune_extra` in `ldl_tune`.
* Semantics as in B §3.2: s(v, c) = L(v) + Σ V(f ∈ features(c)) + N(v, c), with
  identifier-keyed SplitMix64/Box–Muller vectors, `sem_dim` 1000, lexeme SD 4, noise SD 1,
  inflection SD tuned. Deep mode off.
* End-state mappings F = (CᵀC + λI)⁻¹CᵀS and G = (SᵀS + λI)⁻¹SᵀC (JudiLing
  `make_transform_*`, λ = 0.02).
* **Hidden cell of a known verb:** target meaning ŝ = L(v) + Σ V(features(c)) (no noise
  term: the noise belongs to an observed form). Ĉ = ŝG, decoded by `learn_paths`
  (threshold 0.05, `max_can` 10) with `data_train` = training rows, C, F, the full
  adjacency of the training cue inventory, and `max_t` = longest training form + 4.
* **Decoder options (2026-10-09).** Two decoder options are available:
  * `tolerance: true` uses `learn_paths`' tolerant mode. A path may then include up to
    `max_tolerance` n-grams whose support lies in (`tolerance_floor`, `threshold`].
  * `threshold` is configurable as before.

  `decoder: build_paths` is refused. With bigram cues its paths loop (*ssubisasubise*),
  and one Italian item took 129 s with 3 neighbours.
* No per-lemma refit is needed: every queried verb is in the training sample. Items are
  decoded in chunks of `predict_chunk` rows. `learn_paths` treats rows independently, so
  predictions do not depend on chunking or order (tested: chunk 7 vs 400, reversed
  order, one lemma alone).
* A query for a verb with no training row fails the job ("not known lexemes").
* `unseen_target_features`: features of the target cell that occur in no training cell.
  Their vectors exist, but G never saw them. They are reported per item and listed in
  `diagnostics.json`.

### A.2 Leakage audit (what can and cannot see gold)
* The cue inventory, adjacency, decoder training rows and `max_t` are built from the
  training file only, and that file holds only shown cells (audited: no (verb, cell) pair
  of any sample, selector round or tuning file is a hidden cell).
* The query table has no form column (`lemma_id, target_cell, item_set`); `read_queries`
  keeps two columns. Predictions are identical with gold withheld, present or permuted
  in extra columns (tested).
* `make_combined_cue_matrix`, `make_combined_S_matrix`, `cal_max_timestep` and
  `check_gold_path` are not used (B §2). Gold is read only by `score_mapping` after
  `predictions.csv` exists (`mapping_quality.csv`). It never feeds back.

### A.3 Exact rank-one extension by one row (`add_row`, `row_chat`)
Used only by the selector (SELECTION.md A.2). It was kept from the pilot's binding code and
generalised:
* production: P = (SᵀS + λI)⁻¹, α = sPsᵀ, G_ext = [G 0];
  Ĉ = ŝG_ext + (ŝPsᵀ)/(1+α) · (c − sG_ext);
* comprehension: F_h = [F; 0] + P_c (s − c[F; 0])ᵀ/(1 + cP_c), with
  P_c = [(CᵀC+λI)⁻¹c_known; c_novel/λ];
* the decoder's training rows, C and adjacency are extended by the row and its novel cues.

Tested against a full JudiLing refit on the augmented matrices: max |ΔĈ|, max |ΔF| <
1e-8 for bigram and trigram cues with novel cues (`test_rank_one_update_matches_full_refit`).

### A.4 Toy check and the inflection-SD grid
On the synthetic 3-class toy language (150 verbs, k ≤ 4 shown of 9 cells, 40 core verbs),
known-lexeme accuracy and the rate of copying one of the verb's shown forms were:

| cue n-gram | SD 0.4 | 1.0 | 2.0 | 4.0 | 8.0 |
|---|---|---|---|---|---|
| 2: accuracy / copy | 0.12 / 0.82 | 0.34 / 0.42 | 0.35 / 0.18 | 0.33 / 0.08 | 0.27 / 0.08 |
| 3: accuracy / copy | 0.08 / 0.92 | 0.12 / 0.81 | 0.11 / 0.79 | 0.12 / 0.79 | 0.12 / 0.80 |

With a small inflection SD the hidden cell's meaning is dominated by the lexeme vector,
so Ĉ reproduces the verb's own shown forms. This is the PCFP counterpart of the pilot's
source copying. The pilot grid {0.4, 4.0} was therefore widened to {0.4, 1, 2, 4} before
any real-data tuning. The choice is made by `ldl_tune` on auxiliary verbs only
(PROTOCOL §6). The toy numbers did not choose the setting.

### A.5 Batch runner and outputs
Unchanged from B §10, except:
* the query columns are `lemma_id, target_cell` (+ ignored extras);
* the prediction columns are `lemma_id, target_cell, prediction, prediction_segments,
  status, n_candidates, top_candidates, support, unseen_target_features,
  n_train_forms_lemma, max_t`;
* the `source_binding` and `build_paths` options were removed, and a config containing
  `source_binding` is rejected.

## Part B. pilot_v1 record: held-out-lemma paradigm completion (source binding)

Owner: LDL component (`julia/`, `morph_ldl/ldl/`). Status: Phase 1 accepted by the
main agent (2026-10-06). Phase 2 runner implemented (§10). Decisions taken by the main
agent are in `configs/pilot.yaml` (`ldl:`) and `CONTRACT.md` v3. This file documents the
LDL component and does not override them.

Reference: Heitmeier, Chuang & Baayen (2021), *Modeling morphology with Linear
Discriminative Learning: considerations and design choices*, Front. Psychol. 12:720713
(cited below as H21 with section numbers).

## 1. Software pin

| item | value | why |
|---|---|---|
| Julia | 1.12.6 (juliaup), `julia_version` recorded in `julia/Manifest.toml` | installed; JudiLing 1.0.1 allows 1.9–1.12 |
| JudiLing | registered **1.0.1** (`git-tree-sha1 67048fbe…`), `[compat] JudiLing = "=1.0.1"`, pinned in the Manifest | `src/` is byte-identical to the clone at `ca77304` (Project 1.1.0, unregistered; it only widens Julia compat to 1.13). A registered version instantiates offline from the General registry and needs no `Pkg.develop` path. |
| other | CSV 0.10.17, DataFrames 1.8.2, JSON 0.21.4, SHA, LinearAlgebra, SparseArrays, Statistics, Random (stdlib) | exact versions in `julia/Manifest.toml` |

Instantiate: `julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`.

## 2. Audit: where JudiLing can leak target information

These are the JudiLing 1.0.1 entry points that read validation (target) forms or gold
indices. The runner never calls the leaking variants. Every matrix it passes to JudiLing
is built from training forms plus the one permitted source form.

| JudiLing function | What it reads from val/gold | Leak if used naively | Runner policy |
|---|---|---|---|
| `make_combined_cue_matrix(train, val)` | val forms, to build `f2i`/`i2f` and `A` | **yes**: target n-grams enter the cue inventory and the adjacency graph, so unreachable targets become reachable | Not used. The inventory is built from training forms. Each held-out lemma adds the cues of its own source form (§4.2). |
| `make_cue_matrix(val, cue_obj_train)` | val forms → `C_val`, `gold_ind` | Throws `KeyError` on novel cues. The `gold_ind` it returns is gold. | Not used. The source cue vector is built directly, with novel cues counted. |
| `make_combined_adjacency_matrix` | val forms (via combined cue matrix) | **yes** | Not used. |
| `make_cue_matrix(train).A` | training forms only (attested transitions) | no | Available as `adjacency: attested`. |
| `make_full_adjacency_matrix(i2f)` | the inventory only | no, if `i2f` is gold-free | Default `adjacency: full`, built from the background inventory plus the source cues. An own O(k) builder is used; it was checked identical to JudiLing's. |
| `cal_max_timestep(train, val, col)` | **val form lengths** | **yes** (gold length) | Not used. `max_t = max(longest training form, source length) + max_t_margin`, in segments. The trigram path length equals the form length; the bigram path length is the form length + 1. |
| `make_S_matrix` / `make_combined_S_matrix` / `make_L_matrix` | feature *values* of val rows (not forms). Vectors come from `Random.seed!(seed)` in order of first appearance of each lexeme/feature in the data frame, and noise is drawn in row order. | No form leak, but **sample-dependent**: the same lemma gets a different vector in another sample, which violates CONTRACT §6. | Not used. Identifier-keyed vectors are used instead (§3.2). |
| `learn_paths(data_train, data_val, C_train, S_val, F, Chat_val, A, i2f, f2i; …)` | `data_val`: **only `size(data_val,1)`**, unless `check_gold_path=true`. `gold_ind` and `Shat_val` are read only when `check_gold_path=true`. `S_val` is the **semantic target of synthesis-by-analysis** (`eval_can` ranks candidates by `cor(Σ F[path,:], S_val[i,:])`). | `check_gold_path=true` is gold-dependent but only produces diagnostics. **Passing the gold `S_val` would leak** if that `S` were derived from target identity in a gold-dependent way. | `data_val = DataFrame(query_index=1:T)` (no forms). `check_gold_path=false`, `gold_ind=nothing`. `S_val` is the *constructed* target semantics `Ŝ_tgt` (§4.3). `data_train`/`C_train` are background + binding row. `target_col` is read only from `data_train`. |
| `learn_paths(data, cue_obj, …)` (wrapper) | uses `data` as both train and val, `cal_max_timestep(data)`, `gold_ind` | train-only use | Used only for seen-item diagnostics. |
| `build_paths(data_val, C_train, S_val, F, Chat_val, A, i2f, C_train_ind; …)` | `data_val`: only row count. Neighbour candidates come from `cor(Chat_val, C_train)` plus the n-grams of the top-`n_neighbors` **training** rows. | no (C_train is gold-free) | Implemented as `decoder: build_paths`, but **not viable**: it enumerates *all* paths over the neighbour n-grams up to `max_t`. With the full adjacency graph the 100-lemma smoke test did not finish 40 lemmas in 12 min (memory grew), so the run was killed. |
| `eval_SC`, `eval_SC_loose`, `accuracy_comprehension`, `eval_acc`, `eval_acc_loose`, `check_gold_path` / `Gold_Path_Info_Struct`, `iscorrect`, `isnovel` | gold vectors / gold indices | gold-dependent by design | Used only after `predictions.csv` exists (`score_mapping`, seen-item diagnostics). They are never called in the prediction path. |
| `make_transform_fac` / `make_transform_matrix` | X, Y given | no | Used for F (C→S) and G (S→C). Defaults: additive ridge, `shift = 0.02`. **This is ridge, not pinv**: `ridge_lambda: 0.0` in `pilot.yaml` resolves to 0.02 (§9). `learn_paths` also uses `shift=0.02` internally for its positional maps. |
| `eval_can` (inside both decoders) | `S_val` (see above) | see above | Runs under `Threads.@threads`. Each row is independent, and predictions were bit-identical with `-t 1` and `-t 4`. |

The prediction entry point `predict_lemma(bg, lemma_id, source_cell, source_segments,
target_cells)` has **no gold parameter**. `read_queries` keeps only the five CONTRACT §6
query columns. Gold-based diagnostics (`score_mapping`) recompute the same Ĉ in a separate
call that takes gold explicitly.

## 3. Representations (fixed across policies, budgets and folds)

### 3.1 Form cues
* Units are the CONTRACT `segments` (space-separated). JudiLing is called with
  `tokenized=true, sep_token=" ", keep_sep=true` and `start_end_token=boundary` (`#`).
  This makes cues unambiguous for multi-character `phon_custom` symbols. Word spaces (`_`)
  are ordinary symbols.
* Cues are boundary-padded n-grams (`cue_ngram`) built with `JudiLing.make_ngrams`. C is
  binary, as in JudiLing.
* The cue inventory is ordered by first appearance in the training rows, then in the
  source form (the order does not affect predictions).

### 3.2 Simulated semantics (H21 §4.2.1) with identifier-keyed vectors
`s(lemma, cell) = L(lemma) + Σ_{f ∈ features(cell)} V(f) + N(lemma, cell)`.

* `features(cell)` = the `;`-split `cell_norm`. Features are equipollent (H21 §4.2.1), for
  example `1;IND;PRS;SG` → {1, IND, PRS, SG}, and `NFIN` → {NFIN}.
* `L(lemma) ~ N(0, sem_sd_lexeme²)^d`, keyed by `(semantic_seed, "lexeme", lemma_id)`.
  `V(f) ~ N(0, sem_sd_inflection²)^d` is keyed by `(semantic_seed, "feature", f)`.
  `N ~ N(0, sem_sd_noise²)^d` is keyed by `(semantic_seed, "noise", lemma_id, cell_norm)`.
* Generator: the key `sha256(join(parts,"|"))[1:8]` (big-endian UInt64) seeds SplitMix64,
  and Box–Muller turns its output into normals. The generator is independent of the
  Julia/Random version and can be ported to Python exactly (Phase-2 test). Vectors do
  not depend on sample size, row order or which other lemmas are present. In the smoke
  test, rows shared between two different backgrounds had max |Δ| = 0.
* Deep mode is off (mean 0). JudiLing's default `isdeep=true` adds a per-vector mean drawn
  from N(0,1). H21 reports a mean |value| of 0.32 for the reduced inflection vectors, which
  matches N(0, 0.4²) without that offset (E|x| = 0.4·√(2/π) = 0.32).
* `sem_dim` is fixed (config, 1000), independent of the sample size.
* `semantic_seed` must come from `seeds.derive(master, "semantic", unit_id, repetition,
  outer_fold)`, so that it is shared by all policies and budgets of a fold (CONTRACT §8).

**Cell vector (option, 2026-10-09; off by default).** With `sem_sd_cell > 0`,
`s(lemma, cell) = L(lemma) + Σ_f V(f) + V(cell) + N(lemma, cell)`, where
`V(cell) ~ N(0, sem_sd_cell²)^d` is keyed by `(semantic_seed, "cell", cell_norm)`. It is
one vector per full feature combination, shared by all lexemes, and it sits next to the
feature vectors. It carries meaning specific to the combination, so an exponent tied to
one cell (Italian 1SG present *-o*) can be learned. The target meaning of a hidden cell
includes it. `sem_sd_cell = 0` reproduces the additive semantics exactly (no vector is
generated). The Python mirror is `morph_ldl.ldl.semantics.cell_vec`.

### 3.3 Mappings (end-state, type-based, frequency-free)
`F = (CᵀC + λI)⁻¹CᵀS` and `G = (SᵀS + λI)⁻¹SᵀC`, computed with JudiLing's
`make_transform_fac`/`make_transform_matrix`, `λ = ridge_shift = 0.02`. There is no
incremental learning and no frequency weighting.

## 4. Source-binding (wug-style) held-out-lemma protocol

The background state `(inventory, C, S, F, G, A, max_len)` is fitted once per exported
training sample (≈0.1–0.5 s). Each held-out lemma *h* is processed **independently from
that same immutable background**. The supplied source information (one form + its cell)
is counted separately from the background budget: it is one extra row, not part of `B`.

### 4.1 Steps for lemma h (default `source_binding: wug_refit`, H21 §4.3.2 steps 1–5)
1. `c_src`: cue vector of the source form on the **extended inventory** = background cues
   + the source's own novel cues (`n_source_cues_unseen`). This is permitted information.
2. `ŝ_src = c_src F` (comprehension of the source form). Novel cues have no F weights.
3. Production refit with the binding row `(ŝ_src → c_src)` on the extended inventory:
   `G_h = argmin ‖[S; ŝ_src]G − [C 0; c_src]‖² + λ‖G‖²`.
4. Target semantics: `ŝ_tgt = ŝ_src − Σ V(features(source cell)) + Σ V(features(target cell))`.
   This is H21 step 4, generalised from "−SG +PL" to the feature sets of the two cells.
5. `ĉ_tgt = ŝ_tgt G_h`. Decode with `learn_paths`, trained on background + source row
   (`data_train`, `C_train`), with the extended adjacency and gold-free `max_t`.
   Synthesis-by-analysis ranks candidates against `ŝ_tgt` using comprehension `F_h`.

### 4.2 Exact refits as rank-one updates
* **Production.** With `P = (SᵀS+λI)⁻¹`, `α = ŝPŝᵀ` and `G_ext = [G 0]`,
  `G_h = G_ext + Pŝᵀ (c_src − ŝG_ext) / (1+α)`. This is the Sherman–Morrison / recursive
  least squares identity, so it is the exact ridge solution of step 3. Only
  `ĉ_tgt = ŝ_tgt G_ext + (ŝ_tgt Pŝᵀ)/(1+α) · (c_src − ŝG_ext)` is materialised, which
  costs O(d·k) per lemma. In the smoke test it differed from a full JudiLing refit on the
  augmented matrices by ≤1.1e-10.
* **Comprehension.** Refitting F with the binding row `(c_src → ŝ_src)` is
  **algebraically a no-op** when `ŝ_src = c_src F`. The ridge normal equations
  `(CᵀC + c_srcᵀc_src + λI)F_h = CᵀS + c_srcᵀc_src F` are solved by `F_h = F`, and novel-cue
  rows stay 0. Verified: max |Δ| = 5e-12 against a full refit. So the question of whether
  comprehension is refit has no effect under `wug_refit`. The runner still applies the
  generic exact update, which matters for `lexeme_refit`.
* **Positional decoder.** `learn_paths` refits its per-timestep maps `M_t` internally on
  `C_train` = background + binding row. This is exact but is the dominant per-lemma cost
  (§7).

### 4.3 Alternatives implemented for comparison (config `source_binding`)
* `lexeme_refit`: the binding row uses the *simulated* meaning of the source,
  `s(h, source cell)`, instead of the comprehension estimate. F and G are both refit
  (exact rank-one). The target is again `s_src − V(src) + V(tgt)`. This is the standard
  LDL "seen lemma, unseen cell" set-up, in which the lemma is known in exactly one form.
  It assumes the speaker knows the lexeme's meaning. It uses no form information beyond
  the source, and the lexeme vector is keyed by `lemma_id` only.
* `none` (ablation): background G only, no binding row (`ĉ_tgt = ŝ_tgt G`).

### 4.4 Per-item accounting (never drop items)
`predictions.csv` (CONTRACT §6 columns plus `binding_fit`, `max_t`):
* `status ∈ {ok, no_candidate, error}`. `no_candidate` means `learn_paths` returned no
  complete path. `error` means the decoder threw; this is caught per lemma and the rows
  are kept.
* `n_source_cues`, `n_source_cues_unseen`, and `unseen_target_features` (target-cell
  features absent from all training cells; the vector exists but G never saw it).
* `n_candidates` counts the candidates **after** JudiLing's `max_can` truncation (JudiLing
  does not expose the pre-truncation count). `top_candidates` is a JSON list of
  `{prediction, support}`. `support` is the top candidate's synthesis-by-analysis
  correlation.
* `binding_fit = cor(ŝ_src G_h, c_src)` is not gold.
* Gold diagnostics, available only through `score_mapping` after predictions:
  `chat_gold_cor`, `n_gold_cues_outside_inventory` (the target needs a cue that is in
  neither the background nor the source), `n_gold_cues_below_threshold`, and
  `gold_path_longer_than_max_t`.

## 5. Verification run on the real model (smoke, `julia/scripts/smoke.jl`)

Fixtures come from `mgn_data/data/ita-v.csv` via `morph_ldl/ldl/smoke_fixture.py`. They
are written to `outputs/scratch_ldl/fixture_ita*`. Each fixture uses the source NFIN plus
the 8 pilot panel cells, and the background is (lemmas × 9) rows. All checks passed on
both the 60/20 and the 100/40 fixtures:

| check | result |
|---|---|
| (i) predictions identical whether the query file has no gold, real gold, or permuted gold in extra columns | identical |
| (ii) per-lemma reset: lemma alone vs in a batch, singletons in reverse order, whole batch in reverse order | identical (bitwise, including supports) |
| (ii) background refit twice → F, G, S | bitwise equal |
| (iii) rows of lemmas shared by two different background samples (30 resp. 50 lemmas) | max \|ΔS\| = 0 |
| RLS production update vs full JudiLing refit (Ĉ) | ≤ 1.1e-10 |
| comprehension refit with binding row, `wug_refit` | ≤ 5e-12 (no-op) |
| own full adjacency vs `JudiLing.make_full_adjacency_matrix` | equal |
| threads 1 vs 4 | identical `predictions.csv` |
| seen-item diagnostics (background decoded from SG) | production 1.00 / 0.999, comprehension 1.00 |

## 6. Smoke-test results (Italian verbs, NFIN → 8 cells, exploratory)

> **Status of these numbers.** The smoke fixtures are random Italian lemmas drawn from
> the whole resource, so they may overlap outer-test folds. They motivated the
> pre-declared tuning grid `ldl.tune.grid` (cue_ngram {2,3} × sem_sd_inflection
> {0.4, 4.0}), but they **did not choose the setting**. The main agent selects it on
> inner-dev lemmas only (fold-0 seed + pool prefix as background, the 80 fold-0 dev
> lemmas as held-out, both units) before any outer-test fit.

Pilot config (`cue_ngram=3, sem_sd_inflection=0.4, wug_refit, full adjacency,
threshold 0.05, sem_dim 1000`):

| background | held-out | acc | mean ED | norm ED | pred = source form | status | items with unseen src cues | s/lemma (median) |
|---|---|---|---|---|---|---|---|---|
| 60 lemmas (540 rows) | 20 (160 items) | 0.044 | 3.96 | 0.40 | 82% | 160 ok | 152/160 (mean 3.4 cues) | 0.13 |
| 100 (900 rows) | 40 (320 items) | 0.066 | 3.26 | 0.33 | 83% | 320 ok | 280/320 (mean 2.8) | 0.44 |
| 300 (2700 rows) | same 40 | 0.078 | 3.26 | — | 75% | 7 no_candidate | — | 2.4 |

Variants on the 100/40 fixture (60/20 in brackets):

| variant | acc | mean ED |
|---|---|---|
| pilot config (trigram, wug_refit, sd_infl 0.4) | 0.066 (0.044) | 3.26 |
| adjacency attested | 0.066 (0.044) | 3.40 |
| sd_infl 4.0 (JudiLing default) | 0.062 (0.044) | 4.59 |
| lexeme_refit | 0.100 (0.062) | 3.12 |
| none (no binding) | 0.000 (0.000) | 7.83 |
| **bigram** cues, wug_refit | **0.216** (0.169) | 2.88 |
| bigram, lexeme_refit | 0.247 (0.219) | 2.79 |
| bigram, sd_infl 4.0 | 0.197 (0.150) | 3.08 |
| sem_dim 200 (trigram, wug) | 0.056 / 0.075 (sd_infl 0.4 / 4) | |

Bigram curve with the same 40 held-out lemmas: 60 → 0.169 (different held-out set),
100 → 0.216, 300 → 0.263.

Per cell, 100/40, pilot config / bigram: PRS.1SG 0 / 0, PRS.3SG .05 / .35, PRS.3PL .05 / .38,
PST.1SG .05 / .28, PST.3SG 0 / 0, PST.3PL .28 / .28, COND.3SG .05 / .05, IMP.2SG .05 / .40.

Diagnosis (gold-side, after prediction; 100/40, pilot config):
* Gold was among the top-10 candidates in 6.6% of items, which equals the accuracy. The
  bottleneck is **candidate generation (Ĉ → learn_paths)**, not synthesis-by-analysis.
* Only 74/320 items have every gold cue in the extended inventory *and* above threshold
  in Ĉ. 71/320 need a cue that occurs in neither the background nor the source (stem+suffix
  junction trigrams such as `g r o`, `r o #` for *migro*). Mean cor(Ĉ, c_gold) = 0.63.
* The model mostly **copies the source form** (≈80%). With `sem_sd_inflection = 0.4`,
  ŝ_tgt differs little from ŝ_src. The source's cues stay bound to the lemma through the
  binding row, and the feature shift ΔG removes them only partly (for example, Ĉ(`r e #`)
  stays ≈0.25–0.3 for a 1SG target). With sd 4.0 copying drops, but candidates drift to
  forms of similar training lemmas (for example *frondeggiare* → *favoleggiai*), because
  ŝ_src from comprehension is a blend of neighbours' meanings.
* Bigrams cut unseen source cues from ≈2.8 to ≈0.45 per lemma and make suffix cues
  shareable across stems. This is the same change H21 needed for their wug simulation
  (§4.3.2).
* Typical errors are linguistically interpretable: *accertarebbe* for *accerterebbe*
  (a→e stem change in COND), *frondeggiarebbe*, and reflexives (*annettersi* → *mi annetto*
  is never reached because the clitic reordering requires unseen cues).

Example outputs (100/40, bigram, wug_refit; gold → top-3 candidates):
*xerocopiare* PST.3PL `xerocopiarono` → [xerocopiarono, xerocopiaronono, xerocopiare] ✓;
*accertare* COND.3SG `accerterebbe` → [accertarebbe, accertare, accertarerebbe];
*imperlare* IMP.2SG `imperla` → [imperlare, imperlarebbe, imperli];
*migrare* PRS.1SG `migro` → [migrare, migracchebbe, miemigra].

## 7. Runtime

On this Mac, Julia 1.12.6 with `-t 4`, the per-lemma cost is dominated by `learn_paths`
rebuilding and solving its positional maps on the (n+1)-row C at each timestep:

| background rows | cues (tri / bi) | background fit | per held-out lemma (8 targets), tri / bi |
|---|---|---|---|
| 540 | 596 / 216 | 0.07–0.1 s | 0.13 s / 0.07 s |
| 900 | 833 / 257 | 0.14 s | 0.44 s / 0.10 s |
| 2700 | 1416 / 307 | 0.4 s | 2.4 s / 0.43 s |

Julia start-up and compilation add ≈20–30 s per process (first lemma 2.3 s).

**Pilot extrapolation** (2 languages × 3 folds × 3 policies × budget 100 = 18 fitted
backgrounds of ≈900 rows, ~400 held-out lemmas × 8 targets each):
* trigram: 18 × 400 × 0.44 s ≈ 53 min for Italian-like forms. Finnish forms are longer
  (more cues, larger max_t), so assume up to ≈2×. That gives **≈1–1.5 h sequential** in
  one Julia process. Independent jobs can run as 2–3 processes in parallel.
* bigram: ≈12–25 min.
* budget 200 (≈1800 rows) roughly triples the trigram per-lemma cost.
* If needed, Phase 2 can apply the same exact rank-one update to the positional maps
  M_t. That would mean re-implementing `learn_paths`' path search around JudiLing's
  primitives. I propose keeping native `learn_paths` unless runtime becomes binding.

## 8. Decisions (main agent, after Phase 1) and remaining notes

Decided by the main agent:
1. **Binding.** Primary binding is `wug_refit`. `lexeme_refit` and `none` stay as
   labelled options only.
2. **Tuned settings.** `cue_ngram` and `sem_sd_inflection` are chosen on inner-dev lemmas
   via `ldl.tune.grid`, not from §6. The runner accepts per-job `overrides` for this.
3. **Fixed settings.** `ridge_shift: 0.02`, `adjacency: full`, `sem_isdeep: false`,
   `tolerance: false` (not implemented; the runner rejects `true`), and
   `semantic_seed = seeds.derive(master, "semantic", unit_id, rep, fold)`.
4. **Output columns.** Extra columns `binding_fit` and `max_t`; `top_candidates` as a
   JSON list; `n_candidates` counted after the cut; prediction format per CONTRACT §6.
5. **Eligibility.** Reflexive/clitic and multiword lemmas are excluded by
   `task.eligibility` before splitting.

Remaining notes:
* **Low ceiling.** The ceiling of this learner/protocol is low even on perfectly regular
  morphology. On the synthetic 3-class toy language (`morph_ldl/ldl/toy.py`), with 100
  background lemmas and 40 held-out lemmas, accuracy was:

  | cue_ngram | sem_sd_inflection | wug_refit | lexeme_refit |
  |---|---|---|---|
  | 2 | 0.4 | 0.36 | 0.34 |
  | 2 | 4.0 | 0.38 | 0.35 |
  | 3 | 0.4 | 0.15 | 0.12 |
  | 3 | 4.0 | 0.15 | 0.13 |

  The failure modes are source copying and neighbour-stem intrusion (§6). Outcome
  differences between units therefore partly reflect how well a linear cue/semantic
  mapping with n-gram junction cues suits a language's stem–suffix structure. This is an
  interpretation caveat for the study.
* `build_paths` is not viable (combinatorial blow-up) and is not used.

## 9. Config/contract changes

All Phase-1 proposals were accepted and applied by the main agent (pilot.yaml `ldl:`,
CONTRACT v3 §6). The runner still accepts the legacy key `ridge_lambda` (0 → 0.02).
`ridge_shift` takes precedence.

## 10. Phase-2 batch runner

### 10.1 Interface (`morph_ldl/ldl/runner.py`)
```python
run_ldl_jobs(jobs, cfg, n_procs=None, force=False, run_dir=None, raise_on_error=True) -> list[Path]
score_mapping_jobs(jobs, gold_csv_by_job, cfg, n_procs=None, run_dir=None) -> list[Path]
write_gold_csv(forms, queries, path)           # evaluation-side helper only
```
A job is `{train_csv, queries_csv, out_dir, unit_id, repetition, fold, overrides}`. The
resolved model config is `cfg["ldl"]` minus orchestration keys (`tune`, `n_procs`), plus
`overrides`, plus the contract `semantic_seed`. `semantic_seed` cannot be overridden.

### 10.2 Outputs per job (`out_dir/`)
* `predictions.csv`: CONTRACT §6 columns + `binding_fit`, `max_t`. Written atomically.
* `diagnostics.json` contains:
  - runner, JudiLing and Julia versions; thread counts
  - training rows (raw and after the variant-0 / non-missing filter) and training lemmas
  - cue-inventory size, `sem_dim`, `cue_ngram`, `semantic_seed`, and the remaining model
    settings
  - status counts; items with unseen source cues and the total number of unseen source
    cues; items with unseen target features
  - background fit time, per-lemma runtime (median/mean/max/first/total), job time
  - **seen-item diagnostics**: training comprehension accuracy (`JudiLing.eval_SC` with
    homograph handling), training production accuracy (`learn_paths` on SG), and mean
    cor(SG, C)
  - `semantic_probe`: the first 5 dimensions of the first 3 training rows' S, used for
    the cross-sample and Python-parity tests
* `job_config.json`: the resolved config, inputs and their sha256, runner version, Julia
  Manifest hash and `job_hash`. It is **written last** and is the completion marker.
* `error.json`: written only if the job failed.

### 10.3 Resumability and parallelism
* **Skipping finished jobs.** A job is skipped when `predictions.csv`, `diagnostics.json`
  and a `job_config.json` with the same `job_hash` exist. The hash covers the resolved
  config, input file contents, runner version and Julia Manifest. A stale marker is
  deleted before a job is re-run.
* **Processes and threads.** Pending jobs are dealt round-robin into `n_procs` shards
  (default `ldl.n_procs`). Each shard runs in one Julia process
  (`julia/bin/run_jobs.jl`), which amortises the ≈20 s compilation. Each process gets
  `threads = min(limits.julia_threads, limits.max_threads // n_procs)` for both Julia
  threads and BLAS threads.
* **Logs and failures.** Manifests and logs go to `<common out_dir parent>/_ldl_runs/<timestamp>/`.
  A failing job writes `error.json` without stopping its shard. Inside a job, a failing
  lemma yields `status=error` rows; items are never dropped.

### 10.4 Gold-isolated scoring
`score_mapping_jobs` runs only after a finished prediction with a matching hash. It
refits the same background (deterministic) and recomputes Ĉ per lemma. It then writes
`mapping_quality.csv` with these columns:
* `chat_gold_cor`, for the gold variant whose cue vector correlates best with Ĉ
* `gold_variant_idx`
* `n_gold_cues`, `n_gold_cues_outside_inventory`, `n_gold_cues_below_threshold`
* `gold_reachable`, `gold_path_longer_than_max_t`
* `gold_in_top_candidates`, `gold_rank_in_candidates`

It also writes a `mapping_quality.json` marker (job hash + gold sha256). Gold CSV columns
are `lemma_id, target_cell, gold_variants`, with segment strings joined by `" || "`. The
scoring step never writes to `predictions.csv`.

### 10.5 Tests (`.venv/bin/python -m pytest tests/test_ldl_*.py -q`)
* `test_ldl_runner_unit.py` (fast) covers config/seed resolution, hash sensitivity,
  skip/rerun with stale-marker removal, refusing to score before prediction, the thread
  cap, the gold CSV helper, and properties of the semantics port.
* `test_ldl_julia.py` (`@slow`, real JudiLing on synthetic paradigms, one 2-process run)
  covers:
  - contract columns, with no dropped items
  - variant-1 and missing training rows being filtered
  - **decoder isolation**: identical predictions with gold withheld, present or permuted
    in the query file
  - **reset/order invariance**: reversed lemma order and a single lemma alone
  - **gold-free max_t**: a held-out lemma with 43-symbol gold targets gets the same
    max_t formula as the others
  - **unseen-cue accounting**, recomputed independently in Python
  - **semantic stability** across two samples, plus Julia↔Python parity of S rows
  - generator parity on 1001 dimensions (≤1e-13)
  - **resumability** (no relaunch)
  - scoring after prediction, with the predictions file untouched

### 10.6 Real-data check (ita.V.orth.mgn, rep 0, fold 0; no outer-test lemmas)
Background: the fold-0 seed plus the first 80 pool lemmas (100 lemmas, 900 rows). Held-out:
the first 10 **dev** lemmas (80 items). Queries come from `morph_ldl.cv.queries.build_queries`.
Settings are the config defaults, with `cue_ngram` overridden.
Outputs are in `outputs/scratch_ldl/realcheck_ita_fold0/`.

| cue_ngram | acc | mean ED | copies source | cor(Ĉ, gold) | gold in top-10 | gold reachable | cues | per-lemma median (s) | job (s) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.438 | 1.52 | 21% | 0.87 | 0.45 | 71% | 223 | 0.085 | 7.5 |
| 3 | 0.138 | 2.29 | 66% | 0.64 | 0.14 | 39% | 733 | 0.31 | 13.9 |

Other results from this check:
* All items had status ok.
* Training comprehension accuracy was 1.0, and training production accuracy 0.98 (n=2)
  and 0.99 (n=3).
* The first lemma per process costs ≈1.7–2 s (JIT).
* Wall time for both jobs in 2 parallel processes, including compilation: 20 s for
  prediction and 9 s for scoring.

These are 10 dev lemmas. They are a plumbing check, not the tuning run.

## 11. Main-agent integration decisions (2026-10-06)

* **Setting choice moved to auxiliary lemmas.** The setting was first chosen on fold 0's
  dev and pool lemmas, which are test lemmas in folds 1 and 2. That choice was discarded
  before any outer-test fit. Tuning now uses 100 background lemmas plus 80 held-out
  lemmas, all eligible lemmas outside the inventory. It again chose `cue_ngram=2,
  sem_sd_inflection=0.4`, best in both units: Italian 0.373, Finnish 0.245 held-out
  accuracy. Re-tuning is refused once outer-test predictions exist.
* **Fixed Julia thread count.** It now depends on the configured `ldl.n_procs`, not on
  how many jobs are pending, so a resumed run uses the same threads.
* **Training rows by mode.** `training_mode: panel` trains on `budget_{B}.csv`.
  `all_cells` trains on `budget_{B}_allforms.csv`.
