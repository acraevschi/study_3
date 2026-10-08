# Shared contract: morphology sampling, LDL outcomes and GeLaTo linkage

Owner: main agent. Components may propose changes; only the main agent edits this file.
Version 4 (2026-10-06): auxiliary manifest, pool_cap item column, GeLaTo human confirmation; version 3: LDL output columns, semantic seed; version 2: eligibility exclusions, collection names, variant order, epitran segmentation.

## 0. Scope and non-negotiables

* Outcomes are computed freshly from paradigms. MGN accuracies, complexity scores or
  predictions are never used as gold, features, or outcomes.
* No component chooses methods, settings or languages by looking at ancestry values.
* Raw data (`mgn_data/`, `analyses/gelato_feasibility_2026_10_01/`, `data_sources/`)
  is read-only. All new outputs go under `outputs/<experiment_id>/`.
* Components never create or alter splits. They read split manifests (§4).
* Only the declared source anchor of a held-out lemma (one form + its cell) may reach
  prediction code. Gold targets of test, dev-for-scoring, and candidate-pool lemmas are
  read only by (a) the oracle reveal after selection (§5) and (b) evaluation (§7).

## 1. Identifiers

| Name | Definition | Example |
|---|---|---|
| `resource_id` | `<collection>:<file stem>`; collections `mgn_data` (MGN data/), `mgn_custom` (MGN data-custom/), `unimorph`, `paralex` | `mgn_data:ita-v`, `mgn_custom:french-v` |
| `resource_version` | git commit, release tag, or `sha256:<first 16 hex>` of the source file | `sha256:3f2a…` |
| `variety_id` | canonical ISO 639-3 of the variety the resource represents, plus optional `-<subtag>` when a resource documents a sub-variety | `ita`, `fin`, `nor-bokmal` |
| `representation` | `orth` (standard orthography), `ipa_epitran` (automatic G2P), `phon_custom` (resource-native phonemic transcription) | `orth` |
| `unit_id` | morphology outcome unit: `{variety_id}.{pos}.{representation}.{resource_slug}` | `ita.V.orth.mgn` |
| `lemma_id` | `{resource_id}::{lemma_label}` with a `#k` suffix only if the same label has several distinct paradigms in one resource (k = 1-based order of appearance) | `mgn_data:ita-v::lacrimare` |
| `group_id` | leakage group, §3 | `ita.V::g000123` |
| `cell_norm` | UniMorph features, upper-case, de-duplicated, sorted alphabetically, `;`-joined, POS removed | `1;IND;PRS;SG` |

POS values: `N`, `V`, `ADJ`.

## 2. Long-form record table (`forms.csv`)

Produced by the data stage, one row per (lemma, cell, variant). UTF-8 CSV, NFC.

| column | type | notes |
|---|---|---|
| `unit_id`, `resource_id`, `resource_version`, `variety_id`, `iso639_3`, `glottocode`, `pos`, `representation` | str | |
| `lemma_id`, `lemma_label`, `group_id` | str | `lemma_label` exactly as in source |
| `cell_orig` | str | original label |
| `cell_norm` | str | §1; empty if unparseable (row kept, flagged) |
| `form_orig` | str | original cell content, all variants, unchanged |
| `variant_idx` | int | 0-based order within `form_orig` as stored by the source (MGN stores variants sorted alphabetically, so variant 0 is not a preferred form) |
| `n_variants` | int | |
| `form` | str | this variant, NFC, stripped |
| `segments` | str | space-separated symbols. `orth`: one symbol per Unicode character (combining marks attached); `ipa_epitran`: same rule, with length and secondary-articulation modifiers attached to the preceding symbol. `phon_custom`: the source's own tokens. Word spaces inside a form become `_`. `#` is reserved as boundary and must not occur |
| `is_missing` | bool | source marked the cell missing (`NA`, empty). `form`/`segments` empty |
| `source_file`, `source_row` | str/int | provenance |

The data stage never discards variants or missing markers silently; it records them.

## 3. Leakage groups

Within one `variety_id` + `pos` (across all resources and representations of that
variety), lemmas are joined into one `group_id` if they share
(a) the same normalized lemma label (NFC, case-folded, whitespace-collapsed), or
(b) an identical source-cell `form` in the same representation.
Groups are connected components of that relation. All splitting is by group.
Alternative resources for the same variety are measurement variants: one variety
contributes one independent morphology observation per POS/task, and a group may never
straddle train/dev/pool/test.

## 4. Task, eligibility, inventory and split manifests

### Task (primary): `source_known_completion`
* One predefined source cell per unit (`task.source_cell`), and a bounded target panel
  (`task.panel`, abstract slot -> `cell_norm` per unit; default 8 targets).
* Prediction of a held-out lemma's target cell may use only: the lemma's source `form`
  (variant 0), the source cell, the target cell, and the fitted background state.
* Eligible lemma: non-missing source cell and all panel cells non-missing; not a
  derived paradigm listed by the data stage (`eligibility/<unit>_derived_paradigms.csv`:
  e.g. Italian pronominal/clitic verbs whose forms are clitic + base-verb forms, Finnish
  multiword idioms) when `task.eligibility.exclude_derived_paradigms`; and no task form
  containing a word space when `task.eligibility.exclude_multiword_task_forms`.
* Variants: training and source anchors use `variant_idx == 0`; evaluation accepts any
  variant of the gold cell. Variant rates are reported.
* `all_cells` training mode (optional) adds every eligible non-missing cell of the
  selected lemmas to training. Its different exposure must be reported.

### Inventory and splits (main agent only, `morph_ldl/cv/splits.py`)
1. Inventory: from eligible lemmas, sample `inventory_size` groups' lemmas at random
   (`inventory` seed). Declared before any acquisition; recorded.
2. Outer folds: inventory groups are shuffled (`split` seed, per repetition) and dealt
   into K folds.
3. Per outer fold: test = fold k. From the rest (`fold` seed): `dev_size` dev lemmas,
   `seed_size` seed lemmas, then `pool_cap` pool lemmas; leftovers are `pool_overflow`
   (unused). All by group.

`splits/<unit_id>/rep{r}/split_manifest.csv`:
`unit_id, repetition, outer_fold, lemma_id, group_id, role, inventory_seed, split_seed,
role_rank, fold_seed` with role in {`test`, `dev`, `seed`, `pool`, `pool_overflow`}; one
row per lemma per fold. `role_rank` orders lemmas within a role; a smaller pool cap
uses the first lemmas of the pool (nested pools for the pool-size sensitivity).
Use `morph_ldl.cv.splits.roles(manifest, repetition, fold, pool_cap)` to read roles.

`splits/<unit_id>/auxiliary_manifest.csv` (`unit_id, lemma_id, group_id, aux_role,
aux_rank, auxiliary_seed`): eligible lemmas outside every inventory group, with roles
`tune_background`, `tune_heldout` (LDL setting choice) and `copy_anchor` (selector
auxiliary copy items; source forms only). Auxiliary lemmas never enter any fold.

## 5. Active selection interface (`morph_ldl/selection`)

* `budget` = total training lemmas including the seed. Acquisition proceeds from the seed
  in batches of `batch_size` until `max(budgets)`; smaller budgets are prefixes of the
  same trajectory. If `budget - seed_size` is not a multiple of `batch_size`, the last
  round acquires the remainder (logged).
* The scorer sees only `CandidateQuery(lemma_id, source_form, source_cell,
  target_cells)`. Gold candidate targets live in an `Oracle` that reveals full panel
  forms only for lemmas already selected; every reveal is logged.
* Policies: `random` (seeded, matched seed/pool/dev/budget), `low_confidence`
  (highest lemma mean length-normalized surprisal of the top beam hypothesis),
  `high_entropy` (mean per-cell entropy over renormalized beam hypotheses with
  p_i >= 0.05, as in the paper), optional `oracle_*` (labelled separately, never pooled).
* Deterministic tie-break: score descending, then `sha256(f"{tie_seed}:{lemma_id}")`.
* Outputs under `selection/<unit_id>/rep{r}/fold{k}/<policy>/`:
  `order.csv` (`lemma_id, acquisition_rank, round, lemma_score, score_name`),
  `acquisition_log.csv` (every scored candidate per round: `round, lemma_id,
  lemma_score, n_cells_scored, n_nonfinite, rank_in_round, selected`),
  `cell_scores.csv` (`round, lemma_id, target_cell, hyp_rank, hyp, logprob_sum,
  hyp_len, surprisal_norm, prob_renorm`),
  `rounds.json` (per round: n_train_lemmas, n_train_examples, dev accuracy, runtime,
  model hash), `oracle_reveals.csv`,
  `samples/budget_{B}.csv` (forms.csv rows for source + panel cells of the first B
  lemmas) and `samples/budget_{B}_allforms.csv` (all eligible original rows of them).

## 6. LDL interface (`morph_ldl/ldl` + `julia/`)

* Input per fitted model: training rows (`samples/budget_{B}.csv`), test queries
  `test_queries.csv` (`lemma_id, source_cell, source_form, source_segments,
  target_cell`) with NO gold target column, and a JSON config.
* Output `predictions.csv`: `lemma_id, target_cell, prediction, prediction_segments,
  status (ok|no_candidate|error), n_candidates, top_candidates, support,
  n_source_cues, n_source_cues_unseen, unseen_target_features`.
* Gold-dependent diagnostics (cue-vector correlation of Ĉ with the gold target) are
  computed by a separate call after `predictions.csv` is written.
* Extra prediction columns: `binding_fit`, `max_t`. `top_candidates` is a JSON list of
  `{prediction, support}` (top `max_can`); `n_candidates` counts after that cut.
  `prediction` = segments joined, `_` -> space; scoring always uses `prediction_segments`.
* `semantic_seed = seeds.derive(master, "semantic", unit_id, repetition, fold)`.
* Per-held-out-lemma state is reset to the same fitted background.
* Simulated semantics must give each lemma/feature the same vector in every sample
  (derive from `semantic_seed` and the identifier, not from row order).

## 7. Evaluation (main agent)

`item_predictions.csv`: `unit_id, repetition, outer_fold, policy, pool_cap, budget, model
(ldl|selector), lemma_id, group_id, target_cell, gold_variants, prediction, status,
correct, edit_distance, norm_edit_distance`. Missing/failed predictions are scored
incorrect, with edit distance = gold length; they are also counted separately.

## 8. Seeds

`morph_ldl.seeds.derive(master, purpose, *keys)` = first 8 bytes of
`sha256("master|purpose|key1|key2…")` as an unsigned int (mod 2**31-1). Purposes:
`inventory`, `split`, `fold`, `selector_init`, `random_policy`, `tie`,
`semantic`, `bootstrap`. Selector init and semantic seeds are shared by all policies and
budgets within a fold/repetition.

## 9. GeLaTo match status

`accepted` requires a complete review entry **and** a human `confirmed_by` +
`confirmed_date`. Agent-written reviews yield `candidate` with `proposed_status`.

## 10. Provenance

Every stage writes `stage_manifest.json`: stage, config hash, inputs with sha256,
external revisions, package versions, seeds, start/end time, git commit and dirty flag.
