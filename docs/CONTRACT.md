# Shared contract: morphology sampling, LDL outcomes and GeLaTo linkage

Owner: main agent. Components may propose changes; only the main agent edits this file.
Version 5 (2026-10-08): paradigm cell filling (PCFP) replaces source-known completion (§4–§7), LDL selector, exposure manifest, core/seed/pool roles, Grambank typology outcome (§11). Version 4 (2026-10-06): auxiliary manifest, pool_cap item column, GeLaTo human confirmation; version 3: LDL output columns, semantic seed; version 2: eligibility exclusions, collection names, variant order, epitran segmentation. Versions ≤ 4 describe pilot_v1 (commit 24390cf).

## 0. Scope and non-negotiables

* Outcomes are computed freshly from paradigms. MGN accuracies, complexity scores or
  predictions are never used as gold, features, or outcomes.
* No component chooses methods, settings or languages by looking at ancestry values.
* Raw data (`mgn_data/`, `analyses/gelato_feasibility_2026_10_01/`, `data_sources/`)
  is read-only. All new outputs go under `outputs/<experiment_id>/`.
* Components never create or alter splits. They read split manifests (§4).
* Only shown forms (§4) of training verbs reach LDL fitting. A pool candidate reaches
  the selector only as (lemma id, citation label, names of its shown cells) until it is
  acquired. Hidden-cell gold forms are read only by evaluation (§7) and by the
  gold-side mapping diagnostics after prediction.

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

## 4. Task, cells, exposure, eligibility and split manifests

### Task: `pcfp` (paradigm cell filling with known lexemes)
* **Cells.** Per unit, the declared single-word cells (`units[].cells`, citation cell
  `units[].citation_cell`). Rule: a cell is eligible if ≤ `task.cell_rule.max_multiword_share`
  of its variant-0 forms over non-derived lemmas are multiword; remaining multiword forms
  are unavailable. `splits/<unit>/cell_inventory.csv` records the counts and the decision;
  the splits stage refuses a config whose list differs from the rule's output.
* **Eligible verb.** Not in `eligibility/<unit>_derived_paradigms.csv` and, with
  `require_complete_paradigm`, every eligible cell available (variant 0, non-missing,
  single-word). `splits/<unit>/eligible_lemmas.csv`.
* **Exposure.** `splits/<unit>/exposure_manifest.csv`: `unit_id, lemma_id, group_id,
  n_cells, k_max, k, shown_cells, hidden_cells, citation_shown, n_test_cells,
  exposure_seed`. Cell lists are `|`-joined `cell_norm`s. k ~ U{1..min(max_shown,
  n_cells − 1)}; shown cells uniform without replacement; seed
  `derive(master, "exposure", unit_id, lemma_id)`. Use `morph_ldl.cv.pcfp.load_exposure`.
  The draw is identical for every policy, fold, budget, pool cap and repetition.
* **Test items.** Hidden cells minus the citation cell (`citation_cell_rule:
  exclude_from_test`). The citation cell may be shown (counted in k).
* Prediction of a test item may use only: the shown forms of all training verbs (which
  include the queried verb), identifier-keyed simulated semantics, and the target cell.

### Inventory and splits (main agent only, `morph_ldl/cv/splits.py`, `build_pcfp_manifest`)
1. Inventory: `inventory_size` eligible lemmas by whole group (`inventory` seed).
2. Core sets: inventory groups shuffled with the `split` seed (per repetition); K disjoint
   sets of exactly `core_size` lemmas taken in order.
3. Per fold k: all other inventory lemmas (other folds' core verbs included) shuffled with
   the `fold` seed and filled into `dev` (`dev_size`, 0 in pcfp_v1), `seed`, `pool`; the
   rest is `pool_overflow`.

`splits/<unit_id>/rep{r}/split_manifest.csv`: `unit_id, repetition, outer_fold, lemma_id,
group_id, role, inventory_seed, split_seed, role_rank, fold_seed`, role in {`core`, `dev`,
`seed`, `pool`, `pool_overflow`}; one row per inventory lemma per fold. `role_rank`
orders lemmas within a role (a smaller pool cap is a prefix). Read roles with
`splits.roles(manifest, repetition, fold, pool_cap)`.

`splits/<unit_id>/auxiliary_manifest.csv` (`unit_id, lemma_id, group_id, aux_role,
aux_rank, auxiliary_seed`): eligible lemmas outside every inventory group with roles
`tune_core`, `tune_extra` (LDL setting choice) and `aux_unused`.

## 5. Active selection interface (`morph_ldl/selection/ldl_acquisition.py`)

* `budget` = selected verbs including the seed (core verbs are extra and shared).
  Acquisition proceeds in batches of `batch_size` until `max(budgets)`; smaller budgets
  are prefixes. A remainder round is logged.
* The selector (LDL, `morph_ldl/ldl/selector.py` + `julia/bin/selector_server.jl`) sees
  per round: the shown forms of core + seed + acquired verbs (`rounds/r{n}/train.csv`)
  and a candidate table `rounds/r{n}/candidates.csv` with exactly `lemma_id,
  citation_cell, citation_segments, shown_cells` (citation segments = the segmented lemma
  label). `ShownOracle` holds only shown-cell rows and reveals a verb only once it is
  core, seed or acquired; every reveal is logged.
* Policies: `random` (seeded uniform draw over sorted pool ids), `low_confidence`
  (mean over shown cells of 1 − top support), `high_entropy` (mean entropy of
  softmax(supports / T)); scores averaged over `selection.semantic_seeds` seeds.
* Deterministic tie-break: score descending, then `sha256(f"{tie_seed}:{lemma_id}")`.
* Outputs under `selection/<unit_id>/rep{r}/fold{k}/<policy>@<pool_cap>/`: `order.csv`,
  `acquisition_log.csv` (+ `score_seed_sd, n_seeds`), `cell_scores.csv` (per seed,
  candidate and shown cell: supports, top prediction, `top_equals_citation`, u-scores),
  `comprehension_check.csv`, `rounds.json`, `oracle_reveals.csv`, `rounds/r{n}/`,
  `samples/budget_{B}.csv` (forms rows of the shown cells of core + first B selected
  verbs) and `samples/budget_{B}_lemmas.csv` (`lemma_id, role (core|seed|acquired),
  acquisition_rank, round, k, weight`), `selection_summary.json` (form counts per budget).

## 6. LDL interface (`morph_ldl/ldl` + `julia/`)

* Input per fitted model: training rows (`samples/budget_{B}.csv`), a query table
  `queries/…/budget_{B}/queries.csv` with `lemma_id, target_cell, item_set` and no form
  column, and a JSON config. Every queried verb must have training rows (known lexeme);
  otherwise the job fails.
* Output `predictions.csv`: `lemma_id, target_cell, prediction, prediction_segments,
  status (ok|no_candidate|error), n_candidates, top_candidates, support,
  unseen_target_features, n_train_forms_lemma, max_t`, rows in query order.
* Gold-dependent diagnostics (`mapping_quality.csv`) are computed by a separate call
  after `predictions.csv` is written.
* `semantic_seed = seeds.derive(master, "semantic", unit_id, repetition, fold)`, shared by
  all policies and budgets of a fold. The selector's seed 0 equals it; its further seeds
  are `derive(master, "selector_semantic", unit_id, repetition, fold, j)`.
* Simulated semantics give each lemma/feature the same vector in every sample.

## 7. Evaluation (main agent)

`item_predictions.csv`: `unit_id, repetition, outer_fold, policy, pool_cap, budget, model,
lemma_id, group_id, target_cell, gold_variants, prediction, status, correct,
edit_distance, norm_edit_distance, item_set (core|selected), k_shown, citation_shown,
n_test_cells, pred_equals_shown_form, pred_equals_citation`. Missing/failed predictions
are scored incorrect, with edit distance = gold length, and are counted separately.
`core` items are identical across policies and carry the primary outcome; `selected`
items are policy-dependent.

## 8. Seeds

`morph_ldl.seeds.derive(master, purpose, *keys)` = first 8 bytes of
`sha256("master|purpose|key1|key2…")` as an unsigned int (mod 2**31-1). Purposes:
`inventory`, `split`, `fold`, `random_policy`, `tie`, `semantic`, `selector_semantic`,
`exposure`, `bootstrap`, `auxiliary` (`selector_init` only for the archived Transformer).
Semantic seeds are shared by all policies and budgets within a fold/repetition; exposure
seeds depend only on (unit, lemma).

## 9. GeLaTo match status

`accepted` requires a complete review entry **and** a human `confirmed_by` +
`confirmed_date`. Agent-written reviews yield `candidate` with `proposed_status`.

## 10. Provenance

Every stage writes `stage_manifest.json`: stage, config hash, inputs with sha256,
external revisions, package versions, seeds, start/end time, git commit and dirty flag.

## 11. Typology outcome (`morph_ldl/typology`, docs/TYPOLOGY.md)

* `outputs/<exp>/typology/grambank_inflection.csv`: one row per language-level Glottocode
  (`glottocode`), with `n_present`, `n_coded`, `n_features`, `coverage`, `share` for the
  main feature set and each sensitivity set, `no_inflection`, `minimal_inflection`,
  family, macroarea, clitic flags, `has_ldl_unit`. It joins the LDL outcome on
  `glottocode` many-to-one (the LDL table has one row per design cell; filter to one
  design cell first).
* Populations appear only in `grambank_population_links.csv` (`link_basis` in exact,
  dialect_rollup, group_map_down, group_ambiguous, …, manual; §9 statuses apply).
* `feature_set.json` records the declared and the used feature set; the audit requires
  them to be equal.
* The stage opens no ancestry, Q-matrix or genetic-summary file; `stage_manifest.json`
  lists `files_opened` and the pinned source revisions (Grambank v1.0.3, Glottolog CLDF
  v5.3).
