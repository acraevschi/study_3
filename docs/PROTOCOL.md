# Experimental protocol (declared before selection and outer-test evaluation)

Owner: main agent. Status: **declared for pcfp_v1** (`configs/pcfp_v1.yaml`, 2026-10-08).
Component details: [DATA.md](DATA.md), [SELECTION.md](SELECTION.md),
[LDL_PROTOCOL.md](LDL_PROTOCOL.md), [TYPOLOGY.md](TYPOLOGY.md); interfaces:
[CONTRACT.md](CONTRACT.md).

The earlier source-known new-verb task (`pilot_v1`, [REPORT.md](REPORT.md)) is replaced and
is not run again. Its outputs in `outputs/pilot_v1/` are kept unchanged as the archived
record. The code that produced them is commit `24390cf` (merge of the 2026-10-08
reorganisation commit `db9156f`).

## 0. Two outcomes

| | Primary (high power) | Secondary (finer) |
|---|---|---|
| Outcome | Grambank inflectional extent (`typology` stage) | LDL predictability under paradigm cell filling (this protocol, §1–§7) |
| Languages | every GeLaTo-linked language in Grambank with ≥ 60% of the 12 inflectional categories coded (pcfp_v2; each category merges its dependent Grambank features by OR, TYPOLOGY.md §3; pcfp_v1: 35 raw features), including isolating languages | inflecting languages with paradigm data; here the two pilot units |
| Unit | one row per language-level Glottocode | one row per unit × item set × policy × pool cap × budget |
| Key | `glottocode` | `glottocode` (same key; many LDL design cells per language, so filter to one design cell before joining) |

LDL predictability is defined only for languages that inflect. It is stated as
conditional on inflection, and isolating languages never get an LDL score. The two
outcome tables share the language-level Glottocode key (the LDL table has one row per design cell). The Grambank outcome and its
planned analysis are specified in [TYPOLOGY.md](TYPOLOGY.md) (§9 there; summarised in §9
below).

## 1. Estimand (LDL outcome)

For a morphology unit (variety × POS × representation × resource), a sampling policy, a
candidate-pool size and a training budget of B selected verbs: the expected exact-form
accuracy (and edit distance) of an end-state LDL model fitted on the shown forms of the
training verbs when it **fills the hidden cells of known verbs** (paradigm cell filling,
PCFP). The primary items are the hidden cells of a fixed set of core verbs, identical for
every policy.

Lower accuracy or larger edit distance means greater difficulty *for this learner under
this exposure and sampling protocol*. The estimate is conditional on the declared cell
inventory, exposure rule, representation and policy. It is not a language-intrinsic
quantity.

## 2. Task

* **Cells.** The single-word cells of each unit (rule declared in the config): a cell is
  eligible if at most 50% of its variant-0 forms (over non-derived lemmas) are multiword;
  any remaining multiword form is unavailable. Italian MGN: 48 of 48 cells. Finnish MGN:
  35 of 137 cells (102 are mainly periphrastic: negatives, perfects, pluperfects). The
  lists are written into `configs/pcfp_v1.yaml`; the `splits` stage refuses to run if the
  data disagree. Counts are in `splits/<unit>/cell_inventory.csv`.
* **Eligible verb.** Not a derived paradigm (Italian pronominal/clitic verbs, Finnish
  idioms; `eligibility/<unit>_derived_paradigms.csv`) and a complete single-word paradigm:
  every eligible cell non-missing and single-word at variant 0. A missing cell or a
  stray multiword form makes the verb ineligible. Italian: 7,767 verbs (30 of 7,797 lost to the
  completeness rule); Finnish: 7,597.
* **Exposure.** Each verb shows k forms, k ~ Uniform{1, …, min(7, n_cells − 1)}, the same
  cap in every language. The k shown cells are drawn uniformly without replacement from
  the verb's eligible cells; all other eligible cells are hidden. The draw is made once
  per verb from `seeds.derive(master, "exposure", unit_id, lemma_id)` and is identical for
  every policy, fold, budget, pool cap and repetition (`splits/<unit>/exposure_manifest.csv`).
* **Citation cell** (NFIN in both units; the lemma label equals the NFIN form for 99.9%
  of Italian and 100% of Finnish verbs). It may be drawn as a shown cell and then counts
  in k. **It is never a test item** (`citation_cell_rule: exclude_from_test`), because the
  LDL selector treats the lemma label as a known lexical label (§5). Every policy is
  therefore scored on identical items.
* **Test items.** The hidden non-citation cells of a verb: 40–47 per Italian verb and
  27–34 per Finnish verb.
* **Variants.** Training uses variant 0; evaluation accepts any documented variant
  (both units have 0% variants).

## 3. Units

`ita.V.orth.mgn` and `fin.V.orth.mgn` (MGN `data/` files, built from UniMorph without
transliteration). Same ingest set and group ids as pilot_v1.

## 4. Cross-validation design (core + selected)

Per unit: an inventory of 1,200 eligible verbs (by leakage group, `inventory` seed),
one repetition, K = 3 outer folds.

* **Core verbs.** The inventory groups are shuffled with the `split` seed and K disjoint
  core sets of C = 100 verbs (whole groups) are taken in order. Fold k's core verbs are in
  *every* learner's training through their shown forms. Their hidden non-citation cells
  are the primary test items (≈ 4,300 Italian and ≈ 3,000 Finnish items per fold),
  identical for every policy, budget and pool cap. Core sets are disjoint across folds.
* **Selected verbs.** From the other 1,100 inventory verbs (shuffled with the `fold`
  seed): seed 20 (shared by all policies), pool 500 (nested 250 prefix for the pool-cap
  sensitivity run), overflow 580 (unused). The budget B counts selected verbs including
  the seed (B = 100; core verbs are extra). Selected verbs contribute only their shown
  forms.
* **No dev role.** LDL is fitted in closed form, so it needs no early stopping. The dev
  capacity of pilot_v1 goes to overflow. No verb is ever used for monitoring.
* **Auxiliary verbs** (outside every inventory group, `auxiliary` seed): 100 `tune_core`
  verbs (shown forms in training, hidden cells tested) and 100 `tune_extra` verbs (shown
  forms only) for the LDL setting choice (§6). They never enter any fold.
* Each training sample contains 100 core + 100 selected verbs: about 800 shown forms
  (k̄ ≈ 4). The number of forms is reported for every sample
  (`eval/sample_composition.csv`).

## 5. Selection (LDL selector)

Seed 20, batch 20, budget 100 (4 acquisition rounds); policies: matched `random`,
`low_confidence`, `high_entropy`. **The selection model is the evaluated model**: each
round a JudiLing LDL model with the frozen `ldl_tune` settings is fitted on the shown
forms of core + seed + acquired verbs. The Transformer selector of pilot_v1 is not used.

* **What LDL knows about a candidate:** its lemma label (citation form) and its simulated
  lexeme vector. A candidate is scored as a verb seen through one form. Its citation row is
  the citation label's cues paired with its simulated semantics for the citation cell
  (lexeme + citation features + noise). The row is added to the round's model by an
  exact rank-one update. The candidate's pre-drawn shown cells are then decoded from
  lexeme + cell features, and the row is removed. No other form of a candidate enters a
  fit, a cue inventory or a candidate list before the candidate is acquired. The
  evaluated LDL never contains a pool candidate's citation row.
* **Scores** (higher = more uncertain = selected first; declared before any run). For
  each shown cell, s₁ ≥ s₂ ≥ … are the synthesis-by-analysis supports of the top
  `max_can` = 10 decoded candidates.
  - `low_confidence`: u = 1 − s₁.
  - `high_entropy`: H = −Σ wᵢ log wᵢ with w = softmax(s / T), T = 0.02.
    *Pre-run deviation (user decision, 2026-10-08):* T was declared as 0.1. The
    independent review found that at 0.1 the entropy tracks log(number of decoded
    candidates) (r = 0.94 in the smoke run, with 60% of cells at the cap of 10) and is
    nearly uncorrelated with 1 − s₁, because the median top-two support gap is 0.04.
    T = 0.02 (half that gap; r = 0.54) was set before any real selection, from smoke
    score distributions only. No accuracy and no outer-test data were used.
  - A cell with no candidate (or a decoder error, logged separately) counts as s₁ = −1
    (u = 2) and H = log 10.
  - Lemma score = mean over its shown cells, then mean over 3 semantic seeds (seed 0 =
    the evaluated LDL's seed; seeds 1–2 = `selector_semantic`).
* **Ties and order.** Score descending, then `sha256(tie_seed:lemma_id)`. Smaller
  budgets are prefixes of the same trajectory.
* **Comprehension-side check** (logged, never used for selection): per candidate and
  round, cor(c·F, s) and ‖c·F − s‖/‖s‖ for the citation form, and the share of
  citation cues unseen in training. These are reported with their rank correlations with
  the selector scores and with citation length, final letters and the inflection-class
  proxy.
* **Stop condition** (applied to the smoke run before the real run): if the primary
  scores are degenerate, report it and ask the user before choosing another approach.
  Degenerate means near-constant (round CV < 0.01), dominated by copying the citation
  form (top candidate = citation in > 90% of scored cells), or explained by citation
  length and final letters alone (R² > 0.9).

## 6. Downstream LDL (known lexemes)

End-state (frequency-free) JudiLing 1.0.1 fit on the shown forms of the training sample.
Letter n-gram cues. Simulated semantics s(v, c) = L(v) + Σ V(features(c)) + N(v, c), with
identifier-keyed vectors (LDL_PROTOCOL §3). A hidden cell's target meaning is
L(v) + Σ V(features(c)); Ĉ = Ŝ G, decoded with `learn_paths`. Cue inventory, adjacency,
decoder training rows and `max_t` (longest training form + 4) come from training forms
only. No per-lemma refit is needed, because every queried verb is in the training
sample. Features that never occur in training are reported per item
(`unseen_target_features`). The `wug_refit` source binding of pilot_v1 is removed.

**Setting choice (`ldl_tune`, before selection).** The tuning fits use their own semantic
seed (`derive(master, "semantic", unit, 0, -1)`), which no outer fold uses. cue n-gram {2, 3} × inflection SD
{0.4, 1, 2, 4} is fitted on the auxiliary verbs (shown forms of 100 + 100) and scored on
the hidden cells of the 100 `tune_core` verbs. SD 2.0 was added to the grid after the toy
check in LDL_PROTOCOL §6 and before any real-data tuning. The setting with the highest mean
held-out accuracy over the two units is applied everywhere (ties: lower mean edit
distance). If the chosen inflection SD is at a grid edge, the grid is extended once
(0.1 / 10.0; n-gram + 1 at the upper edge) and the stage re-runs. The stage refuses to
run once selection or outer-test LDL outputs exist. The LDL selector uses these settings,
so `ldl_tune` runs before `select`.

## 7. Metrics and uncertainty

* Exact-form accuracy (any variant), micro (item) and macro (mean of verb means), with
  denominators. Levenshtein distance over segments and its length-normalised form.
  Decoder failures and missing predictions are scored incorrect, with distance = gold
  length, and are counted separately.
* Breakdowns: per cell, per k (number of shown forms), the number of test cells per
  verb, and copy rates (prediction = one of the verb's shown forms; = its citation form).
* LDL mapping quality (Ĉ–gold cue correlation, gold reachability, gold in the top
  candidates) is reported separately from decoded production.
* Primary items: core verbs' hidden cells. Secondary items: hidden cells of the selected
  verbs (seed + acquired). These are policy-dependent and labelled so (`item_set`).
* 95% percentile intervals from a leakage-group cluster bootstrap of the pooled
  out-of-fold predictions (2,000 draws, `bootstrap` seed per design cell). Paired
  active − random differences resample the same verbs for both policies on identical
  core items. These intervals are **conditional on the fitted fold models**. Fold
  variability is descriptive; fold SD/√K is not used.

## 8. What is not done here

No admixture–morphology association is fitted. No setting or language is chosen by
looking at ancestry values. GeLaTo populations are linked, not aggregated, and accepted
links need a human `confirmed_by` + `confirmed_date`. The admixture exposure variable is
undefined and was not looked at.

## 9. Planned analysis (documented, not fitted)

* **Grambank extent:** beta-binomial on (`n_present`, `n_coded`) per language (main set,
  ≥ 60% coverage, non-proxy links). Add a hurdle or zero-inflation part only if
  posterior predictive checks show excess zeros (4–6 languages are at or near zero).
* **LDL predictability:** binomial at the item level with language random effects, or
  beta-binomial per language, inflecting languages only.
* **Confounding:** the zero or near-zero languages (Khmer, Vietnamese, Naxi, She, Thai,
  Mandarin, Northern Tujia) are almost all in Mainland Southeast Asia and China and share
  East Asian ancestry. Any no-inflection part is weakly identified once area is
  controlled. It is descriptive, not a main test.
* The admixture exposure variable is still undefined; nothing is fitted with it.
