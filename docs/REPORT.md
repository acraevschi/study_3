# Pilot report: active vs random sampling for LDL morphology outcomes (pilot_v1)

Date: 2026-10-06. Config: `configs/pilot.yaml` (hash a559df1d7b61a544). The outputs are
in `outputs/pilot_v1/`. This pilot is complete in the brief's sense: real active selection
and real JudiLing evaluation ran inside every outer fold. It is still a **bounded pilot**:
2 languages, 1 POS, 3 folds, 1 repetition and budget 100. No admixture–morphology
association was fitted.

## 1. What was built

| Component | Where | Status |
|---|---|---|
| Resource adapters (MGN wide and long, UniMorph, Paralex interface), registry, ISO/Glottolog crosswalk (`languages-of-the-world` d319631 + project overrides) | `morph_ldl/data/`, `outputs/pilot_v1/registry/` | complete; 135 resources registered |
| Eligibility audit, leakage groups, derived-paradigm detection | `morph_ldl/data/`, `outputs/pilot_v1/eligibility/` | complete |
| GeLaTo crosswalk and linkage (no ancestry values copied) | `outputs/pilot_v1/gelato/`, `outcomes/population_links.csv` | complete; 236 rows |
| Inventory, grouped outer folds, inner roles, auxiliary set | `morph_ldl/cv/splits.py`, `outputs/pilot_v1/splits/` | complete |
| Selector (character Transformer), scores, acquisition loop, matched random | `morph_ldl/selection/` | complete; ALmorphinfl 3caf0d0 audited, method kept and code rewritten |
| JudiLing runner (=1.0.1), wug source binding, gold-isolated diagnostics | `julia/`, `morph_ldl/ldl/` | complete |
| Cross-validation orchestration, gold-free query gate, evaluation, cluster bootstrap, outcome and linkage tables, artifact audit, provenance | `morph_ldl/cv/`, `morph_ldl/provenance.py`, `morph_ldl/cli.py` | complete |
| Tests | `tests/` | 97 pass, including real-Julia tests |

## 2. Declared design

The design was fixed before any outer-test evaluation; [PROTOCOL.md](PROTOCOL.md) gives
full detail.

* **Units.** `ita.V.orth.mgn` and `fin.V.orth.mgn`, from MGN `data/`.
  * Italian is identical to UniMorph `ita` at fa2cc6c for all 479,579 forms.
  * Finnish comes from an older UniMorph release; 99.55% of the forms both releases
    share agree.
  * Both are orthographic, untransliterated and have no variants in the task cells.
* **Task.** Source NFIN → 8 target cells: PRS 1SG/3SG/3PL, PST 1SG/3SG/3PL, COND 3SG and
  IMP 2SG. Italian PST is the *passato remoto*.
* **Exposure.** Each lemma contributes 9 forms, so 100 lemmas give 900 training forms
  in every unit.
* **Eligibility exclusions,** declared for both units:
  * Derived paradigms are excluded: 2,208 Italian pronominal/clitic verbs such as
    *abbandonarsi* → *mi abbandonai*, and 127 Finnish idiom or multiword-label entries.
  * Lemmas with any multiword task form are excluded.
  * This leaves 7,767 Italian and 7,597 Finnish eligible lemmas.
* **Cross-validation, per unit:**
  * The inventory is 1,200 lemmas, sampled by leakage group, with 3 outer folds.
  * Each fold has 400 test lemmas, 80 dev, 20 seed and a 500-lemma pool. A nested
    250-lemma pool is used for the sensitivity check.
  * Acquisition runs in batches of 20, so budget 100 takes 4 acquisition rounds.
* **Auxiliary set.** Eligible lemmas outside the inventory are never tested. From them:
  * 100 background + 80 held-out lemmas were used to choose the LDL settings;
  * 270 source forms serve as the selector's copy anchors.
* **LDL settings.** End-state fitting with simulated semantics (dimension 1000, lexeme SD
  4, inflection SD 0.4, noise SD 1), bigram cues, ridge shift 0.02, `learn_paths` with
  threshold 0.05, and `wug_refit` source binding with a reset for every held-out lemma.
* **Uncertainty.** 95% percentile intervals from a 2,000-draw leakage-group cluster
  bootstrap over pooled out-of-fold predictions. The intervals are **conditional on the
  fitted fold models**.

## 3. Results

Each design cell covers 1,200 test lemmas and 9,600 items. There were no missing LDL
predictions and no decoder failures. The selector produced no hypothesis for 8, 8 and 4
of the 9,600 Finnish items in three of its runs.

**LDL exact-form accuracy and edit distance:**

| Unit | Policy (pool) | LDL accuracy [95% CI] | LDL edit distance [95% CI] | Selector accuracy [95% CI] |
|---|---|---|---|---|
| Italian | random (500) | **0.351** [0.337, 0.365] | 1.64 [1.59, 1.69] | 0.862 [0.847, 0.878] |
| Italian | low-confidence (500) | 0.214 [0.202, 0.226] | 2.00 [1.96, 2.05] | 0.901 [0.888, 0.914] |
| Italian | high-entropy (500) | 0.225 [0.213, 0.238] | 1.98 [1.94, 2.02] | 0.898 [0.886, 0.911] |
| Italian | low-confidence (250) | 0.247 [0.235, 0.259] | 1.93 [1.89, 1.98] | 0.897 [0.884, 0.910] |
| Finnish | random (500) | **0.201** [0.187, 0.217] | 1.58 [1.54, 1.63] | 0.777 [0.756, 0.798] |
| Finnish | low-confidence (500) | 0.186 [0.172, 0.200] | 1.66 [1.61, 1.70] | 0.777 [0.758, 0.797] |
| Finnish | high-entropy (500) | 0.185 [0.170, 0.199] | 1.68 [1.63, 1.72] | 0.772 [0.753, 0.792] |
| Finnish | low-confidence (250) | 0.207 [0.192, 0.222] | 1.62 [1.57, 1.66] | 0.794 [0.775, 0.812] |

**Paired active − random differences, LDL accuracy** (same 9,600 items per unit):

| Unit | Comparison | Difference [95% CI] |
|---|---|---|
| Italian | low-confidence − random | −0.137 [−0.153, −0.122] |
| Italian | high-entropy − random | −0.126 [−0.140, −0.112] |
| Finnish | low-confidence − random | −0.015 [−0.027, −0.003] |
| Finnish | high-entropy − random | −0.016 [−0.029, −0.004] |

**Active selection's own benefit does not transfer.** For the selector, active selection
helps in Italian: low-confidence − random = **+0.039 [0.026, 0.050]**. In Finnish the
difference is 0.000 [−0.020, 0.019]. For downstream LDL the same samples are worse in
both units. All differences are in `outcomes/paired_differences.csv`.

**Pool-size sensitivity** (low-confidence, cap 500 − cap 250, LDL): Italian −0.033
[−0.047, −0.019]; Finnish −0.021 [−0.032, −0.009]. A larger pool offers more opportunity
to select atypical lemmas, and LDL accuracy falls further.

**Mechanism: sample composition** (`eval/sample_composition.csv`). Across folds, active
selection draws 44–54 of its 100 Italian lemmas from the *-ere*, *-ire* and *-rre*
conjugations. Random selection draws 14–21, which roughly matches the inventory, where
*-are* is 80%. The selector gains most on those minority classes: on *-ere*, random
reaches 0.58 and active 0.67. LDL accuracy on test *-are* verbs drops from 0.41 (random)
to 0.23–0.28 (active), while its accuracy on minority classes stays near 0.1 under every
policy. Finnish samples are already diverse (22–27 distinct three-letter endings), so the
shift is small.

**Dominant LDL error: copying the source.** The top LDL candidate equals the supplied
infinitive in 38–50% of items; the selector copies in under 5%. Italian PST.3SG (*-ò*) is
almost never produced (0.2–1.5%) and PRS.1SG reaches only 4–6%, against 49% for PST.1SG
and 64% for IMP.2SG under random sampling. The gold form appears among the top 10 LDL
candidates in 23–36% of items, and the mean predicted–gold cue correlation is 0.79–0.88
(`eval/ldl_mapping_quality.csv`). The bottleneck is generating candidates, not ranking
them.

**Fold variability**, descriptive only and not a CI (`eval/fold_variability.csv`): fold
SDs of LDL accuracy range from 0.010 to 0.050.

## 4. Interpretation for the study

* The LDL outcome depends strongly on the **sampling policy**: a 14-point accuracy gap in
  Italian at the same 100-lemma budget. Any cross-language morphology outcome must
  therefore fix one declared policy for all languages.
  * **Random sampling from the declared inventory** is the natural primary policy. It
    estimates performance under representative lexical composition.
  * Active samples answer a different question: which examples are informative for a
    particular selector.
* Equal lemma budgets gave equal exposure here (900 forms for 100 lemmas), but not equal
  lexical composition.
* The Italian–Finnish difference (0.35 vs 0.20 under random sampling) is a pilot
  observation about this learner, task and panel. It is not yet a complexity estimate.
  * LDL's high source-copy rate means the outcome partly measures how far junction n-grams
    and the wug binding can move away from the source form.
  * The *passato remoto* PST.3SG cell is effectively unreachable for this learner.
* `outcomes/ldl_outcomes.csv` is ready for analysis. It is keyed by variety, POS, task,
  resource, representation, policy, pool cap, budget and model, and records the frozen
  LDL settings.

## 5. GeLaTo linkage

`outcomes/population_links.csv` has one row per unit × population. It holds no morphology
values and fixes no aggregation.

* **Italian:**
  * Tuscan (n=8) is `candidate`, with a proposed status of `accepted`.
  * Bergamo (n=13) is `ambiguous`: its vernacular is Lombard.
  * Italian_South, Italian_Calabria, Sicilian_East and Sicilian_West are `excluded`;
    they are other languages linked by proxy.
* **Finnish:** Finnish (n=8) is `candidate`, with a proposed status of `accepted`.
* **Whole crosswalk:** candidate 119, ambiguous 53, unmatched 40, excluded 24, accepted 0.

Acceptance needs **your confirmation**. The variety, community, locality and period
reviews in `morph_ldl/data/gelato_review.yaml` were written by an agent. To accept a
row, add `confirmed_by` and `confirmed_date` to its entry and rerun the `data` and
`outcomes` stages.

Each population row carries its provenance:
* GeLaTo c625fdc population metadata;
* Zenodo 15263706 ADMIXTURE Q-file names for K12–30 and the GeneticInfoID label;
* the derived K23 summary file and row key.

These three layers are kept separate.

## 6. Adaptations and deviations

* **Selector.** A 2+2-layer, d=128 character Transformer replaces the paper's fairseq
  model; data are tiny. It is retrained from scratch each round on CPU with one thread,
  which is deterministic. Additions:
  * shared embeddings;
  * auxiliary source→source copy items from 270 fixed non-inventory source forms. Without
    them the seed-round model scored 0% on dev.
* **Scores.** Lemma score = mean over the 8 panel cells of the length-normalised
  surprisal of the top hypothesis, using the predicted length. Entropy is computed over
  sequence-level probabilities renormalised across the beam, using only p ≥ 0.05.
* **Upstream script issues** (detail in SELECTION.md):
  * Log-likelihoods are sorted as strings, which happened to give the right order.
  * Upstream scores were already length-normalised per token.
  * The script's cumulative 0.95 cut-off is equivalent to the per-hypothesis rule on
    upstream's nearly flat per-token beams, but not on sequence-level probabilities.
* **LDL.** JudiLing's combined cue and S matrices and `cal_max_timestep` would leak gold
  information, so they are not used. Semantic vectors are keyed to identifiers, so each
  lemma has the same vector in every sample. Production is refitted exactly for each
  held-out lemma via a rank-one update.
* **Setting choice.**
  * The first tuning run used fold-0 dev and pool lemmas, which are test lemmas in folds
    1–2. The independent review found this.
  * That run was discarded before any outer-test fit and redone on auxiliary
    non-inventory lemmas, which chose the same setting.
* **Configuration.** Subagents ran on Opus 5.5 at the runtime's default effort, because
  this session could not set "medium thinking". You approved this. The project agent
  definitions in `.claude/agents/` set `effort: medium` for future sessions.

## 7. Review and verification

* **Independent read-only review:** no critical findings and 3 major ones, all fixed:
  * tuning leakage;
  * agent-accepted GeLaTo links;
  * weak resume hashes.

  Of the minor findings, these were fixed:
  * copy-anchor asymmetry across pool caps;
  * `all_cells` mode not reaching LDL;
  * audit ordering;
  * a missing source-tree hash;
  * Julia thread count;
  * missing re-tune lock;
  * paired-table caveat;
  * bootstrap clustering by leakage group.

  Group-id fragility and the Italian doublet caveat (below) are documented.
* **Artifact audit.** Run before and after LDL, it found 0 problems over 28 sample and
  tuning sets. No test or dev lemma or test group entered any sample. Seed sets are
  identical across policies, queries cover exactly the test lemmas, and auxiliary lemmas
  share no group with the inventory.
* **Real-model isolation tests:**
  * predictions are bit-identical with gold withheld or permuted;
  * results don't depend on lemma order, and each lemma's state is reset;
  * semantic vectors are stable across samples;
  * on the selection side, a gold permutation test and enforced oracle reveals.

## 8. Runtime

On an 18-core Apple-silicon machine:

| Stage | Time |
|---|---|
| Data stage | about 2 min |
| LDL setting choice | 1.5 min |
| Selection (24 runs, 6 processes) | 35 min |
| LDL (24 models × 400 held-out lemmas) | 5.5 min |
| Selector evaluation | 13.5 min |
| Evaluation and outcomes | under 1 min |
| **Whole pilot after data** | **54 min** |

## 9. Limitations

* **Scale.** Two languages, one POS, one repetition, budget 100.
  * The intervals omit selection and fitting variability.
  * Full-pipeline uncertainty needs more repetitions, with new split, selector and
    semantic seeds.
* **GeLaTo samples.** They are small (n=8 for both accepted-proposed populations) and
  record no individual's language.
* **Italian doublets.** MGN Italian keeps one form per cell, so legitimate *passato remoto*
  doublets such as *credei* / *credetti* cannot be credited. Within a lemma, MGN's choice
  is sometimes inconsistent.
* **Weak LDL baseline.** LDL production on held-out lemmas has a low ceiling, about
  0.35–0.38 even on a regular synthetic toy language, and copies the source. Outcome
  differences may partly reflect how well bigram junction cues fit a language's
  stem–suffix structure.
* **Policy-neutral tuning is assumed.** Setting choice on a random auxiliary background
  assumes the optimum does not depend on the sampling policy.
* **Unverified provenance.** The `data-custom` sources and the release behind MGN Finnish
  are unverified.

## 10. Next steps

1. **Fix the primary policy before scaling.** Recommended: random sampling, with active
   selection as a labelled sensitivity analysis.
2. **Address source copying on auxiliary lemmas only.** Candidates include `lexeme_refit`
   binding, tolerance mode and inflection-SD or threshold grids. Re-freeze settings
   before any new outer-test fit.
3. **Extend coverage.** Budget 200 (`--set 'selection.budgets=[100,200]'`; nested prefixes),
   2–3 more repetitions, and more GeLaTo-linked verb units. French (`mgn_custom:french-v`)
   is the best-documented next candidate; it is phonemic, so analyse it separately.
   Noun units also have candidates.
4. **Confirm or revise** the proposed GeLaTo acceptances, and review further languages'
   populations.
5. **Then** specify the admixture-side estimand and aggregation rule, using the linkage
   table as is.
