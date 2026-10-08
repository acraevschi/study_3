# Experimental protocol (declared before outer-test evaluation)

Owner: main agent. Status: **declared for pilot_v1** (configs/pilot.yaml). Component
details: [DATA.md](DATA.md), [SELECTION.md](SELECTION.md), [LDL_PROTOCOL.md](LDL_PROTOCOL.md);
interfaces: [CONTRACT.md](CONTRACT.md).

## 1. Estimand

For a morphology unit (variety × POS × representation × resource), a sampling policy,
a candidate-pool size and a training budget of B lemmas: the expected exact-form accuracy
(and edit distance) of a freshly fitted end-state LDL model on **source-known paradigm
completion for held-out lemmas** — given one supplied source form and its cell, produce
the forms of a fixed panel of target cells — over the declared eligible inventory.

Lower accuracy or larger edit distance means greater difficulty *for this learner under
this task and sampling protocol*. The estimate is conditional on the declared inventory,
panel, representation and acquisition policy; it is not a language-intrinsic quantity.
Independent test folds do not remove cross-language differences in lexical composition.

## 2. Task

* Source cell: infinitive (`NFIN`). Target panel (8 cells, identical abstract slots in
  every unit): PRS.1SG, PRS.3SG, PRS.3PL, PST.1SG, PST.3SG, PST.3PL, COND.3SG, IMP.2SG.
  Each slot is mapped to the unit's `cell_norm` in the config.
* Italian PST = synthetic past perfective (*passato remoto*, `IND;PST;PFV`), chosen
  because it is the synthetic past with independent stem alternation; the imperfective
  past is almost entirely predictable from the infinitive (exceptions such as *essere*,
  *fare*, *dire*) and would make the panel easier by construction.
  Finnish PST = active indicative past (*imperfekti*). Both choices are made on
  morphological grounds before any result; they are recorded, not tuned.
* The panel bounds training exposure: every selected lemma contributes 1 source + 8
  target forms (9 forms) whatever the language's paradigm size. All eligible original
  forms of selected lemmas are exported separately (`budget_{B}_allforms.csv`).
* Eligible lemma: non-missing source and all 8 panel cells (variant 0). Training and
  source anchors use variant 0; evaluation accepts any documented variant.
* Held-out information: exactly the source form (variant 0) and its cell per held-out
  lemma, supplied through `morph_ldl.cv.queries.build_queries`. This is counted
  separately from the training budget.

## 3. Units in the pilot

Two GeLaTo-linked languages, one POS (verbs), one representation (orthography):
`ita.V.orth.mgn` and `fin.V.orth.mgn` (MGN `data/` files, built by MGN from UniMorph
without transliteration). Final confirmation recorded in REPORT.md after the data audit.

## 4. Cross-validation design

Per unit: inventory of 1,200 eligible lemmas (by leakage group, fixed `inventory` seed);
3 outer folds by group (one repetition in the pilot, more supported). Per fold: test =
the fold (≈400 lemmas); from the remaining lemmas: dev 80, seed 20, pool 500 (nested
pool 250 for the sensitivity run), overflow unused. Active and random selection share
the seed, pool, dev set, budget, selector architecture/training/decoding settings and
selector initialisation seed. LDL is fitted afresh on each exported sample with
identical settings and evaluated on the identical outer-test lemmas.

Dev lemmas guide selector early stopping only; they are never acquisition or training
data. LDL settings are chosen on auxiliary non-inventory lemmas (§6). The selector's
auxiliary copy items use the source forms (only) of 270 further auxiliary lemmas,
identical for every fold, policy and pool cap, so no candidate is ever a copy anchor.

## 5. Selection

Seed 20 lemmas (random, shared), batch 20, budget 100 (4 acquisition rounds; budget 200
continues the same trajectory). Policies: matched `random`, `low_confidence` (mean
length-normalised surprisal of the top beam hypothesis over the 8 target cells),
`high_entropy` (mean entropy over renormalised beam hypotheses with p ≥ 0.05).
Selector: small character Transformer, retrained from scratch each round (details and
adaptations in SELECTION.md).

## 6. Downstream LDL

End-state (frequency-free) JudiLing fitting; letter n-gram cues; simulated semantics
(lexeme + inflectional-feature vectors + noise) with per-identifier seeding so a lemma
has the same vector in every sample; source-binding wug protocol (`wug_refit`) for
held-out lemmas with per-lemma reset (LDL_PROTOCOL.md).

**Setting choice on never-tested lemmas (stage `ldl_tune`).** Two LDL settings were
left open after the component smoke test: cue n-gram size {2, 3} and inflectional-feature
SD {0.4, 4.0}. They are chosen once, before any outer-test fit, on an **auxiliary set**:
eligible lemmas outside the 1,200-lemma inventory (whole leakage groups, `auxiliary`
seed; `splits/<unit>/auxiliary_manifest.csv`), which are never test, dev, seed or pool
in any fold or repetition. Per unit: background = 100 auxiliary lemmas, held-out = 80
other auxiliary lemmas, source-known protocol. The setting with the highest mean
held-out accuracy across the two units is applied to every unit, policy, budget and
fold (ties: lower mean edit distance). Result: bigram cues, inflection SD 0.4 (best in
both units; `ldl_tune/tune_by_unit.csv`). The stage refuses to re-run once outer-test
LDL outputs exist.

*Correction history.* A first version tuned on fold 0's dev/pool lemmas, which are
outer-test lemmas of folds 1–2 (found by the independent review). That run was
discarded before any outer-test LDL fit (`outputs/pilot_v1/_superseded_2026-10-06/`);
the corrected auxiliary-set tuning selected the same setting. The component smoke runs
(random Italian lemmas) motivated the grid but were not used to choose within it.
Choosing on a random auxiliary background is policy-neutral only insofar as the optimum
does not depend on the sampling policy (stated assumption).

## 7. Metrics and uncertainty

* Exact-form accuracy (any variant), micro (cell-level) and macro (mean of lemma means),
  with denominators; Levenshtein distance over segments and its length-normalised form;
  counts of missing predictions and decoder failures (scored incorrect, distance = gold
  length); LDL mapping quality (Ĉ–gold cue correlation) separately from decoded
  production; selector accuracy on the same test items.
* 95% percentile intervals from a lemma-cluster (leakage-group) bootstrap of the pooled out-of-fold
  predictions (2,000 draws, `bootstrap` seed per design cell); paired active − random
  differences resample the same lemmas for both policies. These intervals are
  **conditional on the fitted fold models**: they omit variability from re-running
  selection and fitting. Split / acquisition / semantic-seed variability is reported
  descriptively across folds and repetitions; fold SD/√K is not used as a CI.
* Full-pipeline uncertainty would require repeating selection and fitting under
  resampling (more repetitions with new split, selector and semantic seeds).

## 8. What is not done here

No admixture–morphology association is fitted; no setting or language is chosen by
looking at ancestry values; GeLaTo populations are linked, not aggregated.
