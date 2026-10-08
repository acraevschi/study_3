# Report: paradigm cell filling with an LDL selector, and a Grambank inflection outcome (pcfp_v1)

Date: 2026-10-08. Config: `configs/pcfp_v1.yaml` (hash 8b088277dfaa1e26). Code: commit
`9c4ee53` on branch `pcfp-v1`; every stage manifest records `pipeline_dirty: false`. The
outputs are in `outputs/pcfp_v1/`, and the report tables are in
`analyses/pcfp_v1_2026_10_08/` (written by `report_tables.py`).

The earlier task, pilot_v1 (source-known new verbs with a Transformer selector), is
archived. Its outputs are unchanged in `outputs/pilot_v1/` and its report is
[REPORT.md](REPORT.md). Its code state is commit **`24390cf`**; `configs/pilot.yaml` and
`configs/smoke.yaml` run only with that commit.

**Status.** Both parts of the experiment ran end to end:

* the real `pcfp_v1` run (step 4 of the brief): splits → ldl_tune → select → audit → ldl
  → evaluate → outcomes → typology → audit;
* the `typology` stage (step 6).

Results:

* Both audits report **0 problems**: 98 artifact sets after selection and 102 at the end,
  including the typology checks.
* 151 tests pass (fast and slow).
* No admixture–morphology association was fitted. No ancestry values were read. The
  admixture exposure variable remains undefined.

**Headline.** At budget 100, LDL-uncertainty selection does **not** beat random
selection. In both languages, every active policy is about 1 percentage point *less*
accurate than random on the shared core test items:

* **Italian:** the paired intervals exclude 0.
* **Finnish:** the intervals just include 0 for two of the three comparisons.

Random is best in 5 of the 6 folds.

---

## 1. What changed from pilot_v1

| | pilot_v1 (archived) | pcfp_v1 |
|---|---|---|
| Task | source NFIN → 8 cells, test verbs never seen | paradigm cell filling: each verb is known through k random forms, and the hidden cells are predicted |
| Selector | character Transformer | the LDL model itself (JudiLing), with an exact rank-one citation-row update |
| Test items | held-out new verbs, different per run | hidden cells of 100 fixed **core** verbs per fold, identical for every policy |
| Downstream LDL | `wug_refit` source binding | known lexemes: the verb's shown forms are in training, and hidden cells are decoded from L + ΣV |
| Outcomes | LDL accuracy only | **Grambank inflectional extent** (primary, all GeLaTo-linked languages) + LDL predictability (secondary, inflecting languages only) |

## 2. Declared design (fixed before any evaluation; full detail in [PROTOCOL.md](PROTOCOL.md))

* **Units.** `ita.V.orth.mgn` (48 cells) and `fin.V.orth.mgn` (35 cells): MGN `data/`,
  orthographic, single-word cells only (`cell_rule.max_multiword_share: 0.5`).
* **Eligible verbs.** Derived paradigms are excluded and complete paradigms are
  required. This leaves 7,767 Italian and 7,597 Finnish verbs.
* **Citation cell.** NFIN is one known form of every verb. The selector uses it as the
  candidate's representation, and it is never a test cell. The MGN lemma label equals
  NFIN for 99.9% of Italian verbs (8 exceptions) and for 100% of Finnish verbs.
* **Exposure.**
  * k ~ Uniform{1..7} shown cells (the cap is min(7, n_cells − 1)). The shown cells are
    drawn without replacement from the non-citation cells.
  * Each verb has exactly one draw, seeded by (master, "exposure", unit, lemma). The same
    draw is used for every policy, fold, budget and pool cap.
  * The manifest is re-derived and validated at every audit.
  * Realised k: mean 3.98 (Italian) and 4.00 (Finnish).
* **Cross-validation.**
  * Each unit has an inventory of 1,200 verbs, sampled by leakage group, with 3 outer
    folds and 1 repetition.
  * Per fold: 100 core verbs (disjoint across folds), a 20-verb seed, a 500-verb pool
    with a nested 250-verb pool for sensitivity, and an overflow.
  * There is no dev role.
* **Budget.** B = 100 verbs selected (seed included) + 100 core verbs = 200 training
  verbs, about 790–800 training forms. The batch size is 20, so selection takes 4
  rounds.
* **Policies.** random@500, low_confidence@500, high_entropy@500 and low_confidence@250.
* **Primary test items.** All hidden non-citation cells of the 300 core verbs per unit:
  * Italian: 12,902 items, 43.0 test cells per verb on average;
  * Finnish: 9,032 items, 30.1 per verb.

  These items are identical across policies, so the comparisons are paired. A secondary
  `selected` item set (the hidden cells of each run's own selected verbs) is reported
  separately.
* **LDL.**
  * JudiLing 1.0.1, end-state ridge fit (λ = 0.02).
  * Simulated semantics L(lemma) + ΣV(features) + N(lemma, cell): dimension 1000,
    lexeme SD 4, noise SD 1. Vectors are identifier-keyed.
  * Decoding with `learn_paths`: threshold 0.05, max_can 10, max_t = longest training
    form + 4.
  * The cue inventory and adjacency come from training forms only.
  * A hidden cell is decoded from the meaning L + ΣV, without noise.
* **Selector.**
  * For each candidate, the citation row is added with an exact rank-one
    (Sherman–Morrison) update of G and F; novel cues get new columns. The pre-drawn
    shown cells are then decoded and the row is removed.
  * Scores:
    * low_confidence: u = 1 − s₁;
    * high_entropy: the entropy of softmax(supports / T), with **T = 0.02**.
  * A cell's scores are averaged over its shown cells, then over 3 semantic seeds; seed
    0 is the evaluated model's seed.
  * Ties are broken deterministically. The comprehension-side check is logged and never
    used for selection.
* **Uncertainty.**
  * 95% percentile intervals come from a 2,000-draw leakage-group cluster bootstrap over
    pooled out-of-fold predictions.
  * Paired differences use identical core items only.
  * The intervals are conditional on the fitted fold models and on the single selection
    run.

### LDL tuning (on auxiliary verbs outside the inventory, own semantic seed; before selection)

Held-out accuracy on the hidden cells of 100 auxiliary core verbs (4,321 Italian items,
3,025 Finnish), with the other 100 auxiliary verbs as training extras:

| cue n-gram | inflection SD | Italian acc | Italian copy | Finnish acc | Finnish copy | mean acc |
|---|---|---|---|---|---|---|
| 2 | 0.4 | 0.091 | 0.743 | 0.087 | 0.467 | 0.089 |
| 2 | 1.0 | 0.192 | 0.424 | 0.122 | 0.241 | 0.157 |
| **2** | **2.0** | **0.197** | **0.237** | **0.128** | **0.152** | **0.162** |
| 2 | 4.0 | 0.170 | 0.143 | 0.125 | 0.097 | 0.148 |
| 3 | 0.4–4.0 | 0.101–0.122 | 0.67–0.85 | 0.064–0.079 | 0.51–0.77 | 0.083–0.100 |

The chosen settings, bigram cues with inflection SD 2.0, were best in both units. The
pilot_v1 setting of SD 0.4 makes known-lexeme LDL copy shown forms most of the time
(74% Italian).

## 3. Results: LDL predictability

No LDL prediction is missing, and no decoder failed. The selector produced a candidate
for every scored cell, with no errors and no no-candidate cells in any round.

### 3.1 Core items (primary, identical across policies)

| Unit | Policy | Accuracy (micro) [95% CI] | Verb-macro | Edit distance | Copy rate |
|---|---|---|---|---|---|
| Italian | **random@500** | **0.172** [0.161, 0.184] | 0.174 | 2.37 | 0.305 |
| Italian | low_confidence@500 | 0.160 [0.150, 0.170] | 0.162 | 2.42 | 0.341 |
| Italian | high_entropy@500 | 0.161 [0.150, 0.171] | 0.163 | 2.45 | 0.337 |
| Italian | low_confidence@250 | 0.162 [0.151, 0.173] | 0.164 | 2.45 | 0.315 |
| Finnish | **random@500** | **0.122** [0.110, 0.134] | 0.125 | 2.64 | 0.151 |
| Finnish | low_confidence@500 | 0.114 [0.103, 0.125] | 0.117 | 2.71 | 0.183 |
| Finnish | high_entropy@500 | 0.114 [0.104, 0.125] | 0.117 | 2.67 | 0.169 |
| Finnish | low_confidence@250 | 0.108 [0.098, 0.119] | 0.110 | 2.69 | 0.171 |

**Paired differences on core items** (accuracy in percentage points; edit distance in
characters; positive edit distance = worse):

| Unit | Comparison | Δ accuracy [95% CI] | Δ edit distance [95% CI] |
|---|---|---|---|
| Italian | low_confidence@500 − random | −1.14 [−1.95, −0.35] | +0.055 [+0.009, +0.105] |
| Italian | high_entropy@500 − random | −1.12 [−1.90, −0.29] | +0.082 [+0.027, +0.137] |
| Italian | low_confidence@250 − random | −0.93 [−1.81, −0.04] | +0.082 [+0.025, +0.137] |
| Italian | low_confidence@500 − @250 | −0.21 [−1.03, +0.56] | −0.026 [−0.078, +0.024] |
| Finnish | low_confidence@500 − random | −0.80 [−1.65, +0.04] | +0.068 [+0.015, +0.121] |
| Finnish | high_entropy@500 − random | −0.75 [−1.54, +0.07] | +0.031 [−0.030, +0.094] |
| Finnish | low_confidence@250 − random | −1.36 [−2.15, −0.53] | +0.050 [−0.011, +0.109] |
| Finnish | low_confidence@500 − @250 | +0.56 [−0.27, +1.36] | +0.017 [−0.032, +0.064] |

**Per fold** (core accuracy):

| Fold | Italian random | Italian low_conf@500 | Italian high_ent | Italian low_conf@250 | Finnish random | Finnish low_conf@500 | Finnish high_ent | Finnish low_conf@250 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.175 | 0.160 | 0.160 | 0.163 | 0.105 | 0.110 | 0.101 | 0.096 |
| 1 | 0.176 | 0.163 | 0.160 | 0.165 | 0.141 | 0.134 | 0.134 | 0.122 |
| 2 | 0.164 | 0.158 | 0.162 | 0.159 | 0.121 | 0.098 | 0.108 | 0.108 |

**Reading.**

* Every active-minus-random difference is negative, in both languages, on accuracy, and
  random is best in 5 of 6 folds.
* Finnish accuracy varies between folds more than the policy effect does: the fold SD
  is about 1.8 points, while the Italian fold SD is 0.1–0.7. The Finnish result therefore
  rests on fewer effective replications than its intervals suggest.
* There was only one selection repetition, so the intervals do not include the variance
  of the selection run itself, for example a different random draw.
* The safe conclusion: **uncertainty sampling with this LDL selector gives no gain over
  random at B = 100 and is probably slightly worse.** The pool cap (250 vs 500) makes no
  detectable difference.

### 3.2 Selected verbs (secondary; not comparable across policies)

This item set holds the hidden cells of each run's own selected verbs. It describes the
verbs chosen, not the outcome:

| Unit | random | low_conf@500 | high_ent@500 | low_conf@250 |
|---|---|---|---|---|
| Italian | 0.171 | 0.124 | 0.127 | 0.156 |
| Finnish | 0.130 | 0.100 | 0.112 | 0.098 |

The active policies do pick verbs that the final model finds harder, as uncertainty
sampling should. Training on those verbs, however, does not transfer to better
predictions for the core verbs.

### 3.3 Copying, compared with pilot_v1

The copy rate is the share of predictions equal to a form already shown for that verb.

| | Copies any shown form | Copies the citation form |
|---|---|---|
| pilot_v1, Italian (prediction = infinitive) | 0.375 (random) – 0.504 | 0.375–0.504 |
| pilot_v1, Finnish | 0.406 (random) – 0.475 | 0.406–0.475 |
| **pcfp_v1, Italian, core** | 0.305 (random) – 0.341 | 0.002–0.003 |
| **pcfp_v1, Finnish, core** | 0.151 (random) – 0.183 | 0.002–0.005 |

**Syncretism.** Some copying is correct: in 6.1% of the Italian core items and 1.0% of
the Finnish ones, the gold form equals a shown form. For example, the Italian 1PL
present indicative, 1PL present subjunctive and 1PL imperative are all *abboniamo*.

| Unit | Accuracy on syncretic items | Accuracy on non-syncretic items | Copy rate (non-syncretic) | Correct share of copies |
|---|---|---|---|---|
| Italian (random) | 0.750 | 0.134 | 0.272 | 0.149 |
| Finnish (random) | 0.556 | 0.118 | 0.146 | 0.037 |

Copying of the citation form, the failure mode that dominated pilot_v1, has almost
disappeared. Copying of other shown forms is still common in Italian: 27% of
non-syncretic items, wrong in about 85% of cases. The active policies copy slightly more
than random in both languages.

`analyses/pcfp_v1_2026_10_08/syncretism.csv` and `copy_rates_vs_pilot.csv` hold the
full tables.

### 3.4 Accuracy by k (core items)

| k shown | Italian random | Italian low_conf@500 | Italian high_ent | Finnish random | Finnish low_conf@500 | Finnish high_ent |
|---|---|---|---|---|---|---|
| 1 | 0.078 | 0.090 | 0.071 | 0.061 | 0.061 | 0.071 |
| 2 | 0.142 | 0.132 | 0.131 | 0.082 | 0.082 | 0.082 |
| 3 | 0.144 | 0.139 | 0.141 | 0.098 | 0.090 | 0.081 |
| 4 | 0.214 | 0.201 | 0.198 | 0.137 | 0.122 | 0.108 |
| 5 | 0.188 | 0.181 | 0.184 | 0.153 | 0.132 | 0.140 |
| 6 | 0.217 | 0.186 | 0.184 | 0.149 | 0.135 | 0.160 |
| 7 | 0.231 | 0.216 | 0.231 | 0.191 | 0.172 | 0.191 |

The low_confidence column is the @500 run. Accuracy roughly triples from k = 1 to
k = 7, so exposure is the strongest driver of item accuracy in this design. Because k is
drawn per verb independently of policy, the paired comparison is balanced on k.

### 3.5 Accuracy by cell (core items, random@500)

| Italian: lowest | acc | Italian: highest | acc |
|---|---|---|---|
| 1;IND;PRS;SG (*-o*) | 0.000 | 2;PRS;SBJV;SG | 0.422 |
| 3;IND;PFV;PST;SG (*-ò*) | 0.021 | 1;PRS;SBJV;SG | 0.409 |
| 3;FUT;IND;PL | 0.044 | 2;IND;PRS;SG | 0.375 |
| 3;COND;PL | 0.057 | 2;PST;SBJV;SG | 0.350 |
| 3;IND;PFV;PL;PST | 0.066 | 3;PRS;SBJV;SG | 0.328 |

| Finnish: lowest | acc | Finnish: highest | acc |
|---|---|---|---|
| 3;ACT;IND;PL;POS;PRS | 0.030 | 3;ACT;IND;POS;PST;SG | 0.259 |
| 1;ACT;IMP;PL;POS;PRS | 0.040 | 3;ACT;COND;POS;PRS;SG | 0.232 |
| 2;ACT;IMP;POS;PRS;SG | 0.041 | 1;ACT;COND;POS;PRS;SG | 0.216 |
| PASS;POS;POT;PRS | 0.045 | 2;ACT;COND;POS;PRS;SG | 0.205 |
| IND;PASS;POS;PRS | 0.047 | 1;ACT;IND;POS;PST;SG | 0.204 |

**The Italian 1SG present is never correct** (0/286), and the 3SG *passato remoto*
almost never is. We checked that this is not a bug:

* The cell is present in training, for example 10 of the 805 rows in fold 0.
* The model predicts the stem plus *-a* or *-i* (*abbonare* → *abbona* for gold
  *abbono*), which is the ending of the 3SG present or the present subjunctive.

The likely cause is how meanings are built. Under additive feature semantics (L + ΣV),
the 1SG present shares its features (1, SG, PRS, IND) with many cells that end in other
vowels. The only meaning that predicts *-o#* is a conjunction of all four features, and
an additive model learns it poorly from about 17 training rows per cell. The best Italian
cells are the subjunctive singulars, which are syncretic with each other: in the
present, 1SG = 2SG = 3SG. Full tables: `accuracy_by_cell_core.csv`, `cell_extremes.csv`.

### 3.6 Mapping quality (core items, mean over folds)

Three measures, defined as follows:

* **cor(Ĉ, C):** the correlation between the predicted cue vector and the gold form's
  cue vector.
* **Gold reachable:** the gold form can be built from the training cue inventory and
  adjacency.
* **Gold in top:** the gold form is among the decoder's candidates.

| Unit | cor(Ĉ, C) | Gold reachable | Gold in top |
|---|---|---|---|
| Italian | 0.75–0.77 | 0.65–0.67 | 0.19–0.21 |
| Finnish | 0.71–0.72 | 0.59–0.60 | 0.14–0.16 |

About a third of Italian and two fifths of Finnish gold forms cannot be built from the
training cues of a 200-verb sample at all, and accuracy is bounded by that.

Training fit is high:

| | Italian | Finnish |
|---|---|---|
| Production accuracy | 0.92 | 0.82 |
| Comprehension accuracy | 0.97 | 0.98 |

## 4. The selector

### 4.1 What it chose: inflection-class composition of the acquired verbs

The table gives shares, averaged over folds. The pool has 500 verbs per fold. pilot_v1 is
the Transformer selector on the archived task.

| Unit | Run | Italian -are | Italian -ire | Italian -ere | Italian -rre | Mean citation length |
|---|---|---|---|---|---|---|
| Italian | pool | 0.809 | 0.088 | 0.091 | 0.013 | 9.93 |
| Italian | pcfp random | 0.812 | 0.075 | 0.100 | 0.013 | 9.85 |
| Italian | pcfp low_conf@500 | 0.763 | 0.075 | 0.129 | 0.033 | 10.49 |
| Italian | pcfp high_ent@500 | 0.788 | 0.125 | 0.088 | 0.000 | 9.85 |
| Italian | pcfp low_conf@250 | 0.779 | 0.096 | 0.108 | 0.017 | 10.32 |
| Italian | pilot Transformer low_conf@500 | 0.442 | 0.179 | 0.321 | 0.058 | 9.93 |
| Italian | pilot Transformer high_ent@500 | 0.479 | 0.188 | 0.271 | 0.062 | 10.35 |

| Unit | Run | Finnish VV | Finnish Vta | Finnish CCa | Finnish da | Mean citation length |
|---|---|---|---|---|---|---|
| Finnish | pool | 0.592 | 0.159 | 0.127 | 0.122 | 8.73 |
| Finnish | pcfp random | 0.608 | 0.179 | 0.108 | 0.104 | 8.60 |
| Finnish | pcfp low_conf@500 | 0.608 | 0.117 | 0.162 | 0.112 | 8.64 |
| Finnish | pcfp high_ent@500 | 0.571 | 0.196 | 0.188 | 0.046 | 8.50 |
| Finnish | pilot Transformer low_conf@500 | 0.483 | 0.337 | 0.096 | 0.083 | 8.38 |
| Finnish | pilot Transformer high_ent@500 | 0.462 | 0.296 | 0.138 | 0.104 | 9.23 |

The classes are a coarse proxy from the citation-form ending, defined in
`pcfp.inflection_class`.

* The pilot_v1 Transformer moved sharply away from the majority classes: 44–48% Italian
  -are against 81% in the pool.
* The LDL selector stays close to the pool. It shifts only mildly:
  * Italian low_confidence takes more -ere/-rre and longer citations;
  * Finnish high_entropy takes more consonant-gradation (CCa) verbs and fewer -da verbs.

**Overlap.** Each run acquires 80 verbs beyond the 20 shared seed verbs.

* Active and random share 8–19 of the 80 per fold (mean 12.8), exactly the chance level
  of 80 · 80 / 500 = 12.8.
* The active policies share 20–42 verbs with each other, well above chance.

The selector's choices are consistent across policies but unrelated to random's, and
only mildly biased towards any class.

### 4.2 Are the scores informative? (selector checks, mean over folds and rounds)

| Unit | Policy | Score CV | Top candidate = citation | Seed SD / candidate SD | η² (final 3 letters) | Adj. R² (length + final 3) |
|---|---|---|---|---|---|---|
| Italian | low_conf@500 | 0.29 | 0.095 | 0.48 | 0.042 | 0.058 |
| Italian | high_ent@500 | 0.52 | 0.100 | 0.72 | 0.012 | 0.013 |
| Finnish | low_conf@500 | 0.28 | 0.069 | 0.57 | 0.286 | 0.226 |
| Finnish | high_ent@500 | 0.37 | 0.083 | 0.81 | 0.204 | 0.145 |

The scores are not degenerate:

* With T = 0.02, entropy is no longer a function of the candidate count, which was review
  finding M1 at T = 0.1.
* Little of the score is explained by surface shape. In Italian at most 6% is, because
  every citation ends in *-re*. In Finnish up to 23% is.

They are noisy, however. The between-seed SD of a candidate's score is 0.5–0.8 times the
spread between candidates, so averaging three semantic seeds removes only part of the
noise. This likely explains why the selection is close to random and slightly worse. The
selector chases verbs whose uncertainty partly reflects the arbitrary simulated
semantics of one seed, and those verbs are not more informative about the core verbs.

### 4.3 Comprehension-side check (logged only; Spearman over candidates, mean over rounds)

For each candidate, the background model's comprehension of the citation form is logged
before the row update: cor(ŝ, s) and the relative distance ‖ŝ − s‖ / ‖s‖. The share of
citation cues unseen in training is logged too.

| Unit | Policy | ρ(score, cor) | ρ(score, rel. distance) | ρ(score, share of unseen cues) |
|---|---|---|---|---|
| Italian | low_conf@500 | +0.08 | −0.27 | −0.30 |
| Italian | high_ent@500 | +0.14 | −0.30 | −0.37 |
| Finnish | low_conf@500 | +0.08 | −0.13 | −0.10 |
| Finnish | high_ent@500 | +0.01 | −0.26 | −0.28 |

Production uncertainty and comprehension difficulty are **not** aligned. If anything they
run the other way: candidates whose citation form is comprehended better, and whose cues
are less novel, get *higher* production uncertainty. A plausible mechanism is the
rank-one update, which fits a row with many novel cues almost exactly through the new
cue columns, so its shown cells decode more confidently. This is consistent with review
finding m1 (candidates whose shown cells include NFIN look certain). The check was not
used for selection.

## 5. Results: Grambank inflectional extent (primary outcome)

Sources: Grambank v1.0.3 (`7ae000c`) and Glottolog CLDF v5.3 (`072ca0d`). The feature
set is `grambank_core_inflection_35_v1` (35 features), with verbal, nominal and
no-agreement sensitivity sets; [TYPOLOGY.md](TYPOLOGY.md) gives the details. The
typology audit passes, and the ancestry firewall logged no forbidden file. The real-run
numbers are identical to the development run (TYPOLOGY §10).

* **GeLaTo linkage.** 558 populations give 619 link rows over 350 language-level
  Glottocodes:
  * 309 have a non-proxy link and 41 are proxy-only;
  * 42 populations reach no language, for example Yi (an ambiguous group), regional
    Vanuatu and Papuan samples, and Dinka.
* **Languages entering the outcome** (non-proxy, in Grambank, coverage ≥ 60%): **173**.
  * 193 non-proxy languages are in Grambank. At other coverage thresholds: ≥ 50% gives
    183 and ≥ 75% gives 168.
  * Including proxy-only languages: 200 at ≥ 60%, out of 223 in Grambank.
* **No inflection** (0 of the coded features present):
  * Central Khmer 0/35, Naxi 0/35, She 0/29 and Vietnamese 0/30, all of which enter;
  * Thai 0/20, which is below the coverage threshold.
* **Minimal inflection** (≤ 1 present): the five above plus Mandarin Chinese 1/35 and
  Northern Tujia 1/35. All are in mainland Southeast Asia or China. These are exactly
  the languages the LDL outcome cannot represent, and the reason Grambank is the primary
  outcome.
* **Clitic flags.** 15 non-proxy languages are flagged by heuristic for review, including
  Burmese, Hakka, Basque and nine Austronesian languages. Basque and Kharia are known
  false positives.
* **GeLaTo languages missing from Grambank:** 116 non-proxy languages; the full list is
  in `typology/missing_from_grambank.csv`. The largest by sampled individuals:
  * Yoruba 75, Scottish Gaelic 43, Taiga Sayan Turkic 40, West Circassian 33,
    Chachapoyas Quechua 31, Tajik 31, Zoroastrian Yazdi 28, Mamusi 26, West !Xoon 26,
    Spanish 25.
  * Also Spanish, German, Romanian and Bulgarian. We verified directly against
    `languages.csv` that Grambank v1.0.3 has no entry for them.
* **Overlap with the LDL outcome.** Both LDL units, Italian (`ital1282`, 19/35) and
  Finnish (`finn1318`, 14/35), are GeLaTo-linked, in Grambank and at full coverage, so
  each has both outcomes. Of the 16 MGN verb resources in the broad audit, 8 languages
  are GeLaTo-linked (non-proxy), in Grambank at ≥ 60% coverage, and so candidates for
  both outcomes: English, French, Catalan, Finnish, Serbo-Croatian, Armenian, Italian
  and Northern Sami. Bulgarian, German, Romanian and Spanish have no Grambank entry.
  Modern Greek is proxy-only. Portuguese, Tibetan and Macedonian are not GeLaTo-linked.
  * The LDL outcome therefore currently covers 2 of the 173 Grambank languages. At most
    8 could have both outcomes without new data collection.
* **Link status.** None of the 619 links is accepted. Acceptance needs a human
  `confirmed_by` and `confirmed_date`, and none has been entered: 485 rows are
  `candidate`, 90 `ambiguous`, 40 `unmatched` and 4 `excluded`.
* **Group map-downs need review.** A map-down applies only when a group Glottocode has
  exactly one language-level Grambank descendant. Some of the 12 mapped links are
  doubtful as language identifications:
  * Mixe → Oluta Popoluca;
  * Malawi_Ngoni → Tanzanian Ngoni;
  * Pathan → Southern Pashto (many Pakistani Pathans speak Northern Pashto);
  * Sardinian → Campidanese;
  * Tu → Mangghuer;
  * Circassian → Kabardian.

  They are candidates only. A reviewer should confirm or reject each one before any
  analysis.

## 6. Decisions and deviations

All were made before any outer-test evaluation unless stated.

| Item | Decision | Who / when |
|---|---|---|
| Task | PCFP replaces the source-known new-verb task; pilot_v1 is untouched | user brief |
| Exposure | k ~ U{1..min(7, n−1)}, one draw per verb, shared across everything | user brief |
| Selection | active selection picks verbs; cells revealed at random (pre-drawn) | user brief |
| Selector | LDL with the citation row added by an exact rank-one update; the Transformer code is kept for pilot_v1 | user brief |
| Single-word cells, C = 100 core, B = 100, no dev role | proposed defaults, adopted | brief defaults |
| **Entropy temperature T = 0.1 → 0.02** | at 0.1, entropy tracked log(candidate count) (r = 0.94); at 0.02, r = 0.54 | **user decision after review M1**, before the real run |
| Inflection SD 2.0 added to the tuning grid | the toy check showed SD 0.4 makes known-lexeme LDL copy; added before any real tuning | pre-run |
| Tuning seed | a separate semantic seed (fold −1), so tuning never shares semantics with a test fold (review m3); re-run, same choice | pre-run |
| Citation rule | NFIN is always known and never tested; it is the selector's candidate representation | declared |
| Multithreading | selector: 6 processes × 1 thread; downstream LDL and tuning: 4 processes × 2 threads (8 total). JudiLing threads only candidate evaluation in `learn_paths`; the path search is serial | config |
| Commit | the pre-run tree was committed as `9c4ee53` on `pcfp-v1` (not pushed), so all manifests point to clean code | user approval |

## 7. Review and verification

An `ldl-reviewer` pass covered leakage, target-dependent decoding, budget accounting,
GeLaTo correspondence and uncertainty.

**Fixed:**

* **M1 (major):** entropy was degenerate at T = 0.1. Fixed with T = 0.02, the user's
  decision.
* **M2 (major):** the run would have been made from an uncommitted tree. Fixed by
  committing it.
* **m2:** decoder errors are now logged separately from no-candidate cells.
* **m3:** separate tuning seed.
* **m4:** a tighter audit, with tests that inject leaks (a hidden cell in training, a
  candidate in the selector's fit, an extra candidate column, a query for a shown cell)
  and check the audit catches each one.
* **m5:** gold is rewritten together with the queries.
* **m6:** wording of the many-to-one join.
* **m7:** capacity is checked for every pool cap.
* **m10:** unit-filtered tuning is refused.

**Documented, not fixed:**

* **m1:** candidates whose shown cells include NFIN look certain (see §4.3).
* **m8:** leakage-group doublets. This has no effect in repetition 0.
* **m9:** an unlogged typology identifier fallback.
* **m11:** exposure re-derivation depends on numpy's random stream; the numpy version is
  pinned in `requirements.lock`.
* The typology "declared == used" check is partly vacuous.
* The links table has no `confirmed_date` column, although `apply_review` requires both
  `confirmed_by` and `confirmed_date`.

**Checks run:**

* The smoke config ran end to end (audit: 66 checks, 0 problems), before and after the
  review fixes.
* The real run's audits: 98 and 102 artifact sets, 0 problems.
* 151 tests pass, including real-Julia tests for:
  * a rank-one update equal to a full refit (< 1e-8);
  * selector determinism and order-independence;
  * only the citation form affecting scores;
  * known-lexeme isolation;
  * chunk- and order-invariance;
  * a gold-free max_t.

## 8. Runtime (Apple Silicon, 6 performance + 12 efficiency cores)

| Stage | Wall time |
|---|---|
| splits | 5 s |
| ldl_tune (16 fits) | 2.5 min |
| select (18 active + 6 random runs; 6 processes) | 35 min; a selector round takes 80–365 s |
| audit | 6 s |
| ldl (24 runs × 2 item sets; 4 processes × 2 threads) | 15 min |
| evaluate, outcomes, typology, audit | 45 s |
| **Total** (excluding the data stage) | **54 min** |

## 9. Limitations

* **Random exposure.** Which forms of a verb are known is uniform at random, not
  frequency-based. Real learners see frequent cells first, such as the 3SG present and
  the infinitive. Accuracy depends strongly on k (§3.4), so the absolute accuracy level
  is a property of this exposure model.
* **Single-word cells only.** Periphrastic and compound cells are outside the task. In
  Italian these are the compound tenses; in Finnish, the perfect and pluperfect. The
  outcome measures synthetic inflection only.
* **Reliance on the citation form.** The selector sees only the citation form (NFIN), and
  every verb is known through it. The design assumes a citation form that is present and
  single-word. Of the PCFP units, this holds for Italian and Finnish; it does not hold
  for every language. NFIN-shown candidates look certain (review m1).
* **Additive semantics.** Simulated meanings are L + ΣV with no feature interactions.
  Exponents that are specific to one feature combination, such as the Italian 1SG
  present *-o*, are learned poorly (§3.5). LDL predictability therefore partly reflects
  how linearly the cell features map to exponents, not only how predictable the
  paradigm is.
* **Low absolute accuracy.** Accuracy is 11–17%, and 33–41% of gold forms are
  unreachable from a 200-verb sample's cues. Differences between languages in LDL
  accuracy will mix inflectional predictability with cue-inventory coverage.
* **One repetition, two languages, budget 100.** The intervals are conditional on the
  fitted fold models and the single selection run. Finnish fold variability is larger
  than the policy effect.
* **Leakage-group doublets** (m8): no effect in repetition 0, but the rule should be
  tightened before more repetitions.
* **Typology.**
  * The links are unconfirmed candidates.
  * The group map-downs need review (§5).
  * The clitic flags are heuristics with known false positives.
  * 116 GeLaTo languages, including Spanish, German and Yoruba, have no Grambank entry.

## 10. Implications and next steps (proposals; nothing here is decided)

1. **The LDL outcome does not need active selection.** At this budget,
   uncertainty-based selection is no better than random, and random is simpler and
   unbiased. A core-only or random-selected training sample is the natural default for
   the secondary outcome. Whether to drop the selector from the outcome pipeline is the
   user's decision.
2. **Before extending LDL to more languages,** consider frequency-weighted exposure and a
   semantics with cell-level interaction terms (for example an additional
   V(cell-combination) vector). Both would change the estimand, so either needs its own
   declaration.
3. **Human review of the GeLaTo–Grambank links** (`confirmed_by`, `confirmed_date`),
   starting with the 12 map-downs and the 15 clitic flags.
4. **Add a `confirmed_date` column** to the typology links table, and log the identifier
   fallback (m9).

---

*Files:*

* `outputs/pcfp_v1/outcomes/ldl_outcomes.csv`, `paired_differences.csv`;
* `outputs/pcfp_v1/eval/*` (summaries, copy rates, selector checks, mapping quality);
* `outputs/pcfp_v1/typology/*` (grambank_inflection, population links, coverage
  summary, missing list, LDL overlap);
* `analyses/pcfp_v1_2026_10_08/*` (report tables).
