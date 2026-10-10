# Report: LDL predictability under repeated random draws (pcfp_v2)

> **Preliminary** (work in progress; see the README banner). The LDL settings used here
> were revised by the 2026-10-09 ablation (`analyses/ldl_ablation_2026_10_09/`).

Date: 2026-10-08. Config: `configs/pcfp_v2.yaml`. Outputs: `outputs/pcfp_v2/`.

The design is declared in [PROTOCOL.md](PROTOCOL.md) §4b. Everything except the
training-sample design is as in pcfp_v1 ([REPORT_pcfp_v1.md](REPORT_pcfp_v1.md)): the
task, cells, exposure draws, LDL protocol, tuning procedure and typology feature set.

**Status.** The full chain ran: splits → ldl_tune → select → audit → ldl → evaluate →
outcomes → typology → audit.

* Both audits report **0 problems**: 52 artifact sets before the LDL fits, 56 at the end.
* 163 tests pass, including 11 new tests for this design.
* No admixture–morphology association was fitted, and no ancestry values were read.

**Provenance caveats.**

* **Uncommitted code.** The run used code not yet committed: the manifests record branch
  `pcfp-v1` at `9c4ee53` with `pipeline_dirty: true`; the source hash is
  `ee4ca9ae…`.
* **Config edited mid-run.** `configs/pcfp_v2.yaml` was edited during the `ldl` stage by
  a parallel session, which added `typology.dialect_entries: substitute` (see §5):
  * splits through ldl recorded config hash `3b9a260e3164ef7b`;
  * evaluate through the final audit recorded `39844bc4fe8d8c0f`.

  The edit touches only the `typology` block, which no LDL stage reads.

## 1. Design

| | pcfp_v1 (`core_selected`) | **pcfp_v2 (`repeated_random`, default)** |
|---|---|---|
| Core verbs | 3 folds × 100 | **5 folds × 100**, fixed across draws (`core_split` seed) |
| Extra training verbs | per fold: seed 20 + 80 acquired (active or random) | **5 independent random draws of 100**, each shared by all 5 folds (`random_draw` seed) |
| LDL fits per language | 3 per policy run (12 in all) | **25** (5 draws × 5 folds) |
| Test items | hidden cells of the 300 core verbs, plus the selected verbs' cells | hidden cells of the 500 core verbs, identical in every draw |
| Semantic seed | per repetition and fold | **per fold** (`semantic_seed_scope: fold`), so draws differ only in their training verbs |
| Parallelism | 4 processes × 2 threads | 8 processes × 1 thread |

More detail:

* **Inventory and pool.** The inventory is the same 1,200 verbs per language as in
  pcfp_v1. After the 500 core verbs, 700 remain, and the draws are taken from those.
* **Draw overlap.** Pairs of draws share 8–20 verbs; chance is about 14.
* **Test items per draw.** Italian 21,570 (43.1 per verb); Finnish 15,000 (30.0 per
  verb). The mean number of shown cells k for core verbs is 3.8–4.4 per fold.
* **Training sample.** Each fit trains on 200 verbs: Italian about 790 forms, Finnish
  about 800.
* **LDL settings.** `ldl_tune` re-chose bigram cues with inflection SD 2.0 on the same
  auxiliary verbs as pcfp_v1. Held-out tuning accuracies differ from pcfp_v1 by at most
  0.0009. The cause is the BLAS thread count (1 here, 2 there), which changes
  floating-point summation order; the choice is the same.

## 2. Result

### 2.1 Pooled outcome (25 fits per language; core items)

| Unit | Accuracy (micro) [95% CI] | Verb-macro | Edit distance | Copy rate (shown form) | Copy rate (citation) |
|---|---|---|---|---|---|
| Italian | **0.171** [0.163, 0.180] | 0.173 | 2.43 | 0.287 | 0.002 |
| Finnish | **0.135** [0.127, 0.144] | 0.138 | 2.53 | 0.149 | 0.003 |

The interval comes from a leakage-group cluster bootstrap. It resamples core verbs
together with their predictions in all 5 draws, so it reflects the sampling of core verbs
and is conditional on the 25 fitted models.

These estimates agree with the pcfp_v1 random policy, which had different core verbs and
3 folds: Italian 0.172 [0.161, 0.184] and Finnish 0.122 [0.110, 0.134]. The Finnish
difference is within the fold-to-fold spread in both runs.

### 2.2 Robustness: how much does the random draw matter?

Accuracy per draw, pooled over the 5 folds, on identical items:

| Draw | 0 | 1 | 2 | 3 | 4 | SD | Range |
|---|---|---|---|---|---|---|---|
| Italian | 0.169 | 0.173 | 0.170 | 0.176 | 0.169 | **0.31 pt** | 0.7 pt |
| Finnish | 0.139 | 0.135 | 0.141 | 0.129 | 0.132 | **0.50 pt** | 1.2 pt |

Two-way decomposition of the 25 fit accuracies (`draw_variability_summary.csv`):

| Unit | SD of draw effects | SD of fold effects | Residual SD | Bootstrap CI half-width |
|---|---|---|---|---|
| Italian | 0.31 pt | 1.62 pt | 0.65 pt | 0.87 pt |
| Finnish | 0.50 pt | 1.19 pt | 0.47 pt | 0.86 pt |

**Reading.**

* Which 100 random verbs complete the sample matters little.
  * The draw-to-draw SD is about a third (Italian) to a half (Finnish) of the bootstrap
    half-width.
  * It is several times smaller than the spread between folds. Folds differ in their
    core verbs and semantic seed.
* With 5 draws, the draw contribution to the pooled estimate is about SD/√5 = 0.14 pt
  (Italian) and 0.22 pt (Finnish), well below the core-verb sampling uncertainty.
* Hence, for the LDL outcome, one random draw would already give a stable estimate, and
  5 draws make the draw contribution negligible. The larger source of variation is which
  core verbs are tested. More core verbs, or more folds, would narrow the interval more
  than more draws would.
* **This bears on pcfp_v1.** Its active − random differences were about 1 pt from a
  single selection run.
  * Italian (−0.9 to −1.1 pt) is 3–4 draw SDs, so it is unlikely to be draw noise alone.
  * Finnish (−0.8 to −1.4 pt) is within about 1.5–3 draw SDs. Part of it may be the luck
    of one random draw.

  The conclusion that active selection does not help stands in both languages.

### 2.3 Breakdowns (pooled over draws)

**By k (shown forms):**

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Italian | 0.086 | 0.131 | 0.165 | 0.198 | 0.199 | 0.223 | 0.217 |
| Finnish | 0.071 | 0.092 | 0.113 | 0.142 | 0.147 | 0.189 | 0.199 |

**By cell:**

| Unit | Weakest cells | Strongest cells |
|---|---|---|
| Italian | 3SG *passato remoto* (0.011), 1SG present (0.017), 3PL conditional (0.064) | the singular subjunctives (0.38–0.39), which are syncretic within each tense |
| Finnish | present passive indicative (0.038), 3PL and 1PL imperatives (0.05) | 3SG past (0.28), 1PL conditional (0.25), 1PL past (0.22) |

The pattern is the same as in pcfp_v1. The cell-specific Italian endings (*-o*, *-ò*)
remain the additive-semantics limitation described in REPORT_pcfp_v1 §3.5.

**Mapping quality:**

| | Italian | Finnish |
|---|---|---|
| cor(Ĉ, C) | 0.76 | 0.72 |
| Gold in the decoder's top candidates | 0.21 | 0.18 |
| Training production accuracy | 0.93 | 0.83 |

## 3. Runtime (8 Julia processes × 1 thread)

| Stage | Wall time |
|---|---|
| splits | 6 s |
| ldl_tune (16 fits) | 2 min 15 s |
| select (writes 50 samples; nothing fitted) | 13 s |
| ldl (50 fits: 87 s Italian, 121 s Finnish per fit) | 13 min 12 s |
| evaluate, outcomes, typology, audits | 45 s |
| **Total** (excluding the data stage) | **16.5 min** |

For 70 languages at these sizes, the LDL fits would take roughly 70 × 25 × 100 s / 8
processes ≈ 6 h on this machine.

## 4. What did not change

The active-selection code (LDL selector, rank-one candidate scoring, acquisition loop,
selector checks) is unchanged and selectable with `cv.design: core_selected`.
`configs/pcfp_v1.yaml` now states this explicitly. That added key changes its config
hash, while the pcfp_v1 outputs keep the original `8b088277dfaa1e26`. Its smoke config
still passes its audit (66 checks, 0 problems).

## 5. Typology in this run

The typology stage ran under the parallel session's rule change,
`typology.dialect_entries: substitute`. Under this rule, a language with no
language-level Grambank entry is represented by its best-covered Grambank dialect entry;
[TYPOLOGY.md](TYPOLOGY.md) §2 has the details.

* **Changed by the rule:** three GeLaTo-linked languages now enter, all at full or
  near-full coverage:
  * Karelian, via Northern Karelian (14/35);
  * Selkup, via Southern Selkup (15/35);
  * Terena-Kinikinao-Chane, via Kinikinao (11/34).
* **Counts:**
  * 195 non-proxy languages are in Grambank and 175 meet the 60% coverage threshold;
    pcfp_v1 had 193 and 173;
  * 113 are missing, against 116.
* **Linked-language split.** It moved by one, from 309 non-proxy / 41 proxy-only to
  308 / 42. Murut's group map-down became ambiguous once a second Murutic language was
  coded through a dialect entry (TYPOLOGY.md §11).

**Feature set: 12 inflectional categories** (user decision, 2026-10-08, after this report
was first written; TYPOLOGY.md §3 and §12). Grambank splits one category into several
logically dependent features: by role, affix position, value or host. A share of 35 raw
features therefore counts one property several times.

* **New measure.** The outcome now counts 12 categories. Each is the logical OR of its
  Grambank sources, the merge rule of the GBI curation (Graff et al. 2025, *Sci. Data*).
  * The categories are tense, aspect, mood, person indexing, negation, polar
    interrogation, nominal number, case, possessor affix, possessed-noun affix, gender
    agreement and number agreement.
  * GB079 and GB080 are dropped.
* **Effect on the language set.** The same 175 languages enter, and the new score
  correlates with the 35-feature share at Pearson 0.92 / Spearman 0.89.
* **No inflection.** The list is unchanged: Central Khmer, Naxi, She and Vietnamese, plus
  Thai below coverage.
* **Compression.** The scale measures breadth, so elaborate systems move closer together:
  Italian 8/12, Finnish 8/12, English 7/12 (19, 14 and 10 of 35 before). This loses
  resolution among inflecting languages.
* Typology was re-run with the categories. The audit reports 56 artifact sets and 0
  problems.

The check of whether "missing" languages are coded under another code or level is in
TYPOLOGY.md §11. None are, apart from these three dialect-coded languages.

## 6. Limitations

As in REPORT_pcfp_v1 §9 (random exposure, single-word cells, citation-form reliance,
additive semantics, low absolute accuracy), plus:

* **Repeated cores, not resampled.** The core sets are fixed across draws. The
  draw-to-draw spread therefore isolates the training-verb sample, but does not cover a
  re-partition of the core verbs. That variation is in the fold SD and the bootstrap
  interval.
* **Fixed semantic seed per fold.** The semantic seed is fixed per fold, so its
  variability is confounded with the core-verb sets in the fold effects.
