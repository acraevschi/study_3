# First LDL run on the cell grid (pcfp_grid_v1, 2026-10-15)

**Exploratory (D33).** The mappings are `reviewed`, not approved, and were not frozen
before this run.

## Setup

* Config: `configs/pcfp_grid_v1.yaml`. Run outputs: `outputs/pcfp_grid_v1/`.
* Summary: [summarize_ldl.py](summarize_ldl.py), which writes the CSVs in this folder.
* **Cells.** Each language has 15 cells: the 14 grid cells plus the citation form, which
  is tested like any other cell (D36).
* **Forms.** Auxiliary cells use option B, "auxiliary + verb" (D35).
* **Design.** As `pcfp_v2`:
  * 100 core + 100 extra training verbs per fit;
  * 5 folds × 5 draws = 25 fits per language;
  * 650 eligible verbs per language (D37).
* **LDL settings.** Chosen by tuning on verbs outside the inventory: bigram cues,
  inflection SD 1.0. The cell vector is off (`sem_sd_cell = 0`), as in `pcfp_v2`.
* **Runtime.** Tuning 4.5 min, 125 fits 11 min. The audit found 0 problems.
* **Not read.** No population, GeLaTo or ancestry file was read; the registry, GeLaTo,
  outcomes and typology stages were not run.

## Accuracy (core items; 95% CI from a cluster bootstrap over core verbs)

| | items | accuracy [95% CI] | edit distance | items copyable from a shown form |
|---|---|---|---|---|
| English (phonemic) | 27,205 | **0.737** [0.717, 0.757] | 0.32 | 0.77 |
| Eastern Armenian | 27,785 | **0.303** [0.285, 0.320] | 2.23 | 0.00 |
| Serbo-Croatian | 27,400 | **0.246** [0.233, 0.259] | 2.36 | 0.02 |
| Finnish | 27,700 | **0.222** [0.207, 0.238] | 1.79 | 0.05 |
| Italian | 27,755 | **0.218** [0.205, 0.231] | 3.51 | 0.07 |

The spread between draws is small: SD 0.2–1.1 points (`ldl_accuracy_by_draw.csv`).

## By cell type

| | citation | imperative | plain PRS | plain PST | auxiliary block |
|---|---|---|---|---|---|
| eng | 0.86 | 0.85 | 0.66 | 0.76 | – |
| fin | 0.09 | 0.12 | 0.17 | 0.34 | – |
| hbs | 0.46 | 0.31 | 0.19 | – | 0.25 (PST) |
| hye | 0.48 | 0.20 | – | 0.30 | 0.31 (PRS) |
| ita | 0.46 | 0.42 | 0.19 | – | 0.14 (PST) |

## Reading

1. **Option B removed the trivial copies.** No item in an auxiliary block can be copied
   from a shown form (copyable share 0.00). Under option C the six participle cells
   would have been copies of each other.
2. **English is high because of syncretism, as the grid intends.**
   * 77% of its test items equal a form the model was shown for the same verb, and these
     are 93% correct.
   * The remaining 23% are 11% correct, which is lower than any other language's
     non-copyable items (Italian 0.18, Finnish 0.20, Serbo-Croatian 0.23, Armenian 0.30).
3. **In the auxiliary blocks, the model fails on the auxiliary, not on the lexical verb.**

   | | auxiliary right | lexical part right | whole form right |
   |---|---|---|---|
   | Italian | 0.16 | 0.52 | 0.14 |
   | Serbo-Croatian | 0.29 | 0.42 | 0.25 |
   | Armenian | 0.32 | 0.65 | 0.31 |

   * The auxiliary is the same in every verb (Italian: per *essere/avere* class), but it is
     specific to one person × number combination.
   * Under additive semantics (lexeme + Σ feature vectors, no cell vector), such
     cell-specific exponents are learned poorly. This is the limitation found in
     REPORT_pcfp_v1 §3.5 and in `analyses/ldl_ablation_2026_10_09`. There, adding a cell
     vector doubled accuracy (Italian 0.21 → 0.41, Finnish 0.13 → 0.27).
   * Italian also often drops the auxiliary altogether: 44% of its predictions in this
     block are one word.
   * The same limitation hits synthetic person endings (plain PRS 0.17–0.19), so it is not
     specific to option B. It is larger for auxiliaries, which are whole words tied to a
     cell.
4. **The Finnish citation form is hard (0.09).** Consonant gradation and the *-a/-ä,
   -da/-dä, -la, -ta* infinitive types are not predictable from a few finite forms
   without more training.

## Baselines on the same items ([baselines.py](baselines.py), 2026-10-15)

Both baselines see only what LDL saw in each fit: the shown forms of that fit's 200
training verbs.

* **copy:** predict the shown form of the cell that is most often identical to the target
  cell among the training verbs.
* **rules:** an analogy baseline. It learns edit rules from training verb pairs (a prefix and
  a suffix change around the longest common substring), conditioned on the source form's
  ending.

The whole baseline script runs in 9 seconds.

| | LDL | copy | rules |
|---|---|---|---|
| English | 0.737 [0.717, 0.757] | 0.768 [0.746, 0.791] | **0.919** [0.906, 0.934] |
| Armenian | 0.303 [0.285, 0.320] | 0.000 | **0.946** [0.932, 0.958] |
| Serbo-Croatian | 0.246 [0.233, 0.259] | 0.022 | **0.872** [0.853, 0.890] |
| Italian | 0.218 [0.205, 0.231] | 0.064 | **0.784** [0.768, 0.798] |
| Finnish | 0.222 [0.207, 0.238] | 0.049 | **0.739** [0.720, 0.759] |

The analogy baseline beats LDL by 18–64 points in every language, and the copy baseline
beats it in English. The two give a similar ranking. Under the rule baseline it is
hye > eng > hbs > ita > fin, which is linguistically plausible. LDL compresses the
differences and puts English first. As set up here, the LDL accuracy is dominated by model
limitations, not by the languages.

## Sensitivity runs (2026-10-15)

* **`pcfp_grid_v1_cellvec`:** adds the cell vector, re-tuned. The tuning chose
  `sem_sd_cell` 2.0, `sem_sd_inflection` 1.0 and bigram cues.
* **`pcfp_grid_v1_gaps`:** leaves out the auxiliary cells, so ita, hye and hbs have 9
  cells. English and Finnish are unchanged and reproduce the main run exactly.
* Both audits found 0 problems.

| | LDL additive | LDL + cell vector | LDL, gaps | rules (15 cells) | rules (gaps) |
|---|---|---|---|---|---|
| English | 0.737 | 0.755 | 0.737 | 0.919 | 0.919 |
| Armenian | 0.303 | 0.547 | 0.443 | 0.946 | 0.989 |
| Serbo-Croatian | 0.246 | 0.487 | 0.351 | 0.872 | 0.920 |
| Italian | 0.218 | 0.316 | 0.416 | 0.784 | 0.768 |
| Finnish | 0.222 | 0.324 | 0.222 | 0.739 | 0.739 |

Accuracy in the auxiliary block (with the rest of the grid in brackets):

| | LDL additive | LDL + cell vector | rules |
|---|---|---|---|
| Italian (aux first) | 0.14 (0.27) | **0.17** (0.41) | 0.89 (0.71) |
| Serbo-Croatian (aux after) | 0.25 (0.25) | 0.58 (0.43) | 0.91 (0.85) |
| Armenian (aux after) | 0.31 (0.30) | 0.62 (0.50) | 0.99 (0.92) |

Reading:

* **The cell vector helps the most where cell-specific material comes last.** In
  Serbo-Croatian and Armenian, the auxiliary block rises from about 0.3 to about 0.6.
* **It does not fix the Italian auxiliary, which comes first** (0.14 → 0.17). The
  sequence decoder seems unable to pick the cell-specific start of a form.
* **For the rule baseline, the auxiliary blocks are the easiest cells** (0.89–0.99),
  as a speaker would expect.
* **The LDL scores depend strongly on model choices.** The cell vector, the auxiliary's
  position, and the number of cells (the gaps run shows more of each verb) each move
  LDL by 10–25 points.
* **The rule baseline is stable** across all three variants.
* **Rankings largely agree.**
  * LDL + cell vector: eng > hye > hbs > ita ≈ fin.
  * Rules: hye > eng > hbs > ita > fin.
  * They differ only in English vs Armenian. LDL compresses the differences.

## Gaps + cell vector (`pcfp_grid_v1_cellvec_gaps`, 2026-10-19)

This run completes the 2×2 {full grid, gaps} × {additive, cell vector}.

* **Config.** `configs/pcfp_grid_v1_cellvec_gaps.yaml`: the gaps grid of
  `pcfp_grid_v1_gaps` with the cell-vector tuning grid of `pcfp_grid_v1_cellvec`.
* **Same items.** The exposure manifests are byte-identical to `pcfp_grid_v1_gaps`, so
  the two gaps runs are paired item for item.
* **Tuning.** Bigram cues, inflection SD 1.0, and `sem_sd_cell` **1.0** (2.0 on the
  full grid). Tuning is pooled over languages, so English and Finnish move slightly
  although their data are unchanged.
* **Audit and runtime.** The audit found 0 problems. Tuning took 7 min, the 125 fits
  4 min.

LDL accuracy, all test items (core; 95% CI from the cluster bootstrap):

| | full, additive | full + cell vector | gaps, additive | **gaps + cell vector** | rules (gaps) |
|---|---|---|---|---|---|
| English | 0.737 | 0.755 | 0.737 | **0.758** [0.737, 0.778] | 0.919 |
| Armenian | 0.303 | 0.547 | 0.443 | **0.602** [0.574, 0.632] | 0.989 |
| Serbo-Croatian | 0.246 | 0.487 | 0.351 | **0.459** [0.435, 0.484] | 0.920 |
| Italian | 0.218 | 0.316 | 0.416 | **0.520** [0.492, 0.547] | 0.768 |
| Finnish | 0.222 | 0.324 | 0.222 | **0.329** [0.310, 0.348] | 0.739 |

The cells shared by all four runs in ita, hye and hbs (everything except the auxiliary
block). The full-grid columns are the "rest of the grid" figures above:

| | full, additive | full + cell vector | gaps, additive | gaps + cell vector |
|---|---|---|---|---|
| Italian | 0.27 | 0.41 | 0.42 | 0.52 |
| Serbo-Croatian | 0.25 | 0.43 | 0.35 | 0.46 |
| Armenian | 0.30 | 0.50 | 0.44 | 0.60 |

Reading:

* **Without the auxiliary cells, the cell vector adds about 10–16 points.**
  * Italian +10, Serbo-Croatian +11, Armenian +16, against Finnish +11 and English +2.
  * On the full grid it added +24 overall in Serbo-Croatian and Armenian: +34 and +31
    in the auxiliary block, and +18 and +20 in the rest of the grid.
  * So the cell vector is a general improvement, not only an auxiliary fix. Its larger
    effect in hbs and hye on the full grid is partly a property of option B.
* **The two changes add up roughly; they do not substitute for each other.** The best
  LDL condition is gaps + cell vector in all three auxiliary languages.
* **Even with the cell vector, removing the auxiliary cells helps the other cells**
  (Italian +11, Serbo-Croatian +3, Armenian +10 against the full-grid rest). The comparison is confounded: with 9 cells and at most 7
  shown, a larger share of each verb is visible.
* **LDL is still far below the rule baseline** (by 16–39 points outside English, and
  below the copy baseline's 0.77 in English).
* **Rankings.**
  * LDL: eng > hye > ita > hbs > fin.
  * Rules (gaps): hye > hbs ≈ eng > ita > fin.
  * Compared with the full-grid cell-vector run, Italian and Serbo-Croatian swap places
    under LDL; the rule baseline keeps hbs above ita in both grids.

## Open (owner decision)

* **Which outcome measure to use:**
  1. **LDL**, after fixing the setup (cell vector on, the decoder for form-initial
     material), with the rule baseline reported next to it as a ceiling check.
  2. **An analogy learner** as the predictability measure, with LDL as a robustness
     check.
  3. **LDL relative to the baseline.**
* The rule baseline here is deliberately simple. A standard PCFP learner (e.g. a
  SIGMORPHON-style baseline) would be a stronger, citable reference.
