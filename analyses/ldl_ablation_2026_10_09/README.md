# LDL ablation (2026-10-09): why is PCFP accuracy low?

`ablation.py` varies three suspected causes of the low LDL accuracy, one stage at a time.
Each stage starts from the best setting of the stages before it.

* **Data.** It uses only the pcfp_v2 tuning sample: the shown forms of 200 auxiliary verbs
  outside the inventory. The scored items are the hidden cells of the 100 `tune_core`
  verbs: 4,321 Italian and 3,025 Finnish items.
* **Seed.** The tuning semantic seed is used.
* **No outer-test verb is read.**
* **Selection criterion.** As in `ldl_tune`: mean held-out accuracy over the two units,
  ties broken by lower edit distance.
* **Files.** Fits are in `outputs/ldl_ablation_v1/` (not committed). The tables are
  `<stage>_by_unit.csv` and `<stage>_summary.csv`. Input hashes are in `inputs.json`.

## Result

| Setting | Italian | Finnish | Mean | Copy rate | Gold among 10 candidates |
|---|---|---|---|---|---|
| pcfp_v2 (dim 1000, noise 1, λ 0.02, threshold 0.05) | 0.197 | 0.127 | 0.162 | 0.20 | 0.21 |
| + no noise (best meaning space) | 0.212 | 0.134 | 0.173 | 0.28 | 0.22 |
| + cell vector (SD 2) | 0.410 | 0.267 | 0.338 | 0.12 | 0.44 |
| + decoder threshold 0.02 | 0.438 | 0.324 | **0.381** | 0.05 | 0.55 |
| + tolerant decoding (1 weak n-gram), threshold 0.05 | 0.440 | 0.320 | **0.380** | 0.04 | 0.58 |

Copy rate is the share of predictions equal to a shown form of the verb.

### 1. Meaning space (`space`, 48 settings)

The grid was `sem_dim` {100, 200, 400, 1000} × `sem_sd_noise` {0, 0.5, 1} ×
`ridge_shift` {0.02, 1, 10, 100}.

* **It is not the bottleneck.** The hypothesis that the model memorises training rows
  because there are more dimensions than training forms is **not supported**:
  * smaller dimensions and larger ridge penalties lower accuracy, monotonically;
  * at λ = 100 and d = 100, accuracy falls to 0.07.
* The best setting, no noise at d = 1000, gains only 1.1 points over pcfp_v2.
* d = 400 is equivalent.
* The gold form is among the candidates for about 21% of items in every setting with
  λ ≤ 1.

### 2. Cell vector (`cell`, 7 settings)

* **This is the main cause.** Adding V(cell) next to the feature vectors doubles accuracy
  (0.173 → 0.338).
* **The size of the vector barely matters.** Every SD from 0.5 to 4 gives 0.330–0.338, so
  the effect comes from having combination-specific meaning at all.
* **Cells only.** With no feature vectors (`sem_sd_inflection` 0), accuracy is 0.327–0.329,
  just below features plus cell.
* **Copying.** The copy rate of shown forms halves.
* **Per cell.** Accuracy improves in 46 of 47 Italian cells and all 34 Finnish cells.
  * The median cell accuracy rises from 0.20 to 0.47 (Italian) and from 0.14 to 0.31
    (Finnish), measured with the final decoder.
  * Italian 1SG present goes from 0.01 to 0.15.
  * Italian 3SG *passato remoto* goes from 0.00 to 0.20.

### 3. Decoder (`decoder`)

* **Threshold.** Lowering the `learn_paths` threshold from 0.05 to 0.02 adds 4 points. Raising
  it to 0.1 costs 9.
* **Tolerant mode** (threshold 0.05, at most one n-gram per path with support in
  (0, 0.05]) gives the same accuracy at a fraction of the runtime.
* **Runtime** (fit plus prediction for the 4,321 / 3,025 items, one process):

  | Decoder | Italian | Finnish |
  |---|---|---|
  | threshold 0.05 | 12 s | 11 s |
  | tolerant, 1 n-gram | 72 s | 54 s |
  | threshold 0.02 | 376 s | 536 s |
* **Settings not run to completion:**
  * Threshold 0.01 in both units, and tolerant mode with 2 n-grams in Finnish, were
    stopped after 60 minutes per fit.
  * Italian with 2 tolerated n-grams finished at 0.430, below 0.440 for 1.
* **`build_paths` was dropped.** It is JudiLing's decoder that chains the n-grams of the
  forms nearest to Ĉ.
  * With bigram cues its paths loop (*subissasubisatero*).
  * One Italian item took 129 s with 3 neighbours.
  * The runner now refuses it.

## Caveats

* **One sample, one seed.** Each number is one fit per unit, on one semantic seed and one
  100-verb tuning sample. Differences under about 1 point (threshold 0.02 vs tolerance)
  are within noise. The cell-vector effect (+16 points) and the decoder effect
  (+4 points) are not.
* **Settings chosen and scored on the same items.** The final numbers are therefore
  slightly optimistic. The outer-test core verbs were not used, so a declared pcfp_v3 run
  with these settings would give an unbiased estimate.
* **Interactions untested.** The stages are greedy: noise and dimension were not re-tuned
  with the cell vector, and inflection SD and cue size were not re-tuned at all.
* **Remaining error.** Accuracy is still 0.44 / 0.32. The gold form is among the
  candidates for 55–58% of items, and is ranked first in 62–74% of those.
