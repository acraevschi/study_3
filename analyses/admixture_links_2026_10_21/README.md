# Admixture → Grambank inflection on the 2026-10-20 links (prepared 2026-10-21)

**Exploratory, and not blind.** The admixture values were unblinded on 2026-10-10 (see
`analyses/draft_admixture_model_2026_10_10/README.md`), so nothing here is a blind or
pre-registered test. This folder lifts the ancestry firewall for itself only.

**Status 2026-10-22: fitted; results in §11.** The user saw the preparation first and gave
the go-ahead on 2026-10-21 for `ADM_XMODE=both` (§10). The full run (244 models) ran
2026-10-21 20:40 to 2026-10-22 03:11, the f3 sampling check (§4) 03:13–04:10. No model failed.

This is the successor of the draft model, which is left unchanged. It was built for 175
languages and the old links.

## 1. What changed against the draft

| Item | Draft (2026-10-10) | Here |
|---|---|---|
| Links | own-language links of `pcfp_v2`, unreviewed | `analysis_links.csv` of `pcfp_v2_links_2026_10_20`: rule-accepted links, Graff stand-ins, the reviewed decision sheet |
| Languages | 175 | 202 (main) |
| Sensitivity | none | the three link sensitivity sets (§2) |
| Population measures | no use of the number of individuals | bootstrap SE over each population's individuals enters stage 1 |
| New measures | | `het` (§3), `f3_z`, `f3_est` (§4) |

`pcfp_v2` in the old path was only the experiment name under which the typology stage wrote
its tables; the draft used none of the LDL work.

## 2. Analysis sets

From `outputs/pcfp_v2_links_2026_10_20/typology/analysis_links.csv` (links to a language in
the outcome set). A sensitivity set drops links, so a language can stay with fewer
populations. Stage 1 is refitted for every set.

| Set | Drops links flagged | Languages | Populations | Individuals |
|---|---|---|---|---|
| `main` | nothing | 202 | 335 | 2,943 |
| `no_doubtful_language` | endangered (AES ≥ 4), diaspora, wrong_code, group_choice | 174 | 293 | 2,587 |
| `no_other_language_stand_ins` | stand_in_other_language | 193 | 319 | 2,838 |
| `strict` | all of the above | 169 | 281 | 2,512 |

Individuals here are those in the ADMIXTURE files (4,768 in all, relatives removed); the
sample sizes in the link tables come from GeLaTo and are a little larger (2,962 in `main`).
Outcome: `n_present` of `n_coded` of the 12 inflection categories, unchanged by the links.

## 3. Admixture measures, and what each could reflect

All values come from the ADMIXTURE best runs of Graff et al. 2025 (Human Origins,
K = 12…30), unless stated. Genes record who had children with whom, not who learned which
language; the readings below are hypotheses to test.

| Measure | Definition | Possible reading |
|---|---|---|
| `ent` (K-averaged), `ent_K12`, `ent_K30` | Shannon entropy of each individual's ancestry proportions | long-term diffuse mixing, any depth. Also high on genetic clines, which is not contact |
| `nonmax` | 1 − largest component, per individual | the same, cruder |
| `pop_ent` | entropy of the population's mean ancestry | as `ent`, plus differences between individuals |
| **`het`** (new) | `pop_ent` − mean individual entropy, times n/(n−1); populations with ≥ 2 individuals | unmixed people of different ancestry living side by side: recent or ongoing incoming groups, the closest genetic sign of L2 speakers being present now |
| `support`, `amount` | Graff's rule: the population looks like a clean two-way mix (`support`), and the minor source's share (`amount`) | one group absorbed into another; the minor share as a stand-in for the share of incoming speakers |
| `graff_target` | the population is named as a curated cross-family admixture target in Graff et al. Table S2 (81 populations, 65 languages) | a published contact event; binary, so weakly identified per language. Names are matched exactly; S2's "Greek" and "Khaskas" are samples of other studies (Hellenthal 2014, Flegontov 2016), not our Greek_* and Khakass_* populations, so these stay 0 |
| `neigh_flow` | −log median FST to populations within 1,000 km (GeLaTo) | ongoing gene flow with neighbours, i.e. general contact. Missing for 110 of 335 populations |
| **`f3_z`, `f3_est`** (new) | §4 | evidence of mixing with another family |

`het` is a Jensen gap and is ≥ 0. The factor n/(n−1) undoes the plug-in bias that makes
small samples look homogeneous. It is exact only to second order, and n = 2 stays noisy.
Seven single-individual populations have no `het`.

Spearman correlations between population measures (`prep_summary.json`): `ent`, `nonmax` and
`pop_ent` ≥ 0.93; `het` 0.2–0.4 with them; `support` and `amount` are 0.8 with each other
and negative with entropy; `f3_z`/`f3_est` correlate 0.6 with entropy.

## 4. f3 (`f3.R`)

Needs the R package `admixtools` (GitHub tag v2.0.9, installed 2026-10-20) and the
precomputed F2 blocks of Graff et al. (`F2results`, 3.6 GB, in `~/Downloads`, not in the
repository). No genotypes are used.

For each of the 335 populations A: f3(A; B, C) for every B in A's Glottolog family (not A) and
every C of another family within 3,000 km of A. A negative f3 means A looks like a mix of a
B-like and a C-like group. Per population we keep the triple with the lowest Z.
* `f3_z` = −Z of that triple, `f3_est` = −f3 of that triple (higher = more evidence).
* The tables also keep the number of triples, the number with Z < −3, and the pair found.

Checks and limits:
* Finnish: the best triple is Nganasan_Volochanka × Spanish_Basque, Z = −10.9. Graff et al.
  report −9.9 for their curated pair.
* 316 of 335 populations have at least one triple; 217 of them have Z < −3. 19 have none:
  18 have no relative in their family in the F2 set (isolates and small families: Japanese,
  Korean, Burusho, Ticuna, …) and Tahiti has no other-family population within 3,000 km.
  They are left out of the `f3` models.
* Coordinates: the Graff mapping, then GeLaTo, then the Glottolog point of the population's
  base language, then (targets only) the Glottolog point of the language it links to; the
  last step was not needed in this build (`coord_source` in `f3_scores.csv`). The first version of this script
  dropped four targets (Assyrian, Cameroon_Bangwa, Mongol_Minorities, Yemen_Marib) for want of
  coordinates; found by the review of 2026-10-21 and fixed. A Glottolog point is coarser
  than a sampling site; it matters only for the 3,000 km window.
* Taking the lowest Z over many triples favours populations with many triples
  (Spearman −0.32 between triple count and best Z). `n_triples` is in `populations.csv`.
* Sampling check (2026-10-21, asked by the user after the main run started): the score
  depends on how well the panel samples a population's family. Of the populations with 1–2
  relatives in the panel, 17% have a best Z < −3; with 3–10 relatives 53%; with more than 10,
  78%. A population that looks isolated only because its relatives were not sampled therefore
  looks unmixed. Two variants were fitted on the main set (§11):
  * `f3k_z`, `f3k_est`: only the 5 nearest B and the 5 nearest C (`f3.R ... 3000 5` →
    `f3_scores_k5.csv`), so at most 25 triples per population instead of up to 13,298. The
    link between score and number of relatives drops from Spearman 0.21 to 0.07; 26% of
    populations instead of 69% have a best Z < −3. Same 316 populations as `f3`.
  * `f3b3_z`, `f3b3_est`: the ordinary scores, only for populations with at least 3 relatives
    in the panel (292 populations, 165 languages).
* Z grows with the precision of the data, so it is partly a sample-size measure; `f3_est`
  is the one that does not.
* A negative f3 shows mixing. A positive one does not show its absence: drift hides it.
  As a continuous score, `f3_z` ranks all 316 populations: about 60 have a best Z above 0
  and about 100 above −3. In that lower range the score mostly reflects drift, so a slope
  can be driven by drifted populations rather than by admixed ones. A truncated score
  (`max(0, f3_z)`) or the indicator "any triple with Z < −3" is the natural check.
* The F2 files give a warning about negative f2 estimates in populations with few
  individuals (admixtools suggests `afprod = TRUE`). Graff et al. used the defaults, so did we.
* The 3,000 km and the other-family rule were fixed before looking at the outcome.

qpAdm (how *much* ancestry comes from a named source) was not run: it needs one hand-built
model per target. A check against Graff's curated cases is possible later.

## 5. Certainty from the number of individuals

Stage 1 gives each language an exposure with a posterior mean and SD; stage 2 passes both in
as measurement error (`me(x, x_sd)`); whether that is the right treatment is an open decision (§10).

| Measures | How the number of individuals counts |
|---|---|
| `ent`, `nonmax`, `ent_K12`, `ent_K30` | individuals are the data: `value_i ~ 1 + (1 | lang) + (1 | lang:population)`. More individuals, narrower SD |
| `pop_ent`, `het`, `support`, `amount` | one value per population with its bootstrap SE over the individuals (400 resamples): `value_p | se(se_p, sigma = TRUE) ~ 1 + (1 | lang)`. In the draft these had no SE |
| `graff_target`, `neigh_flow`, `f3_z`, `f3_est` | one number per population, **no SE**: the number of individuals does not enter. For FST and f3 there is no simple bootstrap |

A bootstrap from fewer than 3 individuals is unreliable. There the SE is the pooled
per-individual SD (median over populations with ≥ 3) divided by √n, as in the draft.
Not modelled: ADMIXTURE's own estimation error, and the choice of K (averaged over 12–30).
No minimum sample size is applied (user decision 2026-10-20).

## 6. Models (`fit.R`)

As in the draft (README §3 and §7 there): beta-binomial on (`n_present`, `n_coded`);
`me(x, x_sd)` with x the standardised language exposure; controls N (none), P (+ phylogeny,
Glottolog tree with Grafen lengths, correlation matrix from the 202-language tree), S (+
exact Gaussian process on Glottolog point coordinates), F (P + S); same priors; PSIS-LOO
against the baseline with the same controls and the same languages.

| | Measures | Control sets | Languages |
|---|---|---|---|
| `main` | 12 | N, P, S, F | 202; 198 for `het`, 183 for `f3_z`/`f3_est`, 140 for `neigh_flow` |
| each sensitivity set | 12 | F only | 169–193; fewer for `het`, `f3`, `neigh` |

160 models in all (48 exposure, 112 outcome including baselines). In the draft the F models
took 5.5 minutes on average and the S models 3.5 (175 languages, exact GP, 2 fits at a time).
Here there are 64 F and 16 S models on up to 202 languages, so expect 4.5–5 hours (an
estimate, not timed). The run uses at most 12 cores: 3 fits at a time × 4 chains × 1 thread
(2026-10-21; first planned as 2 fits × 4 chains × 2 threads = 16, more than the 12 cores
allowed). With about 200 languages per model, separate fits side by side gain more than
threads inside a chain. A failed model is logged and skipped, a failed
set does not stop the next one, and fits are cached in `fits/`, so a rerun resumes.
Results go to `results/` per set; the fits to `fits/` (not for git). In the result files
`ctrl` reads none / phylogeny / space / phylogeny+space, and `xmode` says how stage 2 used the
exposure (§10). Each set standardises its own exposure, so a slope is "per SD" of that set.
`model_results_<set>.csv` therefore keeps `x_scale` and `y_sd`; `b / (x_scale × y_sd)` is the
slope per raw unit of the measure, which is the scale for comparing sets (not for `graff`,
whose stage 1 is on the logit scale). Rerunning a set overwrites its result files, so run each
set in one mode setting. Seeds are fixed, but brms threading is not bit-reproducible: a rerun agrees
up to MCMC noise.

## 7. Files and how to run

| File | Content |
|---|---|
| `f3.R` | f3 scores → `f3_scores.csv` (about 4 minutes); with a third argument k, the k-nearest variant → `f3_scores_k<k>.csv` |
| `prepare.py` | → `language_data.csv`, `individuals.csv`, `populations.csv`, `phylo_tree.nwk`, `prep_summary.json` (3 seconds) |
| `fit.R` | stage 1 and stage 2 per set |

```
Rscript analyses/admixture_links_2026_10_21/f3.R <path to F2results> 3000
.venv/bin/python analyses/admixture_links_2026_10_21/prepare.py
ADM_DRY=1 Rscript analyses/admixture_links_2026_10_21/fit.R     # validates all models, fits none
Rscript analyses/admixture_links_2026_10_21/fit.R [set ...]     # the real run
# as run on 2026-10-21/22:
ADM_XMODE=both Rscript analyses/admixture_links_2026_10_21/fit.R > .../fit.log 2>&1
Rscript analyses/admixture_links_2026_10_21/f3.R <path to F2results> 3000 5
ADM_XMODE=both ADM_MEASURES=f3k_z,f3k_est,f3b3_z,f3b3_est ADM_TAG=f3check \
  Rscript analyses/admixture_links_2026_10_21/fit.R main > .../fit_f3check.log 2>&1
```

`ADM_MEASURES` picks measures (default: the 12 of the main run); `ADM_TAG` adds a suffix to
the result files (`model_results_main_f3check.csv`, …) so they do not overwrite others.

The tables were built on the `gelato-link-rules` branch; they need its
`analysis_links.csv`.

## 8. Known limits (unchanged from the draft unless noted)

* Averaging over K = 12–30; one global GP length scale; one coordinate per language.
* Entropy is reference-free: it cannot tell mixing from position on a cline, and depends on
  which populations are in the ADMIXTURE panel.
* No time depth: old and recent admixture count the same.
* Phylogeny is a topology with Grafen lengths. Sensitivity sets use the sub-matrix of the
  202-language correlation matrix.
* PSIS-LOO is unreliable where the exposure is almost unidentified (`graff_target`).
* Skewed measures under a Gaussian stage 1: `het` (skew about 3, largest standardised value
  about 7), `f3_z`, `support`, `amount`. A log or square-root version is a possible check.
* Hypothesis check: the draft found positive entropy slopes and a negative `amount` slope. Any
  new result is compared with those, with 27 more languages and changed links.

## 9. Review (2026-10-21)

An independent review of the code (Opus subagent, read-only, dry run only) found no blocker.
Changes made after it:
* `fit.R`: the code after the fits had no error guard. One failed model or missing baseline
  would have stopped the run before the sensitivity sets. Now a failed model, LOO comparison
  or set is logged and skipped. Language subsets (`het`, `f3`, `neigh`) are built from the
  data, not from which stage-1 fit succeeded, and a measure whose exposure languages differ
  from its subset is skipped with a message.
* `prepare.py`: the parser for Table S2 target names split on commas inside a bracketed
  comment, so `Aleut_Tlingit` was missed. Fixed (one population changes: 80 → 81 targets).
* `f3.R`: the coordinate fallback (§4). Four more targets now have f3 scores; f3 and
  `populations.csv` were rebuilt, and every other measure is identical to the first build.
* `.gitignore`: the smoke-test folders.
* README: the wrong reason given for populations without triples, and the time estimate.
* Second round (same day, same reviewer, after the fixes above): the post-fit code was run end
  to end with the sampling stubbed (clean run, then injected failures, killed workers and
  cached fits from an earlier run). It found that LOO comparisons could use a cached fit of a
  model that had failed in the current run, that a killed worker was logged without its name,
  and that `ctrl` "F" is read back as FALSE in files with a single control set. All three
  fixed: comparisons need both models fitted in this run, workers are named, controls are
  written as words. The run was repeated with the same stubs and behaves as intended; what the
  stubs cannot test is real brms/CmdStan behaviour (compiling in forks, cached refits).
* Final look before the run (same day): slopes of different sets were on different "per SD"
  scales; the scaling constants are now saved so they can be compared (§6). Checked with the
  dry run in all three modes and the stubbed end-to-end run.

## 10. How stage 2 uses the exposure (`ADM_XMODE`)

Decision 2026-10-21 (user): `both`. What was observed is in §11.

The review found a statistical issue that comes from the draft. Stage 2 gets each language's
stage-1 **posterior mean** and passes its SD to `me(x, x_sd)`, which treats the mean as the
true value plus noise ("classical" error). A posterior mean is the other way round: the true
value scatters around it ("Berkson" error), and a plain regression on it is roughly unbiased.
`me()` therefore makes up for attenuation that is not there, and the slope comes out too large
by about 1 / (1 − mean(x_sd²)). With the draft's stage-1 values (`hier_exposure.csv`, 175
languages; this run's values will differ):

| Measure | mean(x_sd²) | slope too large by |
|---|---|---|
| `ent_K12` | 0.05 | about 5% |
| `ent`, `nonmax`, `pop_ent` | 0.09 | about 9–10% |
| `neigh` | 0.12 | about 13% |
| `amount`, `ent_K30` | 0.15 | about 18% |
| `support` | 0.26 | about 36% |
| `graff` | 1.02 | undefined: the claimed noise exceeds the variance of x, so x is treated as pure noise |

The draft's `graff` slope (−0.01, wide interval) therefore reflected the prior, not the data.

Options:
* `ADM_XMODE=me` (default, as in the draft): comparable with the draft; slopes inflated as
  above; `graff` unidentified.
* `ADM_XMODE=plain`: `x` only. Unbiased for posterior means. Exposure uncertainty enters only
  through the shrinkage of the means (small samples are pulled to the mean), not as extra
  width in the slope.
* `ADM_XMODE=both`: every with-x model in both forms; the baselines are shared. The
  difference between the two is itself informative. About 196 outcome models instead of 112,
  so roughly 7–8 hours instead of 4.5–5 (an estimate).
* Not built: refitting stage 2 on many posterior draws of the exposure and pooling, which
  propagates the uncertainty properly but multiplies the cost.

Sets can be run apart: `ADM_XMODE=both Rscript .../fit.R main`, then another mode for the
others. Compare a sensitivity set only with the main set in the same mode.

## 11. Results (2026-10-22)

Files: `results/model_results_<set>.csv`, `loo_x_vs_baseline_<set>.csv`,
`loo_controls_main.csv`, `exposure_<set>.csv`, `stage1_diagnostics_<set>.csv`; the f3 check
in the same files with the suffix `_f3check`. Logs: `fit.log`, `fit_f3check.log`, `f3_k5.log`; they and the fits (5.7 GB) stay local, not in git.

**Run.** 48 stage-1 and 196 stage-2 models, then 4 and 40 (36 new) for the f3 check, on at most
12 cores (3 fits × 4 chains). Times: main set 3 h, each sensitivity set 55–90 min, f3 check
1 h. An F model took 4–11 minutes.

**Diagnostics.** Stage 1: R-hat ≤ 1.014, bulk ESS ≥ 444, no divergences; one model had low
E-BFMI on 2 of 4 chains (the log does not say which; R-hat and ESS are fine). Stage 2: R-hat ≤
1.022, bulk ESS ≥ 214; 34 of 196 models had divergent transitions, 65 in all, at most 5 of
6,000 draws per model, all in phylogeny + space models. Not refitted.

**Slopes (plain mode, phylogeny + space).** Per SD of the language exposure, on the logit
scale of the share of the 12 categories present; [95% interval]. Δelpd: with-x model minus
the baseline on the same languages and controls (± SE).

| Measure | main | no_doubtful_language | no_other_language_stand_ins | strict | Δelpd, main |
|---|---|---|---|---|---|
| `ent` | +0.19 [+0.03, +0.35] | +0.21 [+0.04, +0.38] | +0.22 [+0.06, +0.38] | +0.22 [+0.05, +0.38] | +1.7 ± 1.9 |
| `pop_ent` | +0.19 [+0.03, +0.34] | +0.21 [+0.04, +0.38] | +0.22 [+0.05, +0.38] | +0.22 [+0.05, +0.39] | +1.8 ± 1.9 |
| `ent_K12` | +0.19 [+0.02, +0.37] | +0.20 [+0.03, +0.38] | +0.22 [+0.05, +0.39] | +0.21 [+0.03, +0.38] | +1.7 ± 1.8 |
| `ent_K30` | +0.16 [+0.01, +0.31] | +0.20 [+0.04, +0.36] | +0.19 [+0.04, +0.35] | +0.20 [+0.03, +0.35] | +1.2 ± 1.8 |
| `nonmax` | +0.16 [−0.00, +0.32] | +0.21 [+0.04, +0.38] | +0.19 [+0.02, +0.35] | +0.20 [+0.04, +0.37] | +0.8 ± 1.8 |
| `f3_z` | +0.17 [+0.02, +0.32] | +0.16 [+0.01, +0.32] | +0.20 [+0.05, +0.36] | +0.17 [+0.02, +0.32] | +1.4 ± 2.1 |
| `f3_est` | +0.18 [+0.04, +0.32] | +0.18 [+0.04, +0.33] | +0.20 [+0.06, +0.35] | +0.19 [+0.05, +0.34] | +1.9 ± 2.3 |
| `het` | +0.13 [−0.00, +0.26] | +0.13 [−0.00, +0.27] | +0.15 [+0.02, +0.28] | +0.15 [+0.01, +0.29] | 0.0 ± 2.2 |
| `amount` | −0.17 [−0.31, −0.03] | −0.13 [−0.28, +0.03] | −0.18 [−0.32, −0.03] | −0.14 [−0.29, +0.01] | +1.0 ± 2.4 |
| `support` | −0.12 [−0.26, +0.02] | −0.12 [−0.27, +0.02] | −0.12 [−0.27, +0.02] | −0.12 [−0.27, +0.04] | +0.1 ± 1.7 |
| `neigh` | −0.03 [−0.23, +0.18] | −0.04 [−0.27, +0.20] | −0.03 [−0.24, +0.19] | −0.02 [−0.24, +0.22] | −0.7 ± 0.5 |
| `graff` | −0.01 [−0.14, +0.13] | +0.04 [−0.12, +0.20] | +0.01 [−0.13, +0.14] | +0.04 [−0.11, +0.20] | −0.2 ± 0.3 |

* More mixed ancestry goes with more inflection. `amount` and `support` correlate
  negatively with entropy (§3), so their negative slopes are the same pattern. Same direction
  and size as the draft (entropy +0.20 to +0.25, `amount` −0.21, `me` mode).
* Size: +0.19 per SD is about half a category of 12 per SD at the average language.
* No measure improves prediction clearly: the largest Δelpd over all sets is +3.8 ± 2.4
  (`f3_est`, `no_other_language_stand_ins`).
* The link sets do not change the picture; dropping doubtful links and stand-ins makes the
  entropy slopes a little larger. On the raw scale (`b / (x_scale × y_sd)`), `ent` is 0.55
  (main) to 0.63 per unit of entropy.
* `het`, the closest sign of L2 speakers present now, is weakly positive, not negative.
* `neigh` is +0.19 [+0.02, +0.37] without controls and about 0 with phylogeny or space:
  gene flow with neighbours is explained by relatedness and place.

**Controls (main, baselines on all 202 languages).** Against phylogeny + space: phylogeny alone
−1.2 ± 2.8, space alone −10.0 ± 5.6, none −73.2 ± 9.1. Phylogeny does most of the work. The
entropy slopes need it: without controls `ent` is +0.09 [−0.04, +0.21], with phylogeny
+0.20 [+0.06, +0.36] (plain). The association is within families, not between them.

**f3 sampling check (main, plain, phylogeny + space).**

| Measure | Slope | Δelpd |
|---|---|---|
| `f3_z` (all triples) | +0.17 [+0.02, +0.32] | +1.4 ± 2.1 |
| `f3k_z` (5 nearest B and C) | +0.21 [+0.07, +0.36] | +3.4 ± 3.5 |
| `f3b3_z` (≥ 3 relatives) | +0.16 [+0.01, +0.32] | +0.9 ± 2.0 |
| `f3_est` (all triples) | +0.18 [+0.04, +0.32] | +1.9 ± 2.3 |
| `f3k_est` (5 nearest B and C) | +0.18 [+0.04, +0.33] | +2.4 ± 2.7 |
| `f3b3_est` (≥ 3 relatives) | +0.18 [+0.04, +0.33] | +2.2 ± 2.1 |

The f3 slope is not an effect of how densely the panel samples a family: it stays the same
when every population gets the same number of tests and when poorly sampled families are left
out. It does not cover the 19 populations (19 languages) without any f3 score.

**`me` against `plain`.** Slopes in `me` mode are larger, as expected (§10): by 5–10% for the
entropy measures, about 30% for `f3`, about 80% for `het`; `graff` is unidentified (−0.02
[−0.49, +0.45]). In `me` mode PSIS-LOO breaks down where the exposure is uncertain, because
every language gets its own latent exposure: `graff` has Pareto k > 0.7 for 179 of 202
languages, `het` for 38, and the Δelpd values (`graff` −138, `het` −54, `f3` −9 to −12) are
artefacts, not losses. Model comparison uses `plain` mode only; in `plain` mode at most 11
languages per model have k > 0.7.

**Where the pattern comes from (descriptive, `ent`, main, stage-1 means; not a model).**
Spearman between exposure and share of categories present: 0.08 over all 202 languages;
within macroareas Africa +0.45 (28), North America +0.63 (12), Papunesia +0.20 (48),
Eurasia +0.17 (98), South America −0.51 (13); within families Turkic +0.68 (12),
Atlantic-Congo +0.31 (13), Austronesian +0.22 (46), Indo-European −0.12 (24), Uralic −0.14
(12). The highest exposures are Turkic and Mongolic languages of Inner Asia; the lowest
include Grassfields Bantu, Mande, Hadza, Taa, Karitiana and Nganasan. Exposure is unrelated
to how many of the 12 features are coded (Spearman 0.01).

**Reading (open).** Two readings fit so far, and these data do not separate them:
* Long-term contact with intermarriage (child bilingualism) adds categories (Trudgill); genetic
  mixing marks that kind of contact better than adult L2 learning.
* The outcome counts transparent, mostly agglutinative categories. It rewards the Inner Asian
  suffixing type and penalises isolating languages. L2 theories mostly predict loss of
  irregularity (fusion, allomorphy), which these 12 features do not measure. Low entropy is
  also partly a property of the panel: drifted, isolated groups get their own ADMIXTURE
  component.

Possible next checks (not run): leave out Turkic + Mongolic, and each macroarea; the slope per
category; Grambank's Fusion and Flexivity scores (Shcherbakova et al. 2023) as outcomes.
