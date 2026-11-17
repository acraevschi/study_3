# Stage-2 Bayesian Berkson, Equivalence & Entrenchment Pipeline (`analyses/paradigm/stage2/`)

Two-stage Bayesian phylogenetic, spatial, equivalence (`BF_01` / ROPE), and token-frequency entrenchment pipeline for Study 3 Track 2 (Morphological Paradigm Predictability), strictly separated into **Verb (`V`)** and **Noun (`N`)** 1-row-per-`glottocode` pipelines matching Track 1 (`analyses/grambank/baseline/fit.R` and `analyses/common/stage1_exposure.R`).

## Scripts
1. **`prepare_stage2.py`:**
   - Merges MGL and Neural outcomes (`uniform120`, `uniform480`, `topfreq120`, `phon`, `malouf`), `unit_summary.csv`, `frequencies/{unit_id}_freq.csv`, `analyses/common/data/analysis_links.csv` (all 4 link sets), `analyses.common` (`genetics`, `demography`, `phylogeny`), and Track 1 `irreg13` overlap into `analyses/paradigm/results/stage2/prepared/`.
2. **`fit_stage2.R`:**
   - Runs `fit_stage1_exposure` (`ent`, `sqrt_het`, `f3k_est_pos`) per `(POS, link_set)`, standardizes `eta_mean` on the exact Stage-2 language set (**Berkson `plain` mode**, plus a 20-draw posterior-imputation Berkson propagation check), and fits precision-weighted `y | se(se_y, sigma = TRUE)` models across Glottolog Grafen + 3D spherical GP, Bouckaert v2.0.0 VCVs, Mundlak within-/between-family splits, paradigm-size/subsampling/Inner Asian sensitivities, and all 4 link sets.
   - Computes Savage–Dickey `BF_01` (Gaussian & KDE), ROPE `[-0.15, +0.15]` SD, `P(beta < 0.44)` against Track 1's breadth slope, and empirical + PGLS correlations with Track 1 `irreg13`.
3. **`fit_entrenchment.R`:**
   - Fits two-level within-language token-frequency entrenchment + cross-level `Admixture × Frequency` interaction regressions (`all_lemmas` and `attested_only`) separately for `V` and `N`.

## Usage
```bash
# 1. Prepare Stage-2 tables
.venv/bin/python analyses/paradigm/stage2/prepare_stage2.py

# 2. Validate all Stan models without MCMC sampling (fast dry-run)
PARA_DRY=1 Rscript analyses/paradigm/stage2/fit_stage2.R
PARA_DRY=1 Rscript analyses/paradigm/stage2/fit_entrenchment.R

# 3. Full MCMC run (respecting PARA_CORES <= 4 default)
PARA_CORES=4 Rscript analyses/paradigm/stage2/fit_stage2.R
PARA_CORES=4 Rscript analyses/paradigm/stage2/fit_entrenchment.R
```
