# Stage-2 Bayesian Berkson, Equivalence & Entrenchment Pipeline (`analyses/paradigm/stage2/`)

Two-stage Bayesian phylogenetic, spatial, equivalence ($\text{BF}_{01}$ / ROPE), cross-track validation, and token-frequency entrenchment pipeline for Study 3 Track 2 (Morphological Paradigm Predictability / *i-complexity*), strictly separated into **Verb (`V`)** and **Noun (`N`)** 1-row-per-`glottocode` pipelines and harmonized with Track 1 ([`analyses/grambank/fit_stage2_bb.R`](../../grambank/fit_stage2_bb.R) and [`analyses/common/stage1_exposure.R`](../../common/stage1_exposure.R)).

---

## 1. Scripts

1. **`prepare_stage2.py`:**
   - Merges MGL and Neural outcomes (`uniform120`, `uniform480`, `topfreq120`, `phon`, `malouf`), `unit_summary.csv`, `frequencies/{unit_id}_freq.csv`, `analyses/common/data/analysis_links.csv` (all 4 link sets), `analyses.common` (`genetics`, `demography`, `phylogeny`, `f3_fixed_scores.csv`, `fst_neighbor_regimes.csv`), and all **5 Track-1 Grambank clean domains** (`v_breadth`, `n_breadth`, `v_irreg`, `n_irreg`, `np_agr`) plus focal 2PL IRT-SEM latent EAPs (`analyses/grambank/latent_sem/results_domain/sem_latent_eap.csv`) into `analyses/paradigm/results/stage2/prepared/`.
2. **`fit_stage2.R`:**
   - Runs `fit_stage1_exposure` for fixed-reference cross-family **`f3_fixed`** (`min_n_ind = 3` primary, `5` sensitivity) and Block-2 genetic SI benchmarks (`f3_all_fixed`, `f3_fixed_iso`, `f3_fixed_maxp`, `f3_fixed_raw`, `f4_est`, `neigh_flow_out`), standardizes `eta_mean` on the exact active Stage-2 language set (**Berkson `plain` mode**), and includes both a classical measurement-error benchmark (`me_baseline`) and a 20-draw Stage-1 posterior-imputation propagation check (`berkson_20draw`).
   - Fits precision-weighted Gaussian regressions `y | se(se_y, sigma = TRUE)` (`Normal(0, 0.5)` slope priors, regularized 3D spherical spatial GP `sdgp ~ exponential(1)`, `lscale ~ inv_gamma(3, 0.4)` on `gp(sx, sy, sz)`) across a two-block grid:
     - **Block 1 (locked complete-case maximal & LOO ablations):** `maximal`, `loo_no_space`, `loo_no_cdist`, `loo_no_demog`, `loo_no_lexprov`, `mundlak_wb`, and `null_baseline`.
     - **Block 2 (sample, phylogeny, learner/regime, and genetics SI checks):** `full_sample_PS`, `cutoff_n5`, `strict_links`, `bouckaert_stem`, `bouckaert_crown`, `no_inner_asia`, `tier2_uniform480`, `topfreq120`, `pcfp_k1`, `pcfp_k3`, `reliability`, `malouf_harmony`, `neural_monotonic` (`V`), `f3_all_fixed`, `f3_fixed_iso`, `f3_fixed_maxp`, `f3_fixed_raw`, `f4_est`, `neigh_flow_out`, `drop_regime_a`, `drop_regime_b`, `plus_regime_b`, `me_baseline`, and `berkson_20draw`.
   - Computes Savage–Dickey $\text{BF}_{01}$ (Gaussian & KDE), ROPE `[-0.15, +0.15]` SD, $P(\beta < 0.44)$, and `71` cross-track empirical & PGLS construct-validity correlations (`cross_track_correlations.csv`) linking `h_tree_bits` and `log_C_distinct` to all 5 Track-1 Grambank domain proportions, Track-1 2PL latent EAPs (`sem_latent_eap.csv`), and cross-regime MGL/Neural benchmarks.
3. **`fit_entrenchment.R`:**
   - Fits two-level within-language token-frequency entrenchment + cross-level `f3_fixed × Frequency` interaction regressions (`y_lemma ~ 1 + x_adm * z_freq_within + z_log_navail + (1 | gr(glottocode, cov = A_phylo)) + (1 + z_freq_within | lang_id)`) across `all_lemmas` and `attested_only` at $n_A \ge 3$ and $n_A \ge 5$, writing `entrenchment_results_{V,N}.csv` and `language_entrenchment_slopes_{V,N}.csv`.

---

## 2. Headline Results (`f3_fixed`, `main`, $n_A \ge 3$)

- **Stage-2 Cross-Linguistic `maximal` (`stage2_results_{V,N}.csv`):**
  - **Verbs (`V`, $N = 42$):** $\beta = -0.157\;[-0.333, +0.072]$, $\text{BF}_{01} = 1.51$, $\text{ROPE} = 0.429$ (`26/27` fitted $H_{\text{tree}}$ models include `0` in their `95%` CrI; `berkson_20draw`: $-0.153\;[-0.344, +0.076]$; `me_baseline`: $-0.182\;[-0.402, +0.076]$; `f4_est` SI [$N = 33$]: $-0.167\;[-0.483, +0.126]$).
  - **Nouns (`N`, $N = 47$):** $\beta = +0.085\;[-0.231, +0.412]$, $\text{BF}_{01} = 2.66$, $\text{ROPE} = 0.576$ (`26/26` fitted $H_{\text{tree}}$ models include `0` in their `95%` CrI; `berkson_20draw`: $+0.078\;[-0.254, +0.409]$; `me_baseline`: $+0.104\;[-0.249, +0.457]$; `f4_est` SI [$N = 42$]: $-0.065\;[-0.436, +0.300]$).
- **Two-Level Token-Frequency Entrenchment (`entrenchment_results_{V,N}.csv`, `all_lemmas`, $n_A \ge 3$):**
  - **Verbs (`V`, $42$ langs / $19,784$ lemmas):** $b_{\text{freq}} = +0.095\;[+0.050, +0.140]$ ($\text{BF}_{10} = 322$, $P > 0 = 1.0000$); $b_{\text{int}} = -0.002\;[-0.044, +0.041]$ ($\text{BF}_{01} = 22.40$, $\text{ROPE} = 1.000$).
  - **Nouns (`N`, $44$ langs / $26,060$ lemmas):** $b_{\text{freq}} = +0.031\;[+0.014, +0.048]$ ($\text{BF}_{10} = 11.05$, $P > 0 = 0.9996$); $b_{\text{int}} = -0.002\;[-0.019, +0.016]$ ($\text{BF}_{01} = 55.35$, $\text{ROPE} = 1.000$).

---

## 3. Methodological Note on HMC Sampler Diagnostics

Across all single-fit (non-Berkson) Track-2 Stage-2 models (`V` and `N`), focal admixture slope convergence is tight (**`b_rhat <= 1.007`**, **`b_ess_bulk >= 985`**), and all two-level entrenchment models converge with **`0` divergent transitions**. However, in small-$N$ ($N = 22\text{--}47$) precision-weighted Gaussian models (`y | se(se_y, sigma = TRUE)`) that simultaneously include a phylogenetic random effect `(1 | gr(lang, cov = A))` and a 3D spherical spatial Gaussian process `gp(sx, sy, sz)`, posterior funnel geometry near the `sigma -> 0` vs. phylogenetic variance boundary can still produce residual HMC divergences on hierarchical variance parameters (and summed across the 20 independent fits in `berkson_20draw`). Users inspecting diagnostics or refitting custom subsets can increase `adapt_delta` or tighten initialization via `PARA_ADAPT_DELTA` (e.g., `0.999`) and `PARA_INIT` (e.g., `0.2`), or increase `max_treedepth` as needed.

---

## 4. Usage & Environment Variables

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# 1. Prepare Stage-2 tables
.venv/bin/python analyses/paradigm/stage2/prepare_stage2.py

# 2. Validate all Stan code & data blocks without MCMC sampling (fast dry-run)
PARA_DRY=1 PARA_POS=V,N Rscript analyses/paradigm/stage2/fit_stage2.R
ENT_DRY=1 ENT_POS=V,N Rscript analyses/paradigm/stage2/fit_entrenchment.R

# 3. Full MCMC run (scale PARA_CORES / ENT_CORES or pass --cores=N; --pos=BOTH also supported)
PARA_CORES=4 PARA_POS=V,N Rscript analyses/paradigm/stage2/fit_stage2.R
ENT_CORES=4 ENT_POS=V,N Rscript analyses/paradigm/stage2/fit_entrenchment.R
```

Key environment variables & CLI flags:
- **`fit_stage2.R`:** `PARA_DRY=1`, `PARA_SMOKE=1`, `PARA_POS=V,N` (or `--pos=BOTH` / `--pos=V`), `PARA_CORES` (or `--cores=N`, default `4`), `PARA_THREADS` (or `--threads=N`, default `1`), `PARA_SETS=main`, `PARA_MEASURES=f3_fixed`, `PARA_MIN_N_IND=3`, `PARA_N_IMP=20`, `PARA_ADAPT_DELTA`, `PARA_INIT`, `PARA_FILE_REFIT` (`never` default), `PARA_DATA_DIR`, `PARA_OUT_DIR`.
- **`fit_entrenchment.R`:** `ENT_DRY=1`, `ENT_SMOKE=1`, `ENT_POS=V,N` (or `--pos=BOTH`), `ENT_CORES` (or `--cores=N`, default `4`), `ENT_THREADS` (or `--threads=N`, default `1`), `ENT_MEASURES=f3_fixed`, `ENT_CUTOFFS=3,5`, `ENT_DATA_DIR`, `ENT_OUT_DIR` (all fall back to `PARA_*` equivalents when unset).
