# Latent-Variable and Multi-Item Morphological Models (`analyses/grambank/latent_sem/`)

**Exploratory.** Extends [`analyses/grambank/baseline/`](../baseline/README.md) by replacing the equal-weight sum-score Beta-Binomial outcome (`n_present | trials(n_coded)`) with item-level latent-variable and hierarchical varying-slope models over the binary language $\times$ feature observation matrix $Y_{ik} \in \{0, 1\}$.

**Status 2026-10-31: fitted (`main`, Phylogeny `P`, 42 core models in §4 + 309 unique unifying synthesis models in §5).** All 42 core models converged with $\max \hat{R} \le 1.007$, minimum bulk ESS $\ge 554$, and $0$ divergent transitions; all 309 unique synthesis models (`237` SEM specs + `72` Beta-Binomial `brms` specs; `333` rows before deduplicating the 24 reused legacy `P` reference rows) converged with $\max \hat{R} \le 1.011$, minimum bulk ESS $\ge 228$, and only $3$ divergent transitions across $9.48 \times 10^5$ post-warmup SEM draws.

## 1. Motivation

Collapsing the $12$ GBI inflectional categories (`cat12`) or the $13$/$26$ Grambank irregularity/flexivity features (`irreg13` / `flex26`) into a single count `n_present` out of `n_coded` imposes three restrictive assumptions:
1. **Equal item weights ($\lambda_k = 1$):** Every feature contributes identically to the outcome, even though item-rest correlations across the 202 `main` languages range from $r_{\text{rest}} = +0.62$ (`tense`) and $+0.58$ (`nominal_number`) down to $+0.17$ (`polar_interrogation`) and $+0.02$ (`possessed_affix`).
2. **Missingness denominator bias:** Base rates across the 12 categories range from $14.7\%$ (`polar_interrogation`, coded in $170/202$ languages) to $79.8\%$ (`person_indexing`, coded in $198/202$ languages). In a Binomial/Beta-Binomial sum model, missing a rare category reduces `n_coded` and artificially inflates `n_present / n_coded`. Modeling observed $(i, k)$ cells directly with item easiness intercepts $\alpha_k$ removes this distortion.
3. **Within-domain local dependence and opposing contact slopes:** Within families, inherent/head-marked categories (`nominal_number`, `negation`, `possessed_affix`, `person_indexing`) correlate positively with genetic admixture, whereas contextual modifier agreement (`gender_agreement`, `number_agreement`) and paradigmatic irregularity (`irreg13`) correlate negatively.

## 2. Model Ladder and Architectures

All models include hierarchical item easiness intercepts $\alpha_k = \mu_\alpha + \sigma_\alpha z_{\alpha, k}$ and multi-item within-language **domain testlet** random effects $u_{i, t(k)} \sim \mathcal{N}(0, \sigma_{\text{testlet}}^2)$ to absorb local item dependence within multi-item grammatical sub-domains (`verbal_tam`, `neg_interrog`, `possession`, `agreement`, `noun_irreg`, `arg_split_classes`, `verb_suppletion_stem`, `gender_nc`, `classifiers`, `evaluative_anim`).

### 2.1 One-Factor Models on `cat12` (`stan/item_1factor.stan`, $K = 12$)

$$\text{logit}\,P(Y_{ik} = 1) = \alpha_k + \eta_{ik} + u_{i, t(k)}$$

* **`M0_1pl` (Baseline 1PL):** $\eta_{ik} = \theta_{0, i}$, where $\theta_{0, i} = u_{\text{phylo}, i} + u_{\text{space}, i} + \sigma_{\text{lang}} z_{\text{lang}, i}$ ($\lambda_k = 1$, no admixture $x_i$).
* **`M1_1pl` (Equal-Weight 1PL with Admixture):** $\eta_{ik} = \theta_i$ with $\theta_i = \beta x_i + \theta_{0, i}$ and $\lambda_k = 1$.
* **`MA0_2pl` / `MA_2pl` (Architecture A — 2PL IRT / MIMIC SEM):**
  $$\eta_{ik} = \lambda_k \, \theta_i, \qquad \lambda_k = K \, \operatorname{softmax}\!\left(\sigma_\lambda (\mathbf{Q}_{\text{zf}} \mathbf{z}_\lambda)_k\right)$$
  where $\mathbf{Q}_{\text{zf}} \in \mathbb{R}^{K \times (K-1)}$ is an isometric Helmert zero-sum basis ($\sum_{k=1}^K \delta_k = 0$, giving every item $k$ identical marginal prior variance $\sigma_\lambda^2$ and equicorrelated off-diagonals $-\frac{1}{K-1}$, strictly invariant to item ordering) and $K \operatorname{softmax}(\cdot)$ locks the **arithmetic mean** $\frac{1}{K}\sum_{k=1}^K \lambda_k = 1.0$ on every MCMC draw so $\beta$ stays on the exact mean-item log-odds scale.
* **`MB_1pl` / `MB_2pl` (Architecture B — Hierarchical Varying Slopes by Domain & Category):**
  $$\eta_{ik} = \beta_k \, x_i + \lambda_k \, \theta_{0, i}, \qquad \beta_k = \beta_0 + \text{centered}(\delta_{\text{dom}, d(k)} + \delta_{\text{item}, k})$$
  Allows admixture to shift different domains/categories in opposite directions (`MB_1pl` with $\lambda_k = 1$, `MB_2pl` with 2PL weights $\lambda_k$ on $\theta_{0, i}$), and tracks the posterior contrast $\Delta\beta_{\text{inh}-\text{agr}} = \bar{\beta}_{\text{inherent}} - \bar{\beta}_{\text{agreement}}$ between the 10 inherent categories (`no_agreement`) and the 2 contextual agreement categories (`gender_agreement`, `number_agreement`).
* **Marginal Language-Level LOO (`log_lik_lang`):** In `generated quantities`, language-specific i.i.d. random effects (`z_lang[i]` and `z_testlet[i, 1:T_test]`) are integrated out via nested Gauss-Hermite quadrature conditioned on the structural surface $u_{\text{phylo}, i} + u_{\text{space}, i}$ and $x_i$ (Merkle et al. 2019). This cuts PSIS-LOO `Pareto k > 0.7` flags on `cat12` from $60\text{--}73$ languages (under conditional LOO) to $5\text{--}19$ languages (concentrated in the 29 singleton-family languages whose phylogenetic innovation $u_{\text{phylo}, i}$ has no within-family relatives; also integrating the conditional phylogenetic tip variance $\sigma_{\text{phylo}}^2 / (\mathbf{A}^{-1})_{ii}$ reduces `k > 0.7` to $0\text{--}4$ without altering any slope estimate or qualitative model ranking).

### 2.2 Two-Factor Joint Models on `cat12_irreg13` ($K = 25$) and `cat12_flex26` ($K = 38$) (`stan/item_2factor.stan`)

Jointly estimates **Factor 1 (Enumerative Breadth, $\theta_{i, 1}$)** on items $1\dots 12$ (`cat12`) and **Factor 2 (Paradigmatic Irregularity/Flexivity, $\theta_{i, 2}$)** on items $13\dots K$ (`irreg13` or `flex26`), with correlated language-level and phylogenetic residuals ($\rho_{\text{latent}}$), factor-specific 2PL loadings $\boldsymbol{\lambda}_1, \boldsymbol{\lambda}_2$ (arithmetic mean $1.0$ within each factor), and factor-specific admixture slopes $(\beta_{\text{breadth}}, \beta_{\text{irreg}})$ with posterior difference $\Delta\beta = \beta_{\text{breadth}} - \beta_{\text{irreg}}$:

* **`MC0_strict` / `MC_strict`:** Strict confirmatory factor assignment ($1\dots 12$ load on $\theta_{i, 1}$; $13\dots K$ load on $\theta_{i, 2}$).
* **`MC0_cross` / `MC_cross`:** Allows the 2 contextual agreement categories in `cat12` (`gender_agreement`, `number_agreement`) to also cross-load ($\gamma_{\text{cross}, 1}, \gamma_{\text{cross}, 2}$) onto Factor 2 ($\theta_{i, 2}$).

## 3. Files and Usage

```bash
# 1. Build item_observations.csv, items_metadata.csv, language_table.csv, latent_prep_summary.json
/usr/bin/python3 analyses/grambank/prepare.py

# 2. Dry-run validation (checks Stan syntax and all model data structures without sampling)
LAT_DRY=1 Rscript analyses/grambank/latent_sem/fit_latent.R

# 3. Smoke test (fast 400-iteration check into fits_smoke/ and results_smoke/)
LAT_SMOKE=1 Rscript analyses/grambank/latent_sem/fit_latent.R

# 4. Core run (ent, sqrt_het, f3k_est_pos under P control on main; 42 models; default LAT_CTRLS=P,F runs 84)
LAT_CTRLS=P Rscript analyses/grambank/latent_sem/fit_latent.R main

# 5. Unifying synthesis checks (Checks 1–3; 237 unique SEM specs + 72 Beta-Binomial brms specs = 309 models)
SYN_DRY=1 Rscript analyses/grambank/latent_sem/fit_synthesis.R
Rscript analyses/grambank/latent_sem/fit_synthesis.R
```

## 4. Results (`main`, Phylogeny `P`, 2026-10-31)

Output tables: `results/model_results_main.csv`, `results/loo_comparisons_main.csv`, `results/item_parameters_main.csv`.
Diagnostics across all 42 models: $\max \hat{R} = 1.007$, $\min \text{ESS}_{\text{bulk}} = 554$, $0$ divergent transitions.

### 4.1 One-Factor and Varying-Slope Models on `cat12` ($N = 202$ for `ent`, $198$ for `sqrt_het`, $183$ for `f3k_est_pos`)

| Exposure | Architecture | Mean Slope $\beta_0$ [95% CI] | Inherent $\bar{\beta}_{\text{inh}}$ ($10$ cats) | Agreement $\bar{\beta}_{\text{agr}}$ ($2$ cats) | Contrast $\Delta\beta_{\text{inh}-\text{agr}}$ [95% CI] | $\Delta\text{elpd}_{\text{obs}}$ vs. Base | $\Delta\text{elpd}_{\text{lang}}$ vs. Base |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ent`** | `M1_1pl` | $+0.34 \; [+0.13, +0.54]$ | — | — | — | $+3.4 \pm 2.4$ | $+4.7 \pm 2.8$ |
| **`ent`** | `MA_2pl` | $+0.35 \; [+0.15, +0.57]$ | — | — | — | $+3.1 \pm 2.7$ | $+5.9 \pm 3.2$ |
| **`ent`** | `MB_1pl` | $+0.32 \; [+0.11, +0.53]$ | $+0.44$ | $-0.31$ | **$+0.75 \; [+0.39, +1.14]$** ($P=1.000$) | **$+17.0 \pm 6.2$** | **$+19.7 \pm 7.3$** |
| **`ent`** | `MB_2pl` | $+0.30 \; [+0.11, +0.51]$ | $+0.42$ | $-0.31$ | **$+0.73 \; [+0.38, +1.10]$** ($P=1.000$) | **$+15.1 \pm 5.8$** | **$+18.8 \pm 7.3$** |
| **`sqrt_het`** | `M1_1pl` | $+0.20 \; [+0.01, +0.38]$ | — | — | — | $-0.1 \pm 1.7$ | $+0.3 \pm 2.3$ |
| **`sqrt_het`** | `MA_2pl` | $+0.22 \; [+0.04, +0.40]$ | — | — | — | $+1.3 \pm 2.0$ | $+1.5 \pm 2.8$ |
| **`sqrt_het`** | `MB_1pl` | $+0.21 \; [+0.02, +0.40]$ | $+0.34$ | **$-0.47$** | **$+0.81 \; [+0.43, +1.20]$** ($P=1.000$) | **$+7.9 \pm 4.7$** | **$+11.0 \pm 7.1$** |
| **`sqrt_het`** | `MB_2pl` | $+0.22 \; [+0.05, +0.40]$ | $+0.35$ | **$-0.42$** | **$+0.77 \; [+0.39, +1.16]$** ($P=1.000$) | **$+7.6 \pm 4.9$** | **$+10.3 \pm 8.1$** |
| **`f3k_est_pos`** | `M1_1pl` | $+0.26 \; [+0.07, +0.46]$ | — | — | — | $+2.4 \pm 2.4$ | $+3.0 \pm 4.1$ |
| **`f3k_est_pos`** | `MA_2pl` | $+0.35 \; [+0.13, +0.59]$ | — | — | — | $+3.2 \pm 2.6$ | $+4.9 \pm 4.4$ |
| **`f3k_est_pos`** | `MB_1pl` | $+0.31 \; [+0.11, +0.53]$ | $+0.37$ | $+0.02$ | $+0.34 \; [-0.01, +0.74]$ ($P=0.972$) | $+4.6 \pm 3.1$ | $+5.5 \pm 4.9$ |
| **`f3k_est_pos`** | `MB_2pl` | $+0.31 \; [+0.11, +0.53]$ | $+0.37$ | $+0.03$ | $+0.34 \; [-0.00, +0.73]$ ($P=0.974$) | $+2.4 \pm 3.1$ | $+4.3 \pm 4.8$ |

* **2PL weights vs. 1PL equal weights (`2pl_vs_1pl`):** Learning unequal item discrimination weights $\lambda_k$ improves PSIS-LOO over equal weights by **$\Delta\text{elpd}_{\text{obs}} = +27.8\text{ to }+35.9 \text{ (\pm 8.5)}$** ($\Delta\text{elpd}_{\text{lang}} = +28.4\text{ to }+34.2 \text{ (\pm 8.2)}$).
* **Varying slopes vs. single latent MIMIC slope (`varslopes_vs_2pl`):** For `ent`, allowing category/domain slopes to vary (`MB_2pl` vs. `MA_2pl`) improves fit by **$\Delta\text{elpd}_{\text{obs}} = +12.0 \pm 5.5$** ($\Delta\text{elpd}_{\text{lang}} = +12.9 \pm 6.6$), and versus the no-$x$ baseline (`MB_2pl` vs. `MA0_2pl`) by **$+15.1 \pm 5.8$** ($+18.8 \pm 7.3$).

### 4.2 Learned 2PL Category Weights ($\lambda_k$) and Category-Specific Admixture Slopes ($\beta_k$) in `cat12`

| Category ($k$) | Domain | Learned Weight $\lambda_k$ (`MA0_2pl`) | $\beta_k$ (`MB_2pl`, `ent`) | $\beta_k$ (`MB_2pl`, `sqrt_het`) | $\beta_k$ (`MB_2pl`, `f3k_est_pos`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `nominal_number` | `nominal_number` | **$2.01 \; [1.41, 2.77]$** | $+0.53 \; [+0.14, +1.06]$ | $+0.42 \; [+0.01, +0.92]$ | $+0.39 \; [+0.00, +0.94]$ |
| `tense` | `verbal_tam` | **$1.95 \; [1.42, 2.62]$** | $+0.64 \; [+0.13, +1.22]$ | $+0.48 \; [+0.07, +0.97]$ | $+0.47 \; [-0.08, +1.17]$ |
| `mood` | `verbal_tam` | **$1.29 \; [0.89, 1.73]$** | $+0.54 \; [+0.08, +1.02]$ | $+0.44 \; [+0.06, +0.88]$ | $+0.40 \; [-0.09, +0.96]$ |
| `gender_agreement` | `agreement` | $1.17 \; [0.71, 1.68]$ | $-0.08 \; [-0.52, +0.35]$ | **$-0.44 \; [-0.88, -0.03]$** | $+0.12 \; [-0.25, +0.48]$ |
| `case` | `case` | $1.01 \; [0.66, 1.42]$ | $+0.45 \; [+0.11, +0.84]$ | $+0.33 \; [-0.02, +0.73]$ | $+0.40 \; [+0.05, +0.91]$ |
| `person_indexing` | `person_indexing` | $0.91 \; [0.57, 1.29]$ | $+0.41 \; [+0.07, +0.75]$ | $+0.21 \; [-0.15, +0.57]$ | $+0.27 \; [-0.11, +0.65]$ |
| `aspect` | `verbal_tam` | $0.85 \; [0.53, 1.19]$ | $-0.04 \; [-0.46, +0.38]$ | $+0.24 \; [-0.17, +0.61]$ | $+0.51 \; [+0.06, +1.11]$ |
| `possessor_affix` | `possession` | $0.74 \; [0.42, 1.11]$ | $+0.35 \; [-0.03, +0.75]$ | $+0.41 \; [+0.07, +0.76]$ | $+0.42 \; [+0.02, +0.91]$ |
| `number_agreement` | `agreement` | $0.69 \; [0.36, 1.06]$ | **$-0.54 \; [-1.00, -0.10]$** | **$-0.40 \; [-0.79, -0.02]$** | $-0.07 \; [-0.47, +0.30]$ |
| `negation` | `neg_interrog` | $0.59 \; [0.28, 0.91]$ | $+0.55 \; [+0.18, +0.92]$ | $+0.25 \; [-0.06, +0.58]$ | $+0.62 \; [+0.18, +1.20]$ |
| `polar_interrogation` | `neg_interrog` | $0.57 \; [0.20, 1.03]$ | $+0.18 \; [-0.27, +0.64]$ | $+0.26 \; [-0.13, +0.65]$ | $+0.12 \; [-0.49, +0.59]$ |
| `possessed_affix` | `possession` | **$0.21 \; [0.05, 0.43]$** | $+0.64 \; [+0.28, +1.04]$ | $+0.44 \; [+0.10, +0.79]$ | $+0.08 \; [-0.32, +0.46]$ |

### 4.3 Two-Factor Joint Models (`cat12_irreg13` [$N = 175, K = 25$] and `cat12_flex26` [$N = 186, K = 38$])

| Item Set | Exposure | Arch | Breadth Slope $\beta_{\text{breadth}}$ | Irreg/Flex Slope $\beta_{\text{irreg}}$ | Difference $\Delta\beta = \beta_{\text{br}} - \beta_{\text{ir}}$ ($P>0$) | Latent Corr $\rho_{\text{latent}}$ | $\Delta\text{elpd}_{\text{obs}}$ (`MC_cross` vs `MC_strict`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`cat12_irreg13`** | `ent` | `MC_strict` | $+0.38 \; [+0.16, +0.61]$ | $-0.13 \; [-0.43, +0.17]$ | **$+0.51 \; [+0.16, +0.86]$** ($0.998$) | $+0.45 \; [+0.10, +0.73]$ | — |
| **`cat12_irreg13`** | `ent` | `MC_cross` | $+0.44 \; [+0.22, +0.66]$ | $-0.13 \; [-0.41, +0.14]$ | **$+0.57 \; [+0.22, +0.92]$** ($0.999$) | $+0.16 \; [-0.20, +0.51]$ | **$+40.7 \pm 9.0$** |
| **`cat12_irreg13`** | `sqrt_het` | `MC_strict` | $+0.23 \; [+0.04, +0.43]$ | $-0.16 \; [-0.46, +0.13]$ | **$+0.39 \; [+0.07, +0.73]$** ($0.990$) | $+0.39 \; [+0.04, +0.70]$ | — |
| **`cat12_irreg13`** | `sqrt_het` | `MC_cross` | $+0.29 \; [+0.10, +0.49]$ | $-0.12 \; [-0.37, +0.12]$ | **$+0.41 \; [+0.10, +0.74]$** ($0.995$) | $+0.11 \; [-0.24, +0.45]$ | **$+40.0 \pm 9.0$** |
| **`cat12_irreg13`** | `f3k_est_pos` | `MC_strict` | $+0.26 \; [+0.05, +0.49]$ | $-0.07 \; [-0.33, +0.17]$ | **$+0.33 \; [+0.02, +0.66]$** ($0.982$) | $+0.47 \; [+0.09, +0.77]$ | — |
| **`cat12_irreg13`** | `f3k_est_pos` | `MC_cross` | $+0.37 \; [+0.14, +0.62]$ | $-0.13 \; [-0.35, +0.07]$ | **$+0.50 \; [+0.18, +0.85]$** ($0.999$) | $+0.09 \; [-0.30, +0.45]$ | **$+41.4 \pm 8.8$** |
| **`cat12_flex26`** | `ent` | `MC_strict` | $+0.38 \; [+0.17, +0.59]$ | $-0.02 \; [-0.22, +0.19]$ | **$+0.39 \; [+0.12, +0.67]$** ($0.997$) | $+0.50 \; [+0.17, +0.75]$ | — |
| **`cat12_flex26`** | `ent` | `MC_cross` | $+0.43 \; [+0.21, +0.65]$ | $-0.04 \; [-0.25, +0.18]$ | **$+0.47 \; [+0.16, +0.78]$** ($0.998$) | $+0.06 \; [-0.30, +0.40]$ | **$+52.2 \pm 9.0$** |
| **`cat12_flex26`** | `sqrt_het` | `MC_strict` | $+0.22 \; [+0.04, +0.42]$ | $+0.01 \; [-0.18, +0.20]$ | $+0.22 \; [-0.03, +0.46]$ ($0.954$) | $+0.45 \; [+0.11, +0.73]$ | — |
| **`cat12_flex26`** | `sqrt_het` | `MC_cross` | $+0.28 \; [+0.09, +0.48]$ | $-0.00 \; [-0.19, +0.19]$ | **$+0.28 \; [+0.00, +0.56]$** ($0.976$) | $-0.02 \; [-0.36, +0.32]$ | **$+50.4 \pm 9.1$** |
| **`cat12_flex26`** | `f3k_est_pos` | `MC_strict` | $+0.28 \; [+0.05, +0.51]$ | $-0.02 \; [-0.15, +0.10]$ | **$+0.30 \; [+0.05, +0.56]$** ($0.992$) | $+0.40 \; [+0.01, +0.72]$ | — |
| **`cat12_flex26`** | `f3k_est_pos` | `MC_cross` | $+0.38 \; [+0.14, +0.65]$ | $-0.04 \; [-0.18, +0.09]$ | **$+0.42 \; [+0.15, +0.73]$** ($0.999$) | $-0.05 \; [-0.40, +0.33]$ | **$+52.6 \pm 8.9$** |

* **Agreement cross-loading (`MC_cross` vs. `MC_strict`):** Allowing `gender_agreement` and `number_agreement` to cross-load onto Factor 2 improves PSIS-LOO by **$\Delta\text{elpd}_{\text{obs}} = +40.0\text{ to }+41.4 \text{ (\pm 9.0)}$** on `cat12_irreg13` ($\Delta\text{elpd}_{\text{lang}} = +45.5\text{ to }+46.8 \text{ (\pm 12.5)}$) and **$\Delta\text{elpd}_{\text{obs}} = +50.4\text{ to }+52.6 \text{ (\pm 9.0)}$** on `cat12_flex26` ($\Delta\text{elpd}_{\text{lang}} = +60.7\text{ to }+62.5 \text{ (\pm 12.0)}$).
* In `MC_cross`, the Factor-1 (Breadth) loadings of `gender_agreement` and `number_agreement` drop to $\lambda_1 = 0.42$, while their cross-loadings onto Factor 2 (Irregularity/Flexivity) are $\gamma_{\text{cross}, \text{gender}} = +1.72 \; [+1.09, +2.44]$ (`irreg13`) / $+2.02 \; [+1.44, +2.71]$ (`flex26`) and $\gamma_{\text{cross}, \text{number}} = +1.05 \; [+0.63, +1.60]$ (`irreg13`) / $+0.97 \; [+0.58, +1.44]$ (`flex26`).
* **Orthogonality of Inherent Breadth and Paradigmatic Irregularity once Agreement is separated:** In `MC_strict`, the two factors appear correlated ($\rho_{\text{latent}} = +0.39\text{ to }+0.50$). Once `gender_agreement` and `number_agreement` are allowed to cross-load onto Factor 2 (`MC_cross`), the residual correlation between Factor 1 (Inherent Breadth) and Factor 2 (Paradigmatic Irregularity/Flexivity) drops to zero ($\rho_{\text{latent}} = +0.09\text{ to }+0.16$ on `irreg13`; $-0.05\text{ to }+0.06$ on `flex26`).

---

## 5. Unifying Synthesis Checks (`fit_synthesis.R`, `stan/item_*_synthesis.stan`)

To unify the item-level SEM findings with the within-family branch-rate and demographic analyses, `fit_synthesis.R` executes **309 unique synthesis models** (`237` item-level SEM specifications using `stan/item_1factor_synthesis.stan`, `stan/item_2factor_synthesis.stan`, and cached spatial `F` models, plus `72` two-stage Beta-Binomial `brms` specifications) and writes three canonical CSV tables:
- `results/synthesis_checks_main.csv` ($309$ rows: all posterior slope, contrast, raw-unit slope, variance-component, and LOO summaries)
- `results/family_slopes_main.csv` ($852$ rows: family-specific posterior slopes $\beta_f = \beta_0 + u_{\text{fam}, f}$ across all $23$ multi-language families)
- `results/loo_synthesis_main.csv` ($258$ rows: paired observation-level, language-marginal, and tip-marginalized LOO comparisons with both language-level SEs and family-clustered SEs)

**Methodological Enhancements in `stan/item_*_synthesis.stan`:**
- **Exact Tip-Marginalized LOO (`log_lik_lang_tip`):** Conditioning on the phylogenetic vector $u_{\text{phylo}, i}$ in `log_lik_lang` leaves the $29$ singleton-family languages (`A_inv_diag[i] == 1`) with high Pareto $k > 0.7$. The synthesis Stan models integrate out the conditional phylogenetic tip variance $\sigma_{\text{phylo}}^2 / (\mathbf{A}^{-1})_{ii}$ around the conditional phylogenetic mean $\mu_{\text{phylo}, i \mid -i} = u_{\text{phylo}, i} - (\mathbf{A}^{-1} \mathbf{u}_{\text{phylo}})_i / (\mathbf{A}^{-1})_{ii}$ using 21-point 1D Gauss-Hermite quadrature (`item_1factor_synthesis.stan`) and factored $31 \times 31$ (for singletons / 2-tip families) and $15 \times 15$ (for multi-relative tips) 2D Gauss-Hermite quadrature (`item_2factor_synthesis.stan`), cutting `Pareto k > 0.7` flags from $5\text{--}19$ down to $1\text{--}6$ languages per model.
- **Family-Clustered LOO SEs:** `loo_synthesis_main.csv` reports both standard language-level SEs (`se_obs`, `se_lang`, `se_lang_tip`) and family-clustered SEs (`se_obs_fam`, `se_lang_fam`, `se_lang_tip_fam`) computed over the $52$ family sums $\sum_{i \in f} \Delta\text{elpd}_i$.

---

### 5.1 Check 1: Family-Varying Admixture Slopes (`(1 + x | family)`), Mundlak Decomposition, and Inner-Asia Sensitivity

#### 5.1.1 Hierarchical Family-Varying Slopes (`famslope`) vs. Fixed Slopes (`x`)

Allowing the admixture slope to vary across Glottolog families ($\beta_{f} = \beta_0 + \sigma_{\text{fam\_b}} z_{\text{fam\_b}, f}$ on within-family centered $x_i - \bar{x}_f$ for multi-language families) tests whether a few extreme families drive the global positive slope on Enumerative Breadth or mask negative family-level slopes on Irregularity/Flexivity:

| Item Set & Architecture | Exposure | Spec | Breadth / Mean Slope $\beta_0$ [95% CI] ($P>0$) | Family Slope SD $\sigma_{\text{fam\_b1}}$ [95% CI] | Irreg / Agr Slope [95% CI] | Family Slope SD $\sigma_{\text{fam\_b2}}$ [95% CI] | Contrast $\Delta\beta$ [95% CI] ($P>0$) | $\Delta\text{elpd}_{\text{tip}}$ (`famslope` vs `fixed`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`cat12` (`MA_2pl`)** | `ent` | `fixed` | $+0.35 \; [+0.15, +0.57]$ ($0.999$) | — | — | — | — | — |
| | `ent` | `famslope` | **$+0.36 \; [+0.11, +0.62]$** ($0.997$) | $0.34 \; [0.01, 0.88]$ | — | — | — | $+0.4 \pm 1.1$ |
| | `sqrt_het` | `famslope` | **$+0.25 \; [+0.03, +0.49]$** ($0.986$) | $0.25 \; [0.01, 0.67]$ | — | — | — | $-0.2 \pm 1.1$ |
| | `f3k_est_pos` | `famslope` | **$+0.36 \; [+0.02, +0.69]$** ($0.981$) | $0.30 \; [0.01, 0.80]$ | — | — | — | $-1.0 \pm 1.5$ |
| **`cat12` (`MB_2pl`)** | `ent` | `famslope` | **$+0.29 \; [+0.06, +0.52]$** ($0.992$) | $0.24 \; [0.01, 0.72]$ | $\bar{\beta}_{\text{agr}} = -0.32$ | — | **$+0.73 \; [+0.38, +1.10]$** ($1.000$) | $-0.6 \pm 0.7$ |
| | `sqrt_het` | `famslope` | **$+0.25 \; [+0.06, +0.48]$** ($0.993$) | $0.22 \; [0.01, 0.61]$ | $\bar{\beta}_{\text{agr}} = -0.39$ | — | **$+0.77 \; [+0.39, +1.16]$** ($1.000$) | $-0.6 \pm 1.2$ |
| | `f3k_est_pos` | `famslope` | **$+0.32 \; [+0.06, +0.58]$** ($0.991$) | $0.16 \; [0.01, 0.51]$ | $\bar{\beta}_{\text{agr}} = +0.04$ | — | $+0.34 \; [+0.01, +0.72]$ ($0.976$) | $-0.7 \pm 0.7$ |
| **`cat12_irreg13` (`MC_cross`)** | `ent` | `fixed` | $+0.44 \; [+0.22, +0.66]$ ($1.000$) | — | $-0.13 \; [-0.41, +0.14]$ | — | $+0.57 \; [+0.22, +0.92]$ ($0.999$) | — |
| | `ent` | `famslope` | **$+0.46 \; [+0.18, +0.75]$** ($0.999$) | $0.40 \; [0.02, 1.02]$ | $-0.11 \; [-0.41, +0.21]$ | $0.22 \; [0.01, 0.67]$ | **$+0.57 \; [+0.15, +0.99]$** ($0.996$) | $+1.2 \pm 1.6$ |
| | `sqrt_het` | `famslope` | **$+0.34 \; [+0.10, +0.59]$** ($0.996$) | $0.23 \; [0.01, 0.65]$ | $-0.12 \; [-0.39, +0.14]$ | $0.16 \; [0.01, 0.49]$ | **$+0.46 \; [+0.10, +0.83]$** ($0.991$) | $-1.3 \pm 1.3$ |
| | `f3k_est_pos` | `famslope` | **$+0.37 \; [+0.05, +0.69]$** ($0.988$) | $0.25 \; [0.01, 0.71]$ | $-0.12 \; [-0.39, +0.16]$ | $0.16 \; [0.01, 0.52]$ | **$+0.49 \; [+0.07, +0.91]$** ($0.987$) | $-0.7 \pm 1.2$ |
| **`cat12_flex26` (`MC_cross`)** | `ent` | `famslope` | **$+0.44 \; [+0.18, +0.71]$** ($0.999$) | $0.35 \; [0.02, 0.89]$ | $-0.02 \; [-0.26, +0.24]$ | $0.22 \; [0.01, 0.56]$ | **$+0.46 \; [+0.09, +0.82]$** ($0.992$) | $-1.0 \pm 1.0$ |
| | `sqrt_het` | `famslope` | **$+0.32 \; [+0.09, +0.58]$** ($0.995$) | $0.25 \; [0.01, 0.66]$ | $+0.00 \; [-0.22, +0.24]$ | $0.20 \; [0.01, 0.51]$ | $+0.32 \; [-0.00, +0.66]$ ($0.975$) | $-1.7 \pm 1.3$ |
| | `f3k_est_pos` | `famslope` | **$+0.38 \; [+0.03, +0.72]$** ($0.984$) | $0.28 \; [0.01, 0.78]$ | $-0.02 \; [-0.20, +0.17]$ | $0.13 \; [0.00, 0.39]$ | $+0.40 \; [-0.01, +0.80]$ ($0.972$) | $-1.4 \pm 1.6$ |

* **Family-Specific Posterior Slopes (`family_slopes_main.csv`, 852 rows across 23 multi-language families):**
  * **Zero families in any model or exposure have a 95% credible interval excluding zero in the negative direction** (`beta_q975 < 0` in $0 / 852$ rows; the lowest 97.5% quantile on any family's Irregularity/Flexivity slope is $+0.069$ in Afro-Asiatic under `f3k_est_pos`).
  * In `MC_cross` (`cat12_irreg13`, `ent`), all 13 `min3` families ($n_f \ge 3$) have positive posterior mean Breadth slopes $\beta_{\text{breadth}, f} \in [+0.27, +0.92]$, led by Sino-Tibetan ($+0.92 \; [+0.22, +2.16], P>0=0.998$), Atlantic-Congo ($+0.87 \; [+0.21, +2.05], P>0=0.997$), Afro-Asiatic ($+0.69 \; [-0.05, +1.83], P>0=0.968$), Turkic ($+0.61 \; [-0.12, +1.65], P>0=0.957$), Tupian ($+0.55, P>0=0.945$), Mongolic-Khitan ($+0.51, P>0=0.905$), Austroasiatic ($+0.50, P>0=0.905$), Tungusic ($+0.49, P>0=0.888$), Nakh-Daghestanian ($+0.42, P>0=0.862$), Dravidian ($+0.42, P>0=0.860$), Austronesian ($+0.36 \; [+0.01, +0.68], P>0=0.978$), Indo-European ($+0.29 \; [-0.40, +0.83], P>0=0.837$), and Uralic ($+0.27 \; [-0.38, +0.75], P>0=0.832$), while every family's Irregularity slope 95% CI straddles zero ($\beta_{\text{irreg}, f} \in [-0.24, +0.02]$).
  * Two-stage Beta-Binomial `brms` models (`bb_cat12`, `bb_irreg13`, `bb_flex26`) show the exact same pattern under both within-family centered and uncentered `(1 + x | family)` parameterizations (`bb_cat12` `ent`: $\beta_0 = +0.21 \; [+0.04, +0.38], \sigma_{\text{fam\_b}} = 0.15$; `bb_irreg13`: $-0.05 \; [-0.29, +0.18]$; `bb_flex26`: $-0.00 \; [-0.15, +0.15]$).

#### 5.1.2 Mundlak Within- vs. Between-Family Decomposition (`c1_mundlak`)

Decomposing $x_i$ into a pure within-family deviation $(x_i - \bar{x}_f)$ for multi-language families ($\beta_{\text{within}}$) and a family-mean / singleton component $\bar{x}_f$ ($\beta_{\text{between}}$) confirms that the positive Enumerative Breadth slope and the Breadth-vs-Irregularity dissociation operate **strictly within language families**:
- In `MC_cross` (`cat12_irreg13`), $\beta_{\text{within, breadth}}$ is **$+0.47 \; [+0.21, +0.74]$** ($P>0 = 1.000$) for `ent`, **$+0.38 \; [+0.13, +0.65]$** ($P>0 = 0.998$) for `f3k_est_pos`, and **$+0.24 \; [+0.01, +0.47]$** ($P>0 = 0.980$) for `sqrt_het`, with within-family dissociation $\Delta\beta_{\text{within}} = +0.64 \; [+0.22, +1.07]$ (`ent`, $P>0 = 0.999$), $+0.50 \; [+0.17, +0.86]$ (`f3k_est_pos`, $P>0 = 0.998$), and $+0.33 \; [-0.03, +0.70]$ (`sqrt_het`, $P>0 = 0.963$).
- Splitting $\beta$ into $\beta_{\text{within}}$ and $\beta_{\text{between}}$ provides no out-of-sample gain over a single pooled slope ($\Delta\text{elpd}_{\text{tip}} \in [-1.9, -0.6]$ across `MA_2pl` and `MC_cross`), showing that within- and between-family slopes are statistically indistinguishable.

#### 5.1.3 Leave-Out-Inner-Asia Sensitivity (`no_turk_mong` [$N = 184$] & `no_inner_asia` [$N = 179$])

| Item Set & Arch | Exposure | `main` ($N=202/175/186$) | `no_turk_mong` ($N=184/159/169$) | `no_inner_asia` ($N=179/155/164$) |
| :--- | :--- | :---: | :---: | :---: |
| **`cat12` (`MA_2pl`) $\beta$** | `ent` | $+0.35 \; [+0.15, +0.57]$ ($0.999$) | **$+0.30 \; [+0.10, +0.50]$** ($0.999$) | **$+0.30 \; [+0.09, +0.51]$** ($0.997$) |
| | `sqrt_het` | $+0.22 \; [+0.04, +0.40]$ ($0.991$) | **$+0.20 \; [+0.02, +0.38]$** ($0.982$) | $+0.18 \; [-0.00, +0.36]$ ($0.975$) |
| | `f3k_est_pos` | $+0.35 \; [+0.13, +0.59]$ ($0.999$) | **$+0.30 \; [+0.08, +0.54]$** ($0.996$) | **$+0.29 \; [+0.06, +0.53]$** ($0.994$) |
| **`cat12` (`MB_2pl`) $\Delta\beta_{\text{inh}-\text{agr}}$** | `ent` | $+0.73 \; [+0.38, +1.10]$ ($1.000$) | $+0.15 \; [-0.11, +0.48]$ ($0.843$) | $+0.12 \; [-0.13, +0.45]$ ($0.798$) |
| | `sqrt_het` | $+0.77 \; [+0.39, +1.16]$ ($1.000$) | **$+0.49 \; [+0.10, +0.89]$** ($0.995$) | **$+0.42 \; [+0.05, +0.81]$** ($0.992$) |
| | `f3k_est_pos` | $+0.34 \; [-0.00, +0.73]$ ($0.974$) | $+0.23 \; [-0.09, +0.62]$ ($0.909$) | $+0.17 \; [-0.14, +0.54]$ ($0.844$) |
| **`cat12_irreg13` (`MC_cross`) $\beta_{\text{breadth}}$** | `ent` | $+0.44 \; [+0.22, +0.66]$ ($1.000$) | **$+0.33 \; [+0.12, +0.56]$** ($0.998$) | **$+0.32 \; [+0.10, +0.55]$** ($0.997$) |
| | `sqrt_het` | $+0.29 \; [+0.10, +0.49]$ ($0.999$) | **$+0.27 \; [+0.07, +0.48]$** ($0.996$) | **$+0.23 \; [+0.03, +0.44]$** ($0.988$) |
| | `f3k_est_pos` | $+0.37 \; [+0.14, +0.62]$ ($0.999$) | **$+0.30 \; [+0.07, +0.56]$** ($0.995$) | **$+0.28 \; [+0.04, +0.55]$** ($0.990$) |
| **`cat12_irreg13` (`MC_cross`) $\Delta\beta_{\text{br}-\text{ir}}$** | `ent` | $+0.57 \; [+0.22, +0.92]$ ($0.999$) | **$+0.36 \; [+0.03, +0.69]$** ($0.983$) | $+0.33 \; [-0.02, +0.68]$ ($0.969$) |
| | `sqrt_het` | $+0.41 \; [+0.10, +0.74]$ ($0.995$) | **$+0.36 \; [+0.05, +0.68]$** ($0.988$) | **$+0.32 \; [+0.01, +0.63]$** ($0.978$) |
| | `f3k_est_pos` | $+0.50 \; [+0.18, +0.85]$ ($0.999$) | **$+0.40 \; [+0.07, +0.76]$** ($0.993$) | **$+0.37 \; [+0.03, +0.73]$** ($0.982$) |

* **Positive Breadth slope is global, not an Inner Asian artifact:** Dropping Turkic + Mongolic-Khitan ($18$ languages) and Tungusic ($23$ languages total) preserves **77%–90%** of the raw-unit Breadth slope across all SEM and Beta-Binomial models (`bb_cat12` `ent`: $+0.20 \to +0.20 \to +0.19$, $P>0 \ge 0.993$; `f3k_est_pos`: $+0.19 \to +0.21 \to +0.20$, $P>0 = 0.993$).
* **Inner Asia's specific role in `MB_2pl` (`ent`):** Turkic and Mongolic languages combine the highest `ent` in the dataset with a complete absence of gender and NP number agreement. Removing them eliminates the negative `ent` slope on the 2 agreement items ($\bar{\beta}_{\text{agr}}: -0.31 \to +0.15$), whereas for **`sqrt_het`** the Inherent-vs-Agreement split remains credibly positive outside Inner Asia ($+0.49 \; [+0.10, +0.89], P = 0.995$ in `no_turk_mong`; $+0.42 \; [+0.05, +0.81], P = 0.992$ in `no_inner_asia`).

---

### 5.2 Check 2: Time-Calibrated Bouckaert `v2.0.0` Covariance (`P_b_stem`, `P_b_crown`, `P_b_kyr` vs. Grafen `P`)

Replaces the global topological Grafen correlation matrix (`P`) across all $N = 202$ `main` languages with three parameterizations derived from the time-calibrated Bouckaert `v2.0.0` phylogeny (`analyses/common/phylogeny.R`):
1. **`P_b_stem`:** Block-diagonal within-family correlation matrix using the global pruned-tree root age ($129.6\text{ kyr}$), so each multi-language family's stem branch acts as an implicit family intercept whose prior scale reflects its stem length.
2. **`P_b_crown`:** Block-diagonal within-family crown-MRCA correlation matrix (unit diagonal per family) paired with explicit i.i.d. family random intercepts $u_{\text{fam}, f} \sim \mathcal{N}(0, \sigma_{\text{fam}}^2)$.
3. **`P_b_kyr`:** True absolute-kyr within-family crown-MRCA covariance blocks (diagonal entries scaled by $\text{crown\_age}_f / \overline{\text{crown\_age}}$, preserving absolute time-depth differences between young families like Mongolic-Khitan [$0.79\text{ kyr}$] and deep families like Afro-Asiatic [$8.29\text{ kyr}$]) paired with explicit family random intercepts $u_{\text{fam}, f} \sim \mathcal{N}(0, \sigma_{\text{fam}}^2)$.

| Item Set & Arch | Exposure | Grafen `P` | Bouckaert `P_b_stem` | Bouckaert `P_b_crown` | Bouckaert `P_b_kyr` | $\Delta\text{elpd}_{\text{tip}}$ (`P_b_kyr` vs `P`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`cat12` (`MA_2pl`) $\beta$** | `ent` | $+0.35 \; [+0.15, +0.57]$ | $+0.34 \; [+0.15, +0.54]$ | $+0.34 \; [+0.14, +0.55]$ | **$+0.34 \; [+0.14, +0.55]$** | **$+5.1 \pm 2.2$** (fam: $3.0$) |
| | `sqrt_het` | $+0.22 \; [+0.04, +0.40]$ | $+0.23 \; [+0.06, +0.41]$ | $+0.22 \; [+0.04, +0.40]$ | **$+0.22 \; [+0.04, +0.41]$** | **$+4.9 \pm 2.1$** (fam: $2.9$) |
| | `f3k_est_pos` | $+0.35 \; [+0.13, +0.59]$ | $+0.31 \; [+0.11, +0.53]$ | $+0.31 \; [+0.11, +0.54]$ | **$+0.31 \; [+0.10, +0.54]$** | **$+6.2 \pm 2.4$** (fam: $3.4$) |
| **`cat12` (`MB_2pl`) $\Delta\beta_{\text{inh}-\text{agr}}$** | `ent` | $+0.73 \; [+0.38, +1.10]$ | $+0.73 \; [+0.38, +1.11]$ | $+0.73 \; [+0.37, +1.09]$ | **$+0.72 \; [+0.36, +1.09]$** | **$+3.8 \pm 1.9$** (fam: $2.5$) |
| | `sqrt_het` | $+0.77 \; [+0.39, +1.16]$ | $+0.79 \; [+0.42, +1.17]$ | $+0.78 \; [+0.39, +1.17]$ | **$+0.77 \; [+0.39, +1.16]$** | **$+4.7 \pm 2.0$** (fam: $2.7$) |
| | `f3k_est_pos` | $+0.34 \; [-0.00, +0.73]$ | $+0.33 \; [+0.00, +0.71]$ | $+0.33 \; [+0.00, +0.71]$ | $+0.33 \; [-0.00, +0.70]$ | **$+7.1 \pm 2.4$** (fam: $3.3$) |
| **`cat12_irreg13` (`MC_cross`) $\Delta\beta_{\text{br}-\text{ir}}$** | `ent` | $+0.57 \; [+0.22, +0.92]$ | $+0.59 \; [+0.25, +0.94]$ | $+0.59 \; [+0.23, +0.94]$ | **$+0.58 \; [+0.23, +0.93]$** | $+2.9 \pm 2.7$ (fam: $3.7$) |
| | `sqrt_het` | $+0.41 \; [+0.10, +0.74]$ | $+0.39 \; [+0.10, +0.69]$ | $+0.39 \; [+0.08, +0.70]$ | **$+0.39 \; [+0.09, +0.69]$** | $+3.5 \pm 3.0$ (fam: $3.5$) |
| | `f3k_est_pos` | $+0.50 \; [+0.18, +0.85]$ | $+0.47 \; [+0.17, +0.79]$ | $+0.49 \; [+0.17, +0.82]$ | **$+0.48 \; [+0.17, +0.80]$** | $+4.0 \pm 3.2$ (fam: $4.1$) |
| **`cat12_flex26` (`MC_cross`) $\Delta\beta_{\text{br}-\text{ir}}$** | `ent` | $+0.47 \; [+0.16, +0.78]$ | $+0.46 \; [+0.17, +0.76]$ | $+0.46 \; [+0.16, +0.77]$ | **$+0.46 \; [+0.15, +0.76]$** | $-0.8 \pm 3.0$ (fam: $3.4$) |
| | `sqrt_het` | $+0.28 \; [+0.00, +0.56]$ | $+0.27 \; [+0.01, +0.55]$ | $+0.26 \; [-0.01, +0.54]$ | $+0.26 \; [-0.01, +0.53]$ | $+0.9 \pm 3.1$ (fam: $3.6$) |
| | `f3k_est_pos` | $+0.42 \; [+0.15, +0.73]$ | $+0.40 \; [+0.14, +0.68]$ | $+0.41 \; [+0.14, +0.70]$ | **$+0.40 \; [+0.13, +0.69]$** | $+2.2 \pm 3.5$ (fam: $4.2$) |

* **Invariance of admixture slopes to phylogenetic time calibration:** Every posterior slope and contrast across `P_b_stem`, `P_b_crown`, and `P_b_kyr` lies within $\le 0.04$ of Grafen `P`.
* **Between- vs. within-family phylogenetic variance partitioning (`P_b_crown` & `P_b_kyr`):** Explicitly separating family intercepts ($\sigma_{\text{fam}}$) from within-family crown-MRCA phylogenetic drift ($\sigma_{\text{phylo}}$) shows that between-family differences account for the bulk of phylogenetic variance ($\sigma_{\text{fam}} = 1.14\text{--}1.17$ on Factor 1 Breadth and $1.66\text{--}1.69$ on Factor 2 Irregularity/Flexivity), while within-family phylogenetic drift accounts for $\sigma_{\text{phylo}} = 0.38\text{--}0.50$ on Factor 1 and $0.32\text{--}0.43$ on Factor 2. Preserving absolute crown ages in kyr (`P_b_kyr`) shifts variance on `cat12` from i.i.d. language noise ($\sigma_{\text{lang}}: 0.23 \to 0.18$) into time-calibrated within-family drift ($\sigma_{\text{phylo}}: 0.38 \to 0.50$), yielding a modest tip-marginalized LOO gain over Grafen `P` ($\Delta\text{elpd}_{\text{tip}} = +3.8\text{ to }+7.1 \pm 2.4$ on `cat12`, $+2.9\text{ to }+4.0 \pm 3.2$ on `cat12_irreg13`).

---

### 5.3 Check 3: Demographic Controls (`z_pop`, `x * z_pop`) and Spatial Gaussian Process (`F`)

#### 5.3.1 Speaker Population Size (`z_pop`) and Interaction (`x * z_pop`) on `has_pop == 1`

Tested on the exact subset of languages with non-missing L1 speaker population (`has_pop == 1`: $N = 195$ on `cat12`, $N = 171$ on `cat12_irreg13`, $N = 181$ on `cat12_flex26`) under Phylogeny `P`:

| Item Set & Arch | Exposure ($N$) | Spec | Breadth / Mean Slope $\beta_x$ [95% CI] | Irreg / Agr Slope [95% CI] | Contrast $\Delta\beta$ [95% CI] ($P>0$) | Population Slope $\beta_{\text{pop}}$ (F1 / F2) | Interaction $\beta_{x \times \text{pop}}$ (F1 / F2) | $\Delta\text{elpd}_{\text{tip}}$ vs. Nested |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`cat12` (`MB_2pl`)** | `ent` ($195$) | `x` | $+0.31 \; [+0.11, +0.51]$ | $\bar{\beta}_{\text{agr}} = -0.32$ | **$+0.75 \; [+0.38, +1.12]$** ($1.000$) | — | — | $+18.4 \pm 7.2$ (vs `base`) |
| | `ent` ($195$) | `x_pop` | $+0.31 \; [+0.11, +0.52]$ | $\bar{\beta}_{\text{agr}} = -0.31$ | **$+0.75 \; [+0.38, +1.13]$** ($1.000$) | $-0.04 \; [-0.26, +0.19]$ | — | $-0.5 \pm 0.4$ (vs `x`) |
| | `ent` ($195$) | `x_int_pop` | $+0.31 \; [+0.10, +0.52]$ | $\bar{\beta}_{\text{agr}} = -0.32$ | **$+0.75 \; [+0.38, +1.13]$** ($1.000$) | $-0.04 \; [-0.26, +0.19]$ | $-0.03 \; [-0.21, +0.15]$ | $-1.9 \pm 0.8$ (vs `x_pop`) |
| | `sqrt_het` ($191$) | `x_int_pop` | $+0.24 \; [+0.05, +0.43]$ | $\bar{\beta}_{\text{agr}} = -0.41$ | **$+0.77 \; [+0.39, +1.16]$** ($1.000$) | $-0.06 \; [-0.28, +0.15]$ | $-0.08 \; [-0.26, +0.09]$ | $-1.5 \pm 1.3$ (vs `x_pop`) |
| | `f3k_est_pos` ($176$) | `x_int_pop` | $+0.33 \; [+0.12, +0.56]$ | $\bar{\beta}_{\text{agr}} = +0.03$ | **$+0.36 \; [+0.01, +0.76]$** ($0.979$) | $-0.06 \; [-0.30, +0.19]$ | $+0.08 \; [-0.15, +0.32]$ | $-0.6 \pm 1.3$ (vs `x_pop`) |
| **`cat12_irreg13` (`MC_cross`)** | `ent` ($171$) | `x` | $+0.43 \; [+0.20, +0.65]$ | $-0.13 \; [-0.41, +0.15]$ | **$+0.56 \; [+0.20, +0.92]$** ($0.999$) | — | — | $+8.6 \pm 3.4$ (vs `base`) |
| | `ent` ($171$) | `x_pop` | $+0.45 \; [+0.22, +0.68]$ | $-0.12 \; [-0.41, +0.16]$ | **$+0.57 \; [+0.21, +0.94]$** ($0.999$) | $-0.15$ / $+0.05$ | — | $-1.8 \pm 1.0$ (vs `x`) |
| | `ent` ($171$) | `x_int_pop` | $+0.45 \; [+0.21, +0.68]$ | $-0.10 \; [-0.38, +0.18]$ | **$+0.55 \; [+0.19, +0.91]$** ($0.999$) | $-0.16$ / $+0.06$ | $+0.00$ / $+0.09$ | $-1.2 \pm 0.8$ (vs `x_pop`) |
| | `sqrt_het` ($167$) | `x_int_pop` | $+0.31 \; [+0.11, +0.52]$ | $-0.11 \; [-0.38, +0.15]$ | **$+0.42 \; [+0.09, +0.76]$** ($0.994$) | $-0.17$ / $+0.05$ | $-0.03$ / $-0.03$ | $-2.4 \pm 1.0$ (vs `x_pop`) |
| | `f3k_est_pos` ($155$) | `x_int_pop` | $+0.42 \; [+0.18, +0.69]$ | $-0.13 \; [-0.36, +0.08]$ | **$+0.55 \; [+0.21, +0.90]$** ($0.999$) | $-0.13$ / $+0.07$ | $+0.07$ / $-0.02$ | $-3.1 \pm 1.5$ (vs `x_pop`) |
| **`cat12_flex26` (`MC_cross`)** | `ent` ($181$) | `x_int_pop` | $+0.43 \; [+0.21, +0.66]$ | $-0.04 \; [-0.25, +0.18]$ | **$+0.47 \; [+0.16, +0.78]$** ($0.999$) | $-0.06$ / $+0.07$ | $+0.02$ / $+0.02$ | $-0.3 \pm 1.6$ (vs `x_pop`) |
| | `sqrt_het` ($178$) | `x_int_pop` | $+0.30 \; [+0.10, +0.50]$ | $-0.00 \; [-0.20, +0.19]$ | **$+0.30 \; [+0.02, +0.59]$** ($0.982$) | $-0.08$ / $+0.07$ | $-0.03$ / $-0.02$ | $-1.6 \pm 1.5$ (vs `x_pop`) |
| | `f3k_est_pos` ($163$) | `x_int_pop` | $+0.40 \; [+0.16, +0.66]$ | $-0.04 \; [-0.19, +0.10]$ | **$+0.44 \; [+0.16, +0.75]$** ($0.999$) | $-0.07$ / $+0.07$ | $+0.06$ / $-0.04$ | $-1.4 \pm 1.7$ (vs `x_pop`) |

* Under phylogenetic control (`P`), speaker population size `z_pop` and its interaction `x * z_pop` have null slopes across both latent factors (every 95% CI straddles zero) and consistently penalize out-of-sample LOO ($\Delta\text{elpd}_{\text{tip}} = -0.0\text{ to }-1.8$ for `x_pop` vs. `x`, and $-0.3\text{ to }-3.1$ for `x_int_pop` vs. `x_pop`), while leaving every admixture slope and contrast intact.

#### 5.3.2 Exact Spatial Gaussian Process (`F`: Phylogeny + Space vs. `P`: Phylogeny)

Adding an exact great-circle Matérn / exponentiated-quadratic Gaussian process $u_{\text{space}, i}$ alongside $u_{\text{phylo}, i}$ (`ctrl_code = F`) across all winning architectures (`MA_2pl`, `MB_2pl` on `cat12`; `MC_cross` on `cat12_irreg13` and `cat12_flex26`) preserves every key slope and contrast:
- **`cat12` (`MA_2pl`, `F`):** `ent` $\beta = +0.32 \; [+0.09, +0.55]$ ($P>0 = 0.997$); `sqrt_het` $+0.20 \; [+0.02, +0.39]$ ($P>0 = 0.985$); `f3k_est_pos` $+0.33 \; [+0.10, +0.59]$ ($P>0 = 0.998$).
- **`cat12` (`MB_2pl`, `F`):** $\Delta\beta_{\text{inh}-\text{agr}} = +0.73 \; [+0.37, +1.11]$ ($P>0 = 1.000$) for `ent`, $+0.78 \; [+0.41, +1.17]$ ($P>0 = 1.000$) for `sqrt_het`, and $+0.33 \; [-0.01, +0.72]$ ($P>0 = 0.971$) for `f3k_est_pos`, with strong predictive gain over the spatial baseline (`MB_2pl` `F` vs. `MA0_2pl` `F`: $\Delta\text{elpd}_{\text{obs}} = +14.8 \pm 6.0$ for `ent`, $+8.6 \pm 5.2$ for `sqrt_het`).
- **`cat12_irreg13` (`MC_cross`, `F`):** `ent` $\beta_{\text{breadth}} = +0.45 \; [+0.21, +0.70]$, $\beta_{\text{irreg}} = -0.16 \; [-0.45, +0.13]$, $\Delta\beta = +0.61 \; [+0.23, +0.98]$ ($P>0 = 0.999$); `sqrt_het` $\Delta\beta = +0.36 \; [+0.03, +0.71]$ ($P>0 = 0.984$); `f3k_est_pos` $\Delta\beta = +0.47 \; [+0.15, +0.82]$ ($P>0 = 0.999$).
- **`cat12_flex26` (`MC_cross`, `F`):** `ent` $\Delta\beta = +0.52 \; [+0.19, +0.86]$ ($P>0 = 0.999$); `sqrt_het` $\Delta\beta = +0.23 \; [-0.05, +0.53]$ ($P>0 = 0.946$); `f3k_est_pos` $\Delta\beta = +0.40 \; [+0.12, +0.70]$ ($P>0 = 0.997$).
- *Note on `F` vs. `P` LOO comparisons:* Because the spatial GP adds a second correlated $N \times N$ latent vector $u_{\text{space}}$ that is conditioned on in `log_lik_lang`, `F_vs_P` LOO differences are not tip-marginalized (`delta_elpd_lang_tip = NA`); nested `x_vs_base_F` comparisons within `F` evaluate admixture predictive gains on a matched spatial + phylogenetic surface.

