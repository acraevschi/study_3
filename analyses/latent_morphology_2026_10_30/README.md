# Latent-Variable and Multi-Item Morphological Models (`latent_morphology_2026_10_30`)

**Exploratory.** Extends `analyses/admixture_links_2026_10_21/` on the `latent-morphology` branch by replacing the equal-weight sum-score Beta-Binomial outcome (`n_present | trials(n_coded)`) with item-level latent-variable and hierarchical varying-slope models over the binary language $\times$ feature observation matrix $Y_{ik} \in \{0, 1\}$.

**Status 2026-10-31: fitted (`main`, Phylogeny `P`, 42 models; results in §4).** All 42 models converged with $\max \hat{R} \le 1.007$, minimum bulk ESS $\ge 554$, and $0$ divergent transitions.

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
* **Marginal Language-Level LOO (`log_lik_lang`):** In `generated quantities`, language-specific i.i.d. random effects (`z_lang[i]` and `z_testlet[i, 1:T_test]`) are integrated out via nested Gauss-Hermite quadrature conditioned on the structural surface $u_{\text{phylo}, i} + u_{\text{space}, i}$ and $x_i$ (Merkle et al. 2019), avoiding leave-one-language-out Pareto-$k$ breakdown and directly evaluating $x_i$ at the language level.

### 2.2 Two-Factor Joint Models on `cat12_irreg13` ($K = 25$) and `cat12_flex26` ($K = 38$) (`stan/item_2factor.stan`)

Jointly estimates **Factor 1 (Enumerative Breadth, $\theta_{i, 1}$)** on items $1\dots 12$ (`cat12`) and **Factor 2 (Paradigmatic Irregularity/Flexivity, $\theta_{i, 2}$)** on items $13\dots K$ (`irreg13` or `flex26`), with correlated language-level and phylogenetic residuals ($\rho_{\text{latent}}$), factor-specific 2PL loadings $\boldsymbol{\lambda}_1, \boldsymbol{\lambda}_2$ (arithmetic mean $1.0$ within each factor), and factor-specific admixture slopes $(\beta_{\text{breadth}}, \beta_{\text{irreg}})$ with posterior difference $\Delta\beta = \beta_{\text{breadth}} - \beta_{\text{irreg}}$:

* **`MC0_strict` / `MC_strict`:** Strict confirmatory factor assignment ($1\dots 12$ load on $\theta_{i, 1}$; $13\dots K$ load on $\theta_{i, 2}$).
* **`MC0_cross` / `MC_cross`:** Allows the 2 contextual agreement categories in `cat12` (`gender_agreement`, `number_agreement`) to also cross-load ($\gamma_{\text{cross}, 1}, \gamma_{\text{cross}, 2}$) onto Factor 2 ($\theta_{i, 2}$).

## 3. Files and Usage

```bash
# 1. Build item_observations.csv, items_metadata.csv, language_table.csv, prep_summary.json
/usr/bin/python3 analyses/latent_morphology_2026_10_30/prepare.py

# 2. Dry-run validation (checks Stan syntax and all model data structures without sampling)
LAT_DRY=1 Rscript analyses/latent_morphology_2026_10_30/fit_latent.R

# 3. Smoke test (fast 400-iteration check into fits_smoke/ and results_smoke/)
LAT_SMOKE=1 Rscript analyses/latent_morphology_2026_10_30/fit_latent.R

# 4. Core run (ent, sqrt_het, f3k_est_pos under P control on main; 42 models; default LAT_CTRLS=P,F runs 84)
LAT_CTRLS=P Rscript analyses/latent_morphology_2026_10_30/fit_latent.R main
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
