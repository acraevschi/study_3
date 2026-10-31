# Within-Family Phylogenetic Branch-Rate (`cov_rrw`) & Contrasts Analysis (2026-10-30)

**Exploratory methodological test.** Tests the proposal to model genetic shifts and morphological evolution along the branches of a time-calibrated language phylogeny via a Relaxed Random Walk (`cov_rrw`), restricted to **within-family branches of multi-language families** ($\ge 5$ languages per family as primary, $\ge 3$ as sensitivity) so that deep pre-proto-family macro-population stems and the 29 singleton families do not confound the branch rates.

---

## 1. Design & Sample Structure

All 202 `main`-set Grambank languages in [`analyses/grambank/data/language_data.csv`](../data/language_data.csv) match the time-calibrated global tree of Bouckaert et al. (`v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz`, branch lengths in **kyr** = thousands of years): 198 by exact 8-character Glottocode and 4 Glottolog 5.3 splits mapped 1-to-1 without collision to their pre-split/sister tips (`iron1242` $\leftarrow$ `osse1243`, `minh1238` $\leftarrow$ `tuuu1240`, `tanz1241` $\leftarrow$ `ngon1269`, `mlah1239` $\leftarrow$ `turo1239`).

Each multi-language family $f$ is pruned separately to its own crown MRCA (`ape::keep.tip`), forming a forest of $F$ independent time-calibrated family trees with zero shared root stems between families:

| Family | Tips (`cat12`) | Within-Family Branches ($2T_f - 2$) | Internal Contrasts ($T_f - 1$) | Crown Age (kyr) | Mean Branch Length (kyr) | Set |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Austronesian | 46 | 90 | 45 | 5.41 | 1.30 | `min5`, `min3` |
| Indo-European | 24 | 46 | 23 | 6.97 | 1.69 | `min5`, `min3` |
| Afro-Asiatic | 14 | 26 | 13 | 8.29 | 1.63 | `min5`, `min3` |
| Atlantic-Congo | 13 | 24 | 12 | 6.78 | 1.98 | `min5`, `min3` |
| Turkic | 12 | 22 | 11 | 2.37 | 0.85 | `min5`, `min3` |
| Uralic | 12 | 22 | 11 | 5.66 | 1.57 | `min5`, `min3` |
| Sino-Tibetan | 7 | 12 | 6 | 7.25 | 2.89 | `min5`, `min3` |
| Mongolic-Khitan | 6 | 10 | 5 | 0.79 | 0.39 | `min5`, `min3` |
| Nakh-Daghestanian | 5 | 8 | 4 | 5.11 | 2.42 | `min5`, `min3` |
| Tungusic | 5 | 8 | 4 | 1.87 | 0.90 | `min5`, `min3` |
| Austroasiatic | 3 | 4 | 2 | 6.64 | 4.86 | `min3` only |
| Dravidian | 3 | 4 | 2 | 5.44 | 3.88 | `min3` only |
| Tupian | 3 | 4 | 2 | 6.26 | 3.60 | `min3` only |
| **Total (`min5`, $\ge 5$ langs)** | **144** | **268** | **134** | — | — | **10 families** |
| **Total (`min3`, $\ge 3$ langs)** | **153** | **280** | **140** | — | — | **13 families** |

Across the three morphological outcomes (all filtered at $\ge 60\%$ coded features, matching `fit.R`):
- **`cat12`** (12 inflectional categories — categorial breadth): $N = 144$ (`min5`, 10 families) / $N = 153$ (`min3`, 13 families).
- **`flex`** (30 Grambank Flexivity features): $N = 136$ (`min5`, 10 families) / $N = 142$ (`min3`, 12 families).
- **`irreg13`** (13 core allomorphy / suppletion / inflection-class features): $N = 125$ (`min5`, 10 families) / $N = 131$ (`min3`, 12 families).

---

## 2. Six Genetic Representations Compared

To test whether the choice of genetic statistic and coordinate embedding affects branch-jump inference, we evaluate six genetic representations side-by-side:

1. **`ent` ($d = 1$):** Stage-1 ADMIXTURE Shannon entropy ($K = 12..30$ averaged, with individual/population hierarchical SE).
2. **`sqrt_het` ($d = 1$):** Stage-1 square-root within-population inter-individual ancestry spread (`het`, with bootstrap SE).
3. **`f3k_est_pos` ($d = 1$):** Stage-1 clamped $5$-nearest within-family $\times$ out-of-family admixture $f_3$ statistic ($\max(0, -f_3)$).
4. **`q_within_d3` ($d = 3$):** Top 3 within-family principal coordinates of the language-level ADMIXTURE $Q$-vectors ($K = 12..30$ concatenated, scaled by a single global constant so natural ancestry distances are preserved across families, with individual-level SE propagated to each axis).
5. **`fst_within_d3` ($d = 3$):** Top 3 within-family PCoA axes (Cailliez-corrected) of the pairwise $F_{ST}$ matrix (`input/FST.csv` from Zenodo 15263706), scaled by a single global constant across families.
6. **`fst_global_d3` ($d = 3$):** Top 3 global PCoA axes of the $202 \times 202$ pairwise $F_{ST}$ matrix (the literal global $F_{ST}$ embedding from the original sketch).

---

## 3. Finding 1: Demographic Audit of Genetic Branch Jumps ($F_{ST}$ Confounds Bottlenecks with Contact)

Both the analytical PIC branch-rate reconstruction (`contrasts_diagnostic.R` $\to$ `results/jump_demographic_audit.csv`) and the Stage-1 Bayesian RRW model (`gen_rrw.stan` $\to$ `results/stage1_branch_jumps.csv`) confirm the central population-genetic critique from the statistical review:

### 3.1 Correlation of Terminal Branch Jump Rates with GeLaTo Effective Population Size ($N_e$) and Sample Size ($n_i$)

| Genetic Representation | PIC Rate $\rho(N_e)$ (`min5`) | PIC Rate $\rho(N_e)$ (`min3`) | Stan $\hat{J}_b$ $\rho(N_e)$ (`min5`) | Stan $\hat{J}_b$ $\rho(\text{ent})$ | Stan $\hat{J}_b$ $\rho(f_3)$ | Diagnostic Reading |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`fst_within_d3`** | **$-0.404$** ($p = 1.0 \times 10^{-5}$) | **$-0.430$** ($p = 8.7 \times 10^{-7}$) | **$-0.334$** ($p = 3.5 \times 10^{-4}$) | $+0.282$ | $+0.007$ | **Heavily confounded by low-$N_e$ isolation bottlenecks** |
| **`fst_global_d3`** | $-0.121$ ($p = 0.20$) | $-0.136$ ($p = 0.14$) | $-0.096$ ($p = 0.31$) | $+0.280$ | $+0.066$ | **Dominated by $n=1\text{--}2$ small-sample noise & intercontinental axes** |
| **`q_within_d3`** | $-0.066$ ($p = 0.49$) | $-0.100$ ($p = 0.27$) | **$-0.010$** ($p = 0.91$) | **$+0.329$** | $+0.098$ | **Unconfounded by $N_e$; captures within-family ancestry shifts** |
| **`ent`** | $-0.043$ ($p = 0.65$) | $-0.011$ ($p = 0.91$) | $-0.123$ ($p = 0.20$) | $+0.081$ | $-0.006$ | Unconfounded by $N_e$ |
| **`sqrt_het`** | $-0.197$ ($p = 0.038$) | $-0.189$ ($p = 0.038$) | $-0.146$ ($p = 0.13$) | $+0.080$ | $+0.143$ | Mild negative $N_e$ correlation |
| **`f3k_est_pos`** | $+0.172$ ($p = 0.070$) | $+0.193$ ($p = 0.034$) | $+0.024$ ($p = 0.80$) | **$+0.338$** | **$+0.415$** | Unconfounded by $N_e$; tracks positive admixture evidence |

### 3.2 Concrete Examples from the Top Stage-1 Stan Branch Jumps (`results/stage1_branch_jumps.csv`)
- **Under `fst_within_d3`:** The #1 largest branch jump in the entire Stan RRW model is **Rennell-Bellona (`renn1242`, $\hat{J}_b = +3.12$, rate multiplier $25.7\times$)**—an isolated Polynesian outlier island with an effective population size of just **$N_e = 267.9$**, strongly *low* admixture entropy (`-1.42 SD`), and clamped $f_3 = 0$. Other top `fst_within_d3` jumps include **Nganasan (`ngan1291`, $\hat{J}_b = +2.00$, $N_e = 772$, `ent` $= -2.04$ SD)** and **Mrassu Shor-Tutal (`shor1247`, $\hat{J}_b = +2.36$, $N_e = 811$)**. Because $F_{ST}$ has within-population heterozygosity in its denominator, isolated populations with strong genetic drift produce massive $F_{ST}$ branch jumps that mimic contact.
- **Under `fst_global_d3`:** The #1 and #2 largest Stan branch jumps in the dataset are **Southern Sotho (`sout2807`, $n = 1$ individual, $\hat{J}_b = +3.68$, rate multiplier $46.4\times$)** and **Fe'fe' (`fefe1239`, $n = 2$ individuals, $\hat{J}_b = +3.33$, rate multiplier $32.9\times$)**.
- **Under `q_within_d3` (ADMIXTURE $Q$ within-family axes):** The #1 and #2 largest Stan branch jumps are **Gagauz (`gaga1249`, Balkan Turkic, $\hat{J}_b = +2.30$, rate multiplier $13.0\times$, $N_e = 13,524$)** and **Maltese (`malt1254`, Semitic in Europe, $\hat{J}_b = +2.21$, rate multiplier $10.2\times$, `graff_target = 1`)**—canonical historical contact languages whose speakers shifted toward neighboring out-of-family gene pools.

---

## 4. Finding 2: Within-Family Phylogenetic Independent Contrasts (`results/pic_contrasts_summary.csv`)

Across the $134$ internal node contrasts in `min5` ($10$ families) and $140$ contrasts in `min3` ($13$ families):

| Outcome | Genetic Measure | `min5` Contrasts | Rate Test: Spearman $\rho(|c^{\text{morph}}|, \|c^{\text{gen}}\|)$ ($p$) | Directional Test: Standardized Slope $\Delta M \sim 0 + \Delta G$ ($\pm$ SE, $p$) |
| :--- | :--- | :---: | :---: | :---: |
| **`cat12`** | `ent` | 134 | **$-0.285$** ($p = 0.00085$) | **$+0.182 \pm 0.085$** ($p = 0.035$) |
| | `sqrt_het` | 133 | $+0.151$ ($p = 0.084$) | $+0.092 \pm 0.087$ ($p = 0.289$) |
| | `f3k_est_pos` | 133 | $-0.003$ ($p = 0.971$) | **$+0.217 \pm 0.085$** ($p = 0.012$) |
| | `q_within_d3` | 134 | **$-0.194$** ($p = 0.025$) | $+0.122 \pm 0.086$ ($p = 0.158$) |
| | `fst_within_d3` | 134 | $-0.069$ ($p = 0.426$) | $+0.106 \pm 0.086$ ($p = 0.221$) |
| | `fst_global_d3` | 134 | $-0.105$ ($p = 0.229$) | **$+0.199 \pm 0.085$** ($p = 0.020$) |
| **`flex`** | `ent` | 126 | $-0.034$ ($p = 0.709$) | $+0.090 \pm 0.089$ ($p = 0.315$) |
| | `sqrt_het` | 125 | $+0.098$ ($p = 0.277$) | $-0.074 \pm 0.090$ ($p = 0.409$) |
| | `f3k_est_pos` | 125 | $-0.148$ ($p = 0.100$) | $+0.087 \pm 0.089$ ($p = 0.332$) |
| | `q_within_d3` | 126 | $-0.029$ ($p = 0.745$) | $+0.072 \pm 0.089$ ($p = 0.422$) |
| | `fst_within_d3` | 126 | $+0.030$ ($p = 0.738$) | $-0.103 \pm 0.089$ ($p = 0.249$) |
| | `fst_global_d3` | 126 | $-0.012$ ($p = 0.895$) | $-0.096 \pm 0.089$ ($p = 0.281$) |
| **`irreg13`** | `ent` | 115 | $+0.005$ ($p = 0.957$) | $-0.035 \pm 0.094$ ($p = 0.708$) |
| | `sqrt_het` | 114 | $+0.080$ ($p = 0.395$) | $+0.001 \pm 0.094$ ($p = 0.995$) |
| | `f3k_est_pos` | 114 | $-0.122$ ($p = 0.197$) | $+0.000 \pm 0.094$ ($p = 0.997$) |
| | `q_within_d3` | 115 | $-0.097$ ($p = 0.301$) | $+0.051 \pm 0.094$ ($p = 0.584$) |
| | `fst_within_d3` | 115 | $-0.088$ ($p = 0.350$) | $-0.067 \pm 0.093$ ($p = 0.472$) |
| | `fst_global_d3` | 115 | $-0.133$ ($p = 0.157$) | $-0.110 \pm 0.093$ ($p = 0.238$) |

- **No positive rate coupling (`|c_morph|` vs. `||c_gen||`):** None of the 18 outcome $\times$ genetic combinations shows significant positive rate coupling; for `cat12`, nodes with larger `ent` or `q_within_d3` contrasts actually have *smaller* absolute morphological contrasts ($\rho = -0.285$ and $-0.194$).
- **Directional contrasts reproduce the cross-sectional `cat12` pattern:** On `cat12`, the sister lineage with higher admixture (`ent` $+0.182, p = 0.035$; `f3k_est_pos` $+0.217, p = 0.012$) has *more* inflectional categories, whereas on `flex` and `irreg13` all directional contrast slopes are statistically indistinguishable from zero.

---

## 5. Finding 3: Two-Stage Stan RRW & Directional Drift Models (`results/stage2_morph_rrw_results.csv`)

### 5.1 MCMC Diagnostics
- **Stage 1 (`gen_rrw.stan`, 12 fits):** All 12 models converged cleanly describing substantial within-family genetic rate heterogeneity ($\hat{\tau}_J \in [0.68, 1.95]$ across representations).
- **Stage 2 (`morph_rrw.stan`, 150 fits):** $\max \hat{R} = 1.013$, minimum bulk $\text{ESS} = 591$, and only 22 divergent transitions across all $150 \times 4,000 = 600,000$ post-warmup draws (140 of 150 models had 0 divergences).

### 5.2 Primary Set Results (`min5`: 10 families, $\ge 5$ languages/family)

Each cell reports the posterior mean and 95% credible interval (with $\Delta\text{elpd}_{\text{LOO}}$ against the strict-clock within-family phylogenetic baseline in parentheses):

| Outcome ($N$) | Genetic Measure | `rate`: Volatility $\gamma_{\text{rate}}$ ($\Delta\text{elpd}$) | `dir_jump`: Cumul. Rate Jump $\beta_{\text{dir}}$ ($\Delta\text{elpd}$) | `dir_signed`: Signed Root Shift $\beta_{\text{dir}}$ ($\Delta\text{elpd}$) |
| :--- | :--- | :---: | :---: | :---: |
| **`cat12`** ($N=144$) | `ent` | $-0.18\;[-1.02, +0.45]$ ($-0.6$) | $-0.05\;[-0.25, +0.15]$ ($-1.3$) | **$+0.16\;[-0.00, +0.32]$** ($+1.2$) |
| | `sqrt_het` | $+0.00\;[-0.85, +0.49]$ ($+0.1$) | $+0.08\;[-0.05, +0.20]$ ($-1.4$) | $+0.09\;[-0.03, +0.22]$ ($-0.6$) |
| | `f3k_est_pos` | $+0.22\;[-0.73, +0.76]$ ($+1.1$) | $+0.11\;[-0.05, +0.27]$ ($-0.9$) | **$+0.21\;[+0.02, +0.40]$** ($+1.3$) |
| | `q_within_d3` | $-0.28\;[-1.01, +0.56]$ ($-0.1$) | $+0.06\;[-0.09, +0.22]$ ($-0.9$) | $+0.17\;[-0.03, +0.38]$ ($+0.9$) |
| | `fst_within_d3` | $-0.26\;[-1.05, +0.54]$ ($-0.0$) | $-0.03\;[-0.21, +0.14]$ ($-0.6$) | $+0.22\;[-0.02, +0.47]$ ($+1.4$) |
| | `fst_global_d3` | $-0.24\;[-1.06, +0.52]$ ($+0.0$) | $+0.04\;[-0.13, +0.20]$ ($-1.4$) | **$+0.21\;[+0.02, +0.39]$** ($+1.7$) |
| **`flex`** ($N=136$) | `ent` | $-0.16\;[-0.94, +0.44]$ ($-0.2$) | $+0.08\;[-0.12, +0.26]$ ($-0.8$) | $-0.03\;[-0.18, +0.12]$ ($-0.7$) |
| | `sqrt_het` | $-0.22\;[-0.99, +0.27]$ ($-0.3$) | $-0.04\;[-0.18, +0.09]$ ($-0.4$) | $-0.04\;[-0.17, +0.08]$ ($-0.7$) |
| | `f3k_est_pos` | $-0.31\;[-1.10, +0.27]$ ($+0.3$) | $-0.07\;[-0.22, +0.08]$ ($-0.1$) | $-0.03\;[-0.19, +0.12]$ ($-1.3$) |
| | `q_within_d3` | $+0.13\;[-0.57, +0.80]$ ($-0.4$) | $-0.00\;[-0.13, +0.13]$ ($-1.2$) | $+0.02\;[-0.16, +0.21]$ ($-0.9$) |
| | `fst_within_d3` | $-0.05\;[-0.76, +0.54]$ ($-0.3$) | $-0.01\;[-0.19, +0.16]$ ($-0.9$) | $-0.01\;[-0.25, +0.24]$ ($-0.8$) |
| | `fst_global_d3` | $-0.14\;[-1.01, +0.60]$ ($-0.4$) | $-0.12\;[-0.25, +0.03]$ ($+0.0$) | $-0.06\;[-0.23, +0.11]$ ($-1.0$) |
| **`irreg13`** ($N=125$) | `ent` | $-0.14\;[-0.96, +0.53]$ ($+0.1$) | $+0.08\;[-0.22, +0.37]$ ($-0.5$) | $-0.07\;[-0.29, +0.17]$ ($-0.9$) |
| | `sqrt_het` | $-0.21\;[-1.00, +0.36]$ ($-0.1$) | $+0.00\;[-0.21, +0.19]$ ($-1.2$) | $-0.04\;[-0.23, +0.14]$ ($-0.9$) |
| | `f3k_est_pos` | $-0.18\;[-1.04, +0.51]$ ($-0.2$) | $-0.11\;[-0.38, +0.13]$ ($-0.5$) | $-0.03\;[-0.28, +0.22]$ ($-1.0$) |
| | `q_within_d3` | $-0.04\;[-0.82, +0.71]$ ($+0.0$) | $-0.08\;[-0.29, +0.13]$ ($-0.5$) | $-0.12\;[-0.38, +0.15]$ ($-0.6$) |
| | `fst_within_d3` | $-0.12\;[-0.93, +0.63]$ ($-0.1$) | $-0.10\;[-0.35, +0.15]$ ($-0.9$) | $-0.16\;[-0.49, +0.19]$ ($-0.5$) |
| | `fst_global_d3` | $-0.15\;[-0.97, +0.60]$ ($-0.1$) | $-0.14\;[-0.34, +0.07]$ ($+0.2$) | $-0.14\;[-0.39, +0.10]$ ($-0.1$) |

*(The `min3` sensitivity set [$13$ families, $N = 153$ on `cat12`] gives virtually identical results: `dir_signed` on `cat12` is $+0.14\;[-0.01, +0.29]$ for `ent`, $+0.21\;[+0.01, +0.44]$ for `f3k_est_pos`, and $+0.21\;[+0.01, +0.41]$ for `q_within_d3`, while all `rate` and `dir_jump` models have $\Delta\text{elpd} \in [-1.8, +1.1]$; see `results/stage2_morph_rrw_results.csv`).*

---

## 6. Key Conclusions

1. **Rescaling the phylogenetic covariance by genetic branch jumps (`cov_rrw` / $\gamma_{\text{rate}}$) adds zero predictive signal:**
   Across all 36 `rate` and `both` models (2 family cutoffs $\times$ 3 outcomes $\times$ 6 genetic measures), the rate-coupling exponent $\gamma_{\text{rate}}$ has wide 95% credible intervals spanning zero (typically $[-1.0, +0.6]$), most point estimates are slightly negative ($-0.32$ to $+0.22$), and $\Delta\text{elpd}_{\text{LOO}}$ against the strict clock is indistinguishable from zero ($-1.7$ to $+1.1$). A single morphological summary count per tip simply cannot support branch-specific variance stretching, and genetic branch jumps do not accelerate undirected morphological volatility.
2. **Unsigned branch rate jumps (`dir_jump`) lose the directional information of admixture:**
   Because an unsigned branch jump $\hat{J}_b > 0$ treats a branch that rapidly *lost* genetic diversity (via bottleneck drift, e.g. Nganasan or Rennell-Bellona) identically to a branch that rapidly *gained* outside ancestry (via admixture, e.g. Northern Uzbek or Maltese), accumulating $\hat{J}_b$ along branches (`dir_jump`) washes out the `cat12` slope ($\beta_{\text{dir}} = -0.05$ for `ent`, $+0.11$ for `f3k_est_pos`).
3. **Signed within-family genetic divergence (`dir_signed`) on the time-calibrated Bouckaert tree reproduces the main `fit.R` findings:**
   When we use the signed divergence from each family's reconstructed root (`dir_signed`) while controlling for the time-calibrated Bouckaert `v2.0.0` within-family phylogeny:
   - **`cat12` (categorial breadth):** Positive association with admixture (`ent` $+0.16$, `f3k_est_pos` $+0.21$, `q_within_d3` $+0.17$ in `min5`, $+0.21$ in `min3`), matching `analyses/grambank/baseline/` ($+0.18$ to $+0.19$).
   - **`flex` and `irreg13` (paradigmatic opacity / irregularity):** Null to slightly negative slopes ($-0.03$ to $-0.16$, with $P(\beta < 0) = 0.80\text{--}0.91$ on `irreg13` for the 3D divergence measures).

---

## 7. Files & How to Reproduce

| File | Description |
| :--- | :--- |
| `prepare.R` | Prunes Bouckaert `v2.0.0` per family, builds incidence matrices $\tilde{\mathbf{A}}_f$, and constructs the 6 genetic representations |
| `contrasts_diagnostic.R` | Runs the $N_e$ / bottleneck audit of branch jumps and within-family Phylogenetic Independent Contrasts (`ape::pic`) |
| `gen_rrw.stan` | Stage 1 Stan model: within-family block-diagonal Relaxed Random Walk on genetic coordinates |
| `morph_rrw.stan` | Stage 2 Stan model: within-family Beta-Binomial morphological model with `clock`, `rate`, `dir_jump`, `both`, and `dir_signed` |
| `fit_rrw.R` | Compiles and runs all 12 Stage-1 and 150 Stage-2 Stan models sequentially on $\le 4$ CPU cores |
| `results/*.csv` | All summary tables (`jump_demographic_audit.csv`, `pic_contrasts_summary.csv`, `stage1_genetic_rrw_diagnostics.csv`, `stage1_branch_jumps.csv`, `stage2_morph_rrw_results.csv`, `top15_jumps_*.csv`) |

```bash
Rscript analyses/grambank/phylo_rates/prepare.R
Rscript analyses/grambank/phylo_rates/contrasts_diagnostic.R
Rscript analyses/grambank/phylo_rates/fit_rrw.R
```
