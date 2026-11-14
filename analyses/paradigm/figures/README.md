# Track 2 Visualization Suite (`analyses/paradigm/figures/`)

Publication-grade figure suite for Study 3 Track 2 (**Morphological Paradigm Predictability / *i-complexity***), styled to match the Track 1 synthesis visual grammar (`outputs/study3_synthesis_2026_11_03/`): `6.5 in` final width, Helvetica typography (`>= 7.0 pt` floor at final size), `300 DPI` PNG + vector PDF masters, orthogonal semantic palettes reserving `#1c5cab` (deep blue) and `#d95f26` (burnt orange) strictly for **Enumerative breadth** vs. **Paradigmatic irregularity** while encoding POS consistently across Track 2 (**Verbs**: `#0d7a68` deep pine-teal / circle / solid; **Nouns**: `#9e6b06` ochre-bronze / triangle / dashed), and verified via greyscale, CVD (Machado et al. 2009 protanopia/deuteranopia), and blur contact sheets.

---

## 1. Script & Output Inventory

| Script | Rendered Figures | Primary Empirical Inputs |
| :--- | :--- | :--- |
| [`_style.R`](_style.R) | — | Shared palette, typography, ROPE fill, and `save_track2_fig()` helpers |
| [`run_qc.R`](run_qc.R) | — | 150-DPI print preview, Machado CVD + greyscale + box-blur QC contact sheet, font-size floor check |
| [`make_fig1_mgl_map_and_complexity.R`](make_fig1_mgl_map_and_complexity.R) | `fig1_mgl_map_and_complexity.{png,pdf}` | `unit_summary.csv`, `unit_outcomes_uniform120.csv`, `language_data_{V,N}.csv` |
| [`make_fig2_mgl_validation.R`](make_fig2_mgl_validation.R) | `fig2_mgl_validation.{png,pdf}` | `unit_outcomes_{uniform120,uniform120_malouf,uniform480,topfreq120}.csv`, `language_data_N.csv`, `cross_track_correlations.csv` |
| [`make_fig3_mgl_admixture_synthesis.R`](make_fig3_mgl_admixture_synthesis.R) | `fig3_mgl_admixture_synthesis.{png,pdf}` | `stage2_results_{V,N}.csv`, Track 1 `synthesis_checks_main.csv` |
| [`make_fig4_mgl_family_slopes.R`](make_fig4_mgl_family_slopes.R) | `fig4_mgl_family_slopes.{png,pdf}` | `language_data_{V,N}.csv`, `stage2_results_{V,N}.csv` |
| [`make_fig5_mgl_frequency_entrenchment.R`](make_fig5_mgl_frequency_entrenchment.R) | `fig5_mgl_frequency_entrenchment.{png,pdf}` | `entrenchment_results_{V,N}.csv`, `language_entrenchment_slopes_{V,N}.csv`, `lemma_freq_data_{V,N}.csv`, `entrench_{V,N}__all_lemmas__ent.rds` |

### Reproduction

```bash
FIG_DIR=analyses/paradigm/figures
Rscript "$FIG_DIR/make_fig1_mgl_map_and_complexity.R"
Rscript "$FIG_DIR/make_fig2_mgl_validation.R"
Rscript "$FIG_DIR/make_fig3_mgl_admixture_synthesis.R"
Rscript "$FIG_DIR/make_fig4_mgl_family_slopes.R"
Rscript "$FIG_DIR/make_fig5_mgl_frequency_entrenchment.R"
```

---

## 2. Figure Captions & Alt Text

### Figure 1 (`fig1_mgl_map_and_complexity.{png,pdf}`)
**Integrative paradigm entropy stays low across large synthetic paradigms in 56 GeLaTo languages.**  
**a**, Geographic distribution of the `56` GeLaTo-linked languages (`94` orthographic paradigm units: `45` Verb, `49` Noun; `12` Glottolog v5.3 families). Marker colour encodes mean individual ancestry entropy (`ent`, nats) and marker shape encodes POS coverage (`38` Verbs & Nouns, `11` Nouns only, `7` Verbs only). **b**, Distinct synthetic paradigm size (`n_cells_distinct`, log scale, up to `256` cells) versus cross-validated Vowel-Tier MGL tree conditional cross-entropy (`h_tree_bits` ± 1 CV SE) across all `94` orthographic units (`Verbs` pine-teal circles, `Nouns` ochre-bronze triangles). **c**, Empirical PCFP conditional-entropy drop across predictor contexts (`k = 1` single-cell baseline -> `k = 3` Product-of-Experts -> `Tree` Chu–Liu–Edmonds optimal branching); faint grey lines show individual units and thick trajectories show POS medians with IQR bars (overall median `2.17 -> 1.20 -> 0.58` bits; Verbs `0.54`, Nouns `0.72` bits), confirming Ackerman & Malouf's Low Conditional Entropy Conjecture.  
*Alt text:* Three-panel figure showing a world map of 56 languages colored by genetic admixture entropy, a scatter plot of paradigm cell count versus MGL tree entropy showing low entropy even at 100–256 cells, and a slopegraph showing conditional entropy dropping from 2.17 bits (1 predictor) to 0.58 bits (optimal tree).

### Figure 2 (`fig2_mgl_validation.{png,pdf}`)
**Multi-Locus Vowel-Tier MGL converges across harmony pre-abstraction, sample size, frequency stratum, orthography/IPA, and Track 1 Grambank irregularity.**  
**a**, Built-in Vowel-Tier MGL versus Malouf (2017) spectral vowel-harmony archiphoneme pre-abstraction across `33` Uralic, Turkic, and Mongolic-Khitan units (`r = 0.992` on `29` orthographic units, `r = 0.988` overall; dashed line marks 1:1 parity). **b**, Stability against `uniform120` baseline for `uniform480` (`n = 73` Tier-2 units, `r = 0.937`; `90.4%` lower or equal at `N_train = 480`) and `topfreq120` (`n = 74` units, `r = 0.980`; `70.3%` higher `H_tree` on frequent core lemmas per Cotterell et al. 2019 fn. 11). **c**, Paired orthographic versus Gold phonemic IPA (`12` Paralex/CELEX pairs), pinpointing orthographic neutralization in Russian unstressed vowel reduction (`0.995 -> 2.434` bits) and Estonian Q2/Q3 quantity (`0.594 -> 1.336` bits) alongside English opaque verbal spelling (`1.846 -> 0.732` bits). **d**, Cross-track construct validation of nominal `H_tree, N` against Track 1 Grambank nominal irregularity share (`irreg_noun_prop`, `n = 37`, `r = 0.502, p = 0.0016, rho = 0.494`) and latent `theta_irreg13` (`n = 35`, `r = 0.332, rho = 0.448`), with OLS fits and 95% CIs.  
*Alt text:* Four-panel validation plot showing tight 1:1 agreement between Vowel-Tier MGL and Malouf harmony abstraction, consistent shifts for 480-lemma and top-frequency splits, expected orthography-versus-IPA offsets for Russian, Estonian, and English, and positive correlation with Grambank nominal irregularity.

### Figure 3 (`fig3_mgl_admixture_synthesis.{png,pdf}`)
**Cross-linguistic admixture shows no Track-1-magnitude erosion of integrative paradigm predictability across Verbs and Nouns.**  
**a**, Posterior mean slopes (SD per SD of admixture) with thick `80%` and thin `95%` CrIs across `Individual entropy (ent)`, `Community heterogeneity (sqrt_het)`, and `Cross-family f3 (f3k_est_pos)` for Track 1 Grambank benchmarks (`MC_cross` Enumerative breadth `+0.44 [+0.22, +0.66]` and `irreg13` `-0.13 [-0.41, +0.14]`) and 12 Track 2 MGL specifications (`Verbs N <= 45`, `Nouns N <= 49`). Shaded grey band marks the `[-0.15, +0.15]` SD ROPE; vertical dotted line marks Track 1 Breadth. Under primary phylogenetic `ent`, `BF_01 = 2.07` (`P(beta < +0.44) = 0.980`) for Verbs and `BF_01 = 2.39` (`P(beta < +0.44) = 0.971`) for Nouns. **b**, Specification curves sorting all `73` Verb and `70` Noun `H_tree` models (`uniform120`, `uniform480`, `topfreq120`, and `neural`) by posterior mean; `61/73` Verb and `67/70` Noun `95%` CrIs include zero (`47/73` Verb and `64/70` Noun models have `P(beta < +0.44) >= 0.95`).  
*Alt text:* Two-panel synthesis figure contrasting Track 1's positive enumerative-breadth slope (+0.44 SD) with Track 2 MGL Verb and Noun admixture slopes centered near zero across 12 key specifications and all 143 specification-curve models.

### Figure 4 (`fig4_mgl_family_slopes.{png,pdf}`)
**Within-family admixture trajectories remain flat across multi-language lineages.**  
**a**, Comparison of Global (`Phylo + GP`), Mundlak within-family, Mundlak between-family, and leave-region-out (`Drop Turkic & Mongolic`, `Drop all Inner Asia`) `ent` posterior slopes (`80%` and `95%` CrIs; shaded `[-0.15, +0.15]` SD ROPE) for Verbs and Nouns. **b**, Observed MGL `H_tree` (± 1 CV SE, square-root scale) versus mean individual ancestry entropy (`ent`, nats) across the four lineages with `n >= 3` languages in both POS (`Indo-European 25 V / 23 N`, `Uralic 9 V / 9 N`, `Turkic 3 V / 7 N`, `Afro-Asiatic 3 V / 3 N`), overlaid with OLS trend lines (95% CI ribbons shown for `n >= 9`).  
*Alt text:* Two-panel plot showing Global, Mundlak within-family, and leave-region-out admixture slopes near zero alongside small-multiple scatterplots of ancestry entropy versus MGL tree entropy across Indo-European, Uralic, Turkic, and Afro-Asiatic.

### Figure 5 (`fig5_mgl_frequency_entrenchment.{png,pdf}`)
**Within-language token frequency entrenches irregular lemmas universally, with decisive null interaction against demographic admixture across 48,034 lemmas.**  
**a**, Language-specific within-language frequency slopes (`80%` and `95%` CrIs from `entrench_{V,N}__all_lemmas__ent.rds`) sorted by posterior mean across `44` Verb (`21,078` lemmas) and `46` Noun (`26,956` lemmas) languages, with population hyper-posterior at bottom (`Verbs: +0.093 [+0.051, +0.134] SD, BF_10 = 651, 40/44 > 0`; `Nouns: +0.030 [+0.013, +0.046] SD, P(>0) = 0.9994, 42/46 > 0`). **b**, Empirical within-language-centered `Delta H_tree` (bits, mean ± SE across 5 frequency quintiles) stratified by median language admixture (`Low` vs. `High`), overlaid with exact Bayesian hierarchical model predictions and 95% CrI ribbons showing parallel upward entrenchment slopes. **c**, Exact posterior densities and `80%`/`95%` CrIs for the cross-level `Admixture x Token Frequency` interaction `b_int`, sitting `100%` inside the `[-0.15, +0.15]` SD ROPE (`BF_01 = 17.8` on Verbs, `46.2` on Nouns).  
*Alt text:* Three-panel figure showing positive within-language frequency-entrenchment slopes across 44 Verb and 46 Noun languages, parallel empirical and modeled entropy gains across frequency quintiles for low- and high-admixture languages, and interaction posteriors concentrated inside the null ROPE.
