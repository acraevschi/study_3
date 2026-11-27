# Track 1: GeLaTo Genetic Admixture $\to$ Grambank Morphology ($N = 225$, 5 Clean Domains)

This directory implements **Track 1** of PhD Study 3, testing whether population-genetic admixture in the GeLaTo / Human Origins panel predicts typological morphological complexity across **$N = 225$ GeLaTo-linked Grambank v1.0.3 languages** (`60` Glottolog v5.3 families, `364` `main` populations, `3,166` genotyped individuals; `189` languages / `300` populations / `2,629` individuals under `strict` links).

All canonical data preparation is unified in [`prepare.py`](prepare.py), which derives Grambank metadata directly from tracked [`morph_ldl/typology/`](../../morph_ldl/typology/) + pinned CLDF repositories (`external/grambank`, `external/glottolog-cldf`) together with [`../common/`](../common/) (`genetics.py`, `demography.py`, `phylogeny.py`) and writes canonical tables into [`data/`](data/).

---

## 1. Scientific Rationale: From Pooled Sums & Discrete-$K$ `.Q` to 5 Clean Domains & `f3_fixed`

The primary Track 1 architecture resolves two methodological confounds identified during the earlier exploratory $N = 202$ phase (`e082562`, preserved in `baseline/`, `phylo_rates/`, and `latent_sem/{fit_latent,fit_irreg13_mb2pl,fit_synthesis}.R`):

1. **Decomposing pooled complexity sums (`cat12` / `irreg13`) into 5 clean, domain-conditioned domains (`29` items, `5,411` observations):**
   Early pooled category counts (`cat12`) combined *inherent* verbal and nominal inflectional categories (which trend weakly positive with admixture) with *contextual* NP modifier agreement (`gender_agreement`, `number_agreement`, which trends weakly negative), masking opposing signals inside a single sum score. Similarly, unconditioned irregularity counts (`irreg13`) coded isolating languages that lack verbal/nominal morphology as `0` ("regular"), conflating the absence of a morphological domain with paradigmatic regularity. The harmonized schema splits `29` clean binary Grambank features into five theoretically distinct domains—**Verb breadth** (`v_breadth`, $k=9$), **Noun breadth** (`n_breadth`, $k=5$), **Verb irregularity** (`v_irreg`, $k=5$, prerequisite NA-masked to languages with verbal inflection), **Noun irregularity** (`n_irreg`, $k=3$, prerequisite NA-masked to languages with nominal number/gender), and **Contextual NP agreement** (`np_agr`, $k=7$)—and pairs every model with a non-morphological Grambank documentation-coverage covariate (`151` features).
2. **Replacing discrete-$K$ ADMIXTURE `.Q` scalars (`ent`, `sqrt_het`) with fixed-reference cross-family `f3_fixed`:**
   Reference-free ADMIXTURE entropy (`ent`) and within-population spread (`sqrt_het`) depend on panel composition, choice of $K$, and isolation-by-distance clines rather than specifically indexing cross-family gene flow. Fixed-reference `f3_fixed` directly tests three-population allele sharing $f_3(A; B^*, C^*)$ between each target population $A$, a fixed clade-stratified same-family source $B^*$, and a fixed out-of-family source $C^*$ (using `n_A >= 3` primary and `n_A >= 5` sensitivity cutoffs, cross-checked against `f3_all_fixed`, `f3_fixed_iso`, `f3_fixed_maxp`, `f3_fixed_raw`, `f4_est`, `neigh_flow_out`, and $F_{ST}$ neighbor regimes).

---

## 2. Primary 5-Domain `f3_fixed` Pipeline

| Component | Script(s) | Outputs | Scope |
|---|---|---|---|
| **Data preparation** | [`prepare.py`](prepare.py) | [`data/`](data/) (`language_data.csv`, `language_table.csv`, `populations.csv`, `individuals.csv`, `items_metadata.csv`, `item_observations.csv`, `phylo_tree.nwk`, `prep_summary.json`, `latent_prep_summary.json`) | `225` languages (`60` families), `364` `main` pops (`3,166` inds), `29` clean items (`5,411` obs), `151` non-morph doc features, audited L1 pop (`218`) & measured area (`215`) |
| **Focal 2PL IRT-SEM & Within-POS 2-Factor SEM** | [`latent_sem/fit_domain_sem.R`](latent_sem/fit_domain_sem.R) (`stan/item_domain_sem.stan`, `stan/item_pos_2factor_sem.stan`) | [`latent_sem/results_domain/`](latent_sem/results_domain/) (`sem_1factor_results.csv`, `sem_2factor_pos_results.csv`, `sem_item_dif.csv`, `sem_latent_eap.csv`, `exposure_sem_f3_fixed.csv`, `maximal_anatomy_summary.csv`) | `60` 1-factor models (`55` `MA_2pl` Block 1 + Block 2 + `5` `MB_2pl` item-DIF) + `4` joint within-POS 2-factor SEMs (`Verb_2fac`, `Noun_2fac` at $n_A \ge 3, 5$) |
| **Two-Stage Beta-Binomial Suite** | [`fit_stage2_bb.R`](fit_stage2_bb.R) | [`results/`](results/) (`stage2_bb_results.csv`, `exposure_bb_all.csv`, `stage1_bb_diagnostics.csv`) | `108` models across all 5 domains (Block 1 locked complete-case maximal + LOO ablations + Mundlak; Block 2 sample, Bouckaert phylogeny, raw unmasked zeros, genetics SI benchmarks, classical `me`, and 20-draw Berkson check) |
| **Publication Figures & QC** | [`figures/make_fig1..6*.R`](figures/README.md), [`figures/run_qc.R`](figures/run_qc.R) | [`figures/fig1..6_grambank_*.{png,pdf}`](figures/README.md) | Six publication figures (`6.5 in`, `300 DPI` PNG + vector PDF) with standalone CSV fallback (`FORCE_CSV_FALLBACK=1`) |

### Five Clean Morphological Domains (`data/items_metadata.csv`)

| Domain | Items ($k$) | Coding Floor ($\ge 60\%$) | Eligible (`main`) | With `f3_fixed` ($n_A \ge 3$) | Maximal Complete-Case ($N$) | Features |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `v_breadth` | `9` | `>= 6` | `213` | `176` | `167` | Present/past/future tense (`GB082`–`084`), aspect (`GB086`), mood (`GB312`), S/A & P indexing (`GB089`–`094`), bound negation (`GB107`), bound polar Q (`GB286`) |
| `n_breadth` | `5` | `>= 3` | `217` | `177` | `168` | Productive plural (`GB044`), dual (`GB043`), core & oblique noun case (`GB070`, `GB072`), possessed-noun affix (`GB431`/`GB433`) |
| `v_irreg` | `5` | `>= 3` | `172` | `140` | `135` | Prerequisite NA-masked conjugation classes (`GB111`), TAM suppletion (`GB110`), person stem alternation (`GB099`), participant-number suppletion (`GB109`), lexical verb suppletion (`GB300`/`GB402`/`GB403`) |
| `n_irreg` | `3` | `>= 2` | `131` | `113` | `108` | Prerequisite NA-masked noun-number allomorphy (`GB039`), noun-number suppletion (`GB041`), unpredictable gender/noun class (`GB321`) |
| `np_agr` | `7` | `>= 5` | `198` | `167` | `158` | Adnominal gender agreement (`GB170`–`172`, `GB198`) and adnominal number agreement (`GB184`–`186`) |

---

## 3. Headline Results (Block-1 Maximal Model, `main`, `f3_fixed`, $n_A \ge 3$)

Standardized posterior slopes per 1 SD of cross-family `f3_fixed` under the locked complete-case maximal specification (Phylogeny + 3D spherical spatial GP + $\log$ speaker population + $\log$ geographic range area + non-morphological documentation coverage, `Normal(0, 0.5)` slope prior, ROPE `[-0.15, +0.15]`):

### 3.1 Single-Domain 2PL IRT-SEM (`MA_2pl`) & Beta-Binomial Slopes

| Domain | $N$ | Focal 2PL IRT-SEM $\beta$ [95% CrI] | $\text{BF}_{01}$ | ROPE | Beta-Binomial $\beta$ [95% CrI] | BB $\text{BF}_{01}$ | BB ROPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v_breadth`** ($k=9$) | `167` | `+0.087 [-0.183, +0.349]` | `3.01` | `0.640` | `+0.044 [-0.141, +0.231]` | `4.73` | `0.849` |
| **`n_breadth`** ($k=5$) | `168` | `+0.260 [-0.091, +0.636]` | `0.99` | `0.266` | `+0.140 [-0.061, +0.347]` | `1.95` | `0.541` |
| **`v_irreg`** ($k=5$) | `135` | `-0.247 [-0.716, +0.215]` | `1.23` | `0.308` | `-0.271 [-0.674, +0.113]` | `0.99` | `0.257` |
| **`n_irreg`** ($k=3$) | `108` | `-0.229 [-0.734, +0.263]` | `1.31` | `0.310` | `-0.194 [-0.609, +0.201]` | `1.56` | `0.378` |
| **`np_agr`** ($k=7$) | `158` | `-0.242 [-0.681, +0.198]` | `1.24` | `0.279` | `-0.167 [-0.418, +0.087]` | `1.65` | `0.441` |

*Across all `163` fitted Track 1 models (`55` `MA_2pl` + `108` Beta-Binomial), `0/163` 95% credible intervals exclude zero, while directional point estimates consistently separate weakly positive enumerative breadth from weakly negative irregularity and contextual NP agreement.*

### 3.2 Joint Within-POS 2-Factor 2PL SEMs (`sem_2factor_pos_results.csv`, $n_A \ge 3$)

| Model | $N$ | Breadth $\beta_{\text{breadth}}$ [95% CrI] | Irregularity $\beta_{\text{irreg}}$ [95% CrI] | Contrast $\Delta\beta = \beta_{\text{br}} - \beta_{\text{ir}}$ [95% CrI] | $P(\Delta\beta > 0)$ | Residual $\rho_{\text{latent}}$ [95% CrI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Verb_2fac`** (`v_breadth` + `v_irreg`) | `168` | `+0.087 [-0.173, +0.353]` | `-0.237 [-0.724, +0.234]` | `+0.325 [-0.206, +0.881]` | `0.882` | `-0.004 [-0.495, +0.522]` |
| **`Noun_2fac`** (`n_breadth` + `n_irreg`) | `169` | `+0.242 [-0.112, +0.608]` | `-0.236 [-0.778, +0.303]` | `+0.479 [-0.165, +1.140]` | `0.919` | `-0.269 [-0.841, +0.415]` |

---

## 4. Reproducing the Primary Track 1 Pipeline

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# 0. Regenerate canonical N = 225 / 5-domain tables in analyses/grambank/data/
.venv/bin/python analyses/grambank/prepare.py

# 1. Dry-run validation (compiles/validates all Stan code & data blocks; zero MCMC sampling)
BB_DRY=1 Rscript analyses/grambank/fit_stage2_bb.R
SEM_DRY=1 Rscript analyses/grambank/latent_sem/fit_domain_sem.R

# 2. Full MCMC runs (scale BB_CORES / SEM_CORES or pass --cores=N as appropriate for your machine)
BB_CORES=4 Rscript analyses/grambank/fit_stage2_bb.R
SEM_CORES=4 Rscript analyses/grambank/latent_sem/fit_domain_sem.R

# 3. Render publication figures (Figs 1–6; works standalone from tracked CSVs via FORCE_CSV_FALLBACK=1)
for s in analyses/grambank/figures/make_fig*.R; do
  FORCE_CSV_FALLBACK=1 Rscript "$s"
done
```

---

## 5. Preserved Earlier / Exploratory Stages (`N = 202`, `38`-Item Phase, Commit `e082562`)

The following subdirectories and scripts are preserved verbatim with their tracked result CSVs as historical snapshots of the earlier $N = 202$ (`38`-item `cat12`/`flex26`/`irreg13`, ADMIXTURE `ent`/`sqrt_het`/`f3k_est_pos`) exploratory phase:

| Preserved Stage | Location | Description |
| :--- | :--- | :--- |
| **Earlier Baseline & Demography** | [`baseline/`](baseline/README.md) (`fit.R`, `run_demog.R`) | Two-stage Beta-Binomial models across `cat12`, `flex`, `flex26`, `irreg13` and 18 exploratory admixture measures on $N = 202$ (`192` with demography). |
| **Within-Family Phylo Rates & PICs** | [`phylo_rates/`](phylo_rates/README.md) (`prepare.R`, `contrasts_diagnostic.R`, `fit_rrw.R`) | Time-calibrated within-family Bouckaert v2.0.0 Relaxed Random Walk (`cov_rrw`) and PIC methodological test ($F = 10$ `min5` / $13$ `min3` families), demonstrating that $F_{ST}$ branch jumps confound isolation bottlenecks with contact and that signed shifts reproduce cross-sectional models. |
| **Earlier `38`-Item IRT & Synthesis** | [`latent_sem/`](latent_sem/README.md) (`fit_latent.R`, `fit_irreg13_mb2pl.R`, `fit_synthesis.R`) | Exploratory 1PL/2PL IRT, varying-slope `cat12`, 2-factor `cat12_irreg13`/`cat12_flex26`, and `309`-model synthesis checks on the $N = 202$ / `38`-item schema. |

Because [`data/`](data/) now stores the canonical $N = 225$ five-domain tables, re-running those archived scripts requires checking out the `e082562` data snapshot.
