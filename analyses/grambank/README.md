# Track 1: GeLaTo Genetic Admixture $\to$ Grambank Morphology ($N = 202$)

This directory implements **Track 1** of PhD Study 3, testing whether population-genetic
admixture in the GeLaTo / Human Origins panel predicts morphological simplification across
the 202 Grambank-linked languages (`outputs/pcfp_v2_links_2026_10_20/typology/`).

All data preparation is unified in [`prepare.py`](prepare.py), which calls the shared
modules in [`../common/`](../common/) (`genetics.py`, `demography.py`, `phylogeny.py`) and
writes canonical tables into [`data/`](data/).

---

## The Three Modeling Stages

| Stage | Directory | Models & Purpose |Detailed Report |
|---|---|---|---|
| **Stage A: Baseline & Demography** | [`baseline/`](baseline/) | Two-stage Bayesian Beta-Binomial models (`brms`) across 4 outcomes (`cat12`, `flex`, `flex26`, `irreg13`), 16 admixture measures, 4 link-sensitivity sets, 4 control sets (`N`, `P`, `S`, `F`), and audited L1 speaker population & range area extensions ($N = 192$). Includes 590 cached `.rds` fits. | [`baseline/README.md`](baseline/README.md) |
| **Stage B: Within-Family Phylo Rates & PICs** | [`phylo_rates/`](phylo_rates/) | Time-calibrated within-family Bouckaert v2.0.0 trees ($F = 10$ families with $\ge 5$ languages; $F = 13$ with $\ge 3$), Phylogenetic Independent Contrasts (PICs), terminal branch-jump $N_e$ bottleneck audits, and custom 2-stage Relaxed Random Walk (RRW) Stan models (`gen_rrw.stan`, `morph_rrw.stan`). Includes 162 cached `.rds` fits. | [`phylo_rates/README.md`](phylo_rates/README.md) |
| **Stage C: Item-Level Latent IRT, 2-Factor SEM & Unifying Synthesis** | [`latent_sem/`](latent_sem/) | Item-level logistic models (`stan/item_1factor*.stan`, `stan/item_2factor*.stan`) over the 6,821 binary language $\times$ item cells: 1PL vs. 2PL IRT/MIMIC, hierarchical varying admixture slopes across items/domains (testing agreement vs. inherent inflection), joint 2-factor SEMs of Enumerative Breadth ($\theta_{1l}$, `cat12`) + Irregularity/Flexivity ($\theta_{2l}$, `irreg13` / `flex26`), and 333 unifying synthesis checks (`fit_synthesis.R`: family-varying slopes `(1 + x \| family)` & Mundlak within/between decomposition, Inner-Asia sensitivity, time-calibrated Bouckaert v2.0.0 covariance `P_b_stem`/`P_b_crown`/`P_b_kyr`, speaker population `z_pop` & `x * z_pop`, and Spatial GP `F`). Includes 351 cached `.rds` fits (`42` core `P` + `23` `F` + `214` `syn_*` SEM fits + `72` `bb_syn__*` Beta-Binomial fits). | [`latent_sem/README.md`](latent_sem/README.md) |

---

## Canonical Data Tables (`data/`)

Running `.venv/bin/python analyses/grambank/prepare.py` generates:

- `language_data.csv`: 202 main-coverage Grambank languages with outcome counts (`n_present`/`n_coded`, `flex_*`, `flex26_*`, `irreg13_*`), Glottolog coordinates, link-set flags, and audited demographic covariates (`population_l1`, `area_km2`, `log_pop`, `log_area`).
- `populations.csv`: 335 GeLaTo populations with K-averaged ADMIXTURE statistics (`pop_ent`, `het`, `sqrt_het`, `support`, `amount`) + 400-resample bootstrap SEs, `graff_target`, `neigh_flow`, and `f3_z`/`f3_est`.
- `individuals.csv`: 2,943 genotyped individuals with `ent`, `nonmax`, `ent_K12`, `ent_K30`.
- `phylo_tree.nwk`: Glottolog 5.3 topology Newick tree across the 202 languages.
- `items_metadata.csv`: 38 binary Grambank items (12 `cat12` categories + 26 non-overlapping `flex26` features, including the 13 `irreg13` lexicalized irregularity/suppletion items).
- `item_observations.csv`: 6,821 observed binary language $\times$ item responses (`0`/`1`).
- `language_table.csv`: Language table aligned with `grambank_entry` for `latent_sem/`.
- `prep_summary.json` & `latent_prep_summary.json`: Dataset, correlation, demographic, and item-level summaries.

---

## Reproducing Each Stage

```bash
# 0. Prepare all canonical Grambank tables
.venv/bin/python analyses/grambank/prepare.py

# Stage A: Two-stage Beta-Binomial & Demography
ADM_DRY=1 Rscript analyses/grambank/baseline/fit.R
Rscript analyses/grambank/baseline/fit.R main
Rscript analyses/grambank/baseline/run_demog.R N,P cat12,flex,irreg13 ent,het,f3k_est both

# Stage B: Within-Family Phylogenetic Branch Rates & PICs
Rscript analyses/grambank/phylo_rates/prepare.R
Rscript analyses/grambank/phylo_rates/contrasts_diagnostic.R
Rscript analyses/grambank/phylo_rates/fit_rrw.R

# Stage C: Item-Level 1PL/2PL IRT, Varying-Slope, 2-Factor SEM & Unifying Synthesis
LAT_DRY=1 Rscript analyses/grambank/latent_sem/fit_latent.R
LAT_CTRLS=P Rscript analyses/grambank/latent_sem/fit_latent.R main
SYN_DRY=1 Rscript analyses/grambank/latent_sem/fit_synthesis.R
Rscript analyses/grambank/latent_sem/fit_synthesis.R
```
