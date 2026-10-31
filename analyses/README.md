# Analyses Directory — PhD Study 3 (Inflectional Complexity & Genetic Admixture)

This directory contains the shared data/modeling infrastructure and the two empirical
modeling tracks of PhD Study 3:

1. **Track 1 (`grambank/`)**: GeLaTo genetic admixture $\to$ Grambank inflectional breadth,
   flexivity, and lexicalized irregularity ($N = 202$ languages, 335 populations, 2,943 individuals).
2. **Track 2 (`ldl/`)**: GeLaTo genetic admixture $\to$ Linear Discriminative Learning (LDL / Wug)
   paradigm cell predictability and production confidence across inflecting languages.

Both tracks share a single, language-set-agnostic library in [`common/`](common/) for
genetic admixture ($Q$-matrices $K = 12\dots30$ and $f_3$), audited speaker demography and
range area, Glottolog and time-calibrated Bouckaert v2.0.0 phylogenies, and Stage-1
hierarchical `brms` exposure models.

---

## Directory Layout

```text
analyses/
├── README.md                     <- Master navigation guide (this file)
├── common/                       <- Shared, language-set-agnostic modules for BOTH Grambank & LDL/Wug tracks
│   ├── __init__.py
│   ├── genetics.py               <- Loads ADMIXTURE Q (K=12..30) & f3 scores; computes individual & population
│   │                                admixture measures (ent, nonmax, pop_ent, het, sqrt_het, support, amount,
│   │                                graff_target, neigh_flow, f3_z, f3_est, f3k_z, f3k_est, f3k_est_pos) +
│   │                                400-resample bootstrap SEs and GeLaTo Ne for ANY (population, glottocode) table
│   ├── f3.R                      <- admixtools f3 computation (all triples and k-nearest triples)
│   ├── demography.py             <- Audited population_l1 & area_km2 lookup for ANY Glottocode (Grambank or LDL/Wug)
│   ├── demography_sources/       <- Bromham, Ethnologue multi-ISO, and Koplenig source tables
│   ├── phylogeny.py              <- Glottolog tree & Newick builder for any set of Glottocodes
│   ├── phylogeny.R               <- Builds Grafen (A_Grafen) & time-calibrated Bouckaert v2.0.0 (A_Bouckaert)
│   │                                phylogenetic correlation matrices for any set of Glottocodes
│   ├── stage1_exposure.R         <- Reusable Stage-1 hierarchical brms exposure fitter (individual/pop -> language
│   │                                posterior mean & SD), usable by both Grambank and LDL/Wug Stage-2 models
│   └── data/                     <- Shared precomputed tables (f3_scores.csv, f3_scores_k5.csv, bouckaert_v2.nwk,
│                                    and gelato_populations_summary.csv for all 558 GeLaTo ADMIXTURE populations)
│
├── grambank/                     <- Track 1: GeLaTo Admixture -> Grambank Morphology (N = 202)
│   ├── README.md                 <- Overview of the 3 Grambank modeling stages & how they connect
│   ├── prepare.py                <- Calls analyses.common to write data/ (language_data.csv, individuals.csv,
│   │                                populations.csv, item_observations.csv, items_metadata.csv, language_table.csv,
│   │                                phylo_tree.nwk, prep_summary.json, latent_prep_summary.json)
│   ├── data/                     <- Canonical prepared Grambank + admixture + demography + item tables
│   ├── baseline/                 <- Stage A: Two-stage Beta-Binomial (cat12, flex, flex26, irreg13) & Demography/Area
│   │                                (fit.R, run_demog.R, README.md, .gitignore, results/, fits/, stan_cache/)
│   ├── phylo_rates/              <- Stage B: Time-calibrated within-family RRW branch rates & PICs
│   │                                (prepare.R, contrasts_diagnostic.R, fit_rrw.R, gen_rrw.stan, morph_rrw.stan,
│   │                                 README.md, .gitignore, data/, results/, fits/)
│   └── latent_sem/               <- Stage C: Item-level 1PL/2PL IRT, Varying-Slope & 2-Factor SEM
│                                    (fit_latent.R, stan/item_1factor.stan, stan/item_2factor.stan,
│                                     README.md, .gitignore, results/, fits/)
│
├── ldl/                          <- Track 2: GeLaTo Admixture -> LDL / Wug Paradigm Predictability & Confidence
│   ├── README.md                 <- Guide showing how to link Wug/LDL paradigm outcomes to analyses/common/
│   │                                (genetics.py, demography.py, phylogeny.R, stage1_exposure.R)
│   └── ablation/                 <- LDL meaning-space, cell-vector, and decoder ablation
│
└── gelato_feasibility_2026_10_01/ <- Initial GeLaTo feasibility audit & cached upstream source archives
                                      (Zenodo 15263706 ADMIXTURE/F2, GeLaTo c625fdc, Bouckaert v2.0.0 tree)
```

---

## Quick Verification Commands

```bash
# 1. Regenerate all canonical Grambank & shared GeLaTo tables via analyses.common
.venv/bin/python analyses/grambank/prepare.py

# 2. Run the Python test suite (including tests/test_analyses_common.py)
.venv/bin/python -m pytest -m "not slow"

# 3. Validate Track 1 Stage A (Two-stage Beta-Binomial & Demography)
ADM_DRY=1 Rscript analyses/grambank/baseline/fit.R

# 4. Run Track 1 Stage B (Within-family PIC & branch-jump demographic audit)
Rscript analyses/grambank/phylo_rates/contrasts_diagnostic.R

# 5. Validate Track 1 Stage C (Item-level 1PL/2PL IRT, Varying-Slope & 2-Factor SEM; 43 cached fits)
LAT_DRY=1 Rscript analyses/grambank/latent_sem/fit_latent.R
```

---

## Archive Note

Six superseded October 2026 exploratory directories (`draft_admixture_model_2026_10_10`,
`gelato_review_2026_10_09`, `gelato_review_2026_10_16`, `gelato_review_2026_10_20`,
`pcfp_v1_2026_10_08`, `unimorph_gelato_estimate_2026_10_08`) are archived outside the repository
at `/Users/acraev/Work/study_3_archive_2026-10-08/analyses_archive_2026_10_31/` (see
[`../ARCHIVE_NOTE.md`](../ARCHIVE_NOTE.md)).
