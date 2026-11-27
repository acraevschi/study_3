# Analyses Directory — PhD Study 3 (Morphological Complexity & Genetic Admixture)

This directory contains the shared language-set-agnostic modeling library and the two primary empirical tracks of PhD Study 3, plus preserved earlier exploratory stages:

1. **Track 1 ([`grambank/`](grambank/README.md)):** GeLaTo population-genetic admixture (`f3_fixed`) $\to$ Grambank v1.0.3 five clean morphological domains (`v_breadth`, `n_breadth`, `v_irreg`, `n_irreg`, `np_agr`; `29` items, `5,411` observations, `151` non-morphological documentation control features) across **$N = 225$ linked languages** (`60` families, `364` `main` populations, `3,166` individuals; `189` languages under `strict` links), modeled via two-stage Beta-Binomial regressions (`fit_stage2_bb.R`) and focal item-level 2PL IRT-SEM & joint within-POS 2-factor SEMs (`latent_sem/fit_domain_sem.R`).
2. **Track 2 ([`paradigm/`](paradigm/README.md)):** GeLaTo population-genetic admixture (`f3_fixed`) $\to$ cross-validated morphological paradigm predictability (*i-complexity*, $H_{\text{tree}}$ in bits) across **`56` linked languages** (`106` paradigm units, `12` families), evaluated with a zero-language-tuned Multi-Locus Autosegmental Vowel-Tier MGL and 3-seed Hard-Monotonic Neural Transducer, paired with separate Verb (`V`) and Noun (`N`) Stage-2 Berkson regressions (`stage2/fit_stage2.R`) and two-level token-frequency entrenchment models (`stage2/fit_entrenchment.R`).
3. **Earlier Pilot & Preserved Stages:**
   - [`ldl/`](ldl/README.md): Earlier/exploratory `JudiLing.jl` LDL/Wug paradigm predictability pilot & ablation track (`ablation/`), superseded as primary Track 2 by `paradigm/`.
   - `grambank/{baseline/, phylo_rates/, latent_sem/{fit_latent,fit_irreg13_mb2pl,fit_synthesis}.R}`: Preserved earlier/exploratory Track 1 stages from the $N = 202$ (`38`-item `cat12`/`flex26`/`irreg13`, ADMIXTURE `.Q` / `f3k_est_pos`) phase (commit `e082562`).

Both primary tracks share a single, language-set-agnostic library in [`common/`](common/) for fixed-reference cross-family $f_3$/$f_4$, neighbor $F_{ST}$ regimes, discrete-$K$ ADMIXTURE diagnostics, audited speaker demography and range area, Glottolog v5.3 and time-calibrated Bouckaert v2.0.0 phylogenies, and Stage-1 hierarchical `brms` exposure shrinkage.

---

## Directory Layout

```text
analyses/
├── README.md                     <- Master navigation guide (this file)
├── common/                       <- Shared, language-set-agnostic library for Track 1 & Track 2
│   ├── __init__.py
│   ├── genetics.py               <- Loads f3_fixed, f4, FST neighbor regimes, discrete-K diagnostics,
│   │                                ADMIXTURE Q (K=12..30), bootstrap SEs & GeLaTo Ne for ANY link table
│   ├── f3.R                      <- admixtools f3/f4 computation (fixed clade-stratified f3_fixed, f4_est,
│   │                                plus legacy all-triple & k-nearest f3)
│   ├── demography.py             <- Audited population_l1 & area_km2 lookup for ANY Glottocode
│   ├── demography_sources/       <- Bromham, Ethnologue multi-ISO, and Koplenig source tables
│   ├── phylogeny.py              <- Glottolog v5.3 tree & Newick builder for any set of Glottocodes
│   ├── phylogeny.R               <- Grafen (A_Grafen) & time-calibrated Bouckaert v2.0.0 (stem/crown/kyr)
│   │                                phylogenetic correlation matrices
│   ├── stage1_exposure.R         <- Reusable Stage-1 hierarchical brms exposure fitter (pop/ind -> language
│   │                                posterior mean & SD; supports f3_fixed, f4_est, neigh_flow_out, etc.)
│   └── data/                     <- Shared canonical tables:
│                                    • analysis_links.csv (1,954 rows across main & 3 sensitivity sets)
│                                    • f3_fixed_scores.csv (517 populations: fixed-reference f3 & f4)
│                                    • fst_neighbor_regimes.csv (558 populations: FST neighbor regimes)
│                                    • discrete_k_diagnostics.csv (558 populations: K=12,20,30 diagnostics)
│                                    • bouckaert_v2.nwk (time-calibrated global phylogeny)
│                                    • gelato_populations_summary.csv (558 GeLaTo populations)
│                                    • f3_scores.csv & f3_scores_k5.csv (335 rows; legacy N = 202 tables)
│
├── grambank/                     <- Track 1: Admixture -> Grambank 5 Clean Domains (N = 225)
│   ├── README.md                 <- Track 1 overview, scientific rationale, headline results & reproduction
│   ├── prepare.py                <- Standalone 5-domain data builder via analyses.common & pinned CLDFs
│   ├── data/                     <- Canonical N = 225 / 29-item / 5,411-obs tables & phylo_tree.nwk
│   ├── fit_stage2_bb.R           <- Primary two-block Beta-Binomial runner (108 models across 5 domains)
│   ├── results/                  <- Canonical Track 1 Beta-Binomial outputs (stage2_bb_results.csv, etc.)
│   ├── latent_sem/               <- Focal 2PL IRT-SEM & 2-factor within-POS SEM (fit_domain_sem.R,
│   │                                stan/item_domain_sem.stan, stan/item_pos_2factor_sem.stan,
│   │                                results_domain/) + preserved N = 202 scripts (fit_latent.R, etc.)
│   ├── figures/                  <- Publication figure suite (fig1..6_grambank_*.{png,pdf}, run_qc.R)
│   ├── baseline/                 <- Preserved earlier N = 202 Beta-Binomial & demography stage (e082562)
│   └── phylo_rates/              <- Preserved earlier N = 202 within-family RRW & PIC study (e082562)
│
├── paradigm/                     <- Track 2: Admixture -> Paradigm Predictability (56 langs, 106 units)
│   ├── README.md                 <- Track 2 overview, reproduction commands & headline summary tables
│   ├── ingestion/                <- Paradigm fetching, orthographic/NFC hygiene, cognate grouping, Malouf & splits
│   ├── frequencies/              <- UD v2.15 / Leipzig / wordfreq fetching, EM homograph matching & topfreq120
│   ├── mgl/                      <- Multi-Locus Autosegmental Vowel-Tier MGL evaluator
│   ├── neural/                   <- 3-seed Hard-Monotonic Neural Transducer validator
│   ├── stage2/                   <- Separate V & N prepare_stage2.py, fit_stage2.R, fit_entrenchment.R
│   ├── data/                     <- SHA-256 manifests, unit_summary.csv, cell_maps/
│   ├── results/                  <- Tracked MGL, Neural, Stage-2 & entrenchment summary CSVs
│   └── figures/                  <- Publication figure suite (fig1..6_mgl_*.{png,pdf}, run_qc.R)
│
├── ldl/                          <- Earlier/exploratory JudiLing.jl LDL/Wug pilot & ablation track
│   ├── README.md                 <- Guide to linking LDL/Wug outcomes to analyses/common/
│   └── ablation/                 <- LDL meaning-space, cell-vector, and decoder ablation
│
└── gelato_feasibility_2026_10_01/ <- Initial GeLaTo feasibility audit & cached third-party archives
                                      (Zenodo 15263706 ADMIXTURE/FST, GeLaTo c625fdc, Bouckaert v2.0.0 tree)
```

---

## Quick Verification Commands

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# 1. Run the automated Python regression suite (common, Track 1, Track 2, morph_ldl)
.venv/bin/python -m pytest -m "not slow"

# 2. Track 1 (Grambank 5 domains): regenerate canonical tables & validate BB + 2PL IRT-SEM Stan blocks
.venv/bin/python analyses/grambank/prepare.py
BB_DRY=1 Rscript analyses/grambank/fit_stage2_bb.R
SEM_DRY=1 Rscript analyses/grambank/latent_sem/fit_domain_sem.R

# 3. Track 2 (Paradigm predictability): regenerate Stage-2 tables & validate Stage-2 + entrenchment Stan blocks
.venv/bin/python analyses/paradigm/stage2/prepare_stage2.py
PARA_DRY=1 Rscript analyses/paradigm/stage2/fit_stage2.R
ENT_DRY=1 Rscript analyses/paradigm/stage2/fit_entrenchment.R
```

*Note: Preserved earlier $N = 202$ Track 1 scripts (`baseline/fit.R`, `phylo_rates/prepare.R`, `latent_sem/{fit_latent,fit_irreg13_mb2pl,fit_synthesis}.R`) targeted the pre-v2 `38`-item (`cat12`/`flex26`/`irreg13`) schema from commit `e082562` and are not part of the active verification commands above; see [`grambank/README.md`](grambank/README.md) for details.*
