# PhD Study 3: Morphological Complexity & Population-Genetic Admixture

Reproducible cross-linguistic repository testing whether population-genetic admixture (GeLaTo / Human Origins panel) predicts morphological simplification across two complementary empirical tracks linked via language-level Glottocodes and shared genetic, demographic, and phylogenetic infrastructure (`analyses/common/`):

1. **Track 1 — Typological Inflectional Complexity ([`analyses/grambank/`](analyses/grambank/README.md)):**
   Harmonized five-domain typological architecture built from **Grambank v1.0.3** and **GeLaTo** across **$N = 225$ linked languages** (`60` Glottolog v5.3 families, `364` `main` populations, `3,166` genotyped individuals; `189` languages / `300` populations / `2,629` individuals under `strict` links), separating `29` clean binary morphological features (`5,411` observations, plus `151` non-morphological documentation-coverage control features) into **Verb breadth** (`v_breadth`, $k=9$), **Noun breadth** (`n_breadth`, $k=5$), domain-conditioned **Verb irregularity** (`v_irreg`, $k=5$), domain-conditioned **Noun irregularity** (`n_irreg`, $k=3$), and **Contextual NP agreement** (`np_agr`, $k=7$). Evaluates fixed-reference cross-family admixture (`f3_fixed`, $n_A \ge 3$ primary and $n_A \ge 5$ sensitivity, plus Block-2 genetic, phylogenetic, demographic, and sample robustness checks) via focal item-level **2PL IRT-SEMs & joint within-POS 2-factor SEMs** (`latent_sem/fit_domain_sem.R`, `64` models), **two-stage Berkson Beta-Binomial regressions** (`fit_stage2_bb.R`, `108` models), and a six-figure publication suite (`figures/make_fig1..6*.R`). Earlier exploratory $N = 202$ (`38`-item `cat12`/`flex26`/`irreg13`, ADMIXTURE `.Q` / `f3k_est_pos`) stages are preserved in `analyses/grambank/{baseline/, phylo_rates/, latent_sem/{fit_latent,fit_irreg13_mb2pl,fit_synthesis}.R}` for provenance.
2. **Track 2 — Morphological Paradigm Predictability / *i-Complexity* ([`analyses/paradigm/`](analyses/paradigm/README.md)):**
   Cross-validated conditional cross-entropy ($H_{\text{tree}}$ in bits) of predicting inflected wordforms across **`56` GeLaTo-linked languages** (`106` paradigm units: `94` orthographic + `12` Gold phonemic IPA; `12` Glottolog v5.3 families), evaluated with a zero-language-tuned **Multi-Locus Autosegmental Vowel-Tier Minimal Generalization Learner (MGL)** across four regimes (`uniform120`, `uniform120_malouf`, `topfreq120`, `uniform480`) and validated against a **3-seed Hard-Monotonic Neural Transducer** (`12` typologically diverse units). Links paradigm predictability to `f3_fixed` via strictly separated Verb (`V`) and Noun (`N`) Stage-2 precision-weighted phylogenetic + 3D spherical GP Berkson regressions (`stage2/fit_stage2.R`), two-level within-language token-frequency entrenchment & `Admixture × Frequency` interaction models across `45,844` lemmas (`stage2/fit_entrenchment.R`), and a six-figure publication suite (`figures/make_fig1..6*.R`). Earlier `JudiLing.jl` Linear Discriminative Learning (LDL / Wug) pilot and ablation infrastructure is preserved in `morph_ldl/`, `julia/`, and [`analyses/ldl/`](analyses/ldl/README.md).

---

## Documentation Index

| Document | Scope |
|---|---|
| [`analyses/README.md`](analyses/README.md) | Master navigation guide to `analyses/common/`, Track 1 (`grambank/`), Track 2 (`paradigm/`), and preserved `ldl/` pilot |
| [`analyses/grambank/README.md`](analyses/grambank/README.md) | Track 1 (`grambank/`): 5-domain `f3_fixed` architecture, headline 2PL IRT-SEM & Beta-Binomial results, and preserved $N = 202$ stages |
| [`analyses/grambank/latent_sem/README.md`](analyses/grambank/latent_sem/README.md) | Track 1 focal 2PL IRT-SEM & 2-factor within-POS SEM (`fit_domain_sem.R`) + archived $N = 202$ item-level SEM stages |
| [`analyses/grambank/figures/README.md`](analyses/grambank/figures/README.md) | Track 1 publication figures (`fig1..6`), captions, alt text, and standalone CSV fallback |
| [`analyses/paradigm/README.md`](analyses/paradigm/README.md) | Track 2 (`paradigm/`): MGL & Neural *i-complexity* pipelines, reproduction commands, and headline Stage-2 & entrenchment tables |
| [`analyses/paradigm/stage2/README.md`](analyses/paradigm/stage2/README.md) | Track 2 Stage-2 Berkson `f3_fixed` regressions, equivalence (`BF_01`/ROPE), sampler notes, and frequency-entrenchment models |
| [`analyses/paradigm/figures/README.md`](analyses/paradigm/figures/README.md) | Track 2 publication figures (`fig1..6`), captions, alt text, and standalone CSV fallback |
| [`analyses/ldl/README.md`](analyses/ldl/README.md) | Earlier/exploratory `JudiLing.jl` LDL/Wug pilot & ablation (`analyses/ldl/ablation/README.md`) |
| [`docs/TYPOLOGY.md`](docs/TYPOLOGY.md) | Upstream GeLaTo linkage rules, decision sheet (`morph_ldl/data/gelato_link_review.csv`), and Grambank curation |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | Upstream `morph_ldl` / `pcfp_v2` pilot specification (estimand, CV, metrics, uncertainty) |
| [`docs/PIPELINE_OVERVIEW.md`](docs/PIPELINE_OVERVIEW.md) | Upstream `pcfp_v2` pilot stage map |
| [`docs/REPORT_pcfp_v2.md`](docs/REPORT_pcfp_v2.md) | Upstream `pcfp_v2` repeated-random LDL pilot report |
| [`docs/CONTRACT.md`](docs/CONTRACT.md), [`docs/DATA.md`](docs/DATA.md), [`docs/SELECTION.md`](docs/SELECTION.md), [`docs/LDL_PROTOCOL.md`](docs/LDL_PROTOCOL.md) | Upstream `morph_ldl` schemas, MGN adapters, active-selection audit, and JudiLing protocol |

---

## Setup & Quick Verification

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# 1. Clone pinned external repositories (Grambank v1.0.3, Glottolog CLDF v5.3, languages-of-the-world, etc.)
bash scripts/fetch_external.sh

# 2. Create Python environment (Python >= 3.12) and install pinned dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install ./external/languages-of-the-world

# 3. Run the automated regression test suite (common, Track 1, Track 2, morph_ldl)
.venv/bin/python -m pytest -m "not slow"

# 4. Validate Track 1 (Grambank 5-domain data prep, Beta-Binomial & 2PL IRT-SEM dry runs)
.venv/bin/python analyses/grambank/prepare.py
BB_DRY=1 Rscript analyses/grambank/fit_stage2_bb.R
SEM_DRY=1 Rscript analyses/grambank/latent_sem/fit_domain_sem.R

# 5. Validate Track 2 (Stage-2 data prep, Berkson regression & frequency-entrenchment dry runs)
.venv/bin/python analyses/paradigm/stage2/prepare_stage2.py
PARA_DRY=1 Rscript analyses/paradigm/stage2/fit_stage2.R
ENT_DRY=1 Rscript analyses/paradigm/stage2/fit_entrenchment.R

# 6. Reproduce publication figures from tracked summary CSVs without local .rds fits
FORCE_CSV_FALLBACK=1 Rscript analyses/grambank/figures/make_fig6_grambank_maximal_effects.R
FORCE_CSV_FALLBACK=1 Rscript analyses/paradigm/figures/make_fig5_mgl_frequency_entrenchment.R
FORCE_CSV_FALLBACK=1 Rscript analyses/paradigm/figures/make_fig6_mgl_maximal_effects.R
```

*Optional — upstream `morph_ldl` / `pcfp_v2` pilot (requires Julia 1.12, JudiLing 1.0.1):*

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
.venv/bin/python -m morph_ldl.cli <stage> --config configs/pcfp_v2_links.yaml
```

---

## Repository Layout

```text
study_3/
├── README.md                                 <- Project overview, setup, repository layout, data provenance
├── requirements.lock                         <- Exact Python package versions
├── pytest.ini                                <- Test settings (`slow` marker for real-Julia tests)
│
├── analyses/                                 <- Primary empirical tracks & shared library (see analyses/README.md)
│   ├── common/                               <- Shared genetics (f3_fixed, f4, FST regimes, ADMIXTURE Q),
│   │   │                                        audited demography, Glottolog/Bouckaert phylogeny & Stage-1 exposure
│   │   └── data/                             <- Canonical analysis_links.csv, f3_fixed_scores.csv,
│   │                                            fst_neighbor_regimes.csv, discrete_k_diagnostics.csv, bouckaert_v2.nwk
│   ├── grambank/                             <- Track 1: Admixture -> Grambank 5 Clean Domains (N = 225)
│   │   ├── prepare.py / data/                <- Standalone 5-domain preparation & canonical CSV/NWK tables
│   │   ├── fit_stage2_bb.R / results/        <- Primary Beta-Binomial two-block runner (108 models) & summaries
│   │   ├── latent_sem/                       <- Focal 2PL IRT-SEM & 2-factor SEM (fit_domain_sem.R, stan/, results_domain/)
│   │   │                                        + archived N = 202 scripts (fit_latent.R, fit_synthesis.R)
│   │   ├── figures/                          <- Track 1 publication figures (fig1..6) & visual QC
│   │   ├── baseline/                         <- Preserved earlier N = 202 Beta-Binomial & demography stage
│   │   └── phylo_rates/                      <- Preserved earlier N = 202 within-family RRW & PIC study
│   ├── paradigm/                             <- Track 2: Admixture -> Paradigm Predictability (56 langs, 106 units)
│   │   ├── ingestion/                        <- Programmatic paradigm acquisition, cognate grouping, Malouf & splits
│   │   ├── frequencies/                      <- Open-corpus token frequency acquisition & EM homograph matching
│   │   ├── mgl/                              <- Multi-Locus Autosegmental Vowel-Tier MGL evaluator
│   │   ├── neural/                           <- 3-seed Hard-Monotonic Neural Transducer validator
│   │   ├── stage2/                           <- Separate V & N Berkson Stage-2 f3_fixed & entrenchment pipelines
│   │   ├── data/ / results/                  <- Provenance manifests, unit outcomes, Stage-2 & entrenchment CSVs
│   │   └── figures/                          <- Track 2 publication figures (fig1..6) & visual QC
│   ├── ldl/                                  <- Earlier JudiLing.jl LDL/Wug pilot & ablation (ablation/)
│   └── gelato_feasibility_2026_10_01/        <- GeLaTo feasibility audit & cached third-party genetic/tree archives
│
├── morph_ldl/                                <- Upstream Python package for GeLaTo linkage rules, typology & pcfp_v2 pilot
│   ├── data/                                 <- GeLaTo crosswalk, confirmed gelato_link_review.csv, MGN adapters
│   ├── typology/                             <- Grambank feature-set & Glottolog linkage stage
│   └── cv/ / ldl/ / selection/               <- pcfp_v2 CV splits, JudiLing runner, active-selection audit
├── julia/                                    <- JudiLing 1.0.1 runner (upstream pcfp_v2 pilot)
├── configs/                                  <- Upstream pcfp_v1 / pcfp_v2 / pcfp_v2_links YAML configs
├── scripts/                                  <- External dependency fetcher (`fetch_external.sh`) & pilot runner
├── docs/                                     <- Upstream protocol, contract, linkage & pilot reports
├── tests/                                    <- Automated pytest suite (common, Track 1, Track 2, morph_ldl)
├── mgn_data/                                 <- MGN paradigm inputs (not committed; MANIFEST.sha256 & README are)
├── external/                                 <- Pinned external clones (not committed; see scripts/fetch_external.sh)
└── outputs/                                  <- Local upstream stage outputs (not committed)
```

---

## Data Provenance

All data used here come from third-party sources, listed below with the versions used.

| Source | What we use | Version | Where |
|---|---|---|---|
| GeLaTo | population metadata, Glottocodes, sample sizes, effective population sizes ($N_e$), pairwise $F_{ST}$ | `gelato-data` `c625fdc` | `analyses/gelato_feasibility_2026_10_01/sources/` |
| Graff et al. 2025 archive | population–language table, Human Origins ADMIXTURE runs ($K = 12\text{--}30$), precomputed $F_2$ blocks (for $f_3$/$f_4$), and $F_{ST}$ | Zenodo `15263706` | `analyses/gelato_feasibility_2026_10_01/sources/` (precomputed $F_2$ external) |
| Grambank (Skirgård et al. 2023) | `29` clean morphological items across 5 domains + `151` non-morphological doc features (and earlier `cat12`/`flex` sets) | `v1.0.3` (`7ae000c`) | `external/grambank` |
| Glottolog CLDF | language roll-up, family taxonomy, macroareas, coordinates, endangerment (AES) | `v5.3` (`072ca0d`) | `external/glottolog-cldf` |
| Bouckaert global language tree | time-calibrated within-family phylogenies (`bouckaert_stem`, `bouckaert_crown`) | `v2.0.0` | `analyses/common/data/bouckaert_v2.nwk` |
| Speaker demography & range area | audited L1 speaker counts & measured polygon range areas (Bromham, Koplenig, Ethnologue multi-ISO) | pinned tables | `analyses/common/demography_sources/` |
| Track 2 morphological paradigms | UniMorph, Paralex, Morph-it!, PoliMorf, VESUM, CELEX/MGN paradigm tables | SHA-256 pinned | `analyses/paradigm/data/provenance_manifest.csv` |
| Track 2 token frequencies | Universal Dependencies `v2.15` (`@r2.15`), Leipzig Corpora Collection, `wordfreq` `v3.1.1` | SHA-256 pinned | `analyses/paradigm/data/freq_provenance.csv` |
| MGN data (Guzmán Naranjo 2024) | upstream `pcfp_v2` pilot paradigm tables | `mgn_data/MANIFEST.sha256` | `mgn_data/` ([README](mgn_data/README.md)) |
| `languages-of-the-world` | ISO 639-3 $\to$ Glottocode & Glottolog tree utilities | `d319631` (`0.2.0`) | `external/` |
| `JudiLing.jl` & `ALmorphinfl` | upstream `pcfp_v2` LDL pilot & active-selection reference | `1.0.1` (`ca77304`), `3caf0d0` | `julia/Manifest.toml`, `external/` |

Licences of the third-party files redistributed in `analyses/gelato_feasibility_2026_10_01/sources/`:
GeLaTo data (`gelato-org/gelato-data`) CC BY-NC 4.0; Graff et al. 2025 archive (Zenodo `15263706`) CC BY 4.0; global language trees (`rbouckaert/global-language-tree-pipeline`) MIT. External paradigm, corpus, Grambank, and Glottolog sources are fetched programmatically via `scripts/fetch_external.sh` and `analyses/paradigm/{ingestion,frequencies}/fetch_*.py` rather than redistributed here.
