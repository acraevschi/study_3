# Morphology sampling → LDL outcomes → GeLaTo linkage

> **Work in progress.** Everything in this repository is preliminary: the pipeline, both
> morphology outcomes, the reports and the analyses. None of the results is a finding.
> * **Grambank outcome.** Exact and rule-based GeLaTo population–language links are accepted
>   by rule (`configs/pcfp_v2_links.yaml`), and all 33 doubtful cases are decided in
>   `morph_ldl/data/gelato_link_review.csv` ([docs/TYPOLOGY.md](docs/TYPOLOGY.md) §13).
> * **LDL outcome.** Under revision: the 2026-10-09 ablation changed the model
>   settings, and the absolute accuracy is still too low to treat the score as a measure
>   of predictability.
> * **Admixture models.** Exploratory two-stage admixture–Grambank models (12 inflectional
>   categories and Grambank Flexivity across 202 languages, link sensitivity sets, $f_3$,
>   and `het`) are in [`analyses/admixture_links_2026_10_21/`](analyses/admixture_links_2026_10_21/README.md).
>   The ancestry values have been looked at, so later analyses are not blind to them.

PhD Study 3: inflectional complexity and genetic admixture.

The pipeline produces two morphology outcomes (keyed by language-level Glottocode, with
GeLaTo populations linked in separate tables) for the admixture–morphology analysis; the
pipeline itself never reads ancestry values:

* **Grambank inflectional extent** (primary). The number of 12 inflectional categories
  present out of those coded, for every GeLaTo-linked language in Grambank, including
  isolating languages. Examples of categories: tense, person indexing, case, gender
  agreement.
  * Each category merges its logically dependent Grambank features by OR, following the
    GBI curation of Graff et al. 2025.
* **LDL predictability** (secondary; inflecting languages only). Every verb is known
  through k randomly drawn forms (k ~ U{1..7}, the same cap in every language). A native
  LDL model (JudiLing.jl, end-state, simulated semantics) fills the hidden cells.
  * Training samples are 100 fixed **core** verbs per outer fold plus 100 **randomly
    drawn** verbs. There are 5 independent draws, each shared by all 5 folds.
  * The test items are the hidden cells of the core verbs, identical in every draw.
  * Uncertainty comes from lemma-cluster bootstrap intervals. The spread over draws is
    reported separately as a robustness check.
  * Option `cv.design: core_selected` (pcfp_v1) instead adds 100 verbs chosen by **active
    selection**, with the LDL model itself as the selector, or by matched random
    selection.

| Document | Content |
|---|---|
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | experimental specification (estimand, task, CV, metrics, uncertainty) |
| [docs/CONTRACT.md](docs/CONTRACT.md) | schemas, identifiers, permitted held-out information, interfaces |
| [docs/DATA.md](docs/DATA.md) | adapters, provenance, identifiers, eligibility, GeLaTo matching and reviews |
| [docs/SELECTION.md](docs/SELECTION.md) | selector, scores, acquisition, ALmorphinfl audit |
| [docs/LDL_PROTOCOL.md](docs/LDL_PROTOCOL.md) | JudiLing audit, held-out-lemma protocol, runner |
| [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md) | one-page diagram of the stages and fitted models |
| [docs/TYPOLOGY.md](docs/TYPOLOGY.md) | Grambank inflection-extent outcome: sources, 12-category feature set, mapping rules, link review |
| [docs/REPORT_pcfp_v2.md](docs/REPORT_pcfp_v2.md) | `pcfp_v2` results: repeated random draws |
| [analyses/ldl_ablation_2026_10_09/README.md](analyses/ldl_ablation_2026_10_09/README.md) | LDL ablation (preliminary): cell-specific meaning vector and decoder settings |
| [analyses/admixture_links_2026_10_21/README.md](analyses/admixture_links_2026_10_21/README.md) | exploratory admixture → Grambank inflection & Flexivity models on the 2026-10-20 links |

## Setup & Run

```bash
# Setup
scripts/fetch_external.sh                       # pinned external sources (incl. Grambank, Glottolog CLDF)
python3 -m venv .venv                           # any Python ≥ 3.12
.venv/bin/pip install -r requirements.lock      # exact versions used
.venv/bin/pip install ./external/languages-of-the-world
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'   # Julia 1.12, JudiLing =1.0.1

# Run a single stage (data | splits | ldl_tune | select | ldl | evaluate | outcomes | typology | audit)
.venv/bin/python -m morph_ldl.cli <stage> --config configs/pcfp_v2_links.yaml

# Run splits -> audit end-to-end, or run the test suite
scripts/run_pcfp.sh configs/pcfp_v2.yaml
.venv/bin/python -m pytest                      # add -m "not slow" to skip real-Julia tests
```

Every stage is independently runnable and writes a `stage_manifest.json` under
`outputs/<experiment_id>/` (see [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md) and
[docs/CONTRACT.md](docs/CONTRACT.md) for stage outputs and schemas).

## Repository layout

```
study_3/
├── README.md                                 <- Project overview, setup, repository layout, data provenance
├── requirements.lock                         <- Exact Python package versions
├── pytest.ini                                <- Test settings (`slow` marker for real-Julia tests)
│
├── morph_ldl/                                <- Python package with all pipeline stages
│   ├── cli.py                                <- Single entry point: `python -m morph_ldl.cli <stage>`
│   ├── config.py / seeds.py / schemas.py     <- Config loading, derived seeds, table schemas
│   ├── provenance.py                         <- Stage manifests (hashes, git state, versions)
│   ├── data/                                 <- `data` stage: MGN adapters, cells, eligibility, registry
│   │   ├── gelato.py                         <- GeLaTo population ↔ language crosswalk
│   │   ├── gelato_link_review.csv            <- GeLaTo link decision sheet (decision + confirmed_by + date)
│   │   ├── gelato_review.yaml                <- Earlier link reviews (archive for the typology stage)
│   │   ├── resources/                        <- cells_to_unimorph.json (MGN cell label → UniMorph features)
│   │   └── vendor/                           <- MGN↔ISO map and parsers
│   ├── cv/                                   <- `splits`, `select`, `evaluate`, `outcomes`, `audit` stages
│   │   ├── pcfp.py                           <- Paradigm cell filling task: cells, exposure draws
│   │   ├── splits.py                         <- Core folds and random-draw manifests
│   │   ├── pipeline.py                       <- Stage orchestration and artifact audit
│   │   └── outcomes.py / bootstrap.py        <- LDL outcome table, cluster bootstrap, draw variability
│   ├── ldl/                                  <- LDL runner (calls Julia), semantics, LDL selector
│   ├── selection/                            <- Active-selection scores and acquisition loop
│   └── typology/                             <- `typology` stage: Grambank inflection-extent outcome
│       ├── grambank.py                       <- Feature-set parsing, 12-category OR merge
│       ├── stage.py                          <- Glottocode mapping, link rules, coverage, links, audit
│       ├── link_review.py                    <- Decision sheet: checks and merge-safe regeneration
│       └── default_config.yaml               <- Declared feature set and mapping rules
│
├── julia/                                    <- JudiLing runner (JudiLing pinned to 1.0.1)
│   ├── Project.toml / Manifest.toml
│   ├── src/LDLRunner.jl                      <- End-state LDL fit and decoding
│   └── bin/                                  <- run_jobs.jl (batch fits), selector_server.jl (LDL selector)
│
├── configs/
│   ├── pcfp_v2_links.yaml                    <- Current experiment with the GeLaTo link rules of 2026-10-20
│   ├── pcfp_v2.yaml / pcfp_v2_smoke.yaml     <- Repeated-random design
│   └── pcfp_v1.yaml / pcfp_smoke.yaml        <- core_selected design (active vs random selection)
│
├── scripts/
│   ├── fetch_external.sh                     <- Clones the pinned external sources into external/
│   └── run_pcfp.sh                           <- Runs every stage from `splits` to `audit`
│
├── docs/                                     <- Protocol, contract, component docs and reports (table above)
│
├── analyses/                                 <- One-off analyses, each with its own README
│   ├── gelato_feasibility_2026_10_01/        <- GeLaTo feasibility audit; sources/ holds the cached
│   │                                            GeLaTo, Zenodo 15263706 and language-tree files
│   ├── ldl_ablation_2026_10_09/              <- Why PCFP accuracy is low (preliminary)
│   ├── draft_admixture_model_2026_10_10/     <- Initial exploratory admixture → Grambank draft (175 languages)
│   └── admixture_links_2026_10_21/           <- Admixture → Grambank inflection & Flexivity on 2026-10-20 links
│
├── tests/                                    <- pytest suite (fixtures in tests/fixtures/)
│
├── mgn_data/                                 <- MGN paradigm inputs (not committed; MANIFEST.sha256 and README are)
├── external/                                 <- Pinned external clones (not committed; see fetch_external.sh)
└── outputs/                                  <- Stage outputs (not committed)
```

## Data provenance

All data used here come from third-party sources, listed below with the versions used.

| Source | What we use | Version | Where |
|---|---|---|---|
| MGN data (Guzmán Naranjo 2024, *J. Language Modelling* 12(2)) | per-language paradigm tables `data/`, `data-custom/`, build scripts `build-data/` | file hashes in `mgn_data/MANIFEST.sha256` | `mgn_data/` ([README](mgn_data/README.md)) |
| UniMorph (Italian, Finnish) | verification of the MGN tables against UniMorph | unimorph/ita fa2cc6c, unimorph/fin fe0a270 | `external/` |
| languages-of-the-world | ISO 639-3 → Glottocode layer, with project corrections; Glottolog tree for the admixture models | d319631 (0.2.0) | `external/`, `morph_ldl/data/vendor/mgn_language_map.py` |
| JudiLing.jl | LDL implementation | 1.0.1 (= ca77304) | `julia/Manifest.toml`, `external/` |
| ALmorphinfl (Muradoğlu & Hulden 2022) | reference implementation for the active-learning scores | 3caf0d0 | `external/` |
| GeLaTo | population metadata, Glottocodes, sample sizes; neighbour FST in the admixture models | gelato-data c625fdc | `analyses/gelato_feasibility_2026_10_01/sources/` |
| Graff et al. 2025 archive | population–language table; ADMIXTURE runs (K = 12–30) and F2 blocks, used in the exploratory admixture models | Zenodo 15263706 | `analyses/gelato_feasibility_2026_10_01/sources/` |
| Grambank (Skirgård et al. 2023) | inflection-extent (12 categories) and Flexivity (30 features) outcomes | v1.0.3 (7ae000c) | `external/grambank` |
| Glottolog CLDF | dialect → language roll-up, family, macroarea, endangerment (AES) | v5.3 (072ca0d) | `external/glottolog-cldf` |

The exact input hashes of each run are in its `outputs/<experiment>/<stage>/stage_manifest.json`
(written locally by every stage; not committed, because they record absolute paths).

Licences of the third-party files redistributed in `analyses/gelato_feasibility_2026_10_01/sources/`:
GeLaTo data (gelato-org/gelato-data) CC BY-NC 4.0; Graff et al. 2025 archive (Zenodo 15263706)
CC BY 4.0; global language trees (rbouckaert/global-language-tree-pipeline) MIT. MGN, UniMorph
and the other external sources are not redistributed here.
