# Morphology sampling → LDL outcomes → GeLaTo linkage

PhD Study 3: inflectional complexity and genetic admixture. The repository holds the
current pipeline at its root. Earlier exploratory tracks (the MGN-accuracy Bayesian models
with demographic and phylogenetic covariates, and the Germanic historical track) were
archived on 2026-10-08; see [Archive](#archive).

**Status.** The current experiment is `pcfp_v1`: paradigm cell filling for Italian and
Finnish verbs (MGN), with LDL as its own active-selection model, plus a Grambank
inflection-extent outcome for all GeLaTo-linked languages. See
[docs/REPORT_pcfp_v1.md](docs/REPORT_pcfp_v1.md). The earlier `pilot_v1` (a source-known
new-verb task with a Transformer selector) is archived: its outputs are in
`outputs/pilot_v1/`, its report is [docs/REPORT.md](docs/REPORT.md), and its code state is
commit `24390cf`.

The pipeline produces two morphology outcomes for a later admixture–morphology analysis
(not fitted here):

* **Grambank inflectional extent** (primary). The number of 12 inflectional categories
  present out of those coded, for every GeLaTo-linked language in Grambank, including
  isolating languages. Examples of categories: tense, person indexing, case, gender
  agreement.
  * Each category merges its logically dependent Grambank features by OR, following the
    GBI curation of Graff et al. 2025.
  * pcfp_v1 used the share of 35 raw features.
* **LDL predictability** (secondary; inflecting languages only). Every verb is known
  through k randomly drawn forms (k ~ U{1..7}, the same cap in every language). A native
  LDL model (JudiLing.jl, end-state, simulated semantics) fills the hidden cells.
  * Training samples are 100 fixed **core** verbs per outer fold plus 100 verbs chosen by
    **active selection** or by matched **random selection**. The LDL model itself is the
    selector (low-confidence or high-entropy scores over its candidate supports).
  * The primary test items are the hidden cells of the core verbs, identical for every
    policy.
  * Uncertainty comes from lemma-cluster bootstrap intervals.

Both outcomes are keyed by the language-level Glottocode. GeLaTo populations are linked
in separate tables.

| Document | Content |
|---|---|
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | experimental specification (estimand, task, CV, metrics, uncertainty) |
| [docs/CONTRACT.md](docs/CONTRACT.md) | schemas, identifiers, permitted held-out information, interfaces |
| [docs/DATA.md](docs/DATA.md) | adapters, provenance, identifiers, eligibility, GeLaTo matching and reviews |
| [docs/SELECTION.md](docs/SELECTION.md) | selector, scores, acquisition, ALmorphinfl audit |
| [docs/LDL_PROTOCOL.md](docs/LDL_PROTOCOL.md) | JudiLing audit, held-out-lemma protocol, runner |
| [docs/REPORT.md](docs/REPORT.md) | pilot_v1 results (archived task) |
| [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md) | one-page diagram of the stages and fitted models |
| [docs/gelato_feasibility_2026-10-01.md](docs/gelato_feasibility_2026-10-01.md) | GeLaTo feasibility review (what the genetic data can and cannot measure) |
| [docs/TYPOLOGY.md](docs/TYPOLOGY.md) | Grambank inflection-extent outcome: sources, 12-category feature set, mapping rules, planned analysis |
| [docs/REPORT_pcfp_v1.md](docs/REPORT_pcfp_v1.md) | pcfp_v1 results (current) |

## Setup

```bash
scripts/fetch_external.sh                       # pinned external sources (incl. Grambank, Glottolog CLDF)
python3 -m venv .venv                           # any Python ≥ 3.12
.venv/bin/pip install -r requirements.lock      # exact versions used (torch 2.14.1 CPU)
.venv/bin/pip install ./external/languages-of-the-world
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'   # Julia 1.12, JudiLing =1.0.1
```

Inputs that are read but never modified: `mgn_data/` (see [Data provenance](#data-provenance))
and `analyses/gelato_feasibility_2026_10_01/` (GeLaTo commit c625fdc, Zenodo 15263706).
Paths in the configs are relative to the repository root.

## Run

Use one entry point. Every stage is independently runnable and writes a `stage_manifest.json`
(config hash, input hashes, git state, external revisions, package versions, seeds).

```bash
.venv/bin/python -m morph_ldl.cli <stage> --config configs/pcfp_v1.yaml [--units ...] [--folds ...] [--policies ...] [--set key=json ...]
```

| Stage | Output under `outputs/<experiment_id>/` |
|---|---|
| `data` | `data/forms/<unit>.csv`, `registry/`, `eligibility/`, `gelato/crosswalk.csv` |
| `splits` | `splits/<unit>/cell_inventory.csv`, `eligible_lemmas.csv`, `exposure_manifest.csv`, `auxiliary_manifest.csv`, `rep{r}/split_manifest.csv` |
| `ldl_tune` | `ldl_tune/chosen_settings.json` (cue n-gram and inflection SD, chosen on auxiliary verbs; runs before `select`) |
| `select` | `selection/<unit>/rep{r}/fold{k}/<policy>@<pool>/`: order, logs, selector rounds, cell scores, comprehension check, `samples/budget_{B}*.csv` |
| `ldl` | `queries/…/queries.csv` (gold-free), `ldl/…/budget_{B}/predictions.csv`, `mapping_quality.csv` |
| `evaluate` | `eval/item_predictions.csv`, summaries (by cell, by k), copy rates, sample composition, selector checks |
| `outcomes` | `outcomes/ldl_outcomes.csv`, `paired_differences.csv`, `population_links.csv` |
| `typology` | `typology/grambank_inflection.csv`, `grambank_population_links.csv`, `feature_set.json`, coverage summary |
| `audit` | `eval/artifact_audit.json` (samples, selector rounds, queries and typology checked against the manifests) |

`scripts/run_pcfp.sh configs/pcfp_v1.yaml` runs every stage from `splits` through `audit`.
`configs/pcfp_smoke.yaml` inherits from it and runs the full chain at tiny sizes.

Tests: `.venv/bin/python -m pytest` runs all tests. Add `-m "not slow"` to skip the real-Julia tests.

## Key outputs for the next analysis stage

* `outputs/pcfp_v1/typology/grambank_inflection.csv` has one row per language-level
  Glottocode. It carries counts (`n_present`, `n_coded`, `n_features`) for the main set
  and the sensitivity sets, coverage, the no/minimal-inflection flags, clitic flags,
  family and macroarea.
* `outputs/pcfp_v1/outcomes/ldl_outcomes.csv` has one row per unit × item set × policy ×
  pool cap × budget. It carries accuracy (micro and verb-macro), edit distance, 95%
  lemma-cluster bootstrap intervals, exposure metadata and the language-level `glottocode`.
* The population link tables (`outcomes/population_links.csv`,
  `typology/grambank_population_links.csv`) carry match statuses, sample sizes and
  identifiers, but no ancestry values.

## Repository layout

```
morph_ldl/        Python package: data, cv, selection, ldl, typology stages; cli.py is the entry point
  data/vendor/    modules vendored from the earlier code (MGN↔ISO map, cell-label parsers)
  data/resources/ cells_to_unimorph.json (curated MGN cell label → UniMorph features)
julia/            JudiLing runner (Project.toml pins JudiLing =1.0.1; Manifest.toml committed)
configs/          pcfp_v1.yaml, pcfp_smoke.yaml (current); pilot.yaml, smoke.yaml (archived pilot_v1)
tests/            pytest suite
scripts/          fetch_external.sh (pinned external sources), run_pcfp.sh, run_pilot.sh (archived)
docs/             protocol, contract, component docs, report, next-task brief
analyses/         GeLaTo feasibility audit (2026-10-01) and coverage estimates (2026-10-08)
mgn_data/         third-party MGN paradigm inputs (not committed; MANIFEST.sha256 is)
external/         pinned external clones (not committed; scripts/fetch_external.sh)
outputs/          stage outputs; only small pilot_v1 result tables are committed
```

## Data provenance

Nothing here is our own primary data. Cite every source below in any write-up.

| Source | What we use | Version | Where |
|---|---|---|---|
| MGN data (Guzmán Naranjo 2024, *J. Language Modelling* 12(2)) | per-language paradigm tables `data/`, `data-custom/`, build scripts `build-data/` | upstream URL/commit not yet recorded; file hashes in `mgn_data/MANIFEST.sha256` | `mgn_data/` ([README](mgn_data/README.md)) |
| UniMorph (Italian, Finnish) | verification of the MGN tables against UniMorph | unimorph/ita fa2cc6c, unimorph/fin fe0a270 | `external/` |
| languages-of-the-world | ISO 639-3 → Glottocode layer, with project corrections | d319631 (0.2.0) | `external/`, `morph_ldl/data/vendor/mgn_language_map.py` |
| JudiLing.jl | LDL implementation | 1.0.1 (= ca77304) | `julia/Manifest.toml`, `external/` |
| ALmorphinfl (Muradoğlu & Hulden 2022) | reference implementation for the active-learning scores | 3caf0d0 | `external/` |
| GeLaTo | population metadata, Glottocodes, sample sizes | gelato-data c625fdc | `analyses/gelato_feasibility_2026_10_01/sources/` |
| Graff et al. 2025 archive | ADMIXTURE Q-matrix identifiers and population–language table (no ancestry values are used yet) | Zenodo 15263706 | `analyses/gelato_feasibility_2026_10_01/sources/` |
| Grambank (Skirgård et al. 2023) | inflection-extent outcome (12 inflectional categories; pcfp_v1: 35 features) | v1.0.3 (7ae000c) | `external/grambank` |
| Glottolog CLDF | dialect → language roll-up, family, macroarea | v5.3 (072ca0d) | `external/glottolog-cldf` |

The exact input hashes of each run are in its `outputs/<experiment>/<stage>/stage_manifest.json`
(written locally by every stage; not committed, because they record absolute paths).

Licences of the third-party files redistributed in `analyses/gelato_feasibility_2026_10_01/sources/`:
GeLaTo data (gelato-org/gelato-data) CC BY-NC 4.0; Graff et al. 2025 archive (Zenodo 15263706)
CC BY 4.0; global language trees (rbouckaert/global-language-tree-pipeline) MIT. MGN, UniMorph
and the other external sources are not redistributed here.

## Archive

Material from earlier implementation paths was moved, unchanged, out of the repository
on 2026-10-08 and is kept privately (a local, gitignored `ARCHIVE_NOTE.md` says where).
Everything that was committed is also in git history before the
archive commit. Archived: the global MGN-accuracy track (`src/`, root R scripts, `fits/`,
`results/`, `plots/`, `global_demographic_registry.csv`, `mgn_modeling_dataset.csv*`,
`phylo_cov_matrix.rds`, `data_sources/`, `docs/METHODS.md`, its tests), the Germanic track
(`germanic/`, `scripts/fetch_cora_ren.sh`), MGN's own results and analysis scripts
(about 8.4 GB of `mgn_data/`), agent scratch folders, and the previous README.

The GeLaTo feasibility audit script (`analyses/gelato_feasibility_2026_10_01/audit.py`)
read `mgn_modeling_dataset.csv`, `mgn_language_map.json` and
`data_sources/Glottolog_lookup_table_Heti_edition.tsv`; these are now in the archive. Its
outputs, which the pipeline reads, are unchanged.

