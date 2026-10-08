# Morphology sampling → LDL outcomes → GeLaTo linkage

PhD Study 3: inflectional complexity and genetic admixture. The repository holds the
current pipeline at its root. Earlier exploratory tracks (the MGN-accuracy Bayesian models
with demographic and phylogenetic covariates, and the Germanic historical track) were
archived on 2026-10-08; see [Archive](#archive).

**Status.** `pilot_v1` (Italian and Finnish verbs, MGN) is complete; see
[docs/REPORT.md](docs/REPORT.md). The next task replaces the new-verb task below with
paradigm cell filling, uses LDL as its own selector and adds a Grambank outcome
([brief](docs/next_task_pcfp_prompt.md)).

A reproducible pipeline that turns inflectional paradigms into **LDL-based difficulty
outcomes** and links them to **GeLaTo genetic populations** for a later
admixture–morphology analysis (not fitted here).

The task is **source-known paradigm completion for held-out lemmas**. For each held-out
lemma, the model sees one supplied source form (the infinitive) and must produce a fixed
panel of 8 target cells. Training samples of 100, 200 or N lemmas are chosen by
**active selection** (low-confidence or high-entropy scores from a character Transformer
selector, after Muradoğlu & Hulden 2022) or by **matched random selection**. Selection
runs inside every grouped outer cross-validation fold. Each sample gets a freshly
fitted **native LDL model** (JudiLing.jl, end-state, simulated semantics, wug-style
source binding). Uncertainty comes from lemma-cluster bootstrap intervals.

| Document | Content |
|---|---|
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | experimental specification (estimand, task, CV, metrics, uncertainty) |
| [docs/CONTRACT.md](docs/CONTRACT.md) | schemas, identifiers, permitted held-out information, interfaces |
| [docs/DATA.md](docs/DATA.md) | adapters, provenance, identifiers, eligibility, GeLaTo matching and reviews |
| [docs/SELECTION.md](docs/SELECTION.md) | selector, scores, acquisition, ALmorphinfl audit |
| [docs/LDL_PROTOCOL.md](docs/LDL_PROTOCOL.md) | JudiLing audit, held-out-lemma protocol, runner |
| [docs/REPORT.md](docs/REPORT.md) | pilot results, adaptations, failures, limitations, next steps |
| [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md) | one-page diagram of the stages and fitted models |
| [docs/gelato_feasibility_2026-10-01.md](docs/gelato_feasibility_2026-10-01.md) | GeLaTo feasibility review (what the genetic data can and cannot measure) |
| [docs/next_task_pcfp_prompt.md](docs/next_task_pcfp_prompt.md) | brief for the next task: PCFP task, LDL selector, Grambank inflection-extent outcome |

## Setup

```bash
scripts/fetch_external.sh                       # pinned external sources
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
.venv/bin/python -m morph_ldl.cli <stage> --config configs/pilot.yaml [--units ...] [--folds ...] [--policies ...] [--set key=json ...]
```

| Stage | Output under `outputs/<experiment_id>/` |
|---|---|
| `data` | `data/forms/<unit>.csv`, `registry/`, `eligibility/`, `gelato/crosswalk.csv` |
| `splits` | `splits/<unit>/rep{r}/split_manifest.csv` |
| `ldl_tune` | `ldl_tune/chosen_settings.json` (dev-only choice of cue n-gram and inflection SD) |
| `select` | `selection/<unit>/rep{r}/fold{k}/<policy>@<pool>/`: order, logs, per-cell scores, `samples/budget_{B}*.csv` |
| `ldl` | `queries/…/test_queries.csv` (gold-free), `ldl/…/budget_{B}/predictions.csv`, `mapping_quality.csv` |
| `selector` | `selector_eval/…/predictions.csv` (selector accuracy on the same test items) |
| `evaluate` | `eval/item_predictions.csv`, summaries, per-cell, fold variability, sample composition |
| `outcomes` | `outcomes/ldl_outcomes.csv`, `paired_differences.csv`, `population_links.csv` |
| `audit` | `eval/artifact_audit.json` (samples, anchors and queries checked against the manifests) |

`scripts/run_pilot.sh configs/pilot.yaml` runs every stage from `select` through `outcomes`.
`configs/smoke.yaml` inherits from the pilot config and runs the full chain at tiny sizes.
Budget 200 and more repetitions only need config changes, for example
`--set 'selection.budgets=[100,200]' 'cv.repetitions=[0,1,2]'`. The trajectories are nested,
so budget 100 is a prefix of budget 200.

Tests: `.venv/bin/python -m pytest` runs all tests. Add `-m "not slow"` to skip the real-Julia tests.

## Key outputs for the next analysis stage

* `outputs/pilot_v1/outcomes/ldl_outcomes.csv` has one row per unit × policy × pool cap × budget × model. It holds accuracy (micro and lemma-macro), edit distance and normalized edit distance. Each estimate has 95% lemma-cluster bootstrap intervals, counts and provenance.
* `outputs/pilot_v1/outcomes/population_links.csv` has one row per unit × GeLaTo population. It carries the match status, sample sizes and ancestry-source identifiers, but no ancestry values. It applies no aggregation and does not duplicate morphology rows.

## Repository layout

```
morph_ldl/        Python package: data, cv, selection, ldl stages; cli.py is the entry point
  data/vendor/    modules vendored from the earlier code (MGN↔ISO map, cell-label parsers)
  data/resources/ cells_to_unimorph.json (curated MGN cell label → UniMorph features)
julia/            JudiLing runner (Project.toml pins JudiLing =1.0.1; Manifest.toml committed)
configs/          pilot.yaml (pilot_v1), smoke.yaml
tests/            pytest suite (97 tests)
scripts/          fetch_external.sh (pinned external sources), run_pilot.sh
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
| Grambank, Glottolog CLDF | planned Grambank inflection-extent outcome | Grambank v1.0.3, Glottolog CLDF v5.3 | to be pinned by the next task |

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

