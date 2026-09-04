# PhD Study 3 — Morphological complexity and speaker demography

Does the **size and ecology of a speech community** predict the **inflectional
predictability** of its language?

Complexity is *I-complexity* in the sense of Ackerman & Malouf (2013), measured as in
Guzmán Naranjo (2024): the accuracy of predicting one paradigm cell from another. One
observation is one directed cell pair in one language, `correct` successes out of
`total` cross-validation trials.

**The question, the design decisions, the model results and the caveats are all in
[docs/METHODS.md](docs/METHODS.md).** Read that before changing anything in the
pipeline. This file covers layout, how to run it, and where the data comes from.

---

## Status

The models have been fitted once, on 2026-09-04. **Neither meets the R̂ < 1.01 /
ESS > 400 convergence target** — the spatial GP length-scale and the phylogenetic SD are
weakly identified, and three terms compete for the same between-language variance across
only 64 languages. The population-level coefficients are well behaved and usable; the
variance decomposition is not. See [docs/METHODS.md §5](docs/METHODS.md).

---

## Two tracks

| Track | What it is | Where |
|---|---|---|
| **A — Global** | 64 languages × demographic, ecological, phylogenetic and spatial covariates → Bayesian beta-binomial models | `src/`, the three root `.R` scripts, `mgn_data/`, `data_sources/` |
| **B — Germanic historical** | Low and High German diachrony: Middle Low German verbs from CorA-ReN, plus historical Germanic UniMorph paradigms | `germanic/`, `src/low_german_extraction.py` |

Track B inherits the demographic registry Track A builds; it is otherwise independent
and has no model yet.

---

## Repository map

```
study_3/
├── README.md                        ← you are here
├── docs/METHODS.md                  ← the study: question, decisions, results, caveats
│
├── src/                             ← Python pipeline (5 modules, all live)
│   ├── mgn_language_map.py          · single source of truth: MGN ↔ ISO ↔ Glottolog
│   ├── cell_normalization.py        · parsers for the six MGN cell-label conventions
│   ├── build_demographic_registry.py· multi-source demographic fusion
│   ├── merge_mgn_features.py        · distance computation + demographic merge
│   └── low_german_extraction.py     · TRACK B: CorA-ReN XML → verb tokens
│
├── build_phylo_matrix.R             ← Glottolog lineages → 64×64 correlation matrix
├── fit_bayesian_models.R            ← the models
├── plot_language_map.R              ← coverage map of the modelled sample
├── scripts/fetch_cora_ren.sh        ← retrieve the Middle Low German corpus
│
├── data_sources/                    ← INPUTS (see §Data)
├── mgn_data/results-final/          ← the four MGN files the pipeline reads
├── germanic/                        ← TRACK B data
│
├── global_demographic_registry.csv  ← GENERATED: 7,872 languages × 17 cols
├── mgn_modeling_dataset.csv.gz      ← GENERATED: 111,315 trials × 38 cols, 64/64 langs
├── mgn_language_map.json            ← GENERATED, read by the R scripts
├── phylo_cov_matrix.rds             ← GENERATED: 64×64 phylogenetic correlation matrix
│
├── results/  plots/  fits/          ← model output (fits/ is not committed, 163 MB each)
└── tests/                           ← 5 tiers, 96 tests
```

---

## Run order

```bash
PYTHONPATH=src python3 src/build_demographic_registry.py   # → global_demographic_registry.csv
PYTHONPATH=src python3 src/merge_mgn_features.py           # → mgn_modeling_dataset.csv
Rscript build_phylo_matrix.R                               # → phylo_cov_matrix.rds
Rscript plot_language_map.R                                # → plots/map_mgn64.png
python3 -m pytest tests/ -q                                # 92 passed, 4 skipped*
```

\* The four skips are the model-fit checks: `fits/*.rds` is not committed, so they skip
on a fresh clone and pass (96) once you have fitted the models.

The modelling dataset is committed gzipped; `gunzip -k mgn_modeling_dataset.csv.gz`
restores it, or just re-run step 2.

Fitting the models — the run reported in `docs/METHODS.md`:

```bash
MGN_UNIT=source_cell MGN_CHAINS=4 MGN_THREADS=4 MGN_ITER=3000 MGN_WARMUP=1500 \
  Rscript fit_bayesian_models.R
```

Knobs: `MGN_UNIT` (`pair` = 111,315 cell pairs, `source_cell` = 3,175 aggregated source
cells), `MGN_MODELS` (`minimal`/`comprehensive`/`both`), `MGN_CHAINS`, `MGN_THREADS`
(cores *within* each chain), `MGN_ITER`, `MGN_WARMUP`, `MGN_ADAPT_DELTA`,
`MGN_MAX_TREEDEPTH`, `MGN_SUBSAMPLE` (rows per language × POS; unset = full data),
`MGN_DROP_ECO_IMPUTED=1` (refit Model 2 on the 53 languages with measured covariates).

Track B, independently:

```bash
bash scripts/fetch_cora_ren.sh          # only if you need the corpus itself
python3 src/low_german_extraction.py    # → germanic/extracted_verbs.csv
```

---

## Data

Nothing in this repository is our own primary data. Everything below is third-party and
should be cited as such in any write-up.

### Morphological complexity

**MGN — inflectional prediction accuracy.**
Guzmán Naranjo, Matías (2024). *An analogical approach to the typology of inflectional
complexity.* Journal of Language Modelling 12(2). ⟨https://jlm.ipipan.waw.pl/⟩

The upstream repository is ~8.7 GB. **Four files are committed**, the only ones the
pipeline reads, under `mgn_data/results-final/`:

| File | Size | Role |
|---|---|---|
| `compressed-lang-pairs-v.csv.gz` | 5.0 MB | verb cell pairs |
| `compressed-lang-pairs-n.csv` | 2.4 MB | noun cell pairs |
| `compressed-lang-pairs-adj.csv.gz` | 0.1 MB | adjective cell pairs |
| `all-accuracies.csv` | 33 KB | per-language aggregate accuracies |

The verb file is committed as its gzip: the plain 48 MB `.csv` has byte-identical
contents (verified by md5). The noun file is committed plain, because the shipped
`.csv` and `.csv.gz` differ in float precision and the `.csv` is what the pipeline used.
Everything else in `mgn_data/` — including the 5.3 GB uncompressed verb pair file and
the per-language paradigm data — is gitignored; get it from the author's repository.

### Demography, ecology and taxonomy

**Bromham et al. — L1 population and macro-ecological covariates.**
Bromham, L., Dinnage, R., Skirgård, H., Ritchie, A., Cardillo, M., Meakins, F.,
Greenhill, S. & Hua, X. (2022). *Global predictors of language endangerment and the
future of linguistic diversity.* Nature Ecology & Evolution 6:163–173.
Data: ⟨https://github.com/huaxia1985/LanguageEndangerment⟩ (`data.Rdata`).
The 11 MB `data.Rdata` is gitignored and auto-downloads; the 388 KB extract the
pipeline actually uses, `data_sources/bromham_extracted.csv`, **is committed**.
*Sole source of contact richness, range area, altitude range and roughness.*

**Ethnologue multi-ISO table — population for the major world languages.**
Distributed with Dinnager's replication materials for the above:
⟨https://github.com/rdinnager/language_endangerment⟩ →
`data/all_multi_ISO_languages.csv` (564 KB, **committed**). The `LMP_POP1` field is the
same L1 variable as Bromham's, and covers English, Spanish, Russian, French,
Portuguese, Hindi, Urdu and Croatian, which Bromham omits.

**Koplenig — L2 proportion, vehicularity, fallback population.**
Koplenig, Alexander (2019). *Language structure is influenced by the number of speakers
but seemingly not by the proportion of non-native speakers.* Royal Society Open Science
6:181274. ⟨https://doi.org/10.1098/rsos.181274⟩
As `data_sources/ethnologue_population_data.csv` (2,143 languages, **committed**).

**Glottolog v5.3 — glottocodes, families, macro-areas, coordinates, lineages.**
Hammarström, H., Forkel, R., Haspelmath, M. & Bank, S. *Glottolog.* Max Planck
Institute for Evolutionary Anthropology. ⟨https://glottolog.org⟩
Two files, both **committed**:
- `data_sources/glottolog_extracted.csv` (7.3 MB) — exported from the `lingtypology` R
  package (Moroz 2017). **Not re-downloadable**: it is generated by R and
  `build_phylo_matrix.R` fails without it, which is why it is in the repository.
- `data_sources/Glottolog_lookup_table_Heti_edition.tsv` (5.7 MB) — the ISO↔Glottocode
  lookup, from the `rdinnager` repository above.

### Germanic historical (Track B)

**CorA-ReN — Reference Corpus Middle Low German / Low Rhenish (1200–1650).**
ReN-Team (2021). Version 1.1, CorA-XML release. Universität Hamburg / ZFDM.
⟨https://doi.org/10.25592/uhhfdm.9195⟩ — **CC BY 4.0.**
694 MB, **not committed**. `bash scripts/fetch_cora_ren.sh` walks you through getting it.
The derived verb table, `germanic/extracted_verbs.csv` (183,450 tokens, 19 MB), **is
committed**, so you only need the corpus if you want to change the extraction.

**UniMorph — historical Germanic verb paradigms.**
⟨https://unimorph.github.io/⟩, per-language repositories at
`https://github.com/unimorph/<iso>`. Committed under `germanic/unimorph/raw/`, filtered
to verbs and reshaped to `lemma,form,paradigm_slot`:

| ISO | Language | Role |
|---|---|---|
| `goh` | Old High German | High German diachrony |
| `gmh` | Middle High German | High German diachrony |
| `deu` | Modern Standard German | High German diachrony, endpoint |
| `osx` | Old Saxon | Low German diachrony |
| `nds` | Low German | Low German diachrony, endpoint |
| `ang` | Old English | nearest West Germanic comparandum |
| `got` | Gothic | East Germanic outgroup |
| `non` | Old Norse | North Germanic outgroup |

The Middle Low German stage comes from CorA-ReN, not UniMorph.

### Hand-curated in this repository

`data_sources/cells_to_unimorph.json` (303 entries) maps MGN's dotted cell labels onto
UniMorph feature bundles. It is the only human input to the normalisation step.

---

## What is not committed, and why

| Path | Size | Why |
|---|---|---|
| `fits/*.rds` | 163 MB each | over GitHub's 100 MB per-file limit; re-fit with the command above |
| `mgn_data/` except the four files above | 8.7 GB | third-party; the committed subset is everything the pipeline reads |
| `germanic/cora_ren_xml_1.1/` | 694 MB | third-party, CC BY 4.0, available by DOI; `scripts/fetch_cora_ren.sh` |
| `data_sources/data.Rdata` | 11 MB | auto-downloads; the extract we use is committed |
| `mgn_modeling_dataset.csv` | 46 MB | the 3 MB `.gz` is committed instead |

Everything else — the registry, the phylogenetic matrix, the language map, the model
results and the plots — **is committed**, so the analysis can be inspected
and the models re-fitted without regenerating anything upstream.
