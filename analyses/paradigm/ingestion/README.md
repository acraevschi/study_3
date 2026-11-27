# Track 2 Morphological Paradigm Ingestion Pipeline (`analyses/paradigm/ingestion/`)

This package implements the reproducible acquisition, SHA-256 provenance verification, paradigm cleaning, three-rule root-cognate grouping, Malouf (2017) vowel-harmony pre-abstraction, and cross-validation split generation for **Study 3 — Track 2 (Paradigm Predictability)**.

---

## Module Overview

| Script | Responsibility |
| :--- | :--- |
| [`fetch_sources.py`](fetch_sources.py) | Programmatically downloads and verifies Priority-1 Paralex Zenodo packages, documented CELEX English Verbs, Priority-2 National Lexicons (`Morph-it! v0.48`, `PoliMorf 0.6.7`, `VESUM v6.8.6`), Priority-3 `github.com/unimorph/*` shallow checkouts (`git rev-parse HEAD`), and SIGMORPHON 2020 Uralic tables (`mdf`, `myv`, `udm`). Writes `analyses/paradigm/data/provenance_manifest.csv`. |
| [`build_lexicons.py`](build_lexicons.py) | Executes the Priority 1 → 2 → 3 source cascade across accepted GeLaTo Glottocodes (`analyses/common/data/analysis_links.csv`), normalizes NFC, strips periphrastic multiword cells (`max_multiword_share <= 0.50`), respects `#DEF#`, deduplicates lexicon-wide 100%-syncretic cells, computes per-lemma `shape_id`, assigns `leakage_group_id`, exports `lexicons/`, `cell_maps/`, `splits/`, `lexicons_malouf/`, and `unit_summary.csv`. |
| [`grouping.py`](grouping.py) | Three-rule Union-Find root-family clustering (`leakage_group_id`) within `(glottocode, pos)` using Pigeonhole-Principle `O(N * L)` cell-form indexing (`100%` exact equivalence to all-pairs checking). |
| [`splits.py`](splits.py) | Deterministic SHA-256 CV split manifests (`{unit_id}_uniform120.csv` with `R=5 × K=5` folds of 150 distinct root groups `[96 subtrain / 24 dev / 30 test]`, and `{unit_id}_uniform480.csv` for `tier2 >= 600` units with `R=3 × K=5` folds of 600 groups `[384 subtrain / 96 dev / 120 test]`). |
| [`harmony_malouf.py`](harmony_malouf.py) | Unsupervised bipartite front/back vowel-harmony discovery and affix archiphoneme abstraction (Malouf 2017) producing `analyses/paradigm/data/lexicons_malouf/{unit_id}.csv` for Uralic, Turkic, Mongolic, Tungusic, and Chukotko-Kamchatkan units. |

---

## Output Schemas

### 1. `analyses/paradigm/data/lexicons/{unit_id}.csv` (and `lexicons_malouf/{unit_id}.csv`)
Exact 14-column long paradigm schema (`unit_id = {glottocode}_{pos}_{representation}`):
```text
unit_id,glottocode,iso639_3,pos,representation,source_id,lemma_id,lemma_label,leakage_group_id,cell,raw_form,segmented_form,variant_idx,shape_id
```
- Complete synthetic paradigms have `variant_idx == 0` non-missing across all `n_cells_distinct >= 2` deduplicated canonical synthetic cells; attested overabundant variants are retained with `variant_idx > 0`.
- `shape_id` hashes the lemma's cell-syncretism equivalence partition over `variant_idx == 0` forms so within-lemma syncretic cells are explicitly identified.
- Paired phonological feature tables (`sounds.csv`) for Paralex `phon` units are stored at `analyses/paradigm/data/cell_maps/{unit_id}_sounds.csv` (recorded in `unit_summary.csv:sounds_file`) so `data/lexicons/*.csv` contains exclusively 14-column paradigm tables.

### 2. `analyses/paradigm/data/splits/{unit_id}_{regime}.csv`
```text
unit_id,regime,draw_idx,fold_idx,lemma_id,leakage_group_id,role,inner_role
```
- `role in {"train", "test"}`, `inner_role in {"subtrain", "dev", "test"}`.
- Strictly 1 canonical simplex lemma per `leakage_group_id` and **zero** `leakage_group_id` overlap between `train` (`subtrain`/`dev`) and `test`.

---

## Reproducible Execution

```bash
# 1. Fetch & verify all external paradigm sources (<= 4 download workers)
.venv/bin/python analyses/paradigm/ingestion/fetch_sources.py --workers 4

# 2. Build cleaned lexicons, cell maps, Malouf abstractions, and CV splits
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python analyses/paradigm/ingestion/build_lexicons.py

# 3. Run unit & end-to-end provenance/leakage verification tests
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_paradigm_ingestion.py -v
```
