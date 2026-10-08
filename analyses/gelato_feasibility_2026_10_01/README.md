# GeLaTo/MGN feasibility audit, 1 October 2026

The substantive assessment is [the review](../../docs/gelato_feasibility_2026-10-01.md).
These files do not modify the study's modelling pipeline.

Run the offline audit from the repository root:

```sh
python3 analyses/gelato_feasibility_2026_10_01/audit.py
```

The script reads the current root modelling CSV and Glottolog lookup, plus pinned
sources in this folder. NumPy is required. Source hashes are in
`outputs/provenance.json`. Outputs record 64 language units, exact identifier
matches, candidate descendant/proxy mappings, sample-size sensitivity, published
contact-case matches, and descriptive ancestry diagnostics. No model is fitted.

Important outputs:

- `outputs/summary.json`: independently reproducible counts and limitations.
- `outputs/mgn_language_coverage.csv`: one row for each of the 64 MGN languages.
- `outputs/population_crosswalk.csv`: all main and expanded population records,
  including mapping status, sample size, and curation evidence.
- `outputs/published_contact_case_overlap.csv`: existing interpreted contact rows
  that reach MGN, with direct and linguistic-proxy matches separated.
- `outputs/unvalidated_candidate_contact_overlap.csv`: candidate search results,
  not accepted contact evidence.
- `outputs/population_ancestry_K23_components.csv`: population means from the
  published individual-level Q matrix, using Table S1's curated language assignments.
- `outputs/population_ancestry_K12_K30_diagnostics.csv` and
  `outputs/ancestry_K_sensitivity.csv`: variation across model resolutions.

`ancestry_heterogeneity` is calculated as 1 minus the sum of squared population-mean
component proportions. `mean_individual_ancestry_heterogeneity` calculates this
quantity within individuals before averaging. They are descriptive, different
quantities. Neither is an identified non-native ancestry proportion, a formal
admixture test, or a contact-intensity measure. `two_component_screen_only` replicates
the archived script's strict greater-than 70%/5% component screen; it does not apply
source assignment, cross-family validation, historical dating, or f3 confirmation.
Component IDs are local to each K and have not been aligned across K.

Sources:

- GeLaTo commit `c625fdcf0225142cc03ae3a1635edf322e9c7778`, retrieved from
  [gelato-org/gelato-data](https://github.com/gelato-org/gelato-data/tree/c625fdcf0225142cc03ae3a1635edf322e9c7778).
- Published 2025 ADMIXTURE results, mappings and case tables from
  [Zenodo record 15263706](https://zenodo.org/records/15263706).
- Bouckaert tree MCC summaries from
  [v1.0.0 and v2.0.0 releases](https://github.com/rbouckaert/global-language-tree-pipeline/releases).
- The root MGN CSV, canonical language mapping, and Glottolog lookup already in
  this repository. Their hashes are recorded without altering them.

`remote_zip.py` is a source-specific downloader using checked HTTP byte ranges.
It reads the ZIP64 end record and the first/last parts of the central directory,
which contain the required metadata and Q matrices. Its inventory explicitly
records that partial scope; it is not a complete archive inventory. Each selected
member is checked against its ZIP CRC and size. The 19 Q matrices can be refreshed
from the pinned archive with:

```sh
python3 analyses/gelato_feasibility_2026_10_01/remote_zip.py --output analyses/gelato_feasibility_2026_10_01/sources/zenodo_15263706 --ancestry-k-range
```

Other files were selected with repeated `--member` arguments matching their archived
paths. The downloader is intentionally pinned to this archive's URL and size.
The snapshots contain published ancestry inference outputs and metadata, not raw
genotype files. Tree coverage checks inspect taxon labels; posterior-tree analyses
and ancestral-state reconstructions are not implemented here.

**Note (2026-10-08).** The repository was reorganised. Three inputs of `audit.py`
(`mgn_modeling_dataset.csv`, `mgn_language_map.json`,
`data_sources/Glottolog_lookup_table_Heti_edition.tsv`) were moved to
the private archive (see the local `ARCHIVE_NOTE.md`). To re-run the audit, copy them back to the
repository root. The outputs here are unchanged and their input hashes are in
`outputs/provenance.json`.
