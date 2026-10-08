# MGN paradigm data (third-party, read-only)

Source: the data repository accompanying Guzmán Naranjo, Matías (2024). *An analogical
approach to the typology of inflectional complexity.* Journal of Language Modelling 12(2).
<https://jlm.ipipan.waw.pl/>

Only the parts the pipeline reads are kept here. They are not committed to git (317 MB);
`MANIFEST.sha256` (committed) lists the SHA-256 of every kept file, so a fresh copy can be
checked against the one used for `outputs/pilot_v1`.

| Path | Size | Read by | Content |
|---|---|---|---|
| `data/` | 167 MB | `data` stage (adapters, registry) | per-language paradigm tables built by MGN from UniMorph (plus Polish from PoliMorf) |
| `data-custom/` | 150 MB | `data` stage | MGN's custom phonological paradigm tables (English, French, Hungarian, Russian, ...) |
| `build-data/` | 60 KB | `data` stage (`morph_ldl/data/provenance.py`) | MGN's R build scripts, parsed to record how each file was derived (orthography vs epitran, cell filters) |

Everything else from the upstream repository (model results `results-*`, TiMBL runs, R
analysis scripts, about 8.4 GB) was moved to the
private archive on 2026-10-08 (see the local `ARCHIVE_NOTE.md`). The pipeline does
not use MGN accuracy results, complexity scores or predictions.

To restore on a new machine: obtain the upstream MGN data repository, copy `data/`,
`data-custom/` and `build-data/` here, and run
`shasum -a 256 -c MANIFEST.sha256` from this directory.

Upstream URL and commit: not recorded in this repository. Add them here when known.
