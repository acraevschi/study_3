# Track 2 — Academic Open-Corpus Token-Frequency & Paradigm-Matching Pipeline

This directory implements the reproducible, zero-hardcoding token-frequency acquisition and paradigm-matching layer for **Study 3, Track 2 (Morphological Paradigm Predictability)** (`analyses/paradigm/frequencies/`).

## Overview & Purpose

Grammar induction in Track 2 (MGL and neural models) is kept strictly **type-uniform** ($N_{\text{train}} = 120$ and $480$), while this pre-tabulated usage-frequency layer supports two key downstream analyses without downloading multi-terabyte raw web crawls:
1. **Dictionary-Skew Control:** Constructing `analyses/paradigm/data/splits_freq/{unit_id}_topfreq120.csv` ($R=5$ draws $\times$ $K=5$ folds of $150$ groups from the top $\min(500, n_{\text{attested}})$ corpus-attested leakage groups whenever $\ge 200$ distinct `leakage_group_id`s are attested).
2. **Within-Language Entrenchment & Contact Interaction:** Providing lemma-level usage frequencies (`log10_freq_lemma`, `freq_lemma_pmw`, `freq_lemma_tokens`) for hierarchical `Admixture × Token Frequency` irregularity-entrenchment regressions (`analyses/paradigm/stage2/fit_entrenchment.R`).

## Three-Tier Academic Source Hierarchy

All frequency assets are programmatically fetched, verified via SHA-256 checksums, and logged in [`freq_provenance.csv`](../data/freq_provenance.csv):

1. **Tier A — Universal Dependencies v2.15 (`tier_a_ud_v2.15`):**
   - Official GitHub repositories (`https://github.com/UniversalDependencies/UD_*`) pinned at release tag `r2.15` with dereferenced 40-character commit SHAs and archive SHA-256 checksums (`Zeman et al. 2024`, LINDAT/CLARIAH-CZ `http://hdl.handle.net/11234/1-5787`).
   - Provides lemmatized, UPOS-disambiguated `(lemma, upos, form)` token counts (`VERB ∪ AUX` for `V`; `NOUN` for `N`).
2. **Tier B — Leipzig Corpora Collection (`tier_b_leipzig_corpora`):**
   - Official static corpus tarballs (`https://downloads.wortschatz-leipzig.de/corpora/{corpus_id}.tar.gz`, `CC-BY-4.0`, `Goldhahn, Eckart, & Quasthoff 2012, LREC`), prioritizing multi-register `news`/`web` + `wikipedia`/`community` releases (`300K` / `100K` / `1M`, with `30K` / `10K` / `community_2017` fallback for minority languages).
   - Verifies upstream `.tar.gz` SHA-256 (`archive_sha256`) in memory and extracts **only** `*-words.txt` and `*-meta.txt` / `*-import.sql` (`sha256` verified on disk), discarding raw sentence/co-occurrence tables to save disk space.
3. **Tier C — `wordfreq` v3.1.1 (`tier_c_wordfreq`):**
   - Multi-register smoothed frequency wordlists (`Speer 2022`, Zenodo DOI `10.5281/zenodo.7199437`, `CC-BY-SA-4.0 / Apache-2.0`), hashed directly from `.venv` msgpack data packs. Supplements paradigm cells (e.g., 1st/2nd-person conversational inflections from OpenSubtitles/dialogue registers) that happen to fall at zero count in finite news/Wikipedia samples.

## Scripts & Reproducible Execution

Always run via `.venv/bin/python` (strictly capped at $\le 2$ threads):

```bash
# 1. Fetch and SHA-256 verify UD v2.15, Leipzig Corpora wordlists, and wordfreq packs
.venv/bin/python -m analyses.paradigm.frequencies.fetch_frequencies --workers 2

# 2. Match surface forms onto orthographic paradigms, run UD + 5-step EM homograph disambiguation,
#    and build splits_freq/{unit_id}_topfreq120.csv manifests
.venv/bin/python -m analyses.paradigm.frequencies.match_paradigms

# 3. Run unit tests
.venv/bin/python -m pytest tests/test_paradigm_frequencies.py -v
```

## Output Schemas

- `analyses/paradigm/data/freq_provenance.csv`:
  `source_id,source_tier,iso639_3,corpus_id,url,version_or_commit,license,citation,local_path,sha256,archive_sha256,n_tokens,n_types`
- `analyses/paradigm/data/frequencies/{unit_id}_freq.csv`:
  `unit_id,lemma_id,leakage_group_id,freq_lemma_tokens,freq_lemma_pmw,log10_freq_lemma,n_cells_attested,primary_freq_source`
- `analyses/paradigm/data/frequencies/freq_summary.csv`:
  Per-unit coverage, corpus token denominator, UD vs. EM homograph counts, and `topfreq120` eligibility.
- `analyses/paradigm/data/splits_freq/{unit_id}_topfreq120.csv`:
  $R=5$ draws $\times$ $K=5$ folds of $150$ distinct `leakage_group_id`s ($120$ `train` = $96$ `subtrain` + $24$ `dev`; $30$ `test`), mirroring `analyses/paradigm/data/splits/{unit_id}_uniform120.csv` 1-to-1.
