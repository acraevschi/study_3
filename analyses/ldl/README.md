# Track 2: GeLaTo Genetic Admixture $\to$ LDL / Wug Paradigm Predictability & Confidence

This directory contains **Track 2** of PhD Study 3: linking population-genetic admixture in
GeLaTo to paradigm cell predictability and production confidence estimated by Linear
Discriminative Learning (`JudiLing.jl`) on the held-out-lemma Wug / Paradigm Cell Filling
Problem (PCFP).

- [`ablation/`](ablation/): Meaning-space, cell-specific semantic vector $V(\text{cell})$, and
  tolerant path-decoder ablation on the `pcfp_v2` tuning sample (see [`ablation/README.md`](ablation/README.md)).

---

## Connecting LDL / Wug Outcomes to `analyses/common/`

The pipeline (`morph_ldl`) writes language-level LDL outcomes and GeLaTo crosswalks under
`outputs/<experiment_id>/`:

- `outputs/<experiment_id>/outcomes/ldl_outcomes.csv`: held-out PCFP exact-match accuracy,
  normalized Levenshtein similarity, rank-1 candidate confidence, and lemma-cluster bootstrap SEs
  keyed by `unit_id` and `glottocode`.
- `outputs/<experiment_id>/gelato/gelato_links.csv` (or `typology/analysis_links.csv`):
  accepted `(population, glottocode)` links between GeLaTo populations and the experiment's
  languages.

Because [`analyses/common/`](../common/) is completely language-set-agnostic, Stage-2 LDL/Wug
admixture models reuse the exact same genetics, demography, phylogeny, and Stage-1 exposure
machinery as Track 1 (`analyses/grambank/`):

### 1. Python: Prepare Admixture, Demography & Phylogeny for Any LDL Language Set

```python
from pathlib import Path
import pandas as pd

from analyses.common.demography import resolve_demography_df
from analyses.common.genetics import (
    compute_individual_measures,
    compute_population_measures,
    load_admixture_q,
)
from analyses.common.phylogeny import (
    build_newick,
    resolve_glottolog_coords,
    resolve_glottolog_paths,
)

# 1. Load your LDL experiment's (population, glottocode) links
links_df = pd.DataFrame([
    {"population": "Bergamo", "glottocode": "ital1282"},
    {"population": "Tuscan", "glottocode": "ital1282"},
    {"population": "Finnish", "glottocode": "finn1318"},
    {"population": "Russian", "glottocode": "russ1263"},
    {"population": "French_Central", "glottocode": "stan1290"},
])
glottocodes = sorted(links_df["glottocode"].unique())

# 2. Compute individual-level (ent, nonmax) and population-level (het, sqrt_het, f3k_est_pos, Ne)
info, Q = load_admixture_q()
ind_df = compute_individual_measures(info, Q, links_df=links_df)
pop_df, pooled_sd = compute_population_measures(
    info,
    Q,
    links_df["population"].unique(),
    include_extended_metadata=True,
    warn_missing_f3=False,
)
pop_df = pop_df.merge(links_df, on="population")

# 3. Resolve audited L1 speaker population & range area (Bromham / Koplenig / Ethnologue)
demog_df = resolve_demography_df(glottocodes)

# 4. Build Glottolog coordinates and topology Newick tree
coords_df = resolve_glottolog_coords(glottocodes)
paths, tree_src = resolve_glottolog_paths(glottocodes)
nwk_str = build_newick(paths)
```

### 2. R: Compute Grafen / Bouckaert Phylogenetic Covariance & Stage-1 Exposure

```r
source("analyses/common/phylogeny.R")
source("analyses/common/stage1_exposure.R")

# 1. Phylogenetic correlation matrices for any vector of LDL Glottocodes
A_grafen    <- build_grafen_corr("ldl_phylo_tree.nwk", ldl_langs$glottocode)
A_bouckaert <- build_bouckaert_corr(ldl_langs$glottocode)

# 2. Fit Stage-1 hierarchical exposure (individual/population -> language eta_mean & eta_sd)
pop_df <- augment_population_f3_columns(pop_df, k5_file = "analyses/common/data/f3_scores_k5.csv")
s1_ent <- fit_stage1_exposure("ent", ind_df = ind_df, pop_df = pop_df, set_name = "ldl_main")
s1_het <- fit_stage1_exposure("sqrt_het", ind_df = ind_df, pop_df = pop_df, set_name = "ldl_main")
s1_f3  <- fit_stage1_exposure("f3k_est_pos", ind_df = ind_df, pop_df = pop_df, set_name = "ldl_main")

# 3. Merge Stage-1 exposure (eta_mean, eta_sd) with LDL outcome table for Stage-2 brms / Stan modeling
#    e.g. y_ldl | se(y_se, sigma = TRUE) ~ 1 + me(x, x_sd) + z_pop + (1 | gr(lang, cov = A_bouckaert)) + gp(sx, sy, sz)
```
