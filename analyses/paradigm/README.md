# Track 2 — Morphological Paradigm Predictability (*i-Complexity*) (`analyses/paradigm/`)

End-to-end reproducible pipeline evaluating **integrative morphological complexity** (*i-complexity*: Ackerman & Malouf 2013; Cotterell et al. 2018; Wu et al. 2019)—the cross-validated conditional cross-entropy (in bits) of predicting inflected wordforms across paradigms—and its relationship with fixed-reference cross-family population-genetic admixture (`f3_fixed`) and lexical token frequency across **56 GeLaTo-linked languages** (**106 paradigm units**, **12 linguistic families**).

---

## 1. Directory Layout

| Path | Description |
| :--- | :--- |
| [`ingestion/`](ingestion/) | Programmatic acquisition (`fetch_sources.py`), NFC/orthographic hygiene, syncretism deduplication, 3-rule Union-Find root-cognate grouping (`grouping.py`), Malouf (2017) vowel-harmony pre-abstraction (`harmony_malouf.py`), and deterministic SHA-256 CV splits (`splits.py`). |
| [`frequencies/`](frequencies/) | Open-corpus token-frequency acquisition (`fetch_frequencies.py`: UD v2.15 `@r2.15`, Leipzig Corpora Collection static archives, `wordfreq` v3.1.1), script-pure UD-POS + 5-step EM homograph matching (`match_paradigms.py`), and `topfreq120` splits. |
| [`mgl/`](mgl/) | Zero-language-tuned **Multi-Locus Autosegmental Vowel-Tier Minimal Generalization Learner** (Albright & Hayes 2002/2003 Mikheev $\pi_L(0.75)$ + residue impugnment $\pi_U(0.95)$ + proper Dirichlet escape over $\Sigma^*$) and Chu–Liu–Edmonds / $k=3$ PoE paradigm topology evaluator (`evaluate_mgl.py`). |
| [`neural/`](neural/) | **3-Seed Hard-Monotonic Neural Transducer** (Wu & Cotterell 2019 exact anti-diagonal wavefront forward DP over $\Sigma^*$ + batched string-marginal beam search on `mps`/`cuda`/`cpu`). |
| [`stage2/`](stage2/) | Separate 1-row-per-`glottocode` Verb (`V`) & Noun (`N`) data assembly (`prepare_stage2.py`), precision-weighted phylogenetic/spatial Berkson Stage-2 `f3_fixed` regressions (`fit_stage2.R`), and two-level token-frequency entrenchment models (`fit_entrenchment.R`). |
| [`data/`](data/) | Cryptographic provenance manifests (`provenance_manifest.csv`, `freq_provenance.csv`), `unit_summary.csv`, `frequencies/freq_summary.csv`, `cell_maps/` (`*_cells.csv`, `*_sounds.csv`), plus gitignored `lexicons/`, `lexicons_malouf/`, `splits/`, `frequencies/*_freq.csv`, `splits_freq/`. |
| [`results/`](results/) | Tracked unit-level MGL (`results/mgl/unit_outcomes_*.csv`) & Neural (`results/neural/unit_outcomes_uniform120.csv`) outcomes and Stage-2 / Entrenchment / Figure-fallback summary CSVs (`results/stage2/*.csv`). |
| [`figures/`](figures/) | Publication-grade cross-linguistic, validation, specification-curve, family-trajectory, entrenchment, and maximal-model anatomy figures (`fig1..6_mgl_*.{png,pdf}`; see [`figures/README.md`](figures/README.md)). |

---

## 2. End-to-End Reproduction Commands

All Python and R scripts run with explicit single-thread BLAS/OpenMP caps:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# 0. Clone pinned external repositories (external/glottolog-cldf, external/grambank, external/languages-of-the-world)
bash scripts/fetch_external.sh

# 1. Fetch & SHA-256-verify paradigm sources + build lexicons, Malouf tables, and CV splits
.venv/bin/python analyses/paradigm/ingestion/fetch_sources.py --workers 4
.venv/bin/python analyses/paradigm/ingestion/build_lexicons.py

# 2. Fetch & SHA-256-verify open token-frequency corpora + match onto paradigms & build topfreq120 splits
.venv/bin/python analyses/paradigm/frequencies/fetch_frequencies.py --workers 4
.venv/bin/python analyses/paradigm/frequencies/match_paradigms.py

# 3. Evaluate Multi-Locus Vowel-Tier MGL across all 4 regimes
.venv/bin/python -m analyses.paradigm.mgl.evaluate_mgl --regime uniform120 --max-workers 6
.venv/bin/python -m analyses.paradigm.mgl.evaluate_mgl --regime uniform120 --malouf --model-name malouf --max-workers 6
.venv/bin/python -m analyses.paradigm.mgl.evaluate_mgl --regime topfreq120 --max-workers 6
.venv/bin/python -m analyses.paradigm.mgl.evaluate_mgl --regime uniform480 --max-workers 6

# 4. Evaluate 3-seed Hard-Monotonic Neural Transducer on curated 12-unit typological validation set
TORCH_NUM_THREADS=2 .venv/bin/python -m analyses.paradigm.neural.train_eval \
  --regime uniform120 --device mps --num-threads 2 \
  --units stan1288_V_orth,esto1258_V_orth,nucl1301_N_orth,ital1282_V_orth,stan1290_V_orth,finn1318_V_orth,russ1263_N_orth,poli1260_V_orth,stan1295_V_orth,kore1280_V_orth,stan1293_V_orth,sout1528_N_orth

# 5. Prepare Stage-2 datasets, run dry-run validation, and fit Bayesian Berkson + entrenchment models
.venv/bin/python analyses/paradigm/stage2/prepare_stage2.py
PARA_DRY=1 PARA_POS=V,N Rscript analyses/paradigm/stage2/fit_stage2.R
ENT_DRY=1 ENT_POS=V,N Rscript analyses/paradigm/stage2/fit_entrenchment.R
PARA_CORES=4 PARA_POS=V,N Rscript analyses/paradigm/stage2/fit_stage2.R
ENT_CORES=4 ENT_POS=V,N Rscript analyses/paradigm/stage2/fit_entrenchment.R

# 6. Render publication figures (Figs 1–6; works standalone from tracked CSVs via FORCE_CSV_FALLBACK=1)
for s in analyses/paradigm/figures/make_fig*.R; do
  FORCE_CSV_FALLBACK=1 Rscript "$s"
done

# 7. Run automated repository test suite (Track 2, Track 1, common, and morph_ldl tests)
.venv/bin/python -m pytest -m "not slow"
```

---

## 3. Headline Empirical Summary Tables

### 3.1 Multi-Regime Paradigm Predictability & Validation (`results/mgl/` & `results/neural/`)

| Regime / Model | Units | Held-Out Lemmas | Median $H_{\text{tree}}$ (IQR) [bits] | Median $\text{Rel}_{\text{tree}}$ | Median $\text{Acc}_{\text{tree}}$ | Key Validation Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **MGL `uniform120`** (Primary) | `106` (`94` orth + `12` phon) | `58,896` | `0.5924` (`0.3050–1.3759`) | `0.9479` | `0.9652` | Baseline cross-linguistic *i-complexity* across 56 languages / 12 families |
| **MGL `uniform120_malouf`** | `33` (`29` orth + `4` phon) | `18,655` | `0.3679` (`0.2065–0.8222`) | `0.9630` | `0.9740` | $r = 0.9879$ ($0.9917$ `orth`) vs. raw Vowel-Tier MGL |
| **MGL `topfreq120`** | `74` (`74` orth) | `28,454` | `0.7635` (`0.3362–1.5743`) | `0.9332` | `0.9532` | $r = 0.9801$ vs. `uniform120`; higher $H_{\text{tree}}$ in `70.3%` (`52/74`) of units |
| **MGL `uniform480`** | `73` (`61` orth + `12` phon) | `83,588` | `0.4351` (`0.2235–0.7441`) | `0.9709` | `0.9774` | $r = 0.9375$ vs. `uniform120`; lower $H_{\text{tree}}$ in `90.4%` (`66/73`) of units |
| **Neural `uniform120`** (3-seed) | `12` (`9 V + 3 N` orth) | `8,419` | `0.5952` (`0.4009–1.1366`) | — | `0.9055` | $r = 0.8543$, Spearman $\rho = 0.8811$ vs. Vowel-Tier MGL |

### 3.2 Stage-2 Cross-Linguistic `f3_fixed` Regressions (`results/stage2/stage2_results_{V,N}.csv`)

Standardized posterior slopes per 1 SD of cross-family `f3_fixed` under the maximal Phylogenetic + 3D spherical GP specification (`y | se(se_y, sigma = TRUE) ~ 1 + x + z_log_C_distinct + z_log_navail + source_prov + z_pop + z_area + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)`, `Normal(0, 0.5)` slope prior, ROPE `[-0.15, +0.15]`):

| POS | Specification (`f3_fixed`) | $N$ | Posterior Slope $\beta$ [95% CrI] | $\text{BF}_{01}$ (Null) | ROPE `[-0.15, +0.15]` | $P(\beta < 0.44)$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Verbs (`V`)** | **`maximal`** (`main`, $n_A \ge 3$) | `42` | `-0.157 [-0.333, +0.072]` | `1.51` | `0.429` | `1.000` |
| **Verbs (`V`)** | `mundlak_wb` (within-family) | `42` | `-0.197 [-0.392, +0.016]` | `0.76` | `0.300` | `1.000` |
| **Verbs (`V`)** | `cutoff_n5` (`main`, $n_A \ge 5$) | `41` | `-0.171 [-0.354, +0.056]` | `1.26` | `0.384` | `1.000` |
| **Verbs (`V`)** | `strict_links` (`strict`, $n_A \ge 3$) | `37` | `-0.058 [-0.343, +0.209]` | `3.28` | `0.694` | `1.000` |
| **Verbs (`V`)** | `tier2_uniform480` ($H_{\text{tree}, 480}$) | `22` | `+0.005 [-0.184, +0.228]` | `5.02` | `0.877` | `0.999` |
| **Verbs (`V`)** | `topfreq120` ($H_{\text{tree}, \text{topfreq}}$) | `31` | `+0.156 [-0.143, +0.436]` | `1.94` | `0.454` | `0.976` |
| **Nouns (`N`)** | **`maximal`** (`main`, $n_A \ge 3$) | `47` | `+0.085 [-0.231, +0.412]` | `2.66` | `0.576` | `0.983` |
| **Nouns (`N`)** | `mundlak_wb` (within-family) | `47` | `+0.062 [-0.299, +0.414]` | `2.64` | `0.583` | `0.982` |
| **Nouns (`N`)** | `cutoff_n5` (`main`, $n_A \ge 5$) | `46` | `+0.114 [-0.220, +0.438]` | `2.38` | `0.524` | `0.976` |
| **Nouns (`N`)** | `strict_links` (`strict`, $n_A \ge 3$) | `42` | `+0.073 [-0.212, +0.369]` | `3.01` | `0.645` | `0.992` |
| **Nouns (`N`)** | `tier2_uniform480` ($H_{\text{tree}, 480}$) | `36` | `+0.179 [-0.172, +0.485]` | `1.69` | `0.380` | `0.952` |
| **Nouns (`N`)** | `topfreq120` ($H_{\text{tree}, \text{topfreq}}$) | `40` | `-0.058 [-0.465, +0.354]` | `2.30` | `0.516` | `0.991` |

*Across all fitted $H_{\text{tree}}$ models, `26/27` Verb and `26/26` Noun `95%` credible intervals include zero (see [`stage2/README.md`](stage2/README.md) for the full Block 1 / Block 2 grid and sampler diagnostics).*

### 3.3 Two-Level Token-Frequency Entrenchment (`results/stage2/entrenchment_results_{V,N}.csv`, `all_lemmas`, `f3_fixed`, $n_A \ge 3$)

| POS Sample | Languages / Lemmas | Within-Language Frequency Slope $b_{\text{freq}}$ [95% CrI] | $P(b_{\text{freq}} > 0)$ | Cross-Level `f3_fixed × Freq` ($b_{\text{int}}$) | $\text{BF}_{01}$ (`b_int = 0`) | ROPE (`b_int`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Verbs (`V`)** | `42` / `19,784` | `+0.095 [+0.050, +0.140]` SD | `1.0000` ($\text{BF}_{10} = 322$) | `-0.002 [-0.044, +0.041]` | `22.40` | `1.000` |
| **Nouns (`N`)** | `44` / `26,060` | `+0.031 [+0.014, +0.048]` SD | `0.9996` ($\text{BF}_{10} = 11.05$) | `-0.002 [-0.019, +0.016]` | `55.35` | `1.000` |
