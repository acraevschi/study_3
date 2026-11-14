# Track 2 — Morphological Paradigm Predictability (*i-Complexity*) (`analyses/paradigm/`)

End-to-end reproducible pipeline evaluating **integrative morphological complexity** (*i-complexity*: Ackerman & Malouf 2013; Cotterell et al. 2018; Wu et al. 2019)—the cross-validated conditional cross-entropy (in bits) of predicting inflected wordforms across paradigms—and its relationship with population-genetic admixture and lexical token frequency across **56 GeLaTo-linked languages** (**106 paradigm units**, **12 linguistic families**).

---

## 1. Directory Layout

| Path | Description |
| :--- | :--- |
| [`ingestion/`](ingestion/) | Programmatic acquisition (`fetch_sources.py`), NFC/orthographic hygiene, syncretism deduplication, 3-rule Union-Find root-cognate grouping (`grouping.py`), Malouf (2017) vowel-harmony pre-abstraction (`harmony_malouf.py`), and deterministic SHA-256 CV splits (`splits.py`). |
| [`frequencies/`](frequencies/) | Open-corpus token-frequency acquisition (`fetch_frequencies.py`: UD v2.15 `@r2.15`, Leipzig Corpora Collection static archives, `wordfreq` v3.1.1), script-pure UD-POS + 5-step EM homograph matching (`match_paradigms.py`), and `topfreq120` splits. |
| [`mgl/`](mgl/) | Zero-language-tuned **Multi-Locus Autosegmental Vowel-Tier Minimal Generalization Learner** (Albright & Hayes 2002/2003 Mikheev $\pi_L(0.75)$ + residue impugnment $\pi_U(0.95)$ + proper Dirichlet escape over $\Sigma^*$) and Chu–Liu–Edmonds / $k=3$ PoE paradigm topology evaluator (`evaluate_mgl.py`). |
| [`neural/`](neural/) | **3-Seed Hard-Monotonic Neural Transducer** (Wu & Cotterell 2019 exact anti-diagonal wavefront forward DP over $\Sigma^*$ + batched string-marginal beam search on `mps`/`cuda`/`cpu`). |
| [`stage2/`](stage2/) | Separate 1-row-per-`glottocode` Verb (`V`) & Noun (`N`) data assembly (`prepare_stage2.py`), precision-weighted phylogenetic/spatial Berkson Stage-2 `brms` regressions (`fit_stage2.R`), and two-level token-frequency entrenchment models (`fit_entrenchment.R`). |
| [`data/`](data/) | Cryptographic provenance manifests (`provenance_manifest.csv`, `freq_provenance.csv`), `unit_summary.csv`, `frequencies/freq_summary.csv`, `cell_maps/` (`*_cells.csv`, `*_sounds.csv`), plus gitignored `lexicons/`, `lexicons_malouf/`, `splits/`, `frequencies/*_freq.csv`, `splits_freq/`. |
| [`results/`](results/) | Tracked unit-level MGL (`results/mgl/unit_outcomes_*.csv`) & Neural (`results/neural/unit_outcomes_uniform120.csv`) outcomes and Stage-2 / Entrenchment posterior summary CSVs (`results/stage2/*.csv`). |
| [`figures/`](figures/) | Publication-grade cross-linguistic, validation, specification-curve, family-trajectory, and entrenchment figures (`fig1..5_mgl_*.{png,pdf}`). |

---

## 2. End-to-End Reproduction Commands

All Python scripts run via `.venv/bin/python` with explicit single-thread BLAS/OpenMP caps:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# 1. Fetch & SHA-256-verify paradigm sources + build lexicons, Malouf tables, and CV splits
.venv/bin/python analyses/paradigm/ingestion/fetch_sources.py --workers 4
.venv/bin/python analyses/paradigm/ingestion/build_lexicons.py

# 2. Fetch & SHA-256-verify open token-frequency corpora + match onto paradigms & build topfreq120 splits
.venv/bin/python analyses/paradigm/frequencies/fetch_frequencies.py --workers 4
.venv/bin/python analyses/paradigm/frequencies/match_paradigms.py

# 3. Evaluate Multi-Locus Vowel-Tier MGL across all 4 regimes (<= 6 workers)
.venv/bin/python -m analyses.paradigm.mgl.evaluate_mgl --regime uniform120 --max-workers 6
.venv/bin/python -m analyses.paradigm.mgl.evaluate_mgl --regime uniform120 --malouf --model-name malouf --max-workers 6
.venv/bin/python -m analyses.paradigm.mgl.evaluate_mgl --regime topfreq120 --max-workers 6
.venv/bin/python -m analyses.paradigm.mgl.evaluate_mgl --regime uniform480 --max-workers 6

# 4. Evaluate 3-seed Hard-Monotonic Neural Transducer on curated 12-unit typological validation set
TORCH_NUM_THREADS=2 .venv/bin/python -m analyses.paradigm.neural.train_eval \
  --regime uniform120 --device mps --num-threads 2 \
  --units stan1288_V_orth,esto1258_V_orth,nucl1301_N_orth,ital1282_V_orth,stan1290_V_orth,finn1318_V_orth,russ1263_N_orth,poli1260_V_orth,stan1295_V_orth,kore1280_V_orth,stan1293_V_orth,sout1528_N_orth

# 5. Prepare Stage-2 datasets & run Bayesian Berkson cross-linguistic + two-level entrenchment models
.venv/bin/python analyses/paradigm/stage2/prepare_stage2.py
PARA_CORES=4 Rscript analyses/paradigm/stage2/fit_stage2.R --pos BOTH
PARA_CORES=4 Rscript analyses/paradigm/stage2/fit_entrenchment.R --pos BOTH

# 6. Run full automated test suite (43 Track-2 tests + repository regression suite)
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

### 3.2 Stage-2 Cross-Linguistic Admixture Regressions (`results/stage2/stage2_results_{V,N}.csv`)

| POS Sample | Exposure | Primary `full_covs` (Phylogenetic + GP) $\beta$ [95% CrI] | $\text{BF}_{01}$ (Null) | ROPE `[-0.15, +0.15]` | $P(\beta < 0.44)$ vs. Track 1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Verbs (`V`, $n=45$)** | `ent` (Admixture Entropy) | `+0.147 [-0.134, +0.427]` | `2.07` | `0.484` | `0.980` |
| **Verbs (`V`, $n=45$)** | `ent` (`+ z_log_C_distinct`) | `+0.075 [-0.250, +0.381]` | `2.76` | `0.591` | `0.991` |
| **Verbs (`V`, $n=32$)** | `f3k_est_pos` ($f_3$-Admixture) | `-0.211 [-0.438, +0.075]` | `0.99` (`4.22` w/ $\|C\|$) | `0.335` (`0.800` w/ $\|C\|$) | `1.000` |
| **Nouns (`N`, $n=49$)** | `ent` (Admixture Entropy) | `+0.106 [-0.231, +0.452]` | `2.39` | `0.527` | `0.971` |
| **Nouns (`N`, $n=49$)** | `sqrt_het` (Heterozygosity) | `-0.181 [-0.509, +0.154]` | `1.66` | `0.425` | `1.000` |
| **Nouns (`N`, $n=37$)** | `f3k_est_pos` ($f_3$-Admixture) | `-0.045 [-0.351, +0.275]` | `3.00` | `0.631` | `0.998` |

### 3.3 Two-Level Token-Frequency Entrenchment (`results/stage2/entrenchment_results_{V,N}.csv`)

| POS Sample | Languages / Lemmas | Within-Language Frequency Slope $b_{\text{freq}}$ [95% CrI] | $P(b_{\text{freq}} > 0)$ | Cross-Level `Admixture × Freq` ($b_{\text{int}}$, `ent`) | $\text{BF}_{01}$ (`b_int = 0`) | ROPE (`b_int`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Verbs (`V`)** | `44` / `21,078` | `+0.093 [+0.051, +0.134]` SD | `1.0000` ($\text{BF}_{10} = 651$) | `+0.016 [-0.026, +0.058]` | `17.80` (`13.2–23.2`) | `1.000` |
| **Nouns (`N`)** | `46` / `26,956` | `+0.030 [+0.013, +0.046]` SD | `0.9994` | `-0.006 [-0.023, +0.011]` | `46.20` (`26.7–50.1`) | `1.000` |
