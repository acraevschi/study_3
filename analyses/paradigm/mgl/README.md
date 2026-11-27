# Multi-Locus Vowel-Tier Minimal Generalization Learner (`analyses/paradigm/mgl/`)

Primary morphological predictability learner for **Study 3 — Track 2 (Paradigm Predictability)**.

## Module Overview

| Module | Responsibility |
| :--- | :--- |
| [`vowel_tier.py`](vowel_tier.py) | Multi-script Unicode vowel classification (Latin/IPA, Cyrillic incl. Turkic/Uralic `ә ө ү і ӱ ӧ ы э я ю е ё`, Greek, Armenian, Georgian, Hebrew/Arabic vocalics, Devanagari/Indic matras, Hangul jamo), Sukhotin's (1962) unsupervised co-occurrence vowel discovery fallback (for unknown scripts and cross-linguistic glide disambiguation `y`/`w`), and `{unit_id}_sounds.csv` binary feature bundles (`phon` mode). Enforces strict source-only scansion (`w_s` only; zero target `w_t` leakage). |
| [`align.py`](align.py) | Symmetric global Needleman–Wunsch sequence aligner between `# x_1 ... x_m #` and `# y_1 ... y_n #` (`cost(x==y)=0`, `cost(V,V')=cost(C,C')=1.0`, `cost(V,C)=2.5`, `indel=1.1`) with contiguous-locus tie-breaking (`_ -> _t`, `_ -> ge_t`, `_e_o -> _ie_e`). |
| [`mgl_learner.py`](mgl_learner.py) | C-speed Python `int` bitmask Minimal Generalization Learner (`scope_mask`, `hit_mask`) with globally fixed Mikheev lower bound $\alpha = 0.75$ (one-sided Student's $t_{0.75, n-1}$) and Albright & Hayes (2002 §3.7) residue impugnment $\alpha_{\text{upper}} = 0.95$, generalizing in parallel over local slot environments and source-only Vowel-Tier contexts. |
| [`escape.py`](escape.py) | Within-candidate `max` rule competition ($\text{Score}(y \mid w_s) \in [0, 1]$), bounded parameter-free reliability (**Metric B**), and strictly proper generative cross-entropy over $\Sigma^*$ (**Metric A**) via Dirichlet escape ($\gamma = 0.05$) into an exact forward-DP memoryless stochastic character-edit model $P_{\text{edit}}(y \mid w_s)$. |
| [`topology.py`](topology.py) | Exact recursive Chu–Liu–Edmonds minimum-cost directed spanning arborescence $T_{r^*}$ (`H_tree`) selected strictly on inner `subtrain -> dev` cross-entropy $\hat{M}[s, t]$, syncretism-aware `shape_id` equivalence-class normalization ($|s_l| - 1$), fast plug-in conditional entropy parent pre-screening for $|C_{\text{distinct}}| > 40$, and dynamic $k = 1, 3$ Product-of-Experts PCFP ($\tau = 0.1$) with Hölder upper partition normalization over $\Sigma^*$. |
| [`evaluate_mgl.py`](evaluate_mgl.py) | CLI & multiprocessing runner enforcing `OMP_NUM_THREADS=1`, strict `leakage_group_id` disjointness assertions, full-`train` refitting after inner-split topology selection, and 500-resample `leakage_group_id` cluster bootstrap SEs. |

## CLI Usage

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m analyses.paradigm.mgl.evaluate_mgl \
  --lexicons-dir analyses/paradigm/data/lexicons \
  --splits-dir analyses/paradigm/data/splits \
  --results-dir analyses/paradigm/results/mgl \
  --regime uniform120 \
  --max-workers 6
```

## Produced Artifacts (`analyses/paradigm/results/mgl/`)

- `unit_outcomes_uniform120.csv` (`106` units) & `lemma_outcomes_uniform120.csv` (`58,896` lemmas)
- `unit_outcomes_uniform120_malouf.csv` (`33` units) & `lemma_outcomes_uniform120_malouf.csv` (`18,655` lemmas)
- `unit_outcomes_topfreq120.csv` (`74` units) & `lemma_outcomes_topfreq120.csv` (`28,454` lemmas)
- `unit_outcomes_uniform480.csv` (`73` units) & `lemma_outcomes_uniform480.csv` (`83,588` lemmas)
