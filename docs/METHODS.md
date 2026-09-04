# Methods and design decisions

*PhD Study 3. Last revised 2026-09-04, after the first full model run.*

This is the single reference document for the study: what it asks, where the numbers
come from, which choices shape them, what the models found, and what should not be
claimed from them. Repository layout and how to run things are in
[../README.md](../README.md).

---

## 1. The question

**Do the demographic properties of a speech community predict the inflectional
complexity of its language?**

- **Complexity** is *I-complexity* in the sense of Ackerman & Malouf (2013): how
  predictable one inflected form is from another. It is measured as in Guzmán Naranjo
  (2024), who reports, for every ordered pair of paradigm cells in a language, how
  often a classifier trained on cell A predicts the correct form in cell B. One
  observation is one **directed cell pair**: `correct` successes out of `total`
  cross-validation trials.
- **Demography** is L1 speaker population, plus macro-ecological covariates (contact
  richness, range area, terrain roughness) and, where available, the L2 proportion.

Two things follow from this operationalisation and are easy to forget. Accuracy is a
*proportion of trials*, so the response is binomial-shaped, not continuous. And it is
comparable across cell pairs only after controlling for how many inflection classes
the classifier is choosing among — the chance baseline — which is what `nvar` does.

The Low/High German diachronic arm is a separate track that inherits the demographic
registry built here but has no model yet.

---

## 2. Data sources

Full citations and download locations are in [../README.md](../README.md#data). In
outline, four sources are fused into one language-level registry:

| Tier | Source | Supplies | Coverage |
|---|---|---|---|
| 1 | **Bromham et al. (2022)** | `population_l1`, and the **only** source of contact richness, range area, altitude range and roughness | 6,511 |
| 1b | **Ethnologue multi-ISO table** (`LMP_POP1`) | `population_l1` for the major world languages Bromham omits | 1,061 |
| 2 | **Koplenig (2019) / Ethnologue** | `l2_proportion`, `vehicularity`; fallback population and area | 2,143 |
| 3 | **Glottolog v5.3** | glottocodes, families, macro-areas, coordinates, genealogical lineages | ~8,000 |

**Resulting registry: 7,872 languages, 6,272 with a real population.** Provenance:
Bromham 6,164 · Ethnologue multi-ISO 60 · Koplenig 43 · pluricentric aggregate 3 ·
manual 2. **1,600 languages have no population and keep `NaN`.**

**Why tier 1b exists.** Bromham omits English, Spanish, Russian, French, Portuguese,
Hindi, Urdu, Croatian and others. `LMP_POP1` in the multi-ISO table is the *same* L1
variable as Bromham's `lang_L1.POP_lang` — verified identical for `deu`, `als`, `azj`,
`pbt`, `pes` — so using it keeps population on one definition rather than mixing
Ethnologue editions.

**Are Bromham and Koplenig commensurable?** Yes. Over the 2,071 languages both cover,
the median ratio of Koplenig `Population` to Bromham `lang_L1.POP_lang` is exactly
**1.00**, with only 9.9% differing by more than 2×. Both are L1 counts derived from
Ethnologue. Area agrees equally well (median ratio 1.0000 over 2,058 languages).

---

## 3. Decisions that shape the numbers

These are the choices a reader should know about before trusting any coefficient.
Each one is a place where a defensible-looking alternative would quietly change the
results, so the reasoning is given rather than just the rule.

### 3.1 No population is ever invented

Languages with no population in any source keep `population_l1 = NaN` and are filtered
out of the modelling dataset. They stay in the registry because their taxonomy and
coordinates remain useful.

A silent default fill would be invisible downstream while anchoring the low end of the
study's primary predictor, so a test fails if any single population value accounts for
≥5% of the registry.

### 3.2 Macro-ecological covariates are measured or flagged, never invented

Bromham's measurements are authoritative and never overwritten. Where a language has
no polygon in any source, the four covariates are imputed by **family → macro-area →
global median** and the row is flagged `eco_imputed`.

**53 of the 64 modelled languages have measured covariates.** The 11 imputed are
`arb, eng, fra, hbs, hin, lav, por, rus, spa, urd, ydd` — which are, unhelpfully, among
the highest-population languages in the sample. `MGN_DROP_ECO_IMPUTED=1` refits on the
53 measured languages as a sensitivity check.

Two guards matter here, because country-level proxies are a tempting and badly wrong
substitute: a nation's territory is not a language's range area, and a nation's highest
mountain is not a language's altitude range — both are an order of magnitude outside the
observed distributions (Bromham's altitude range tops out at 4,858 m). So Bromham's
values are read straight through, a test checks they are reproduced exactly, and every
imputed value is flagged rather than blended in silently.

### 3.3 Chatino disambiguation

In ISO 639-3, `yai` is **Yaghnobi** (Iranian, Tajikistan) and `zen` is **Zenaga**
(Berber, Mauritania). In MGN they are Oaxaca Chatino fieldwork corpora. Glottolog
treats Yaitepec and San Juan Quiahije Chatino as dialects of **Western Highland
Chatino** (`ctp`/`west2644`), which is the language-level unit carrying data.

| MGN | → ISO | Glottocode | Name |
|---|---|---|---|
| `yai` | `ctp` | `west2644` | Western Highland Chatino |
| `zen` | `czn` | `zenz1235` | Zenzontepec Chatino |

This is independently validated: with `yai → ctp`, the two least-predictable languages
in the assembled dataset are `ctp` (0.325) and `nav` (0.328) — exactly the two the
paper's abstract names as the clear exceptions. A test asserts it.

### 3.4 One canonical language map

The MGN → ISO mapping lives only in `src/mgn_language_map.py`, which emits
`mgn_language_map.json` for the R scripts and is imported by the test suite. Keeping it
in one place is not tidiness: the Python pipeline, the R scripts and the tests must agree
on what `hbs`, `lav`, `nob` and `pus` mean, and a second copy that drifts changes which
languages enter the model. `hbs` and `lav` stay as themselves because Glottolog carries
Serbian-Croatian-Bosnian and Latvian as language-level units, while `hrv`/`srp`/`bos`
and `lvs` are only dialects with no record to attach to.

### 3.5 Pluricentric standards

Three MGN datasets describe one shared standard spanning several ISO varieties, so L1
population is summed over members and recorded in `population_scope`:

| Anchor | Members | Population |
|---|---|---|
| `hbs` | hrv + srp + bos | 19,982,910 |
| `als` | als + aln | 8,258,000 |
| `arb` | 9 Arabic vernaculars | 126,035,000 |

**Arabic is a lower bound.** Modern Standard Arabic has no native speakers by
definition, so the relevant community is the aggregate of vernacular speakers, but
Koplenig covers only 9 of ~30 Ethnologue Arabic varieties. The true figure is nearer
300M — about 0.4 dex higher on the log10 scale. Arabic contributes 5,886 trials.

### 3.6 Two manual population values, both unverified

**These are the only two numbers in the pipeline that do not come from a data file.**

| ISO | Language | Value | Status |
|---|---|---|---|
| `lav` | Latvian | 1,500,000 | **Unverified** |
| `ydd` | Eastern Yiddish | 600,000 | **Unverified** |

Both were entered by hand during the 2026-09-03 audit from recalled Ethnologue figures.
Neither was read from, or checked against, any primary source. They are round numbers of the right order of magnitude, not citations.
Latvian is absent from all three population sources; Bromham carries only Western
Yiddish (`yih`, 5,000), a different variety from the Eastern Yiddish in MGN. Without
them both languages drop out.

**Before publication, look both up and replace them with a cited figure and edition.**
Both carry `population_source = "Manual_Ethnologue"`, so one filter isolates every
affected row; together they contribute 114 of 111,315 trials.

### 3.7 Training-set size, and rescuing Navajo

MGN reports each cell pair at `num` ∈ {200, 500, 1000, 2000, 5000} training lexemes.
Accuracy rises with `num`, so it must be held constant across languages or complexity
is confounded with corpus availability. We prefer **`num == 200`**.

Per `(language, POS)`, we use `num == 200` where available, else the **smallest
available** size, recorded in `num_used`. This recovers Navajo verbs and Belarusian
verbs, both of which exist only at 500. **108,747 of 111,315 rows (97.7%) are at 200.**

Navajo is why the fallback exists: it is one of the two headline high-complexity
languages in the paper, and a hard `num == 200` filter drops it entirely. Language
coverage is therefore asserted on the **modelling dataset**, not on the registry — the
registry can be complete while the data actually fitted is not.

### 3.8 Cell-label normalisation

MGN cell labels use several conventions inherited from their origin corpora. Before any
distance between them is meaningful they must be put on a common footing.
`src/cell_normalization.py` handles six, in order: the curated map in
`cells_to_unimorph.json`; Navajo (`FUT.3apl:IPA` → `{FUT, 4, PL}`); native UniMorph;
the Polish colon format; uncurated dotted labels; and bare single features.

Anything else is **dropped and reported**, not turned into an opaque token that would
give every pair involving it the same meaningless distance. This reduced unparseable
trials from 2,831 to **116** (6 idiosyncratic labels).

**POS tags.** Curated labels expand with an explicit POS feature; native UniMorph labels
have none. Comparing the two added a spurious **+1** to the distance for 1,872 pairs
across 15 languages (English was 57% tagged / 43% untagged). Bare POS features are now
stripped — POS is already a column — while sub-POS tags carrying real morphosyntax
(`V.PTCP`, `V.CVB`, `V.MSDR`) are kept.

**Distance** is the symmetric set difference of the two normalised feature bundles,
|F₁ △ F₂|, an integer in 1…14.

### 3.9 Chance level

Raw accuracy is not comparable across cell pairs: two inflection classes give a 0.5
baseline, twenty give 0.05. MGN's `nvar` tracks this and correlates **−0.32** with
accuracy. `log10_nvar_z` therefore enters both models as a control, and it turns out to be the best-identified effect in the fit (§5).

### 3.10 Standardisation

Language-level covariates are standardised over the **64 languages**, not the 111,315
trials, because trial counts per language range from 6 to ~12,000 and trial-level
standardisation would let paradigm size weight the centring. The two trial-level
covariates (`log10_nvar_z`, `log10_paradigm_size_z`) are standardised over whichever
rows are actually fitted, so they are recomputed when the data are aggregated (§4).

### 3.11 Phylogenetic covariance

Glottolog lineages are assembled into an explicit tree with unit branch lengths per
classification step, and `ape::vcv.phylo(corr = TRUE)` gives the standard Brownian-motion
correlation matrix. Nodes are keyed by full path so identically-named subgroups in
different families never merge, and a missing lineage raises rather than silently
grouping languages under a shared root.

Result: 64×64, λ_min = 0.083, condition number 164. Chatino 0.756, Turkic 0.667, Slavic
0.577, Germanic 0.417–0.456, cross-family exactly 0.

Using a genuine BM covariance rather than an ad-hoc similarity score matters for
interpretation. Normalising shared lineage levels by the deepest lineage *in the sample*
rescales every entry whenever a language is added; it gives the variance component no
evolutionary reading; and it leaves a gap between the largest off-diagonal and a diagonal
of 1 that bakes in independent variance inseparable from the phylogenetic part.

---

## 4. Observation unit: cell pairs vs. source cells

The dataset is 111,315 directed cell pairs. The models can be fitted on those directly
(`MGN_UNIT=pair`) or on **source cells** (`MGN_UNIT=source_cell`), which is what the
reported run uses.

Source-cell aggregation groups on (language, POS, source cell) and sums `correct` and
`total` over every target cell, giving **3,175 observations** — a median of 24 targets
collapsed per row. Pair-level morphological measures (`nvar`, `nph`, `nmarkers`) become
trial-weighted means.

Two consequences:

- **It reduces pseudo-replication.** At the pair level, `hye` contributes 11,967 rows and
  `czn` 6; the top five languages are 47% of the data while the effective N for the
  population effect is 64.
- **`mo(distance)` cannot survive it.** A source cell reaches targets a median of 6
  distinct distances away (max 13), so every aggregated row mixes all of them. The term
  is dropped under this unit rather than fed a meaningless average. This is the one
  thing the source-cell model cannot say anything about.

Summing rather than averaging is deliberate. Taking the *median accuracy rate* per
source cell and modelling it as a beta proportion — the obvious alternative — discards
the trial counts entirely, so a source cell tested on 12,000 trials would weigh the same
as one tested on 60.

---

## 5. The models and what they found

### Specification

```r
correct | trials(total) ~ log10_pop_z * pos
                        + log10_paradigm_size_z * pos
                        + log10_nvar_z
                        + mo(distance)                      # pair unit only
                        + (1 | iso_sanitized)               # independent language variance
                        + (1 | gr(phylo, cov = A))          # phylogenetically structured
                        + gp(geo_x, geo_y, geo_z, k = 15, c = 5/4, gr = TRUE)
family = beta_binomial(link = "logit", link_phi = "log")
```

Model 2 adds `contact_richness_scaled + log10_area_scaled + roughness_scaled`.

Why each piece:

- **beta-binomial, not binomial.** MGN's per-pair accuracy is the *best of three*
  edge-weighting configurations under cross-validation, not an i.i.d. Bernoulli sequence.
  A plain binomial is badly overdispersed, making the interval on the population effect
  far too narrow — the most likely route to a false positive. The fitted overdispersion
  (φ ≈ 20) confirms this was necessary.
- **both random effects.** With only the phylogenetic term, all residual between-language
  variation is forced through the phylogenetic covariance, inflating apparent
  phylogenetic signal. `phylo` is a duplicate column of the language factor because brms
  refuses two group-level terms naming the same factor.
- **`mo(distance)`, not linear.** Distance is an ordinal count with no reason to act
  linearly on the logit scale.
- **GP on 3-D unit-sphere coordinates.** A degree of longitude is 111 km at the equator
  and 55 km at 60°N, and raw degrees do not wrap at the antimeridian. `gr = TRUE` groups
  by the 64 unique locations rather than fitting over all rows.
- **`altitude_range_scaled` dropped** from Model 2 (decision of 2026-09-03), reducing but
  not removing the imputed-covariate load.

### Run of 2026-09-04

Source-cell unit, 3,175 observations, 64 languages. 4 chains × 3000 iterations
(1500 warmup), `adapt_delta = 0.99`, 4 threads per chain. About 35 minutes per model.

**Convergence — both models fall short of the R̂ < 1.01 / ESS > 400 target:**

| | Model 1 | Model 2 |
|---|---|---|
| max R̂ | 1.042 | 1.015 |
| min bulk ESS | 118 | 218 |
| divergences | 52 / 6000 | 60 / 6000 |
| max-treedepth hits | 0 | 1498 / 6000 |

The failure is **localised**: 9 of 3,521 parameters in Model 1. The GP length-scale
(R̂ 1.042, ESS 132, posterior [0.017, 0.820]) and the phylogenetic SD (R̂ 1.015,
posterior [0.019, 0.914]) are both weakly identified and pressed against zero. Every
population-level coefficient is fine, with R̂ ≤ 1.013 and bulk ESS 1,200–10,500.

The cause is structural: three terms compete for the same between-language variance
across an effective N of 64 — the independent language effect (SD 0.60), the
phylogenetic effect (SD 0.40) and the spatial GP (amplitude 1.71). **The fixed effects
below are usable; the decomposition of variance into phylogenetic, spatial and
independent components is not.**

**Population effect** (slope of `log10_pop_z`, reference POS = adjectives):

| POS | mean | 95% CI | P(slope > 0) |
|---|---|---|---|
| adj | +0.208 | [−0.070, +0.495] | 0.93 |
| n | −0.162 | [−0.403, +0.077] | 0.09 |
| v | +0.019 | [−0.209, +0.250] | 0.57 |

No credible population effect within any part of speech. But the **contrast between
them is credible**: the noun-vs-adjective interaction is −0.370 [−0.561, −0.174],
excluding zero. The finding is a difference in how population relates to predictability
across parts of speech, not a main effect of population.

**Other terms.** `log10_nvar_z` = −0.282 [−0.319, −0.246] with bulk ESS 10,565: the
chance-level control is by a wide margin the best-identified effect in the model, and
omitting it would leave the population terms to absorb it. φ = 20.1 [19.0, 21.2].

**Model comparison.** elpd_diff = **−0.33 (SE 0.43)** — Model 2 is indistinguishable
from Model 1. All three macro-ecological covariates straddle zero: contact richness
−0.185 [−0.410, +0.030], log area −0.014 [−0.373, +0.348], roughness +0.060
[−0.180, +0.296]. No bad Pareto-k values, so the comparison itself is reliable. Given
that Model 2's covariates are imputed for 11 of 64 languages, this is a convenient
outcome: nothing rests on the imputed values.

### Before these numbers go anywhere

The convergence problem is one of identification, not sampler tuning — raising
`adapt_delta` further would suppress the divergences while leaving the ridge, which is
worse. In order of expected value: put an informative prior on the GP length-scale
(brms's default is diffuse and the posterior runs into zero); then decide whether 64
languages can support all three language-level terms at once.

---

## 6. Limitations to state in the write-up

- **Trial imbalance**, even after aggregation. Source cells per language range from 3 to
  212.
- **Ceiling effect.** 12.3% of cell pairs sit at accuracy exactly 1.0.
- **Arabic population is a lower bound** (§3.5), and two populations are unverified (§3.6).
- **Classification depth is not phylogeny.** Glottolog subclassification depth partly
  reflects research attention, so finely-subclassified families (Indo-European, 43 of the
  64 languages) get systematically higher covariances than shallowly-classified ones.
- **`l2_proportion` covers only 26 of 64 languages**, so the L2 arm of the question
  cannot yet be answered.
- **Coverage is heavily Eurasian**: 60 of 64 languages, with the Americas contributing
  Navajo and two Chatino languages and Africa only Zulu. See `plots/map_mgn64.png`.
- **The variance decomposition is not identified** in this run (§5).

---

## 7. What the tests check

`python3 -m pytest tests/ -q` → **96 passed** with the models fitted, or **92 passed,
4 skipped** on a fresh clone: the four skips check `fits/*.rds`, which is not committed
(163 MB per fit).

Tiers 1–4 check that artefacts exist and have the right shape. **Tier 5** checks that the
numbers in them are real. Each of its 30 tests targets a failure mode that a purely
structural test cannot see:

- no constant population imputation; provenance labels name sources actually loaded;
  population and provenance co-occur; manual overrides minimal and declared
- macro-ecological covariates inside their source distribution; Bromham values reproduced
  exactly; `eco_imputed` set honestly
- all six cell conventions parse correctly; unparseable labels rejected rather than made
  opaque
- 64/64 coverage **on the modelling dataset**, not the registry; Navajo present via the
  `num` fallback; training size constant within language; chance-level control present
- the phylogenetic matrix is a valid Brownian-motion correlation matrix with sane
  relatedness ordering
- **the paper's headline finding is reproduced** — Navajo and Western Highland Chatino
  are the two least predictable languages — an end-to-end check on the language mapping,
  the `num` fallback and the merge

`tests/conftest.py` imports the language inventory from `src/mgn_language_map.py` rather
than restating it, so a test cannot validate against a mapping the pipeline has stopped
using.
