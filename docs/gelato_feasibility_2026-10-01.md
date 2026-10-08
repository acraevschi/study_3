# GeLaTo, inflectional predictability, and historical contact: feasibility review

Audit date: 1 October 2026. This reviews a proposed design; it does not adopt a new
design or change the existing modelling pipeline. The Low/High German component
remains outside the present scope.

**Assessment.** Genetic evidence can identify some episodes of demographic contact
and can complement the present morphology study. GeLaTo is not a ready-made measure
of historical adult L2 participation. Its public repository supplies genetic
distances and demographic summaries, while a separate 2025 research archive supplies
inferred ancestry proportions and curated contact cases. The current MGN sample
overlaps the former for 32 languages. The proposed historical simplification-jump
test needs additional temporal and population alignment evidence.

**What the genetic terms mean.**

| Term | Meaning for this project |
|---|---|
| SNP | A DNA position at which people can carry different variants. The Human Origins panel measures hundreds of thousands of these positions. |
| Autosomal | Refers to the chromosomes other than the sex chromosomes. These data draw ancestry information from many lines of descent. |
| Genetic admixture or gene flow | Ancestry contributed by previously differentiated populations through reproduction. A historical demographic process. |
| Genetic drift | Random changes in variant frequencies across generations, especially consequential in small populations. Drift can make a population distinct without implying an absence of earlier contact. |
| Ancestry component | A statistical pattern of DNA-variant frequencies inferred by a model. Components are not intrinsically named historical peoples or language groups. |
| ADMIXTURE and K | ADMIXTURE estimates each sampled individual's proportions in K inferred components. Choosing a different K changes the resolution and often the component definitions. |
| Q matrix | The ADMIXTURE result: one individual per row, one component per column, with proportions summing approximately to one. |
| FST | Differentiation in DNA-variant frequencies between populations. Low differentiation can reflect shared ancestry or gene flow; it is not itself an admixture percentage. |
| f3 admixture statistic | A test involving a target and two reference populations. A sufficiently negative value supports admixture under its assumptions. A nonnegative or nonsignificant value does not establish its absence. |
| IBD | Identity by descent: shared stretches of DNA inherited from a common ancestor. The length and distribution of such stretches can inform demographic history. |
| Ne | Effective population size: a model-based measure of the size of a population's reproducing ancestry. It is not a census population or speaker count. |

For the genetic interpretation, see [Korunes and Goldberg (2021)](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009374),
[Patterson et al. (2012)](https://academic.oup.com/genetics/article/192/3/1065/5935193),
and [GeLaTo's variable documentation](https://github.com/gelato-org/gelato-data/blob/c625fdcf0225142cc03ae3a1635edf322e9c7778/GeneticVariable.md).

**Why a native/non-native genetic split needs reformulation.** A language has no
intrinsically native DNA component. To interpret a contribution as ancestry acquired
through contact, one must specify a target population, reference populations, and
a historical period. The largest component is not automatically the ancestry of the
people who introduced the language, and a minor component is not automatically
ancestry from L2 learners. Language shift can occur with little gene flow; gene flow
can occur while speakers retain their languages. The GeLaTo founding study explicitly
examines these mismatches. [Barbieri et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9704691/)

A hypothetical 70/30 ancestry profile could follow recent intermarriage, an older
event predating the relevant language lineage, or a language shift in either
direction. Those scenarios imply different linguistic histories. Consequently,
ancestry proportions can support a historically interpreted demographic-contact
measure, but their magnitude alone does not measure contact duration, frequency,
adult-learning pressure, or the direction of language transmission.

The sampling model also matters. In the downloaded published runs, the largest
component in the Finnish population ranges from **50.0% to 86.4%** across K=12..30.
These are different model resolutions applied to the same eight individuals, not
observed changes through time. Calling one minus the largest proportion
"non-native ancestry" would therefore introduce an unvalidated and highly
model-dependent definition. [Calculated K sensitivity](../analyses/gelato_feasibility_2026_10_01/outputs/ancestry_K_sensitivity.csv)
See also [Lawson, van Dorp and Falush (2018)](https://www.nature.com/articles/s41467-018-05257-7)
on interpretation of ancestry clusters.

**What is available, and how it can be extracted.**

The pinned [GeLaTo repository](https://github.com/gelato-org/gelato-data/tree/c625fdcf0225142cc03ae3a1635edf322e9c7778)
has 397 Human Origins populations, 4,030 individuals, and 295 distinct assigned
Glottocodes. Some codes designate broader linguistic groupings, so this last number
is an identifier count rather than an assertion that every entry represents an
equally defined language community. Its regenerated CLDF export contains one active
panel. An older STR panel is present in the source folders but is not part of that
CLDF export; it should not be silently pooled with the SNP panel.

Relevant files are `cldf/populations.csv`, `cldf/variables.csv`, and
`datasets/HumanOrigins_AutosomalSNP/data.csv`. Population metadata include sample
size, coordinates, publication, linguistic identity, and curation notes. Available
measurements include global, regional, and neighbouring-population FST summaries,
Ne, and pairwise distances. The repository's `data_pairwise.csv` holds pairwise
FST, geographic distance, and approximate divergence times. Its documentation
describes a rough split-time calculation from Ne and linearized FST. **That
divergence time is not an admixture date or a date of language contact.**

The public repository does not contain the ancestry Q matrices needed for the
supervisor's proposed percentage measure. For those, use the separate
[Graff et al. 2025 archive](https://zenodo.org/records/15263706).
It includes published ADMIXTURE runs, individual-row metadata, population-language
mappings, a candidate-contact list, and a curated final contact list.
The associated [Science Advances article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12396315/)
uses genetic admixture to study **structural convergence**, which establishes a
methodological precedent rather than evidence that contact simplifies paradigms.

Extraction is already demonstrated locally:

1. Read the Q matrices for K=12..30 and `GeneticInfoID.csv`.
2. Check each matrix's row count, component count, bounds, and row sums. Check that
   metadata `Order` is sequential and that population counts match Table S1.
3. Match rows by their documented order, then average component proportions within
   each sampled population, following the archived analysis script.
4. Attach the curated Table S1 population-language mapping. Do not use the older
   individual file's Glottocode without checking it: the archive corrects several
   assignments, including Mala, Bergamo, and Saami.
5. Keep population identities separate, and join MGN by Glottocode. Record a
   descendant-language mapping or linguistic proxy as a distinct mapping status.

All 19 matrices passed the structural checks: 4,768 individuals and 558 population
means per K. Table S1 has 373 distinct nonmissing assigned Glottocodes. The 653-row
auxiliary mapping file is a wider lookup, not the set of 558 genetically analysed
populations. The script uses Table S1 as the population denominator.

The resulting [K=23 population profiles](../analyses/gelato_feasibility_2026_10_01/outputs/population_ancestry_K23_components.csv)
and [multi-K diagnostics](../analyses/gelato_feasibility_2026_10_01/outputs/population_ancestry_K12_K30_diagnostics.csv)
are descriptive extraction outputs. No component has been assigned "native" status
and no contact predictor has been approved. Component column numbers are local to
each K; identical column numbers across K do not identify the same ancestry.

**Coverage with the current MGN modelling sample.** These counts use the actual
111,315-row, 64-language modelling dataset and its existing language identifiers.
Verbs, nouns, and adjectives refer to languages with those MGN subsystems.

| Join rule | Languages | Genetic populations | Individuals | Verbs | Nouns | Adjectives |
|---|---:|---:|---:|---:|---:|---:|
| Main GeLaTo panel: identical base Glottocode | 32 | 62 | 585 | 20 | 22 | 11 |
| Expanded archive: identical base Glottocode, all sample sizes | 35 | 80 | 643 | 22 | 24 | 11 |
| Expanded archive: identical base Glottocode, minimum 5 individuals per population | 32 | 62 | 585 | 20 | 22 | 11 |
| Main panel: exact matches plus screened descendant/Greek candidates | 35 | 75 | 692 | 23 | 24 | 13 |
| Expanded archive: exact matches plus screened candidates, all sample sizes | 37 | 100 | 771 | 24 | 26 | 13 |
| Expanded archive: exact matches plus screened candidates, minimum 5 individuals | 35 | 75 | 692 | 23 | 24 | 13 |

The final three rows are catalogue-screening upper bounds. They have not passed
morphological-resource/community alignment review. The minimum-five rule is a
sensitivity screen motivated by the original GeLaTo panel, not a claim that five
individuals are sufficient for every ancestry inference.

The exact 32-language overlap consists of:

| Family | Matched languages |
|---|---|
| Indo-European, 21 | Belarusian, Bulgarian, Catalan, Czech, Eastern Armenian, Eastern Yiddish, English, French, Galician, German, Icelandic, Irish, Italian, Lithuanian, Northern Kurdish, Norwegian, Polish, Romanian, Russian, Ukrainian, Western Farsi |
| Turkic, 4 | Bashkir, North Azerbaijani, Tatar, Turkish |
| Uralic, 3 | Estonian, Finnish, Hungarian |
| Abkhaz-Adyge, 2 | Kabardian, West Circassian |
| Kartvelian, 1 | Georgian |
| Dravidian, 1 | Telugu |

The main exact overlap retains 57,790 MGN cell-pair records, but only 32 independent
language-level predictor units. Every matched language is classified as Eurasian
in the current MGN dataset. Thus the join reduces geographical breadth: it does
not repair the global sample's imbalance.

The expanded archive adds exact identifiers for Spanish, North Saami, and Zulu,
but the relevant directly matched populations have only four, three, and one
individuals respectively. Spanish can also be linked through sampled descendant
varieties with larger samples. The main-panel candidate additions are Spanish
(Castilian/Murcian varieties), Serbian-Croatian-Bosnian (Croatian), and Modern Greek
(Greek populations assigned to the broader `gree1276` grouping, with published
GBI/TLI mappings to `mode1248`). No family-wide mapping is otherwise used.

Several exact identifiers still require substantive scrutiny. Mala is one Telugu
community, Ashkenazi Jewish is an ancestry-defined sample mapped to Eastern Yiddish,
and English is represented by Cornwall and Kent. Bergamo and Tuscan are mapped to
Italian. Neither a sample label nor an exact Glottocode establishes that sampled
people spoke the same variety represented by the morphological resource, or that
they represent the entire language's population. Jewish samples cannot be assigned
to Modern Hebrew by ethnicity. Generic Albanian and Pashto groupings do not
automatically identify MGN's Northern Tosk Albanian or Southern Pashto; Standard
Arabic cannot be treated as a genetically homogeneous vernacular community.

The complete [language coverage ledger](../analyses/gelato_feasibility_2026_10_01/outputs/mgn_language_coverage.csv)
and [population crosswalk](../analyses/gelato_feasibility_2026_10_01/outputs/population_crosswalk.csv)
preserve both matches and nonmatches.

**How much coverage already has interpreted contact evidence?**

The 2025 final list contains five contact rows whose genetic target population has
an exact MGN language match, representing four target languages. These are existing
published case assignments, not independently reanalysed genetic events.

| MGN target | Archived case | Evidence in the archive |
|---|---|---|
| Russian | Khanty-related contact | Literature-derived case; no f3 result in this table |
| Finnish | Contact associated with an Indo-European source | Genetically inferred case; minimum reported f3 Z = -9.92 |
| Bulgarian | Contact assigned to a West Oghuz source clade | Literature-derived case; no f3 result in this table |
| Zulu | Two cases, with Naro- and Taa-related proxies | One target individual; minimum reported f3 Z = -1.21 and -1.31, respectively |

The Finnish result supports admixture in the archived analysis. It does not identify
a specific historical donor language, establish a contact date, or show morphological
simplification. Modern reference populations approximate past genetic sources.

Two further published linguistic-proxy assignments reach MGN: a Hazara target
represented by Western Farsi, and a literature-derived Greek target represented
by Modern Greek. The Hazara-to-Western-Farsi substitution requires particular
scrutiny. Counting these gives six target languages, not a validated six-language
comparison. See the [contact-case ledger](../analyses/gelato_feasibility_2026_10_01/outputs/published_contact_case_overlap.csv).

The uncurated longlist intersects MGN for 22 target languages and 379 candidate
rows. It is a search space, not 379 established contacts. An absent final pair also
cannot be coded as "no contact": the original contact search selects particular
two-source, cross-family scenarios and depends on available reference populations.
Its screening thresholds require multi-K consistency and subsequent source
interpretation. Structural threshold replication alone does not validate an event.

**The exact global language tree.**

The supervisor's references identify the Bouckaert et al. global tree. The
[Nature Human Behaviour article](https://doi.org/10.1038/s41562-025-02325-z)
reports using the 2022 tree's 902 posterior draws, downsampled to 100. The
[2023 Grambank paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10115409/)
also uses this global analysis. The current
[release page](https://github.com/rbouckaert/global-language-tree-pipeline/releases)
distinguishes v1.0.0, with 6,635 tips, from v2.0.0, with 6,636 tips and associated
posterior trees. The later version accompanies the
[updated diversification preprint](https://osf.io/preprints/socarxiv/f8tr6_v2).

Both downloaded maximum-clade-credibility (MCC) summary trees contain **all 64
current MGN Glottocodes**. Tip-identifier coverage therefore imposes no further
loss on the joins above. This verifies the taxon labels, not every node's historical
interpretation. The local downloads are summary trees; posterior sets have not
been downloaded or used for fitting.

The tree is a constrained global reconstruction incorporating family
classifications, independent family analyses and timing information, geography,
and evidence about human expansion. It is more informative than the project's
current unit-length Glottolog classification tree, but deep relationships and
dates depend on those inputs. A complete version-specific audit of calibrations
and priors was not possible from the accessible materials. Geography and genetic
history used in the scaffold also motivate sensitivities that do not rely on the
same global scaffold to assess genetic contact.

For an association analysis, map the harmonized language-level morphology results
and population-aligned predictors to tips and propagate posterior-tree uncertainty.
Check results against within-family trees or a covariance with unrelated families
treated separately, as well as geographical controls. Preserve the complete tree
before pruning: branch lengths should not become counts of the remaining sample.

**Can the proposed historical jumps be tested?** Ancestral-state reconstruction can
infer earlier morphological states under an explicit evolutionary model. It does
not turn modern endpoints into observed changes. With only a few dozen matched
languages, inferred branch-specific jumps will be particularly sensitive to trait
measurement, tree uncertainty, and the assumed evolutionary process.

There is a second problem: genetic admixture is not vertically inherited along
a language tree in the same way as linguistic descent. Gene flow connects
populations across branches, and language shift can change the population linked
to a lineage. A modern ancestry proportion placed on a language tip cannot simply
be reconstructed as that language's ancestral ancestry proportion using Brownian
motion. Correlating two such reconstructed changes could mainly correlate the
models' assumptions.

The desired event-level estimand would be something like: **Following a dated
demographic-contact event on a specified lineage, did a comparable morphological
subsystem become more predictable than its pre-contact state and an appropriate
comparison lineage?** That requires earlier and later matched paradigms, comparable
learner settings and representation, independently interpreted ancestry sources,
event timing, and an account of continuity or language shift. Genetic dating based
on ancestry-block or linkage-disequilibrium patterns may help in selected cases;
GeLaTo's rough FST split dates do not supply that evidence.

Present-day MGN accuracy is not itself a historical simplification rate. Increased
predictability would be simplification under this measure, but the loss of cells,
borrowing, or increased word length can change other dimensions of morphology
independently. A cross-sectional association must therefore be described as such.
The nine historical language entries in upstream MGN are useful leads, but do not
give matching ancient genetic observations or guarantee measurement comparability.

**Recommended next design decision.**

Keep the paradigm-based outcome. Treat GeLaTo as a candidate additional evidence
stream whose exposure variable must be defined and validated. Two routes merit
consideration:

1. **An exploratory present-day association:** ask whether inflectional
   predictability covaries with a validated, population-aligned measure of
   ancestry mixing or genetic differentiation. Plan around roughly 32 exact
   language identifiers, or at most 35 screened candidates with populations of
   at least five individuals, before community exclusions. FST or ancestry-profile
   heterogeneity can be exploratory descriptors, but should not be relabelled
   adult-L2 pressure. For a genuinely admixture-based predictor, extend independent
   source/event validation beyond the small existing final case list.
2. **A smaller historical study:** select documented contact histories with
   independently supported admixture and dated morphological evidence. Genetics
   would corroborate the demographic exposure; the morphological outcome would
   be measured before and after contact in comparable paradigms. This is the
   route that most directly addresses simplification following contact.

For route 1, predefine the exposure before examining its morphological association.
An event-specific ancestry proportion is interpretable only relative to specified
sources and a time window. A validated contact category can be used when percentages
are not identifiable, but the comparison group must have its own evidence audit;
"not selected" is not "unadmixed". The 2025 two-component criteria are a useful
starting protocol, not a universal measure of all contact.

For an explicitly specified donor, the candidate quantity is the sum of the target
population's mean Q proportions over components independently assigned to that
donor. The assignment must hold under reasonable alternative references and K
values, and the historical evidence must distinguish donor ancestry from shared
older ancestry. This sum cannot be recovered merely by subtracting the largest
component from one. Where identifiable, source-specific modelling of ancestry
proportions (for example with qpAdm) and independent admixture dating would provide
a stronger basis; they require additional genetic work and assumptions beyond
the current summary-table join.

Keep one demographic predictor unit per independently matched language/community.
Multiple genetic populations mapped to the same MGN morphology cannot create
independent morphology outcomes. Select samples representing the morphology's
community where possible; otherwise report population ranges and sensitivity to
alternative aggregation. Averaging individuals estimates the sampled panel, not
automatically the ancestry profile of the worldwide speech community. Retain
genetic sampling uncertainty and K/source-model sensitivity separately from
uncertainty in the morphology learner.

Before fitting, repair the inherited methodological interpretation of `nvar`:
the present `mgn_data/functions.R` counts variable slots with
`str_count(x$analogy, "<")`; it does not count competing inflection classes.
The existing fits also need their convergence problems resolved. No new fit was
run for this audit, and the present Methods document was not rewritten.

**Reproducibility.** The
[audit directory](../analyses/gelato_feasibility_2026_10_01/README.md)
contains pinned source snapshots, a coverage/extraction script, outputs, and SHA-256
input provenance. The Zenodo archive is approximately 11.6 GB; selective HTTP range
extraction retrieved the metadata and published Q matrices without downloading the
whole archive or raw genomes. The audit demonstrates extraction and overlap, not
the substantive validity of an admixture-intensity variable.
