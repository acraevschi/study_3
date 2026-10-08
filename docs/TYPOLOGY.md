# Grambank inflection-extent outcome (`typology` stage)

Owner: typology component (`morph_ldl/typology/`). Status: declared 2026-10-08, before
any linkage output of the stage was written. §1–§8 are the declaration; §9 is the
proposed analysis protocol (not fitted); §10 records the first development run.

The stage gives a second, language-level outcome next to the LDL predictability outcome:
the extent of bound inflectional morphology from Grambank, for every language reached
from a GeLaTo population. It never reads ancestry values (§7).

## 1. Sources (pinned; recorded in `typology/stage_manifest.json`)

| source | pin | files read |
|---|---|---|
| Grambank CLDF (`grambank/grambank`) | tag `v1.0.3`, commit `7ae000cf740f93cdb3e4ec67010668d6795337a9` | `cldf/values.csv`, `cldf/parameters.csv`, `cldf/codes.csv`, `cldf/languages.csv` |
| Glottolog CLDF (`glottolog/glottolog-cldf`) | tag `v5.3`, commit `072ca0d0410039fb8b779be8fc165bac575d2cda` | `cldf/languages.csv` (Level, Language_ID, Family_ID, Macroarea); `cldf/values.csv`, only the `classification` rows (for group map-down) |
| GeLaTo main_397 | gelato-data commit `c625fdc…` | `cldf/populations.csv` |
| GeLaTo expanded_558 | Zenodo 15263706 | `tables/tableS1.csv` |

`scripts/fetch_external.sh` clones Grambank and Glottolog at the tags into
`external/grambank` and `external/glottolog-cldf` (shallow clones; `external/` is
gitignored). The stage checks that the clone's `HEAD` equals the pinned commit when the
directory is a git checkout, and fails on a mismatch. The manifest records, per source,
the tag, the declared and resolved commit, and the sha256 of every file read.

The GeLaTo files are the same cached files as the data-stage crosswalk
(`morph_ldl/data/gelato.py`), but the typology stage reads only the two population
metadata tables. It does not use `gelato.load_sources`, because that function also opens
`GeneticInfoID.csv` and the derived K23/diagnostic tables (with key-only `usecols`). It
also does not read the audit's `population_crosswalk.csv` (it carries FST/Ne values) or
the Zenodo population mapping (not needed).

## 2. Unit of observation and mapping rules

One outcome row per **language-level Glottocode** (Glottolog 5.3 `Level == language`),
column `glottocode`. This is the same key as the LDL outcome table (`glottocode`, the
language-level Glottocode of the unit), so the two tables join one-to-one.

Each GeLaTo population has its *own* Glottocode: the main-panel `Glottocode` and/or the
Table S1 `GeLaTo_Glottocode` (in the cached files the two never disagree for the 397
shared populations; if they did, both would be resolved and the conflict reported).
It is resolved with Glottolog 5.3:

| own code level | rule | `link_basis` | status |
|---|---|---|---|
| language | the code itself | `exact` | `candidate` |
| dialect | Glottolog `Language_ID` | `dialect_rollup` | `candidate` |
| family/group | the language-level descendants (Glottolog `classification`) that have a language-level Grambank entry: exactly one → that language | `group_map_down` | `candidate` |
| family/group, several such descendants | no automatic match; candidates listed | `group_ambiguous` | `ambiguous` (no `glottocode`) |
| family/group, no such descendant | no automatic match | `group_no_grambank_descendant` | `unmatched` (no `glottocode`) |
| `NA`, empty, or not in Glottolog 5.3 | none | `unresolved` | `unmatched` |
| config `typology.manual_links` | as declared | `manual` | `candidate` |

Table S1 `GBI_Glottocode` / `TLI_Glottocode` (database proxies) are resolved by the same
rules. A proxy whose language-level code differs from the population's own
language-level code becomes a separate link row with `link_basis` `proxy_gbi`,
`proxy_tli` or `proxy_gbi_tli`, `is_proxy = True` and status `ambiguous`, as in the
data-stage crosswalk. A proxy that resolves to the population's own language adds
nothing.

Roll-up and map-down are initial matches, never acceptance. Statuses follow contract §9
and the data-stage crosswalk: overrides only come from `gelato_review.yaml` (keys
`glottocode` = language-level code, `population`; `resources: all` entries apply). An
`accepted` review takes effect only with a human `confirmed_by` + `confirmed_date`;
otherwise the link stays `candidate` with `proposed_status = accepted`. The current
reviews therefore give `proposed_status = accepted` for Tuscan (ital1282), Finnish
(finn1318) and French_Central (stan1290), and `excluded`/`ambiguous` for the other
reviewed Italian/French populations.

Edge cases:

* Several populations of one language stay separate link rows; nothing is averaged. The
  outcome row only counts them (`n_populations`, `n_populations_nonproxy`) and sums their
  individuals, per population the larger of the two panel sizes:
  `n_individuals_total` over non-proxy links, `n_individuals_proxy_links` over
  populations reached only by a proxy. These sums only rank coverage gaps.
* A Grambank entry is used only if its ID is a language-level Glottocode in Glottolog 5.3.
  Grambank dialect entries (70 in v1.0.3), the 3 Grambank "language" entries that are
  dialects in Glottolog 5.3 (lito1235, tamb1254, temp1235) and the one that is a family
  there (nuuu1241, N||ng-Danster !Ui) are not used; they are listed in
  `missing_from_grambank.csv` (`grambank_dialect_entries`) when they would fill a gap.
  **pcfp_v2 changes this rule** (`typology.dialect_entries: substitute`, user decision
  2026-10-08; pcfp_v1 used `ignore`, the default). A language with no language-level
  Grambank entry is represented by one of the non-language-level Grambank entries that
  roll up to it: the one with most coded main-set features (ties: lowest ID). Entries are
  never merged. `grambank_inflection.csv` records the entry used (`grambank_entry`,
  `grambank_entry_level`), and `coverage_summary.json` lists all substitutes. Group
  map-down counts substituted languages as Grambank-coded.
* Group map-down counts only descendants with a language-level Grambank entry. It is
  therefore a Grambank-conditional rule: it selects the only codable language, and it is
  reported as such.
* A language reached only by links with status `excluded` is not an outcome row.

**Which languages are rows.** Every language-level Glottocode reached by a non-excluded
link (exact, roll-up, map-down, manual or proxy), plus the language of every configured
LDL unit (`in_scope_reason` says which). Languages reached only through proxies are
kept, with `link_basis_proxy_only = True`; the main typology analysis uses
`link_basis_proxy_only == False`.

**`has_ldl_unit`.** For each configured unit (`cfg["units"]`, or the `units` argument),
the Glottocode comes from `outputs/<exp>/data/forms/<unit>.csv` (`glottocode`, first
row); if the data stage has not run, from `outputs/<exp>/registry/registry.csv`; if that
is absent too, from `morph_ldl.data.identifiers.resolve_identifier` on the ingest entry's
ISO code. The manifest records which source was used per unit. The code is rolled up to
language level with Glottolog. A unit whose Glottocode cannot be found is reported in the
manifest and in `coverage_summary.json`; the stage does not fail.

## 3. Feature set (declared: `typology.feature_set`, id `grambank_inflection_categories_12_v1`; pcfp_v2 and the default block)

**Why categories, not features.** Grambank splits one inflectional category into several
binary features: by argument role (S/A/P), affix position (prefix/suffix), value
(singular/dual/plural/…) or host (adjective/demonstrative/article). Many of these features
depend logically on each other:
* a language without person-indexing affixes is coded 0 six times (GB089–GB094);
* one without case is coded 0 four times (GB070–GB073);
* every gender-agreement feature presupposes a gender system.

A share of such features therefore weights a category by how finely Grambank splits it
(person indexing was 6/35 of the old score, mood 1/35). It also overstates the number of
independent observations behind each language's count.

The GBI curation (Graff et al. 2025, *Sci. Data* 12:106) removes these logical
dependencies:
* it merges such features into "X at all" features by logical OR;
* it codes sub-features NA unless the parent is present.

Graff et al. 2025 (*Sci. Adv.*) use GBI rather than raw Grambank for the same reason. The
GBI rules were read from the curation table `input/GBI/parameters.csv` in that paper's
archive.

**The measure.** The outcome counts 12 inflectional categories. Each category is the
logical OR of its binary Grambank sources (GBI merge rule), computed by the stage from the
pinned Grambank v1.0.3 values (`category_matrix`):
* `1` if any source is `1`;
* `0` if every source is `0`;
* empty if no source has a row;
* otherwise `?` (not coded).

| category | domain | Grambank sources | GBI equivalent |
|---|---|---|---|
| tense | verbal_tam | GB082, GB083, GB084 (present / past / future marking on the verb) | "overt morphological tense" (GB580mcc; GBI's GB557drm also counts non-morphological tense) |
| aspect | verbal_tam | GB086 (perfective/imperfective on the verb) | morphological part of GB559drm |
| mood | verbal_tam | GB312 (mood marking on the verb) | morphological part of GB558drm |
| person_indexing | person_indexing | GB089–GB094 (S/A/P by suffix or prefix) | GB625m |
| negation | neg_interrog | GB107 (negation by affix/clitic/verb modification) | GB107 (unchanged in GBI) |
| polar_interrogation | neg_interrog | GB285, GB286 (polar question by verbal morphology, with or without a particle) | GB947m |
| nominal_number | nominal_number | GB042, GB043, GB044, GB165, GB166 (sg/du/pl/trial/paucal on nouns) | GB991drm |
| case | case | GB070–GB073 (core/oblique × pronominal/non-pronominal) | GB480m |
| possessor_affix | possession | GB430, GB432 (prefix/suffix on the possessor) | GB590m |
| possessed_affix | possession | GB431, GB433 (prefix/suffix on the possessed noun) | GB591m |
| gender_agreement | agreement | GB170, GB171, GB172, GB198 (adjective/demonstrative/article/numeral) | GB560drm |
| number_agreement | agreement | GB184, GB185, GB186 (adjective/demonstrative/article) | GB561drm |

The merged categories are used unconditioned (GBI's `c`/`C` variants set NA where a
parent is absent). For a breadth count, "no gender, hence no gender agreement" must stay a
`0`, not become NA.

**Changes from the 35-feature set:**
* GB079 and GB080 are excluded (`catch_all_verbal_affixes`). These "verbal
  prefixes/suffixes other than pure S/A/P markers" overlap the TAM features and are not in
  GBI.
* GB285 joins GB286 as a source of `polar_interrogation`.
* The other exclusions are unchanged (see the table below).
* Sources must be binary in Grambank; the stage checks this.

**Domains and sensitivity sets.** The domain names are kept, now holding categories:
`verbal` has 6 categories, `nominal` 4, `no_agreement` 10.

The clitic heuristic's thresholds are rescaled from 5 of 35 to 2 of 12. So is
`minimal_inflection`, which stays at `n_present <= 1`. That cut is now relatively looser,
1 of 12 instead of 1 of 35 (§12).

**Remaining dependencies (documented, not removed).** Merging removes the logical
dependencies *within* a category. Statistical dependencies *between* categories remain,
for example those that GBI's "statistical" curation handles by extra conditioning:
* possessed-noun affixes with person indexing;
* number agreement with nominal number;
* gender agreement with number agreement;
* negation affixes with TAM morphology.

The beta-binomial model (§9) absorbs these as overdispersion.

### 3a. pcfp_v1 feature set (archived: id `grambank_core_inflection_35_v1`, `configs/pcfp_v1.yaml`)


35 binary Grambank features (codes `0`/`1` in v1.0.3). One line each, checked against
the parameter text in `parameters.csv`.

| ID | domain | Grambank question (short) | why included |
|---|---|---|---|
| GB079 | verbal_tam | verb prefixes/proclitics other than pure A/S/P markers | prefixal verbal inflection (TAM and other categories) |
| GB080 | verbal_tam | verb suffixes/enclitics other than pure A/S/P markers | suffixal verbal inflection (TAM and other categories) |
| GB082 | verbal_tam | overt morphological present tense on verbs | tense inflection |
| GB083 | verbal_tam | overt morphological past tense on verbs | tense inflection |
| GB084 | verbal_tam | overt morphological future tense on verbs | tense inflection |
| GB086 | verbal_tam | morphological perfective/imperfective distinction | aspect inflection |
| GB312 | verbal_tam | overt morphological mood marking on verbs | mood inflection |
| GB089 | person_indexing | S indexed by suffix/enclitic | person/argument indexing on the verb |
| GB090 | person_indexing | S indexed by prefix/proclitic | person/argument indexing on the verb |
| GB091 | person_indexing | A indexed by suffix/enclitic | person/argument indexing on the verb |
| GB092 | person_indexing | A indexed by prefix/proclitic | person/argument indexing on the verb |
| GB093 | person_indexing | P indexed by suffix/enclitic | person/argument indexing on the verb |
| GB094 | person_indexing | P indexed by prefix/proclitic | person/argument indexing on the verb |
| GB107 | neg_interrog | standard negation by affix, clitic or verb modification | polarity inflection of the verb |
| GB286 | neg_interrog | polar question by verbal morphology only | sentence-type (interrogative mood) inflection |
| GB042 | nominal_number | productive overt singular marking on nouns | number inflection (incl. singulatives) |
| GB043 | nominal_number | productive dual marking on nouns | number inflection |
| GB044 | nominal_number | productive plural marking on nouns | number inflection |
| GB165 | nominal_number | productive trial marking on nouns | number inflection |
| GB166 | nominal_number | productive paucal marking on nouns | number inflection |
| GB070 | case | morphological case for non-pronominal core arguments | case inflection |
| GB071 | case | morphological case for pronominal core arguments | case inflection of pronouns (Grambank Boundness 0.5: suppletive pronoun forms count) |
| GB072 | case | morphological case for oblique non-pronominal NPs | case inflection |
| GB073 | case | morphological case for oblique pronominal arguments | case inflection of pronouns (Boundness 0.5, as GB071) |
| GB430 | possession | possession prefix on the possessor | possessive inflection |
| GB431 | possession | possession prefix on the possessed noun | possessive inflection |
| GB432 | possession | possession suffix on the possessor | possessive inflection (genitive-type) |
| GB433 | possession | possession suffix on the possessed noun | possessive inflection |
| GB170 | agreement | adnominal property word agrees in gender | agreement inflection on modifiers |
| GB171 | agreement | adnominal demonstrative agrees in gender | agreement inflection on modifiers |
| GB172 | agreement | article agrees in gender | agreement inflection on modifiers |
| GB184 | agreement | adnominal property word agrees in number | agreement inflection on modifiers |
| GB185 | agreement | adnominal demonstrative agrees in number | agreement inflection on modifiers |
| GB186 | agreement | article agrees in number | agreement inflection on modifiers |
| GB198 | agreement | adnominal numeral agrees in gender | agreement inflection on modifiers |

**Present rule.** For every one of the 35 features, `Value == "1"` is present, `"0"` is
coded absent, `"?"`, an empty value or a missing row is *not coded*. The stage checks
`codes.csv`: if a declared feature has more than the codes `0`/`1` and no explicit
`present_codes.per_feature` entry, it refuses to run. Any value that is neither a code
in `codes.csv` nor `?`/empty stops the stage.

**Exclusions (declared in `feature_set.excluded`).**

| IDs | why excluded |
|---|---|
| GB047, GB048, GB049 | productive derivation of action, agent and object nouns: word formation, not inflection |
| GB187, GB188 | diminutive and augmentative: evaluative derivation, not paradigmatic inflection |
| GB119, GB120, GB121, GB298 | mood, aspect, tense and negation by an inflecting auxiliary word: periphrastic (free-word) marking, not bound inflection of the lexical verb |
| GB103, GB104 | benefactive and instrumental applicatives: valency-changing (derivational) verb morphology |
| GB113 | transitivising affixes/clitics: valency-changing derivation |
| GB147, GB148 | morphological passive and antipassive: voice/valency morphology, usually classed as derivational and highly lexically restricted |
| GB155 | causative affixes/clitics: valency-changing derivation |
| GB275 | bound comparative on property words: degree morphology restricted to one word class; not part of the noun/verb paradigms modelled by LDL |

**Notes on the set.** (a) The features are not independent: one TAM suffix can make
GB080 and GB082–GB084 present together; agreement and indexing features co-vary within
gender systems. The beta-binomial model (§9) treats this as overdispersion; it is not a
count of morphemes. (b) GB071 and GB073 have Grambank Boundness 0.5, because pronoun
case can be suppletive rather than affixal. They are kept: suppletive case forms are
still inflectional paradigms. (c) The agreement questions do not say how the agreement is
expressed; in practice it is bound marking on the modifier (Grambank Boundness 1).
(d) The count is 35, unique, no overlap with the exclusions; the stage checks this.

**Sensitivity sets** (`typology.sensitivity_sets`, each a union of domains):

| set | domains | n features |
|---|---|---|
| `verbal` | verbal_tam, person_indexing, neg_interrog | 15 |
| `nominal` | nominal_number, case, possession | 13 |
| `no_agreement` | all domains except agreement | 28 |

`nominal` is inflection of nouns and pronouns themselves; agreement (inflection of
modifiers controlled by the noun) is its own domain and is left out of `nominal`, so that
`nominal` and `no_agreement` answer different questions.

## 4. Outcome representation (`grambank_inflection.csv`)

Per language and feature set S (main set: no prefix; sensitivity sets: prefix
`verbal_`, `nominal_`, `no_agreement_`):

* `n_features` = |S|; `n_coded` = features with value `0` or `1`; `n_present` = features
  with a present code; `coverage = n_coded / n_features`; `share = n_present / n_coded`
  (empty if `n_coded == 0`).
* Counts are kept for binomial-type models.
* Coverage threshold for the main analysis: `coverage >= 0.6` (`meets_coverage_main`).
  Reported in addition: `meets_coverage_50` (>= 0.5) and `meets_coverage_75` (>= 0.75).
* `no_inflection` = main `n_present == 0`; `minimal_inflection` = main `n_present <= 1`.
  Both are empty for languages without a Grambank entry, and are computed regardless of
  coverage (combine with `meets_coverage_main` before use).
* Family and macroarea from Glottolog 5.3: `family_id` (top-level family; isolates: the
  language itself, `is_isolate = True`), `family`, `macroarea`.

## 5. Clitics

Grambank counts clitics as bound morphology. The parameter text of 9 of the 35 features
says so explicitly (GB079, GB080, GB089–GB094, GB107), and the "morphological marking"
features (tense, aspect, mood, case) follow the same Grambank convention. Largely analytic
languages with verbal particles or postpositional clitics therefore receive moderate
scores (the prototype found Burmese at 15/35). The stage **does not recode** Grambank. It
adds two transparent flags; `clitic_flag` is their union and `clitic_flag_reason` names
the rule(s):

1. `clitic_flag_family`: the top-level family is one of Sino-Tibetan (`sino1245`),
   Tai-Kadai (`taik1256`), Hmong-Mien (`hmon1336`) or Austroasiatic (`aust1305`), and the
   main `n_present >= 5`. These are the families of the Mainland Southeast Asia / China
   area whose grammars are usually described as isolating or particle-based, so a
   non-trivial score there is more likely to come from clitics or particles. Known
   over-inclusion: Sino-Tibetan includes richly inflecting languages (e.g. Kiranti,
   rGyalrong), and some Austroasiatic (Munda) languages are agglutinating.
2. `clitic_flag_profile` (family-blind): main `n_present >= 5`, but no feature present
   in `nominal_number` and none in `agreement`, with each of those two domains coded on at
   least half its features. A language with a fair score but neither noun number nor
   agreement inflection usually gets its score from verbal suffixes/particles and case
   postpositions, which Grambank may count although they are clitics (e.g. Japanese and
   Korean case particles). It flags real agglutinating languages too; it is a warning, not
   a classification.

Use: sensitivity analysis that drops `clitic_flag` languages; never a recode.

## 6. Linkage outputs

* `grambank_population_links.csv`: one row per (population, linked language, basis);
  population metadata (panels, sample sizes, own and proxy Glottocodes and their
  Glottolog levels), `glottocode` (language level, empty for unresolved/ambiguous
  groups), `candidate_glottocodes`, `link_basis`, `is_proxy`, `match_status`,
  `proposed_status`, `confirmed_by`, `decision_reason`, `unresolved_issues`,
  `review_ref`. No ancestry columns.
* `grambank_inflection.csv`: one row per language (§2, §4) with `gelato_linked`,
  `link_bases`, `link_basis_proxy_only`, `best_link_status` (accepted > candidate >
  ambiguous over non-proxy links, else over proxy links), `has_accepted_link`,
  `n_populations`, `n_populations_nonproxy`, `populations`, `n_individuals_total`,
  `n_individuals_proxy_links`, `has_ldl_unit`, `ldl_units`, `clitic_flag*`.
* `feature_set.json`: the declared set (copied from the config) and the set actually used
  (IDs with Grambank parameter names, code lists and present codes; config hash).
* `coverage_summary.json`: counts (links by basis/status, languages, in Grambank,
  coverage thresholds, no/minimal inflection lists, clitic flags, LDL overlap).
* `missing_from_grambank.csv`: GeLaTo-linked languages without a language-level Grambank
  entry; non-proxy languages first, then by `n_individuals_total` (descending). Gaps are
  listed, not hand-coded.
* `ldl_overlap.csv`: configured units plus the verb resources of the broad audit with
  suitability `suitable`, `suitable_with_caveats` or `limited` (read-only, from
  `typology.ldl_eligibility_audit`), with GeLaTo/Grambank status of their language.
* `stage_manifest.json` (`StageRecorder`) with `typology_sources` (tags, commits, file
  hashes), `files_opened` (every file the stage read, with sha256), `ldl_unit_glottocodes`
  and the output hashes.

Output rows are sorted by key and floats rounded to 6 decimals; two runs give
byte-identical CSVs.

## 7. Ancestry firewall

The stage opens files only through one reader that refuses paths matching ancestry or
genetic-value patterns (`*.Q`, `GeneticInfoID`, `best_runs/`, `*ancestry*`,
`*K12_K30*`, `*diagnostics*`, the audit's `population_crosswalk.csv`, GeLaTo
`datasets/*/data.csv`), and records every opened path in the manifest.
`audit_typology` re-checks that list. No ancestry value, Q matrix or genetic summary is
read, and no choice in this document was made after looking at one.

## 8. Audit (`audit_typology(cfg) -> (checks, problems)`)

1. `feature_set.json` declared == used, and both equal the current `cfg["typology"]`
   feature set and present-code rules.
2. Manifest status `ok`; Grambank and Glottolog tags/commits recorded and equal to the
   config pins; `files_opened` contains no forbidden path.
3. Outcome rows: `glottocode` non-empty and unique, and `Level == language` in Glottolog
   5.3 (re-read from the pinned `languages.csv`); `0 <= n_present <= n_coded <= n_features`
   for every set.
4. Link rows with a `glottocode` point to an outcome row (unless status `excluded`); no
   `accepted` status without `confirmed_by`.

## 9. Proposed analysis PROTOCOL (not fitted)

* **Grambank extent.** Beta-binomial on (`n_present`, `n_coded`) per language, main set,
  languages with `meets_coverage_main` and `link_basis_proxy_only == False`. The
  beta-binomial absorbs the dependence between features (§3 note a). A hurdle or
  zero-inflation part is added only if posterior predictive checks show more zeros than
  the beta-binomial predicts; the prototype found 4–6 languages at or near zero.
  Sensitivity: 50% and 75% coverage, the three sensitivity sets, dropping
  `clitic_flag` languages, including proxy-only languages.
* **LDL predictability.** Binomial at the item level (item correct/incorrect) with
  language random effects, or beta-binomial per language on (correct, n items).
  Inflecting languages only (the LDL task needs paradigms).
* **Confounding.** The zero / near-zero inflection languages are almost all in
  Mainland Southeast Asia and China and share East Asian ancestry. Once area (and
  family) is controlled, any no-inflection component is weakly identified: there is
  hardly any within-area variation left. The no-inflection part is therefore reported
  descriptively, not as a main test.
* **Exposure.** The admixture exposure variable is not defined here. Nothing in this
  stage looks at, defines or fits it.

## 10. Development run (2026-10-08, experiment `typology_dev`, outputs in a scratch directory)

Config: `configs/pilot.yaml` + the block of `morph_ldl/typology/default_config.yaml`.
Grambank v1.0.3 @ `7ae000c`, Glottolog v5.3 @ `072ca0d` (both verified). Audit: no
problems; 10 files opened, none forbidden. Two runs gave byte-identical outputs.

* **Links.** 558 populations, 619 link rows: exact 421 (416 candidate, 5 ambiguous by
  review), dialect roll-up 57, group map-down 12, group ambiguous 28, group without
  Grambank descendant 5, unresolved (own code `NA`) 35, proxy 61 (57 ambiguous, 4 excluded
  by review). 42 populations reach no language at all (e.g. Yi = Loloish, 22
  Grambank-coded candidates; Vanuatu and Papuan regional samples; Dinka).
* **Languages.** 350 language-level Glottocodes: 309 with a non-proxy link, 41
  proxy-only. In Grambank: 193 non-proxy (+30 proxy-only).
* **Coverage** (non-proxy, in Grambank): >= 50%: 183; >= 60% (main): 173; >= 75%: 168.
  Including proxy-only languages: 211 / 200 / 194.
* **No inflection** (main `n_present == 0`): Central Khmer (Austroasiatic, 0/35),
  Naxi (Sino-Tibetan, 0/35), She (Hmong-Mien, 0/29), Vietnamese (Austroasiatic, 0/30),
  Thai (Tai-Kadai, 0/20, below the 60% threshold). **Minimal** (<= 1) adds Mandarin
  Chinese (1/35) and Northern Tujia (1/35). All Eurasia, Mainland Southeast Asia / China.
* **Clitic flags** (15 non-proxy): family rule: Burmese 15/35, Hakka 6/34, Kharia 13/32;
  profile rule: Basque, Kusunda, Xavante and nine Austronesian languages (Amis,
  Kankanaey, Minamanwa, Notsi, Tutuba, Paama, Akei, Apma, Tigak). Basque and Kharia are
  known false positives of the heuristics.
* **Missing from Grambank** (116 non-proxy), top by individuals: Yoruba 75, Scottish
  Gaelic 43 (the GeLaTo "Scottish" samples carry scot1245; see unresolved issues), Taiga
  Sayan Turkic 40, West Circassian 33, Chachapoyas Quechua 31, Tajik 31, Zoroastrian Yazdi
  28, Mamusi 26, West !Xoon 26, Spanish 25. East Asian gaps: Hmong Daw (Miao, 10), Lü (Dai,
  10); Yi is unlinked (ambiguous group); Tujia maps down to Northern Tujia (in Grambank).
* **LDL overlap.** Italian (ital1282, 19/35) and Finnish (finn1318, 14/35): GeLaTo-linked
  (best status `candidate`, acceptance proposed, not confirmed), in Grambank, full
  coverage. Of 16 broad-audit verb resources (suitable / with caveats / limited), 13
  languages are GeLaTo-linked, 12 non-proxy, 8 also in Grambank above 60% (English,
  French, Catalan, Finnish, Serbo-Croatian, Armenian, Italian, Northern Sami); Bulgarian,
  German, Romanian and Spanish are not in Grambank; Modern Greek is proxy-only.
* **Against the prototype** (`analyses/unimorph_gelato_estimate_2026_10_08`): 301 shared
  languages, identical `n_present`/`n_coded` and Grambank membership. The prototype kept
  36 group-level codes as rows; here 11 map down (12 population links; e.g. Japanesic ->
  Japanese, Tujia -> Northern Tujia, Pashto -> Southern Pashto), 24 give no automatic
  match (ambiguous or no Grambank-coded descendant), and nuuu1241 (a Grambank entry that
  is a family in Glottolog 5.3) is not used. New here: 8 map-down target languages and the
  41 proxy-only languages (the prototype ignored GBI/TLI proxies). Individual totals
  differ for 3 languages (Iron Ossetian, Kabardian, Western Farsi) because map-down adds
  populations (e.g. Circassian -> Kabardian); proxy populations are counted separately.

## 11. Real run (2026-10-08, experiment `pcfp_v1`, `outputs/pcfp_v1/typology/`)

Config `configs/pcfp_v1.yaml` (hash 8b088277dfaa1e26), code `9c4ee53`. Every count in §10
was reproduced exactly:

* 558 populations and 619 link rows;
* 350 languages (309 non-proxy, 41 proxy-only);
* 193 non-proxy languages in Grambank, 183 / 173 / 168 at 50 / 60 / 75% coverage;
* the same no-inflection, minimal-inflection and clitic-flag lists;
* 116 non-proxy languages missing from Grambank;
* the same LDL overlap.

The typology audit inside the final `audit` stage passed (0 problems across 102 artifact
sets). The results are in [REPORT_pcfp_v1.md](REPORT_pcfp_v1.md) §5.

**Are the missing languages really absent?** (checked 2026-10-08,
`analyses/pcfp_v1_2026_10_08/grambank_missing_check.py` → `grambank_missing_check.csv`)

Each of the 127 languages in `missing_from_grambank.csv` (116 non-proxy, 11 proxy-only)
was searched in Grambank v1.0.3 in four ways:

* by ISO 639-3 code;
* for Grambank entries below the language in Glottolog (dialects);
* for Grambank entries above it (groups or families);
* by name.

Result:

* **No language is coded under a different language-level Glottocode or ISO code.**
  Spanish (stan1288), German (stan1295), Scottish Gaelic (scot1245), Romanian (roma1327),
  Bulgarian (bulg1262), Yoruba (yoru1245), Tajik, Nogai, Sindhi and the rest have no
  Grambank entry at any level. Grambank v1.0.3 does not cover them.
* **Three languages are coded only through dialect entries,** which §2 declares unused:
  * Selkup (selk1253, 24 individuals): Southern Selkup (sout3262);
  * Karelian (kare1335, 15): Northern Karelian (nort2673) and Tver (tver1240);
  * Terena-Kinikinao-Chane (tere1279, 1): Terena (tere1281) and Kinikinao (guan1270).

  Letting a dialect entry stand for its language would add them. That is a rule change
  to be declared before any outcome analysis, not a correction.
* **The other hits are not matches:**
  * Oceanic (ocea1241) is a family-level Grambank entry (a reconstruction), not a code
    for the Oceanic languages Mamusi, Duke, Sowa and others.
  * Aromanian ≠ Romanian, Shuar ≠ Shua, Narom ≠ Naro, Basay ≠ Basa (Cameroon).
* **The GeLaTo side may be off.** The Scottish population samples (43 individuals) are
  linked to Scottish Gaelic (scot1245) through GeLaTo's own code, which may not reflect
  what they speak (§10). Whether they should link to English (stan1293, in Grambank) is
  a GeLaTo-link review decision; it must not be decided by looking at either outcome.

**Dialect substitution in pcfp_v2** (`outputs/pcfp_v2/typology/`; audit: no problems).

* Karelian (via Northern Karelian, 14/35), Selkup (via Southern Selkup, 15/35) and
  Terena-Kinikinao-Chane (via Kinikinao, 11/34) enter the main set.
* For Terena, the coverage rule picked Kinikinao over Terena, which has fewer coded
  features. The GeLaTo population (1 individual) is probably Terena, so this needs review.
* Murut's group map-down (to Timugon Murut) becomes ambiguous, because a second Murutic
  language is now Grambank-coded through a dialect entry.
* Main set: 173 → 175 languages (≥ 50%: 185, ≥ 75%: 170); 203 including proxy-only.
* Other languages also gain substitutes (Hopi, Eastern Mari and more). They are not
  GeLaTo-linked, so they do not affect the outcome set.

## 12. pcfp_v2 run with the 12 categories (2026-10-08, `outputs/pcfp_v2/typology/`)

The run used `dialect_entries: substitute` (§2) and the 12-category set (§3). The typology
audit passes; the full audit reports 56 artifact sets and 0 problems. Comparisons are with
the same run under the 35-feature set.

* **Entering the main analysis** (non-proxy, coverage ≥ 60%): **175** languages, the same
  175 as with the 35-feature set. Mean coverage is 0.88 (0.90 before).
* **Agreement with the old score:** Pearson 0.92 and Spearman 0.89 with the 35-feature
  share across the 175. Mean share is 0.62 (SD 0.23).
* **Distribution of n_present** (0–12):

  | n_present | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | languages | 4 | 4 | 5 | 8 | 15 | 11 | 9 | 23 | 42 | 33 | 16 | 5 |
* **No inflection** (0 categories): Central Khmer 0/11, Naxi 0/12, She 0/10 and
  Vietnamese 0/11; Thai 0/5 is below coverage. The list is unchanged.
* **Minimal (≤ 1).** Mandarin and Northern Tujia stay; Central Maewo, Mussau-Emira, Tagalog
  and Tonga are added. ≤ 1 of 12 is a looser cut than ≤ 1 of 35. Tagalog's voice/focus
  morphology falls under the declared `valency_voice` exclusion. The flag is descriptive.
* **Ceiling and compression.** No language reaches 12/12. A few reach share 1 through
  missing codes (Ingush 10/10, North-Central Dargwa 9/9).
  * The scale measures breadth, so it compresses elaborate systems: Italian 8/12 (19/35
    before), Finnish 8/12 (14/35), English 7/12 (10/35).
  * The largest upward shifts are in languages whose many 35-feature zeros were sub-values
    of present categories (Ingush, Cocama-Cocamilla, Dargwa: +0.38 to +0.42 in share).
* **Clitic flags:** 16 non-proxy languages, with Lahu and Nakanai new at the rescaled
  threshold.
