# Grambank inflection-extent outcome (`typology` stage)

Owner: typology component (`morph_ldl/typology/`). §1–§8 specify the sources, mapping
rules, 12-category feature set, outputs, ancestry firewall and audit; §9 summarises the
outcome modeling design; §10–§12 record the missing-language audit, the 12-category
`pcfp_v2` run, and the 2026-10-20 link-rules run (202 main-set languages).

The stage gives a second, language-level outcome next to the LDL predictability outcome:
the extent of bound inflectional morphology from Grambank, for every language reached
from a GeLaTo population. It never reads ancestry values (§7).

## 1. Sources (pinned; recorded in `typology/stage_manifest.json`)

| source | pin | files read |
|---|---|---|
| Grambank CLDF (`grambank/grambank`) | tag `v1.0.3`, commit `7ae000cf740f93cdb3e4ec67010668d6795337a9` | `cldf/values.csv`, `cldf/parameters.csv`, `cldf/codes.csv`, `cldf/languages.csv` |
| Glottolog CLDF (`glottolog/glottolog-cldf`) | tag `v5.3`, commit `072ca0d0410039fb8b779be8fc165bac575d2cda` | `cldf/languages.csv` (Level, Language_ID, Family_ID, Macroarea); `cldf/values.csv`, the `classification` rows (for group map-down) and, since 2026-10-20, the `aes` rows (endangerment, for the `endangered` flag) |
| GeLaTo main_397 | gelato-data commit `c625fdc…` | `cldf/populations.csv` |
| GeLaTo expanded_558 | Zenodo 15263706 | `tables/tableS1.csv` |
| decision sheet (configs with `typology.decision_sheet`, since 2026-10-20) | in the repository | `morph_ldl/data/gelato_link_review.csv` |

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
the Zenodo population mapping (not needed). Only the sheet regeneration
(`python -m morph_ldl.cli link_review`) reads that mapping, for the place and Graff
comment of each sheet row.

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

Roll-up and map-down are initial matches. **Link rules and one decision sheet (2026-10-20, user decisions; `configs/pcfp_v2_links.yaml`).**
Checking each link by hand is too slow, and a check of an exact link adds nothing. So:

* **Accepted by rule** (`typology.link_rules.accept_by_rule`, column `accepted_by_rule`):
  `exact`, `dialect_rollup`, `group_map_down`, and `graff_gbi_stand_in`. The last is
  Graff et al.'s Grambank-side (GBI) stand-in, used only when the population has no own
  link to a Grambank language. Their WALS-side (TLI) stand-ins are never accepted by rule:
  they replace languages missing from WALS/AUTOTYP, not from Grambank. Every case where a
  stand-in disagrees with an own main-set link is a TLI stand-in (Kove → Lusi, Sicilian →
  Italian), so the order of the rules does not matter.
* **The decision sheet** `morph_ldl/data/gelato_link_review.csv` is the only place where
  a person decides; `gelato_review.yaml` is an archive. One row per link, keyed by
  (`population`, `link_glottocode`). `decision` is `accept`, `relink` (a manual link to
  `decision_glottocode` replaces the link), `exclude` or `ambiguous`; it takes effect with
  `confirmed_by` + `confirmed_date` (rule of 2026-10-16), otherwise it is a proposal.
  Malformed rows are reported and not applied. A confirmed YAML decision missing from the
  sheet stops the stage. The sheet holds only what the rules cannot settle: group codes
  without a Graff choice, the 12 decisions of 2026-10-16, the Yukaghir swap (relink
  proposals), and doubt labels.
* **Main analysis** = links with `in_main` (accepted, or a non-proxy candidate) to a
  Grambank language at the coverage threshold (`in_main_analysis` in the outcome table).
* **Sensitivity analyses** (`typology.sensitivity_analyses`) drop the main-analysis links
  that carry a flag (`sensitivity_flags`); they never add a link:
  * `endangered`: the linked language's Glottolog AES is at least
    `link_rules.endangered_min_aes` (4, moribund; 3 would add "shifting"). The sampled
    community may have shifted to another language.
  * `diaspora`, `wrong_code`: doubt labels from the sheet. They apply to every link of the
    population and need no confirmation, because they never change the main analysis.
    Diaspora = Graff et al. mapping location `migrants_collected_in_…`; wrong code = the
    source check of 2026-10-16. `group_choice` (added 2026-10-20, user decision): a person
    chose one language of a group code; the choice is a judgement, unlike an exact code.
    For a sample labelled only by island, the proposal follows the rule Graff et al. used
    for Tanna: the group's language with most speakers; if two are about equal, no choice.
  * `stand_in_other_language` (a Graff stand-in replaces the population's own language,
    for example Saudi → Gulf Arabic) and `stand_in_no_own_language` (the population's own
    code is a group, for example Greek → Modern Greek).

  pcfp_v2_links declares `no_doubtful_language` (endangered, diaspora, wrong_code, group_choice),
  `no_other_language_stand_ins`, and `strict` (both). Sample size is not a flag: the
  exposure model carries it as uncertainty (user decision 2026-10-20).
* `python -m morph_ldl.cli link_review --config …` rebuilds the sheet from the typology
  outputs: rows are merged by key, filled cells are kept, new cases are added, and a filled
  row whose link no longer exists is kept and marked `STALE (no such link)`. It refuses to
  write if a filled row would be lost or changed.

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
`link_basis_proxy_only == False`. With the link rules of 2026-10-20 it uses
`in_main_analysis` instead (accepted stand-ins count).

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
1 of 12 instead of 1 of 35 (§11).

**Remaining dependencies (documented, not removed).** Merging removes the logical
dependencies *within* a category. Statistical dependencies *between* categories remain,
for example those that GBI's "statistical" curation handles by extra conditioning:
* possessed-noun affixes with person indexing;
* number agreement with nominal number;
* gender agreement with number agreement;
* negation affixes with TAM morphology.

The beta-binomial model (§9) absorbs these as overdispersion.

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
| GB079, GB080 | catch-all verbal prefixes/suffixes other than pure A/S/P markers: overlap the TAM features and are not in GBI |

**Sensitivity sets** (`typology.sensitivity_sets`, each a union of category domains):

| set | domains | n categories |
|---|---|---|
| `verbal` | verbal_tam, person_indexing, neg_interrog | 6 |
| `nominal` | nominal_number, case, possession | 4 |
| `no_agreement` | all domains except agreement | 10 |

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
  `proposed_status`, `accepted_by_rule`, `confirmed_by`, `confirmed_date`,
  `decision_reason`, `unresolved_issues`, `review_ref`, `aes`, `sensitivity_flags`,
  `in_main`. No ancestry columns.
* `analysis_links.csv` (since 2026-10-20): the links of the main analysis and of each
  sensitivity analysis (`analysis`, population, language, basis, `accepted_by` = rule or
  person, flags, `n_individuals`, `in_grambank`, `meets_coverage_main`).
  `coverage_summary.json` gives the counts per analysis (`analyses`), the rules, and the
  decision sheet (rows by reason, confirmed and proposed decisions, malformed rows, rows
  without a link).
* `grambank_inflection.csv`: one row per language (§2, §4) with `gelato_linked`,
  `link_bases`, `link_basis_proxy_only`, `best_link_status` (accepted > candidate >
  ambiguous over non-proxy links, else over proxy links), `has_accepted_link`,
  `n_populations`, `n_populations_nonproxy`, `n_populations_main`, `in_main_analysis`,
  `populations`, `n_individuals_total`,
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

**Update 2026-10-21.** The firewall still holds for this stage. Outside it, the exploratory
admixture models (`analyses/draft_admixture_model_2026_10_10/` and
`analyses/admixture_links_2026_10_21/`, now `analyses/grambank/baseline/`) read the ancestry
values. The choices in this document (12-category feature set, mapping rules, coverage
threshold) predate those runs.

## 8. Audit (`audit_typology(cfg) -> (checks, problems)`)

1. `feature_set.json` declared == used, and both equal the current `cfg["typology"]`
   feature set and present-code rules.
2. Manifest status `ok`; Grambank and Glottolog tags/commits recorded and equal to the
   config pins; `files_opened` contains no forbidden path.
3. Outcome rows: `glottocode` non-empty and unique, and `Level == language` in Glottolog
   5.3 (re-read from the pinned `languages.csv`); `0 <= n_present <= n_coded <= n_features`
   for every set.
4. Link rows with a `glottocode` point to an outcome row (unless status `excluded`); no
   `accepted` link, and no link ambiguous or excluded by a review, without
   `confirmed_by` + `confirmed_date` (`gelato.review_link_problems`; since 2026-10-16). The
   final `audit` stage applies the same check to the data-stage `gelato/` link tables.
   Since 2026-10-20 a link accepted by a configured rule (`accepted_by_rule`) needs no
   person; a rule not in the config is a problem.
5. `analysis_links.csv`: the main analysis equals the `in_main` links, each sensitivity
   analysis is a subset of it, the analyses equal the config, and every link has an
   outcome row.

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
* **Exposure.** The admixture exposure variable is not defined inside this stage; the
  two-stage hierarchical admixture → Grambank inflection and Flexivity models are
  implemented in [`analyses/grambank/baseline/`](../analyses/grambank/baseline/README.md).

## 10. Missing-language audit and dialect substitution (2026-10-08)

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
* **Three languages are coded only through dialect entries,** which §2 declares unused
  unless `dialect_entries: substitute` is enabled:
  * Selkup (selk1253, 24 individuals): Southern Selkup (sout3262);
  * Karelian (kare1335, 15): Northern Karelian (nort2673) and Tver (tver1240);
  * Terena-Kinikinao-Chane (tere1279, 1): Terena (tere1281) and Kinikinao (guan1270).
* **The other hits are not matches:**
  * Oceanic (ocea1241) is a family-level Grambank entry (a reconstruction), not a code
    for the Oceanic languages Mamusi, Duke, Sowa and others.
  * Aromanian ≠ Romanian, Shuar ≠ Shua, Narom ≠ Naro, Basay ≠ Basa (Cameroon).
* **The GeLaTo side may be off.** The Scottish population samples (43 individuals) are
  linked to Scottish Gaelic (scot1245) through GeLaTo's own code, which may not reflect
  what they speak.

**Dialect substitution in `pcfp_v2`** (`outputs/pcfp_v2/typology/`; audit: no problems):

* Karelian (via Northern Karelian), Selkup (via Southern Selkup) and
  Terena-Kinikinao-Chane (via Kinikinao) enter the main set.
* Murut's group map-down (to Timugon Murut) becomes ambiguous, because a second Murutic
  language is now Grambank-coded through a dialect entry.
* Non-proxy main set (before link rules): 173 → 175 languages (≥ 50%: 185, ≥ 75%: 170);
  203 including proxy-only.

## 11. `pcfp_v2` run with the 12 categories (2026-10-08, `outputs/pcfp_v2/typology/`)

The run used `dialect_entries: substitute` (§2) and the 12-category set (§3). The typology
audit passes; the full audit reports 56 artifact sets and 0 problems.

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

## 12. Link rules run (2026-10-20, `configs/pcfp_v2_links.yaml`, `outputs/pcfp_v2_links_2026_10_20/typology/`)

Typology stage with rule-based link acceptance and all 33 rows of the decision sheet
(`morph_ldl/data/gelato_link_review.csv`) confirmed by the user. The typology audit passes.

* **Main set: 202 languages** (175 before link rules; none lost). 335 populations, 2,962
  individuals (2,943 in the ADMIXTURE files).
  * Graff et al.'s Grambank-side (`GBI`) stand-ins add 23 languages (14 resolving a group
    code such as Modern Greek, Halh Mongolian, Lenakel; 9 replacing a language missing from
    Grambank such as Gulf Arabic, Chaldean Neo-Aramaic, Forest Enets, Central Mansi).
  * Group codes confirmed in the decision sheet (label `group_choice`): Buka → Halia,
    Yi → Sichuan Yi, Vanuatu_Ambrym → North Ambrym, Vanuatu_Banks → Mwotlap;
    Vanuatu_Malekula `ambiguous` (not in the main analysis).
  * Yukaghir swap corrected: Northern Yukaghir is represented by Yukagir_Tundra (n=19)
    instead of Yukagir_Forest (n=5), which drops out (Southern Yukaghir has no Grambank data).
  * Sicilian_East/West: the WALS-side stand-in Italian is excluded; they stay linked to Sicilian.
* **Sensitivity analyses** (main-set languages / populations):

  | analysis | drops | languages | populations |
  |---|---|---|---|
  | `main` | – | 202 | 335 |
  | `no_doubtful_language` | endangered (AES ≥ 4), diaspora, wrong_code, group_choice | 174 | 293 |
  | `no_other_language_stand_ins` | the 16 stand-ins for another language | 193 | 319 |
  | `strict` | all of the above | 169 | 281 |
