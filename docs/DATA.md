# Data stage: adapters, registry, eligibility and GeLaTo linkage

Owner: data subagent. Version 1, 2026-10-06. Follows `docs/CONTRACT.md` v1; proposed
contract changes are listed in section 11 and are not applied here.

## 1. Entry points and outputs

| Function (in `morph_ldl.data`) | Purpose |
|---|---|
| `run_data_stage(cfg, with_registry=True) -> dict` | The full stage for a config from `morph_ldl.config.load_config`. |
| `load_forms(cfg, unit_id) -> DataFrame` | Reads `data/forms/<unit_id>.csv` with the contract dtypes (ints, bool `is_missing`). |
| `build_population_links(outcome_units, cfg) -> DataFrame` | Returns one row per (unit, population candidate): status plus sample and ancestry-source identifiers. It returns no morphology values and does no aggregation. |

The stage writes only to `outputs/<experiment_id>/`. `util.guard_path` refuses any other
target, including `scratch_data/`. With `with_registry=False`, the stage skips the
135-file registry and broad audit (about 1 min) and builds `gelato/crosswalk.csv` for the
configured units only.

```
data/forms/<unit_id>.csv          contract §2 rows, group_id filled
data/lemmas/<unit_id>.csv         lemma_id, group_id, group_size, join_reasons, eligible
data/unparseable_cells.csv        labels with empty cell_norm (pilot: none)
data/stage_manifest.json          contract §9
registry/registry.csv             every MGN data/ and data-custom/ file (135)
registry/identifier_crosswalk.csv original id -> canonical ISO -> low Glottocode -> override
registry/build_provenance.csv     per-file derivation parsed from the MGN build scripts
registry/unimorph_verification.csv MGN ita-v / fin-v vs pinned UniMorph clones
eligibility/<unit>_cells.csv      per-cell coverage, missing/variant/multiword rates, role
eligibility/<unit>_duplicates.csv multi-lemma leakage groups with the join reason
eligibility/<unit>_derived_paradigms.csv  near-duplicate paradigms (see §7)
eligibility/summary.csv           one row per configured unit
eligibility/config_cell_check.csv configured cell_norm strings vs normaliser output
eligibility/broad_audit.csv       all MGN V/N resources, plausible NFIN/NOM.SG + 8-cell task
gelato/crosswalk.csv              resource x population candidate (all 135 resources)
gelato/ancestry_sources.csv       population metadata vs Q matrices vs derived summaries
gelato/population_links.csv       build_population_links() for the configured units
```

## 2. Adapters

Every adapter yields raw long records. `to_forms()` then adds the contract columns.
The source cell content is kept unchanged in `form_orig`. `source_row` is the 1-based line
number in the source file, with the header on line 1.

* **`mgn_wide`** (`mgn_data/data/*.csv`) has a `lexeme` column plus one column per cell.
  Variants are `;`-joined and missing cells are the literal `NA`. Each variant becomes one
  row with `variant_idx` in stored order. **MGN stores variants in alphabetical order**
  (`paste(sort(unique(x)), collapse=";")`), so "variant 0" means the alphabetically first
  variant, not a preferred form. A repeated `lexeme` row is a distinct paradigm and gets
  `#k` (contract §1).
* **`mgn_long`** (`data-custom/*.csv`) has the columns `lexeme,cell,form,language_ID,POS`.
  Extra columns are tolerated, such as Navajo's unnamed index. Each row is one variant,
  and `variant_idx` is the order of appearance within (lexeme, cell). A long table cannot
  separate a repeated paradigm from a variant, so repeated rows are treated as variants.
  Variant rates: Russian nouns 3.0 %, Arabic and Navajo verbs have duplicates. `form_orig`
  is that row's form.
* **`unimorph`** is the standard `lemma<TAB>form<TAB>features[<TAB>extra…]` format. Blank
  lines are skipped and an optional POS filter applies. The POS tag stays in `cell_orig`
  and is removed from `cell_norm`.
* **`paralex`** is a Paralex package directory. Its `forms.csv` has
  `form_id, lexeme, cell, phon_form|orth_form`. `representation` selects `phon_form`
  (`phon_custom`) or `orth_form` (`orth`), and `#DEF#` (defective) counts as missing.
  Cells resolve to UniMorph through a `unimorph` column in `cells.csv`. Failing that, the
  `unimorph` column of `features-values.csv` is applied to the `.`-separated value ids of
  the cell id. Failing that, the normaliser is applied to the raw id. The interface is
  tested on a fixture only.

Config `resources.ingest` entries use `{adapter, file, iso, pos, representation}`. The
`file` value is relative to `paths.mgn_data` (wide), `paths.mgn_custom` (long), or the
pipeline root (unimorph/paralex). `resource_id` is `mgn_data:<stem>`, `mgn_custom:<stem>`,
`unimorph:<stem>` or `paralex:<dir>`.

## 3. Representation, segmentation, cell normalisation

**Representation** comes from the MGN build scripts, never from the forms.
`provenance.parse_clean_paradigms` reads `clean-paradigms.R`. In each block,
`df_x <- read_tsv(".../unimorph/<code>/<code>")` names the UniMorph source, and
`df_x$form <- epi_transliterate(df_x$form, "<epi>")` marks automatic G2P.
`write_csv(df_x_<pos>, "../data/<file>.csv")` names the output. The result is:

* `ipa_epitran`, 25 files: 23 UniMorph files of aze, cat, deu, fas, hin, kmr, nld, ron, spa, swe, tel, tur, ukr
  and zul (UniMorph + epitran), plus `pol-n` and `pol-adj` (PoliMorf 0.6.7 + epitran
  `pol-Latn`, from `clean-pol.R`). epitran is applied to forms only. **Lexeme labels stay
  orthographic**, so the label and the source form differ in representation.
* `orth`, 98 files: untouched UniMorph orthography.
* `phon_custom`, 12 files: all `data-custom`. No build script exists. The label and
  transcription conventions suggest Flexique for French and LatInfLexi for Latin; this
  is unverified.

A heuristic check (`representation_check`) found no file inconsistent with its declared
representation. The stage refuses a config whose declared representation differs from
the build provenance. It also refuses any unit that mixes representations.

**Segments** (contract §2):

* `orth`: one symbol per character, with combining marks attached.
* `phon_custom`: the source's tokens.
* Word spaces become `_`.
* *Proposal (contract silent):* `ipa_epitran` uses the orth rule, but also attaches the
  IPA length and secondary-articulation modifiers `ː ˑ ʰ ʲ ʷ ˠ ˤ ˀ ⁿ` to the preceding
  symbol. epitran ran with `ligatures=True`, so affricates are already single code points.

No MGN form contains `#`. One file (`ckb-n`) contains `_`, which would clash with the word
separator. The registry flags it.

**`cell_norm`** follows contract §1: upper-case, de-duplicated, sorted, `;`-joined, with
bare POS removed. The normaliser tries these in order:

1. Data-stage parsers for data-custom conventions:
   * English 8-cell table: `presothers`, `past13` and `pastnot13` use UniMorph
     `LGSPEC1/2`.
   * French dotted `prs.1sg`.
   * Hungarian `in+ess;sg`, with `inst` mapped to `INS`.
   * Latvian and Russian case words. Russian prepositional maps to `ESS`, as in UniMorph.
   * Latin nouns `NOUN:Abl+Plur`.
   * Portuguese `PresIndic1..6`: 1–3 are SG and 4–6 are PL. The personal infinitive is
     `NFIN;LGSPEC1`.
   * Chatino aspect labels.
2. The project normaliser `src/cell_normalization.py` with `cells_to_unimorph.json`
   (imported read-only). It handles the curated dotted map, Navajo, native UniMorph,
   Polish colon labels, uncurated dotted labels and bare features.
3. A bare upper-case tag such as `CAUSE` or `PSS3P`.

The only unparseable labels are the 254 Latin verb labels (`VERB:Fin+…`). Latin is
extinct and out of scope.

**Pilot config check.** All 18 configured `cell_norm` strings in `configs/pilot.yaml`
match the normaliser's output for `ita-v` and `fin-v` (`eligibility/config_cell_check.csv`).
There are no mismatches.

## 4. MGN provenance findings (from the build scripts; affect eligibility)

1. **Cells with ≤100 forms are dropped** (`process_subset`, `.n = 100`; Polish adjectives
   use 400).
2. **Fully duplicated columns are dropped, including duplicates of the `lexeme` column.**
   The rule is `.df[!duplicated(as.list(.df))]`, and it runs after pivoting, with `lexeme`
   as the first column. Two consequences:
   * A citation-form cell whose form equals the label for every lemma disappears. This
     happens to NFIN in many verb files (hbs, mkd, lit, hye, sme, bul, est, tur …) and to
     NOM;SG in many noun files (hye, bak, tat, nob …).
   * A cell syncretic with an earlier cell everywhere disappears. Examples: Finnish
     NOM;SG = ACC;SG for all nouns (verified on UniMorph `fin`), and German 3PL = 1PL.

   An absent cell is therefore not evidence that the language lacks it. For the broad
   audit only, a missing source cell is reconstructed from the lexeme label (orth files
   only) and flagged `reconstructed … unverified`. This is not done for units. The flag
   matters most for languages whose UniMorph lemma is not the infinitive, such as
   Macedonian and Bulgarian.
3. `ast-v.csv` is written by two blocks of `clean-paradigms.R`, and the later block
   (lines 1502–1529) determines it. `pus-v.csv` is written by the script but absent from
   `data/`.
4. **UniMorph verification** (`registry/unimorph_verification.csv`, clones in `external/`):
   * `unimorph-ita` @ `fa2cc6ce173643e748bd8f4162709365bdc98805`: MGN `ita-v` is
     identical. All 10,009 lemmas, all 48 cells and 479,579/479,579 form sets match.
     MGN Italian is untouched UniMorph orthography.
   * `unimorph-fin` @ `fe0a2707244ed2ce2fe5d92a4a57c92271b32e1f` (current, split into
     `fin.1`/`fin.2`): MGN `fin-v` comes from an **older** UniMorph `fin` release (a
     single `fin` file). MGN has 7,724 verb lemmas and the current release has 7,222, with
     5,030 shared. MGN's plain `NFIN` corresponds to the current `NFIN;1;ACT`
     (A-infinitive). The current release's other infinitive and participle-like cells
     are absent from MGN. Shared (lemma, cell) form sets are 99.55 % identical. The exact
     MGN source release is unidentified.
   * Both pilot resources have **0 % variants** in every cell.
   * Italian NFIN has 14 erroneous UniMorph entries; for example, `accidere` NFIN is
     `accendere`. The source-form grouping rule joins these to the other lemma.

## 5. Identifier crosswalk

The layers are applied in order:

1. File code. data-custom names map to MGN codes, for example `french` to `fre`.
2. Project correction `src/mgn_language_map.canonical_iso`, read-only.
3. `low` (languages-of-the-world @ `d319631e916ae7f363d2e1c78f9dc9f3d61d3e51`, v0.2.0)
   for ISO→Glottocode.
4. The contract variety subtag: `nob` becomes `nor-bokmal`.

Overrides recorded in `registry/identifier_crosswalk.csv`:

| original | canonical | low(original) | final Glottocode | flag |
|---|---|---|---|---|
| gal | glg | galo1243 (Galoli) | gali1258 | false friend in low |
| yai (Yaitepec Chatino) | ctp | yagn1238 (Yaghnobi) | west2644 | false friend in low |
| zen (Zenzontepec Chatino) | czn | zena1248 (Zenaga) | zenz1235 | false friend in low |
| fre, est, fas, ara, aze, yid, nob, pus, sqi | fra, ekk, pes, arb, azj, ydd, nor, pbt, als | – (macro/legacy code not in low) | stan1290, esto1258, west2369, stan1318, nort2697, east2295, norw1258, sout2649, tosk1239 | original not in low |

`frm` (Middle French) is not in `low`. It is extinct and excluded.

## 6. Grouping (contract §3)

The stage builds union-find components within variety_id+pos, across all ingested
resources. Two lemmas join if they share:

* the same normalised label (NFC, casefold, whitespace-collapsed), or
* an identical source-cell form within one representation. Every non-missing variant of
  the source cell is used, which is conservative.

Group ids are `{variety}.{pos}::g{n:06d}`, numbered by the sorted smallest lemma_id. Ids
are deterministic for a given ingested set. Adding a resource renumbers them.

## 7. Pilot eligibility (outputs/pilot_v1)

The unit is source NFIN plus the 8 configured cells. A lemma is eligible if the source
and all panel cells are non-missing at variant 0.

| | ita.V.orth.mgn | fin.V.orth.mgn |
|---|---:|---:|
| lemmas / groups | 10,009 / 10,001 | 7,724 / 7,704 |
| cells | 48 | 137 |
| eligible lemmas / groups | 9,983 / 9,975 | 7,724 / 7,704 |
| missing rate: all cells / task cells | 0.18 % / 0.11 % | 0 / 0 |
| variant rate | 0 | 0 |
| multiword task forms | 17.2 % | 1.35 % |
| multi-lemma groups (max size) | 7 (3) | 16 (5) |
| meets inventory_size 1200 | yes | yes |

**Near-duplicate paradigms not caught by contract §3** (`*_derived_paradigms.csv`):

* **Italian pronominal verbs.** 2,208 eligible lemmas are clitic infinitives such as
  `abbandonarsi`, `andarsene` or `farcela`. Their finite forms are the base verb's forms
  with proclitics (`mi abbandonai`). For 1,917 of them the base verb is present. These
  forms are the 17 % multiword task forms. Contract §3 does not join them to their base,
  so a test `abbandonarsi` can be predicted from a trained `abbandonare`. With pronominals
  joined to their base, 8,059 eligible groups remain. Excluding all derived lemmas leaves
  7,775 eligible lemmas.
* **Finnish phrasal/idiomatic lemmas.**
  * 104 eligible lemmas have a multiword NFIN, for example `ajaa karille`.
  * 23 more have a multiword label but a one-word NFIN. These are already grouped with
    the base verb by the source-form rule, for example `ajaa partansa` with `ajaa`.

Both need a main-agent decision (§11).

## 8. Broad audit and suitability (`eligibility/broad_audit.csv`, `registry/registry.csv`)

The audit derives a plausible task for each MGN V/N resource without outcome information:

* **Verbs:** source NFIN. The 8 pilot slots map to the cell with the slot's features and
  the fewest extra features. Non-indicative moods, NEG/PASS, participles and similar
  cells are avoided, and ties go to the cell with higher coverage. A slot that cannot be
  filled falls back to the best-covered remaining cell, which is flagged.
* **Nouns:** the source preference is NOM;SG, then INDF/NDEF;SG, then SG, then ACC;SG
  (flagged). The panel is the 8 best-covered other cells.

Suitability is assigned as follows:

* `out_of_scope`: extinct.
* `not_assessed`: ADJ.
* `insufficient`: fewer than 600 eligible groups (inventory_size/2), or no source.
* `limited`: fewer than 1,200 eligible groups.
* `suitable_with_caveats`: at least 1,200 groups, but epitran, more than 5 % multiword
  panel forms, fallbacks, variants above 5 %, or unparseable cells.
* `suitable`: at least 1,200 groups and none of those caveats.

Living resources with at least 600 eligible groups:

| resource | repr. | source | eligible groups | panel multiword | GeLaTo statuses | suitability |
|---|---|---|---:|---:|---|---|
| mgn_data:ita-v | orth | NFIN | 9975 | 0.194 | accepted/ambiguous/excluded | suitable_with_caveats |
| mgn_data:fin-v | orth | NFIN | 7704 | 0.013 | accepted | suitable |
| mgn_custom:english-v | phon_custom | NFIN | 6037 | 0 | candidate | suitable_with_caveats (only 7 non-source cells) |
| mgn_data:spa-v | ipa_epitran | NFIN | 5430 | 0.043 | ambiguous/candidate | suitable_with_caveats |
| mgn_custom:french-v | phon_custom | NFIN | 5206 | 0 | accepted/ambiguous | suitable |
| mgn_data:hbs-v | orth | NFIN* | 4329 | 0.505 | ambiguous | suitable_with_caveats |
| mgn_data:deu-v | ipa_epitran | NFIN | 2878 | 0.102 | candidate | suitable_with_caveats (3PL slots by fallback) |
| mgn_custom:portuguese-v | phon_custom | NFIN | 1993 | 0 | unmatched | suitable |
| mgn_data:mkd-v | orth | NFIN* | 1961 | 0.145 | unmatched | suitable_with_caveats |
| mgn_data:cat-v | ipa_epitran | NFIN | 1505 | 0.001 | candidate | suitable_with_caveats |
| mgn_data:ron-v | ipa_epitran | NFIN | 1215 | 0.009 | ambiguous/candidate | suitable_with_caveats |
| hye-v, sme-v, bod-v, bul-v, ell-v | orth | (NFIN*) | 669–903 | | | limited |
| mgn_data:pol-n | ipa_epitran | NOM;SG | 169918 | 0 | candidate | suitable_with_caveats |
| mgn_custom:russian-n | phon_custom | NOM;SG | 43439 | 0 | candidate (15 pops) | suitable |
| mgn_data:fin-n | orth | ACC;SG (=NOM;SG) | 42752 | 0.007 | accepted | suitable_with_caveats |
| mgn_data:kmr-n | ipa_epitran | INDF;NOM;SG | 14680 | 0.066 | candidate | suitable_with_caveats |
| mgn_custom:hungarian-n | phon_custom | NOM;SG | 12727 | 0 | candidate | suitable |
| mgn_data:deu-n | ipa_epitran | NOM;SG | 11095 | 0 | candidate | suitable_with_caveats (6 cells only) |
| hbs-n, ell-n | orth | NOM;SG | 9964, 7274 | | ambiguous | suitable |
| hye-n*, nob-n*, kat-n, isl-n, ces-n | orth | | 3092–4916 | | candidate | suitable / with caveats |
| tur-n, ukr-n | ipa_epitran | | 2922, 1226 | | candidate | with caveats |

\* source reconstructed from the lexeme label (§4.2), unverified.

## 9. English, German and Romance: MGN versus UniMorph

The UniMorph repositories other than `ita` and `fin` were not fetched (no broad
downloads). Statements about them are therefore about *type*, not verified counts.

* **English.** MGN uses a data-custom phonemic table: 6,064 verbs, 8 cells (inf, pres1s,
  pres3s, presothers, past13, pastnot13, ppart, prespart). Source undocumented, with a
  CELEX-like layout. It cannot supply 8 distinct target cells: there are 7 non-source
  cells, and `past13`/`pastnot13` differ only for *be*. UniMorph `eng` (Wiktionary)
  offers more lemmas in orthography but only about 5 cells (NFIN, 3SG PRS, PST, PRS/PST
  participles). Neither supports the 8-cell panel. English is better treated as a
  low-cell control than as a pilot language. Its GeLaTo populations (Cornwall, Kent) are
  candidates only.
* **German.** MGN `deu-v` (2,970 verbs, 21 cells) and `deu-n` (12,090 nouns, 6 cells) are
  epitran `deu-Latn` transcriptions of UniMorph `deu`. Spot checks show systematic G2P
  errors:
  * initial *b* is devoiced (`bratschen` → `praːʧən`);
  * vowel length is wrong before double consonants (`erstrecken` NFIN `eəstreːkən`
    vs 3SG `eəstrekt`).

  Separable verbs make 9.7 % of forms multiword (`kyndiçt an`). The 3PL cells were
  dropped as lexicon-wide duplicates of 1PL. UniMorph `deu` in orthography would remove
  the G2P noise and restore the dropped syncretic cells, and it would match the pilot's
  `orth` representation. Replacing MGN would be a substitution of resource and should be
  a declared decision. It is not applied here.
* **Romance.**
  * `ita-v` (orth) is exactly UniMorph `ita` and adequate. Its pronominal-verb
    duplication is a property of UniMorph `ita`, not of MGN.
  * `spa-v`, `cat-v` and `ron-v` are epitran over UniMorph. Spanish epitran uses
    *seseo* (`mezclar` → `mesklaɾ`), a Latin-American rather than Castilian
    pronunciation. That variety does not match the GeLaTo Spanish samples from Spain.
    Spanish also has clitic multiword forms (11.8 %).
  * French (5,249 verbs, 51 cells) and Portuguese (1,996 verbs, 69 cells) are data-custom
    phonemic resources. The French one uses Flexique-style archiphonemes E/O. Both are
    adequate in size and coverage.
  * gal, ast, oci, fur, lld and vec are small UniMorph orthographic sets (168–486 verbs)
    and insufficient for the inventory.
  * UniMorph `spa`, `cat`, `ron`, `fra` and `por` would give orthographic alternatives.
    For spa/cat/ron they are the very sources that MGN transcribed.

## 10. GeLaTo crosswalk

**Sources.** The stage reads the cached files of
`analyses/gelato_feasibility_2026_10_01`, read-only. It never reads ancestry values. The
derived files are opened with `usecols` for population keys only. `ancestry_sources.csv`
keeps three layers apart:

* **population metadata:** GeLaTo c625fdc `cldf/populations.csv` (main_397), Zenodo
  15263706 Table S1 (expanded_558, curated GeLaTo/GBI/TLI Glottocodes), and the 653-row
  population mapping;
* **ancestry inference:** `GeneticInfoID.csv`, whose row order is the Q-row order, and
  the 19 best-run Q matrices `GelatoHO_mergedSetMarchBEDnorelatives_pruned_autosomes_K{12..30}.Q`
  (HO merged set, no relatives, LD-pruned autosomes; component ids are local to each K);
* **derived summaries:** `audit.py` K23 population means, K12–K30 diagnostics and K
  sensitivity.

All sha256 hashes match the audit's `provenance.json`.

**Candidates.** For each resource Glottocode, the stage collects:

* populations with that Glottocode in main_397 or as Table S1 `GeLaTo_Glottocode`
  (exact);
* Table S1 rows linking to it only through GBI/TLI proxies;
* the audit's descendant or Greek candidates.

Each population is one row, with panel membership, sample sizes per panel, coordinates,
location, publication, curation notes, the GeneticInfoID label and row count, the Q-file
pattern and K range, and the derived-summary file and row key. Populations are never
averaged.

**Statuses.**

* Exact Glottocode: `candidate`.
* Proxy or descendant only: `ambiguous`.
* No population: `unmatched`.
* Extinct language: `excluded`.
* `accepted` (and reviewed ambiguous/excluded) come **only** from
  `morph_ldl/data/gelato_review.yaml`. An `accepted` entry must have variety, community,
  locality and source-period reviews, plus evidence, reviewer note, reviewer and date.
  Incomplete entries are ignored and reported.

Counts over 236 rows and 135 resources: candidate 114, ambiguous 53, unmatched 40,
excluded 24, accepted 5. Of the living languages, 37 have at least one exact-Glottocode
population. That is more than the audit's 32 because the registry also covers MGN files
outside the 64-language modelling sample (for example ast, mlt) and expanded-panel
exact matches.

**Reviews (summary; full text in the YAML).**

* **Italian**
  * `Tuscan` (n=8) is **accepted**: Standard Italian is Tuscan-based, and Tuscan
    vernacular is Italian; the locality is a regional centroid.
  * `Bergamo` (n=13) is **ambiguous**. The traditional vernacular is Eastern Lombard
    (Bergamasque), and the community underwent a shift to Italian. GeneticInfoID
    glottocodeBase is NA. Use it in sensitivity analyses only.
  * `Italian_South` (napo1241, n=3), `Italian_Calabria` (neap1235, n=1) and
    `Sicilian_East/West` (sici1248, TLI proxy only) are **excluded**.
* **Finnish:** `Finnish` (n=8, Helsinki, automatic assignment) is **accepted**, with
  small-sample and L1-not-recorded caveats. The same review applies to fin-n and fin-adj.
* **French** (the recommended alternative):
  * `French_Central` (n=26, national centroid) is **accepted**.
  * `French_Northwest` (Pas-de-Calais, Picard area), `French_East` (Moselle,
    Lorraine-Franconian area), `French_South` (Lourdes, Gascon) and `French_West`
    (Finistère, Breton area) are **ambiguous**, on the same shift logic as Bergamo.

## 11. Recommendation, unresolved issues and proposals

**Pilot recommendation.** Keep **Italian + Finnish verbs, orthographic**. Both are
untouched UniMorph orthography: Italian is verified identical, Finnish comes from an
older release. Each has more than 7,700 eligible groups, 0 % variants, almost no missing
cells, all configured cells present, and exactly one accepted population (n=8 each).

The Italian pronominal duplicates must be resolved first. The preferred option is to
join `pronominal_of` lemmas to their base group (contract §3 amendment). The alternative
is to exclude derived lemmas from the inventory. The best-supported alternative or
additional verb unit is **French verbs** (`mgn_custom:french-v`). It has 5,206 eligible
groups, all 8 template slots without fallback, 0 % multiword, and an accepted
`French_Central` population. It is `phon_custom`, however, so it adds a representation
difference. Do not mix it with orth units without declaring that difference. For a noun
unit, Russian (`phon_custom`, 15 candidate populations) and Hungarian (`phon_custom`)
are the strongest; neither is reviewed.

**Proposals for the main agent:**

1. Contract §3: add a rule joining derived paradigms (Italian clitic infinitives to
   their base). Decide whether multiword-source lemmas (Finnish phrasal idioms, 104) are
   eligible.
2. Contract §1/§2: define segmentation for `ipa_epitran` (the proposal in §3 above).
   Note that "variant 0" in MGN-wide is alphabetical.
3. Contract §1: add `mgn_custom:<stem>` as the collection name for data-custom.
4. Treat MGN cell absence as possibly a dropped duplicate (§4.2) whenever a configured
   cell is missing for a new language.

**Unresolved:**

* The exact UniMorph release behind MGN `fin`.
* The sources of the data-custom files (French, English, Portuguese, Russian, Hungarian,
  Latvian, Arabic, Navajo, Chatino).
* Individual L1 is unrecorded for all GeLaTo samples.
* All accepted populations have small samples (8–26).
* The epitran quality of every `ipa_epitran` file is unverified beyond spot checks.
* No review exists yet for nouns other than Finnish, nor for any language besides
  Italian, Finnish and French.

## Main-agent integration decisions (2026-10-06, after the independent review)

* **Acceptance needs human confirmation.** The reviews in `gelato_review.yaml` were written
  by an agent. They can now only *propose* acceptance. A row is `accepted` only when its
  review also carries `confirmed_by` and `confirmed_date` filled in by a person. Until then
  the outputs show `match_status = candidate` and `proposed_status = accepted` for Tuscan,
  Finnish and French_Central, and for the two other proposed rows. The crosswalk counts
  are now: candidate 119, ambiguous 53, unmatched 40, excluded 24, accepted 0. Where the
  sections above say **accepted**, read "proposed accepted, pending confirmation".
* **Eligibility exclusions.** These are declared in `task.eligibility`, before any
  acquisition, and apply to every unit. Lemmas listed in
  `eligibility/<unit>_derived_paradigms.csv` are excluded: Italian pronominal and clitic
  verbs, and Finnish multiword or idiom entries. Lemmas with any multiword task form are
  also excluded. This leaves 7,767 Italian and 7,597 Finnish eligible lemmas.
* **Auxiliary set.** Some eligible lemmas lie outside the inventory. A further auxiliary
  sample of them is used for LDL setting choice and for selector copy anchors
  (`splits/<unit>/auxiliary_manifest.csv`).
* **Group ids depend on the ingest set.** Adding another resource for the same variety
  renumbers groups and therefore reshuffles inventories and folds. Freeze the ingest set
  for a given experiment id.

## PCFP integration (2026-10-08, pcfp_v1)

* **Same ingest set, same group ids.** `configs/pcfp_v1.yaml` ingests the same two MGN
  files as pilot_v1, and the grouping cell is still NFIN (the citation cell), so group ids
  are identical. The data stage now accepts a unit's `cells` as a list of declared
  eligible cells (`config.unit_cells`). Its 8-slot panel checks then run over all
  declared cells.
* **Single-word cell rule** (`splits/<unit>/cell_inventory.csv`). Italian: all 48 cells
  are single-word. 45 of them contain one stray multiword form (≤ 0.14%), which makes 12
  verbs (1 of them only through it) ineligible under the completeness rule. Finnish: 35 of 137 cells are single-word.
  The other 102 are ≥ 99% multiword (negative forms *en aakkosta*, perfect/pluperfect
  *olen aakkostanut*, and so on). They exist by resource construction (UniMorph
  periphrastic cells), and Italian MGN has no compound tenses.
* **Eligibility.** Derived paradigms are excluded as before, and a complete single-word
  paradigm is required. This leaves 7,767 Italian verbs (7,797 non-derived, 30 lost: 29
  with a missing cell, 11 of which also have a multiword form, and 1 with only a stray
  multiword form) and 7,597 Finnish verbs (none lost).
* **Citation label.** The selector reads the lemma label (`lemma_label`) and segments it
  with the orth rule (`pcfp.citation_segments`). The label equals the NFIN form for
  99.9% of eligible Italian verbs and 100% of Finnish verbs. The exceptions are the
  UniMorph NFIN errors noted in §4.4.
* **Grambank typology outcome.** The new `typology` stage reads Grambank v1.0.3 and
  Glottolog CLDF v5.3 (pinned in `scripts/fetch_external.sh`) and the GeLaTo population
  metadata only: `populations.csv` and Table S1, never `GeneticInfoID.csv`, the Q
  matrices or the audit's derived files. See [TYPOLOGY.md](TYPOLOGY.md). Its population
  links reuse the §10 statuses and the human-confirmation rule.
