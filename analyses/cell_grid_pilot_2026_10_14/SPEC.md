# Universal verb-cell grid: draft specification (Phase 1, for review)

Status: **draft, 2026-10-14. Partly superseded by [DECISIONS.md](DECISIONS.md)** (D1 option C, D2 clitics are separate words, D3 future deferred → G14, D8 R5 changed). No language has been mapped.
Files in this folder: [README.md](README.md) (summary and review questions),
[grid.csv](grid.csv) (the cells and context frames),
[mapping_template.yaml](mapping_template.yaml) (per-language mapping format),
`grambank_frequencies.py/.csv` and `availability_check.py/.csv` (the numbers below).

No ancestry or admixture data was read. Nothing below was chosen by looking at an
admixture association.

---

## 1. Opinion on the fixed-grid design

I agree with the design. The grid gives every language the same paradigm size and the
same LDL meaning space: the same feature vectors and the same `sem_sd_cell` vectors for
the same 16 cells. Syncretism then makes a language easier, which is the intended notion
of simplicity. These are the points I would watch:

1. **Overlap with Grambank.** The main-verb rule turns periphrasis into syncretism. So the
   whole-grid score largely encodes how many grid meanings a language expresses
   synthetically, and Grambank `tense` and `person_indexing` already code that. The
   distinct-cells score addresses this. I also suggest reporting the number of distinct
   syncretism classes per language (§4, R6) and checking its correlation with Grambank
   extent before the LDL score is used.
2. **Copy items.** If a hidden cell has a shown cell in the same syncretism class, the
   model only has to copy that form. English has about 3 distinct forms in 16 cells, so
   its whole-grid accuracy will be near the ceiling for most k. This is intended, but it
   compresses the scale among strongly inflecting languages. Scoring by class (one item
   per class that has no shown member) may be the better form of the distinct-cells
   score. The scoring rule is part of the later LDL decision, but the mapping files must
   carry the class ids, and they do (R6).
3. **Judgement-sensitive languages.** Some single decisions change a language's class
   count a lot:
   * Italian PST: with *passato prossimo* and the main-verb rule, the past has 1–2
     classes; with *passato remoto* it has 6.
   * Serbo-Croatian FUT: 1 class if the clitic is a separate word (*pisat ću*), 2 if it
     is part of the word (Serbian spelling *pisaću*).

   The mapping format therefore keeps the rejected alternative for each judgement call
   (`alternatives:`), so the outcome can be recomputed under it as a sensitivity check.
4. **Narrower construct.** 16 cells replace 48 (Italian) and 35 (Finnish). The LDL score
   becomes "predictability of a core finite grid". The subjunctive, conditional,
   imperfect and non-finite forms are left out. This is acceptable if the reports say
   so.
5. **What the grid does not equalise:**
   * **Representation.** English and Hindi data are phonemic; the other data are
     orthographic.
   * **Lemma inventory.** Light-verb lemmas (Hindi), aspect partners (Slavic), and gaps
     in common verbs: UniMorph `ita` has no *parlare*, *mangiare*, *cantare*, *essere*
     or *avere*.

   These need their own rules (R8, §5.2). They are outside the grid itself.

---

## 2. The grid (G16, recommended)

Sixteen cells: present × 6 persons, past × 6 persons, future 1SG and 3SG, and imperative
2SG and 2PL. The context frames are in [grid.csv](grid.csv).

| block | cells | context (summary) |
|---|---|---|
| PRS | 1SG 2SG 3SG 1PL 2PL 3PL | habitual present: "Every morning I V (it)" |
| PST | 1SG 2SG 3SG 1PL 2PL 3PL | one completed event yesterday: "Yesterday I V-ed (it), once" |
| FUT | 1SG, 3SG | 1SG intention: "[I have decided:] Tomorrow I will V (it)"; 3SG prediction: "[I am sure:] Tomorrow he will V (it)" |
| IMP | 2SG, 2PL | positive command now, to one friend or to a large group |

### 2.1 Cross-linguistic frequency

Grambank shares are from the pinned v1.0.3: language-level rows, codes 0/1 only. The
family-weighted share is the mean of the per-family shares, with isolates counted as
families ([grambank_frequencies.csv](grambank_frequencies.csv)). WALS counts were read
from WALS Online on 2026-10-14.

| category | Grambank (raw / family-weighted) | WALS | role in grid |
|---|---|---|---|
| person indexing of S | GB089 ∨ GB090: 0.68 / 0.82 (n = 2,338) | 102A: the A argument is marked in 266/378 (only A 73 + both 193) | 6 persons in PRS and PST |
| past | GB083: 0.52 / 0.62 | 66A: past/non-past marked in 134/222 | PST block |
| future | GB084: 0.45 / 0.62 | 67A: inflectional future in 110/222 | 2 cells (see 2.2) |
| present | GB082: 0.34 / 0.42 | (no chapter) | PRS block: every language has *some* form for this context, even when it is unmarked |
| imperative | (GB312 covers all moods: 0.70 / 0.81) | 70A: a dedicated 2nd-person imperative in 426/548; 2SG ≠ 2PL in 292/548 | IMP 2SG, 2PL |
| perfective/imperfective | GB086: 0.67 / 0.72 | 65A: 101/222 | not a dimension. The contexts fix aspect (PRS habitual = imperfective, PST = perfective), so aspect languages fill the cells with their aspect forms |
| optative | — | 73A: 48/319 | excluded (rare) |
| P (object) indexing | GB093 ∨ GB094: 0.51 / 0.66 | 102A: P marked in 217/378 (only P 24 + both 193), plus 6 with A or P | default, not cells (§5.3) |
| clusivity | GB028: 0.53 / 0.45 (pronouns **or** indexing) | 39A (verbal inflection) not checked | default, not cells |
| dual for all persons | GB031: 0.26 / 0.30 | — | default, not cells |

Subjunctive and conditional are excluded. Their contexts are construction-specific
(complement clauses, conditionals). WALS has no single comparable chapter for them, and
the frequency of a dedicated form is not established here.

### 2.2 Why only two future cells

* **Room for the imperative.** PRS × 6 + PST × 6 + FUT × 6 already reaches the 18-cell
  cap, which leaves no room for the imperative. The imperative is more frequent than an
  inflectional future (78% vs 50% in WALS).
* **Six future cells add little.** Future person marking almost always repeats the
  present's agreement set. In a periphrastic-future language (English, Finnish,
  Bulgarian, Croatian spelling), six future cells would be six rule-filled copies: 6 of
  18 cells (a third of the grid) would encode "future is not synthetic".
* **Two cells with different meanings.** FUT.1SG (intention) and FUT.3SG (prediction)
  are the two future uses distinguished by Dahl (1985) and by Bybee, Perkins & Pagliuca
  (1994). They also show whether the future carries person.

### 2.3 Alternatives (for the review)

* **G18:** PRS × 6, PST × 6, FUT × 6, no imperative.
* **G14:** PRS × 6, PST × 6, IMP × 2, no future.
* **G13:** PRS × 6, PST × 6, FUT.3SG. This is the minimum.

In every variant the grid is fixed before any mapping, and the same for all languages.

**No citation cell.** Many languages have no infinitive (Bulgarian, Modern Greek). A
citation-form cell would therefore not mean the same thing across languages. A form
that the main-verb rule needs (e.g. the English or Croatian infinitive in the future)
enters the grid only as the form of a grid cell.

---

## 3. Context conventions (defaults that fill the extra dimensions)

These are part of every context frame ([grid.csv](grid.csv)):

| dimension | default | why |
|---|---|---|
| subject | a pronoun, not a full NP | some languages agree only with pronominal subjects (Welsh, Irish) |
| gender | masculine speaker, addressee and 3SG referent | the least-marked agreement value in all pilot languages; arbitrary but fixed |
| 2nd person | informal; 2SG = one addressee, 2PL = a large group | avoids polite-plural syncretism being counted as number |
| plural | a large group ("all the villagers") | selects the plural, not a dual or paucal |
| 1PL | exclusive (speaker and others, not the addressee) | the natural extension of 1st person; inclusive is the alternative (§5.3) |
| object | a verb with a transitive use takes a definite 3SG object "it" (D29) | see §5.3 |
| evidentiality | the speaker witnessed the event (direct evidence) (D23) | selects the witness past where a language marks it (Armenian aorist); no effect elsewhere |
| polarity, clause | affirmative, declarative main clause (IMP: positive command) | negative and subordinate forms are separate paradigms in many languages |
| PST distance | "yesterday" | as decided. In languages with remoteness tenses (38 + 2 of 222, WALS 66A) this selects the hesternal or recent past, not a general past. Flag this per language |

---

## 4. Rules

**R1. Context rule.** A grid cell's form is the form the lexical verb has when the
context sentence is said in the language's standard variety. The mapping is made at
paradigm level, from grammars: one mapping per cell, not per lemma. Lexical classes
(e.g. Italian auxiliary selection) are allowed, but each must be declared with its
membership source.

**R2. Main-verb rule (unexpressed = maximal syncretism).** The meaning may have no
dedicated inflection: it is periphrastic (auxiliary or particle + verb) or unmarked.
Then the cell's form is the single-word form that the lexical verb has in the
construction that expresses the meaning. Examples:
* English FUT → bare *walk*;
* French PST → *mangé*;
* Bulgarian FUT (*ще* + present) → the present form;
* a tenseless language's PST → the bare or perfective form.

If the lexical verb agrees inside the construction (Italian *siamo arrivati*, Croatian
*pisali smo*), the cell takes that agreeing form. (Option B in §5.1 replaces this rule
for auxiliary constructions.)

**R3. Competing forms.**
1. Use the form that the reference grammar gives for the context.
2. If the grammar gives several forms as equally natural, use the least-marked one, in
   this order:
   * not restricted by register or region;
   * single word over multiword;
   * fewest added morphemes.
3. Record the rejected form under `alternatives:` (§1 point 3).

Corpora are not used.

**R4. Syncretism.** If the language uses one form for two grid cells, the mapping says
so (`rule: syncretism`, `same_as: <cell>`) and cites the grammar.
* MGN sometimes drops a cell because it duplicates another cell in every lemma:
  * Serbo-Croatian FUT 3PL = 3SG (*će*);
  * Bulgarian aorist 3SG = 2SG;
  * Bulgarian PRS 1SG, which is the lexeme label.

  Such a cell is restored by a syncretism rule with a grammar reference. Where possible
  the restored cell is checked against another source.
* Accidental identity of forms in some lemmas only (Italian *parla* PRS.3SG =
  IMP.2SG in first-conjugation verbs) is not a paradigm-level syncretism unless the
  grammar states it for the class. Declare it per class.

**R5. Missing forms.** If a verb lacks a needed source form after R2–R4, the verb is
excluded, never filled ad hoc. Defective verbs (impersonals without 1SG) are excluded
too. The mapping reports the counts.

**R6. Formal-distinctness flag.** Each language partitions the 16 cells into syncretism
classes (`distinct_class`). Cells joined by R2 or R4 share a class. A cell is
*formally distinct* if its class has one member. The filled grid carries both the class
id and the flag, so that either scoring rule (cell-level or class-level) can be applied
later.

**R7. Representation.** Forms are taken as the dataset writes them (orthographic or
phonemic). Nothing is transcribed again. Some source forms contain material that is
not part of the cell:
* Hindi and Welsh forms include the subject pronoun;
* MGN Serbo-Croatian and Armenian forms include the auxiliary.

Such material is removed by a declared token rule (`transform:`), never by hand.

**R8. Lemma scope.** The lemmas are non-derived, single-word verbs with a complete grid.
The following are excluded and counted:
* multiword lemmas (Hindi *X karnā* compounds, 85 of 258);
* clitic or pronominal derived paradigms (as in the existing eligibility rules).

The Slavic aspect rule is open (§5.2).

**R9. Sources.**
* Every cell mapping cites a grammar (author, year, section or page).
* The Grambank codings GB082–GB084, GB086, GB089–GB094, GB119–GB121 and their cited
  sources are a cross-check. A mapping that contradicts a Grambank coding must explain
  why.
* Where the form for a context is uncertain, the mapping says so (`confidence: low`) and
  does not guess.

---

## 5. Open questions: options

### 5.1 Inflection carried by an auxiliary

* **A. Single word (main-verb rule only).** The cell is the lexical verb's form, and the
  auxiliary is dropped. The cell gets `aux_carried: true` when the auxiliary carries a
  grid distinction.
* **B. Multiword.** The cell is "lexical verb + auxiliary" as one string, joined by a
  space, in the order of the context sentence. The auxiliary is in the form it has in
  that sentence.
* **C. Hybrid.** The grid form is single-word (A), but the distinctness flag counts
  auxiliary distinctions (as B). LDL scores the lexical verb only, and the class count
  follows B.

Pilot illustration (one verb each, forms from the data; English and Italian auxiliaries
are added from the grammar and are not in the data):

| language, verb | cell | A (single word) | B (multiword) |
|---|---|---|---|
| Italian *arrivare* (essere) | PST.1SG / PST.1PL | arrivato / **arrivati** (not in UniMorph) | sono arrivato / siamo arrivati |
| Italian, avere verb (*abalienare*) | PST.1SG / PST.3PL | abalienato / abalienato | ho abalienato / hanno abalienato |
| English *walk* | FUT.1SG = FUT.3SG | wɔːk | wɪl wɔːk |
| Finnish *syödä* | FUT.1SG / FUT.3SG | syön / syö (= PRS; no future) | same |
| Serbo-Croatian *pisati* | PST.1SG / PST.3SG / PST.1PL | pisao / pisao / pisali | pisao sam / pisao je / pisali smo |
| Serbo-Croatian *pisati* | FUT.1SG / FUT.3SG | pisat / pisat | pisat ću / pisat će |
| Eastern Armenian *աբսորբել* | PRS.1SG / PRS.3SG / PRS.1PL | աբսորբում ×3 | աբսորբում եմ / է / ենք |
| Eastern Armenian | FUT.1SG / FUT.3SG | աբսորբելու ×2 | աբսորբելու եմ / է |
| Hindi *khelnā* | PRS.1SG / PRS.3SG / PRS.1PL | kʰeltaː / kʰeltaː / kʰelte | kʰeltaː ɦũː / kʰeltaː ɦæː / kʰelte ɦæːn |
| Hindi *khelnā* | PST.1SG | kʰelaː (no auxiliary in either option) | kʰelaː |

Number of syncretism classes in G16 (out of 16). These are **provisional**, worked out
from the grammars for regular verbs, and will be checked in Phase 2:

| language | A | B | what changes |
|---|---:|---:|---|
| English | 3 | 4 | FUT becomes *will V* |
| Finnish | 14 | 14 | nothing (FUT = PRS either way) |
| Italian (avere verbs) | ~9 | ~14 | PST 1 → 6 classes |
| Serbo-Croatian | 11 | 16 | PST 2 → 6, FUT 1 → 2 |
| Eastern Armenian | 10 | 16 | PRS 1 → 6, FUT 1 → 2 |
| Hindi | ~8 | ~10 | PRS 2 → 4. The 2SG pronoun choice (*tū* vs *tum*) matters |

Consequences:
* Under A, Armenian and Serbo-Croatian look less inflected than English-type languages
  only in the auxiliary tenses. Under B they are fully distinct.
* Under B, the auxiliary part of a form is the same in every lemma, except where the
  auxiliary depends on the lemma's class (Italian *essere/avere*, Basque intransitive vs
  transitive). The LDL model can therefore produce it from the cell's meaning alone. B
  mainly raises the distinctness count, and its LDL cost is the lemma's class
  (auxiliary selection).
* B needs a representation decision. The current pipeline excludes cells that are
  mostly multiword (PROTOCOL §2), so the space would have to become a segment, with
  n-gram cues that cross it.
* B also needs a fixed word order. Croatian clitics follow the first constituent
  (*pisao sam* vs *Jučer sam pisao*). The proposal is the verb-initial order that the
  data use.
* C keeps LDL single-word and lets the distinct-cells score credit the auxiliary
  distinctions.

My lean: **C**, or B if the LDL representation can handle spaces. A alone hides real
inflection in Armenian-type languages.

### 5.2 Slavic aspect pairs

MGN `hbs-v` and `bul-v` contain both partners as separate lemmas, with no aspect tag
(*pisati / napisati*, *пиша / напиша*). The grid contexts select aspect: the PRS
habitual takes the imperfective; the PST "once, yesterday" context, the intention FUT
and the single-action IMP prefer the perfective.

| option | Serbo-Croatian grid for 'write' | pro | con |
|---|---|---|---|
| **S-a.** Imperfective (and biaspectual) lemmas only; each lemma's own forms | pišem … / pisao (sam) … / pisat (ću) / piši | lexeme = unit, as in other languages; no derivation in the task | departs from R1 in PST/FUT/IMP (the imperfective general-factual past is acceptable but not the default); needs aspect tags (not in MGN; Wiktionary categories would be the source) |
| **S-b.** All lemmas, own forms, aspect treated as lexical | for *napisati*, PRS = *napišem* (formal present) | no aspect tags needed; simplest | perfective PRS forms do not mean "habitual present" by default; R1 is broken for half the lemmas |
| **S-c.** Linked pairs: grid lexeme = pair, PRS from imperfective, PST/FUT/IMP from perfective | pišem / napisao / napisat / napiši | follows R1 exactly | needs a pair list; the prefix choice (*na-*, *u-*, *s-*…) is derivational and unpredictable, so the Slavic LDL difficulty partly measures derivation; unpaired verbs have no grid |
| **S-d.** S-a, with the PST context relaxed to "aspect-neutral past" for all languages | as S-a | consistent rule across languages | changes the fixed grid; aspect languages lose the PFV/IPFV contrast that the context now carries |

My lean: **S-a**, with the departure from R1 declared as a language-family rule. The
same problem exists outside Slavic: Georgian preverbs, Hindi compound verbs (*khā
liyā*), and Lithuanian prefixed perfectives. So the choice is a general "lexicalised
aspect" rule, not a Slavic exception.

### 5.3 Object agreement, dual, clusivity

| dimension | default proposed | alternative | what is lost | share of languages affected |
|---|---|---|---|---|
| object | transitive verbs: definite 3SG object; intransitive verbs: none | (ii) the intransitive or indefinite-object form for every verb (Hungarian indefinite conjugation); (iii) separate object rows | all object person and number contrasts; in Hungarian, the definite/indefinite contrast | P indexing: 0.51 raw / 0.66 family-weighted (Grambank) |
| dual | not in grid; PL = a large group | add 1DU/2DU/3DU cells only where they exist (breaks the equal grid) | Slovene and Sorbian dual forms; Austronesian and Australian dual indexes | dual for all persons: 0.26 / 0.30 |
| clusivity | 1PL = exclusive | 1PL = inclusive | the other 1PL form | 0.53 / 0.45 (pronouns or indexing; verbal-only share not computed) |
| gender | masculine | feminine; or a gender-neutral form where one exists | feminine agreement forms (Hindi, Slavic past, Italian *essere* participles) | — |
| politeness | informal | polite | polite forms (Hindi *āp*, Czech *vy*) | GB415 not computed |

**Data note.** The object default needs the transitivity of each lemma. MGN and UniMorph
do not code it, except as argument tags in `.args` files. The pilot languages have no
object indexing, so the question does not arise in the pilot. A pilot with Basque,
Georgian or Hungarian would be needed to measure the cost.

### 5.4 New question: what counts as a word

Grambank counts clitics as bound ("forms a phonological unit with the verb", GB083/GB084
definitions). Orthography is not consistent about this: Croatian writes *pisat ću*,
Serbian *pisaću*. The main-verb rule and the multiword option both depend on the word
boundary. The options are:

* **(a)** the orthographic word of the dataset (simple, but Croatian and Serbian differ);
* **(b)** clitic auxiliaries are always separate words, whatever the spelling;
* **(c)** follow Grambank's boundness.

My lean: (b), so that A and B are defined the same way across standards.

---

## 6. Pilot languages: availability

Counts are from [availability.csv](availability.csv). A verb counts as "complete" when it
has every source cell that a plausible mapping would draw on. This is an availability
check, not a mapping.

| language | resource | verbs | complete | single-word lemmas | multiword share of needed forms | notes |
|---|---|---:|---:|---:|---:|---|
| Italian | UniMorph `ita` (pinned, `external/`) | 10,009 | 9,964 | 9,964 | 0.18 | has `V.PTCP;PST` (*arrivato*), which R2 needs for *passato prossimo*. MGN `ita-v` (9,975 complete) **has no participle**. Plural or feminine participles (*arrivati*) and auxiliary selection are not in the data. Common verbs are missing (*parlare*, *essere*, *avere*) |
| Finnish | MGN `fin-v` | 7,724 | 7,724 | 7,597 | 0.01 | clean. FUT = PRS (no future), IMP 2SG/2PL present |
| English | MGN `data-custom/english-v` (phonemic) | 6,064 | 6,059 | 5,935 | — | needs 6 source cells (inf, pres1s, pres3s, presothers, past13, pastnot13). Orthographic UniMorph `eng` would need a download |
| Eastern Armenian | MGN `hye-v` | 903 | 903 | 903 | 0.55 | PRS = converb + auxiliary, FUT = future converb + auxiliary, PST = synthetic aorist; converbs are separate cells. Grambank source: Dum-Tragut 2009 |
| Hindi | MGN `hin-v` (IPA) | 258 | 258 | 173 | 0.60 | small. Cell labels are inconsistent: `2;SG;HAB;PRS` holds the *tū*/3SG form, and future forms are tagged `SBJV` and include the pronoun (*mæːn kʰelũːɡaː*). UniMorph `hin` is 4.0 MB, so it may hold more lemmas than MGN kept; checking needs a download |
| Basque | UniMorph `eus` | 26 | — | — | — | synthetic verbs only; no periphrastic paradigms in MGN, UniMorph or the Paralex records surveyed on 2026-10-08. **Not usable** |
| Serbo-Croatian | MGN `hbs-v` | 4,348 | 4,324 | 4,324 | 0.58 | PST = perfect (l-participle + clitic), FUT = infinitive + clitic (Croatian spelling), aorist sparse. FUT 3PL dropped by MGN (= 3SG). No aspect tags |
| Bulgarian | MGN `bul-v` | 699 | 698 | 669 | 0.04 | synthetic aorist and imperfect; FUT = *ще* + PRS; PRS 1SG = lexeme label; aorist 3SG dropped (= 2SG). No aspect tags |
| Czech | MGN `ces-v` | 500 | 0 | — | — | **column labels inconsistent** (`IND;PST;2;SG` holds the present *absolvuješ*; `IND;PRS;2;SG` is NA in 367 verbs). MorphyNet `ces` (3,232 verbs) would need a download |
| Slovene | MGN `slv-v` | 104 | 103 | 101 | 0.20 | too small; PST/FUT/COND are unsplit by person (`IND;PST` = *sem amputíral* only) |

Proposed pilot set:

1. **Italian:** UniMorph `ita`, not MGN, because of the participle.
2. **Finnish:** MGN.
3. **English:** MGN phonemic.
4. **Eastern Armenian** instead of Basque/Hindi. Basque has no data. Hindi has only 173
   single-word verbs and needs relabelling, which is itself a finding for the cost
   estimate. Armenian is the clearest case of person carried by an auxiliary and has 903
   verbs. If you prefer Hindi, I can map it as well (a 6th language) or instead.
5. **Serbo-Croatian:** MGN `hbs-v`. It is the largest Slavic set, it has an auxiliary
   past and a clitic future (so it tests §5.1 and §5.4 as well as §5.2), and it has
   both aspect partners. The variety (Croatian vs Serbian standard) must be fixed. The
   spellings in the data (*pisat ću*, *grđaše*) need checking.

---

## 7. Mapping files and filled grids

* **Format:** [mapping_template.yaml](mapping_template.yaml). There is one file per
  language (`mappings/<iso>.yaml` in Phase 2), version-controlled.
* **Filled grid** (Phase 2, `outputs/cell_grid_pilot/<iso>_grid.csv`, not committed;
  small summaries go in this folder). It has one row per verb × grid cell, with these
  columns:
  * `lemma`, `grid_cell`, `form`;
  * `form_multiword` (option B, if adopted);
  * `rule` (`source` / `syncretism` / `main_verb` / `restored`);
  * `source_cell`, `transform`, `lexical_class`;
  * `distinct_class`, `is_formally_distinct`, `aux_carried`;
  * `mapping_version`.
