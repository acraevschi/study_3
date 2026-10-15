# English (eng) mapping notes

Mapping: `mappings/eng.yaml`. Data: MGN `data-custom/english-v` (CELEX, phonemic). Class lists:
`mappings/classes/eng_modal_aux.csv` (may, shall, must, ought after the fix round; see J5) and `mappings/classes/eng_excluded.csv` (would; M7).
Variant order fixed 2026-10-14 with `variant_overrides` (J7). Status: reviewed (Opus reviewer + fix round 2026-10-14); mapped by Claude Sonnet 5.5, 2026-10-14. Check after the fix round: 0 errors, 0 warnings (see "Fix round 2026-10-14", section 8). Sections 1-7 are the original notes; where the fix round changed them, a line marked **[fix round]** says so.

## 1. Summary table

| cell | rule | source cell | class | aux_carried | confidence |
|---|---|---|---|---|---|
| PRS.1SG | source | pres1s | PRS.NON3SG | false | high |
| PRS.2SG | source | presothers | PRS.NON3SG | false | high |
| PRS.3SG | source | pres3s | PRS.3SG | false | high |
| PRS.1PL | syncretism (= PRS.2SG) | presothers | PRS.NON3SG | false | high |
| PRS.2PL | syncretism (= PRS.2SG) | presothers | PRS.NON3SG | false | high |
| PRS.3PL | syncretism (= PRS.2SG) | presothers | PRS.NON3SG | false | high |
| PST.1SG | source | past13 | PST.ALL | false | high |
| PST.2SG | source | pastnot13 | PST.ALL | false | high |
| PST.3SG | syncretism (= PST.1SG) | past13 | PST.ALL | false | high |
| PST.1PL | syncretism (= PST.2SG) | pastnot13 | PST.ALL | false | high |
| PST.2PL | syncretism (= PST.2SG) | pastnot13 | PST.ALL | false | high |
| PST.3PL | syncretism (= PST.2SG) | pastnot13 | PST.ALL | false | high |
| IMP.2SG | source | inf | PRS.NON3SG | false | high |
| IMP.2PL | syncretism (= IMP.2SG) | inf | PRS.NON3SG | false | high |

No token transforms (English forms are phoneme strings; none are needed). No cell has an auxiliary
carrying a distinction (option C is not triggered: the preterite is single-word, and the perfect
with *have* is not used for the PST frame, see J1). No `confidence: low` cells. Confidence is `high` for the
cells because the forms are standard and a source was seen. The weak points are in the lemma-level
data (section 4), not in the cell mapping.

## 2. Judgement calls

### J1. PST: preterite vs present perfect
* Options: (a) simple past (preterite) *walked*; (b) present perfect *has/have walked* (main verb = past participle).
* Choice: (a), source `past13` / `pastnot13`.
* Reason: the frame is "Yesterday I V-ed (it), once": a finished time with a definite time word.
  The simple past is the tense used for a completed action at a definite past time; the perfect is
  for an open time period. English has no remoteness tenses, so the "yesterday" selection is
  unmarked.
* References (all seen, see section 6):
  * Purdue OWL, "Active Verb Tenses": simple past is for "Completed Action" (example *We visited the museum yesterday*).
  * UVic Study Zone, "Present Perfect: Present Perfect or Simple Past?": "The simple past is usually used with definite times".
* Confidence: high.
* Alternative (`alternatives:`): present perfect under D1 (form = past participle `ppart`; `have`/`has`
  would be aux_carried; differs from the preterite for about 200 irregular verbs).

### J2. past13 vs pastnot13 (differ only for *be*)
* The dataset has two preterite cells, `past13` (1st/3rd person) and `pastnot13` (other persons).
  They differ only for *be* (*was* vs *were*). In the checker's pair table they are identical
  in every other lemma.
* Choice: PST.1SG = past13 and PST.3SG = same cell (syncretism); PST.2SG and PST.1PL/2PL/3PL = pastnot13.
  Mapping for *be*: 1SG, 3SG *w ɒ z*; 2SG, 1PL, 2PL, 3PL *w ɜː*.
* Reason: Pullum's slides give the preterite of *be* as "was" for 1st and 3rd singular and "were" for the others
  (the slide table has the columns 1st sg / 3rd sg / other). Standard *you were* for singular 2SG; *you was*
  is non-standard and not mapped.
* Confidence: high.
* The irrealis *were* ("if I were") is outside the grid contexts.

### J3. IMP = the bare form
* Options: `inf` (plain form); `presothers` (plain present, *are* for *be*); `LEMMA`.
* Choice: `inf`. For all verbs except *be* it equals `presothers`; for *be* it gives *b iː* ("Be quiet!"), not *ɑː*.
* References: Pullum slides ("IMPERATIVE: Be kind", listed under the plain form); Wikipedia "Imperative mood" (Formation: bare form).
* `LEMMA` is rejected: in this dataset it is an orthographic label, not a phoneme string.
* IMP.2PL = IMP.2SG (syncretism): the English imperative has no number distinction.
  UVic Study Zone, "Imperatives": "It is the same for singular and plural subjects."
* Confidence: high.
* **[fix round] Superseded by D25: IMP.2SG and IMP.2PL are now in `PRS.NON3SG` and *be* is declared under `class_exceptions` (section 8, item 1).** Original decision, kept for the record: IMP gets its own class `IMP.2`, not merged with `PRS.NON3SG`. Reason: the grammar treats the plain form and the plain
  present as different cells (they differ for *be*, and modals lack the plain form). Alternative: merge into `PRS.NON3SG`
  (no change in forms; 10 warnings disappear; PRS.NON3SG then has 7 members).

### J4. Person split of `presothers`
* Dataset: `pres1s`, `pres3s`, `presothers`. The grid has six persons.
* Choice: PRS.1SG = `pres1s` (*am*, otherwise = presothers); PRS.3SG = `pres3s`;
  PRS.2SG = `presothers`; PRS.1PL, 2PL, 3PL = same as PRS.2SG (syncretism).
* Reason: Pullum's slide on English verb inflection has the columns "1st sg / 3rd sg / other" (*am / is / are*); for regular
  lexical verbs: plain present *fill* (all persons but 3sg), 3rd singular present *fills*. The 3PL takes the
  plain present (*they walk*), not the -s form.
* Class decision: `PRS.NON3SG` holds 1SG, 2SG, 1PL, 2PL, 3PL, `PRS.3SG` is alone (formally distinct).
  The class structure follows the lexical-verb paradigm: only *be* has a dedicated 1SG form, and the
  checker's class test is paradigm-wide. *be* therefore has three present forms in a two-class
  structure (*am* and *ɑː* in the same class). Alternative: `PRS.1SG` as its own class (PRS has 3 classes;
  PRS.1SG formally distinct, although its form equals presothers in all lemmas but one).
  This is the main way the grid under-represents the single most irregular, most frequent verb.
* Modals (may, shall, would, must, ought): no 3rd-singular -s, so pres3s = bare form (the data already has this).
* Confidence: high for the forms; medium for the class decision (analytic, not sourced).

### J5. Modals and defective verbs (D8)
* Dataset lemmas that are modals: `may`, `shall`, `would`, `must`, `ought`. `can`, `will`, `could`, `might`, `should`,
  `need`, `used` are NOT in the data (section 4).
* The class `modal_aux` (eng_modal_aux.csv) labels these five in the grid (`lexical_class` column).
* **[fix round] The list below is the original. Current state: *would* is excluded from scope (M7), *beware* PRS non-3SG are defective (M6), *ought* PST has a cited source (P5); see section 8.**
* Defective cells declared:
  * IMP.2SG/2PL for all five modals: no plain form, and the imperative is the plain form (Pullum slides: modal
    *will*, *may*, *must* have starred secondary forms, and the imperative is listed under the plain form). 10 cells.
  * PST (6 cells each) for `must`: "Must is even more defective: it has no preterite" (Pullum slides). 6 cells.
  * PST for `would`: this lemma is the preterite of *will* in H&P's analysis (slide on *will* / *would*); the lemma has no
    preterite of its own. The data also lacks it. 6 cells.
  * PST for `ought`: the data lacks it; no source seen that states the grammar facts (**unsourced**, `locus_verified: false`).
    6 cells.
  * `beware`: PRS.3SG and all PST cells, because beware has only the plain form (Pullum slide on *beware*; exercise key to
    Student's Introduction ch. 3: "She bewares" ungrammatical). 7 cells. The PRS non-3SG cells and IMP are kept from the data.
    The slide says beware "seems to lack" the plain present too (with a hedge by Georgia Green), so PRS non-3SG for *beware* is questionable (medium).
* `may` (*might*) and `shall` (*should*) keep their PST cells from the data: the preterite form exists in the paradigm
  (Pullum slide: "may ... preterite might"), although it does not express the "yesterday, once" reading (Wikipedia, "English modal verbs",
  'Preterite forms': preterites do not always modify the meaning simply). Alternative: mark all modal PST cells defective
  (would add 12 defective cells). This is a **low-sourcing, medium-confidence** call.
* A further consequence of the grid contexts: modals cannot take the object "it" or occur as main verbs in the habitual PRS context. We
  keep their PRS cells (the primary forms exist) and do not go further.
* Confidence: medium (J5 overall).

### J6. Lemma scope
* 124 multiword lemmas (e.g. *abut on*, *beaver away*) are excluded by `exclude_multiword_lemmas`. Hyphenated lemmas such as
  *French-polish* are single words and stay in. **[fix round]** 5,939 lemmas in scope.
* **[fix round]** One lemma is excluded by list: *would* (`classes/eng_excluded.csv`, M7).

### J7. Variant order (R3), set with `variant_overrides`
* File: `mappings/classes/eng_variant_overrides.csv` (240 rows: lemma, grid_cell, form). The form must be one of the data's variants; the checker raised no warning for a missing form.
* Principle, from SPEC R3: (1) the form the reference gives for the context; (2) if several are equally natural: not restricted by register or region, then single word, then fewest added morphemes; (3) ties keep the data order (an unprincipled tie-break, flagged below). The rejected variants stay in the `variants` column.
* What I used as reference for the standard form: the table in Wikipedia, "List of English irregular verbs" (which states that the preferred or more common form is generally listed first, and that US usage prefers the regular form of *spell*, *learn*, *spill*, and the irregular *dove* is mainly American; *hanged* is chiefly for execution), and Wiktionary "read" and "use" for pronunciation. Purdue OWL's irregular verbs page lists *find* found, *shine* shone, *swim* swam, *wake* woke, *quit* quit (partial list). Wikipedia is a secondary source; the choices for ties are therefore medium confidence.
* Changed from the data order (to the second or third variant), with the form chosen:
  * the past equals the present in the data first variant: *find* found, *bind* bound, *grind* ground, *wind(air)* wound, *unwind*, *rebind*, *unbind*, *spellbind* (bound / wound by analogy to the verb they contain).
  * *bend* bent (*unbend*), *bid* bid, *outbid* outbid (*bade* is the other variant), *hang* hung (*hanged* is for execution), *slink* slunk, *cleave* cleft (cleft/clove/cleaved: *clove* is restricted; *cleft* is first-listed), *beseech* beseeched (first-listed; besought is the other), *bide* bided (first-listed; *bode* the other), *sweat* sweated (first-listed).
  * R3 ties between a regional pair, resolved to the first-listed form of the Wikipedia table: *dream* dreamed, *lean* leaned, *leap* leaped, *dwell* dwelt (*indwell*), *dive* dived (*crash-dive*, *power-dive*, *skin-dive*; the *dove* variant is mainly American).
  * *read* family (*read*, *lip-read*, *misread*, *proofread*, *sight-read*): PRS.* and IMP.* = long vowel (r iː d); PST.* = r ɛ d (Wiktionary: past tense pronounced /rɛd/). *use*: PRS and IMP j uː z; PST j uː z d (Wiktionary: *used* is /juːzd/ in most senses, /juːst/ in the *used to* sense).
* Left at the data order because it already gives the first-listed or standard form: *alight* alit, *beat*, *bet*, *bleed*, *blend* blended, *breed* and compounds, *broadcast*, *burn* burned, *cost*, *feed* and compounds, *forecast*, *gird* girded, *inset*, *learn* learned (*unlearn*), *light* lit (*floodlight*), *misspell*, *quit*, *ring*, *shine* shone (*outshine*), *smell*, *spell*, *spill*, *spoil*, *spring* sprang, *swim* swam, *wake* woke, *weave* wove (*interweave*), *wet*, *countersink* (sank), *overleap*, *daresay* (as *say* said, by analogy).
* **[fix round] Previously unsourced ties, now resolved:** *abide* (abode first, Wiktionary), *betide* (betid first, Wiktionary), *stavein* (stove first, Wikipedia irregular-verb table), *grit* (override to *gritted*, M8; Wiktionary calls bare *grit* nonstandard). See section 8, item 6.
* Residual uncertainty: for pairs like *burned/burnt*, *learned/learnt* the two forms are equally natural by region; the grid form is the regular one (first-listed), and the other is in `variants`. Confidence medium for these ties, high for *find*, *bind*, *grind*, *wind*, *read*, *use*, *hang*, *bend*, *bid*, *slink*.

## 3. Where the rules broke or were unclear

* **R6 / class ids when the irregularity is in one lemma.** English syncretisms hold for all lemmas except
  *be* (PRS.1SG, PST.1SG vs PST.2SG; IMP vs PRS). The class ids express the lexical-verb paradigm, so *be* contradicts the
  class structure in the grid. No way to express "class holds except for lemma X" in the mapping, except by a lexical class with
  `by_class`, which can only change the source cell, not the class id. A per-class `distinct_class` override would help.
* **R3 variant order (resolved).** The first version of the tool took the first variant in data order, which gave wrong forms for *find*, *read*, *use* and others. With the new `variant_overrides` the order is now set per lemma and cell (J7). The rule `variant_rule` (regex on the phoneme string) was not used: the choice depends on the lemma, not on a phonetic pattern.
* **R5 text vs D8.** SPEC R5 (exclude verbs) is superseded by D8; used as D8.
* **Modals in a grid defined by a transitive habitual frame.** The frames assume a lexical verb with an object; modals do not fit. We kept
  to a minimal rule (J5).

## 4. Data problems

1. **Variant order in the data (78 lemmas have variants; 72 in PST only).** The data order puts a wrong or non-standard form first in many lemmas (for example *find*: f aˑɪ n d, the present form, before *found*; *bend* and *dive* as in J7). Fixed by `variant_overrides` (J7). The data also lists *beaten* as a variant of the preterite of *beat* (b iː t n̩, second variant); it stays in `variants`.
2. **`read` (and *lip-read*, *misread*, *proofread*, *sight-read*): the present cells hold two forms with the preterite vowel** (r ɛ d | r iː d). Data order gave *r ɛ d* for PRS.* and IMP.*, wrong. Fixed by override (J7).
3. **`use`**: present cells hold both /juːs/ (noun-like) and /juːz/; the verb present is /juːz/. Fixed by override (J7). Past: j uː s t | j uː z d; the first is the *used to* pronunciation.
4. **Missing frequent verbs:** *can, will, could, might, should, need, used* are absent. The data is incomplete exactly for the most frequent auxiliaries. *would* is a lemma only as the preterite of *will* without a preterite of its own. **[fix round, M12(b)]** Also absent are the frequent lexical verbs *need, seem, show, wait, die, raise*. This is a known limitation of the dataset (D24): no action is taken; the English scores must be read with it in mind.
5. **Disambiguated lemma labels:** `ring(sound)`, `wind(air)`, `stavein` (= stave in). Lemma labels are not uniform; not harmful.
6. **Missing forms:** `rendezvous` lacks pres3s and the preterite (data_gap; the forms exist). 7 cells.
7. **`LEMMA` cell** is orthographic and not usable in a phonemic grid (not used).
8. `be`: forms are correct (æ m, ɪ z, ɑː, w ɒ z, w ɜː, b iː). Note the dataset keeps the present 1SG separate for all verbs: pres1s = presothers for all except *be*.
9. Transcription: `ʊ`, `ɒ` etc. are CELEX-derived British RP; no editing done.

## 5. Check results (copied from `mappings/eng_check.md`)

* **[fix round] Original numbers; the current ones are in section 8.** Errors 0; warnings 10.
* Lemmas in data 6,064; out of scope: multiword 124, by list 0; **in scope 5,940**.
* Complete grids: 5,933 (all 14 cells filled).
* Empty cells: defective 35 (beware 7, may 2, shall 2, must 8, ought 8, would 8), data_gap 7 (rendezvous).
* Classes: 4 (PRS.NON3SG ×5 cells, PRS.3SG ×1, PST.ALL ×6, IMP.2 ×2). Formally distinct cells: 1 (PRS.3SG).
* Empirical distinctness (from the check report): distinct forms per complete grid: 2 forms in 45 grids (verbs whose preterite equals the bare form, e.g. *cut*, *put*), 3 forms in 5,887 grids, 6 forms in 1 grid (*be*); mean 2.99. Share of lemmas whose form in a cell is unique within the grid: PRS.3SG 1.0; every other cell 0.0 (the formally distinct cell is the one with an empirical unique form).
* Rules: source 6, syncretism 8, main_verb 0, restored 0. aux_carried cells: 0. Dropped tokens: none (no transforms).
* The 10 warnings are all the same kind: "different classes but forms identical in 1.000 of 5935 lemmas": PRS.1SG, PRS.2SG, PRS.1PL, PRS.2PL, PRS.3PL
  each against IMP.2SG and IMP.2PL. The imperative is formally the plain form and the present non-3sg is the plain present; they are
  identical in all lemmas but *be* (*b iː* vs *ɑː*, *æ m*), and the modals have no imperative. I kept them in separate classes by grammar (J3). I did not use
  `aux_carried` to silence them, because no auxiliary is involved. The alternative is a merged class.
* Filled-grid inspection (`outputs/cell_grid_pilot/eng_grid.csv`): 20 random lemmas (core, section, boggle, festoon, compose, putt, peculate,
  poultice, tame, maintain, drown, challenge, prorogue, assay, metabolize, overcloud, space, about-face, utter, park) are regular; plus *be*, *have*, *do*, *go*, *say*,
  *see*, *take*, *make*, *get*, *know*, *come*, *think*, *give*, *tell*, *work*, *walk*, *watch*, *try*, *cut*, *put*, *let*, *dare*, and the modals. All correct
  after the variant fix (J7): *find* found, *read* r iː d / r ɛ d, *use* j uː z / j uː z d, *bend* bent, *dive* dived, *hang* hung, *bid* bid, *slink* slunk, *cleave* cleft, *sweat* sweated checked in the grid.
  *be* gets 3 present forms and 2 preterite forms as intended; modals have PRS.3SG = bare form.

## 6. Sources seen

Added for J7 (summarised by the fetch tool): Wikipedia, "List of English irregular verbs", https://en.wikipedia.org/wiki/List_of_English_irregular_verbs (past forms and the note on first-listed preferred forms; first 100,000 characters); Wiktionary "use", https://en.wiktionary.org/wiki/use (verb /juːz/, past /juːzd/); Wiktionary "read", https://en.wiktionary.org/wiki/read (present /riːd/, past /rɛd/); Purdue OWL "Irregular Verbs", https://owl.purdue.edu/owl/general_writing/grammar/irregular_verbs/index.html (partial list).


I opened these in this session. **Caveat on method:** the fetch tool returns a summary produced by a small model, not the raw page,
for HTML pages. Only the Pullum slides PDF was extracted and read as raw text (via a PDF text extractor). Quotes below for the web pages are as returned by the
fetch tool. I did not see Huddleston & Pullum 2002 itself (no preview was reachable); no page numbers of the book are claimed.
The Grambank page references (g_HuddlestonPullum_English[...]) are copied from the Grambank `Source` field only.
Cambridge Dictionary grammar pages returned HTTP 403; British Council LearnEnglish URLs returned 404.

* Pullum, Geoffrey K., Edinburgh course slides (based on CGEL/SIEG), https://pullum.ppls.ed.ac.uk/grammar/verb_sli.pdf
  (redirected from lel.ed.ac.uk/~gpullum/grammar/verb_sli.pdf). Slides "English verb inflection" (paradigm of *be*: present am / is / are, preterite was / were; regular verb
  with plain present, 3rd singular present, preterite; imperative listed under plain form), "English verb inflection: LEXICAL VERBS"
  (six paradigmatic forms), the slides on modal *will*, *may*, *must* ("Must is even more defective: it has no preterite") and on *beware*.
* Huddleston & Pullum, A Student's Introduction to English Grammar, chapter 3 exercise key, http://pullum.ppls.ed.ac.uk/grammar/exx_ch3.html
  (via summarising fetch): exercise 4 (*beware* has no 3rd singular present), exercise 8 (modals have no distinct -s form).
* Purdue OWL, "Active Verb Tenses", https://owl.purdue.edu/owl/general_writing/grammar/verb_tenses/active_verb_tenses.html: simple present "Habitual Action" (*I run on Tuesdays and Sundays*); simple past "Completed Action" (*We visited the museum yesterday*).
* UVic English Language Centre Study Zone:
  * https://continuingstudies.uvic.ca/elc/studyzone/330/grammar/pres, "Simple Present": he/she/it take simple form + S; I, you, we, they take the simple form.
  * https://continuingstudies.uvic.ca/elc/studyzone/330/grammar/pasted, "Simple Past: Regular Verbs": "Its form is the same with all subjects."
  * https://continuingstudies.uvic.ca/elc/studyzone/330/grammar/irpast, "Simple Past: Irregular Verbs": BE differs by subject; table gives *was* for I and he/she/it, *were* for you, we, they.
  * https://continuingstudies.uvic.ca/elc/studyzone/330/grammar/upperf2, "Present Perfect or Simple Past?": simple past "usually used with definite times".
  * https://continuingstudies.uvic.ca/elc/studyzone/330/grammar/imperat, "Imperatives": same for singular and plural subjects; *Be careful!*.
* Wikipedia, "Imperative mood", https://en.wikipedia.org/wiki/Imperative_mood, section Formation: bare form; same as 2nd person present except *be*; no singular/plural indication.
* Wikipedia, "English modal verbs", https://en.wikipedia.org/wiki/English_modal_verbs (first 100,000 characters only): lead (no -s), "Mote and must" (must lacks a preterite), "Preterite forms", "Lists of modal auxiliary verbs" (ought marginal).
* Wikipedia, "English verbs", https://en.wikipedia.org/wiki/English_verbs: -s suffix; forms of *be*; imperative = base form (confirmation only).
* Grambank v1.0.3 `external/grambank/cldf/values.csv`, stan1293: GB082 0, GB083 1, GB084 0, GB086 0, GB089 1, GB091 1, GB120 1, GB121 1. No mapping contradicts a coding.
  (GB082 = 0 means no overt present marking; the -s of the 3SG is a person marker, as the Grambank comment states. Not a conflict.)

### Unsourced decisions
* **[fix round]** The `ought` PST defective entry is now sourced (Wikipedia, "English modal verbs", section Ought; a secondary source that reports H&P 2002). It is the only source for this cell, so a primary reading of H&P 2002 pp. 108-115 would be better.
* The class structure (PRS.NON3SG with 1SG; IMP separate from PRS; PST.ALL across the *be* difference) is an analytic choice, not a sourced fact.
* `may`/`shall` keep their preterite forms in PST (J5); the choice not to mark them defective is based on the slide showing the preterite but not on any statement
  about the grid context.
* **[fix round]** Variant order for *abide*, *betide*, *stavein*, *grit* is now sourced (section 8, item 6). 
* Variant order for regional ties (*burned/burnt* etc.) follows the first-listed form in the Wikipedia table (secondary source).
* **[fix round]** PRS non-3SG for *beware* are now `defective` (M6).

## 7. Effort

Variant fix: about 12 further tool calls (4 page fetches, script, 4 checks).

About 45 tool calls: ~10 web searches, ~22 page fetches (8 failed with 403/404/429), the rest local file reads, data
queries and checks (the mapping passed `check` at the first run). Main time sinks: finding readable sources (search results are
summaries, the Cambridge Dictionary and Cambridge Grammar are not reachable, so the Edinburgh slides were the only primary-grade source) and inspecting the
variant lists.

## 8. Fix round 2026-10-14

Instructions: `FIXROUND.md` (section "eng"), REVIEW.md items M6, M7, M8, M12(b), P4, P5, and DECISIONS D24, D25, D29. Sources were opened in this session. The Pullum slides PDF was downloaded to a scratch directory and read as raw extracted text. The Wikipedia pages and Wiktionary entries were downloaded as HTML and read as text with a script (no summarising tool), so `locus_verified: true` rests on raw text.

1. **D25, imperative class.** IMP.2SG and IMP.2PL moved from `IMP.2` to `PRS.NON3SG`. A top-level `class_exceptions` entry lists *be* (the only lemma the grammar names) for PRS.1SG, the PST person split (was/were) and IMP (be vs are). Reference: Pullum slides, "English verb inflection" (paradigm of *be*: *am/is/are*, *was/were*, plain form *be*; "Be kind" as the imperative). Exceptions used: 1 of at most 10. No other lemma breaks the identities: the modals have no imperative (defective), and *beware* keeps IMP = plain form = PRS form. Result: the 10 IMP warnings are gone, and there are no warnings of any kind. `distinct_class` now has 3 classes (PRS.NON3SG ×7, PRS.3SG ×1, PST.ALL ×6).
2. **M6, *beware*.** PRS.1SG, 2SG, 1PL, 2PL, 3PL are now `defective` (with the earlier PRS.3SG and PST ×6, 12 cells in all). Reference: the Pullum slide on the defective verb *beware*, whose paradigm stars the 3rd singular present, the gerund-participle, the preterite and the past participle, and says "it seems to lack the former" (the plain present). Green's hedge is recorded in the `reason`: she suggested that uses like *I always beware of their dog* are a pragmatic effect of warning, not a plain-present form. IMP is kept (*b ɪ w ɛˑə*), because the imperative is the plain form that *beware* has. Confidence medium: the slide itself says "seems".
3. **M7, *would*.** Excluded through `classes/eng_excluded.csv` (header `lemma`, one row) in `exclude_lemma_files`. It is the preterite of *will* in H&P's analysis (slide on modal *will*: preterite *would*), not a lemma. Its defective entry and its row in `eng_modal_aux.csv` were removed. Lemmas in scope: 5,939.
4. **M8, *grit*.** PST.1SG-3PL overridden to *ɡ r ɪ t ɪ d* in `eng_variant_overrides.csv` (6 rows; now 246 rows). The bare form stays in `variants`. Reference: Wiktionary "grit": simple past and participle "gritted or (nonstandard) grit".
5. **M12(b), missing verbs.** Added to section 4: *need, seem, show, wait, die, raise* are not in the dataset. Known limitation (D24); nothing was added to the mapping.
6. **P4, variant order.** I scanned all 78 lemmas with variants (list printed from the filled grid) against the Wikipedia "List of English irregular verbs" table, re-read as raw HTML rows. The grid already follows the table's first-listed form for: *beseech, bid, bide, bend, bind, burn, cleave, dive, dream, dwell, find, gird, grind, hang, lean, leap, learn, light, quit, shine, slink, smell, spell, spill, spoil, spring, sweat, swim, wake, weave, wet, wind*, and the compounds. No change was needed. The four earlier unsourced ties are resolved: *stavein* (*stove* first, same table), *abide* (*abode* before *abided*, Wiktionary "abide") and *betide* (*betid* before *betided*, Wiktionary "betide"), both kept as in the data; *grit* changed (item 4). Residual: *abide* and *betide* sources are Wiktionary only; *burned/burnt*-type regional ties follow the first-listed form of the table (a secondary source).
7. **P5, defective entries.** Re-checked all of them:
   * modal IMP.2SG/2PL (*may, shall, must, ought*): Pullum slides (modals have no plain form; the imperative is the plain form). Cited.
   * *must* PST: Pullum slide on *must*. Cited.
   * *ought* PST: the entry had `locus_verified: false`. Now cited: Wikipedia "English modal verbs", section "Ought": "Ought differs ... in lacking a preterite"; the same page says "As they lack preterite forms, must, ought and need cannot be used" for backshift, citing H&P 2002. Kept `defective`. The page also says that for some speakers *may* and *might* are distinct lexemes that each lack a preterite; I keep the data PST for *may* and *shall* (they have a preterite *might/should* in the paradigm slide), which is the medium-confidence call of J5.
   * *beware*: Pullum slide. Cited.
   * *would*: excluded, no entry.
   * Every other empty cell is a `data_gaps` cell: only *rendezvous* (7 cells, missing in the data). Nothing was moved.
8. **Frame.** D23 (witnessed) changes nothing for English. D29 (object "it"): the modals do not take an object; the PRS and PST cells are kept as in J5.
9. **Check numbers after the round** (`gridtools.py check eng`): errors 0, warnings 0. Lemmas in data 6,064; out of scope: multiword 124, by list 1 (*would*); **in scope 5,939**. Complete grids 5,933. Empty cells: defective 32 (*beware* 12, *may* 2, *shall* 2, *must* 8, *ought* 8), data_gap 7 (*rendezvous*). Classes 3, formally distinct cells 1 (PRS.3SG). `emp_mean_classes` 2.99 (2 classes in 44 grids, 3 in 5,888, 6 in 1: *be*), against 3 `distinct_class` ids. Rules: source 6, syncretism 8. `class_exceptions`: *be*.
10. **Could not fix.** The H&P 2002 text was not available, so the *ought* and *beware* claims rest on slides by one of the authors and on Wikipedia. The "missing frequent verbs" limit stays.

### Sources seen in the fix round
* https://pullum.ppls.ed.ac.uk/grammar/verb_sli.pdf, slides "English verb inflection" (*be*), "LEXICAL VERBS", "The defective verb beware", "AUXILIARY VERBS" (will, may, must): "Must is even more defective: it has no preterite."
* https://en.wikipedia.org/wiki/English_modal_verbs, sections "Ought" and "Preterite forms" (see item 7).
* https://en.wikipedia.org/wiki/List_of_English_irregular_verbs, table rows for the lemmas of item 6: "stave – stove/staved – stove/staved/stoven".
* https://en.wiktionary.org/wiki/grit: "gritted or (nonstandard) grit"; https://en.wiktionary.org/wiki/abide: "abode or abided or abid"; https://en.wiktionary.org/wiki/betide: "betid or betided".

### Effort (fix round)
About 38 tool calls, 3 web searches. Main time sink: reading the PDF slides without a PDF library (a small script extracted the text).
