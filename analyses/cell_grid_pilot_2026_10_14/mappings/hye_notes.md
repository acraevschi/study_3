# Notes: Eastern Armenian (hye) mapping

Mapping file: `mappings/hye.yaml` (status reviewed after the fix round of 2026-10-14; see the last section, which supersedes the numbers in sections 1-5 where they differ). Data: MGN `hye-v` (UniMorph-derived), 903 lemmas in the data, 855 in scope after the fix round, orthographic.
Mapped by the Sonnet 5.5 mapping agent, 2026-10-14. Page numbers below are the printed page numbers of Sakayan 2007 as they appear in the text layer I read (the PDF page minus 25); they were seen in the text, not guessed.

## 1. Summary table

| cell | rule | source cell | class | aux_carried | confidence |
|---|---|---|---|---|---|
| PRS.1SG | main_verb | `1;SG;PRS;IND`, transform `strip_present_aux` | PRS.1SG | true | high |
| PRS.2SG | main_verb | `2;SG;PRS;IND`, same transform | PRS.2SG | true | high |
| PRS.3SG | main_verb | `3;SG;PRS;IND`, same transform | PRS.3SG | true | high |
| PRS.1PL | main_verb | `1;PL;PRS;IND`, same transform | PRS.1PL | true | high |
| PRS.2PL | main_verb | `2;PL;PRS;IND`, same transform | PRS.2PL | true | high |
| PRS.3PL | main_verb | `3;PL;PRS;IND`, same transform | PRS.3PL | true | high |
| PST.1SG | source | `1;SG;PST;IND` (aorist) | PST.1SG | false | high |
| PST.2SG | source | `2;SG;PST;IND` | PST.2SG | false | high |
| PST.3SG | source | `3;SG;PST;IND` | PST.3SG | false | high |
| PST.1PL | source | `1;PL;PST;IND` | PST.1PL | false | high |
| PST.2PL | source | `2;PL;PST;IND` | PST.2PL | false | high |
| PST.3PL | source | `3;PL;PST;IND` | PST.3PL | false | high |
| IMP.2SG | source | `2;SG;IMP` | IMP.2SG | false | high |
| IMP.2PL | source | `2;PL;IMP` | IMP.2PL | false | medium |

No syncretism or restored cells. No lexical classes, no exclusion or defective lists.

## 2. Judgement calls

**J1. PRS is periphrastic: grid form = the imperfective converb (D1 option C, D2).**
* Options: (a) the converb in -ում alone, with the auxiliary removed by a transform; (b) the full multiword form.
* Choice: (a). Each PRS cell takes its own person-specific source cell and the declared transform `strip_present_aux`, which drops the six present forms of the auxiliary *եմ, ես, է, ենք, եք, են*. The check shows the transform dropped exactly these 6 strings, 903 times each, so the removed set is small and lemma-invariant.
* Reason: Sakayan 2007 (Unit 1, "The present tense of regular verbs", printed p. 11): the present is "a complex formation" of the auxiliary and the present participle in -ում. Fairbanks & Stevick 1958, Lesson 2, Note 2.2: the present is made from the present of "to be" plus a verb form in /um/. Both say there is one present, covering the habitual and the progressive (Sakayan p. 13, "Uses of the present tense": habitual actions are expressed by it). So the habitual frame needs no other construction.
* Consequence: all six PRS cells have the same grid form (converb), identity 1.000 for all 15 pairs. The person distinction is carried by the auxiliary, so each cell has its own class id with `aux_carried: true` (D1). **For the owner:** this makes the check report 14 formally distinct cells out of 14 classes, although the six PRS cells are identical strings. `is_formally_distinct` here counts auxiliary-carried distinctions, by D1.
* Confidence: high. Alternative recorded: none needed (the multiword form is option B, rejected by D1).

**J2. PST for "yesterday, once": synthetic aorist, not the perfect.**
* Options: (a) aorist `…;PST;IND` (*աբսորբեցի*), (b) perfect `…;PRS;PRF;IND` (*աբսորբել եմ*, participle + auxiliary), (c) imperfect `…;IPFV;IND`.
* Choice: (a). Since D23 the frame also fixes "the speaker witnessed the event": the aorist follows from the frame (Sakayan p. 76: the aorist is "the witness' tense"; the perfect can be reportative, p. 99). See the fix round section.
* Reason: Sakayan 2007, Unit 5 "Uses of the perfect tense" (printed p. 99): the perfect refers to a past action that still holds, or to a more general and distant past; "yesterday" words appear with the aorist ("Yerek l'setsi", with a note that the perfect is not used there). Fairbanks & Stevick 1958, Lesson 14 Note 14.1: the preterite is the equivalent of the simple past and refers to an event that happened once and is completed. The Eastern aorist is the only synthetic indicative tense (Sakayan p. 74). R3 is also met: single word over multiword.
* The imperfect is a progressive or habitual past (Sakayan p. 76, aorist vs imperfect), not a single completed event.
* Alternative recorded under `alternatives:` on PST.1SG: perfect with the auxiliary removed, which would make all six PST cells the same participle string with aux-carried classes.
* Confidence: high.

**J3. IMP 2SG and 2PL: use the dataset's cells (`2;SG;IMP`, `2;PL;IMP`), orthography kept with ՛.**
* 2SG: Sakayan 2007, Unit 6 "The imperative mood" (printed p. 126): the singular imperative is -ել replaced by -ի՛ր and -ալ by -ա՛; verbs with the suffixes -ան-, -են-, -ն-, -ց- (and causatives -ցն-) form it from the aorist stem (e.g. -ի՛ր, -ո՛ւ). Fairbanks & Stevick 1958, Lesson 16 Note 16.1 agree that there are two special imperative forms, with the singular irregular and based on the preterite stem. The data matches this (e.g. *խմի՛ր*, *գնա*-type *աղա՛*, *հագեցրո՛ւ*). Confidence: high.
* 2PL: **a data/grammar mismatch.** Sakayan p. 127 derives the regular plural from the aorist stem + -ե՛ք (*երգեցե՛ք*) and adds that, in everyday speech, plural imperatives from the infinitive stem are also common (*վախե՛ք*). Dolatian's slide 11 ("Imp. 2PL" *ləs-e-tsʰ-ekʰ*) and the Wikipedia verb table (*սիրեցե՛ք*) also give the aorist-stem form. The data gives the infinitive-stem form for the 678 plain -ել verbs (*աբսորբե՛ք*), and the aorist-stem form for all -ալ verbs (*ագահանացե՛ք*), the 52 -ցնել causatives and 11 -նել verbs. So the dataset mixes the two stems.
* Choice: map the dataset cell and mark `confidence: medium`. R3 would pick the grammar's first form, but the dataset has one form per cell and R7 forbids re-deriving it. Rejected option recorded under `alternatives:` and `competing_forms:` of IMP.2PL. **Decided (D30): the data form is kept**; the aorist-stem form is recorded under `alternatives` (see the fix round section).
* The emphasis mark ՛ is kept exactly as in the data. 2PL is a plural-addressee form; it is the same as the polite singular (Sakayan p. 15), which does not matter for the grid.

**J4. Defaults.**
* `subject_gender: not_marked`: Sakayan 2007, Unit 3 (printed p. 57), grammatical gender is unmarked.
* `second_person_sg: informal դու`: Sakayan p. 15: informal *դու* for one close addressee, formal/plural *դուք*.
* `first_person_pl: not_distinguished`: WALS 39A (Cysouw, after Kozintseva 1995) gives no inclusive/exclusive distinction. I opened the WALS value page.
* `plural_reading: plural`, `object: not_indexed`: see "Unsourced decisions". Neither affects any cell.

## 3. Where the rules broke or were unclear

* **D1 and distinctness (see J1):** under option C the six PRS cells are identical strings but count as six classes. The checker accepted it because every pair has `aux_carried: true` on both cells. A downstream analysis should read "formally distinct" for hye PRS as "distinct by the auxiliary only".
* **Perfect/aorist with "once":** SPEC §3 "PST distance: yesterday" fits the aorist directly in Eastern Armenian, so no rule conflict. The aorist also has no remoteness split in the sources read.
* **R3 vs R7 (J3):** R3 asks for the grammar's form, R7 forbids transcribing forms again. For IMP.2PL they point to different forms. I followed R7 and flagged it.
* **Grambank:** GB089 (S indexed on the verb) = 0 contradicts GB091 (A indexed) = 1 and the person suffixes on intransitive aorists (Sakayan p. 74). Reported in `grambank_check.conflicts`; I did not change any cell because of it. GB120 is "?"; GB121 = 1 agrees with the use of the auxiliary *լինել*/*եմ*.
* **Checker behaviour:** no error. The tool does not warn when six cells have identical strings and distinct classes, if all are aux-carried. That is by design (D1).

## 4. Data problems

* **Frequent verbs are missing.** Of 48 common verbs I looked up (including the non-verb *պետք*, so 47 verbs), 39 are not in the 903 lemmas, among them *լինել, ունենալ, գնալ, գալ, տալ, անել, ասել, տեսնել, ուտել, գրել, սիրել, խոսել, լսել, նայել, սկսել*. The 903 lemmas are mostly rare, derived or archaic (741 in -ել, 162 in -ալ; 455 have a PASS cell, 237 a CAUSE cell). Presumably UniMorph lacks complete paradigms for the irregular and frequent verbs. The grid is complete for every lemma present (no `data_gap`), but the sample under-represents suppletive and high-frequency verbs.
* **Irregular or suppletive items in the data:** *երթալ/էթալ* have the present *երթում/էթում* but a *գնա-* aorist (*գնացի*) and imperative plural *գնացե՛ք*; *տենալ* has the aorist *տեսա*. The suppletion is the dataset's; it is not a mapping issue.
* **Mixed orthography:** a few lemmas keep classical orthography (*խօսել*, *կարօտել*, imperative plural *խօսէ՛ք*). Four -ել verbs have plural imperative in -է՛ք.
* **Odd imperative:** *քելել* has the singular imperative *քե՛լ*.
* **IMP.2PL stem inconsistency** (J3).
* **PST 3SG endings:** -ալ verbs of the "-ա" class have -ավ (*կամեցավ, մերկացավ*), matching Sakayan's -ա, -ար, -աւ class (p. 74).
* No cell has more than one form, so there are no `variants` and no R3 variant-order decisions inside the dataset (re-checked in the fix round: 0 of 11,970 cells have variants).
* The lemma set includes derived verbs (passives in -վել, causatives in -ցնել) as ordinary lemmas. I did not exclude them; R8 mentions only multiword and clitic-derived paradigms.

## 5. Check results (from `mappings/hye_check.md`)

* Errors 0; warnings 0.
* Lemmas in data 903; in scope 903 (0 multiword, 0 excluded by list); complete grids 903; empty cells none (no `defective`, no `data_gap`).
* Classes 14; formally distinct cells 14 (counting auxiliary-carried distinctions); aux_carried cells 6.
* Rule counts: main_verb 6, source 8.
* Transform `strip_present_aux` dropped 6 distinct strings (*եմ, ես, է, ենք, եք, են*), each 903 times.
* Pair identity: the 15 PRS pairs are identical in 1.000 of lemmas (explained by aux_carried); no other pair above 0.05.
* Filled-grid inspection: 20 random lemmas, 20 chosen lemmas (*խմել, պառկել, ուզել, կարողանալ, սովորել, հասկանալ, աշխատել, կանգնել, արթնանալ, կարծել, մոռանալ, սպանել, երթալ, էթալ, տենալ, վախենալ, համբուրել, հագել, խօսել, աղալ*). All PRS forms end in -ում; the aorist is -եցի / -ացի / -ացա / -րի as the grammar says. Rows with non-empty `variants`: none, so there was no variant order to decide. I could not inspect the most frequent verbs of the language because they are not in the data.

## 6. Sources seen (opened in this session)

1. Sakayan, Dora 2007, *Eastern Armenian for the English-speaking World: A Contrastive Approach*, Yerevan State University Press. https://lib.armedu.am/files/resource/files/2023-01-26/6ea102f4eca086199c186f54c1aecfc3.pdf. The file has a text layer in a transliterating font encoding that I decoded by hand. Used: Unit 1, present tense pp. 11-13 (paradigm *groum em…*; "complex formation" of the auxiliary and the participle in -oum), pronouns p. 15 (informal *dou*, formal and plural *douq*); Unit 3 p. 57 (gender unmarked); Unit 4 pp. 74-76 (aorist, the only simple tense; paradigm; aorist vs imperfect); Unit 5 pp. 96-99 (perfect and pluperfect; p. 99: "yesterday" goes with the aorist, perfect for "years ago"); Unit 6 pp. 126-127 (imperative singular and plural, footnote on infinitive-stem plurals).
2. Fairbanks, Gordon H. & Stevick, Earl W. 1958, *Spoken East Armenian*, ACLS. https://archive.org/details/micro_IA41153117_0852 (OCR text read through the archive.org text file). Used, from OCR text only: Lesson 2 Note 2.2 (present = verb in /um/ + present of 'to be', "gentiim ém"); Lesson 14 Notes 14.1-14.2 (preterite = simple past; once and completed; /i/ and /a/ endings); Lesson 15.1 heading (present perfect; text garbled); Lesson 16 Note 16.1 (two imperative forms, 2SG irregular and based on the preterite stem, 2PL regular); Lesson 20 B.1 (present perfect "I have gone"; past perfect for more distant events). The paradigm tables in the OCR are garbled, so I took no endings from this source.
3. Dolatian, Hossep 2022, *Armenian Morphology* (Glottothèque LACIM slides). https://spw.uni-goettingen.de/projects/lacim/doc/ARM-MOR-001.pdf. Slide 9: Eastern indicative present = converb + auxiliary; slide 11: Imp. 2PL formed on the aorist stem (*ləs-e-tsʰ-ekʰ*).
4. Wikipedia, "Eastern Armenian verb table". https://en.wikipedia.org/wiki/Eastern_Armenian_verb_table. Used only as a cross-check of the paradigm of *sirel* (present *sirum em…*, aorist *sirec'i…*, imperative *sirir* / *sirec'ek'*).
5. WALS 39A, Armenian (Eastern). https://wals.info/valuesets/39A-arm. Value "No inclusive/exclusive" (Kozintseva 1995).
6. Grambank v1.0.3 `external/grambank/cldf/values.csv`, rows for `nucl1235` (GB082, 083, 084, 086, 089, 091, 120, 121), whose Source field names Dum-Tragut 2009 pp. 196-235, 313-316 and Zorc 1995 pp. 306-308.

Seen only in search results and not opened: Dum-Tragut 2009 (John Benjamins catalogue page). I found no readable preview text of Dum-Tragut 2009, so none of its pages is marked `locus_verified: true`.

## Unsourced decisions

* `defaults.plural_reading: plural` and `defaults.object: not_indexed`: no source read for either (Dum-Tragut 2009 pp. 313-316, cited by Grambank, was not opened; Sakayan describes only singular and plural verb forms). No cell depends on them.
* Dum-Tragut 2009 page references in the YAML (pp. 196-197, 217-219) are copied from Grambank and carry `locus_verified: false`. Every cell also has at least one seen reference, so no cell is unsourced.

## 7. Effort

About 45 tool calls, of which about 5 web searches and 14 web fetches (several of the archive.org OCR text read in 100,000-character chunks). Main time sinks: (a) getting readable text from the PDFs (the fetch tool could not read them, so I extracted the text layer with pypdf from the saved copy in the tool-results folder, outside the repository, and decoded Sakayan's font-encoded Armenian by hand); (b) the Fairbanks OCR, whose paradigm tables are garbled; (c) finding that no preview of Dum-Tragut 2009 is readable.

## Fix round 2026-10-14

Agent: a second Sonnet 5.5 agent, working from `hye.yaml`, these notes, the check report, REVIEW.md (rows for hye, §3, §6) and DECISIONS D23-D32. No `.prompts/`, no ancestry or GeLaTo data, no git. Working files (downloads) were kept in the session scratchpad outside the repository.

### Items and what was done

| item | what I did |
|---|---|
| D23 evidentiality | PST references and the PST.1SG `competing_forms` now say that the frame fixes "the speaker witnessed the event". New reference: Sakayan 2007, Unit 4, "Uses of the aorist tense" (printed p. 76): the aorist "is the witness' tense" and is used "only if the speaker has either witnessed the activity" or can speak of it with absolute certainty. Sakayan p. 99, points c-d: the perfect "does not necessarily indicate activity experienced by the speaker" and has a "reportative or mediative meaning". So the aorist now follows from the frame, and the perfect is the reportative alternative (kept under `alternatives` of PST.1SG). PST confidence stays high. |
| D30 IMP.2PL | Data form kept (infinitive-stem -ե՛ք for the plain -ել verbs). The aorist-stem form -եցե՛ք is recorded under `alternatives` of IMP.2PL (Sakayan 2007 p. 127 point 2a; the Wikipedia Eastern Armenian verb table gives *սիրեցե՛ք*), with Sakayan's footnote that infinitive-stem plurals are "common" in everyday speech. Counts in scope: 650 plain -ել verbs have the short form; 148 -ալ verbs, 51 -ցնել causatives and 6 other -նել verbs have the aorist-stem form. Confidence medium. |
| P4 variants | Checked every cell: the 14 mapped source cells have exactly one form per lemma in all 855 in-scope lemmas (the filled grid has 0 non-empty `variants`). No variant order to set and no `variant_overrides` needed. I also scanned the PST and IMP cells for typos and classical spellings (see the next item). |
| P4 odd lemmas / R8 | Decision: **exclude both groups** under R8 (standard variety). Two lists, in `exclude_lemma_files`. Details below. |
| P5 | There are no `defective` and no `data_gaps` entries, so nothing to re-check. |
| Optional aux | Added `aux:` to the six PRS cells: եմ, ես, է, ենք, եք, են. The grid column `aux` now carries them; `aux_carried` stays true. PST and IMP are synthetic, no auxiliary. |
| D24 | Nothing done for missing frequent verbs (listed as a known limitation in section 4 above: 39 of 47 checked frequent verbs absent). The dataset is used as it is. |
| P1/D26 | The checker reports `emp_mean_classes` = 14.0 (see check numbers). |

### Scope decision for classical-orthography and colloquial lemmas (R8)

**The question.** The data contains lemmas that are not standard Eastern Armenian in the sense of R8: spellings in the classical (Mashtotsian) orthography, and lemmas that are only colloquial, dialectal or Western Armenian.

**Rule 1, classical orthography (6 lemmas).** List `classes/hye_classical_orthography_excluded.csv`: *խօսալ, խօսել, կարօտել, մօտենալ, սիրուել, սիրւել*.
* Standard EA is written in the reformed orthography. Sakayan says EA adopted a "simplifying spelling reform" in the early Soviet years while traditional spelling was kept in WA (About this textbook, point 1, printed p. xix); his rule for օ in EA says it occurs word-initially and in compounds or derived words (Unit 4, "Orthography: O o or Ø ø", printed p. 84), so a medial օ as in *խօսել* is not EA spelling. The Wikipedia article "Armenian orthography reform" lists the changes (classical *խօսել* → reformed *խոսել*; classical ու in *բացուել* → *բացվել*; classical *-է-* in endings → *-ե-*) and says the reformed orthography is the official one in Armenia.
* The list is mechanical and double-checked: (a) the Wiktionary entry of each of the 6 lemmas is an `alt sp ... from=TAO` (traditional orthography) of the reformed lemma (*խոսալ, խոսել, կարոտել, մոտենալ, սիրվել*); (b) a script over all 903 lemmas and their PST and IMP forms, looking for a lone ւ, a medial օ that is not in a compound, classical diphthongs and -է- in endings, finds exactly these 6 (plus *տնօրինել*, a compound, which stays; and lemmas with an initial է, which are not misspellings). The classical forms also create wrong IMP.2PL forms (*խօսէ՛ք*) and a 3SG in -աւ (*մօտեցաւ*).
* Cost: the reformed *խոսել* and *մոտենալ* are not in the data, so these two verbs are lost; *խոսալ, կարոտել, սիրվել* are in the data and stay (*խոսալ* is excluded by Rule 2).

**Rule 2, colloquial, dialectal or Western-only (42 lemmas).** List `classes/hye_nonstandard_register_excluded.csv`. A lemma is excluded when **every** non-inflectional sense in its English Wiktionary Armenian entry carries a label *colloquial*, *dialectal*, *Western Armenian* or a regional label (Mush, Akn, Sivas, Tokat, Istanbul, Izmit, Partizak, Ararat), or the lemma is only an "alternative form of X (colloquial/dialectal)". I read the raw wikitext of each entry through the MediaWiki API (894 of 903 pages found; the 9 missing titles are spelled with *եւ* in the data and with *և* on Wiktionary, and they are regular verbs). Examples: *երթալ* ("dialectal or Western Armenian"; "used mainly dialectally in Eastern Armenian"), *էթալ* ("colloquial form of երթալ"), *տենալ* ("colloquial form of տեսնել"), *ըլնել* (dialectal for *լինել*), *ընգնել* (Ararat dialectal for *ընկնել*), *հելնել*, *քցել*, *խոսալ* (colloquial *-ալ* variant of *խոսել*).
* Why the label is used: the standard EA is the literary language (Sakayan: EA "is the official language of the Republic of Armenia", based on the dialect of the Ararat valley and Yerevan; the Wikipedia Yerevan dialect article separates the standard from the colloquial Yerevan speech), and R8 asks for the standard variety. The dataset's suppletive aorist of these items (*երթալ, էթալ* → *գնացի*; *տենալ* → *տեսա*; *ըլնել* → *էղա*) shows they borrow forms from the standard verbs.
* What is not excluded: lemmas with a mixed entry (a standard sense plus a labelled one), and lemmas labelled only *archaic, literary, poetic, rare, dated, informal, slang* or *vulgar*. These are register or period labels inside the standard lexicon, not another variety. This keeps rare and derived verbs, which are most of the data.
* Effect: 48 lemmas leave the scope (903 to 855). Every remaining lemma still has 14 non-empty cells and the class structure does not change.
* **Limits.** The only source for the membership of Rule 2 is Wiktionary labels (raw text read this session), a volunteer dictionary; I could not open an Armenian academic dictionary (a search for one found nothing readable). The rule itself rests on Sakayan and the Wikipedia article. Confidence: Rule 1 high, Rule 2 medium. If the owner prefers the dataset unfiltered, delete the two entries from `exclude_lemma_files`; no other cell depends on them.

### Check numbers after the fix round (from `hye_check.md`)

* Errors 0, warnings 0.
* Lemmas in data 903; out of scope 48 (0 multiword, 48 by list); in scope 855; complete grids 855; no empty cells, no `defective`, no `data_gaps`.
* Classes 14; formally distinct cells 14 (counting auxiliary-carried distinctions); aux-carried cells 6.
* **`emp_mean_classes` 14.0** (all 855 grids have 14 empirical classes), against 14 declared classes. With `aux:` given, the six PRS cells keep six empirical classes (same converb, six different auxiliaries). No `class_exceptions`.
* Rule counts: main_verb 6, source 8. The transform `strip_present_aux` dropped the six auxiliary forms 855 times each.

### Sources seen in the fix round (opened in this session, raw text unless noted)

1. Sakayan 2007, *Eastern Armenian for the English-speaking World*, https://lib.armedu.am/files/resource/files/2023-01-26/6ea102f4eca086199c186f54c1aecfc3.pdf. Text layer read with pypdf from a copy saved in the scratchpad. Printed page = PDF page minus 25. Seen: p. xix (About this textbook, point 1: EA vs WA, "a simplifying spelling reform was adopted for EA"); p. 15 (*dou* informal, *douq* formal and plural); p. 57 (gender unmarked); pp. 11-12 (present tense: auxiliary plus the participle in -um); p. 74 (the aorist is "the only simple tense"); p. 76 (aorist "is the witness' tense"); p. 84 (orthography of O o and Ø ø); p. 98-99 (perfect vs aorist, points b-d); pp. 126-127 (imperative singular and plural, footnote on infinitive-stem plurals).
2. Fairbanks & Stevick 1958, https://archive.org/details/micro_IA41153117_0852 (djvu OCR text via archive.org/download). Seen: Note 2.2 (present = verb in /um/ plus the present of "to be"), Note 14.1 (preterite: "an event which happened once in the past and is completed"), Lesson 16 imperative table (2PL regular, 2SG irregular, based on the preterite stem). The paradigm tables are garbled; I took no endings from them.
3. Wikipedia, "Armenian orthography reform", https://en.wikipedia.org/wiki/Armenian_orthography_reform (raw wikitext). Seen: Rules section and example table (classical *խօսել* → reformed *խոսել*); "Today it is the officially used orthography ... in Armenia". It cites Dum-Tragut 2009 p. 12 for the changes; I could not open that page (Google Books blocked the request).
4. Wikipedia, "Yerevan dialect", https://en.wikipedia.org/wiki/Yerevan_dialect (raw). Seen: the dialect is the basis of "colloquial Eastern Armenian" and a basis of the literary language.
5. Wikipedia, "Eastern Armenian verb table", https://en.wikipedia.org/wiki/Eastern_Armenian_verb_table (raw). Seen: Imperative *սիրի՛ր / սիրեցե՛ք*.
6. English Wiktionary entries (raw wikitext through https://en.wiktionary.org/w/api.php for the 894 lemma pages; the individual entries *էթալ, տենալ, երթալ, խօսել, ըլնել* by index.php?action=raw). Used only for the register labels and the "traditional orthography" tag, never as the only reference for a decision.
7. Seen only as search results or summaries (not used for any `locus_verified`): a Yerevan State University article by Khachatryan (2022) on conjugation shifts between literary and colloquial EA (abstract only); the WebFetch summaries of Wiktionary that I replaced with raw text.
8. Not opened: Dum-Tragut 2009 (all pages stay `locus_verified: false`), Dolatian 2022 slides, WALS 39A (their earlier readings stand as recorded in section 6 above, but I did not repeat them).

### Unsourced decisions (after the fix round)

* `defaults.plural_reading` and `defaults.object`: unchanged (see above).
* Membership of the Rule 2 list rests on Wiktionary labels alone (see Limits).

### Effort (fix round)

About 60 tool calls, of which 4 web searches and 4 summarising web fetches (whose output I did not use for any `locus_verified`). The rest were local reads and raw downloads (curl) of Wiktionary, Wikipedia, Sakayan and Fairbanks to the scratchpad. Main time sink: reading 894 Wiktionary entries and turning their labels into a rule, and finding the printed pages in Sakayan's font-encoded text.
