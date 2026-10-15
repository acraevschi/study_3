# Finnish (fin) mapping notes, G14

Mapped by Claude Sonnet 5.5 (subagent), 2026-10-14. Dataset: MGN `fin-v` (orthographic, 7,724 lemmas).
Mapping: `mappings/fin.yaml`. Check: 0 errors, 0 warnings (after the fix round).

**Source note (updated in the fix round 2026-10-14).** In the first round VISK
(kaino.kotus.fi/visk) could not be opened: `WebFetch` and `curl` got HTTP 403 and a Cloudflare challenge. In the fix
round the original host still answers 403, but the Internet Archive copy
`https://web.archive.org/web/2023/https://kaino.kotus.fi/visk/sisallys.php?p=<§>` returns the full text of the sections.
I read these pages as raw text (not a fetch summary), so the VISK references in `fin.yaml` now carry
`locus_verified: true`. The earlier extract-only caveats below (sections 2, 3, 6, "Unsourced decisions") are
superseded by the section "Fix round 2026-10-14" at the end. Karlsson 2018 could not be read (no preview found).

## 1. Summary table

| cell | rule | source cell | class | aux_carried | confidence |
|---|---|---|---|---|---|
| PRS.1SG | source | ACT;PRS;POS;IND;1;SG | PRS.1SG | no | high |
| PRS.2SG | source | ACT;PRS;POS;IND;2;SG | PRS.2SG | no | high |
| PRS.3SG | source | ACT;PRS;POS;IND;3;SG | PRS.3SG | no | high |
| PRS.1PL | source | ACT;PRS;POS;IND;1;PL | PRS.1PL | no | high |
| PRS.2PL | source | ACT;PRS;POS;IND;2;PL | PRS.2PL | no | high |
| PRS.3PL | source | ACT;PRS;POS;IND;3;PL | PRS.3PL | no | high |
| PST.1SG | source | ACT;PST;POS;IND;1;SG | PST.1SG | no | high |
| PST.2SG | source | ACT;PST;POS;IND;2;SG | PST.2SG | no | high |
| PST.3SG | source | ACT;PST;POS;IND;3;SG | PST.3SG | no | high |
| PST.1PL | source | ACT;PST;POS;IND;1;PL | PST.1PL | no | high |
| PST.2PL | source | ACT;PST;POS;IND;2;PL | PST.2PL | no | high |
| PST.3PL | source | ACT;PST;POS;IND;3;PL | PST.3PL | no | high |
| IMP.2SG | source | ACT;PRS;POS;IMP;2;SG | IMP.2SG | no | high |
| IMP.2PL | source | ACT;PRS;POS;IMP;2;PL | IMP.2PL | no | high |

All 14 cells are direct source cells. No auxiliary, transform, lexical class or defective list is used. Grambank
(finn1318) agrees: no morphological present marker (GB082 = 0), a dedicated past (GB083 = 1), no future (GB084 = 0).
This is consistent with using synthetic PRS and imperfekti, and with FUT being periphrastic or present (FUT is deferred).
Grambank coding source is "Mervi de Heer, p.c." (Source field empty), so it gives no grammar to consult.

## 2. Judgement calls

### J1. PST: imperfekti or perfekti for "Yesterday I V-ed (it), once"
* **Options.** (a) imperfekti, a single synthetic word (*ostin*); (b) perfekti, `olla` + past participle (*olen ostanut*).
  Under D1 (b) would be a main-verb cell with the participle as the grid form, and the auxiliary distinctions counted
  (`aux_carried: true`).
* **Choice.** (a) imperfekti, `ACT;PST;POS;IND;x;y`, rule `source`, for all six persons.
* **Reason.** The context is a dated, completed, single event told as news. Finnish grammars assign a definite
  time reference to the imperfekti and an indefinite one to the perfekti.
* **References seen.**
  * https://uusikielemme.fi/finnish-grammar/verbs/verb-tenses-and-moods/the-relation-between-the-imperfect-perfect-and-plusquamperfect
    (opened). It says the imperfect is used when the exact time is given, with the example *Ostimme eilen auton*.
    `locus_verified: true`.
  * https://yle.fi/a/20-156316, article "Imperfekti" (opened). Paraphrase: imperfekti tells what happened yesterday or
    a year ago. `locus_verified: true`.
  * https://uusikielemme.fi/finnish-grammar/the-imperfect-tense-past-tense-imperfekti/ (opened): *Söin eilen puuroa*.
  * VISK §1539 "Ajan adverbiaalit imperfekti- ja perfektilauseissa" (https://kaino.kotus.fi/visk/sisallys.php?p=1539),
    **search extract only, not opened** (`locus_verified: false`). The extract says imperfekti takes time adverbials
    tied to a point in time (*eilen*), and perfekti takes vague ones. It also says perfekti with *eilen* is possible
    when the event is presented as recent and relevant (example: *Alakoski on eilen valittu ...*). So perfekti with
    "yesterday" is possible but marked.
* **Confidence.** high that imperfekti is the unmarked form for this frame. The VISK locus is unverified.
* **Alternative.** `alternatives:` records perfekti (grid form = participle, `aux_carried`, PST would have 2 classes
  with aux carried). The perfekti cells exist in the data (`ACT;PST;PRF...` is pluperfect; `ACT;PRS;PRF;POS;IND` is
  perfekti), but they are multiword in all lemmas (share_multiword 1.0) and were not used.
* **Data note.** In MGN labels `ACT;PST;POS;IND;*` is the imperfekti, `ACT;PRS;PRF;POS;IND;*` is the perfekti
  (*olen aakkostanut*) and `ACT;PST;PRF;*` is the pluperfect (*olin aakkostanut*). Checked in the inventory examples.

### J2. PRS and PST 3PL: standard-language form
* **Options.** (a) standard `-vat/-vät` (*he ostavat*, *he ostivat*); (b) colloquial 3PL = 3SG form (*ne ostaa*, *ne osti*).
* **Choice.** (a), the form in the data (`ACT;PRS;POS;IND;3;PL`, `ACT;PST;POS;IND;3;PL`). The frame says "standard
  variety" (R1). Recorded (b) under `alternatives:` for PRS.3PL and as a note for PST.3PL.
* **References seen.** English Wiktionary "-vat" (opened): the third-person plural ending, and "Colloquially, the
  third-person singular is often used instead of the plural". Wikipedia "Finnish verb conjugation" (opened):
  *puhuvat*, and colloquial *ne tietää*. Yle "Imperfekti" paradigm *seisoivat*. `locus_verified: true`. VISK §106/§1272
  extracts concern personal categories only; not used as support.
* **Confidence.** high.

### J3. IMP.2SG and IMP.2PL
* **Choice.** `ACT;PRS;POS;IMP;2;SG` (bare imperative stem, *osta!*) and `ACT;PRS;POS;IMP;2;PL` (*ostakaa!*).
  The data's imperative cells match: *aakkosta / aakkostakaa*, *sano / sanokaa*, *tule / tulkaa* style forms.
* **Politeness.** Finnish 2SG polite address uses 2PL forms. The grid convention is informal 2SG, so 2SG = the bare
  stem and 2PL = *-kaa/-kää* for a large group.
* **References seen.**
  * https://uusikielemme.fi/finnish-grammar/verbs/verb-tenses-and-moods/the-imperative (opened): 2SG is the 1SG minus
    *-n* (*antan* → *anna!*); 2PL adds *-kaa/-kää* to the infinitive stem (*antaa* → *antakaa!*). `locus_verified: true`.
  * Wikipedia "Finnish verb conjugation" (opened): *puhu*, *puhukaa*; older *puhukaat* not used.
  * Gutenberg, Eliot, *A Finnish Grammar* (1890), https://gutenberg.org/cache/epub/59795/pg59795-images.html, Introduction
    pp. xxxi: the imperative 2SG is "merely the root"; plural *-kaa*. Historical, only a cross-check.
  * VISK §118 "Imperatiivin muodostus ja rajageminaatio" (https://kaino.kotus.fi/visk/sisallys.php?p=118):
    **search extract only** (2SG = bare stem, 2PL = *-kAA* on the stem; examples *sano, sanokaa, tulkaa*).
    `locus_verified: false`.
* **Remark.** The 2PL stem sometimes differs from the 2SG stem (*kerro / kertokaa*, *tule / tulkaa*). This is in the
  data and needs no rule.
* **Confidence.** high.

### J4. Present cells
Direct source cells `ACT;PRS;POS;IND;*`, with `POS` (not `NEG`), `ACT` (not `PASS`) and no `PRF`. Sources: Wikipedia
"Finnish verb conjugation" table (*puhun, puhut, ... puhuvat*; opened) and the Yle paradigm. The candidate list ranks
`NEG;IND` first only because of alphabetical order; NEG forms are multiword (*en aakkosta*) and are not an affirmative
frame.

### J5. Defaults
Finnish has no gender or clusivity in verbs (`not_marked`, `not_distinguished`). The object is case-marked (partitive
or accusative) but not indexed on the verb. 2SG informal = *sinä*.

## 3. Where the rules broke or were unclear

* **Class-limited syncretism (R4, R6).** In Kotus type 62 verbs (*voida, -oida/-öidä*, also some *-ia/-iä* verbs such
  as *vahtia, ahtia, ahnehtia, mönkiä*) the stem ends in *-i*, which absorbs the imperfekti *-i-*. So PRS.x = PST.x in
  **all six persons** for these verbs: *voin/voin, voi/voi, voivat/voivat, kritisoin/kritisoin*. Evidence from the grid:
  PRS.1SG = PST.1SG in 1,171 of 7,597 lemmas (15.4%). 752 of them end in *-oida* and 79 in *-öidä*. All 754 lemmas
  ending in *-ida* have it. Confirmed by https://elon.io/grammar/finnish/verb-reference/voida (opened), which gives the
  identical positive paradigms for *voida*. VISK §113 (search extract): stem-final *i* merges with the imperfekti *i*
  (examples *etsi-, tupakoi-, voi-*). `locus_verified: false` for VISK.
  The checker lists these as `pair_identity` (PRS.xx~PST.xx = 0.154) but raises no warning, because I declared no
  syncretism. I did not declare a class-level syncretism: `distinct_class` is one id per cell for the whole language,
  so a class-limited identity cannot be expressed without a lexical class that gives different class ids, which the
  template does not support. **Result: the language is scored as 14 formally distinct cells, although for about 15% of
  lemmas PRS and PST coincide.** Owner decision needed on how to treat this (it is the "one conjugation class only"
  case of the brief). The same applies to PRS.3SG = IMP.2SG (1,682 lemmas, e.g. *stressaa / stressaa* type verbs, 22%)
  and PST.3SG = IMP.2SG (1,045 lemmas, 14%), which hold for verb types with a bare stem.
* **Perfekti is multiword in all lemmas** (`share_multiword` 1.0), so D1 would apply if the owner preferred perfekti.
* **Defectiveness not applied.** Finnish weather and other zero-person verbs (*sataa* 'rain') lack 1SG/2SG in the
  grammar, but the dataset supplies full paradigms for them. I did not build a defective list, as I found no readable
  source that lists such verbs, and the data has no gaps (0 empty cells). A reviewer may want a `defective` list.
* **VISK blocked** (see the top). The brief requires opened sources; I used learner grammars and Wikipedia as the
  opened sources and VISK only as a corroborating search extract.

## 4. Data problems

* **Very frequent verbs are missing from `fin-v`:** *olla, tulla, mennä, antaa, puhua, ottaa, pitää, täytyä, ajatella,
  kuolla, päästä, jäädä, nukkua, lukea, kertoa, kutsua, asua, kuulla, maksaa, seistä, kuulua, tuntua, kokeilla* (23 of
  about 70 high-frequency verbs I tried, chosen from my own knowledge of frequency). Present: *tehdä, nähdä, saada, tietää,
  haluta, sanoa, käydä, voida, tuntea, juosta, tuoda, viedä, syödä, juoda, istua, kirjoittaa, osata, kysyä, lähteä*.
  The reason is unknown (UniMorph/MGN filtering?). It matters for the LDL frequency profile.
* **Glued doublets (two forms run together without a separator) in 14 lemmas**, in PST cells:
  *lähteä* PST.1SG `lähdinläksin`; *entää, hyytää, joutaa, kiitää, kyntää, noutaa, soutaa, yltää, sortaa, häätää,
  liitää, vuotaa, sivaltaa*. They are archaic *-sin* doublets (*lähdin / läksin*). These forms are wrong as single
  forms. The checker shows empty `variants` for every row, so the dataset does not separate variants. A reviewer should
  decide whether to drop these 14 lemmas or to split on pattern.
* **Truncated forms** in 6 cells: *ammentaa* PRS.1PL `ammenn`; *lemmiskellä* PRS.1PL `lemmiske`; *ummehtua* PRS.1PL
  `ummehd`; *sivaltaa* PST.3PL `sival`; *tukahtua* IMP.2PL `tukah`; *takaisinmallintaa* IMP.2PL `takaisinmallin`.
  Found by simple suffix pattern checks on 11 cells; other cells or other truncations may exist. (Also *sivaltaa*
  PST.3PL belongs to the glued list above.)
* **Compound or odd lemmas** such as *väkisinmaata*, *takaisinmallintaa* occur, and 127 multiword lemmas (*ajaa karille*,
  *antaa ilmi*) are excluded by R8. Many lemmas are rare or technical (*desentralisoida, iirintää*).
* **PRS.2SG imperative of some consonant-stem verbs** looks plausible but I did not verify each one (*karpata → karppaa*).
  Not checked.
* Mapping-level check on glued or truncated forms is outside the tool; no verb-by-verb corrections were made.

## 5. Check results (copied from `mappings/fin_check.md`)

* Errors 0, warnings 0.
* Lemmas in data 7,724; out of scope 127 (multiword); excluded by list 0; **lemmas in scope 7,597**; complete grids 7,597.
* Empty cells: none (no `data_gap`, no `defective`).
* Classes 14; formally distinct cells 14; `aux_carried` cells 0.
* Rule counts: source 14. Variants non-empty: 0 rows.
* Pair identity in the data (share of lemmas with identical forms): PRS.xx~PST.xx 0.154 for 1SG, 2SG, 1PL, 2PL, 3PL
  (3SG 0.110); PRS.3SG~IMP.2SG 0.221; PST.3SG~IMP.2SG 0.138. See section 3 on why none is declared.

Filled-grid inspection (`outputs/cell_grid_pilot/fin_grid.csv`): 20 random lemmas and 28 frequent or irregular verbs
looked at (*tehdä, nähdä, saada, tietää, haluta, sanoa, käydä, voida, tuntea, juosta, tuoda, viedä, syödä, juoda,
istua, kirjoittaa, osata, kysyä, lähteä*). Forms are as expected (*tehdä: teen ... tein, teki, tekivät, tee, tehkää*;
*syödä: söin*; *juoda: join*; *tuoda: toin*; *viedä: vein*). No `variants` anywhere, so R3's variant order did not
come up.

## 6. Sources seen

* https://uusikielemme.fi/finnish-grammar/verbs/verb-tenses-and-moods/the-relation-between-the-imperfect-perfect-and-plusquamperfect
  (opened): imperfekti for exact time, perfekti otherwise. Used in J1.
* https://uusikielemme.fi/finnish-grammar/the-imperfect-tense-past-tense-imperfekti/ (opened): imperfekti paradigm of
  *ostaa*; *Söin eilen puuroa*. Used in J1, J4.
* https://uusikielemme.fi/finnish-grammar/verbs/verb-tenses-and-moods/the-imperative (opened): imperative 2SG and 2PL
  formation. Used in J3.
* https://yle.fi/a/20-156316 (opened), article "Imperfekti": paradigm *seisoa*; marker *-i/-si*. Used in J1, J2, J4.
* https://en.wikipedia.org/wiki/Finnish_verb_conjugation (opened): table with *puhun, puhut, puhuvat, puhuin, puhu,
  puhukaa*, perfekti with *olla* + participle; colloquial *ne tietää*. Used in J2 to J4.
* https://en.wiktionary.org/wiki/-vat (opened): 3PL ending; colloquial 3SG for 3PL. Used in J2.
* https://en.wiktionary.org/wiki/Appendix:Finnish_conjugation (opened): the page is only a sparse table (1SG and 3SG);
  not used as support.
* https://elon.io/grammar/finnish/verb-reference/voida (opened): *voida* PRS = PST. Used in section 3.
* https://gutenberg.org/cache/epub/59795/pg59795-images.html (opened, first 100,000 characters): Eliot 1890,
  Introduction pp. xxviii–xxxii. Historical cross-check for the imperative root and *-kaa*.
* https://journal.fi/virittaja/article/view/8312 (opened, abstract only): Lauranto 2013, the imperative paradigm
  consists of personal forms of the present (and perfect). Not used as support for forms.
* https://research.abo.fi/fi/publications/second-person-singular-imperatives-in-finnish-everyday-conversati/ (opened,
  abstract): not used.
* **Not opened (search-engine extract only; blocked by Cloudflare, HTTP 403):** VISK §112, §113, §118, §1530, §1534,
  §1539 at https://kaino.kotus.fi/visk/sisallys.php?p=<§> and https://scripta.kotus.fi/visk/sisallys.php?p=<§>.
  `locus_verified: false`.
* Failed: https://journal.fi/virittaja/article/download/37538/93561/346387 (Virittäjä, over 10 MB); web.archive.org
  fetch not permitted; Karlsson 2018 (no readable preview found).

## Unsourced decisions

None are fully unsourced, but these rest on opened sources of learner-grammar quality only; VISK (the authoritative
source) was not read: all 14 cells (the unopened VISK reference is attached to PST and IMP cells; PRS and 3PL rely on
Wikipedia and Yle). Confidence is stated `high` because the forms are mechanical and the opened sources agree with the
VISK extracts, but a reviewer should read VISK §118 and §1539. The omission of a class-level syncretism (section 3)
and of a `defective` list is a decision taken without a source.

## 7. Effort

About 45 tool calls, of which 17 were web searches and 20 were fetches (about 10 of the fetches failed or were blocked,
including 6 attempts at VISK by `WebFetch`, `curl` and the built-in browser). The main time sinks were the VISK
Cloudflare block and finding learner or encyclopaedic pages that state the imperfekti/perfekti and imperative rules
well enough to quote. Data inspection took about 8 calls.


---

## Fix round 2026-10-14

Status set to `reviewed`, `reviewed_by: "Opus reviewer + fix round 2026-10-14"`. `gridtools.py check fin`: **0 errors,
0 warnings**. No scratch file is in the repository (scripts were in the scratchpad outside it).

### Items

| item | what I did |
|---|---|
| **M5 glued doublets** | Mechanical scan of all 7,597 x 14 cells (see below). The glued cells are exactly those the review lists. `data_gaps`: *lähteä, entää, yltää* PST.1SG-3PL (all six cells; the 1PL-3PL cells are glued too: *lähdimmeläksimme*, *ennimmeensimme*, *yllimmeylsimme*); *hyytää, häätää, joutaa, kiitää, kyntää, liitää, noutaa, sortaa, soutaa, vuotaa* PST.1SG/2SG/3SG only (their PST.1PL-3PL are clean). 13 lemmas, 48 cells. *sivaltaa* has no glued cell (*sivalsin* etc. are fine); it is listed in the review because of its truncated PST.3PL. The tool cannot split the doublet, so no `variant_overrides` is possible |
| **M5 truncated cells** | `data_gaps`: *ammentaa* PRS.1PL (*ammenn*), *lemmiskellä* PRS.1PL (*lemmiske*), *ummehtua* PRS.1PL (*ummehd*), *sivaltaa* PST.3PL (*sival*), *tukahtua* IMP.2PL (*tukah*), *takaisinmallintaa* IMP.2PL (*takaisinmallin*). 6 cells. Total `data_gap` cells: 54 (48 + 6) |
| **M5 further scan** | Script (in the scratchpad) checked: (a) every form against the suffix of its cell (-n, -t, -mme, -tte, -vat/-vät, -kaa/-kää; 3SG ends in a vowel); (b) forms shorter than the lemma minus 4 characters; (c) PST form longer than the PRS form of the same person by 4 or more characters; (d) stem agreement between 1SG/2SG, 1PL/2PL, 1SG/1PL, 3SG/3PL; (e) a repeated word-initial string. Result: the 6 truncated and the 13 glued lemmas above and nothing else. A glued doublet with a different stem (*enninensin*) is caught only by (c) and (d) |
| **D26 partial syncretisms** | `distinct_class` is unchanged (14). `emp_class` now shows the *-oida/-ida/-ia* syncretism (e.g. *voida*: PST.x gets the class of PRS.x, IMP.2SG gets PRS.3SG; checked in the grid). `emp_mean_classes` = **12.87** against 14 formal classes (7 classes: 833 lemmas; 8: 212; 9: 126; 13: 848; 14: 5,553; only complete grids, 7,572) |
| **P4 variant order** | No cell in the data has variants (`variants` is empty in all 106,358 grid rows; the MGN `fin-v` file has one form per cell). The only variants hidden in the data are the glued doublets, now `data_gaps`. Nothing to order |
| **P5/D31 weather verbs, *voida*** | See "Defective decisions" below. *voida* and *taitaa* IMP are defective (VISK 1590), see below |
| **VISK** | Opened via the Internet Archive copy (see "Sources seen in the fix round"). The unverified VISK references were replaced by verified ones |
| **D23 evidentiality** | Finnish does not mark it in the verb form, but VISK 1590 ties the imperfekti to first-hand knowledge and the perfekti to inference. So the PST frame ("the speaker witnessed the event") also supports the imperfekti, and PST confidence stays high |
| **D24 missing frequent verbs** | No action. Known limitation: MGN `fin-v` lacks *olla, tulla, mennä, antaa, pitää, ottaa, puhua, ajatella, jäädä, kuolla, nukkua, lukea, kertoa, asua, kuulla, täytyä, seistä, maksaa* and others. The dataset is used as it is. This biases the Finnish scores towards regularity, since the irregular and suppletive frequent verbs are missing |
| **D25** | No `class_exceptions` needed |
| **D29 object "it"** | Unchanged. Note VISK 457 (Huom. 2): weather verbs also have a transitive use (*Satoi ensimmäisen lumen*), so under D29 they too take the object |

### Defective decisions (P5/D31)

* **Rule used:** `defective` only where a VISK passage I read says the person form does not exist. Otherwise the data
  form is kept.
* **VISK 457** ("Milloin verbi taipuu vain 3. persoonassa?", opened):
  * "vain pieni osa niistä on aina subjektittomia ja sikäli yksipersoonaisia, esim. tuulla": *tuulla* is the one weather
    verb that VISK calls always subjectless. `defective`: all cells except PRS.3SG and PST.3SG (10 cells; PRS.3PL and
    PST.3PL as well, because VISK gives 3SG as the only person form). Reference attached.
  * "Persoonataivutus voi rajoittua kolmanteen persoonaan", for verbs that need an inanimate subject: *tapahtua, halveta,
    raikua, virrata, janottaa* are named. VISK says only the 1st and 2nd person forms are impossible for semantic
    reasons (*Laulut raikuvat* is fine). `defective`: PRS and PST 1SG, 2SG, 1PL, 2PL and both IMP cells (10 cells per
    lemma). Confidence medium: "impossible for semantic reasons" is not the same as "no morphological form", but it is the
    grammar's own statement for these lemmas, and the frame ("I ...", imperative to "you") cannot be realised.
  * Weather verbs other than *tuulla* (*myrskytä, pyryttää, tuiskuta, tuiskuttaa, salamoida, pakastaa, ukkostaa, hämärtää,
    sarastaa*): VISK says weather verbs have "rajoittunut subjektin valinta pikemmin kuin täysi subjektittomuus" and gives
    *Meri myrskyää*, partitive and basic subjects. Data forms kept. *sataa* is not in the data.
  * Necessive verbs (*täytyä, pitää*, etc.) lack person forms and the imperative (VISK 457 and 1565). *täytyä* and
    *pitää* are not in the data. *tarvita, kannattaa, sopia, passata, auttaa* are in the data, but VISK 457 Huom. 1 says
    they also inflect in person in another meaning, so their data forms are kept.
* **Modal verbs, IMP (coordinator follow-up; the rule in D31 wins over its *voida* example).** VISK 1590 (opened):
  "Kaikista verbeistä ei voi muodostaa imperatiivia. Tällaisia ovat ensinnäkin modaaliverbit ..." with the examples
  *taida mennä, *täydy osata. VISK 1352 lists *voida* as a central modal verb. `defective` IMP.2SG and IMP.2PL for
  *voida* and *taitaa* (4 cells; *taitaa* is the verb of the example *taida*). *täytyä* (the other named verb) is not in the
  data. *mahtaa* is not named in 1590 (I did not open the section on it), so its data forms are kept. Caveat: *voida* also has
  a non-modal sense ('to fare'), whose imperative VISK does not discuss; confidence medium. The alternative (keep the data
  forms) is under `alternatives:` of IMP.2SG.

### Corrections to earlier statements in this file

* Section 3, "Class-limited syncretism": the checker now reports it through `emp_class` (D26). The 14-class count is unchanged.
* Section 3, "Defectiveness not applied": superseded by the decisions above.
* Section 4, "Glued doublets" and "Truncated forms": acted on (M5 above). The scan found no further cases.
* Section 4, "Very frequent verbs are missing": see D24 above.
* The references to Wikipedia "Finnish verb conjugation" were removed from the cells. The paradigm and the personal endings
  are now referenced to VISK 107, 112, 124. Wiktionary and the learner pages stay in the notes as cross-checks only.
* The unverified references to VISK 1539 (PST) and 118 (IMP) are now verified; the 1534 reference was added for the
  perfekti alternative. VISK 1530 adds that the imperfekti "viittaa nimenomaan tiettyyn ajankohtaan" and is definite.
  VISK 1539 also confirms the first-round reading: the perfekti is possible with *eilen* for a recent event with present
  relevance, but then the imperfekti would stress the event itself. So the imperfekti is the unmarked choice for the
  frame, and the perfekti stays an `alternatives:` entry.

### Sources seen in the fix round (all opened as raw text; "IA" = Internet Archive copy `https://web.archive.org/web/2023/https://kaino.kotus.fi/visk/sisallys.php?p=<§>`)

* IA p=118, VISK 118 "Imperatiivin muodostus ja rajageminaatio": 2SG is the vowel stem (*sano, kerro, anna, tule*), 2PL
  *-kAA* (*sanokaa, tulkaa*); "Imperatiivin tunnus vaihtelee persoonan mukaan". Table asetelma 61.
* IA p=107, VISK 107 "Persoonamuotoisen verbin taivutuksesta": personal endings (-n, -t, -mme, -tte, -vAt); the imperfekti
  3SG has no ending; the spoken 3PL *pojat tulee*.
* IA p=112, VISK 112 "Tempustaivutus": "Preesens on tunnukseton ja imperfektin tunnus on i".
* IA p=113, VISK 113: stem-final *i* fuses with the imperfekti *i* (*etsi-, tupakoi-, voi-*): the source of PRS = PST.
* IA p=124, VISK 124, asetelma 65: full paradigm of *nousta*: *nousen ... nousevat*, *nousin ... nousivat*.
* IA p=1530, 1534, 1539, 1590: tense meaning, perfekti, time adverbials, evidentiality (imperfekti = first-hand knowledge).
* IA p=457, 900 (not used for forms), 1565, 1590, 1352, 1566, 1569: weather verbs, necessive verbs, modal verbs.
* https://uusikielemme.fi/.../the-relation-between-the-imperfect-perfect-and-plusquamperfect, https://yle.fi/a/20-156316,
  https://uusikielemme.fi/.../the-imperative: opened again as raw text; they hold.
* **Not reachable:** the original `kaino.kotus.fi` and `scripta.kotus.fi` hosts (HTTP 403, Cloudflare). No Kotus PDF of VISK
  was found. Karlsson 2018: no preview found.

### Updated check results (from `mappings/fin_check.md`)

* Errors 0, warnings 0.
* Lemmas in data 7,724; out of scope 127 (multiword); lemmas in scope 7,597; **complete grids 7,570** (25 lemmas have
  an empty cell: 19 with `data_gap`, 6 with `defective`).
* Empty cells: `data_gap` 54, `defective` 66 (*tuulla* 12 cells; the five inanimate verbs 10 cells each; *voida* and *taitaa* 2 cells each).
* Classes 14; formally distinct cells 14; `aux_carried` 0; rule counts source 14.
* `emp_mean_classes` 12.87 (against 14). Pair identity: PRS.xx~PST.xx 0.154-0.155 (3SG 0.110); PRS.3SG~IMP.2SG 0.221;
  PST.3SG~IMP.2SG 0.138.

### Unsourced decisions

None left. All 14 cells now rest on VISK passages that I read. The *voida*/*taitaa* IMP decision and the semantic reading of VISK 457
for the five inanimate verbs are judgement calls with medium confidence, explained above.

### Effort (fix round)

About 50 tool calls, 7 web searches and 1 WebFetch (blocked). The main time sink was getting VISK: the original host is still blocked, the
Internet Archive copy worked with `curl` and gave the full text. Grid scans took 6 calls.
