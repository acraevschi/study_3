# Serbo-Croatian (hbs) mapping notes

Mapping: `mappings/hbs.yaml` (reviewed, fix round 2026-10-14; see section 9). Check: `mappings/hbs_check.md` (0 errors, 1 explained warning).
Mapped by a Claude Sonnet 5.5 agent, 2026-10-14. Standard: ijekavian (D21); the forms are common to the Croatian, Bosnian and Serbian ijekavian standards.

## 1. Summary table

| cell | rule | source cell | class | aux_carried | confidence |
|---|---|---|---|---|---|
| PRS.1SG | source | PRS;1;SG (strip_accents) | PRS.1SG | no | high |
| PRS.2SG | source | PRS;2;SG (strip_accents) | PRS.2SG | no | high |
| PRS.3SG | source | PRS;3;SG (strip_accents) | PRS.3SG | no | high |
| PRS.1PL | source | PRS;1;PL (strip_accents) | PRS.1PL | no | high |
| PRS.2PL | source | PRS;2;PL (strip_accents) | PRS.2PL | no | high |
| PRS.3PL | source | PRS;3;PL (strip_accents) | PRS.3PL | no | high |
| PST.1SG | main_verb | PST;1;SG (strip_perfect_aux, strip_accents; aux sam) | PST.1SG | yes | high |
| PST.2SG | main_verb | PST;2;SG (strip_perfect_aux, strip_accents; aux si) | PST.2SG | yes | high |
| PST.3SG | main_verb | PST;3;SG (strip_perfect_aux, strip_accents; aux je) | PST.3SG | yes | high |
| PST.1PL | main_verb | PST;1;PL (strip_perfect_aux, strip_accents; aux smo) | PST.1PL | yes | medium |
| PST.2PL | main_verb | PST;2;PL (strip_perfect_aux, strip_accents; aux ste) | PST.2PL | yes | medium |
| PST.3PL | main_verb | PST;3;PL (strip_perfect_aux, strip_accents; aux su) | PST.3PL | yes | medium |
| IMP.2SG | source | IMP;2;SG (strip_accents) | IMP.2SG | no | high |
| IMP.2PL | source | IMP;2;PL (strip_accents) | IMP.2PL | no | high |

No syncretism or `restored` cells are needed: every cell has its own source column. No lexical classes. No `confidence: low` cell.

## 2. Judgement calls

**J1. PST: perfect vs aorist (vs imperfect).** Options: (a) perfect = clitic present of *biti* + l-participle (data `PST;x;y`); (b) aorist (`PST;x;y;LGSPEC2`, only 956 lemmas); (c) the `PROG;PST` columns (imperfect for imperfective, aorist for perfective lemmas, inconsistent). Choice: (a). Reason: both grammars I opened call the perfect the general past, and treat aorist and imperfect as optional or marked. Browne & Alt 2.2.1.2 says one can always safely use the perfect instead of the aorist or imperfect; HŠG calls the perfect "opće prošlo glagolsko vrijeme". For "yesterday I V-ed it, once", a news report in a main clause, the perfect is the unmarked choice. Confidence high. Alternative recorded: aorist (single word, person-marked, but only 24% of lemmas have it, so most cells would be gaps).

**J2. PST under D1/D2: which participle.** The l-participle agrees with the subject in gender and number (HŠG: *pjevao/pjevala sam, pjevali/pjevale smo*). Defaults (SPEC §3) are masculine, so SG persons take the masculine singular (-o) and PL persons the masculine plural (-li). The data already stores exactly these forms (e.g. *pisao sam*, *pisali smo*). The clitic auxiliaries *sam, si, je, smo, ste, su* are dropped by `drop_tokens`; they are separate words under D2 (HŠG: unstressed present of *biti*). Because the grid forms of 1SG/2SG/3SG are identical, and so are 1PL/2PL/3PL, each person is its own class with `aux_carried: true` (D1), giving 6 PST classes. Check: forms in these pairs are identical in 100% of lemmas, which the report accepts because both cells are `aux_carried`. Confidence high for SG; medium for PL because the masculine plural for a mixed group is a SPEC convention for which I saw no source (see section "Unsourced decisions").

**J3. 3SG *je* and the reflexive.** HŠG notes that *je* is dropped in the 3SG of reflexive verbs (*Djevojka se smijala*), and in headlines (*krnji perfekt*). The dataset has no reflexive lemma and no `se` token in any form (checked: 0 lemmas ending in ` se`, no token `se` in any cell), so no transform for `se` is needed. If reflexives are added later, `se` is a clitic and must be dropped like the auxiliary; the 3SG cell would then have no `je` to drop.

**J4. Aspect S-b; departure from R1 for perfective lemmas.** R1 says the grid form is the form the lexical verb has in the context sentence. The PRS context is habitual ("Every morning I V it"). For a perfective lemma such as *napisati* the present is a perfective non-past that does not normally express a habitual present in a main clause; Browne & Alt 2.2.1.4 say the perfective present does not mean future outside when/if clauses, and gives a "typical action" reading only when the context signals generalisation. Strictly R1 would send the habitual context to the imperfective partner (*pisati*), with a different stem class. Under D13 (S-b) every lemma instead keeps its own present, perfective or not. The PST context is natural for both aspects (HŠG: the perfect forms from perfective and imperfective verbs), and the IMP context has a perfective command reading. So the departure affects only the PRS cells for perfective lemmas (a large share, because the data includes prefixed perfective verbs); the forms are still the lemma's own present forms. Confidence high that this is what D13 asks; the interpretation cost is for the LDL phase.

**J5. IMP.** The 2SG and 2PL imperative forms are taken as given. HŠG lists the endings (-ø, -j, -i, -ji; 2PL -te, -jte, -ite); Gall's handout says verbs with 3PL in -ju/-je take -j, all others -i. No auxiliary, so `aux_carried: false`. Confidence high.

**J6. 2PL = plural (large group).** "you all" is mapped to 2PL; the polite singular *vi* is excluded by SPEC §3. The perfect 2PL takes the plural participle (*pisali ste*), which is what the data has.

## 3. Where the rules broke or were unclear

* **Mislabelled imperatives.** The 268 lemmas with past conditionals in the IMP columns (section 4) were first entered under `defective` because the checker had no way to say data_gap for a list. After the tool update they are in `data_gaps:` and appear as `na_reason: data_gap` with a note. `defective` is now empty (the grammar defines no defective cell in this data).
* D1 asks for aux-carried distinctions, which makes PST six classes with identical grid forms. This is as designed, but the "different class, identical forms" flag is silenced only because both cells carry `aux_carried: true`. Fine.
* SPEC §4 R5 (exclude verbs) is superseded by D8/D11; I followed D8.
* The `locus_verified: true` flags rest on pages I opened, but I could not access Alexander 2006 or Browne 1993.

## 4. Data problems

1. **268 lemmas (6.7%) have a mislabelled imperative.** `IMP;2;SG` holds the past conditional (*bio bi halucinirao*) and `IMP;2;PL` holds *bili biste halucinirali*; `IMP;1;PL` holds *bili bismo ...*. The real imperative is absent. Saved as `mappings/classes/hbs_imp_mislabelled.csv` (column `lemma`, 268 lemmas; selected as any lemma whose IMP;2;SG or IMP;2;PL has a space). It includes *htjeti*. These cells are empty in the grid (`data_gaps`, na_reason data_gap). They were not repaired by rule (no ad hoc filling, D8).
2. **Frequent verbs are absent:** *biti, moći, ići, znati, imati, vidjeti, jesti, stati, pasti, živjeti, voljeti* (not in the raw file either; the loader's doublet filter is not the cause). The suppletive and irregular verbs most relevant to inflection are therefore missing; *htjeti, dati, doći, reći, uzeti, smjeti* are present.
3. **17 lemmas have no usable forms at all** (all cells NA): the *-pasti* family (*napasti, ispasti, zapasti, dopasti, propasti, pripasti, prepasti, upasti, otpasti, opasti*), *mnijeti*, and Cyrillic lemmas (*брецати, гарантовати, гњурити, дирнути, стењати, шљемати*). They stay as lemmas with all cells `data_gap`. Two further lemmas have one missing cell (*postizati* PRS.3PL, *vapiti* PRS.1SG).
4. **Cyrillic lemmas and tokens:** 10 lemmas contain Cyrillic or odd characters (*pośetiti, śeđeti, ...*). One (*garantovati*) has the Cyrillic auxiliaries *су/смо/сте*; I added the Cyrillic tokens to `strip_perfect_aux` (they dropped once each).
5. **35 lemmas lack the auxiliary** in the PST cells (e.g. *izmaknuo* alone, the *-nuti* family). The bare participle is already the grid form, so the transform leaves it unchanged. The labelling is inconsistent but harmless.
6. **Accent marks:** 28 lemmas have accented forms in some cells (*dȃm*, *dojàvljujēm*, *opòreknēm*). Kept as written (R7). They will count as different strings from the unmarked forms if the representation is not normalised; the LDL phase should decide. (Fixed in the fix round: the `strip_accents` transform removes them, section 9, M10.)
7. **3PL typo:** *grebati* and *sevdisati* have 3PL in *-eu* (*grebeu*) instead of *-u/-e*.
8. **Archaic present of *reći*:** the data gives *reknem, rekneš...* and imperative *rekni*, while Gall's handout says *reknem/rečem* are archaic and the standard imperative is *reci* (Gall, Imperativ handout: reci regelmässig gebraucht). Data forms kept (data is not changed by hand); flagged for the reviewer. (Fixed in the fix round: `reći` PRS.* and IMP.* are now `data_gaps`, section 9, M9.)
9. **Ekavian-only lemmas remaining after the loader's doublet filter:** a conservative heuristic (lemma in *-eti*, present in *-im*, l-participle plural in *-eli* not *-jeli*) finds 8: *mrzeti, pizdeti, pobledeti, posedeti, vreti, zavredeti, śeđeti, bledeti*; I add *sazreti* by inspection (present *sazrem*, participle *sazreli*; 9 in total). Verbs of the other ekavian patterns cannot be found without a lexicon, so 9 is a lower bound. Serbian-flavoured vocabulary is also present: 30 *-isati* verbs vs 0 *-izovati* (the Croatian-leaning bias already noted in D21).
10. **Vocabulary:** 302 *-irati* verbs (45 *-izirati*), 87 *-ovati*; most are biaspectual loans. No effect on the grid forms.

## 5. Check results (copied from `hbs_check.md`; first-round numbers. The numbers after the fix round are in section 9)

* Lemmas in data 3,999 (349 ekavian doublets dropped by the loader); in scope 3,999 (0 multiword).
* Complete grids: 3,712 of 3,999.
* Empty cells: `data_gap` 704 (= 536 from the `data_gaps` list, 268 lemmas x 2 IMP cells, plus 168 from missing data: 17 lemmas with all cells NA and 2 single cells); `defective` 0.
* Classes 14; formally distinct cells 14 (every cell is its own class).
* Rules: source 8, main_verb 6; aux_carried cells 6.
* Dropped clitic auxiliaries: `sam` 3957, `si` 3957, `je` 3957, `smo` 3959, `ste` 3959, `su` 3959, Cyrillic `смо/сте/су` 1 each: 9 distinct strings (small).
* Empirical distinctness (complete grids, 3,712): distinct forms per grid 8 in 1,288 grids, 9 in 2, 10 in 2,422; mean 9.31 of 14. Share of lemmas where the cell's form is unique in the grid: PRS.1SG/2SG/1PL/3PL 1.0; PRS.3SG, PRS.2PL, IMP.2SG, IMP.2PL 0.653 (accidental identity of PRS.3SG with IMP.2SG and PRS.2PL with IMP.2PL in -i- verbs); PST cells 0.0 (identical within SG and within PL, carried by the auxiliary, D1). So the 14 formally distinct classes overstate the empirical distinctness (about 9-10 distinct forms).
* Warnings: 0. Pair identity: PST.1SG~2SG~3SG and PST.1PL~2PL~3PL are 1.0 (by design, aux_carried); PRS.3SG~IMP.2SG and PRS.2PL~IMP.2PL are 0.347 (accidental identity in *-i-* verbs, not a paradigm syncretism, so not declared; R4 last bullet).
* `variants` is empty for every row (the data has one form per cell), so there is no R3 variant order to confirm.
* Filled-grid inspection: 20 random lemmas and 28 frequent lemmas (*raditi, reći, činiti, htjeti, dati, doći, uzeti, misliti, trebati, morati, sjediti, pitati, željeti, smjeti, pisati, govoriti, dobiti, napraviti, ostati, početi, pokazati, tražiti, vratiti, smatrati, poći, naći, pomoći, prihvatiti*) show the expected forms: the PST forms are bare masculine l-participles (*pisao, pisali*). The imperative of *htjeti* is empty (data problem 1).

## 6. Sources seen

Seen means I opened the page or file and read its text, through a web fetch.

1. Hrvatska školska gramatika (Institut za hrvatski jezik i jezikoslovlje), https://gramatika.hr/pravilo/perfekt/50/ : perfect = unstressed present of *biti* + l-participle; "opće prošlo glagolsko vrijeme"; participle agrees in gender and number; 3SG *je* dropped with reflexive verbs; formed from both aspects.
2. HŠG, https://gramatika.hr/pravilo/glagolski-pridjev-radni/40/ : l-participle endings m -o/-li, f -la/-le, n -lo/-la; *-io* for *-iti/-jeti* verbs.
3. HŠG, https://gramatika.hr/pravilo/aorist/52/ : aorist endings; formed mainly from perfective verbs; no statement on register in the part I could read.
4. HŠG, https://gramatika.hr/pravilo/imperativ/55/ : imperative endings -ø/-j/-i/-ji, 2PL -te/-jte/-ite; examples *pjevaj, budi*.
5. HŠG, https://gramatika.hr/pravilo/sadasnje-vrijeme-prezent/45/ : present endings -am/-im/-em/-jem sets with 3PL -aju/-e/-u/-ju.
6. Browne & Alt 2004, *A Handbook of Bosnian, Serbian, and Croatian* (SEELRC). I could not open the SEELRC original (the project page returned 404); I read a copy at https://vdoc.pub/documents/a-handbook-of-bosnian-serbian-and-croatian-70k125oj58r0 (hosted by a third party, permission unverified). Sections used: 2.2.1.2 (aorist and imperfect optional; "can always safely use the perfect"), 2.2.1.3.2 (perfect "all-purpose past tense", enclitic *sam*), 2.2.1.4 (aspect; perfective present), 2.2.1.8 Table 15 (l-participle gender/number), 2.2.2.1.9.3, 2.2.2.2, 2.2.2.3 (present paradigms incl. 3PL), 2.2.1.6.1 and 3.2.2 (imperative), 3.1.5.2 (clitic order; "se + je" becomes se).
7. Gall, J., BKMS G2 Perfekt (ijekavisch), Slavisches Seminar, Universität Zürich, https://dlf.uzh.ch/openbooks/bkmsa2/wp-content/uploads/sites/107/2022/03/PB-03.2-BKMS-ije-Perfekt-gramatika.pdf : perfect table, short auxiliary + l-participle m/f/n and pl; *reći* present *reknem/rečem* archaic (a one-page table).
8. Gall, J., BKMS G2 Imperativ, https://dlf.uzh.ch/openbooks/bkmsa2/wp-content/uploads/sites/107/2022/04/Imperativ.pdf : -ju/-je verbs take -j, all others -i; *reci, idi, dođi*.
9. Grambank sout1528 values for GB082-GB121 (local file `external/grambank/cldf/values.csv`); source Alexander 2006 pages as listed in the YAML.

**Added in the fix round** (each page or PDF was read as full text in this session: pages saved with curl to a temporary folder outside the repository, and the PDFs opened with the PDF reader; no fetch summary was used for a `locus_verified` claim):

10. HŠG, https://gramatika.hr/pravilo/naglasak-i-naglasne-cjeline/3/ : accent signs are notation; quote "Naglasak je isticanje jednoga sloga u riječi silinom, duljinom i visinom tona".
11. HŠG imperative page re-read as text (source 4): "ne postoji imperativ prvoga lica jednine"; no modal verb is named.
12. Gall, BKMS G2 Perfekt (source 7), re-read in full: *reći* prezent "(reknem, rečem – archaische Formen, nicht mehr im Standard)"; l-participle *rekao*.
13. Gall, BKMS G2 Imperativ (source 8), re-read in full: "reći (unreg. reknu) reci, recimo, recite"; "Imperativ von reći wird regelmässig gebraucht". The page also lists *nemoj, nemojte* and *hajde* as defective verbs; no modal verb of the D31 list is mentioned.
14. Browne & Alt, vdoc copy (source 6), full text searched: 2.2.1.6.1 "Reci! Recite!"; 3.2.2 negative command *nemoj* + infinitive; the text has no statement on the imperative of *morati, trebati, smjeti, htjeti* (these verbs do not occur in the copy's text at all). The SEELRC original was not opened again.
15. HŠG perfect, l-participle and present pages re-read as text (sources 1, 2, 5): *htio/htjeli*; perfect "opće prošlo glagolsko vrijeme"; present endings.

**Not seen:** Alexander 2006 (search returned only catalogue records; no preview reached) and Browne 1993 (record only). Neither is cited as verified.

## 7. Unsourced decisions

* **Masculine plural participle for 1PL/2PL/3PL (mixed large group).** HŠG shows agreement but I did not see a source stating that a mixed group takes the masculine plural. The choice follows SPEC §3 (masculine default) and the data's own default cell. PST.1PL/2PL/3PL: `confidence: medium`.
* **Perfective-present habitual reading (J4):** supported only by a paraphrase from Browne & Alt 2.2.1.4 (typical action when context signals it); no source seen on how natural it is for a given lemma.
* No cell was left without any source; Alexander 2006 and Browne 1993 are unsourced cross-checks only.

## 8. Effort

Fix round: about 52 tool calls and 3 web searches (plus 6 page or PDF fetches, 3 pages saved as text with curl). First round: about 45 tool calls in total: about 6 web searches and about 16 page fetches (3 of which failed with 403/404/429), the rest local file and data inspection. Main time sinks: finding an openable copy of Browne & Alt (no SEELRC original), reading binary PDFs from the Zürich site, and discovering the mislabelled imperative columns in the data.

## 9. Fix round 2026-10-14

Instructions: FIXROUND.md ("For every language" and "hbs"), REVIEW.md M9-M11, §3 decision 7, DECISIONS D23-D32. Provenance is now `status: reviewed`, `reviewed_by: "Opus reviewer + fix round 2026-10-14"`.

| item | what I did |
|---|---|
| M10/D32 accents | Declared the transform `strip_accents` (type `strip_marks`) with codepoints 0300, 0301, 0302, 0304, 030F, 0311 and protected letters ć, Ć, ś, Ś. Applied it to every cell: `transform: strip_accents` for the 8 PRS and IMP cells, and `[strip_perfect_aux, strip_accents]` for the 6 PST cells (the auxiliary is dropped first). Which letters need protection: I listed every letter that carries one of the six marks in the grid after NFD. Acute occurs on c (1,590 forms, this is ć), s (28 forms, ś in *pośetiti*, *śeđeti*) and on e/u/ú in stray accent marks (*stégnuo*, *zaljutiti*), so ć and ś are the only letters that need protection (ć/Ć and ś/Ś are listed in both cases). The caron (U+030C, č š ž) is not in the list. đ and Cyrillic ћ have no decomposition. Result: *dȃm* -> *dam*, *pȉjēm* -> *pijem*, *hoću/hoćeš* keep ć, *pośetim* keeps ś. No other mark remains in the grid except ć and ś (checked by script: 0). 43 distinct forms changed (see the warning below). Reference: HŠG, gramatika.hr/pravilo/naglasak-i-naglasne-cjeline/3/ ("Naglasak je isticanje jednoga sloga u riječi", the accent signs and length mark are notation). |
| M9 *reći* | `reći` PRS.1SG-3PL and IMP.2SG/2PL are `data_gaps` (8 cells). Reason as instructed: archaic form only (*reknem/rekni*), standard *rečem/reci*. **What Gall says exactly** (both PDFs opened and read in full this session, see section 6): the Perfekt handout gives for the present "(reknem, rečem – archaische Formen, nicht mehr im Standard)", so Gall calls both *reknem* and *rečem* archaic; the Imperativ handout gives "reći (unreg. reknu) reci, recimo, recite" and says the imperative of *reći* is used regularly. So Gall supports the gap for the data's *rekni*; for the present it says that neither form is standard. PST (*rekao, rekli*) stays. Browne & Alt 2.2.1.6.1 also give *Reci! Recite!* as the imperative of *reći*. **Not done (outside the instruction):** seven prefixed derivatives have the same *-knem/-kni* pattern (*izreći, odreći, poreći, proreći, zareći, oporeći, опорећи*; *uteći, zateći* have *-knem* too, but *teći* is not a case of Gall's statement). Gall does not mention them, so they keep the data forms. A reviewer may want to extend the gap to them. |
| M11 wrong l-participles | `bdjeti` and `bosti` PST.1SG, 2SG, 3SG are `data_gaps`. The data has *bdije* (a present form) for *bdjeti*. For *bosti* the data has *bȏ*; **note:** after `strip_accents` this would be *bo*, which looks like the correct l-participle (bosti, bola; HŠG shows the *-ao*/*-o* pattern but not this verb, so I have no source for *bo*). I followed the instruction and made the cells gaps; if the reviewer accepts *bo*, drop *bosti* from this gap entry. PL cells (*bdjeli, boli*) are fine. I also scanned all PST.1SG cells mechanically: no other lemma lacks the endings -o/-ao/-io. |
| D31 modal imperatives | Looked up *morati, trebati, smjeti, htjeti*. HŠG 'Imperativ' (gramatika.hr/pravilo/imperativ/55/, read in full as text): the only gap it states is that there is no first-person singular imperative ("ne postoji imperativ prvoga lica jednine"); it does not name any verb without an imperative, and it names no modal. HŠG has no page on modal verbs (checked the site navigation). Browne & Alt (vdoc copy, full text read; sections 2.2.1.6.1 and 3.2.2): they say nothing on the imperative of these verbs; *nemoj* + infinitive is given as the polite negative command. Two web searches (*morati*, *htjeti* imperative) found no grammar that says the imperative does not exist (one learner site lists *htij, htijte* for *htjeti*; not used). So under D31/P5 no cell is `defective`: *moraj/morajte, trebaj/trebajte, smij/smijte* keep the data forms. *htjeti* IMP.2SG/2PL stay `data_gaps`, but for the data reason (the columns hold the past conditional, section 4 item 1), not because the grammar lacks the form. Possible risk, recorded: *moraj* and *trebaj* are not standard in common usage; the sources I opened do not say so, so I did not act on it. |
| P4 variant order | `variants` is empty in all 55,986 rows (the data has one form per cell), so there is no variant order to rank. Nothing to do. |
| Aux (optional) | Added `aux:` to the PST cells: *sam, si, je, smo, ste, su* (1SG..3PL). The empirical class now separates cells by the auxiliary as before; no change in `emp_mean_classes` from this. |
| D23 evidentiality | The PST frame now says the speaker witnessed the event. For hbs it changes nothing: the perfect is the general past (HŠG "opće prošlo glagolsko vrijeme"; Browne & Alt 2.2.1.3.2), and it has no evidential contrast. The aorist is still not chosen (J1). |
| D29 transitivity ("it") | The l-participle and the present do not depend on the object, so no change. |
| D24 missing frequent verbs | Not acted on. Still a known limitation: *biti, moći, ići, znati, imati, vidjeti, jesti, stati, pasti, živjeti, voljeti* are not in the dataset (section 4 item 2). |
| D25/D26 | No `class_exceptions` needed (all 14 cells are separate classes). `emp_mean_classes` is reported below. |

**Check after the fix round** (`gridtools.py check hbs`): errors 0; warnings 1: "transform strip_accents: 43 distinct dropped strings (an auxiliary set should be small)". The warning is expected: for `strip_marks` the tool counts the forms it changed (*mȉo*, *stégnuo*, *opòreknēm*, *dȃm*, and the *ìznjedr-* forms of *iznjedriti*, 15 listed in all), which is not an auxiliary set. It is the 79-form accent problem of M10, split by form. Nothing to fix.

* Lemmas in data/scope 3,999; complete grids 3,709 (was 3,712; minus *reći, bdjeti, bosti*).
* Empty cells: `data_gap` 718 (was 704; +8 *reći*, +3 *bdjeti*, +3 *bosti*); `defective` 0.
* Classes 14; formally distinct cells 14; rules source 8, main_verb 6; aux_carried cells 6.
* Dropped clitic auxiliaries unchanged (*smo/ste/su* 3,959, *sam/si/je* 3,957, Cyrillic 1 each).
* Empirical: distinct forms per complete grid 8 in 1,290 grids and 10 in 2,419 (the former 9-form grids, 2 lemmas with accents, are gone: the accent made two cells look different). `emp_classes_per_complete_grid`: 12 in 1,290 grids, 14 in 2,419 grids; **`emp_mean_classes` 13.3** (formal: 14; was 13.31 before the fix). PRS.3SG = IMP.2SG and PRS.2PL = IMP.2PL hold in 34.8% of lemmas (-i- verbs): they appear in `emp_class` (12 classes) and `distinct_class` stays 14 (D26).
* The notes in sections 1-8 describe the first round; where this section differs, this section holds.

### Follow-up from the main agent (same day)

1. **bosti (M11) reversed.** HJP (hjp.znanje.hr, entry *bosti*, page fetched as text with a POST search and read in full) gives "prid. rad. bȏ/bȍla ž" and, in its table, the active participle *bo, bola, bolo; boli, bole, bola* and the perfect *boli smo*. Wiktionary (en.wiktionary.org/wiki/bosti, Serbo-Croatian conjugation table, read as text) agrees: perfect *bȏ sam*, *bòli smo*. So the data form *bȏ* is the correct l-participle, and after `strip_accents` the grid form is *bo*. I removed *bosti* from the M11 gap; `bosti` PST.1SG-3SG = *bo*. *bdjeti* stays a gap (data *bdije* is the present). The earlier remark in the M11 row above is superseded.
2. **Derivatives of reći (M9).** The same gap (PRS.1SG-3PL, IMP.2SG/2PL; same Gall references) now covers *izreći, odreći, poreći, proreći, zareći, oporeći*: all have the data forms *-reknem... / -rekni* (checked in the grid). 7 lemmas x 8 cells. Not included: *uteći, zateći* (*uteknem*, *utekni*; Gall's statement is about *reći*, and I opened no source on *teći*) and other *-knuti* verbs, which are regular.
3. **Cyrillic-script lemmas excluded.** `classes/hbs_cyrillic_excluded.csv` (8 lemmas, built by a script as the lemmas with a character in U+0400-U+04FF): *брецати, гарантовати, гњурити, дирнути, одмеравати, опорећи, стењати, шљемати*. Rule: a lemma written in Cyrillic is a script duplicate or a script-mixed entry of a Latin lemma; the Latin forms are mapped. It is listed in `exclude_lemma_files` (8 lemmas out of scope). The Cyrillic auxiliaries *смо/сте/су* were therefore removed from `strip_perfect_aux` (only *garantovati* used them).
4. **Check after the follow-up:** errors 0, warnings 0 (the tool no longer warns for `strip_marks`). Lemmas in data 3,999; excluded by list 8; in scope 3,991; complete grids 3,702; `data_gap` cells 685; `defective` 0; `emp_mean_classes` 13.3 (distinct forms per complete grid: 8 in 1,290 grids, 10 in 2,412; emp classes 12 in 1,290, 14 in 2,412). These numbers replace those above.
