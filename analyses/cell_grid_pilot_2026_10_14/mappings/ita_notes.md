# Italian (ita): mapping notes

Mapping: `mappings/ita.yaml`. Data: Morph-it! 0.48 (UniMorph-style labels from `gridtools.py prepare`). First written 2026-10-14 by the ita mapping agent (Sonnet 5.5); revised in the fix round of the same day (see section 8, "Fix round 2026-10-14"). Check after the fix round: 0 errors, 0 warnings.

Reading note: the first mapping agent read web pages through a summarising fetch tool. The fix round re-opened every cited page as raw text (HTTP download, HTML stripped) and quotes from that text; quotations are at most 15 words. Printed grammars (Maiden & Robustelli) were not readable; no page numbers are given.

## 1. Summary table

| cell | rule | source cell | class | aux (D1) | aux_carried | confidence |
|---|---|---|---|---|---|---|
| PRS.1SG | source | IND;PRS;1;SG | PRS.1SG | - | no | high |
| PRS.2SG | source | IND;PRS;2;SG | PRS.2SG | - | no | high |
| PRS.3SG | source | IND;PRS;3;SG | PRS.3SG | - | no | high |
| PRS.1PL | source | IND;PRS;1;PL | PRS.1PL | - | no | high |
| PRS.2PL | source | IND;PRS;2;PL | PRS.2PL (shared with IMP.2PL) | - | no | high |
| PRS.3PL | source | IND;PRS;3;PL | PRS.3PL | - | no | medium (R3/P4 variant ranking, 2.5) |
| PST.1SG | main_verb | V.PTCP;PST;SG;MASC | PST.1SG | ho / sono | yes | medium |
| PST.2SG | main_verb | V.PTCP;PST;SG;MASC | PST.2SG | hai / sei | yes | medium |
| PST.3SG | main_verb | V.PTCP;PST;SG;MASC | PST.3SG | ha / è | yes | medium |
| PST.1PL | main_verb | V.PTCP;PST;SG;MASC (aux_essere: PL;MASC) | PST.1PL | abbiamo / siamo | yes | medium |
| PST.2PL | main_verb | same as PST.1PL | PST.2PL | avete / siete | yes | medium |
| PST.3PL | main_verb | same as PST.1PL | PST.3PL | hanno / sono | yes | medium |
| IMP.2SG | source | IMP;PRS;2;SG | IMP.2SG | - | no | medium |
| IMP.2PL | source | IMP;PRS;2;PL | PRS.2PL (shared) | - | no | medium |

No transform is needed: Morph-it stores the participle alone, and Italian auxiliaries are not clitic. Nothing is dropped, so `transforms: []`. No cell has `confidence: low`. The `aux` column is the optional `aux:` field of the mapping (first value: avere verbs; second value: the `aux_essere` class, set through `by_class`).

## 2. Judgement calls

### 2.1 PST: passato prossimo or passato remoto for "yesterday, once"

* **Options:** (a) passato prossimo (aux + participle); (b) passato remoto (synthetic, `IND;PST;PFV`).
* **Choice:** (a), main_verb rule, grid form = participle (D1). D28: the prossimo stays; the remoto is the planned sensitivity alternative.
* **Frame:** D23 adds "the speaker witnessed the event (direct evidence)". Italian does not mark evidentiality in the past tense, so the frame changes nothing here.
* **Evidence for the prossimo (opened this session, raw text):**
  * Crusca, Mortara Garavelli and Nencioni (https://accademiadellacrusca.it/it/consulenza/sulluso-del-passato-remoto/126, 12 March 2004): the choice "non dipende dalla distanza temporale degli avvenimenti"; the example "Stamattina (ma anche 'ieri' ...) è morto il mio vicino di casa" takes the prossimo because the event is felt as current. Nencioni: the best grammars say the prossimo prevails in northern Italy; in Tuscany the alternation is still alive.
  * Crusca, Mauroni (https://accademiadellacrusca.it/it/consulenza/ancora-sulluso-del-passato-remoto/241, 18 September 2009): informal usage keeps the remoto in the Mezzogiorno, while the northern and part of the central regions replace it "perlopiù dal passato prossimo"; in formal usage the remoto is "ben saldo", above all for concluded events with no link to the present.
  * Treccani, E. Wiberg, s.v. *passato prossimo* (2011, https://www.treccani.it/enciclopedia/passato-prossimo_(Enciclopedia-dell'Italiano)/): the aoristic prossimo ("senza legame con il momento dell'enunciazione") is typical of the North and spreads to the rest of the country except Tuscany.
* **Contrary evidence (real, not garbled; M12a):** Treccani, E. Wiberg, s.v. *passato remoto* (2011, https://www.treccani.it/enciclopedia/passato-remoto_(Enciclopedia-dell'Italiano)/), "Differenze regionali" and the section on texts: "si nota al Nord una netta preferenza per il passato remoto" for personal narration (the example is a Gazzetta di Modena match report, "giocai una buona partita"); in the Centre the two tenses are about equally frequent; for impersonal narration the North splits about equally. The first mapping agent called this summary "garbled"; the raw text shows that it is correct. The two Treccani entries and the Crusca pages therefore disagree for the North, depending on text type and register: the prossimo for informal, news-like and "felt as current" events, the remoto for written personal narration. The frame ("Asked what happened yesterday: Yesterday I V-ed (it), once", reported as news) is informal speech about a recent event, which favours the prossimo. This is why the confidence stays medium.
* **Reason for the choice:** R3(1) (the form the grammar gives for the context) beats the single-word preference of R3(2).
* **Alternatives (in `alternatives:`):** passato remoto (six synthetic cells, no aux_carried, no participle-agreement class); and, from D27, "aux_both verbs take essere" (2.3).
* **Grambank:** GB083 = 1 and GB086 = 1 describe the synthetic past; GB121 = 0 says tense is not marked by an auxiliary, GB120 = 1 says aspect is (perfect). Listed as a conflict with the reason (Grambank counts the perfect as aspect). Maiden & Robustelli could not be read, so the codings are unchecked.

### 2.2 Participle agreement under D1

* **essere-verbs:** the participle agrees with the subject (Treccani, L. Gaeta, s.v. *accordo*, 2010: with auxiliary essere agreement is obligatory and the participle agrees in gender and number with the subject, example "Mario e Giovanna sono arrivati ieri"; Treccani, V. Egerland, s.v. *participio*, 2011: "siamo tornati"). The masculine default gives masculine singular for SG persons and masculine plural for PL persons. Class `aux_essere` overrides the plural cells to `V.PTCP;PST;PL;MASC`.
* **avere-verbs:** the participle is invariant. Gaeta, s.v. *accordo*: with avere and a postposed object the grammar "prevede la selezione del valore non marcato di maschile singolare" (examples "i bambini hanno mangiato la torta").
* **Clitic object "it":** Treccani, E. De Roberto, s.v. *oggetto*, 3.1.3 "Accordo del participio" (2011): the participle agrees in gender and number with a third-person clitic object (*l'ho vista*, *le ho viste*). The frame's object "it" is a proclitic *lo*; I take it as masculine singular (the SPEC §3 masculine default), so agreement gives the same form as the invariant default. A feminine or plural "it" would give *-ata/-ati/-ate* in avere verbs. Not mapped; an open assumption (the frame does not give the gender of "it").
* **Consequence under D1/D26:** the six PST cells are six classes, all `aux_carried: true`, and now each names its auxiliary in `aux:` (ho/hai/ha/abbiamo/avete/hanno; for `aux_essere` through `by_class`: sono/sei/è/siamo/siete/sono). Per lemma, the empirical class (`emp_class`) of a PST cell is therefore never merged with another PST cell or with a PRS cell.

### 2.3 aux_both, unclassified lemmas and the retired proxy class (D27)

* **Rule now:** every `aux_both` verb (598 in Wiktionary, 597 in scope) is an *avere* verb with an invariant participle. Lemmas without auxiliary information (28 left in scope after M4) are also avere verbs. Only the 152 in-scope `aux_essere` lemmas (essere only in Wiktionary) have an agreeing participle.
* **Why:** the first mapping split `aux_both` by the first Wiktionary sense (203 lemmas to essere). The review showed that this proxy splits natural pairs (*salire* essere but *scendere* avere) and has no source behind it. D24 rules out a frequency-based hand check; option (b) of the review is mechanical, and avere is grammatical for every verb that takes both auxiliaries. It also fits the frame: a verb with a transitive use takes the object "it" (D29), and the transitive verb takes avere (Treccani, M. Cennamo, s.v. *transitivi e intransitivi, verbi*, 2011: "nei tempi composti i verbi biargomentali hanno sempre l'ausiliare avere"; Treccani, R. Cimaglia, s.v. *ausiliari, verbi*, 2010: "di regola, si usa avere nei tempi composti della diatesi attiva").
* **Cost, stated plainly:** for intransitive verbs of the *aux_both* list whose standard auxiliary is essere (*arrivare, partire, morire, tornare, piacere*, ...), the grid now gives the avere auxiliary (*ho, abbiamo* ...) and an invariant participle. This is a known bias of the decision, not an Italian fact. The sensitivity alternative "aux_both → essere" is recorded under `alternatives:` of PST.1SG; the PL participle of PST.1PL/2PL/3PL would change for those lemmas.
* **Retired:** `classes/ita_both_intransitive_first.csv` is no longer a class in the mapping (the file stays in the folder only as the record of the retired proxy).
* **Unchanged weak points:** modal verbs (*potere, volere, dovere*) take the auxiliary of the governed infinitive (Treccani, s.v. *ausiliari, verbi*: the modal can take avere when the governed infinitive is intransitive, and with a pronominal infinitive the auxiliary depends on the clitic position); paradigm-level mapping cannot represent that; they are avere verbs.
* **Confidence:** medium for the mechanical part; the lexical class assignment is a documented simplification.

### 2.4 IMP.2SG and IMP.2PL: class identity and irregulars

* **Regular forms (opened, raw):** Treccani, R. Cimaglia, s.v. *imperativo* (2010, https://www.treccani.it/enciclopedia/imperativo_(Enciclopedia-dell'Italiano)/), "Definizione e peculiarità morfologiche": the tu form of the first conjugation is the only imperative-specific one; in the other two it coincides with the PRS.2SG ("temi, senti = tu temi, tu senti"); the voi form "coincide sempre" with PRS.2PL. *-isc-* verbs have the augment in the imperative (Treccani, s.v. *coniugazione verbale*, 2010: "finisci!").
* **IMP.2SG choice:** source cell `IMP;PRS;2;SG` (the data fits the grammar). The identity with the present is class-specific and has a different partner per class (PRS.3SG for -are, PRS.2SG for -ere/-ire), so no single class id fits; the cell stays its own class. D26: the per-lemma `emp_class` now records the identity (IMP.2SG = PRS.3SG in 79.1 percent, = PRS.2SG in 20.5 percent of the lemmas).
* **IMP.2PL choice (D25):** the grammar states "coincide sempre" with PRS.2PL, so the cell shares the class id `PRS.2PL`. It uses rule `source` from the imperative cell (not `syncretism`) because *essere* (siate), *avere* (abbiate), *sapere* (sappiate) and *volere* (vogliate) differ (Cimaglia: "avere → abbiate; essere → siate; sapere → sappiate; volere → vogliate"). These four are listed under `class_exceptions` for PRS.2PL and IMP.2PL. Pair identity PRS.2PL~IMP.2PL is 1.000 once they are ignored. Alternative: own class (13 formally distinct cells instead of 12).
* **Irregular tu forms (opened, raw):**
  * Cimaglia: "dire ha un imperativo monosillabico di'"; fare, stare, dare, andare have fa', sta', da', va' and also fai, stai, dai, vai.
  * Treccani, A. Viviani, s.v. *imperativo, forme di [prontuario]* (2011): the imperatives of fare, andare, stare, dare are conventionally written fa', va', sta', da'; essere and avere have dedicated 2SG forms sii and abbi.
  * Treccani, *La grammatica italiana* (2012), s.v. *imperativo*: for essere, avere, sapere, volere the 2PL uses the regular subjunctive forms (siate, abbiate, sappiate, vogliate) and the 2SG "le antiche forme sii, abbi, sappi, vogli".
  * Treccani, Vocabolario on line: fare "imperat. fai o fa'"; dare "imperat. dai o da' ... o dà"; dire "imperat. di' o dì".
* **Variant order (P4):** the Vocabolario lists the full forms (fai, dai) first, the Enciclopedia and its prontuario give the apostrophe forms (fa', da') as the convention; none ranks the two. The Morph-it order (apostrophe form first) is kept and flagged as unranked (P4 rule 3). *disfare* and *soddisfare* IMP.2SG begin with the unapostrophed form (Vocabolario s.v. *disfare*: "imperat. disfà o disfài o disfa'"); kept in data order for the same reason.
* **Defective (P5, sourced):** *potere* and *dovere*: no imperative (Cimaglia: stative verbs "come dovere, parere, potere"; Viviani prontuario: "Non hanno imperativo i verbi modali potere, dovere e volere"; Vocabolario s.v. *potere* and *dovere*: "manca l'imperativo"). *volere* is in `data_gaps`, not `defective`: the sources disagree (Vocabolario s.v. *volere*: "imperat. vògli"; Cimaglia: vogliate; Viviani prontuario: no imperative), so P5 does not allow `defective`. The data form (vuoi, volete) is not an imperative, so the cells are emptied.

### 2.5 PRS.3PL and variant order (P4)

* **Apocopes.** Morph-it lists the apocopated 3PL first: `parlan | parlano` in 5,758 of 6,071 lemmas. The mapping decision is the full form: Treccani, L. Bafile, s.v. *troncamento* (2011, https://www.treccani.it/enciclopedia/troncamento_(Enciclopedia-dell'Italiano)/): truncation (*andaron tutti, venivan tutti*) happens inside an intonation group ("non appare ugualmente probabile in tutte le forme"), in the Centre-South it is rare and "percepito come caratteristica di un parlato ricercato". `variant_rule: {avoid: "n$"}` moves the -n forms last (stable sort). Same rule type on PRS.3SG: `avoid: "^vuol$"`. The wrong accent in *è|é* needed no rule.
* **Other first variants (fix round; M2, P4).** A scan of all cells with variants (after the apocope rule) found these problems, fixed through `classes/ita_variant_overrides.csv` (55 rows). Each was checked in Treccani's Vocabolario on line (https://www.treccani.it/vocabolario/<lemma>/, entry text opened raw). Quotations are from the entry headers.

| lemma and cells | first variant in data | grid form now | source and rule |
|---|---|---|---|
| soddisfare PST ×6 | soddisfato (typo) | soddisfatto | typo never first; participle of *soddisfare*, Vocabolario s.v. soddisfare (tr. e intr.): "il resto della coniug. segue fare" |
| disfare PST ×6 | disfato (typo) | disfatto | typo never first; Vocabolario s.v. disfare: "come fare" |
| dovere PRS.3PL | debbono | devono | Vocabolario s.v. dovere: "dèvono o dèbbono" (devono listed first) |
| sedere PRS.1SG, PRS.3PL | seggo, seggono | siedo, siedono | Vocabolario s.v. sedere: "siedo [letter. seggo ...] ... siedono [letter. seggono ...]" (literary last) |
| possedere, risedere, soprassedere PRS.1SG, PRS.3PL | posseggo etc. | possiedo, possiedono; risiedo, risiedono; soprassiedo, soprassiedono | Vocabolario s.v. possedere ("possièdo [letter. possèggo]"); risedere and soprassedere "coniug. come sedere" |
| scompartire PRS.3PL | scompartano (an -are form) | scompartiscono | Vocabolario s.v. scompartire: "io scompartisco o scomparto, tu scompartisci o scomparti" (-isc- listed first) |
| scompartire PRS.2SG, IMP.2SG | scomparti | scompartisci | same entry |
| putire PRS.2SG, IMP.2SG, PRS.3SG | puti, pute | putisci, putisce | Vocabolario s.v. putire: "io putisco o puto, tu putisci o puti" |
| incartapecorire PRS.2SG, IMP.2SG | incartapecori | incartapecorisci | Vocabolario s.v. incartapecorire lists only the -isc- forms |
| ripartire PRS.2SG, PRS.3SG | riparti, riparte | ripartisci, ripartisce | Vocabolario has two verbs: ripartire¹ tr. "io ripartisco" (divide) and ripartire² intr. "io riparto" (leave again). The frame has the object "it" (D29), so the transitive verb with -isc- is taken; PRS.1SG/3PL already began with ripartisco/ripartiscono. Sense-dependent; medium |
| premorire PRS.1SG, PRS.3PL | premoro, premorono | premuoio, premuoiono | Vocabolario s.v. premorire: "coniug. come morire"; s.v. morire: "muoio ... muoiono [pop. mòio; ant. o dial. mòro ..., mòrono]" |
| perdere PST ×6; sperdere PST ×6 | perduto, sperduto | perso, sperso | Vocabolario s.v. perdere: "part. pass. pèrso o perduto" (perso listed first); sperdere "coniug. come perdere" |
| succedere PST.1SG-3SG | succeduto | successo | Vocabolario s.v. succedere: "part. pass. succèsso o, escluso il sign. 3, succeduto" (the PL cells are data gaps, 2.6) |
| nuocere PRS.1PL; PRS.2PL, IMP.2PL | nociamo; nocete | nuociamo; nuocete | Vocabolario s.v. nuocere: "nociamo, nocéte [oggi più spesso nuociamo, nuocéte]" (the form that is more frequent today comes first). PRS.1SG, PRS.3PL keep noccio, nocciono ("nòccio o nuòccio") and PST nociuto ("raro nuociuto") |
| percuotere, riscuotere, scuotere PRS.1PL | percotiamo etc. | percuotiamo, riscuotiamo, scuotiamo | Vocabolario s.v. percuotere: forms with the diphthong "oggi sono preferite"; scuotere "tendono a prevalere" |
| cuocere IMP.2PL | cocete | cuocete | Vocabolario s.v. cuocere: "cuocete o cocete" |

Left in Morph-it order, with the reason:

* *inferire* PST (inferito | inferto) and *disseppellire* PST (dissepolto | disseppellito): the Vocabolario lists both as standard for different senses (inferto = "arrecare, causare"; inferito = "trarre una conseguenza"; dissepolto or disseppellito without a ranking). The sense is not fixed by the frame, so the data order stays (unsourced order, P4 rule 3). Treccani lists inferto first for *inferire*; the data has inferito first; I did not override because both senses have a transitive use.
* *dare* (do|dò, dai|dài: the Vocabolario lists "do <dò> o dò", "dai o dài"; the data order agrees), *disfare* and *soddisfare* PRS (disfaccio|disfo etc.: the Vocabolario lists disfaccio first; agrees), *incartapecorire* PRS.1SG (isc first; agrees), *essere* è|é (accent typo second), *volere* vuole|vuol.
* *percuotere*, *riscuotere*, *scuotere* PRS.2PL and IMP.2PL: the data has only *percotete / riscotete / scotete* plus, as a second "variant", the 1PL form (percuotiamo, riscuotiamo, scuotiamo), which is mislabelled. The correct non-diphthong form is kept; the second variant is a data error (listed in section 4).

### 2.6 Plural participle mislabelled (M1)

`V.PTCP;PST;PL;MASC` holds a singular form in five lemmas, found by script (a plural form that is also in the singular cell, or that does not end in -i/-e): *succedere* (succeduto, successo), *trasparire* (trasparito), *nuocere* (nociuto, nuociuto), *perdere* (perduto, perso), *sperdere* (sperduto, sperso). Only the first two are essere verbs, whose PL cells use the plural participle, so `data_gaps` ("plural participle mislabelled in Morph-it") empties PST.1PL/2PL/3PL for *succedere* and *trasparire*. The other three are avere verbs: their PL cells take the singular participle (valid), so nothing is lost there. No other essere lemma has this defect.

## 3. Where the rules broke or were unclear

* **R4 class-specific identity** (IMP.2SG): needs a per-class class id; resolved by D26 (`emp_class`), `distinct_class` stays at language level.
* **R1 lexical classes by transitivity:** the frame's "(it)" is conditional on the verb; D29 states it. The aux_both problem is handled by D27 (2.3).
* **D1 vs R2:** the form of the agreeing participle comes from the lexical verb's class, while the auxiliary distinction is carried by `aux`. Works with `by_class` overrides.
* **P2 (class_exceptions):** the four IMP.2PL exceptions fit in the list (at most 10).
* **P4 (variant order):** sources often list several standard forms without ranking them (fai vs fa', inferito vs inferto); rule 3 applies and is flagged.
* **P5:** `defective` entries now all carry a reference; `volere` stays `data_gap` because the sources disagree.
* **Tool:** no problem found in the fix round.

## 4. Data problems

* PRS.3PL: apocope listed first in Morph-it (fixed by `variant_rule`). `é` after `è`, `vuol` before `vuole` (fixed by order rules).
* **Wrong first variants:** typos (soddisfato, disfato), archaic or literary first forms (debbono, seggo, posseggo, premoro, perduto) and -are-type or short forms in -ire verbs (scompartano, puti): corrected through the override CSV (2.5).
* **Plural participle tagged as singular:** 5 lemmas (2.6).
* **Wrong second variants:** *percuotere, riscuotere, scuotere* PRS.2PL and IMP.2PL carry the 1PL form (percuotiamo ...) as a variant. *ridire* IMP.2SG has *ridici*. *incartapecorire* PRS.3PL has *incartapecorono*. These are never first, so the grid forms are right.
* **Inherently pronominal lemmas:** 110 lemmas excluded (3, section 8).
* **Pseudo-verbs:** Morph-it generates verb paradigms for lemmas that Wiktionary has no verb entry for, including adjectives in *-are* (*interstellare, interdisciplinare, impopolare, irregolare, volgare, solare, lunare*) and many rare or coined verbs. 227 excluded (8, item 4). This also removes some genuine rare verbs (*fascinare, balbutire*) and many *auto-* coinages that Wiktionary lacks; the exclusion is by rule, not by hand.
* **Missing imperatives:** the pronominal imperative (accorgiti) is clitic-suffixed and was dropped by `prepare`; 90 lemmas lacked the imperative. After the exclusions 7 lemmas (IMP.2SG) and 5 (IMP.2PL) keep a `data_gap` there; the rest were pronominal or pseudo-verbs.
* **Missing participle:** 6 lemmas (*esimere, concernere, dirimere, incombere, procombere, soccombere*) are now `defective` for all PST cells with the Vocabolario as the source; *molcere* is `defective` for every cell except PRS.3SG (Vocabolario: "usato solo nella 3 a pers. sing. dell'indic. pres."). *delinquere* and *vigere* lack the participle in the data; the Vocabolario gives *delinquito* as "ant. e raro" (delinquere) and says nothing on the participle of *vigere*, so they stay `data_gap`.
* **Lexicon-wide duplicates** dropped by Morph-it (homograph noun forms): *cerchiare* lacks 2SG (*cerchi*), 1PL, 2PL; *sonare, spengere* (variant lemmas) have gaps. *consumere, licere, plaudere, recere* lack many forms.
* **Pronominal lemmas ending in -rsi:** *infiltrarsi* is a Morph-it lemma that is itself a reflexive infinitive; excluded with the pronominal list.
* **Homograph participles:** *stato* for both *essere* and *stare* (correct data; two lexemes).
* **Missing frequent verbs:** a known limitation (D24). For Italian, 48 of 48 checked frequent verbs are present (essere, avere, fare, dire, andare, potere, volere, sapere, stare, dare, vedere, venire all checked).
* **Participle forms of *aux_both* intransitives** (arrivare, partire, tornare, piacere ...) now carry *ho* as the auxiliary after D27 (2.3).

## 5. Check results (copied from `mappings/ita_check.md`, after the fix round)

* Errors 0, warnings 0.
* Lemmas in data 6,077; excluded by list 337 (110 pronominal + 227 pseudo-verbs); in scope 5,740; complete grids 5,715.
* Empty cells: data_gap 82; defective 53 (potere/dovere IMP ×2 each = 4; esimere, concernere, dirimere, incombere, procombere, soccombere PST ×6 each = 36; molcere 13). By cell, data_gap: PRS.1SG 9, PRS.2SG 6, PRS.3SG 2, PRS.1PL 8, PRS.2PL 4, PRS.3PL 5, PST ×6: 5 (SG) and 7 (PL), IMP.2SG 7, IMP.2PL 5.
* Classes 13; formally distinct cells 12; rule counts: source 8, main_verb 6; aux_carried cells 6.
* Lexical classes in PST.1PL: aux_essere 152, default 5,588.
* `class_exceptions`: avere, essere, sapere, volere.
* Pair identity: PRS.3SG~IMP.2SG 0.791, PRS.2SG~IMP.2SG 0.205, PRS.2PL~IMP.2PL 1.000; PST SG cells 1.000; PST SG vs PL 0.974; PST PL cells 1.000.
* **Empirical classes (D26):** `emp_mean_classes` = **12.00** (distribution on the 5,715 complete grids: 11 classes 1, 12 classes 5,688, 13 classes 23, 14 classes 3). The language-level count is 13 classes / 12 formally distinct cells (the grammar-level structure). The earlier "mean distinct forms" (now replaced as the reported number) was 7.03; it treated the six participle cells of the perfect as one form, while the empirical class counts them as six (D1).
* Share of lemmas in which a cell's form is unique among the 14: PRS.1SG 1.000, PRS.2SG 0.797, PRS.3SG 0.208, PRS.1PL 1.000, PRS.2PL 0.001, PRS.3PL 1.000, PST ×6 0.000, IMP.2SG 0.005, IMP.2PL 0.001.
* Filled grid inspected after the fix round: 24 frequent or irregular lemmas (essere, avere, dovere, potere, volere, sedere, possedere, perdere, nuocere, scuotere, cuocere, soddisfare, disfare, succedere, trasparire, scompartire, putire, premorire, esimere, molcere, finire, cominciare, arrivare, piacere) and every cell with variants (list in 2.5).

## 6. Sources seen (all re-opened as raw text in the fix round)

* https://accademiadellacrusca.it/it/consulenza/sulluso-del-passato-remoto/126 : Mortara Garavelli and Nencioni (2.1).
* https://accademiadellacrusca.it/it/consulenza/ancora-sulluso-del-passato-remoto/241 : Mauroni, 18 September 2009 (2.1).
* https://www.treccani.it/enciclopedia/passato-prossimo_(Enciclopedia-dell'Italiano)/ (E. Wiberg, 2011): aoristic prossimo, regional variation.
* https://www.treccani.it/enciclopedia/passato-remoto_(Enciclopedia-dell'Italiano)/ (E. Wiberg, 2011): "Differenze regionali", personal vs impersonal narration (2.1, M12a).
* https://www.treccani.it/enciclopedia/accordo_(Enciclopedia-dell'Italiano)/ (L. Gaeta, 2010): agreement with essere and avere (2.2).
* https://www.treccani.it/enciclopedia/participio_(Enciclopedia-dell'Italiano)/ (V. Egerland, 2011): "siamo tornati" (2.2).
* https://www.treccani.it/enciclopedia/oggetto_(Enciclopedia-dell'Italiano)/ (E. De Roberto, 2011): 3.1.3 "Accordo del participio" (2.2).
* https://www.treccani.it/enciclopedia/verbi-ausiliari_(Enciclopedia-dell'Italiano)/ (R. Cimaglia, 2010): "Distribuzione nell'uso degli ausiliari" (2.3).
* https://www.treccani.it/enciclopedia/transitivi-e-intransitivi-verbi_(Enciclopedia-dell'Italiano)/ (M. Cennamo, 2011): biargomentali take avere (2.3).
* https://www.treccani.it/enciclopedia/imperativo_(Enciclopedia-dell'Italiano)/ (R. Cimaglia, 2010) (2.4).
* https://www.treccani.it/enciclopedia/imperativo_(La-grammatica-italiana)/ (2012) and https://www.treccani.it/enciclopedia/forme-di-prontuario-imperativo_(Enciclopedia-dell'Italiano)/ (A. Viviani, 2011) (2.4).
* https://www.treccani.it/enciclopedia/troncamento_(Enciclopedia-dell'Italiano)/ (L. Bafile, 2011): truncated 3PL (2.5).
* https://www.treccani.it/enciclopedia/coniugazione-verbale_(Enciclopedia-dell'Italiano)/ (R. Cimaglia, 2010): conjugation classes, *-isc-* (PRS references).
* https://en.wikibooks.org/wiki/Italian/Grammar/Verbs (Wikibooks contributors; last edited 28 August 2026): present indicative endings table. Confirmation only; the mapping does not rest on it.
* Treccani, Vocabolario on line, entries opened raw (the URL pattern is https://www.treccani.it/vocabolario/<entry>/, with the entry number where the homographs have one): sedere2, possedere, soprassedere, risedere, dovere2, potere2, nuocere, succedere, soddisfare, disfare, scompartire, perdere, sperdere, putire, incartapecorire, ripartire1, ripartire2, inferire, disseppellire, percuotere, scuotere, riscuotere, cuocere, morire, premorire, dare2, dire, fare2, volere2, esimere, concernere, dirimere, incombere, procombere, soccombere, delinquere, molcere.
* Not readable: Maiden & Robustelli 2007 (Grambank source). The Treccani vocabolario/stare2 page is empty.
* Data used for decisions: Wiktionary (kaikki.org) dump in `external/kaikki-ita-verbs` (read with a script, not copied) for the exclusion lists and the auxiliary classes; Morph-it long table; `external/unimorph-ita` lemma list as a second dictionary for the pseudo-verb check (none of the 227 excluded lemmas is in it).

### Unsourced decisions

* **Masculine "it" object (invariant participle):** the frame does not give the gender of "it"; assumption.
* **Default avere for aux_both and the 28 lemmas without auxiliary information (D27):** the owner decision rests on avere being grammatical for all of them, not on a per-verb source.
* **Variant order:** *inferito|inferto*, *dissepolto|disseppellito*, *da'|dai*, *fa'|fai*, *disfa|disfa'* (rule 3: sources list both; flagged). The sense-dependent choice for *ripartire* (transitive, -isc-) follows D29 and is a judgement.
* **Exclusion rules** (3 and 8) are mechanical; they depend on the Wiktionary dump and are not a grammar statement.

## 7. Effort

First round: about 55 tool calls (9 web searches, about 24 page fetches). Fix round: see section 8.

## 8. Fix round 2026-10-14

Instructions: `FIXROUND.md` (ita), REVIEW.md §2, §3, §6, DECISIONS D23–D32. Status set to `reviewed`, `reviewed_by: "Opus reviewer + fix round 2026-10-14"`.

1. **M1, mislabelled plural participles. Done.** A script over all lemmas found `V.PTCP;PST;PL;MASC` holding a singular form in *succedere, trasparire, nuocere, perdere, sperdere*. Only *succedere* and *trasparire* are essere verbs (they use the PL cell). Their PST.1PL/2PL/3PL are `data_gaps` with the reason "plural participle mislabelled in Morph-it" (2.6).
2. **M2, variant overrides. Done.** `classes/ita_variant_overrides.csv` (55 rows) covers the typos in *soddisfare* and *disfare*, *dovere* PRS.3PL (*devono*), *sedere* (*siedo, siedono*), *scompartire* PRS.3PL (*scompartiscono*), and the checks asked for: *succedere* (*successo*), *possedere, soprassedere, risedere* (all -siedo/-siedono). The general P4 scan found more (perdere, sperdere, premorire, putire, incartapecorire, ripartire, nuocere, percuotere, riscuotere, scuotere, cuocere); each has a Vocabolario source (2.5). The remaining ranked-or-unranked doublets are listed with reasons.
3. **M3, inherently pronominal verbs. Done.** `classes/ita_pronominal_excluded.csv` has 110 lemmas, listed in `exclude_lemma_files`:
   * 86 lemmas for which the kaikki dump has no verb entry of any kind (neither lemma, nor alternative form, nor inflection) but has the lemma entry *L*-si (rule: no entry for *L*, a lemma entry for *L* minus final *e* plus *si*). Four lemmas that look like this (*adoprare, cagare, comperare, spiacere*) have an alternative-form verb entry and are kept. The review counted 88; the difference is in how the four are treated.
   * 1 Morph-it lemma that is itself a reflexive infinitive (*infiltrarsi*).
   * 23 lemmas replacing the earlier "30 flagged lemmas". The first agent's flag list was never saved, so I rebuilt it by a rule that includes the examples named in its notes (*accaldare, accanire, adirare, ostinare, vergognare*): Morph-it has no plain IMP.2SG for the lemma (only the clitic-suffixed imperative, which `prepare` dropped), and the dump has lemma entries for both *L* and *L*-si. The count is 23, not 30; I could not reproduce the other seven, and I did not widen the rule because the wider candidates (89 lemmas whose non-reflexive senses are all tagged archaic or rare, e.g. *celare, ergere, pacare*) are real verbs.
4. **M4, lemmas without auxiliary information (341). Done.**
   * `classes/ita_pseudoverbs_excluded.csv` has 227 lemmas. Rule: the Morph-it lemma has no entry of any kind in the kaikki verb dump, and it is not an -rsi lemma (those are in item 3). All 227 are also absent from the UniMorph Italian lemma list (a second dictionary), and 8 of them are adjectives in Morph-it itself (*impopolare, interdisciplinare, interstellare, irregolare, lunare, solare, volgare, contitolare*). The list is in `exclude_lemma_files`. Cost: some genuine rare verbs and *auto-* coinages are excluded; the exclusion is by rule.
   * 86 of the 341 are the pronominal lemmas of item 3; 28 lemmas without auxiliary information stay in scope (alternative-form entries such as *annunziare, obbedire, pronunziare*) and default to avere (D27).
   * *esimere* PST: `defective`, with the Vocabolario as the source ("difett. del part. pass. e quindi dei tempi comp."). Also made `defective` on the same kind of statement: *concernere, dirimere, incombere, procombere, soccombere* PST (P5), and *molcere* for all cells except PRS.3SG. *delinquere* and *vigere* stay `data_gap` (no statement that the participle does not exist).
5. **D27, aux_both. Done.** The class `aux_both_intransitive_first` is retired (removed from `lexical_classes` and from every `by_class`); `aux_essere` stays (152 lemmas in scope). "aux_both → essere" is recorded under `alternatives` of PST.1SG. Effect on the numbers: PST SG vs PL pair identity rises from 0.942 to 0.974.
6. **D25, IMP.2PL. Done.** IMP.2PL stays merged with PRS.2PL; `class_exceptions` lists *essere, avere, sapere, volere* for PRS.2PL and IMP.2PL, with Cimaglia's imperativo entry as the reference.
7. **D28 and M12. Done.**
   * The *passato prossimo* is kept (confidence medium).
   * M12(a): the notes (2.1) now report Treccani's northern *remoto* preference as real, with its register condition (personal narration), and set it against Crusca's northern *prossimo* statements.
   * M12(c): the PRS references now read "Wikibooks contributors, 2026" (the page was last edited 28 August 2026), each with a Treccani entry added (coniugazione verbale, 2010). The other Treccani references were checked against the page headers and given their real years (2010 or 2011), where before all said 2011.
8. **Optional auxiliaries. Done.** `aux:` on the six PST cells (ho, hai, ha, abbiamo, avete, hanno), and through `by_class: aux_essere` sono, sei, è, siamo, siete, sono.
9. **Further changes.**
   * P5: `volere` IMP stays `data_gap` (the sources disagree). All `defective` entries have a reference.
   * `variant_overrides` is declared in the YAML.
   * References re-opened as raw text; every `locus_verified: true` now rests on raw text of the page, not on a fetch summary.

**Check numbers after the fix round:** errors 0, warnings 0; 6,077 lemmas in data, 337 excluded, 5,740 in scope, 5,715 complete grids; empty cells data_gap 82, defective 53; 13 classes, 12 formally distinct cells; **`emp_mean_classes` 12.00**; class_exceptions 4 lemmas.

**What I could not do:** reproduce the exact earlier list of 30 pronominal lemmas (rebuilt as 23, item 3); read Maiden & Robustelli (Grambank page references stay unverified); rank *fai/fa'*-type doublets (no source ranks them).

**Effort (fix round):** about 80 tool calls: 2 web searches, 1 fetch-tool call, and about 45 raw page downloads through `curl` in a scratch folder outside the repository (Treccani Enciclopedia, Vocabolario, Crusca, Wikibooks). Main time sinks: finding that the Treccani Vocabolario is reachable only with the homograph number in the URL (sedere2, dovere2); rebuilding the pronominal lemma list; deciding the variant order for sense-dependent doublets.
