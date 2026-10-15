# Review of the five G14 mappings (Opus reviewer, 2026-10-14)

Scope: `mappings/{ita,fin,eng,hye,hbs}.yaml`, notes, check reports, class lists, and the filled grids. The grid
`mapping_version` hashes match the current YAML files, so the grids are up to date. I opened 18 cited sources again and
read their raw text (HTML, or text taken from the PDF streams), not a fetch summary. No mapping, notes, tool or
decision file was changed. No ancestry or admixture data was read.

## 1. Verdicts

| lang | verdict | in one line |
|---|---|---|
| ita | **accept with fixes** | The construction choices are sound. Some forms are wrong (participles, first variants), inherently pronominal verbs and pseudo-verbs are still in scope (against R8), and the auxiliary choice for labile verbs is arbitrary |
| fin | **accept with fixes** | The mapping is correct and mechanical. 20 glued or truncated cells must become `data_gap`. The sources are learner-grade (VISK was blocked), and 18 of 51 frequent verbs are missing |
| eng | **accept with fixes** | The forms are correct. Three small fixes (*beware*, *would*, *grit*). The IMP class is inconsistent with PRS.1SG and with Italian (owner decision 1) |
| hye | **accept** | The mapping is correct and well sourced (Sakayan 2007 checked again). It has a data problem (24 of 28 frequent verbs missing) and a frame gap (evidentiality) |
| hbs | **accept with fixes** | The mapping is correct. The forms of *reći* are archaic, 79 forms carry prosodic accents, and 2 forms are wrong |

## 2. Must-fix issues (12)

| # | file / cell | evidence | fix |
|---|---|---|---|
| M1 | ita, PST.1PL–3PL, *succedere*, *trasparire* (essere class) | In Morph-it `V.PTCP;PST;PL;MASC` holds the singular: *succeduto/successo*, *trasparito*. The grid gives PST.3PL *succeduto* | Set these PL cells to `data_gaps`, with the reason "plural participle mislabelled in Morph-it" |
| M2 | ita, first variants | *soddisfare* PST *soddisfato* and *disfare* PST *disfato* are typos; the correct forms *soddisfatto* and *disfatto* are the second variants. *dovere* PRS.3PL gives *debbono* (archaic) before *devono*. *sedere* PRS.1SG/3PL gives *seggo/seggono* before *siedo/siedono*. *scompartire* PRS.3PL gives *scompartano* (a subjunctive) | Add a `variant_overrides` CSV for these rows (the tool supports it). Also check the overrides for *succedere* (*successo*) and *possedere*/*soprassedere*/*risedere* |
| M3 | ita, scope (R8) | R8 excludes "clitic or pronominal derived paradigms". 88 Morph-it lemmas exist in Wiktionary only as *-rsi* verbs (*accorgere, pentire, vergognare, inginocchiare, impadronire, autodefinire …*). The 30 lemmas the agent itself flags are added to these. They are mapped as avere verbs with an invariant participle (*accorto* ×6), which is wrong: the language says *ci siamo accorti*. They also cause most of the 91 IMP gaps | Build `classes/ita_pronominal_excluded.csv` from the kaikki entries (lemma attested only as *-rsi*) and put it in `exclude_lemma_files`. Report the count |
| M4 | ita, scope | Morph-it generates verb paradigms for adjectives in *-are*: *interstellare, antitubercolare, interdisciplinare, multilivellare, irregolare, impopolare, papillare, volgare* (e.g. *interstellai*). 313 of the 341 lemmas without auxiliary information have no Wiktionary verb entry at all. *esimere* has an invented participle *esimuto* | Hand-check the 313 lemmas and exclude the pseudo-verbs. Make *esimere* PST `defective` if a source is found, otherwise `data_gap` |
| M5 | fin, PST cells of 14 lemmas; 6 single cells | Glued doublets: *lähteä* PST.1SG *lähdinläksin* (all 6 PST cells), and *entää, hyytää, häätää, joutaa, kiitää, kyntää, liitää, noutaa, sortaa, soutaa, vuotaa, yltää, sivaltaa*. Truncated cells: *ammentaa* PRS.1PL *ammenn*, *lemmiskellä* PRS.1PL, *ummehtua* PRS.1PL, *sivaltaa* PST.3PL, *tukahtua* IMP.2PL, *takaisinmallintaa* IMP.2PL. The agent found these but did not act; `fin.yaml` has no `data_gaps` | Add `data_gaps` entries with these lists. The tool cannot split the doublets |
| M6 | eng, *beware* PRS.1SG/2SG/1PL/2PL/3PL | The agent's own source (Pullum slides, checked again) gives *beware* as lacking the plain present ("seems to lack the former"), not only the 3SG. The grid fills these five cells | Declare them `defective`, keep IMP, and note Green's hedge |
| M7 | eng, *would* | *would* is the preterite of *will* (agent J5), yet its 5 PRS cells hold *w ʊ d*, which is not a present form | Exclude *would* through a lemma list, or make all its cells `defective` |
| M8 | eng, *grit* PST.* | The grid gives *ɡ r ɪ t*; the agent itself says *gritted* is standard | Add an override to *ɡ r ɪ t ɪ d* |
| M9 | hbs, *reći* PRS.* and IMP.* | The data gives *reknem…reknu, rekni*. Gall's Imperativ handout (checked again) has *reći* → *reci*, with *reknu* as irregular/archaic; the standard present is *rečem*. *reći* is a top-10 verb | Make these 8 cells `data_gaps` with the reason "archaic form only". The alternative is that the owner accepts the forms |
| M10 | hbs, 79 forms in 31 lemmas | Prosodic accents: *dati* PRS.1SG *dȃm*, *piti* *pȉjēm…*, *htjeti* (6), *nemati* *nȇmām*. The same morph is then two different strings | Strip U+0300/0301/0304/030F/0311/0302 from Latin letters other than *ć, ś*. This needs a tool change (rule P6). Until then, make these cells `data_gaps` |
| M11 | hbs, *bdjeti* PST.1SG–3SG, *bosti* PST.1SG–3SG | The forms are *bdije* and *bȏ*; the l-participles should be *bdio* and *bo* | Make these cells `data_gaps` |
| M12 | notes: ita §2.1, eng §4, ita PRS references | (a) The Italian agent calls the Treccani *passato remoto* claim "garbled". It is not: the page really says *al Nord una netta preferenza per il passato remoto* for personal narration, so the conflicting evidence is real. (b) The English notes list only the modals as missing, but *need, seem, show, wait, die, raise* are also absent. (c) The ita PRS references give "Wikibooks … 2011", which is Treccani's year, not Wikibooks' | Correct the notes. The ita PST confidence stays medium |

## 3. Owner decisions needed

1. **Merging cells that differ only in a few irregular verbs (English IMP).** English merges PRS.1SG with PRS.NON3SG although *am* differs, and puts all PST cells in one class although *was/were* differ. But it keeps IMP in its own class, where only *be* differs. Italian merges IMP.2PL with PRS.2PL despite 4 exceptions (*essere, avere, sapere, volere*). Options: (a) cells are merged when only a closed list of irregular verbs differs; (b) any exception keeps the cells apart. **Recommendation: (a)**, with rule P2 below. Under (a) English IMP moves into PRS.NON3SG (4 → 3 classes) and Italian stays as it is. Under (b), English PRS.1SG and the PST person split would have to be separated too.
2. **Syncretisms that hold in only some conjugation classes.** Examples: Finnish PRS = PST in *-oida/-ida* verbs (15%); Finnish PRS.3SG = IMP.2SG (22%); Italian IMP.2SG = PRS.3SG in *-are* verbs and PRS.2SG elsewhere (99.5% together); hbs PRS.3SG = IMP.2SG and PRS.2PL = IMP.2PL in *-i-* verbs (35%). All agents left these cells distinct, which is consistent, but `n_formally_distinct_cells` then overstates distinctness (ita 12, but empirically 7.06 forms per grid). Options: (a) keep the language-level count only; (b) add a per-lemma class (rule P1) and use it for class-level LDL scoring. **Recommendation: (b).** It is mechanical, it handles D1, and it removes most of the judgement in `distinct_class`.
3. **Italian labile and `aux_both` verbs (D20).** The Wiktionary first-sense proxy splits natural pairs: *salire* → essere (*saliti*) but *scendere* → avere (*sceso*). It also gives *correre*/*vivere* essere, although Treccani gives avere without a goal. Options: (a) keep the proxy; (b) treat every `aux_both` verb as avere (transitive frame); (c) keep the proxy, but hand-check the ~150 most frequent `aux_both` verbs against Treccani or DOP. **Recommendation: (c).** The PL participle changes in 203 lemmas, and the frequent ones matter most.
4. **Italian PST: *passato prossimo* (chosen) or *remoto*.** The sources hold (Crusca; Treccani on the neostandard "aoristic" *prossimo*). But Treccani also reports a northern preference for the *remoto* in personal narration (M12). **Recommendation: keep the *prossimo*** (neostandard, D1), and run the *remoto* alternative as the planned sensitivity check. This choice moves Italian the most: the PST block has 6 aux-carried classes with one participle, against 6 synthetic forms with the *remoto*.
5. **Evidentiality default (new; Armenian).** Sakayan (checked again) calls the aorist "the witness tense": it is used only if the speaker witnessed the event or is certain of it. Otherwise the perfect is used. So PST.3SG/3PL "Yesterday he V-ed" is not fixed by the frame. **Recommendation:** add to SPEC §3 "the speaker witnessed the event (direct evidence)". The hye mapping then stands.
6. **Armenian IMP.2PL of *-ել* verbs.** The data gives the infinitive-stem form (*աբսորբե՛ք*); Sakayan's regular form is the aorist-stem form (*-եցե՛ք*), and Sakayan's footnote calls the short form "common" in everyday speech (checked again). **Recommendation: keep the data form** (R7, 678 lemmas). It does not change the class structure.
7. **Modal and stative imperatives.** These are not treated the same way:
   * Italian *potere/dovere* are `defective` (Treccani);
   * English modals are `defective`;
   * hbs *morati, trebati, smjeti* keep the data forms (*moraj, trebaj, smij*), and *htjeti* is `data_gap`;
   * Finnish *voida* keeps *voi*.

   **Recommendation:** a cell is `defective` only where a reference grammar says so. The hbs agent should look this up for these modals (HŠG, Browne & Alt). If the grammar has no statement, keep the data form.
8. **Re-sourcing frequent verbs.** See §7. Options per language: accept the gaps; rebuild from a fuller source; or add a frequency-matched lemma sample. **Recommendation:**
   * Finnish: rebuild from the local `external/unimorph-fin`, which has *olla, tulla, mennä, antaa, puhua, nukkua*. MGN `fin-v` dropped them.
   * hye and hbs: ask whether a download of UniMorph `hye`/`hbs` is allowed.
   * Until then, report frequent-verb coverage next to each LDL score.

## 4. Cross-language consistency

| issue | ita | fin | eng | hye | hbs | bias / comment |
|---|---|---|---|---|---|---|
| PST construction | aux + participle (*prossimo*) | synthetic imperfekti | preterite | synthetic aorist | aux + l-participle (perfect) | Every choice follows R3(1) (the grammar for the context); ita and hbs override the "single word" preference of R3(2). Consistent |
| aux-carried blocks (D1) | PST ×6 | – | – | PRS ×6 | PST ×6 | Consistent (6 classes each). Under C, person carried by an auxiliary counts as distinct, but person carried by a pronoun (English) does not. This is intended, but report it |
| classes / formally distinct | 13 / 12 | 14 / 14 | 4 / 1 | 14 / 14 | 14 / 14 | — |
| empirical distinct forms per grid (mean) | 7.06 | 12.87 | 2.99 | 9.00 | 9.31 | The ranking differs from the class count (ita < hye/hbs < fin) |
| merge despite exceptions in a few irregular verbs | IMP.2PL merged | – | PRS.1SG and PST merged, IMP not | – | – | Inconsistent → decision 1 |
| syncretism in some conjugation classes only | IMP.2SG kept distinct | PRS/PST, IMP kept distinct | – | – | PRS/IMP kept distinct | Consistent, but not visible in the class count → decision 2 |
| defaults | masc.; *tu*; masc. clitic "it" (unsourced) | n/a | n/a | n/a; *դու* | masc.; *ti* | No default for evidentiality → decision 5 |
| `defective` | *potere, dovere* IMP | none (weather verbs not marked) | modals IMP; *must, would, ought* PST; *beware* | none | none | Modals → decision 7. *ought* PST is unsourced |
| `data_gap` use | 289 (pronominal IMP, *volere*) | 0 (glued forms not flagged → M5) | 7 | 0 | 704 (536 mislabelled IMP) | fin under-uses it |
| variant order | Morph-it order + 2 regex rules | none | per-lemma overrides (Wikipedia) | none | none | No common procedure → rule P4 |
| main source type | Treccani, Crusca (good) | learner pages, Wikipedia (VISK blocked) | Pullum slides + learner pages | Sakayan, Fairbanks (good) | HŠG, Browne & Alt (good) | fin and eng are weakest, but their cells are mechanical |

## 5. Source spot-check (raw text read again)

| lang | URL (short) | claim | result |
|---|---|---|---|
| ita | accademiadellacrusca.it …/sulluso-del-passato-remoto/126 | the choice "non dipende dalla distanza temporale"; *ieri … è morto il mio vicino*; *prossimo* in the north | **holds**. The *prossimo* is chosen when the event is felt as current; in Tuscany the alternation is still alive |
| ita | treccani …/imperativo_(Enciclopedia-dell'Italiano) | IMP.2PL "coincide sempre" with PRS.2PL; stative *dovere, potere* have no imperative | **holds** |
| ita | treccani …/transitivi-e-intransitivi-verbi | "i verbi biargomentali hanno sempre l'ausiliare avere" | **holds**. It does not support the first-sense proxy, as the agent admits |
| ita | treccani …/passato-prossimo; …/passato-remoto | *prossimo* aoristic in the neostandard and the north; the remoto passage was called "garbled" | prossimo **holds**; the "garbled" reading **does not hold** (M12) |
| fin | uusikielemme.fi …/relation-between-the-imperfect-perfect… | imperfect when the exact time is given (*Ostimme eilen auton*) | **holds**; learner page |
| fin | yle.fi/a/20-156316 | imperfekti for "what happened yesterday"; marker -i/-si | **holds**; learner page |
| fin | elon.io …/voida | *voida* PRS = PST | **holds**; commercial learner page |
| fin | uusikielemme.fi …/the-imperative | 2SG = 1SG minus *-n* | **holds** |
| eng | pullum.ppls.ed.ac.uk/grammar/verb_sli.pdf | *must* has no preterite; *beware* is defective; "IMPERATIVE Be kind" is the plain form | **holds**. It also shows *beware* lacks the plain present (M6) |
| eng | continuingstudies.uvic.ca …/upperf2 | simple past "usually used with definite times" | **holds**; learner page |
| eng | continuingstudies.uvic.ca …/imperat | the imperative is "the same for singular and plural subjects" | **holds** |
| hye | lib.armedu.am … Sakayan 2007 PDF | p. 99 *yesterday* goes with the aorist; p. 74 the aorist is "the only simple tense"; p. 127 the footnote on the IMP.PL | **holds**. It also gives the aorist as the "witness tense" (decision 5) |
| hye | spw.uni-goettingen.de …/ARM-MOR-001.pdf | Imp. 2PL *ləs-e-tsʰ-ekʰ*; EA present = converb + auxiliary | **holds** |
| hye | wals.info/valuesets/39A-arm | no inclusive/exclusive distinction | **holds** |
| hbs | gramatika.hr/pravilo/perfekt/50 | "opće prošlo glagolsko vrijeme"; *je* dropped with reflexive verbs | **holds** |
| hbs | vdoc.pub … Browne & Alt 2004 | "can always safely use the perfect"; "all-purpose past tense" | **holds**. The URL is a third-party mirror; cite the SEELRC original |
| hbs | dlf.uzh.ch …/Imperativ.pdf (Gall) | -j vs -i; *reći* → *reci* (*reknu* irregular) | **holds**, and it supports M9 |

**Verified only through a summary.** The ita and eng agents (except for the Pullum PDF), and probably the fin and hbs
agents, read the web pages through a summarising fetch and still set `locus_verified: true`. Every page re-checked
above holds. But the *remoto* case shows that a summary can mislead the agent. **Wikipedia, Wiktionary and learner pages
as the main source:** fin (all cells; the PST call rests on uusikielemme and Yle); eng (variant order from the Wikipedia
list of irregular verbs; IMP formation); ita PRS (Wikibooks).

## 6. Rule-wording proposals

* **P1 (per-lemma class, R6).** "For each lemma, cells *a* and *b* are in the same empirical class if their grid forms
  are identical and their auxiliaries (D1) are identical or both absent. `gridtools check` writes `emp_class` per row.
  `distinct_class` remains the language-level class from the grammar." This needs an `aux:` field per cell, with
  `by_class` (e.g. ita PST.1SG `{default: ho, aux_essere: sono}`), so that the tool can compare auxiliaries and not only
  the `aux_carried` flags.
* **P2 (exceptions in irregular verbs, R4/R6).** "Two cells share a `distinct_class` if the grammar states the identity
  for every regular conjugation class. A closed list of irregular or suppletive lemmas that the grammar names, at most
  10 per language, does not break the identity. The list goes into `class_exceptions:`. If the identity holds in only
  some regular conjugation classes, the cells stay distinct and the identity is reported through P1."
* **P3 (frame, SPEC §3).** Add two lines:
  * evidentiality: the speaker witnessed the event;
  * transitivity: "a verb takes the object 'it' if its primary sense in the reference dictionary is transitive; for
    the 150 most frequent labile verbs the primary sense is checked by hand."
* **P4 (variant order, R3).** Rank a cell's variants in this order:
  * (i) the form a cited grammar or dictionary gives as standard;
  * (ii) otherwise, forms marked archaic, literary, poetic or regional go last (e.g. apocope, *debbono*, *seggo*);
  * (iii) otherwise, data order, flagged as unsourced.

  Typos are never kept first.
* **P5 (`defective` vs `data_gap`).** "`defective` only where a cited grammar says the form does not exist. Every other
  empty cell is `data_gap`. This includes a cell whose only data form is wrong, glued, truncated, mislabelled, or marked
  non-standard by a cited grammar." Also amend R8: drop "non-derived" (D13 and Italian practice keep derived lemmas). Keep
  the exclusion of multiword lemmas and of inherently pronominal or clitic lemmas, with a list built mechanically from a
  dictionary.
* **P6 (tool).** Allow a list of transforms per cell, and add a transform type
  `strip_marks {codepoints: [...]}` that works on NFD and recomposes to NFC. This is needed for the hbs accents (M10).
  It could also strip the Armenian ՛ if the owner wants (the mark is standard orthography and does not change any
  syncretism, so I leave it as it is).

## 7. Data problems

Frequent-verb coverage was checked against short lists I wrote (a common-sense top list per language, not corpus
counts).

| lang | lemmas in scope | frequent verbs present | missing (examples) | other problems |
|---|---:|---|---|---|
| ita | 6,077 | 48/48 | – | ~88–118 inherently pronominal verbs (M3); adjective pseudo-verbs (M4); PL participle tagged as SG (M1); typos *soddisfato*, *disfato*; archaic first variants (M2); the apocope is first in 95% of PRS.3PL (fixed) |
| fin | 7,597 | 33/51 | *olla, tulla, mennä, antaa, pitää, ottaa, jäädä, ajatella, puhua, kuulla, lukea, asua, nukkua, kertoa, kuolla, seistä, täytyä, maksaa* | 14 glued and 6 truncated forms (M5). Local `external/unimorph-fin` has the missing verbs (*olla* 374 rows) |
| eng | 5,940 | 98/109 (top ~100 + modals) | *can, will, could, should, might, need, seem, show, wait, die, raise* | *would* as a lemma (M7); variant order needed 240 overrides; `LEMMA` is orthographic |
| hye | 903 | 4/28 | *լինել, ունենալ, անել, ասել, գնալ, գալ, տալ, տեսնել, գրել, կարդալ, սիրել, խոսել …* | Most lemmas are rare or derived (455 have PASS cells); colloquial lemmas (*երթալ, էթալ, տենալ*); classical spellings (*խօսել, մօտենալ*); the IMP.2PL stem varies (decision 6) |
| hbs | 3,999 | 32/46 | *biti, imati, moći, ići, znati, vidjeti, stati, živjeti, jesti, voljeti, stajati, spavati, čuti, kupiti* | In 268 lemmas the IMP columns hold conditionals, and the COND columns hold aorists and imperfects; 17 lemmas have no forms (*-pasti* family, Cyrillic); accents (M10); *reći* archaic (M9); ≥ 9 ekavian-only lemmas. The ekavian-doublet filter was spot-checked: none of the 349 drops looks false |

The frequent and suppletive verbs are missing exactly where inflection is least predictable (fin, hye, hbs; English
modals). The LDL scores for these languages will therefore read as more regular than the languages are.
