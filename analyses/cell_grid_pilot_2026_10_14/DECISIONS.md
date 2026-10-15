# Decision log: cell-grid pilot

Decisions are listed in order, with their date. They take precedence over the draft
proposals in [SPEC.md](SPEC.md). Decisions are made by the project owner; the findings
and recommendations come from the main agent.

## 2026-10-14: first review of the Phase 1 draft

| # | topic | decision | status |
|---|---|---|---|
| D1 | auxiliary-carried inflection (SPEC §5.1) | **Option C.** The grid form is the lexical verb alone (main-verb rule). Distinctions carried by an auxiliary still count in `distinct_class` and `is_formally_distinct`, with `aux_carried: true`. Rationale: auxiliaries are few and highly frequent, so they are not expected to cause production or perception problems | decided |
| D2 | word boundary (SPEC §5.4) | **Clitic auxiliaries are always separate words**, whatever the orthography. So Croatian *pisat ću* and Serbian *pisaću* both have the lexical-verb form *pisat*. The same applies to every language: a clitic auxiliary or tense particle is not part of the grid form | decided |
| D3 | future cells | **Deferred.** The working grid is **G14**: PRS × 6, PST × 6, IMP 2SG and 2PL. FUT.1SG and FUT.3SG stay in `grid.csv` with status `deferred`. If they are reinstated, this must happen before any cross-language LDL run, and all languages must be mapped for them | deferred |
| D4 | imperative | kept (IMP 2SG, 2PL) | decided |
| D5 | number of pilot languages | 5 | decided |
| D6 | auxiliary language | Eastern Armenian (MGN `hye-v`), not Hindi; Basque has no usable data | decided |
| D7 | Czech | out: the MGN `ces-v` columns are mislabelled | decided |
| D8 | R5, missing forms | **Verbs are not excluded for missing cells.** A missing grid cell stays empty (NA) and is never shown or tested. Each empty cell gets a reason code: `defective` (the grammar says the form does not exist) or `data_gap` (the form exists but the dataset lacks it). Counts are reported per language. Paradigm-wide syncretism rules (R4) may still fill a cell; ad hoc filling is not allowed. This departs from the current PCFP eligibility rule (complete paradigms only). The LDL phase must allow incomplete paradigms, which it can do because exposure is drawn from available cells | decided; whether `data_gap` verbs are kept too is open (Q3 below) |
| D9 | English source | keep MGN `data-custom/english-v` (CELEX, phonemic); UniMorph `eng` is not used | decided |
| D10 | Italian source | UniMorph `ita` (identical to MGN `ita-v`) is **rejected**: 16 of 40 very frequent verbs are missing (*essere, avere, potere, parlare, trovare, sentire, prendere, chiamare, mangiare, finire, capire, dormire, leggere, aprire, conoscere, vivere*). The Romance Verbal Inflection Dataset 2.0.4 is also **rejected**: Modern Standard Italian has 46 lexemes (median over 74 varieties: 24), too few for LDL. Replacement: see Q1 | open |

## Findings behind the open questions (2026-10-14)

**Why MGN uses non-UniMorph data for some languages.** Guzmán Naranjo (2024, JLM
12(2), §3.1, pp. 429–430) relies mostly on UniMorph (state of January 2021), with "only
minimal hand corrections". He adds curated datasets from earlier work, all converted to
phonemic form:
* English verbs: CELEX (Baayen et al. 1996);
* French verbs: Flexique (Bonami et al. 2014);
* Portuguese verbs: Beniamine et al. 2021;
* Russian nouns: Guzmán Naranjo 2020;
* Latin: Pellegrini & Passarotti 2018;
* Navajo; Yaitepec and Zenzontepec Chatino.

Footnote 21 acknowledges "some issues with the Unimorph datasets". The Hungarian,
Latvian and Arabic datasets are hand-corrected UniMorph. The paper gives no reason per
language. CELEX's English lemma list comes from a corpus, which agrees with the
observation that UniMorph `eng` (Wiktionary) contains rarely used verbs. Italian,
Finnish, Hindi, Serbo-Croatian and Armenian are plain UniMorph in MGN.

**Serbo-Croatian data (MGN `hbs-v`):**
* It mixes the ekavian and ijekavian standards: 349 doublets such as *htjeti/hteti* and
  *razumjeti/razumeti*.
* The future is always spelled with a space (Croatian convention). With D2 and D3 this
  no longer matters.
* The column labelled `PROG;PST;1;SG` holds imperfects for imperfective verbs (*govorah*,
  *rađah*) but aorists for perfective verbs (*rekoh*, *dadoh*, *uzeh*). These cells are
  outside G14.
* Aspect partners are separate lemmas with no aspect tag. The leakage grouping (DATA.md
  §6) does not join them. This is no different from Italian prefixed derivatives
  (*fare/rifare*), which are also separate lexemes.

**Prefix heuristic for aspect (tested on `hbs-v`).** 1,166 of 4,348 lemmas are "prefix +
another lemma in the data". A prefix is a poor perfective marker:
* secondary imperfectives are prefixed too (*zapisivati, iskazivati, obuzimati*);
* some prefixed verbs are imperfective without a suffix (*ovisiti* 'depend');
* frequent simplex verbs are perfective (*dati, baciti, reći, kupiti, staviti, pasti*);
* many pairs are suffixal (*baciti/bacati, odgovoriti/odgovarati, kupiti/kupovati*) or
  suppletive (*uzeti/uzimati, reći/govoriti*);
* prefixation often creates a new lexeme rather than an aspect partner (*staviti →
  nastaviti* 'continue', *pisati → prepisati* 'copy');
* 427 loans in *-irati/-isati/-ovati* are mostly biaspectual.

Russian and Bulgarian have the same architecture. A suffix-plus-prefix rule with a list
of simplex perfectives would be right for most lemmas but wrong on exactly the frequent
ones. If tags are needed, Wiktionary aspect categories (hbs, ru) or OpenCorpora (ru)
are the sources.

**Does aspect change inflection within G14?** No, not within one lemma:
* A verb's endings and alternations follow its conjugation class, not its aspect:
  *napisati* inflects like *pisati* (*napišem, napisao, napiši*).
* Partners formed by suffixation belong to different classes, so they have different
  suffixes and stem alternations in the same cell: *kupim/kupujem*, *kupio/kupovao*,
  *odgovorim/odgovaram*, *ostanem/ostajem*. But this is a difference between two
  lexemes, created by derivation.
* Aspect does condition cells outside G14:
  * Serbo-Croatian imperfect (mainly imperfectives), aorist (mainly perfectives), and
    the present adverbial participle (*pišući*, imperfectives only);
  * the Russian future: imperfective periphrastic *буду писать* vs perfective synthetic
    *напишу* (relevant if FUT is reinstated).

So with G14, the aspect options S-a and S-b give formally the same kind of grid. They
differ in which lemmas are included, and in whether the habitual-PRS and
completed-PST contexts are respected for every lemma.

## Open questions after 2026-10-14 (for the owner)

* **Q1. Italian source.** Proposed: Morph-it! (Zanchetta & Baroni 2005). It is built from
  the *la Repubblica* corpus, so its lemmas are verbs in use; about 28,500 lemmas, all
  parts of speech, version 0.11. It is orthographic, has participles with gender and
  number, and is free (CC BY-SA 2.0 / LGPL). It lacks auxiliary selection (*essere* vs
  *avere*), which D1 needs for the plural participle in PST. An auxiliary list from
  Wiktionary (kaikki.org extract) would fill this. Both need a download.
* **Q2. Slavic language.** The options are:
  * keep `hbs-v` (local; fix the standard: ijekavian, dropping 349 ekavian doublets);
  * switch to Russian (UniMorph or OpenCorpora; aspect tags available; the owner is a
    native speaker and can check the judgement calls; download needed).
* **Q3. R5.** Keep `data_gap` verbs as well as `defective` ones?
* **Q4. Aspect option** under G14: S-b (all lemmas, own forms, no tags; recommended now)
  or S-a (imperfective only; needs tags).
* **Q5. R1–R3 procedure:** see the proposal below.

## Proposed procedure for R1–R3 (Q5; superseded by D16)

Most of the 70 cell decisions (14 cells × 5 languages) are mechanical; only a few per
language need judgement. Language models are therefore used only where judgement is
needed.

1. **Candidates (script).** For each grid cell, list the dataset cells whose features
   contain the cell's features, ranked by the fewest extra features (DATA.md §8 rule).
   Flag a cell for judgement if it has no candidate, more than one candidate, or a
   multiword (periphrastic) form.
2. **Data checks (script).**
   * Each declared syncretism (R4) is tested for identical forms across all verbs. This
     shows whether it holds for the whole paradigm or only for one inflection class.
   * Each periphrastic cell is split into lexical verb and auxiliary. The script checks
     that the auxiliary set is small and the same in every verb, which is the D1
     assumption.
3. **Judgement cells (main agent).** Write each flagged cell with:
   * the competing forms;
   * the R3 reasoning;
   * a grammar citation. Grambank's page reference is reused where it exists. Otherwise
     the citation is at section level and marked `locus_verified: false`, because page
     numbers cannot be checked without the books.
4. **Blind second coding (subagent).** One Sonnet 5.5 subagent (high effort) per
   language maps only the flagged cells. It works from the grid, the rules and the
   dataset's cell list, and does not see the first mapping. A script compares the two
   mappings, and every disagreement goes to the review list. Both coders are language
   models, so this catches slips, not shared misconceptions.
5. **Owner review.** One short list per language, of the flagged cells only. The owner
   decides.

## 2026-10-14: second review

| # | topic | decision | status |
|---|---|---|---|
| D11 | R5, missing forms (Q3) | keep **both** `defective` and `data_gap` verbs; every empty cell carries its reason code | decided |
| D12 | Slavic language (Q2) | the owner has no preference. The main agent chose **Serbo-Croatian** (MGN `hbs-v`): it is local, needs no new adapter, and Grambank cites page-referenced sources for it. Ijekavian is kept: the loader drops the 349 ekavian doublets. Ekavian-only lemmas remain and are counted | decided |
| D13 | aspect (Q4) | **S-b**: all lemmas, each mapped with its own forms; no aspect tags | decided, with D14 |
| D14 | stem families (owner concern) | **Note for the LDL phase.** Derived verbs that share a stem (*сказать, рассказать, пересказать*; *pisati, napisati, prepisati*) make learning easier. Oversampling one family must be controlled (see "Stem families" below) | open; for the LDL phase |
| D15 | Italian source (Q1) | **Morph-it! 0.48** (6,077 verb lemmas, including *essere, avere, parlare*), with *essere/avere* classes from Wiktionary (kaikki.org extract). Both were fetched by `fetch_sources.sh`; hashes in `sources.sha256`. Converted by `gridtools.py prepare` | decided |
| D16 | R1–R3 procedure (Q5) | **One Sonnet 5.5 subagent (high effort) per language**, following [BRIEF.md](BRIEF.md). Each mapping must pass `gridtools.py check`, whose evidence is written to `mappings/<iso>_check.md`. Each agent writes its judgement calls to `mappings/<iso>_notes.md`, with references, `locus_verified` flags and confidence. Review: the main agent checks the check reports and notes; the owner, or a stronger agent, decides the judgement calls | decided; launch pending the owner's go. Steps 1–2 of the earlier proposal are now `gridtools.py inventory` and `check`. Step 3 moves from the main agent to the subagents. The blind second coding (step 4) is dropped for now; a stronger reviewer agent can be added for the judgement calls |

### Stem families (D14)

`stem_families.py` joins two lemmas when one is a listed prefix + the other. This is a
lower bound: suffixal partners and secondary imperfectives are not joined. Results
([stem_families_summary.csv](stem_families_summary.csv)):

| language | lemmas | in a family with ≥ 2 members | largest families |
|---|---:|---:|---|
| Serbo-Croatian (MGN) | 4,348 | 39% | *voditi* 17, *staviti* 14, *pisati* 12, *dati* 12 |
| Italian (MGN/UniMorph) | 10,009 | 31% | *porre* 20, *mettere* 18, *tendere* 12, *dire* 10 |

This is not a Slavic-only issue. Italian (*porre: comporre, deporre, imporre…*) is
almost as affected, and the current leakage grouping (DATA.md §6) joins neither.

A fix belongs to the sampling design, not to the grid:
* a per-language prefix list, plus suffix stripping, to build a family key;
* families used as leakage groups;
* a cap per family in each training sample.

Plain edit distance or LCS is not enough on its own. It joins unrelated short verbs
(*pisati/pitati*, distance 1), and it misses suffixal partners whose stems alternate
(*kupiti/kupovati*).

The effort is moderate: about a day, mostly building and checking prefix lists for each
language. Open design question for the owner: should a cap remove derivational families
that speakers really have? Without a cap, family structure counts as part of a
language's predictability.

## 2026-10-14: third review (before launching the mapping agents)

| # | topic | decision | status |
|---|---|---|---|
| D17 | stem families (D14 follow-up) | **The pilot and the next runs sample at random, with no family control.** Family control (limit oversampling, but not remove it entirely) is required for the **main run**. Owner idea for building families: align the forms of lemma pairs to detect shared stems and the affixes around them. This should generalise across languages better than hand-made prefix and suffix lists | open; required before the main run |
| D18 | sources for mappings | **No mapping from memory.** The mapping agents must find and open authoritative sources (open grammars, academy portals, previews, archive.org) and record the URL, the section and a short supporting quote. Cells without a seen source are flagged "unsourced" with confidence at most medium. See BRIEF.md "Sources" | decided |
| D19 | review | an **Opus reviewer subagent** reviews the five mappings and the judgement calls first; the owner reviews after it | decided |
| D20 | Italian edge cases | the Italian agent decides how to treat the 598 verbs taking both auxiliaries and the 341 without auxiliary information, and flags its choice | decided |
| D21 | Serbo-Croatian standard (follow-up to D12) | **Ijekavian is kept.** A Serbian-only subset would lose little: about 3,960 lemmas vs 3,999. The 349 ekavian doublets would be dropped instead, plus about 14–37 ijekavian-only lemmas such as *sjediti, uspjeti, šutjeti, utjecati*. But it would not be more coherent, because the data's vocabulary is Croatian-leaning: 45 *-izirati* verbs and no *-izovati*, ijekavian main entries, and the Croatian future spelling. Ekavian-only lemmas cannot be detected without a lexicon; they remain as a small mix | decided (main agent; owner may override) |
| D22 | representation | English stays phonemic (CELEX); the other languages stay orthographic. Rationale: English spelling is far from its phonology, while the other four have fairly transparent grapheme–phoneme mappings | decided |

## 2026-10-14: fourth review (after the Phase 2 review)

The owner decided D23 and D24 and left the other review decisions (REVIEW §3) to the main
agent ("make reasonable choices and report them back").

| # | topic | decision | by | status |
|---|---|---|---|---|
| D23 | evidentiality (REVIEW decision 5, P3) | **The PST frame adds "the speaker witnessed the event (direct evidence)"** (grid.csv, SPEC §3). Most languages do not mark it, so it changes nothing for them; in Armenian it selects the aorist | owner | decided |
| D24 | missing frequent verbs (REVIEW decision 8) | **No frequency lists, no coverage gate and no re-sourcing.** Checking frequent-verb coverage for every language does not scale and would itself cut the number of usable languages. Each dataset is used as it is. That UniMorph-derived sources lack many frequent irregular verbs (fin, hye, hbs) is recorded as a known limitation: it biases their scores towards regularity. Finnish stays on MGN `fin-v`. The P3 hand check of frequent labile verbs is dropped for the same reason | owner | decided |
| D25 | merging despite a few irregular lemmas (REVIEW decision 1) | **Option (a), with rule P2.** Cells share a `distinct_class` when the grammar states the identity for every regular conjugation class. At most 10 named irregular lemmas may be listed under `class_exceptions:`, and the identity checks ignore them. English IMP joins PRS.NON3SG (*be* is the exception); Italian IMP.2PL stays merged with PRS.2PL (exceptions *essere, avere, sapere, volere*) | main agent | decided |
| D26 | syncretism in some conjugation classes only (REVIEW decision 2) | **Option (b), rule P1.** `gridtools.py check` writes a per-lemma `emp_class`: two cells share it when their grid forms are identical and their auxiliaries (D1) are the same. A cell may name its auxiliary in `aux:` (also inside `by_class`). Without it, an aux-carried cell counts as having its own auxiliary. `emp_mean_classes` replaces `empirical_mean_distinct_forms` as the reported distinctness, because the old count treated the six participle cells of the Italian perfect as one form. `distinct_class` stays the language-level class from the grammar | main agent | decided; implemented |
| D27 | Italian verbs taking either auxiliary (REVIEW decision 3, D20) | **Every `aux_both` verb is treated as an *avere* verb.** Option (c) of the review (hand-check of the ~150 most frequent) needs frequency data and is ruled out by D24. Option (b) is mechanical, and *avere* is grammatical for every verb that takes both auxiliaries. It also fits the frame, where a verb with a transitive use takes the object "it" (D29). The Wiktionary first-sense proxy (`ita_both_intransitive_first.csv`) is retired. *essere* for these verbs is kept as an alternative for sensitivity. Only the PL participle of the PST cells changes | main agent | decided |
| D28 | Italian PST (REVIEW decision 4) | **The *passato prossimo* stays.** The *passato remoto* is the planned sensitivity alternative. The notes must report Treccani's northern *remoto* preference correctly (M12a) | main agent | decided |
| D29 | rule wording P3–P5 (REVIEW §6) | **Adopted**, with these changes. P3: evidentiality as D23; transitivity becomes "a verb takes the object 'it' if it has a transitive use", with no hand check (D24). P4: variants are ranked by a cited standard form first, then archaic, literary, poetic or regional forms go last, then data order (flagged unsourced); a typo is never first. P5: a cell is `defective` only where a cited grammar says the form does not exist; every other empty cell is a `data_gap` (wrong, glued, truncated, mislabelled, or marked non-standard by a cited grammar). The checker now refuses `defective` entries without a reference. R8 drops "non-derived" (D13); it keeps excluding multiword lemmas and inherently pronominal or clitic lemmas, with the list built mechanically from a dictionary | main agent | decided |
| D30 | Armenian IMP.2PL (REVIEW decision 6) | **The data form is kept** (the short *-ե՛ք* form, which Sakayan calls common). The class structure is unchanged | main agent | decided |
| D31 | modal imperatives (REVIEW decision 7) | **`defective` only where a reference grammar says so (P5).** The hbs agent checks *morati, trebati, smjeti, htjeti*. Without a statement, the data form is kept. Finnish *voida*: the fin agent found that VISK §1590 says modal verbs lack an imperative, so its IMP cells are `defective` | main agent | decided |
| D32 | tool (P6) | **Implemented.** A cell's `transform` may be a list, applied in order. A new transform type `strip_marks {codepoints, protect}` removes combining marks (via NFD) but keeps protected letters such as *ć, ś* | main agent | decided; implemented |

The 12 must-fix errors M1–M12 ([REVIEW.md](REVIEW.md) §2) go to the mapping agents in
one fix round, together with D23–D32.

## 2026-10-15: before the first LDL run on the grids

| # | topic | decision | by | status |
|---|---|---|---|---|
| D33 | mapping status for the LDL pilot | **The mappings are used as they are (status `reviewed`, not `approved`) and not committed before the first LDL run.** The owner will review them after seeing the results, because the runs are cheap. This departs from docs/PROTOCOL.md (all values declared before any outer-test fit); the first grid-LDL results are therefore exploratory | owner | decided |
| D34 | how LDL reads the grids | **A new input adapter (`grid`) and a new config (`configs/pcfp_grid_v1.yaml`).** Existing configs are unchanged. The pipeline code gained three opt-in switches, all off by default: `task.eligibility.allow_multiword`, `task.citation_cell_rule: include_in_test`, and `data.registry_and_gelato: false` (the grid run reads no population or ancestry file) | owner | decided; implemented |
| D35 | auxiliary cells in LDL (replaces D1 for the LDL forms) | **Option B: the form is "auxiliary + lexical verb"** (*ho parlato*, *pisao sam*, *գրում եմ*). The order follows the context sentence and the data (Italian auxiliary first; Armenian and Serbo-Croatian after). Under option C (participle × 6), filling one auxiliary cell from another is a trivial copy, so periphrastic tenses would look maximally syncretic. `gridtools.py check` writes `form_b`/`variants_b` from each cell's `aux:` field. Check: the reconstructed forms equal the data's own multiword forms in 5,130/5,130 Armenian cells and 23,733/23,937 Serbo-Croatian cells (the rest are data forms without an auxiliary). Option C stays available (`form: c`); dropping the auxiliary cells ("gaps") is the planned sensitivity check | owner | decided; implemented |
| D36 | citation cell | **The citation form (NFIN) is a 15th cell like any other**: shown or hidden at random and tested when hidden. It is not needed for semantics: lexeme vectors come from the lemma id (`morph_ldl/ldl/semantics.py`). Source: the infinitive cell (ita, fin: `NFIN`; eng: phonemic `inf`), or the lemma label where the data have none (hye, hbs). Italian prefers the full infinitive over Morph-it's apocopated first variant | owner | decided; implemented |
| D37 | sample size | **As `pcfp_v2`: 100 core + 100 extra training verbs per fit, 5 folds × 5 draws.** The inventory is 650 verbs in every language: Armenian has 855 complete grids, and the 200 tuning verbs must lie outside the inventory. The random draws of 100 therefore come from 150 non-core verbs and overlap | owner (100 + 100), main agent (650) | decided |
| D38 | small languages (main run) | **Training size per fit is fixed for every language; folds and draws adapt.** Proposal: the number of folds is `min(5, (N − extras) / core size)`; draws may overlap; the uncertainty is taken from the bootstrap over verbs, not from the spread between draws (it is narrower for small languages only because their draws share verbs); languages below a declared minimum (e.g. N ≥ 1.5 × training size) are excluded, not fitted on fewer verbs | main agent | proposal; to decide before the main run |
| D39 | the morphological model (note for the future) | **Consider replacing LDL with an analogical rule learner.** On the same test items, a simple edit-rule baseline (9 s, no tuning) beats LDL by 18–64 points in every language (ldl/RESULTS.md). The direction: align form pairs per cell pair; infer rules with conditioning contexts (minimal generalisation); test each rule on the training verbs (reliability with a small-sample confidence, as in Albright & Hayes 2002/2003); combine the proposals of all shown cells and output probabilities, not only the top form. Related work: Qumin and implicative entropy (Beniamine; Bonami & Beniamine; Ackerman & Malouf). To decide: the role of LDL (drop, secondary, or "cognitive model" next to a predictability measure); freezing the model on the five pilot languages before any new language; bias towards suffixing morphology. No detailed plan yet | owner | open; note only |

## Where notes are kept

Every decision and every note the owner asks to keep goes into this file. Each entry is
dated and numbered (D…), and open items are marked "open". Findings that support a
decision are kept under the date of the decision.
