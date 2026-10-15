# Brief for the per-language mapping agents (R1–R3 coding)

One agent maps one language. The agent receives this file plus its language code
(`ita`, `fin`, `eng`, `hye` or `hbs`).

## What you produce

1. `analyses/cell_grid_pilot_2026_10_14/mappings/<iso>.yaml`: the mapping file, in the
   format of `mapping_template.yaml`.
2. `analyses/cell_grid_pilot_2026_10_14/mappings/<iso>_notes.md`: your notes (see
   "Notes" below).
3. Optional: class, exclusion or defective lemma lists. Save them as
   `analyses/cell_grid_pilot_2026_10_14/mappings/classes/<iso>_<name>.csv`, with one
   column `lemma`. Each list must be explained in the notes.

The checker writes `mappings/<iso>_check.md` and `outputs/cell_grid_pilot/<iso>_grid.csv`.
Do not edit them by hand.

## Rules for you

* **Read first:** `README.md`, `grid.csv` (the cells with status `active`; FUT is
  deferred), `SPEC.md` §3 and §4, then `DECISIONS.md`. DECISIONS overrides SPEC. Then
  read `mapping_template.yaml` and your `inputs/<iso>_inventory.csv` and
  `inputs/<iso>_candidates.csv`. All paths are in this folder.
* **Files you may write:** only the files listed above. Do not change `gridtools.py`,
  the grid, the specification or any other repository file. If the tool seems wrong,
  say so in your notes and in your final message.
* **Off limits:**
  * the `.prompts/` folder;
  * any ancestry, admixture or GeLaTo population data;
  * `analyses/draft_admixture_model_2026_10_10/`.
* **No git commands. Do not save downloaded files into the repository.** You may read
  web pages and PDFs with your web tools (search, fetch). That is the main part of the
  job (see "Sources").
* **Paradigm level.** You map grid cells to source cells; you do not decide forms verb
  by verb. Lexical classes are allowed when the grammar requires them (R1).
* **Uncertainty.** Where you are not sure which form the language uses in a context,
  set `confidence: low` and explain. Do not guess silently.

## Sources (owner requirement, 2026-10-14: no mapping from memory)

Your decisions must rest on sources that you **find and open in this session**, not on
what you remember. Memory may guide the search but is never the reference.

* **Search first.** For every judgement call, and for the language's basic present, past
  and imperative paradigms, search for authoritative sources that you can read:
  * open-access reference grammars and grammar portals of national language
    institutes or academies;
  * university course grammars;
  * Language Science Press books;
  * scanned grammars on archive.org;
  * previews of the reference grammars (Google Books).

  The starting points are under each language below. Wikipedia and Wiktionary may point
  you to sources, or confirm a paradigm, but they are never the only reference for a
  judgement call.
* **Record what you saw.** Each reference you cite as seen must be in the notes, with:
  * the URL;
  * the section or page you saw;
  * a quotation of at most 15 words, or a short paraphrase, that supports the
    decision.

  Only then set `locus_verified: true`. One seen paradigm table can support all the
  mechanical cells it covers.
* **Unsourced decisions.** If you cannot find a readable source for a cell, set
  `locus_verified: false` and `confidence` at most `medium`. List the cell under a
  heading "Unsourced decisions" in the notes. Never fill a gap silently from memory,
  and never invent page numbers.
* **The data is evidence too.** For example, MGN multiword forms show the construction a
  dataset gives for a cell. Say so when you use it, but data does not replace a grammar
  for choosing among competing forms (R3).
* Grambank's `Source` field (`external/grambank/cldf/values.csv`) names the reference
  grammar and pages for the tense and person features. Use it to find the grammar.

## Procedure

1. For each of the 14 active cells, decide which construction the language uses in the
   context frame of `grid.csv`. This is R1, with competing forms decided by R3.
2. Classify the cell. It is one of:
   * a direct source cell (`source`);
   * the same form as another grid cell (`syncretism`, R4);
   * a periphrastic cell (`main_verb`, R2 with D1 and D2): the grid form is the lexical
     verb's form; auxiliaries and clitic auxiliaries are removed by a declared
     transform;
   * a cell dropped by the dataset as a lexicon-wide duplicate (`restored`).
3. Assign `distinct_class` ids (R6). Cells share an id when their grid forms are the
   same by grammar. Under D1, cells that differ only in their auxiliary get **different**
   ids, with `aux_carried: true`.
4. Fill `defaults` and `grambank_check` (copy the values and their `Source` from
   `external/grambank/cldf/values.csv` for your glottocode).
5. Run:
   ```
   .venv/bin/python analyses/cell_grid_pilot_2026_10_14/gridtools.py check <iso>
   ```
   Repeat until it reports 0 errors. For every warning, either fix the mapping or
   explain the warning in the notes. A typical example to explain is a syncretism that
   holds for one conjugation class only.
6. Look at the filled grid (`outputs/cell_grid_pilot/<iso>_grid.csv`):
   * about 20 random lemmas;
   * the 10–15 most frequent or most irregular verbs of the language that are in the
     data;
   * every row whose `variants` is non-empty for at least a few lemmas, to choose or
     confirm the variant order under R3.

   Fix the mapping where the forms are wrong.

## Notes (`<iso>_notes.md`), in this order

1. **Summary table:** one row per cell, with columns rule, source cell, class,
   aux_carried and confidence.
2. **Judgement calls:** for each, the options, your choice, the reason, the reference,
   your confidence, and the alternative recorded under `alternatives:`.
3. **Where the rules broke or were unclear** (SPEC §4, DECISIONS).
4. **Data problems** found: mislabelled cells, wrong forms, missing frequent verbs.
5. **Check results:** lemmas in scope, complete grids, empty cells by reason, number of
   classes and of formally distinct cells, rule counts. Copy these from the check
   report.
6. **Sources seen:** one line per source, with the URL and the sections used.
7. **Effort:** the number of tool calls and web searches, and the main time sinks.

## Final message (at most 200 words)

Give the check numbers (errors 0, warnings explained), the list of judgement calls, the
cells with `confidence: low`, the unsourced decisions, and anything blocking.

---

## Language notes

### ita: Italian

* **Data.** Morph-it! 0.48, converted to UniMorph-style labels by `gridtools.py prepare`:
  * `IND;PST;PFV;…` = *passato remoto*;
  * `IND;PST;IPFV;…` = *imperfetto*;
  * `V.PTCP;PST;SG;MASC` etc. = past participle, with gender and number.

  There are 6,077 lemmas. Clitic-suffixed forms were dropped. Variants keep Morph-it
  order.
* **Auxiliary classes** come from Wiktionary (`outputs/cell_grid_pilot/inputs/`):
  * `ita_aux_avere.csv`: 4,985 verbs;
  * `ita_aux_essere.csv`: 153 verbs;
  * `ita_aux_both.csv`: 598 verbs;
  * 341 lemmas have no information.
* **Expected judgement calls:**
  * The PST context: *passato prossimo* or *remoto* for "yesterday, once" in the
    standard language.
  * Under D1, the participle's agreement:
    * with *essere* it agrees with the subject (masculine default; plural persons
      take the plural);
    * with *avere* it is invariant, but check agreement with a preceding clitic
      object (*l'ho mangiato*) given the frame's object "it".
  * What to do with `aux_both` verbs and with lemmas without auxiliary information.
  * IMP.2SG: first-conjugation verbs use the PRS.3SG form (*parla*); the other
    conjugations use PRS.2SG (*vedi*). Irregular imperatives (*va'/vai, fa'/fai, di',
    sii, abbi*) need checking.
* **Sources to try:**
  * Treccani, *Enciclopedia dell'Italiano* (online; entries on the *passato prossimo*,
    *passato remoto*, the auxiliaries and the imperative);
  * Accademia della Crusca, *consulenza linguistica* pages;
  * Maiden & Robustelli 2007 (the Grambank source; previews).

### fin: Finnish

* **Data.** MGN `fin-v`, with labels such as `ACT;PRS;POS;IND;1;SG`.
* **Expected judgement calls:**
  * PST: *imperfekti* vs *perfekti* for "yesterday, once";
  * the standard-language 3PL;
  * IMP 2SG and 2PL.
* **Sources to try:**
  * Hakulinen et al. 2004, *Iso suomen kielioppi*, online as VISK
    (kaino.kotus.fi/visk): the sections on tenses and on the imperative;
  * Karlsson 2018 (previews).

### eng: English

* **Data.** MGN `data-custom/english-v` (CELEX), with labels `inf, pres1s, pres3s,
  presothers, past13, pastnot13, ppart, prespart`.
* **Forms are space-separated phoneme strings.** Never use token transforms on English.
* **Expected judgement calls:**
  * past13 vs pastnot13 (they differ only for *be*);
  * IMP = the bare form;
  * the person split of `presothers`.
* **Sources to try:**
  * Huddleston & Pullum 2002 (the Grambank source; previews);
  * Cambridge Dictionary's "Grammar" pages;
  * university grammar pages.

### hye: Eastern Armenian

* **Data.** MGN `hye-v`. For example, `1;SG;PRS;IND` is a converb + auxiliary
  (*աբսորբում եմ*).
* **Expected judgement calls:**
  * PRS is periphrastic: the main verb is the imperfective converb, and the auxiliary
    is removed by a transform;
  * PST: the synthetic aorist (`…;PST;IND`) vs the perfect (`…;PRS;PRF;IND`) for
    "yesterday, once";
  * IMP 2SG and 2PL. Keep the data's orthography, including the emphasis mark ՛.
* **Sources to try:**
  * Dum-Tragut 2009, *Armenian: Modern Eastern Armenian* (the Grambank source;
    previews);
  * Fairbanks & Stevick 1958, *Spoken East Armenian* (public domain, archive.org or
    ERIC);
  * other scanned grammars on archive.org.

### hbs: Serbo-Croatian

* **Data.** MGN `hbs-v`, ijekavian standard: the loader drops the 349 ekavian doublets.
  Some ekavian-only lemmas remain; count them and report.
* **Labels:**
  * `PRS;1;SG`;
  * `PST;1;SG` = perfect (*pisao sam*);
  * `PST;1;SG;LGSPEC2` = aorist (sparse);
  * the `PROG;PST` columns are inconsistent (imperfects or aorists): avoid them.
* **Expected judgement calls:**
  * PST: perfect vs aorist. Under D1 and D2 the grid form is the l-participle, with the
    clitic auxiliary removed: masculine singular for the SG persons, masculine plural
    for the PL persons;
  * reflexive *se* if it occurs in forms.
* **Aspect is S-b:** every lemma is mapped with its own forms. Say in the notes how this
  departs from R1 for perfective lemmas.
* **Sources to try:**
  * Browne & Alt 2004, *A Handbook of Bosnian, Serbian, and Croatian* (SEELRC, free
    online);
  * *Hrvatska školska gramatika* (gramatika.hr, Institut za hrvatski jezik);
  * Alexander 2006 (the Grambank source; previews);
  * Browne 1993.
* **Variety.** The data's vocabulary leans Croatian: 45 *-izirati* verbs, no *-izovati*,
  and the Croatian future spelling. Map the ijekavian standard (DECISIONS D21).
