# Fix round (2026-10-14): instructions for the mapping agents

One agent per language. Each agent fixes its own mapping after the Opus review
([REVIEW.md](REVIEW.md)) and the decisions D23–D32 ([DECISIONS.md](DECISIONS.md)).
The original mapping agents could not be resumed, so these are new agents. They start from
the existing `mappings/<iso>.yaml` and `<iso>_notes.md`.

## For every language

* **Read first:**
  * [BRIEF.md](BRIEF.md): its rules still hold;
  * your `mappings/<iso>.yaml`, `<iso>_notes.md` and `<iso>_check.md`;
  * REVIEW.md: §2 (the rows for your language), §3 and §6;
  * DECISIONS.md D23–D32. DECISIONS overrides REVIEW where they differ.
* **Files you may write:**
  * `mappings/<iso>.yaml`;
  * `mappings/<iso>_notes.md`;
  * `mappings/classes/<iso>_*.csv`.
* **Off limits:**
  * the `.prompts/` folder;
  * ancestry, admixture and GeLaTo population data;
  * `analyses/draft_admixture_model_2026_10_10/`;
  * git.
* **Sources.** Sources must be opened in this session. Record each one with:
  * the URL;
  * the section;
  * a quotation of at most 15 words.

  Never set `locus_verified: true` after seeing only a fetch summary of a page.
* **New tool features** (see `mapping_template.yaml`):
  * `transform` may be a list of transforms, applied in order;
  * a transform type `strip_marks`;
  * an optional `aux:` field per cell, also allowed inside `by_class`;
  * a top-level `class_exceptions:` list, with at most 10 lemmas (P2/D25);
  * the checker writes `emp_class` on every row and reports `emp_mean_classes` (P1/D26);
  * a `defective` entry without a reference is now an error (P5).
* **P4, variant order:**
  1. a form that a cited grammar gives as standard comes first;
  2. archaic, literary, poetic and regional forms go last;
  3. otherwise keep the data order, flagged as unsourced.

  A typo is never first.
* **P5.** A cell is `defective` only where a cited grammar says the form does not exist.
  Every other empty cell is a `data_gaps` entry. Re-check your existing entries against
  this rule.
* **Frame:**
  * PST adds "the speaker witnessed the event" (D23);
  * a verb with a transitive use takes the object "it" (D29).
* **Missing frequent verbs.** Take no action. Keep them described in the notes as a known
  limitation of the dataset (D24).
* **When done:**
  * Run `.venv/bin/python analyses/cell_grid_pilot_2026_10_14/gridtools.py check <iso>`
    from the repository root until it reports 0 errors, and explain every warning.
  * Add a section "Fix round 2026-10-14" to the notes. List each item and what you did,
    and update the check numbers.
  * Set `provenance.status: reviewed` and
    `reviewed_by: "Opus reviewer + fix round 2026-10-14"`.
* **Final message** (at most 150 words):
  * errors and warnings;
  * what changed;
  * `emp_mean_classes`;
  * anything you could not fix.

## ita: Italian

1. **M1: mislabelled plural participles.** *succedere* and *trasparire* are affected,
   and possibly other *essere*-class lemmas. Find them mechanically: lemmas whose
   `V.PTCP;PST;PL;MASC` holds a singular form. For these lemmas, make the PL PST cells
   `data_gaps` with the reason "plural participle mislabelled in Morph-it".
2. **M2: variant overrides.** Add rows to a `variant_overrides` CSV for:
   * the typos in *soddisfare* and *disfare*;
   * *dovere* PRS.3PL: *devono*;
   * *sedere*: *siedo*, *siedono*;
   * *scompartire* PRS.3PL.

   Check *succedere* (*successo*), *possedere*, *soprassedere* and *risedere*. Then apply
   P4 generally: scan the grid for other typos or archaic first variants.
3. **M3: inherently pronominal verbs.** Build `classes/ita_pronominal_excluded.csv`
   mechanically from the kaikki dump: the lemmas that Wiktionary attests only as *-rsi*
   verbs. The dump is
   `external/kaikki-ita-verbs/`; read it with a script and do not copy it. Add your
   earlier 30 flagged lemmas. Put the list in `exclude_lemma_files` and report the
   count.
4. **M4: lemmas without auxiliary information** (341).
   * Exclude the pseudo-verbs (adjectives in *-are*, lemmas with no Wiktionary verb
     entry) through a list in `exclude_lemma_files`. State the rule you used.
   * *esimere* PST: make it `defective` only if you cite a source; otherwise use
     `data_gaps`.
5. **D27: verbs taking either auxiliary.** Every `aux_both` verb is now an *avere* verb.
   * Retire `ita_both_intransitive_first.csv` as a class.
   * Keep `aux_essere` for the essere-only list.
   * Record "aux_both → essere" under `alternatives`.
6. **D25: IMP.2PL** stays merged with PRS.2PL. Add `class_exceptions` for *essere, avere,
   sapere, volere*, with a reference.
7. **D28 and M12: past tense and notes.**
   * Keep the *passato prossimo*.
   * M12(a): correct the notes. Treccani's northern *remoto* preference is real, not
     garbled. The PST confidence stays medium.
   * M12(c): fix the PRS references that give "Wikibooks … 2011".
8. **Optional: auxiliaries.** Add `aux:` to the PST cells:
   * *ho/hai/ha/abbiamo/avete/hanno*;
   * for the `aux_essere` class, through `by_class`: *sono/sei/è/siamo/siete/sono*.

## fin: Finnish

1. **M5: glued and truncated forms.**
   * Add `data_gaps` for the glued doublets, in the affected cells only:
     * *lähteä* PST ×6 (*lähdinläksin*);
     * *entää, hyytää, häätää, joutaa, kiitää, kyntää, liitää, noutaa, sortaa, soutaa,
       vuotaa, yltää, sivaltaa*.
   * Add `data_gaps` for the truncated cells:
     * *ammentaa* PRS.1PL;
     * *lemmiskellä* PRS.1PL;
     * *ummehtua* PRS.1PL;
     * *sivaltaa* PST.3PL;
     * *tukahtua* IMP.2PL;
     * *takaisinmallintaa* IMP.2PL.
   * Then scan the whole grid mechanically for more. Examples: a form with two personal
     endings, or a form shorter than the stem.
2. **D26: partial syncretisms.** The *-oida/-ida* syncretisms (PRS = PST, and PRS.3SG =
   IMP.2SG) now show in `emp_class`. Keep `distinct_class` as it is, and report
   `emp_mean_classes` against 14.
3. **P4.** Check the variant order in cells that have variants.
4. **P5/D31.** Weather verbs and *voida* IMP keep the data form, unless a grammar you can
   open says the form does not exist.
5. **VISK.** It was blocked last time; try once more with other URL forms or the Kotus
   PDF, for the tense and imperative sections. If it is still blocked, record this.

## eng: English

1. **D25: imperative class.** IMP.2SG and IMP.2PL join the PRS non-3SG class.
   * Add `class_exceptions` for *be*, covering PRS.1SG, the PST person split and IMP,
     with a reference.
   * Add any other lemma the grammar names, up to 10 in total.
   * The 10 IMP warnings should disappear. Explain any that remain.
2. **M6: *beware*.** PRS.1SG/2SG/1PL/2PL/3PL become `defective` (cite Pullum, and note
   Green's hedge). Keep IMP.
3. **M7: *would*.** Exclude it through `classes/eng_excluded.csv` in
   `exclude_lemma_files`. It is the preterite of *will*, not a lemma.
4. **M8: *grit*.** Override PST.* to `ɡ r ɪ t ɪ d`.
5. **M12(b): missing verbs.** Add *need, seem, show, wait, die, raise* to the notes as
   missing, as a known limitation (D24).
6. **P5: defective entries.** *ought* PST and the other `defective` entries need a cited
   source. Otherwise make them `data_gaps`.

## hye: Eastern Armenian

The review accepted this mapping, so this round is short.

1. **D23: evidentiality.** The frame now fixes "the speaker witnessed the event". Update
   the PST references and judgement-call notes: the aorist now follows from the frame
   (Sakayan's "witness tense").
2. **D30: IMP.2PL.** Keep the data form. Record Sakayan's *-եցե՛ք* form under
   `alternatives`.
3. **P4: variants and odd lemmas.**
   * Check the cells with variants.
   * Decide, with a source, whether to exclude the classical-orthography lemmas (e.g.
     *խօսել*) and the colloquial-only lemmas (e.g. *երթալ, էթալ, տենալ*) under R8
     (standard variety). If you exclude them, use a list and explain the rule.
4. **Optional: auxiliaries.** Add `aux:` (*եմ, ես, է, ենք, եք, են*) to the PRS cells.

## hbs: Serbo-Croatian

1. **M10/D32: accents.** Declare a `strip_marks` transform:
   * codepoints 0300, 0301, 0302, 0304, 030F, 0311;
   * protect *ć, ś, Ć, Ś*, and check whether other letters need protecting.

   Apply it to every cell. In PST, give it as a list with the auxiliary transform.
   Confirm that *dȃm* → *dam*, *pȉjēm* → *pijem*, and that *ć* is kept.
2. **M9: *reći*.** PRS.* and IMP.* become `data_gaps`, with the reason "archaic form only
   (*reknem/rekni*); standard *rečem/reci*". Cite Gall.
3. **M11: wrong l-participles.** *bdjeti* PST.1SG–3SG and *bosti* PST.1SG–3SG become
   `data_gaps`.
4. **D31: modal imperatives** (*morati, trebati, smjeti, htjeti*). Look them up in HŠG
   (gramatika.hr) and Browne & Alt. Mark an imperative `defective` only where a grammar
   you open says it does not exist; otherwise keep the data form.
5. **P4.** Check the variant order in cells that have variants.
6. **Optional: auxiliaries.** Add `aux:` (*sam, si, je, smo, ste, su*) to the PST cells.
