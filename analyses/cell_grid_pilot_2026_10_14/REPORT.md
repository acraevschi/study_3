# Cell-grid pilot: Phase 2 report (2026-10-14)

Status: **5 languages mapped on grid G14, reviewed by an Opus agent, and fixed in one
round** (2026-10-14).
* Decisions: D23–D32 in [DECISIONS.md](DECISIONS.md).
* Fixes: [FIXROUND.md](FIXROUND.md).
* All five mappings pass `check` with 0 errors and 0 warnings. Their status is
  `reviewed`; the owner's own review is still to come.
* No LDL was fitted. No ancestry or admixture data was read.

## 1. What was done

1. **Shared tool** ([gridtools.py](gridtools.py)), written by the main agent:
   * `prepare`: converts Morph-it! and builds the Wiktionary *essere/avere* classes.
   * `inventory`: lists the source cells and the ranked candidates for each grid cell.
   * `check`: validates a mapping, applies it, and tests it against the data. It checks:
     * declared syncretism classes against form identity;
     * the auxiliary sets removed by transforms;
     * coverage;
     * the reference fields;
     * the empirical number of distinct forms per verb.
2. **Mapping.** One Sonnet 5.5 agent (high effort) per language followed [BRIEF.md](BRIEF.md).
   Each searched the web for sources (D18), wrote `mappings/<iso>.yaml` and
   `<iso>_notes.md`, and iterated until `check` gave 0 errors. After the first round
   the tool gained `data_gaps`, `variant_rule` and `variant_overrides`. Three agents then
   applied them to their own mappings.
3. **Review.** An Opus agent re-read 18 cited sources from their raw text (all hold up),
   spot-checked the grids, and compared the five mappings
   ([REVIEW.md](REVIEW.md)).

## 2. Results per language

Numbers are from `gridtools.py check` ([pilot_summary.csv](pilot_summary.csv)).
"Rule-filled" means `main_verb` cells, where the auxiliary was dropped under D1.
"Syncretic share" is `1 − mean distinct forms per complete grid / 14`.

After the fix round (§7). "Per-verb classes" is P1/D26: per verb, cells with the same
form and the same auxiliary count as one. It replaces "distinct forms", which counted the
six participle cells of an auxiliary-carried past as one form. Both are means over
complete grids, out of 14.

| | Italian | Finnish | English | E. Armenian | Serbo-Croatian |
|---|---|---|---|---|---|
| source | Morph-it! 0.48 | MGN `fin-v` | MGN CELEX (phon.) | MGN `hye-v` | MGN `hbs-v` (ijekavian) |
| verbs in scope (in data) | 5,740 (6,077) | 7,597 (7,724) | 5,939 (6,064) | 855 (903) | 3,991 (3,999) |
| complete grids | 5,715 | 7,570 | 5,933 | 855 | 3,702 |
| empty cells (data_gap / defective) | 82 / 53 | 54 / 66 | 7 / 32 | 0 / 0 | 685 / 0 |
| rules: source / syncretism / main_verb | 8 / 0 / 6 | 14 / 0 / 0 | 6 / 8 / 0 | 8 / 0 / 6 | 8 / 0 / 6 |
| declared classes / formally distinct cells | 13 / 12 | 14 / 14 | 3 / 1 | 14 / 14 | 14 / 14 |
| **per-verb classes (P1)** | **12.00** | **12.87** | **2.99** | **14.00** | **13.30** |
| distinct forms (old measure) | 7.03 | 12.87 | 2.99 | 9.00 | 9.30 |
| auxiliary-carried cells | 6 (PST) | 0 | 0 | 6 (PRS) | 6 (PST) |
| frequent verbs present (reviewer's list; D24: not acted on) | 48/48 | 33/51 | 98/109 | 4/28 | 32/46 |
| status | reviewed | reviewed | reviewed | reviewed | reviewed |

**Effort.**
* Agent tokens are cumulative per agent. Minutes are wall time, with all agents running
  in parallel.
* The main agent's own work (the tool, the brief, data preparation) is not counted
  below; it was about one working session.
* The owner's time was about four review rounds of the design.

| | ita | fin | eng | hye | hbs | review |
|---|---|---|---|---|---|---|
| tokens (k) | 202 | 159 | 206 | 171 | 164 | 289 |
| tool calls | 74 | 63 | 69 | 55 | 62 | 61 |
| minutes | 10.6 | 7.3 | 9.2 | 6.1 | 7.2 | 11.9 |

## 3. Judgement calls (details in each `<iso>_notes.md`)

* **PST.**
  * Italian: *passato prossimo*, medium confidence. Crusca and Treccani support it, but
    Treccani also reports a northern *remoto* preference. Owner decision 4.
  * Serbo-Croatian: perfect, not aorist.
  * Armenian: aorist, not perfect. This depends on a "witnessed" reading: owner
    decision 5.
  * Finnish: imperfekti.
  * English: preterite.
* **Auxiliary-carried cells (D1):**
  * Italian PST: participle; *essere* verbs take the plural participle in PL cells.
  * Armenian PRS: the *-ում* converb.
  * Serbo-Croatian PST: the l-participle.
* **Lexical classes.** Italian auxiliary selection for verbs that take both auxiliaries:
  the agent used a Wiktionary first-sense proxy, which the reviewer calls arbitrary
  (decision 3).
* **Syncretism classes.**
  * English keeps IMP apart from the plain present, and Italian merges IMP.2PL with
    PRS.2PL: an inconsistency (decision 1).
  * Syncretisms that hold only in some conjugation classes stay "distinct" in every
    language (decision 2):
    * Finnish PRS = PST in *-oida* verbs;
    * Italian IMP.2SG;
    * Serbo-Croatian IMP vs PRS in *-i-* verbs.
* **Variants and defectiveness:**
  * English needed 240 per-lemma variant overrides;
  * Italian needs overrides for typos and archaic first variants;
  * modal imperatives are treated inconsistently (decision 7).

## 4. Where the rules broke

1. **R6 cannot express syncretism that holds in only some conjugation classes.** The
   declared count overstates distinctness. Italian has 12 formally distinct cells
   declared, but 7.06 distinct forms per verb. The proposed fix is P1: an empirical
   per-lemma class written by the tool.
2. **Merging despite a few irregular exceptions** (English *be*, Italian *essere/avere*)
   was handled differently in each language. The proposed fix is P2: a closed exception
   list.
3. **The context frame was underspecified:**
   * evidentiality (the Armenian aorist is a witness tense);
   * transitivity for the object "it" (Italian labile verbs).

   The proposed fix is P3.
4. **Variant order** had no procedure. The fix is P4, ranking forms as standard,
   non-archaic, then data order.
5. **`defective` vs `data_gap`.** Wrong, glued or mislabelled data forms need
   `data_gap`. The fix is P5. R8 "non-derived" contradicts D13.
6. **Tool gaps:**
   * removing accents (Serbo-Croatian prosodic marks, fix P6);
   * splitting glued doublets (Finnish *lähdinläksin*).
7. **Sources.**
   * Most reference grammars are not open: Maiden & Robustelli, Huddleston & Pullum,
     Alexander, Dum-Tragut and Karlsson could not be read.
   * VISK blocks automated access.
   * Finnish and English therefore rest on learner and encyclopaedic pages. Their cells
     are mechanical, but this is weak for any judgement call.
   * Agents marked pages `locus_verified` after reading only a fetch summary. In one
     case (Treccani on the *remoto*) the summary misled the agent.

## 5. The largest finding: the data, not the mapping

**Three of the four UniMorph-derived datasets lack their most frequent verbs:**
* Finnish lacks *olla, tulla, mennä, antaa*;
* Armenian lacks *լինել, գնալ, ասել*: only 4 of 28 frequent verbs are present;
* Serbo-Croatian lacks *biti, moći, ići, znati*.

UniMorph `ita` had the same problem, which is why Italian was moved to Morph-it!.

The missing verbs are the irregular and suppletive ones, so LDL scores on these data
will read as **more regular than the languages are**. The owner accepted this as a known
limitation (D24): no frequency-based coverage check and no re-sourcing. This also affects the existing
pcfp_v2 outcomes, which use MGN `fin-v` and UniMorph `ita`. The local
`external/unimorph-fin` (current release) does contain the missing Finnish verbs:
*olla* has 160 rows. MGN's older Finnish release dropped them.

Other data errors found:
* mislabelled columns in Serbo-Croatian: imperatives holding conditionals in 268
  verbs;
* the same in Czech;
* glued and truncated forms in Finnish;
* typos and pseudo-verbs in Morph-it! (*-are* adjectives);
* inherently pronominal verbs mapped as *avere* verbs.

## 6. Recommendation: does the procedure generalise to about 70 languages?

**The mapping procedure generalises; the data does not, without curation.**

* **Mapping.** The rules (SPEC §4 + DECISIONS) handled five typologically different
  systems:
  * a synthetic language with no auxiliaries (Finnish);
  * a heavily syncretic one (English);
  * an auxiliary-carried present (Armenian);
  * an auxiliary-carried past (Italian, Serbo-Croatian).

  All five mappings pass the checker, and the reviewer found no wrong construction
  choice. The breaks are fixable by rule wording (P1–P6). After P1, most `distinct_class`
  judgement becomes mechanical.
* **Not tested:**
  * object agreement and polypersonal verbs (Basque, Georgian, Hungarian), where the
    object default (SPEC §5.3) will matter;
  * clusivity and dual systems.

  A second pilot with one such language is advisable before scaling up.
* **Cost per language** (mapping agent + share of review):
  * about 250k tokens and about 10 minutes of agent wall time; agents run in parallel;
  * plus about 15 minutes of owner review once P1–P6 are in place (now: about 1–2
    decisions per language).

  For 70 languages: about 17–18M tokens and about 15–20 hours of owner review, spread
  over batches.
* **The real cost is data curation.** Each language needs:
  * a check for mislabelled columns (the identity checks catch many of these);
  * lemma-scope lists.

  Missing frequent verbs are accepted as a known limitation (D24). A coverage check against
  frequency lists would not scale and would itself cut the number of usable languages.

  For a clean source this is about half a day of agent time plus a short owner check.
  For UniMorph-only languages it is more, and some will be unusable.

  The number of GeLaTo-linked languages with usable verb paradigms is itself not yet
  established. The 2026-10-08 estimate found few beyond MGN, so "about 70" may be an
  upper bound set by data, not by the procedure.
* **Before scaling up:**
  1. ~~Adopt P1–P6~~: done (D25–D32).
  2. ~~Decide the data policy for UniMorph-derived sources~~: done; data are used as they are (D24).
  3. Run the second pilot on an object-agreement language.

## 7. Fix round (2026-10-14)

The owner decided D23 (the speaker witnessed the event) and D24 (no frequency-based
coverage work). The main agent decided the other review questions (D25–D32), following
the reviewer except where D24 ruled a recommendation out (D27). The tool gained:
* P1: `emp_class` per row;
* P2: `class_exceptions`;
* P5: `defective` must cite a reference;
* P6: transform lists and `strip_marks`.

The original mapping agents could not be resumed, so five new Sonnet agents applied
[FIXROUND.md](FIXROUND.md).

| | changes | tokens (k) / tool calls / min |
|---|---|---|
| ita | M1 (2 lemmas `data_gap`); M2 (55 sourced variant overrides, P4 scan); M3: 110 inherently pronominal lemmas excluded (from kaikki); M4: 227 pseudo-verbs excluded, *esimere, concernere, molcere* `defective` (Treccani); D27: `aux_both` → avere (*salire* now *salito* ×6); D25: exceptions *essere, avere, sapere, volere*; notes corrected (M12) | 273 / 79 / 14.5 |
| fin | M5: 54 glued/truncated cells → `data_gap` (no others found by scan); VISK read via the Internet Archive, all VISK references now verified; weather verbs `defective` (VISK §457); *voida, taitaa* IMP `defective` (VISK §1590, D31 corrected) | 195 / 56 / 9.4 |
| eng | D25: IMP joins PRS.NON3SG (exception *be*), 3 classes; M6 *beware* `defective`; M7 *would* excluded; M8 *grit* → *gritted*; M12(b) notes | 139 / 39 / 4.0 |
| hye | D23: aorist now follows from the frame (Sakayan p. 76); D30 alternative recorded; 48 lemmas excluded (6 classical spelling, 42 colloquial/dialectal by Wiktionary labels, medium confidence) | 186 / 58 / 7.6 |
| hbs | M10: accents stripped on all cells; M9: *reći* and 6 derivatives PRS/IMP → `data_gap`; M11: *bdjeti* `data_gap` (*bosti* lifted: HJP gives *bȏ*); 8 Cyrillic duplicates excluded; D31: modal IMPs kept (no grammar statement found) | 155 / 48 / 5.2 |

Still weak:
* English *ought* and *beware* rest on Pullum's slides and Wikipedia (Huddleston &
  Pullum was not readable).
* The Armenian colloquial list rests on Wiktionary labels.
* Maiden & Robustelli and Dum-Tragut were not readable.

