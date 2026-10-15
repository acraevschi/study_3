# Universal verb-cell grid: design and pilot (2026-10-14)

**Status (2026-10-14, end of day): Phase 2 mapped, reviewed and fixed. The five mappings are in `mappings/`, with status `reviewed` and 0 errors and 0 warnings each. Report: [REPORT.md](REPORT.md) (§7 covers the fix round). Opus review: [REVIEW.md](REVIEW.md). Fix instructions: [FIXROUND.md](FIXROUND.md). Decisions log: [DECISIONS.md](DECISIONS.md), with D23–D32 resolving the review. Open: the owner's review, and a second pilot on an object-agreement language.**

## Why this exists

The LDL outcome is not comparable across languages, because paradigm size differs. This
pilot defines one small, fixed grid of verb cells (meanings), the same for every
language. Each grid cell is defined by a context sentence, and every cell is filled in
every language. A cell whose meaning has no dedicated form takes the form of the main
verb in the construction that expresses the meaning. This equalises paradigm size and
the LDL meaning space, so that syncretism makes a language easier.

The full specification is in [SPEC.md](SPEC.md).

No ancestry or admixture data was read. No pipeline stage or config was changed.

## Files

| file | content |
|---|---|
| [SPEC.md](SPEC.md) | opinion on the design (§1), grid and frequency justification (§2), context conventions (§3), rules R1–R9 (§4), options for the open questions with pilot examples (§5), data availability and proposed languages (§6), filled-grid format (§7) |
| [grid.csv](grid.csv) | the 16 draft cells: UniMorph features and context frames |
| [mapping_template.yaml](mapping_template.yaml) | per-language mapping format |
| [REPORT.md](REPORT.md) | Phase 2 report: results per language, effort, rule breaks, data findings, recommendation for scaling up |
| [REVIEW.md](REVIEW.md) | Opus review of the five mappings: verdicts, must-fix errors, owner decisions, source spot-checks, rule proposals P1–P6 |
| [FIXROUND.md](FIXROUND.md) | Instructions for the fix round after the review (M1–M12, D23–D32), per language |
| [ldl/RESULTS.md](ldl/RESULTS.md) | First LDL run on the grids (`configs/pcfp_grid_v1.yaml`, exploratory): accuracy by language and cell type, copy analysis, auxiliary-error split |
| `mappings/` | per-language mapping files, the agents' notes and the check reports |
| `summarize.py` → `pilot_summary.csv` | per-language numbers from the check reports |
| [DECISIONS.md](DECISIONS.md) | dated decision log and owner notes (takes precedence over SPEC.md) |
| [BRIEF.md](BRIEF.md) | instructions for the per-language mapping agents (Phase 2) |
| `gridtools.py` | Phase 2 tool: `prepare` (Italian inputs), `inventory` (source cells and candidates in `inputs/`), `check` (validates and applies a mapping; evidence in `mappings/<iso>_check.md`) |
| `fetch_sources.sh` → `sources.sha256` | download of Morph-it! and the Wiktionary Italian verbs into `external/` |
| `stem_families.py` → `stem_families_summary.csv` | size of prefix-based stem families (DECISIONS D14) |
| `grambank_frequencies.py` → `grambank_frequencies.csv` | world-wide Grambank v1.0.3 shares (raw and family-weighted) for the verbal categories |
| `availability_check.py` → `availability.csv` | verbs with every needed source cell, per candidate dataset |

Both scripts run from the repository root with `.venv/bin/python`. They read only
`external/grambank`, `external/unimorph-ita` and `mgn_data`.

## Draft proposals in brief

* **Grid G16:**
  * present × 6 persons (habitual);
  * past × 6 persons (one completed event yesterday);
  * future 1SG (intention) and 3SG (prediction);
  * imperative 2SG and 2PL.

  The alternatives are G18 (full future, no imperative), G14 and G13.
* **Defaults:** pronoun subject, masculine, informal 2nd person, plural = a large group,
  exclusive 1PL, definite 3SG object for transitive verbs.
* **Open questions (my leans):**
  * auxiliaries: **C**, a single-word form with auxiliary distinctions counted in the
    distinctness flag, or **B** (multiword) if the LDL representation can handle
    spaces;
  * Slavic aspect: **S-a**, imperfective lemmas only, declared as a general
    "lexicalised aspect" rule;
  * word boundary: clitic auxiliaries are always separate words.
* **Pilot languages:**
  * Italian, from UniMorph `ita`: MGN lacks the past participle;
  * Finnish, from MGN;
  * English, from MGN phonemic;
  * **Eastern Armenian** in place of Basque or Hindi: Basque has 26 verbs, and Hindi has
    173 single-word verbs with inconsistent labels;
  * **Serbo-Croatian**, from MGN `hbs-v`, 4,324 complete verbs. The MGN Czech file has
    mislabelled columns.

## Questions for review

1. Do you accept the grid G16, or prefer G18, G14 or G13? Do the context frames and
   defaults (SPEC §3) work for you?
2. Do you accept the rules R1–R9? In particular, R3 (least-marked order) and R6 (class
   ids).
3. Auxiliaries: which of A, B or C? Slavic aspect: which of S-a to S-d? Word boundary:
   which of (a), (b) or (c)?
4. Are the languages acceptable? Should Armenian replace Hindi, or should Hindi be added
   as a 6th language? For Serbo-Croatian, should the Croatian or the Serbian standard be
   mapped?
5. Some of these would need a download (not done):
   * UniMorph `hin` (it may hold more verbs than MGN);
   * UniMorph `eng` (orthographic English);
   * a Serbo-Croatian aspect list (Wiktionary categories), needed for S-a.

   May I download them?
