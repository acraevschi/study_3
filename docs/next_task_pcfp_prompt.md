# Task: replace the new-verb task with paradigm cell filling (PCFP), and the Transformer selector with an LDL selector; add a Grambank inflection-extent outcome

You are the main agent for the study_3 repository (the pipeline is at the repository root
on `main`; the archived earlier work is outside the repository, see the local `ARCHIVE_NOTE.md`). Read `docs/PROTOCOL.md`, `docs/CONTRACT.md`,
`docs/LDL_PROTOCOL.md`, `docs/SELECTION.md`, `docs/DATA.md`, `docs/REPORT.md` and
`docs/PIPELINE_OVERVIEW.md` first. They describe the current pipeline and the `pilot_v1`
results.

## Why

The current task gives LDL a verb it has never seen: the model gets the infinitive and
must produce 8 other cells. With simulated (random) lexeme vectors, LDL cannot know a new
verb's meaning. The pipeline therefore estimates the meaning from the infinitive and
binds it to the infinitive's cues (`wug_refit`). This step is weak. The top candidate is
the unchanged infinitive in 38–50% of items, so the pilot accuracies mostly measure this
step, not morphology. The user wants the classic PCFP instead. Every test verb is known
from some of its forms, and the model fills the cells it has not seen.

The selection model is also wrong for this study. Active selection currently uses a
character Transformer (after Muradoğlu & Hulden 2022) to choose the verbs that LDL is
then trained on. In the paper the model that chooses the data is the model that is
evaluated; here it is not. The pilot shows the cost: the Transformer's choices helped
the Transformer (Italian +0.039) but hurt LDL (Italian −0.137 against random). The
user wants LDL itself to choose its training verbs.

The study also needs a second outcome. LDL predictability is defined only for languages
that inflect, and paradigm data reach roughly 25 GeLaTo-linked languages. Leaving out
languages with little or no inflection (Vietnamese, Khmer, Mandarin, Thai) would cut off
the low end of the outcome. The user has decided on two outcomes:

- **Primary, high-power outcome: Grambank inflectional extent.** It covers every
  GeLaTo-linked language with Grambank data, including isolating ones: about 166 languages
  with ≥ 60% of the core features coded.
- **Secondary, finer outcome: LDL predictability (PCFP).** It covers inflecting languages
  only and is stated as conditional on inflection. Isolating languages are never given an
  LDL score.

A crude version of the Grambank extraction already exists in
`analyses/unimorph_gelato_estimate_2026_10_08/` (`README.md`,
`grambank_inflection_gelato.csv`). Treat it as a prototype, not as pipeline output.

## Decisions already made by the user

1. **Replace** the source-known new-verb task. Do not keep it as a secondary task in
   future runs. Leave `outputs/pilot_v1` untouched as the archived record.
2. **Full paradigms, with the same exposure cap in every language.**
   - Each verb shows k forms, k ~ Uniform{1, …, min(7, n_cells(v) − 1)}.
   - The cap is the same in every language, so every learner sees at most the same
     number of forms per verb. Small paradigms (e.g. English, about 5 cells) are capped
     by n_cells − 1.
   - Shown cells are drawn uniformly at random, without replacement, from the verb's
     eligible cells.
   - All other eligible cells of the verb are hidden. For core verbs they are the test
     items.
3. **Active selection chooses verbs, and cells are revealed at random.**
   - When a verb enters a training sample, it reveals its pre-drawn shown cells.
   - Draw k and the shown cells once per verb. Use a seed derived from `master_seed`, a
     new purpose (e.g. `exposure`), `unit_id` and `lemma_id`.
   - The draw is identical for every policy, fold, budget and pool cap. The policy never
     influences it.
4. **LDL replaces the Transformer as the selection model.**
   - Active policies score candidates with a JudiLing LDL model fitted on the current
     training sample, using the frozen settings from the new `ldl_tune` run.
   - Do not use the Transformer selector in `pcfp_v1`. Keep its code so that `pilot_v1`
     stays reproducible.
5. **Add a Grambank inflection-extent outcome as the primary admixture outcome** (see
   "Grambank workstream" below).
   - Build it for all GeLaTo-linked languages, not just the two pilot units.
   - It is independent of LDL. LDL remains the secondary outcome.
   - Delegate it to a subagent (see "Subagents").

## Proposed defaults (presented to the user; keep them configurable)

- **Single-word cells only.**
  - Italian MGN has 48 cells, all single-word.
  - Finnish has 137 cells, of which 102 are mainly multiword (periphrastic negatives and
    perfects such as *en aakkosta*, *olen aakkostanut*) and 35 are single-word.
  - Italian MGN has no compound tenses, so multiword cells would inflate Finnish
    complexity by resource construction.
  - Declare the rule before any run: exclude a cell if most of its variant-0 forms are
    multiword, and exclude any remaining multiword form. Report the cell lists and counts.
  - Keep the existing lemma eligibility exclusions (derived/pronominal paradigms,
    multiword lemmas). Declare how missing cells affect eligibility.
- **Core + selected design, so that policies are scored on identical items.**
  - **Core verbs:** per outer fold, a fixed set of C = 100 verbs (whole leakage groups)
    is in every learner's training through its shown forms. Its hidden cells are the
    primary test items, the same for every policy.
  - **Selected verbs:** the seed (shared) plus active or random acquisition from the
    pool, up to budget B = 100 verbs. As now, the budget counts verbs and includes the
    seed.
  - Report the number of forms in every sample. Report that core forms are part of every
    learner's training.
  - Choose the inventory, number of folds and repetitions so that core sets are disjoint
    across folds. Check capacity, and never silently shrink a budget.
  - Secondary outcome: accuracy on the hidden cells of the selected verbs. It is
    policy-dependent and must be labelled that way.
- The dev set existed for Transformer early stopping. LDL is fitted in closed form, so
  it needs no early stopping.
  - Either drop the dev role and declare where its capacity goes, or keep it only to
    monitor selector accuracy per round.
  - In both cases dev verbs are never LDL training or test data. Document the choice.

## What to change

- **Data / cells:**
  - Build the full eligible cell inventory per unit, with no fixed 8-slot panel.
  - `unit_cells` and the configs now list the eligible cells.
  - Export the exposure draw (lemma_id, k, shown cells, hidden cells) as a manifest
    written by the `splits` stage, and validate it.
- **Splits:**
  - New roles: core / seed / pool / overflow (plus dev if kept), and the auxiliary
    non-inventory set for LDL tuning, as before.
  - Selector copy anchors were Transformer-specific. Do not use them in `pcfp_v1`.
  - Use nested pool prefixes for the pool-cap sensitivity run, as before.
- **Selector (LDL instead of the Transformer):**
  - Each acquisition round, fit a JudiLing model on the shown forms of the current
    training verbs (core + seed + verbs acquired so far). Use the frozen `ldl_tune`
    settings and the same semantics as the evaluated LDL.
  - A candidate's score is LDL's uncertainty over its pre-drawn shown cells: the cells
    are known, the forms are not. Candidate forms never enter the fit, the cue
    inventory or the candidate lists.
  - **What LDL knows about a candidate.** Only the lemma label (citation form) and its
    simulated lexeme vector, which comes from the lemma ID and carries no information
    about its forms. A lexeme vector that LDL has never seen paired with a form says
    nothing about the stem. The user has decided how to resolve this:
    - **Primary method (decided): the citation form as one known form.**
      - Score each candidate as a verb that has been seen through one form. Build its
        citation row: the citation form's cues, paired with the candidate's simulated
        semantics for the citation cell (lexeme vector + the citation cell's feature
        vectors + noise), generated exactly as for any training row.
      - Add this row to the round's fitted model with the exact rank-one update that
        already exists for `wug_refit`. Decode the candidate's pre-drawn shown cells
        from its lexeme vector + each cell's feature vectors. Then remove the row, so
        that no candidate influences another candidate's score.
      - This is the k = 1 case of the evaluated task. It uses no comprehension-based
        estimate of the meaning.
      - Only the citation form enters; no other form of the candidate does. The
        evaluated LDL never contains pool candidates' citation rows.
    - **Check (run alongside, not used for selection):** a comprehension-side score for
      every candidate, from the same round's model:
      - how far the meaning that LDL reads from the citation form (c·F) is from the
        candidate's simulated semantics for that cell;
      - the share of the citation form's cues that are unseen in the training cues.
      Log both per round. Report their rank correlations with the primary scores and
      with simple form properties: citation-form length, final letters, inflection
      class where it is known.
    - **Alternative (not the default):** estimate the meaning by comprehension and bind
      it to the citation form, as `wug_refit` did in `pilot_v1`. Use it only if the user
      asks.
    - **Stop condition:** if a smoke check shows that the primary scores are
      degenerate, report it and ask the user before choosing another approach.
      Degenerate means near-constant, dominated by copying the citation form, or
      explained by citation-form length or final letters alone.
  - **Uncertainty measures.** LDL gives candidate supports, not probabilities, so
    redefine the policies and declare the definitions before any run:
    - low_confidence: low support of the top candidate, averaged over the shown cells
      (a top-two margin is an acceptable alternative; pick one);
    - high_entropy: entropy over the candidate supports turned into weights. This needs
      a temperature or normalisation; declare it and do not tune it on outer-test data.
  - Keep deterministic tie-breaking, batch size, budgets and nested pool prefixes.
  - Simulated semantics add noise to the scores. Consider averaging scores over a few
    semantic seeds, and record what you choose.
  - Cost: each round needs a fit plus decoding of every pool candidate's shown cells.
    Avoid restarting Julia per round if that dominates runtime.
  - The separate `selector` evaluation stage (Transformer accuracy on test items) no
    longer applies. Remove it from `pcfp_v1`, or report the LDL selector's own accuracy
    there, and label it.
- **Citation-form issue:**
  - In Italian and Finnish the citation form equals the infinitive cell. For the selector,
    that cell's form is therefore known even when it is hidden.
  - Resolve this explicitly and document it. The LDL selector's citation row (above)
    relies on this cell. The two options:
    - **Recommended:** treat the lemma label as a known lexical label for the selector
      only, and exclude the citation-form cell from the test items, so that every
      policy is scored on identical items.
    - **Alternative:** treat the citation cell as always shown, counted inside k, for
      every verb.
- **LDL (JudiLing 1.0.1):**
  - Use standard known-lexeme production. Training rows are the shown forms of training
    verbs. Semantics are lexeme + inflectional-feature vectors + noise, generated from
    per-identifier seeds as now.
  - Target semantics for a hidden cell are the verb's lexeme vector plus that cell's
    feature vectors. Decode with `learn_paths`.
  - Remove `wug_refit` and the source-binding path from the evaluated LDL. Keep the
    exact rank-one update routine, because the selector's citation row reuses it.
  - Gold hidden forms must not influence cue inventories, adjacency, max lengths,
    candidate lists or settings. Keep avoiding `make_combined_cue_matrix`,
    `make_combined_S_matrix` and `cal_max_timestep`.
  - Features that never occur in training (possible in large paradigms) must be handled
    and reported, not silently dropped.
- **LDL tuning:**
  - Re-run `ldl_tune` for the new task on auxiliary lemmas only (core-like partial
    exposure on auxiliary verbs, tested on their hidden cells). Do it before any
    outer-test fit, and keep the refusal guard.
  - The LDL selector uses these settings, so `ldl_tune` must now run before `select`.
    Update the stage order and the refusal guard (refuse once selection outputs exist).
  - Revisit the grid if it hits a boundary.
- **Evaluation:**
  - Report exact-form accuracy (any variant), micro and macro by verb, and edit distance.
  - Report per-cell accuracy, accuracy by k (number of shown forms) and the number of
    test cells per verb. Report LDL mapping quality separately.
  - Use lemma-cluster (leakage-group) bootstrap intervals and paired policy differences
    on identical items. State that they are conditional on the fitted fold models. Do not
    use fold SD/√K.
- **Outcomes:**
  - Write the outcome table and the population linkage table as before.
  - Add exposure-cap, k-distribution and cell-inventory metadata to each outcome row.
- **Audit:** add these checks:
  - hidden cells of core verbs never appear in any training file, cue inventory,
    selector training data or tuning data;
  - pool candidates' forms never enter a selector fit before the candidate is acquired;
  - exposure draws are identical across policies, folds and budgets;
  - every core verb has at least one shown form;
  - budgets and form counts match the manifests.
- **Docs and configs:**
  - Write a new experiment config (e.g. `configs/pcfp_v1.yaml`) and a matching smoke
    config, with outputs in a new directory.
  - Update PROTOCOL, CONTRACT, DATA, SELECTION, LDL_PROTOCOL and PIPELINE_OVERVIEW (with
    the mermaid diagram).
  - Write a new report. Record every decision and deviation.

## Grambank workstream (delegate to a subagent)

This part has no dependency on the PCFP or selector work. Start it early and let it run in
parallel. Spawn one `ldl-implementer` subagent for it. Give it this section, the
GeLaTo-related parts of `docs/DATA.md` and `docs/CONTRACT.md`, the prototype in
`analyses/unimorph_gelato_estimate_2026_10_08/`, and the tests it must pass. Review and
integrate its work yourself.

- **Sources (pin them and record them in the stage manifest):**
  - Grambank CLDF v1.0.3 (`grambank/grambank`, tag `v1.0.3`): `values.csv`,
    `parameters.csv`, `codes.csv`, `languages.csv`.
  - Glottolog CLDF v5.3 (`glottolog/glottolog-cldf`, tag `v5.3`), `languages.csv`. Use it
    for the dialect → language roll-up (`Level`, `Language_ID`) and for family and
    macroarea.
  - GeLaTo populations: the same pinned sources as the existing crosswalk.
  - Add the downloads to `scripts/fetch_external.sh`.
- **Unit of observation:**
  - One row per language-level Glottocode.
  - GeLaTo populations coded at dialect level (e.g. `Karelian_Northern`, `Irish_Munster`)
    are rolled up to their language with Glottolog.
  - GeLaTo populations coded at group level (e.g. Japanese `japa1256`) are mapped down
    only when exactly one language-level Grambank entry matches.
  - Record the mapping basis (exact, dialect roll-up, group map-down, manual) as its own
    column. Roll-up is an initial match, not acceptance.
- **Feature set (declare it in the config and docs before any linkage output is written):**
  - Start from the 35 core inflectional features of the prototype:
    - verbal TAM and other verbal affixes: GB079, GB080, GB082, GB083, GB084, GB086,
      GB312;
    - person indexing on the verb: GB089–GB094;
    - negation and interrogation verb morphology: GB107, GB286;
    - nominal number: GB042, GB043, GB044, GB165, GB166;
    - case: GB070–GB073;
    - possession affixes: GB430–GB433;
    - agreement: GB170, GB171, GB172, GB184, GB185, GB186, GB198.
  - Exclude derivation (GB047–GB049), diminutive/augmentative (GB187, GB188), auxiliary
    verbs (GB119–GB121, GB298), valency and voice morphology (GB103, GB104, GB113, GB147,
    GB148, GB155), and the bound comparative (GB275).
  - Justify every inclusion and exclusion in one line each.
  - Declare sensitivity sets: verbal only, nominal only, and without agreement.
  - Grambank counts clitics as bound morphology, which inflates some analytic languages
    (e.g. Burmese 15/35). Document this. Flag languages where the score may be driven by
    clitics, but do not recode Grambank values.
- **Outcome representation:**
  - Per language and feature set: `n_present`, `n_coded`, `n_features`, `coverage` and
    `share = n_present / n_coded`.
  - `?` and missing values count as not coded.
  - Keep the counts, not only the share: the planned analysis is binomial-type.
  - Declare a coverage threshold for the main set (the prototype used ≥ 60%). Report how
    many languages enter at 50% and 75%.
  - Add a `no_inflection` flag (`n_present == 0` on the main set) and a
    `minimal_inflection` flag (≤ 1), with family and macroarea.
- **GeLaTo linkage:**
  - Reuse the existing crosswalk logic and statuses:
    - one outcome row per language;
    - populations only in the separate link table;
    - no averaging of populations;
    - acceptance requires a human `confirmed_by` and `confirmed_date`.
  - For languages that also have an LDL outcome, use the same key (language-level
    Glottocode), so that the two outcome tables join without ambiguity.
- **Coverage report:**
  - GeLaTo languages missing from Grambank, by GeLaTo sample size. The prototype found
    Yoruba (75 individuals), Hmong, Dai/Lü, Yi and Tujia among them. Do not hand-code
    them in this task; list them as gaps.
  - The overlap with the LDL-eligible languages.
- **Pipeline integration:**
  - A new module (e.g. `morph_ldl/typology/grambank.py`) and a new CLI stage (e.g.
    `typology`), independent of `select`, `ldl` and `evaluate`.
  - Outputs go to `outputs/<experiment>/typology/`: `grambank_inflection.csv`,
    `grambank_population_links.csv`, `feature_set.json` and a coverage summary.
  - `stage_audit` checks:
    - the declared feature set equals the one used;
    - no ancestry file was read;
    - every outcome row has exactly one language-level Glottocode.
- **Planned analysis (document it in PROTOCOL; do not fit it):**
  - **Grambank extent:** beta-binomial on (`n_present`, `n_coded`) per language. Add a
    hurdle or zero-inflation part only if posterior predictive checks show excess zeros:
    the prototype found 4–6 languages at or near zero.
  - **LDL predictability:** binomial at the item level with language random effects, or
    beta-binomial per language. Inflecting languages only.
  - **Confounding:** the zero/near-zero languages are almost all in Mainland Southeast
    Asia and China and share East Asian ancestry. Any no-inflection part is therefore
    weakly identified once area is controlled. State this, and describe it as
    descriptive, not as a main test.
  - The admixture exposure variable is still undefined. Do not define it, look at it or
    fit anything with it.
- **Tests the subagent must add:**
  - feature-set counts on a small fixture of Grambank values, including `?` and missing;
  - dialect roll-up and group map-down on Glottolog fixtures, including the ambiguous
    case (several matching languages → no automatic match);
  - deterministic output;
  - no ancestry or Q-matrix path is opened (fail the test if one is);
  - sources and pinned versions recorded in the stage manifest.

## Constraints that still apply

- **Preservation:**
  - Preserve existing datasets, results and unrelated changes.
  - Do not overwrite raw data or reinterpret old scores in place.
  - Write new outputs separately.
  - Do not reuse MGN accuracy results, complexity scores or predictions as gold or as
    outcomes.
- **Removing the old task:** the code that produced `pilot_v1` is committed on `main`
  (the 2026-10-08 reorganisation commit). Removing old-task code in this task is
  therefore allowed. Record that commit hash in the new report as the `pilot_v1` code
  state. Commit or push only when the user asks.
- **Selection:** candidate target values stay unavailable to scoring. Use deterministic
  tie-breaking and nested prefixes. Never silently reduce budgets. Run acquisition inside
  every outer fold.
- **GeLaTo:**
  - Identifier equality is only an initial match. Accepted links still need a human
    `confirmed_by` and `confirmed_date`.
  - Several populations linked to one outcome must not create independent observations.
    Do not average populations or pool incompatible varieties.
- **No ancestry fitting:** do not fit any admixture–morphology model, and do not choose
  settings or languages through ancestry associations.
- **Scope:** no broad data collection, no large multilingual campaign, no paid remote
  inference. The PCFP/LDL work stays with the two pilot units (Italian and Finnish verbs,
  MGN). The one exception approved by the user is the Grambank workstream: it reads one
  pinned typological dataset (plus Glottolog) for all GeLaTo-linked languages. It does
  not download paradigm data for new languages.
- **Subagents:**
  - Use implementation subagents (`ldl-implementer`, Opus) for bounded components and a
    separate read-only `ldl-reviewer` for leakage and design review before the real run.
  - You may delegate the LDL selector to an `ldl-implementer` subagent instead of
    writing it yourself. Give it the selection interface (`run_acquisition` in
    `morph_ldl/selection/acquisition.py`), the JudiLing runner, the decisions above and
    the tests it must pass. Review and integrate its work yourself.
  - The Grambank workstream must go to its own `ldl-implementer` subagent (see "Grambank
    workstream"), run in parallel with the PCFP work. Do not implement it in the main
    agent's context.
  - The reviewer must also check the LDL selector for leakage of candidate forms.
  - The reviewer must also check the Grambank stage:
    - the feature set was declared before linkage;
    - the roll-up and map-down rules are correct;
    - no ancestry data were read;
    - one row per language.
  - The main agent owns cross-validation, integration and verification.
  - If the requested model or effort is unavailable, report it and ask before
    substituting.
- **Questions:** ask focused questions only for consequential choices not settled above.

## Done means

1. All tests pass, including new tests for:
   - the exposure draw;
   - core/test disjointness from training;
   - the citation-cell rule;
   - LDL known-lexeme decoding on a toy paradigm;
   - the LDL selector:
     - candidate scores do not change when a candidate's hidden or shown gold forms are
       altered (only its citation form may matter);
     - scores do not depend on the order in which candidates are scored, because the
       citation row is removed after each candidate;
     - the rank-one update matches a full refit on a toy example;
     - scoring and tie-breaking are deterministic.
2. The smoke config runs end to end.
3. The independent review has no unresolved major findings.
4. The real `pcfp_v1` run has completed, with real active selection and JudiLing
   evaluation in every outer fold. The audit reports 0 problems.
5. The final report gives per-language accuracy with intervals, paired active − random
   differences, the source/stem-copy rate compared with `pilot_v1`, and per-cell and
   per-k breakdowns. It describes what the LDL selector chose (e.g. inflection-class
   composition against random) and compares this with the Transformer choices in
   `pilot_v1`. It reports the comprehension-side check: how strongly it correlates with
   the selector scores and with citation-form properties. It lists limitations: random
   rather than frequency-based exposure, the single-word-cell restriction, and the
   selector's reliance on the citation form as its one known form.
6. The `typology` stage has run. It has written the Grambank inflection-extent table for
   all GeLaTo-linked languages in Grambank, with the link table, the declared feature set
   and sensitivity sets, and the coverage report. The audit checks for it pass. The
   report states:
   - how many languages enter the Grambank outcome;
   - how many have no or minimal inflection;
   - which GeLaTo languages are missing from Grambank;
   - how the Grambank and LDL outcomes overlap.

Do not report the pilot as complete before steps 4 and 6.
