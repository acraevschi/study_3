# Pipeline overview (pcfp_v1)

High-level map of the stages; details in [PROTOCOL.md](PROTOCOL.md), results in
[REPORT_pcfp_v1.md](REPORT_pcfp_v1.md). The pilot_v1 pipeline (source-known new verbs,
Transformer selector) is described in [REPORT.md](REPORT.md) and preserved at commit
24390cf.

```mermaid
flowchart TD
    data["Morphology data<br/>MGN verbs: Italian, Finnish"]
    cells["Cell inventory and eligibility<br/>single-word cells: 48 / 35<br/>complete paradigms"]
    expo["Exposure draw, once per verb<br/>k ~ U{1..7} shown cells, rest hidden"]
    cv["Grouped 3-fold design<br/>100 core verbs per fold (disjoint)<br/>seed 20 · pool 500"]
    tune["LDL setting choice<br/>auxiliary verbs, partial exposure"]
    sel["Active or random selection<br/>LDL selector: citation row per candidate"]
    audit["Leakage audit<br/>samples, selector rounds, queries"]
    ldl["JudiLing LDL fit<br/>shown forms of core + selected verbs<br/>fills hidden cells"]
    score["Scoring and bootstrap<br/>core items (primary), selected items"]
    out["LDL outcome table<br/>one row per design cell"]
    gb["Grambank + Glottolog<br/>v1.0.3 / v5.3"]
    typ["Typology stage<br/>inflection extent per language"]
    gelato["GeLaTo population links<br/>await human sign-off"]

    data --> cells --> expo --> cv --> sel --> ldl --> score --> out
    cv --> tune --> sel
    tune --> ldl
    cv -.- audit
    gb --> typ
    gelato --> out
    gelato --> typ

    classDef fitted fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class tune,sel,ldl fitted
```

Teal boxes are stages where models are fitted. The LDL outcome (inflecting languages
only) and the Grambank outcome (all GeLaTo-linked languages in Grambank) join on the
language-level Glottocode.

| Stage | What is fitted | Count in pcfp_v1 |
|---|---|---|
| LDL setting choice | JudiLing on the shown forms of 200 auxiliary verbs, scored on the hidden cells of 100 of them; grid cue n-gram {2,3} × inflection SD {0.4,1,2,4} | 16 (8 settings × 2 languages) |
| Selection | LDL selector refitted every round (3 semantic seeds), each pool candidate scored via a rank-one citation row | 18 active runs × 4 rounds × 3 seeds; 6 random runs (no model) |
| LDL fit | JudiLing end-state model on 100 core + 100 selected verbs | 24 |
| Typology | none (counts only) | – |
