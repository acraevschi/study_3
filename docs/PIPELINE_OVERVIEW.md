# Pipeline overview (pilot_v1)

High-level map of the stages; details in [PROTOCOL.md](PROTOCOL.md), results in
[REPORT.md](REPORT.md).

```mermaid
flowchart TD
    data["Morphology data<br/>MGN verbs: Italian, Finnish"]
    inv["Eligible lemma inventory<br/>1,200 lemmas, infinitive + 8 cells"]
    cv["Grouped 3-fold CV<br/>test ~400 · dev 80 · seed 20 · pool 500"]
    audit["Leakage audit<br/>0 problems"]
    sel["Active or random selection<br/>Transformer chooses 100 lemmas"]
    tune["LDL setting choice<br/>tuned on unused lemmas"]
    ldl["JudiLing LDL fit<br/>infinitive in, 8 forms out"]
    seleval["Selector evaluation<br/>same test lemmas"]
    score["Scoring and bootstrap<br/>accuracy, 95% lemma intervals"]
    out["Outcome tables<br/>one row per design cell"]
    gelato["GeLaTo population links<br/>await human sign-off"]

    data --> inv --> cv --> sel --> ldl --> score --> out
    cv -.- audit
    tune --> ldl
    sel --> seleval --> score
    gelato --> out

    classDef fitted fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class sel,tune,ldl,seleval fitted
```

Teal boxes are stages where models are fitted.

| Stage | What is fitted | Count in pilot_v1 |
|---|---|---|
| LDL setting choice | JudiLing on 100 auxiliary lemmas, scored on 80 others; grid cue n-gram {2,3} × inflection SD {0.4, 4.0} | 8 (4 settings × 2 languages) |
| Selection | Character Transformer, retrained from scratch at 20, 40, 60, 80, 100 lemmas | 24 runs (2 languages × 3 folds × 4 policy/pool settings), 5 trainings each |
| LDL fit | JudiLing end-state model on the 100 selected lemmas | 24 |
| Selector evaluation | Final selector of each run, scored on the same test lemmas | 24 |
