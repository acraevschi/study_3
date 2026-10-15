# Check report: eng

Mapping: `mappings/eng.yaml`. Data: mgn_custom:english-v (CELEX, phonemic, space-separated segments). Loader notes: none.

## Errors

none

## Warnings

none

## Summary

```json
{
 "lemmas_in_data": 6064,
 "out_of_scope": {
  "multiword_lemma": 124,
  "excluded_by_list": 1
 },
 "lemmas_in_scope": 5939,
 "complete_grids": 5933,
 "empty_cells": {
  "defective": 32,
  "data_gap": 7
 },
 "n_classes": 3,
 "n_formally_distinct_cells": 1,
 "rule_counts": {
  "source": 6,
  "syncretism": 8
 },
 "aux_carried_cells": 0,
 "classes": {
  "PRS.NON3SG": [
   "PRS.1SG",
   "PRS.2SG",
   "PRS.1PL",
   "PRS.2PL",
   "PRS.3PL",
   "IMP.2SG",
   "IMP.2PL"
  ],
  "PRS.3SG": [
   "PRS.3SG"
  ],
  "PST.ALL": [
   "PST.1SG",
   "PST.2SG",
   "PST.3SG",
   "PST.1PL",
   "PST.2PL",
   "PST.3PL"
  ]
 },
 "dropped_tokens": {},
 "empirical_distinct_forms_per_complete_grid": {
  "2": 44,
  "3": 5888,
  "6": 1
 },
 "empirical_mean_distinct_forms": 2.99,
 "empirical_cell_unique_share": {
  "PRS.1SG": 0.0,
  "PRS.2SG": 0.0,
  "PRS.3SG": 1.0,
  "PRS.1PL": 0.0,
  "PRS.2PL": 0.0,
  "PRS.3PL": 0.0,
  "PST.1SG": 0.0,
  "PST.2SG": 0.0,
  "PST.3SG": 0.0,
  "PST.1PL": 0.0,
  "PST.2PL": 0.0,
  "PST.3PL": 0.0,
  "IMP.2SG": 0.0,
  "IMP.2PL": 0.0
 },
 "emp_classes_per_complete_grid": {
  "2": 44,
  "3": 5888,
  "6": 1
 },
 "emp_mean_classes": 2.99,
 "class_exceptions": [
  "be"
 ],
 "pair_identity": {
  "PRS.1SG~PRS.2SG": 1.0,
  "PRS.1SG~PRS.1PL": 1.0,
  "PRS.1SG~PRS.2PL": 1.0,
  "PRS.1SG~PRS.3PL": 1.0,
  "PRS.1SG~IMP.2SG": 1.0,
  "PRS.1SG~IMP.2PL": 1.0,
  "PRS.2SG~PRS.1PL": 1.0,
  "PRS.2SG~PRS.2PL": 1.0,
  "PRS.2SG~PRS.3PL": 1.0,
  "PRS.2SG~IMP.2SG": 1.0,
  "PRS.2SG~IMP.2PL": 1.0,
  "PRS.1PL~PRS.2PL": 1.0,
  "PRS.1PL~PRS.3PL": 1.0,
  "PRS.1PL~IMP.2SG": 1.0,
  "PRS.1PL~IMP.2PL": 1.0,
  "PRS.2PL~PRS.3PL": 1.0,
  "PRS.2PL~IMP.2SG": 1.0,
  "PRS.2PL~IMP.2PL": 1.0,
  "PRS.3PL~IMP.2SG": 1.0,
  "PRS.3PL~IMP.2PL": 1.0,
  "PST.1SG~PST.2SG": 1.0,
  "PST.1SG~PST.3SG": 1.0,
  "PST.1SG~PST.1PL": 1.0,
  "PST.1SG~PST.2PL": 1.0,
  "PST.1SG~PST.3PL": 1.0,
  "PST.2SG~PST.3SG": 1.0,
  "PST.2SG~PST.1PL": 1.0,
  "PST.2SG~PST.2PL": 1.0,
  "PST.2SG~PST.3PL": 1.0,
  "PST.3SG~PST.1PL": 1.0,
  "PST.3SG~PST.2PL": 1.0,
  "PST.3SG~PST.3PL": 1.0,
  "PST.1PL~PST.2PL": 1.0,
  "PST.1PL~PST.3PL": 1.0,
  "PST.2PL~PST.3PL": 1.0,
  "IMP.2SG~IMP.2PL": 1.0
 }
}
```
