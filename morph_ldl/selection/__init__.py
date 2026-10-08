"""Active lemma selection (docs/CONTRACT.md §5, docs/SELECTION.md)."""

from morph_ldl.selection.acquisition import (POLICIES, AcquisitionResult, CandidatePoolView, CandidateQuery,
                                             GoldAccessError, Oracle, SelectionSeeds, SelectionTask,
                                             build_examples, policy_dir, predict_queries, round_plan,
                                             run_acquisition, score_candidates, train_selector_on_sample)
from morph_ldl.selection.model import Example, Hypothesis, SelectorConfig, TrainedSelector, train_selector
from morph_ldl.selection.scoring import (aggregate_lemmas, rank_lemmas, renormalise, score_cell,
                                         surprisal_norm, thresholded_entropy)
