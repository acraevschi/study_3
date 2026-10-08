"""Column names shared by all stages. The authoritative description is docs/CONTRACT.md."""

from __future__ import annotations

FORMS_COLUMNS = [
    "unit_id", "resource_id", "resource_version", "variety_id", "iso639_3", "glottocode",
    "pos", "representation", "lemma_id", "lemma_label", "group_id", "cell_orig",
    "cell_norm", "form_orig", "variant_idx", "n_variants", "form", "segments",
    "is_missing", "source_file", "source_row",
]

SPLIT_COLUMNS = [
    "unit_id", "repetition", "outer_fold", "lemma_id", "group_id", "role",
    "inventory_seed", "split_seed",
]
SPLIT_ROLES = ("test", "dev", "seed", "pool", "pool_overflow")

ORDER_COLUMNS = ["lemma_id", "acquisition_rank", "round", "lemma_score", "score_name"]
ACQ_LOG_COLUMNS = [
    "round", "lemma_id", "lemma_score", "n_cells_scored", "n_nonfinite",
    "rank_in_round", "selected",
]
CELL_SCORE_COLUMNS = [
    "round", "lemma_id", "target_cell", "hyp_rank", "hyp", "logprob_sum", "hyp_len",
    "surprisal_norm", "prob_renorm",
]

TEST_QUERY_COLUMNS = ["lemma_id", "source_cell", "source_form", "source_segments", "target_cell"]
PREDICTION_COLUMNS = [
    "lemma_id", "target_cell", "prediction", "prediction_segments", "status",
    "n_candidates", "top_candidates", "support", "n_source_cues", "n_source_cues_unseen",
    "unseen_target_features",
]
ITEM_COLUMNS = [
    "unit_id", "repetition", "outer_fold", "policy", "pool_cap", "budget", "model", "lemma_id",
    "group_id", "target_cell", "gold_variants", "prediction", "status", "correct",
    "edit_distance", "norm_edit_distance",
]

POLICIES_ACTIVE = ("low_confidence", "high_entropy")
POLICY_RANDOM = "random"
