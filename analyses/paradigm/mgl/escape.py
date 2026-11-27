"""Memoryless stochastic character-edit backoff `P_edit(y | w_s)` and Dirichlet-escaped MGL scoring.

Implements Study 3 Track 2 §2.1 step 4:
- Within-candidate maximization:
    `Score(y | w_s) = max_{R: w_s => y} Confidence(R) in [0, 1]`
    `Z(w_s) = sum_{y' in Y(w_s)} Score(y' | w_s)`
- Metric B (Bounded Parameter-Free Reliability):
    `Rel(w_t | w_s) = Score(w_t | w_s) in [0, 1]` (`0.0` if suppletive/unattested) + top-1 `Acc`.
- Metric A (Proper Generative Cross-Entropy over `Sigma*` in bits):
    Memoryless stochastic character-edit transducer `P_edit(y | w_s)` fitted via Laplace-smoothed
    MLE on training alignments (`p_copy + p_sub + p_del + p_ins = 1` for `i < |w_s|`;
    `p_ins_end + p_stop_end = 1` at `i == |w_s|`; target unigram `P_char(c)` summing to `1`),
    marginalized over all alignment paths via exact forward dynamic programming so
    `sum_{y in Sigma*} P_edit(y | w_s) == 1.0` identically.
    Combined with MGL via Dirichlet escape (`gamma = 0.05` fixed globally):
    `P(y | w_s) = (Score(y | w_s) + gamma * P_edit(y | w_s)) / (Z(w_s) + gamma)`.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .align import align_pair
from .mgl_learner import MGLLearner
from .vowel_tier import VowelClassifier

DEFAULT_GAMMA_ESCAPE: float = 0.05
_LN2: float = math.log(2.0)
_NEG_INF: float = -1e300


def _logaddexp(a: float, b: float) -> float:
    if a <= _NEG_INF:
        return b
    if b <= _NEG_INF:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass(frozen=True)
class DualScore:
    """Dual evaluation metrics for a `(w_s -> w_t)` pair."""

    nll_bits: float
    log_prob: float
    prob: float
    reliability: float
    exact_match: float
    predicted: tuple[str, ...]
    z_mass: float


class StochasticEditModel:
    """Strictly proper memoryless character-edit distribution `P_edit(y | x)` over `Sigma*`."""

    def __init__(
        self,
        classifier: VowelClassifier | None = None,
        alphabet: Sequence[str] | None = None,
        laplace: float = 1.0,
        oov_prior_mass: float = 1e-6,
        oov_bucket_size: int = 256,
    ) -> None:
        self.classifier = classifier if classifier is not None else VowelClassifier()
        self._explicit_alphabet = (
            tuple(dict.fromkeys(s for s in alphabet if s != "#")) if alphabet is not None else None
        )
        self.laplace = laplace
        # When an explicit closed alphabet is supplied (e.g. small-alphabet unit tests verifying
        # sum_{y in Sigma*} P == 1), set `oov_prior_mass = 0.0` so 100% of mass lives on `alphabet`.
        self.oov_prior_mass = 0.0 if self._explicit_alphabet is not None else oov_prior_mass
        self.oov_bucket_size = max(1, int(oov_bucket_size))

        self.alphabet: tuple[str, ...] = self._explicit_alphabet or ()
        self.p_copy: float = 0.70
        self.p_sub: float = 0.10
        self.p_del: float = 0.10
        self.p_ins: float = 0.10
        self.p_ins_end: float = 0.25
        self.p_stop_end: float = 0.75

        self._log_p_copy: float = math.log(self.p_copy)
        self._log_p_sub: float = math.log(self.p_sub)
        self._log_p_del: float = math.log(self.p_del)
        self._log_p_ins: float = math.log(self.p_ins)
        self._log_p_ins_end: float = math.log(self.p_ins_end)
        self._log_p_stop_end: float = math.log(self.p_stop_end)

        self.char_probs: dict[str, float] = {}
        self._log_char_probs: dict[str, float] = {}
        self._log_single_oov_prob: float = -30.0

    def fit(
        self,
        train_pairs: Iterable[tuple[Sequence[str], Sequence[str]]],
        precomputed_columns: Sequence[tuple[tuple[str, str], ...]] | None = None,
    ) -> StochasticEditModel:
        """Fit edit operation and unigram emission probabilities by Laplace-smoothed MLE."""
        pairs = [
            (
                tuple(s for s in src if s != "#"),
                tuple(s for s in tgt if s != "#"),
            )
            for src, tgt in train_pairs
        ]

        c_copy = 0.0
        c_sub = 0.0
        c_del = 0.0
        c_ins = 0.0
        c_ins_end = 0.0
        c_stop_end = 0.0
        char_counts: Counter[str] = Counter()
        vocab_seen: set[str] = set()

        for idx_p, (x, y) in enumerate(pairs):
            vocab_seen.update(x)
            vocab_seen.update(y)
            for tok in y:
                char_counts[tok] += 1

            if precomputed_columns is not None and idx_p < len(precomputed_columns):
                cols = precomputed_columns[idx_p]
            else:
                cols, _ = align_pair(x, y, self.classifier)
            inner_cols = cols[1:-1]
            m = len(x)
            i_ptr = 0
            for a, b in inner_cols:
                if i_ptr < m:
                    if a == "" and b != "":
                        c_ins += 1.0
                    elif a != "" and b == "":
                        c_del += 1.0
                        i_ptr += 1
                    elif a == b:
                        c_copy += 1.0
                        i_ptr += 1
                    else:
                        c_sub += 1.0
                        i_ptr += 1
                else:
                    if a == "" and b != "":
                        c_ins_end += 1.0
            c_stop_end += 1.0

        alpha = self.laplace
        denom_inner = c_copy + c_sub + c_del + c_ins + 4.0 * alpha
        self.p_copy = (c_copy + alpha) / denom_inner
        self.p_sub = (c_sub + alpha) / denom_inner
        self.p_del = (c_del + alpha) / denom_inner
        self.p_ins = (c_ins + alpha) / denom_inner

        denom_end = c_ins_end + c_stop_end + 2.0 * alpha
        self.p_ins_end = (c_ins_end + alpha) / denom_end
        self.p_stop_end = (c_stop_end + alpha) / denom_end

        self._log_p_copy = math.log(self.p_copy)
        self._log_p_sub = math.log(self.p_sub)
        self._log_p_del = math.log(self.p_del)
        self._log_p_ins = math.log(self.p_ins)
        self._log_p_ins_end = math.log(self.p_ins_end)
        self._log_p_stop_end = math.log(self.p_stop_end)

        if self._explicit_alphabet is not None:
            vocab = self._explicit_alphabet
        else:
            vocab = tuple(sorted(vocab_seen)) if vocab_seen else ("a",)
        self.alphabet = vocab

        total_char_weight = (
            sum(char_counts[c] + alpha for c in vocab) + self.oov_prior_mass
        )
        self.char_probs = {
            c: (char_counts[c] + alpha) / total_char_weight for c in vocab
        }
        self._log_char_probs = {
            c: math.log(p) for c, p in self.char_probs.items()
        }
        if self.oov_prior_mass > 0.0:
            p_oov_each = (self.oov_prior_mass / self.oov_bucket_size) / total_char_weight
            self._log_single_oov_prob = math.log(p_oov_each)
        else:
            self._log_single_oov_prob = _NEG_INF

        return self

    def log_char_prob(self, seg: str) -> float:
        return self._log_char_probs.get(seg, self._log_single_oov_prob)

    def log_prob(self, source_segs: Sequence[str], target_segs: Sequence[str]) -> float:
        """Exact marginal log-likelihood `ln P_edit(y | x)` via forward dynamic programming."""
        x = tuple(s for s in source_segs if s != "#")
        y = tuple(s for s in target_segs if s != "#")
        m = len(x)
        n = len(y)

        # Precompute log P_char(y[j]) for j = 0 .. n-1
        log_py = [self._log_char_probs.get(yj, self._log_single_oov_prob) for yj in y]

        # dp[i][j] = log forward probability of consuming x[:i] and emitting y[:j]
        dp = [[_NEG_INF] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0

        log_p_del = self._log_p_del
        log_p_ins = self._log_p_ins
        log_p_sub = self._log_p_sub
        log_p_copy = self._log_p_copy
        log_p_ins_end = self._log_p_ins_end

        for i in range(m + 1):
            row_i = dp[i]
            if i < m:
                xi = x[i]
                row_next = dp[i + 1]
                for j in range(n + 1):
                    cur = row_i[j]
                    if cur <= _NEG_INF:
                        continue
                    # 1. DELETE x[i]: (i, j) -> (i + 1, j)
                    row_next[j] = _logaddexp(row_next[j], cur + log_p_del)

                    if j < n:
                        lpy = log_py[j]
                        # 2. INSERT y[j]: (i, j) -> (i, j + 1)
                        row_i[j + 1] = _logaddexp(row_i[j + 1], cur + log_p_ins + lpy)
                        # 3. COPY or SUBSTITUTE x[i] -> y[j]: (i, j) -> (i + 1, j + 1)
                        diag_sub = log_p_sub + lpy
                        if xi == y[j]:
                            diag_step = _logaddexp(log_p_copy, diag_sub)
                        else:
                            diag_step = diag_sub
                        row_next[j + 1] = _logaddexp(row_next[j + 1], cur + diag_step)
            else:
                # Terminal state i == m: only INSERT_END or STOP
                for j in range(n):
                    cur = row_i[j]
                    if cur <= _NEG_INF:
                        continue
                    row_i[j + 1] = _logaddexp(row_i[j + 1], cur + log_p_ins_end + log_py[j])

        return dp[m][n] + self._log_p_stop_end

    def prob(self, source_segs: Sequence[str], target_segs: Sequence[str]) -> float:
        lp = self.log_prob(source_segs, target_segs)
        return 0.0 if lp <= _NEG_INF else math.exp(lp)

    def greedy_decode(self, source_segs: Sequence[str]) -> tuple[str, ...]:
        """Fallback top-1 candidate when MGL has `Z(w_s) == 0`."""
        x = tuple(s for s in source_segs if s != "#")
        if not self.char_probs:
            return x
        best_char = max(self.char_probs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        max_sub = self.p_sub * self.char_probs[best_char]
        if self.p_copy >= max(max_sub, self.p_del):
            base = list(x)
        elif self.p_del >= max_sub:
            base = []
        else:
            base = [best_char] * len(x)
        return tuple(base)


class DirichletEscapeScorer:
    """Combines `MGLLearner` rule scores with `StochasticEditModel` via Dirichlet escape (`gamma`)."""

    def __init__(
        self,
        mgl: MGLLearner,
        edit_model: StochasticEditModel,
        gamma: float = DEFAULT_GAMMA_ESCAPE,
    ) -> None:
        if gamma <= 0.0:
            raise ValueError(f"Dirichlet escape concentration gamma must be > 0, got {gamma}")
        self.mgl = mgl
        self.edit_model = edit_model
        self.gamma = gamma
        self._log_gamma = math.log(gamma)

    @classmethod
    def fit_from_pairs(
        cls,
        train_pairs: Sequence[tuple[Sequence[str], Sequence[str]]],
        classifier: VowelClassifier | None = None,
        gamma: float = DEFAULT_GAMMA_ESCAPE,
        alphabet: Sequence[str] | None = None,
        enable_vowel_tier: bool = True,
        enable_impugnment: bool = True,
    ) -> DirichletEscapeScorer:
        clf = classifier if classifier is not None else VowelClassifier()
        mgl = MGLLearner(
            classifier=clf,
            enable_vowel_tier=enable_vowel_tier,
            enable_impugnment=enable_impugnment,
        ).fit(train_pairs)
        pre_cols = [aln.columns for aln in mgl._last_alignments] if mgl._last_alignments else None
        edit = StochasticEditModel(classifier=clf, alphabet=alphabet).fit(
            train_pairs, precomputed_columns=pre_cols
        )
        return cls(mgl=mgl, edit_model=edit, gamma=gamma)

    def candidate_summary(
        self,
        source_segs: Sequence[str],
    ) -> tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float], tuple[str, ...], float]:
        """Compute `(cand_scores, cand_log_probs, top1_pred, Z)` for `source_segs` (`w_s`)."""
        x = tuple(s for s in source_segs if s != "#")
        raw_scores = self.mgl.score_candidates(x)
        cand_scores = {y: sc for y, sc in raw_scores.items() if sc > 0.0}
        z_mass = sum(cand_scores.values())
        log_denom = math.log(z_mass + self.gamma)

        cand_log_probs: dict[tuple[str, ...], float] = {}
        for y_cand, sc in cand_scores.items():
            lp_edit = self.edit_model.log_prob(x, y_cand)
            log_num = _logaddexp(math.log(sc), self._log_gamma + lp_edit)
            cand_log_probs[y_cand] = log_num - log_denom

        if cand_log_probs:
            # Deterministic tie-breaking: highest P(y|w_s), then highest Score(y|w_s), then string
            top1 = min(
                cand_log_probs.keys(),
                key=lambda y: (-cand_log_probs[y], -cand_scores[y], " ".join(y)),
            )
        else:
            top1 = self.edit_model.greedy_decode(x)

        return cand_scores, cand_log_probs, top1, z_mass

    def log_prob_with_cache(
        self,
        source_segs: Sequence[str],
        target_segs: Sequence[str],
        cand_scores: dict[tuple[str, ...], float],
        cand_log_probs: dict[tuple[str, ...], float],
        z_mass: float,
    ) -> float:
        y = tuple(s for s in target_segs if s != "#")
        cached = cand_log_probs.get(y)
        if cached is not None:
            return cached
        lp_edit = self.edit_model.log_prob(source_segs, y)
        if z_mass <= 0.0:
            return lp_edit
        return (self._log_gamma + lp_edit) - math.log(z_mass + self.gamma)

    def log_prob(self, source_segs: Sequence[str], target_segs: Sequence[str]) -> float:
        """Exact natural log probability `ln P(w_t | w_s)` over `Sigma*`."""
        cand_scores, cand_log_probs, _, z_mass = self.candidate_summary(source_segs)
        return self.log_prob_with_cache(
            source_segs, target_segs, cand_scores, cand_log_probs, z_mass
        )

    def prob(self, source_segs: Sequence[str], target_segs: Sequence[str]) -> float:
        lp = self.log_prob(source_segs, target_segs)
        return 0.0 if lp <= _NEG_INF else math.exp(lp)

    def score_pair(self, source_segs: Sequence[str], target_segs: Sequence[str]) -> DualScore:
        """Evaluate both Metric A (`-log2 P(w_t | w_s)`) and Metric B (`Rel`, `Acc`) on `(w_s, w_t)`."""
        x = tuple(s for s in source_segs if s != "#")
        y = tuple(s for s in target_segs if s != "#")
        cand_scores, cand_log_probs, top1, z_mass = self.candidate_summary(x)
        rel = cand_scores.get(y, 0.0)
        lp = self.log_prob_with_cache(x, y, cand_scores, cand_log_probs, z_mass)
        nll_bits = max(0.0, -lp / _LN2)
        p_val = 0.0 if lp <= _NEG_INF else math.exp(lp)
        acc = 1.0 if top1 == y else 0.0
        return DualScore(
            nll_bits=nll_bits,
            log_prob=lp,
            prob=p_val,
            reliability=rel,
            exact_match=acc,
            predicted=top1,
            z_mass=z_mass,
        )
