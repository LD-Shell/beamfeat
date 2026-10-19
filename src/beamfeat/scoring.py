"""Candidate scoring strategies for guided beam search.

A :class:`Scorer` answers one question: how useful is this candidate feature,
given the target and the features already selected? The "given the features
already selected" part is what keeps a beam from filling with near-duplicates
of the same underlying signal, which is the dominant failure mode of scoring
candidates in isolation.

Four strategies are provided, trading signal quality against cost:

:class:`CorrelationScorer`
    Absolute Pearson correlation against the residual left by the incumbent
    features. Cheapest by a wide margin; detects only monotone-linear
    relationships to the residual, but that is often enough to *find* a feature
    whose nonlinearity is already baked into its expression.

:class:`SpearmanScorer`
    Absolute rank correlation against the same residual. One sort per column
    dearer than Pearson; sees any monotone relationship and is insensitive to
    outlying values.

:class:`MutualInformationScorer`
    Nearest-neighbour mutual information between candidate and residual.
    Detects nonmonotone dependence that correlation misses, at roughly one to
    two orders of magnitude more compute.

:class:`GradientBoostingScorer`
    Measured out-of-fold improvement in predictive performance when the
    candidate is added to the incumbent set. The most faithful signal, since it
    scores what is actually being optimised, and the most expensive.

All scorers are deterministic given a ``random_state``, and all handle the
empty-incumbent case (the first beam round, where there is no residual yet) by
scoring against the raw target.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

__all__ = [
    "CorrelationScorer",
    "GradientBoostingScorer",
    "MutualInformationScorer",
    "Scorer",
    "SpearmanScorer",
    "is_constant",
    "make_scorer",
]

ProblemType = Literal["regression", "classification"]


def _check_problem_type(problem_type: str) -> ProblemType:
    if problem_type not in ("regression", "classification"):
        raise ValueError(f"problem_type must be 'regression' or 'classification', got {problem_type!r}")
    return problem_type  # type: ignore[return-value]


def is_constant(values: np.ndarray) -> bool:
    """Whether a column carries no variation a scorer can use.

    Constancy is judged relative to the values' magnitude rather than by an
    absolute floor on the spread, since an absolute floor makes the verdict
    depend on the column's units: a product of small-unit columns sits at
    the product of their magnitudes and lands under any fixed floor while
    carrying real signal. A spread twelve orders below the magnitude is far
    inside accumulated float64 rounding noise, so nothing informative is
    discarded.

    Exposed so that a caller screening a wide table can apply the same test
    the estimator applies, and so that the diagnostic reported at fit time
    cannot drift from the behaviour it describes.
    """
    centred = values - values.mean()
    scale = float(np.sqrt(np.mean(centred**2)))
    magnitude = float(np.mean(np.abs(values)))
    if np.isfinite(magnitude) and magnitude > 0.0:
        return scale <= 1e-12 * magnitude
    return scale == 0.0


def _standardise(values: np.ndarray) -> np.ndarray:
    """Centre and scale to unit variance. Constant input returns zeros.

    Constancy is decided by :func:`is_constant`; filtering of informationally
    constant columns happens in the expression layer before candidates reach
    a scorer.
    """
    centred = values - values.mean()
    if is_constant(values):
        return np.zeros_like(centred)
    return centred / float(np.sqrt(np.mean(centred**2)))


class Scorer(ABC):
    """Base class for candidate scoring strategies.

    Subclasses implement :meth:`score_batch`. Scores are compared only against
    one another within a single beam round, so their absolute scale is
    unconstrained; only the ordering matters. Higher is better, and scores must
    be non-negative and finite.
    """

    #: Human-readable identifier, used in logs and reports.
    name: str = "scorer"

    def __init__(self, problem_type: ProblemType = "regression", random_state: int | None = 0) -> None:
        self.problem_type = _check_problem_type(problem_type)
        self.random_state = random_state

    @abstractmethod
    def score_batch(
        self,
        candidates: np.ndarray,
        target: np.ndarray,
        incumbent: np.ndarray | None = None,
    ) -> np.ndarray:
        """Score every candidate column against the target.

        Args:
            candidates: ``(n_samples, n_candidates)`` matrix of candidate
                feature values.
            target: ``(n_samples,)`` target vector.
            incumbent: Optional ``(n_samples, n_selected)`` matrix of features
                already selected. Scorers use this to measure *marginal* value,
                so that a candidate duplicating an incumbent scores near zero.

        Returns:
            ``(n_candidates,)`` array of non-negative finite scores.
        """

    def score(self, candidate: np.ndarray, target: np.ndarray, incumbent: np.ndarray | None = None) -> float:
        """Score a single candidate. Convenience wrapper over :meth:`score_batch`."""
        return float(self.score_batch(candidate.reshape(-1, 1), target, incumbent)[0])

    # -- shared helpers ----------------------------------------------------- #

    def _validate(
        self, candidates: np.ndarray, target: np.ndarray, incumbent: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Coerce shapes and dtypes, and check row counts agree."""
        candidates = np.asarray(candidates, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(-1, 1)
        if candidates.ndim != 2:
            raise ValueError(f"candidates must be 2-D, got {candidates.ndim}-D")

        target = np.asarray(target).ravel()
        if target.shape[0] != candidates.shape[0]:
            raise ValueError(f"target has {target.shape[0]} rows, candidates have {candidates.shape[0]}")

        if incumbent is not None:
            incumbent = np.asarray(incumbent, dtype=np.float64)
            if incumbent.ndim == 1:
                incumbent = incumbent.reshape(-1, 1)
            if incumbent.shape[0] != candidates.shape[0]:
                raise ValueError(f"incumbent has {incumbent.shape[0]} rows, candidates have {candidates.shape[0]}")
            if incumbent.shape[1] == 0:
                incumbent = None

        if self.problem_type == "classification":
            # Labels are categories, so they are scored through integer codes:
            # string and object labels then behave like their numeric
            # equivalents, and no path casts a label to float.
            target = _as_codes(target)
        else:
            target = target.astype(np.float64)

        return candidates, target, incumbent

    def _residualise(self, target: np.ndarray, incumbent: np.ndarray | None) -> np.ndarray:
        """Return the part of the target not explained by the incumbent set.

        Regression uses a least-squares projection (cheap and stable via
        ``lstsq``). Binary classification uses the *working residual* of a
        logistic fit on the incumbent set, ``y - p_hat`` — one step of
        iteratively reweighted least squares, so a candidate scores by how
        well it explains what the incumbent logistic model cannot. A linear
        least-squares residual of a 0/1 target underweights multiplicative
        boundaries such as ``x0*x1 > x2*x3``; the working residual does not.
        Multiclass targets fall back to the numeric projection of the integer
        class codes, documented as a heuristic.
        """
        numeric_target = target.astype(np.float64)

        if self.problem_type == "classification":
            classes = np.unique(target)
            if len(classes) == 2:
                binary = (target == classes[1]).astype(np.float64)
                if incumbent is None or incumbent.shape[1] == 0:
                    return binary - binary.mean()
                from sklearn.linear_model import LogisticRegression

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = LogisticRegression(max_iter=500)
                    model.fit(incumbent, binary)
                    probabilities = model.predict_proba(incumbent)[:, 1]
                residual = binary - probabilities
                if np.all(np.isfinite(residual)):
                    return residual
                return binary - binary.mean()  # pragma: no cover - defensive

        if incumbent is None or incumbent.shape[1] == 0:
            return numeric_target

        design = np.column_stack([np.ones(len(numeric_target)), incumbent])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coefficients, *_ = np.linalg.lstsq(design, numeric_target, rcond=None)
        residual = numeric_target - design @ coefficients
        if not np.all(np.isfinite(residual)):  # pragma: no cover - defensive
            return numeric_target
        spread = float(np.sqrt(np.mean((numeric_target - numeric_target.mean()) ** 2)))
        if float(np.sqrt(np.mean(residual**2))) <= 1e-9 * spread:
            # The incumbent reproduces the target to float precision; what is
            # left is rounding dust, and standardising dust would hand the
            # scorers a unit-variance noise vector to chase. The comparison is
            # made against the target's spread so the verdict does not depend
            # on the target's units.
            return np.zeros_like(residual)
        return residual

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"{type(self).__name__}(problem_type={self.problem_type!r})"


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #


class CorrelationScorer(Scorer):
    """Absolute Pearson correlation with the residualised target.

    The cheapest scorer, and the default. Each candidate costs one
    standardisation and one dot product against the residual, so scoring is
    linear in the batch size and handles beam widths in the thousands without
    difficulty.

    The apparent weakness — that Pearson correlation sees only linear
    relationships — matters less here than it would elsewhere. A candidate such
    as ``log(a) / sqrt(b)`` already encodes its nonlinearity in the expression
    itself, so a linear scorer applied to the *transformed* column still
    detects it. What this scorer genuinely misses is nonmonotone dependence
    between a candidate and the residual, such as a candidate tracking the
    residual's magnitude but not its sign.

    Args:
        problem_type: ``"regression"`` or ``"classification"``.
        random_state: Unused; accepted for interface uniformity.
    """

    name = "correlation"

    def score_batch(
        self,
        candidates: np.ndarray,
        target: np.ndarray,
        incumbent: np.ndarray | None = None,
    ) -> np.ndarray:
        candidates, target, incumbent = self._validate(candidates, target, incumbent)
        residual = self._residualise(target, incumbent)

        scaled_residual = _standardise(residual)
        if not np.any(scaled_residual):
            # The incumbent set explains the target completely; nothing to add.
            return np.zeros(candidates.shape[1])

        return _absolute_correlations(candidates, scaled_residual)


def _absolute_correlations(candidates: np.ndarray, scaled_residual: np.ndarray) -> np.ndarray:
    """|Pearson correlation| of each candidate column with a standardised residual."""
    scores = np.empty(candidates.shape[1])
    n_samples = candidates.shape[0]
    for index in range(candidates.shape[1]):
        scaled_candidate = _standardise(candidates[:, index])
        scores[index] = abs(float(scaled_candidate @ scaled_residual) / n_samples)

    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _average_ranks(matrix: np.ndarray) -> np.ndarray:
    """Rank each column from 1, giving tied values the mean of the ranks they span.

    Ordinal ranks would break ties by row order, so a column with ties would
    score differently depending on how the rows happen to be arranged.
    """
    n_rows = matrix.shape[0]
    order = np.argsort(matrix, axis=0, kind="stable")
    ordered = np.take_along_axis(matrix, order, axis=0)
    positions = np.broadcast_to(np.arange(n_rows)[:, None], matrix.shape)
    starts = np.ones(matrix.shape, dtype=bool)
    starts[1:] = ordered[1:] != ordered[:-1]
    ends = np.ones(matrix.shape, dtype=bool)
    ends[:-1] = starts[1:]
    first = np.maximum.accumulate(np.where(starts, positions, 0), axis=0)
    last = np.minimum.accumulate(np.where(ends, positions, n_rows - 1)[::-1], axis=0)[::-1]
    ranks = np.empty(matrix.shape)
    np.put_along_axis(ranks, order, (first + last) / 2.0 + 1.0, axis=0)
    return ranks


# --------------------------------------------------------------------------- #
# Mutual information
# --------------------------------------------------------------------------- #


class MutualInformationScorer(Scorer):
    """Nearest-neighbour mutual information with the residualised target.

    Detects dependence that correlation cannot, including nonmonotone
    relationships. The estimator is the Kraskov-style k-nearest-neighbour
    method from scikit-learn, which is nonparametric but noticeably slower than
    a dot product; expect roughly one to two orders of magnitude more time than
    :class:`CorrelationScorer`.

    For classification, mutual information is computed against the class labels
    directly rather than against a residual, since a least-squares residual of
    a categorical target is not meaningful. This means the classification path
    scores absolute rather than marginal relevance, so redundancy control falls
    to the search layer's redundancy pass (``redundancy_threshold``).

    Args:
        problem_type: ``"regression"`` or ``"classification"``.
        random_state: Seed for the estimator's tie-breaking noise. Fixed by
            default so scores are reproducible.
        n_neighbors: Neighbourhood size for the estimator. Smaller values
            reduce bias and increase variance.

    Cost and variance caveat: k-nearest-neighbour mutual-information
    estimation is orders of magnitude slower than correlation scoring and
    high-variance at moderate sample sizes, so a wide beam over thousands of
    candidates can both blow the time budget and rank on estimator noise.
    Prefer it for narrow searches where non-monotone structure is suspected.
    """

    name = "mutual_information"

    def __init__(
        self,
        problem_type: ProblemType = "regression",
        random_state: int | None = 0,
        n_neighbors: int = 3,
    ) -> None:
        super().__init__(problem_type=problem_type, random_state=random_state)
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be at least 1, got {n_neighbors}")
        self.n_neighbors = n_neighbors

    def score_batch(
        self,
        candidates: np.ndarray,
        target: np.ndarray,
        incumbent: np.ndarray | None = None,
    ) -> np.ndarray:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        candidates, target, incumbent = self._validate(candidates, target, incumbent)

        if candidates.shape[0] <= self.n_neighbors:
            # Too few samples for a meaningful neighbourhood estimate.
            return np.zeros(candidates.shape[1])

        if self.problem_type == "classification":
            reference = target
            estimator = mutual_info_classif
        else:
            reference = self._residualise(target, incumbent)
            estimator = mutual_info_regression
            if not np.any(_standardise(reference)):
                return np.zeros(candidates.shape[1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = estimator(
                candidates,
                reference,
                discrete_features=False,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )

        scores = np.asarray(scores, dtype=np.float64)
        # The estimator can return small negative values from its bias
        # correction; clip rather than propagate them.
        return np.clip(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


# --------------------------------------------------------------------------- #
# Gradient boosting
# --------------------------------------------------------------------------- #


class GradientBoostingScorer(Scorer):
    """Out-of-fold predictive improvement from adding the candidate.

    Fits a small gradient-boosted model on the incumbent set, then on the
    incumbent set plus each candidate, and scores the candidate by the
    improvement in cross-validated performance. This is the only scorer that
    measures the quantity actually being optimised, and correspondingly the
    only one whose ranking cannot be fooled by a statistic that correlates with
    the target without improving prediction.

    The cost is substantial and grows with the candidate count: one model fit
    per candidate per fold, so a wide beam over hundreds of candidates can run
    for minutes where correlation scoring takes milliseconds. It is practical
    only with a narrow beam (roughly 15 or below) and a bounded
    ``subsample_size``; the estimators apply a single scorer throughout the
    search, so there is no cheaper-scorer-first pass to reserve it for.

    Args:
        problem_type: ``"regression"`` or ``"classification"``.
        random_state: Seed for the estimator and fold assignment.
        n_folds: Number of cross-validation folds.
        max_iter: Boosting iterations per fit. Kept low deliberately; this is a
            ranking signal, not a final model.
        max_depth: Tree depth per fit.
        subsample_size: If set, score on a random subsample of this many rows
            rather than the full data. The single most effective way to make
            this scorer affordable.
    """

    name = "gradient_boosting"

    def __init__(
        self,
        problem_type: ProblemType = "regression",
        random_state: int | None = 0,
        n_folds: int = 3,
        max_iter: int = 40,
        max_depth: int = 3,
        subsample_size: int | None = 2000,
    ) -> None:
        super().__init__(problem_type=problem_type, random_state=random_state)
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.subsample_size = subsample_size

    def _make_estimator(self):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        kwargs = {
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "early_stopping": False,
            "random_state": self.random_state,
        }
        if self.problem_type == "classification":
            return HistGradientBoostingClassifier(**kwargs)
        return HistGradientBoostingRegressor(**kwargs)

    def _cv_score(self, features: np.ndarray, target: np.ndarray, folds) -> float:
        """Mean out-of-fold score across folds, or ``-inf`` if unfittable."""
        from sklearn.metrics import accuracy_score, r2_score

        metric = accuracy_score if self.problem_type == "classification" else r2_score
        fold_scores: list[float] = []

        for train_index, test_index in folds:
            if self.problem_type == "classification" and len(np.unique(target[train_index])) < 2:
                continue
            estimator = self._make_estimator()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    estimator.fit(features[train_index], target[train_index])
                    predictions = estimator.predict(features[test_index])
                except (ValueError, RuntimeError):  # pragma: no cover - defensive
                    continue
            fold_scores.append(float(metric(target[test_index], predictions)))

        if not fold_scores:  # pragma: no cover - defensive
            return float("-inf")
        return float(np.mean(fold_scores))

    def score_batch(
        self,
        candidates: np.ndarray,
        target: np.ndarray,
        incumbent: np.ndarray | None = None,
    ) -> np.ndarray:
        from sklearn.model_selection import KFold, StratifiedKFold

        candidates, target, incumbent = self._validate(candidates, target, incumbent)

        # Subsample for affordability, deterministically.
        n_samples = candidates.shape[0]
        if self.subsample_size is not None and n_samples > self.subsample_size:
            rng = np.random.default_rng(self.random_state)
            keep = rng.choice(n_samples, size=self.subsample_size, replace=False)
            keep.sort()
            candidates = candidates[keep]
            target = target[keep]
            if incumbent is not None:
                incumbent = incumbent[keep]

        n_samples = candidates.shape[0]
        n_folds = min(self.n_folds, n_samples)
        if n_folds < 2:
            return np.zeros(candidates.shape[1])

        if self.problem_type == "classification":
            _, counts = np.unique(target, return_counts=True)
            n_folds = min(n_folds, int(counts.min()))
            if n_folds < 2:
                return np.zeros(candidates.shape[1])
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            folds = list(splitter.split(candidates, target))
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            folds = list(splitter.split(candidates))

        # Baseline: performance of the incumbent set alone. With no incumbent
        # the baseline is the trivial predictor, which scores 0 under r2 and
        # majority-rate under accuracy.
        if incumbent is None:
            baseline = 0.0 if self.problem_type == "regression" else float(np.bincount(_as_codes(target)).max() / n_samples)
        else:
            baseline = self._cv_score(incumbent, target, folds)
            if not np.isfinite(baseline):  # pragma: no cover - defensive
                baseline = 0.0

        scores = np.empty(candidates.shape[1])
        for index in range(candidates.shape[1]):
            column = candidates[:, index : index + 1]
            features = column if incumbent is None else np.column_stack([incumbent, column])
            improvement = self._cv_score(features, target, folds) - baseline
            scores[index] = max(0.0, improvement) if np.isfinite(improvement) else 0.0

        return scores


def _as_codes(target: np.ndarray) -> np.ndarray:
    """Map class labels to contiguous integer codes."""
    _, codes = np.unique(target, return_inverse=True)
    return codes


# --------------------------------------------------------------------------- #
# Rank correlation
# --------------------------------------------------------------------------- #


class SpearmanScorer(CorrelationScorer):
    """Rank correlation: Pearson on the rank-transformed candidate and target.

    Invariant to monotone transforms of either variable and insensitive to
    outliers in the tails, so it is the natural choice when the relationship
    is monotone but nonlinear, or when heavy-tailed values would let a few
    extreme rows dominate a product-moment correlation. It shares the
    structural limits of any single-column score: a non-monotone relationship
    (a symmetric bump) and a marginally-quiet-but-jointly-essential feature
    are both invisible to it, exactly as to Pearson.

    Residualisation follows :class:`CorrelationScorer`; ranks are taken after
    residualising, so the score measures monotone association with what the
    incumbent set has not explained. Tied values share their average rank, so
    a score never depends on row order; a constant candidate, or an incumbent
    set that explains the target exactly, scores zero, and a candidate with a
    non-finite value scores zero, as under :class:`CorrelationScorer`.

    For classification only the candidates are ranked. A classification
    residual is a label minus a fitted function of the incumbents, so within
    each class its order is set by the incumbents alone; ranking it would
    turn that order into signal, and a copy of an incumbent would score well.
    Without incumbents the two coincide, since the ranks of a binary residual
    are an affine function of it.

    Leverage caveat: rank transformation hides magnitudes, so a feature with
    a near-singularity (a reciprocal or logarithm evaluated close to its
    pole) can rank-correlate well while carrying extreme values that
    destabilise the downstream least-squares fit. The estimators'
    holdout fit check (:class:`~beamfeat.estimators.DegenerateFitWarning`)
    flags the resulting failures; treat that warning seriously when this
    scorer is selected.
    """

    name = "spearman"

    def score_batch(
        self,
        candidates: np.ndarray,
        target: np.ndarray,
        incumbent: np.ndarray | None = None,
    ) -> np.ndarray:
        candidates, target, incumbent = self._validate(candidates, target, incumbent)
        residual = self._residualise(target, incumbent)
        if not np.all(np.isfinite(residual)):  # pragma: no cover - defensive
            return np.zeros(candidates.shape[1])

        # An exactly explained target leaves a zero residual, whose ranks are
        # all tied: the standardised ranks are then zero, as for Pearson.
        if self.problem_type == "regression":
            residual = _average_ranks(residual.reshape(-1, 1))[:, 0]
        scaled_residual = _standardise(residual)
        if not np.any(scaled_residual):
            return np.zeros(candidates.shape[1])

        scores = _absolute_correlations(_average_ranks(candidates), scaled_residual)
        # Ranks would order an infinite or NaN value like any other; Pearson
        # scores such a column zero, and so does this scorer.
        scores[~np.all(np.isfinite(candidates), axis=0)] = 0.0
        return scores


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


_SCORERS: dict[str, type[Scorer]] = {
    "correlation": CorrelationScorer,
    "spearman": SpearmanScorer,
    "mutual_information": MutualInformationScorer,
    "gradient_boosting": GradientBoostingScorer,
}

_ALIASES = {
    "corr": "correlation",
    "pearson": "correlation",
    "rank": "spearman",
    "mi": "mutual_information",
    "mutual_info": "mutual_information",
    "gb": "gradient_boosting",
    "boosting": "gradient_boosting",
}


def make_scorer(
    scorer: str | Scorer,
    problem_type: ProblemType = "regression",
    random_state: int | None = 0,
    **kwargs,
) -> Scorer:
    """Resolve a scorer name or instance into a :class:`Scorer`.

    Args:
        scorer: A scorer name (``"correlation"``, ``"spearman"``,
            ``"mutual_information"``, ``"gradient_boosting"``, or an alias
            such as ``"mi"`` or ``"rank"``), or an already-constructed
            :class:`Scorer`, which is returned unchanged.
        problem_type: Passed to the constructed scorer. An instance keeps its
            own problem type; a warning is issued when it differs from this
            one.
        random_state: Passed to the constructed scorer.
        **kwargs: Additional scorer-specific arguments.

    Returns:
        A :class:`Scorer` instance.

    Raises:
        ValueError: If ``scorer`` names no known strategy, or
            ``problem_type`` is not recognised.
        TypeError: If ``scorer`` is neither a string nor a :class:`Scorer`.
    """
    problem_type = _check_problem_type(problem_type)
    if isinstance(scorer, Scorer):
        if scorer.problem_type != problem_type:
            warnings.warn(
                f"{type(scorer).__name__} instance has problem_type={scorer.problem_type!r}, but "
                f"{problem_type!r} was requested; the instance is used unchanged. Construct it with "
                f"problem_type={problem_type!r} to score a {problem_type} target as one",
                UserWarning,
                stacklevel=2,
            )
        return scorer
    if not isinstance(scorer, str):
        raise TypeError(f"scorer must be a string or Scorer, got {type(scorer).__name__}")

    key = _ALIASES.get(scorer.lower(), scorer.lower())
    scorer_class = _SCORERS.get(key)
    if scorer_class is None:
        known = sorted(set(_SCORERS) | set(_ALIASES))
        raise ValueError(f"unknown scorer {scorer!r}; choose from {known}")

    return scorer_class(problem_type=problem_type, random_state=random_state, **kwargs)
