"""False-discovery-rate controlled feature selection.

Feature construction proposes far more candidates than any dataset can
support, so selection is where spurious features are either excluded or
silently admitted. This module provides selectors that control the false
discovery rate — the expected proportion of selected features that are
spurious — and is explicit about the assumptions under which each guarantee
holds.

:class:`PermutationSelector` (default)
    A permutation test of marginal association, corrected for
    multiplicity. The test statistic is a fixed function of the data
    (|Pearson correlation| for regression, eta-squared for classification),
    so permuting the target yields p-values valid in finite samples for the null hypothesis
    that a feature is marginally independent of the target (Phipson & Smyth,
    2010). Multiplicity is handled by Benjamini-Hochberg, valid under
    positive regression dependence, or Benjamini-Yekutieli, valid under
    arbitrary dependence (Benjamini & Yekutieli, 2001). This class defaults
    to the former; the estimators default to the latter, since engineered
    candidates share parents and positive dependence cannot be assumed.

    The marginal null composes correctly with constructed features: two
    near-duplicate expressions of one true signal are *both* genuinely
    associated with the target, so selecting both is not a false discovery
    under this null. De-duplication is the job of the redundancy pass, not
    the error-control procedure.

:class:`KnockoffSelector`
    The knockoff filter. Two constructions are provided and routed
    automatically:

    - **Fixed-X** (Barber & Candès, 2015), used when ``n > 2p``. Treats the
      design as fixed and makes *no distributional assumption about the
      features* — deterministic engineered columns are fine. The guarantee
      requires the linear model ``y = X beta + noise`` with Gaussian,
      homoskedastic noise. On near-singular designs the construction remains
      valid but its power degrades toward zero, because the knockoffs become
      nearly identical to the originals.
    - **Model-X Gaussian** (Candès et al., 2018), used when ``n <= 2p``.
      Requires the features to be jointly Gaussian, which engineered
      features are not; a warning is recorded when the design is visibly
      degenerate.

**Selective inference caveat.** Every guarantee above is for testing a
candidate set that is *fixed before seeing the data used for testing*. If
candidates were chosen by a search on the same data — chosen, in part,
because they correlate with this sample's target — the p-values are
optimistically biased and the nominal FDR is not guaranteed. The estimators
in :mod:`beamfeat.estimators` therefore hold out a split of the training
data for selection by default; see ``selection_holdout``.

References:
    Barber, R. F. & Candès, E. J. (2015). Controlling the false discovery
    rate via knockoffs. *Annals of Statistics*, 43(5), 2055-2085.

    Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery
    rate. *JRSS-B*, 57(1), 289-300.

    Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery
    rate in multiple testing under dependency. *Annals of Statistics*,
    29(4), 1165-1188.

    Candès, E., Fan, Y., Janson, L. & Lv, J. (2018). Panning for gold:
    model-X knockoffs. *JRSS-B*, 80(3), 551-577.

    Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be
    zero. *Stat. Appl. Genet. Mol. Biol.*, 9(1), Article 39.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "KnockoffSelector",
    "PermutationSelector",
    "SelectionResult",
    "Selector",
    "knockoff_threshold",
    "make_selector",
]

logger = logging.getLogger(__name__)

ProblemType = Literal["regression", "classification"]
Offset = Literal[0, 1]


@dataclass(slots=True)
class SelectionResult:
    """Outcome of a selection run.

    Attributes:
        selected: Indices of the selected features, ascending.
        statistics: Importance statistic per feature. For the permutation
            selector this is ``1 - p_value`` so that larger is better; for
            knockoffs it is the antisymmetric statistic W.
        p_values: Permutation p-values, valid in finite samples, where the method produces them.
        threshold: The data-dependent cut applied to :attr:`statistics`.
            ``inf`` means nothing met the criterion.
        target_fdr: The nominal FDR level requested.
        n_candidates: Number of features considered.
        method: Name of the selector used.
        warnings_raised: Assumption or configuration issues detected.
    """

    selected: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    statistics: np.ndarray = field(default_factory=lambda: np.empty(0))
    p_values: np.ndarray | None = None
    q_values: np.ndarray | None = None
    threshold: float = float("inf")
    target_fdr: float = 0.1
    n_candidates: int = 0
    method: str = ""
    warnings_raised: list[str] = field(default_factory=list)

    @property
    def n_selected(self) -> int:
        """Number of features selected."""
        return int(self.selected.size)

    def mask(self) -> np.ndarray:
        """Boolean mask over the candidate features."""
        out = np.zeros(self.n_candidates, dtype=bool)
        out[self.selected] = True
        return out

    def summary(self) -> str:
        """Human-readable summary of the run."""
        lines = [
            f"{self.method}: selected {self.n_selected}/{self.n_candidates} "
            f"at target FDR {self.target_fdr:.2f}"
        ]
        lines.extend(f"  warning: {message}" for message in self.warnings_raised)
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.n_selected

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"SelectionResult(n_selected={self.n_selected}/{self.n_candidates}, method={self.method!r})"


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class Selector(ABC):
    """Base class for FDR-controlled selectors."""

    name: str = "selector"

    def __init__(
        self,
        target_fdr: float = 0.1,
        problem_type: ProblemType = "regression",
        random_state: int | None = 0,
    ) -> None:
        if not 0.0 < target_fdr < 1.0:
            raise ValueError(f"target_fdr must be in (0, 1), got {target_fdr}")
        if problem_type not in ("regression", "classification"):
            raise ValueError(f"problem_type must be 'regression' or 'classification', got {problem_type!r}")
        self.target_fdr = float(target_fdr)
        self.problem_type = problem_type
        self.random_state = random_state

    @abstractmethod
    def select(self, features: np.ndarray, target: np.ndarray) -> SelectionResult:
        """Select features controlling the FDR at :attr:`target_fdr`."""

    def _validate(self, features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got {features.ndim}-D")
        if features.shape[1] == 0:
            raise ValueError("features has no columns")
        target = np.asarray(target).ravel()
        if target.shape[0] != features.shape[0]:
            raise ValueError(f"target has {target.shape[0]} rows, features have {features.shape[0]}")
        if not np.all(np.isfinite(features)):
            raise ValueError("features contains non-finite values")
        # Unchecked, a NaN in the target would make every permutation statistic
        # NaN: no null draw would count as an exceedance and every candidate
        # would be selected at the p-value floor. As a class label it would
        # form a class of its own.
        if target.dtype.kind in "fc":
            if not np.all(np.isfinite(target)):
                raise ValueError("target contains non-finite values")
        elif target.dtype.kind == "O" and any(
            value is None or (isinstance(value, float) and not np.isfinite(value)) for value in target
        ):
            raise ValueError("target contains missing or non-finite values")
        return features, target

    def _lasso_importance(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Cross-validated lasso coefficient magnitudes (knockoff statistic)."""
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        n_samples = design.shape[0]
        n_folds = int(min(5, max(2, n_samples // 10)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.problem_type == "classification":
                if len(np.unique(target)) < 2:  # pragma: no cover - defensive
                    return np.zeros(design.shape[1])
                _, counts = np.unique(target, return_counts=True)
                n_folds = int(min(n_folds, counts.min()))
                if n_folds < 2:  # pragma: no cover - defensive
                    return np.zeros(design.shape[1])
                model = LogisticRegressionCV(
                    cv=n_folds, penalty="l1", solver="liblinear", max_iter=2000,
                    random_state=self.random_state,
                )
                model.fit(design, target)
                coefficients = np.max(np.abs(model.coef_), axis=0)
            else:
                model = LassoCV(cv=n_folds, max_iter=5000, random_state=self.random_state)
                model.fit(design, target)
                coefficients = np.abs(model.coef_)
        return np.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"{type(self).__name__}(target_fdr={self.target_fdr}, problem_type={self.problem_type!r})"


def _standardise(features: np.ndarray) -> np.ndarray:
    """Centre and scale each column; constant columns become zero."""
    centred = features - features.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(centred**2, axis=0, keepdims=True))
    scale = np.where(scale < 1e-12, 1.0, scale)
    return np.nan_to_num(centred / scale, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Multiple-testing corrections
# --------------------------------------------------------------------------- #


def _benjamini_hochberg(p_values: np.ndarray, target_fdr: float) -> np.ndarray:
    """Indices surviving the Benjamini-Hochberg (1995) step-up procedure.

    Controls the FDR at ``target_fdr`` when the p-values are independent or
    positively regression dependent (PRDS; Benjamini & Yekutieli, 2001).
    """
    n_tests = p_values.size
    if n_tests == 0:  # pragma: no cover - defensive
        return np.empty(0, dtype=int)
    order = np.argsort(p_values, kind="stable")
    sorted_p = p_values[order]
    critical = target_fdr * np.arange(1, n_tests + 1) / n_tests
    below = np.flatnonzero(sorted_p <= critical)
    if below.size == 0:
        return np.empty(0, dtype=int)
    return order[: below[-1] + 1]


def _adjusted_p_values(p_values: np.ndarray, correction: str) -> np.ndarray:
    """Step-up adjusted p-values ("q-values") for BH or BY.

    ``q_i`` is the smallest level at which feature ``i`` would be selected by
    the given procedure: ``q_(i) = min_{j >= i} c(m) * m * p_(j) / j`` in rank
    order, capped at 1, with ``c(m) = 1`` for Benjamini-Hochberg and the
    harmonic number for Benjamini-Yekutieli. Reported for auditability; the
    selection decision itself uses the step-up procedures directly.
    """
    n_tests = p_values.size
    if n_tests == 0:  # pragma: no cover - defensive
        return p_values.copy()
    scale = 1.0 if correction == "bh" else float(np.sum(1.0 / np.arange(1, n_tests + 1)))
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order] * scale * n_tests / np.arange(1, n_tests + 1)
    monotone = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n_tests)
    adjusted[order] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def _benjamini_yekutieli(p_values: np.ndarray, target_fdr: float) -> np.ndarray:
    """Benjamini-Yekutieli (2001): FDR control under arbitrary dependence.

    Identical to Benjamini-Hochberg with the level divided by the harmonic
    number ``sum(1/i)``; strictly more conservative, but valid without any
    dependence condition on the p-values.
    """
    n_tests = p_values.size
    harmonic = float(np.sum(1.0 / np.arange(1, n_tests + 1))) if n_tests else 1.0
    return _benjamini_hochberg(p_values, target_fdr / harmonic)


# --------------------------------------------------------------------------- #
# Permutation selector
# --------------------------------------------------------------------------- #


class PermutationSelector(Selector):
    """Permutation test of marginal association, with FDR correction.

    **Statistic.** For regression, the absolute Pearson correlation between
    each feature and the target; for classification, eta-squared (the
    between-class share of each feature's variance). Both are fixed functions
    of ``(X_j, y)``, which is the condition for a permutation test to be
    valid: comparing the observed statistic to the same statistic computed on
    ``(X_j, pi(y))`` over random permutations ``pi`` yields a valid p-value
    for the null that ``X_j`` is marginally independent of ``y``, via the
    add-one estimator ``(1 + #exceedances) / (B + 1)`` (Phipson & Smyth,
    2010). Statistics that re-tune themselves on the observed target — a
    cross-validated lasso penalty, for instance — do not satisfy this
    condition unless the tuning is repeated inside every permutation.

    **What the null means for constructed features.** Marginal independence
    is the appropriate null here: an expression that genuinely co-varies with
    the target is a true discovery even if another selected expression
    carries the same information. Redundancy among true discoveries is
    resolved by the redundancy pass in the search, not by the error-control
    procedure.

    **Multiplicity.** Benjamini-Hochberg by default, which controls FDR under
    positive regression dependence; set ``correction="by"`` for the
    Benjamini-Yekutieli procedure, valid under arbitrary dependence at the
    cost of a ``log(m)``-factor in power.

    **Resolution.** The smallest attainable p-value is ``1/(B + 1)``, and the
    leading feature has to reach the threshold its correction sets:
    ``target_fdr / m`` under Benjamini-Hochberg, and
    ``target_fdr / (m c(m))`` under Benjamini-Yekutieli, where
    ``c(m) = sum_{j<=m} 1/j``. ``B`` therefore must be at least
    ``m / target_fdr - 1`` in the first case and ``c(m)`` times that in the
    second for selection to be possible at all, and one further null
    exceedance doubles the requirement. When ``auto_permutations`` is on
    (default), B is raised to twice the bound belonging to the configured
    correction — satisfiable with headroom for a single exceedance — capped
    at ``max_permutations``. The requirement binds hardest when few
    candidates reach the floor together: several tied true signals relax the
    threshold by their count, whereas a single one carries it alone.
    Because the statistic is a chunked matrix product, the larger counts
    remain inexpensive; raising ``B`` ninefold on a 4088-column problem left
    the fit time unchanged within noise.

    Args:
        target_fdr: Nominal false discovery rate.
        problem_type: ``"regression"`` or ``"classification"``.
        random_state: Seed for the permutation draws.
        n_permutations: Minimum number of permutations.
        auto_permutations: Raise B to the satisfiability bound when needed.
        max_permutations: Hard cap on B.
        correction: ``"bh"`` (Benjamini-Hochberg, default) or ``"by"``
            (Benjamini-Yekutieli, arbitrary dependence).
    """

    name = "permutation"

    def __init__(
        self,
        target_fdr: float = 0.1,
        problem_type: ProblemType = "regression",
        random_state: int | None = 0,
        n_permutations: int = 2000,
        auto_permutations: bool = True,
        max_permutations: int = 1_000_000,
        correction: Literal["bh", "by"] = "bh",
    ) -> None:
        super().__init__(target_fdr=target_fdr, problem_type=problem_type, random_state=random_state)
        if n_permutations < 2:
            raise ValueError(f"n_permutations must be at least 2, got {n_permutations}")
        if max_permutations < n_permutations:
            raise ValueError(
                f"max_permutations ({max_permutations}) must be at least n_permutations ({n_permutations})"
            )
        if correction not in ("bh", "by"):
            raise ValueError(f"correction must be 'bh' or 'by', got {correction!r}")
        self.n_permutations = n_permutations
        self.auto_permutations = auto_permutations
        self.max_permutations = max_permutations
        self.correction = correction

    # -- statistics --------------------------------------------------------- #

    @staticmethod
    def _regression_stat(standardised: np.ndarray, target: np.ndarray) -> np.ndarray:
        """|Pearson correlation| per column, given a standardised design."""
        centred = target - target.mean()
        scale = float(np.sqrt(np.mean(centred**2)))
        if scale < 1e-12:
            return np.zeros(standardised.shape[1])
        return np.abs(standardised.T @ (centred / scale)) / standardised.shape[0]

    @staticmethod
    def _classification_stat(standardised: np.ndarray, codes: np.ndarray, n_classes: int) -> np.ndarray:
        """Eta-squared per column: between-class variance over total variance.

        Computed on a standardised design, total variance is 1 per column, so
        eta-squared reduces to the class-size-weighted mean of squared class
        means.
        """
        n_samples = standardised.shape[0]
        stat = np.zeros(standardised.shape[1])
        for klass in range(n_classes):
            members = codes == klass
            n_members = int(members.sum())
            if n_members == 0:
                continue
            class_means = standardised[members].mean(axis=0)
            stat += (n_members / n_samples) * class_means**2
        return stat

    def _required_permutations(self, n_features: int) -> int:
        """Permutations for the configured correction to be satisfiable.

        Satisfiability alone needs ``1/(B+1) <= t``, where ``t`` is the
        threshold the leading p-value has to clear. Under
        Benjamini-Hochberg that threshold is ``target_fdr/m``, giving
        ``B >= m/target_fdr - 1``. Under Benjamini-Yekutieli it is
        ``target_fdr/(m c(m))`` with ``c(m) = sum_{j<=m} 1/j``, so the same
        argument requires ``c(m)`` times as many draws; at m in the hundreds
        that factor is already above five, and using the Benjamini-Hochberg
        bound under Benjamini-Yekutieli leaves the correction unsatisfiable
        whenever few candidates reach the floor together.

        At either exact bound a single null draw exceeding the top feature
        makes its p-value ``2/(B+1)`` and selection impossible again, so
        twice the bound is used: enough for one exceedance.
        """
        scale = 1.0
        if self.correction == "by":
            scale = float(np.sum(1.0 / np.arange(1, n_features + 1)))
        return int(np.ceil(2.0 * scale * n_features / self.target_fdr))

    def select(self, features: np.ndarray, target: np.ndarray) -> SelectionResult:
        features, target = self._validate(features, target)
        n_samples, n_features = features.shape
        rng = np.random.default_rng(self.random_state)
        messages: list[str] = []

        n_permutations = self.n_permutations
        required = self._required_permutations(n_features)
        if self.auto_permutations and n_permutations < required:
            if required <= self.max_permutations:
                n_permutations = required
            else:
                n_permutations = self.max_permutations
                messages.append(
                    f"{n_features} features at target FDR {self.target_fdr} need "
                    f"~{required} permutations for {self.correction.upper()} to be satisfiable, above "
                    f"max_permutations={self.max_permutations}; selection may be impossible. "
                    "Reduce the candidate count, raise target_fdr, raise max_permutations, "
                    "or select with correction='bh', whose bound is smaller by the "
                    "harmonic factor at the cost of assuming positive dependence"
                )
                logger.warning("[beamfeat.selection] %s", messages[-1])
        elif not self.auto_permutations and n_permutations < required:
            messages.append(
                f"n_permutations={n_permutations} with auto_permutations=False is below the "
                f"~{required} that {self.correction.upper()} needs for {n_features} features at "
                f"target FDR {self.target_fdr}; selection may be impossible, and an empty "
                "selection would reflect the budget rather than an absent signal. Raise "
                "n_permutations or set auto_permutations=True"
            )
            logger.warning("[beamfeat.selection] %s", messages[-1])

        standardised = _standardise(features)

        if self.problem_type == "classification":
            classes, codes = np.unique(target, return_inverse=True)
            n_classes = len(classes)
            observed = self._classification_stat(standardised, codes, n_classes)

            null_counts = np.zeros(n_features)
            for _ in range(n_permutations):
                permuted = rng.permutation(codes)
                null_counts += self._classification_stat(standardised, permuted, n_classes) >= observed
        else:
            numeric = target.astype(np.float64)
            observed = self._regression_stat(standardised, numeric)

            # Chunked so B x n permutation matrices never get large; each chunk
            # is a single (m x n) @ (n x chunk) product.
            centred = numeric - numeric.mean()
            scale = float(np.sqrt(np.mean(centred**2)))
            null_counts = np.zeros(n_features)
            if scale < 1e-12:
                null_counts[:] = n_permutations  # constant target: p-values 1
            else:
                unit = centred / scale
                chunk_size = 512
                remaining = n_permutations
                while remaining > 0:
                    current = min(chunk_size, remaining)
                    permuted = np.empty((n_samples, current))
                    for column in range(current):
                        permuted[:, column] = rng.permutation(unit)
                    null_stats = np.abs(standardised.T @ permuted) / n_samples
                    null_counts += np.sum(null_stats >= observed[:, None], axis=1)
                    remaining -= current

        # Add-one estimator: valid in finite samples, and p-values can never be zero.
        p_values = (null_counts + 1.0) / (n_permutations + 1.0)

        correct = _benjamini_hochberg if self.correction == "bh" else _benjamini_yekutieli
        selected = correct(p_values, self.target_fdr)
        q_values = _adjusted_p_values(p_values, self.correction)

        statistics = 1.0 - p_values
        threshold = float(np.min(statistics[selected])) if selected.size else float("inf")

        return SelectionResult(
            selected=np.sort(selected),
            statistics=statistics,
            p_values=p_values,
            q_values=q_values,
            threshold=threshold,
            target_fdr=self.target_fdr,
            n_candidates=n_features,
            method=self.name,
            warnings_raised=messages,
        )


# --------------------------------------------------------------------------- #
# Knockoff threshold
# --------------------------------------------------------------------------- #


def knockoff_threshold(statistics: np.ndarray, target_fdr: float, offset: Offset = 1) -> float:
    """Data-dependent threshold of the knockoff filter.

    Finds the smallest positive cut ``t`` at which
    ``(offset + #{W_j <= -t}) / #{W_j >= t} <= target_fdr``. ``offset=1``
    gives knockoff+ (finite-sample FDR control); ``offset=0`` controls a
    modified FDR.
    """
    if offset not in (0, 1):
        raise ValueError(f"offset must be 0 or 1, got {offset}")
    statistics = np.asarray(statistics, dtype=np.float64)
    positive = statistics[statistics > 0]
    if positive.size == 0:
        return float("inf")
    for cut in np.sort(np.unique(positive)):
        n_negative = int(np.sum(statistics <= -cut))
        n_positive = int(np.sum(statistics >= cut))
        if n_positive == 0:  # pragma: no cover - defensive
            continue
        if (offset + n_negative) / n_positive <= target_fdr:
            return float(cut)
    return float("inf")


# --------------------------------------------------------------------------- #
# Knockoff selector
# --------------------------------------------------------------------------- #


class KnockoffSelector(Selector):
    """Knockoff filter with automatic fixed-X / model-X routing.

    **Fixed-X** (Barber & Candès, 2015) is used when ``n > 2p``. The design
    is treated as fixed, so no assumption about the distribution of the
    features is required — deterministic engineered columns are admissible.
    The finite-sample FDR guarantee (with ``offset=1``) requires the linear
    model ``y = X beta + eps`` with i.i.d. Gaussian noise. On a near-singular
    design the construction is still valid but the equicorrelated ``s``
    shrinks toward zero, the knockoffs become nearly identical to the
    originals, and power degrades toward zero; a warning records this.

    **Model-X Gaussian** (Candès et al., 2018) is used when ``n <= 2p``, where
    the fixed-X construction does not exist. It requires the features to be
    jointly Gaussian, an assumption engineered features violate; a warning is
    recorded when the design is visibly degenerate. Prefer
    :class:`PermutationSelector` in that regime.

    **Power and the offset.** ``offset=1`` requires
    ``(1 + #negatives)/#positives <= target_fdr``, unsatisfiable with fewer
    than ``1/target_fdr`` features regardless of signal; measured on a
    25-feature Gaussian design at level 0.1, ``offset=1`` recovered almost
    nothing while ``offset=0`` recovered everything. ``offset=1`` remains the
    default because it is the stated finite-sample guarantee; a warning fires
    when the design is too narrow for it.

    Args:
        target_fdr: Nominal false discovery rate.
        problem_type: ``"regression"`` or ``"classification"``. The fixed-X
            guarantee is stated for regression; classification uses the same
            machinery heuristically and records a warning.
        random_state: Seed (used by the model-X sampler and the orthogonal
            complement basis).
        offset: ``1`` for knockoff+ (default), ``0`` for the modified-FDR
            filter.
        shrinkage: Ridge term added to the covariance diagonal for numerical
            stability.
        construction: ``"auto"`` (default), ``"fixed"``, or ``"gaussian"``.
        warn_on_violation: Whether to check assumptions and record findings.
    """

    name = "knockoff"

    def __init__(
        self,
        target_fdr: float = 0.1,
        problem_type: ProblemType = "regression",
        random_state: int | None = 0,
        offset: Offset = 1,
        shrinkage: float = 1e-6,
        construction: Literal["auto", "fixed", "gaussian"] = "auto",
        warn_on_violation: bool = True,
    ) -> None:
        super().__init__(target_fdr=target_fdr, problem_type=problem_type, random_state=random_state)
        if offset not in (0, 1):
            raise ValueError(f"offset must be 0 or 1, got {offset}")
        if shrinkage < 0:
            raise ValueError(f"shrinkage must be non-negative, got {shrinkage}")
        if construction not in ("auto", "fixed", "gaussian"):
            raise ValueError(f"construction must be 'auto', 'fixed', or 'gaussian', got {construction!r}")
        self.offset = offset
        self.shrinkage = shrinkage
        self.construction = construction
        self.warn_on_violation = warn_on_violation

    # -- shared pieces ------------------------------------------------------ #

    def _covariance(self, standardised: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Return (Sigma, min eigenvalue, Sigma^-1) with shrinkage applied."""
        n_samples, n_features = standardised.shape
        raw_covariance = standardised.T @ standardised / n_samples
        # The reported eigenvalue is the raw one: shrinkage exists to make the
        # inverse computable, and folding it into the diagnostic would make
        # the near-singularity check unsatisfiable (min eigenvalue could never
        # drop below the shrinkage itself).
        smallest = max(float(np.min(np.linalg.eigvalsh(raw_covariance))), 0.0)
        covariance = raw_covariance + self.shrinkage * np.eye(n_features)
        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            precision = np.linalg.pinv(covariance)
        return covariance, smallest, precision

    @staticmethod
    def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
        """Symmetric PSD square root via eigendecomposition, clipping noise."""
        symmetric = (matrix + matrix.T) / 2.0
        values, vectors = np.linalg.eigh(symmetric)
        return vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None))) @ vectors.T

    def _check_common(self, smallest: float, n_features: int) -> list[str]:
        messages: list[str] = []
        if self.problem_type == "classification":
            messages.append(
                "knockoff FDR guarantees are stated for the Gaussian linear model; "
                "the classification path is heuristic"
            )
        # knockoff+ needs (1 + #negatives) / #positives <= q, so at least
        # ceil(1/q) positive statistics; the tolerance keeps 1/0.1 at 10.
        minimum = int(np.ceil(1.0 / self.target_fdr - 1e-9))
        if self.offset == 1 and n_features < minimum:
            messages.append(
                f"offset=1 (knockoff+) with {n_features} features at target FDR "
                f"{self.target_fdr}: selection requires at least {minimum} features "
                "to be satisfiable; consider offset=0 or a higher target_fdr"
            )
        if smallest < 1e-6:
            messages.append(
                f"design covariance is near-singular (min eigenvalue {smallest:.2e}); "
                "the knockoff construction remains valid for fixed-X but its power "
                "degrades toward zero, since knockoffs become nearly identical to the "
                "originals"
            )
        return messages

    # -- constructions ------------------------------------------------------ #

    def _fixed_x_knockoffs(self, standardised: np.ndarray) -> np.ndarray:
        """Fixed-X equicorrelated knockoffs (Barber & Candès, 2015).

        Requires ``n >= 2p + 1``. Constructs ``X~ = X (I - Sigma^-1 S) + U C``
        where ``S = s I`` with ``s = min(2 lambda_min, 1)``, ``U`` is an
        orthonormal basis of a p-dimensional subspace orthogonal to the
        column span of ``X`` and to the constant vector, and
        ``C' C = 2 S - S Sigma^-1 S``. The result satisfies ``X~' X~ = X' X``
        and ``X' X~ = X' X - S``, which is the exchangeability property the
        filter's guarantee rests on.

        ``U`` is kept orthogonal to the constant vector so the knockoffs are
        centred like the standardised design. The lasso that computes the
        statistic fits an intercept, which centres every column; knockoffs
        with non-zero means would lose part of their norm to that centring
        and the identities above would no longer hold for what the lasso
        sees. That extra direction is why one more row than ``2p`` is needed.
        """
        n_samples, n_features = standardised.shape
        covariance, smallest, precision = self._covariance(standardised)
        s_value = min(2.0 * smallest, 1.0)
        s_matrix = s_value * np.eye(n_features)

        part_parallel = standardised - standardised @ precision @ s_matrix

        gram_c = 2.0 * s_matrix - s_matrix @ precision @ s_matrix
        c_factor = self._matrix_sqrt(gram_c)

        # Orthonormal basis of a p-dimensional subspace orthogonal to col(X)
        # and to the constant vector (see the docstring).
        # Any valid basis yields a valid knockoff copy; the seed only fixes
        # which one. Two robustness measures matter here:
        #
        # 1. The Gaussian draw uses a seed sequence spawned from random_state
        #    rather than random_state itself, so it cannot collide with a data
        #    matrix generated from the same seed elsewhere. A draw that is
        #    (nearly) a linear image of X projects to a rank-deficient
        #    residual, and QR of a rank-deficient matrix pads the missing
        #    directions arbitrarily — including back inside col(X).
        # 2. After QR, orthogonality to col(X) is re-imposed by a second
        #    projection and re-orthonormalisation, then verified. This turns a
        #    silent geometry failure into an explicit error.
        rng = np.random.default_rng(np.random.SeedSequence(self.random_state).spawn(1)[0])
        q_x, _ = np.linalg.qr(np.column_stack([np.ones(n_samples), standardised]))
        raw = rng.standard_normal((n_samples, n_features))
        raw -= q_x @ (q_x.T @ raw)
        u_basis, _ = np.linalg.qr(raw)
        u_basis -= q_x @ (q_x.T @ u_basis)
        u_basis, _ = np.linalg.qr(u_basis)
        residual = float(np.abs(q_x.T @ u_basis).max())
        if residual > 1e-8:  # pragma: no cover - requires adversarial input
            raise np.linalg.LinAlgError(
                f"could not construct a complement basis orthogonal to the design "
                f"(residual {residual:.2e}); the design may be numerically rank-deficient"
            )

        # Scaling: the Gram identities are stated at the Sigma = X'X/n
        # convention. With U orthonormal, (aU C)'(aU C)/n = (a^2/n) C'C, so
        # a = sqrt(n) makes the complement term contribute exactly C'C,
        # giving X~'X~/n = Sigma and X'X~/n = Sigma - S.
        return part_parallel + np.sqrt(n_samples) * (u_basis @ c_factor)

    def _model_x_knockoffs(self, standardised: np.ndarray) -> np.ndarray:
        """Gaussian model-X knockoffs by conditional sampling."""
        n_samples, n_features = standardised.shape
        covariance, smallest, precision = self._covariance(standardised)
        s_value = min(2.0 * max(smallest, 1e-10), 1.0)
        s_diag = np.full(n_features, s_value)

        conditional_mean = standardised - standardised @ precision @ np.diag(s_diag)
        conditional_cov = 2.0 * np.diag(s_diag) - np.diag(s_diag) @ precision @ np.diag(s_diag)
        factor = self._matrix_sqrt(conditional_cov)

        # A seed spawned from random_state, as for the fixed-X basis: drawn from
        # random_state itself, the noise would equal a data matrix generated
        # from the same seed and shape, and the knockoffs would be a fixed
        # linear image of the features instead of an independent copy.
        rng = np.random.default_rng(np.random.SeedSequence(self.random_state).spawn(1)[0])
        noise = rng.standard_normal((n_samples, n_features))
        return conditional_mean + noise @ factor.T

    # -- selection ---------------------------------------------------------- #

    def select(self, features: np.ndarray, target: np.ndarray) -> SelectionResult:
        features, target = self._validate(features, target)
        n_samples, n_features = features.shape
        standardised = _standardise(features)

        use_fixed = self.construction == "fixed" or (
            self.construction == "auto" and n_samples > 2 * n_features
        )
        if self.construction == "fixed" and n_samples <= 2 * n_features:
            raise ValueError(
                f"fixed-X knockoffs require n >= 2p + 1 (a complement orthogonal to the "
                f"p columns and the intercept); got n={n_samples}, p={n_features}"
            )

        _, smallest, _ = self._covariance(standardised)
        messages = self._check_common(smallest, n_features) if self.warn_on_violation else []
        if not use_fixed and self.warn_on_violation:
            messages.append(
                "n <= 2p: falling back to model-X Gaussian knockoffs, which assume "
                "jointly Gaussian features — an assumption engineered features "
                "violate; prefer the permutation selector in this regime"
            )
        for message in messages:
            logger.warning("[beamfeat.selection] %s", message)

        knockoffs = self._fixed_x_knockoffs(standardised) if use_fixed else self._model_x_knockoffs(standardised)

        design = np.column_stack([standardised, knockoffs])
        importance = self._lasso_importance(design, target)
        statistics = importance[:n_features] - importance[n_features:]

        threshold = knockoff_threshold(statistics, self.target_fdr, self.offset)
        selected = np.flatnonzero(statistics >= threshold) if np.isfinite(threshold) else np.empty(0, dtype=int)

        if self.offset == 1 and selected.size == 0 and self.warn_on_violation:
            # Distinguish "no evidence" from "the knockoff+ threshold could not
            # have selected at this configuration": report what offset=0 (the
            # modified-FDR filter) would have done, without applying it.
            alternative = knockoff_threshold(statistics, self.target_fdr, offset=0)
            if np.isfinite(alternative):
                n_alternative = int(np.sum(statistics >= alternative))
                messages.append(
                    f"knockoff+ (offset=1) selected nothing at target FDR {self.target_fdr}, "
                    f"but offset=0 — which controls only a modified FDR — would have "
                    f"selected {n_alternative} feature(s). If that trade-off is acceptable, "
                    "pass KnockoffSelector(offset=0) as the selector; the permutation "
                    "selector is the recommended default for engineered candidates"
                )
                logger.warning("[beamfeat.selection] %s", messages[-1])

        return SelectionResult(
            selected=np.sort(selected),
            statistics=statistics,
            p_values=None,
            threshold=threshold,
            target_fdr=self.target_fdr,
            n_candidates=n_features,
            method=self.name,
            warnings_raised=messages,
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


_SELECTORS: dict[str, type[Selector]] = {
    "knockoff": KnockoffSelector,
    "permutation": PermutationSelector,
}

_ALIASES = {"knockoffs": "knockoff", "model_x": "knockoff", "fixed_x": "knockoff", "perm": "permutation"}
# Aliases that name a knockoff construction select it, unless one is passed explicitly.
_ALIAS_KWARGS = {"model_x": {"construction": "gaussian"}, "fixed_x": {"construction": "fixed"}}


def make_selector(
    selector: str | Selector,
    target_fdr: float = 0.1,
    problem_type: ProblemType = "regression",
    random_state: int | None = 0,
    **kwargs,
) -> Selector:
    """Resolve a selector name or instance into a :class:`Selector`.

    An instance is returned as it is: its own settings (``target_fdr``
    among them) apply, and the other arguments here are not used.
    """
    if isinstance(selector, Selector):
        return selector
    if not isinstance(selector, str):
        raise TypeError(f"selector must be a string or Selector, got {type(selector).__name__}")
    key = _ALIASES.get(selector.lower(), selector.lower())
    kwargs = {**_ALIAS_KWARGS.get(selector.lower(), {}), **kwargs}
    selector_class = _SELECTORS.get(key)
    if selector_class is None:
        known = sorted(set(_SELECTORS) | set(_ALIASES))
        raise ValueError(f"unknown selector {selector!r}; choose from {known}")
    return selector_class(target_fdr=target_fdr, problem_type=problem_type, random_state=random_state, **kwargs)
