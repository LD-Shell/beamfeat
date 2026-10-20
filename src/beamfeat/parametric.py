"""Fitting the constant of a parametric operator.

A parametric operator, such as ``exp_scaled`` (``exp(a*x)``) or ``inv_shift``
(``1/(x + a)``), carries one constant. :func:`fit_constants` chooses it for a
given operand by a one-dimensional search: a coarse grid from the operator's
:attr:`~beamfeat.expression.OperatorSpec.constant_grid`, then a golden-section
refinement between the grid neighbours of the best point.

**Objective.** The constant maximises the absolute Pearson correlation of
the transformed operand with the target, whatever scorer the search uses.
Correlation is smooth in the constant and costs one pass over the rows, so
the fit stays cheap for any scorer; the configured scorer then ranks the
fitted node against every other candidate. Correlation is also what the
final linear model rewards: a feature with correlation ``r`` leaves a
fraction ``1 - r^2`` of the target's variance unexplained on its own.

A constant is often only identifiable in context: ``1/(x0*x1 + a)`` is the
denominator of ``(x0 + x1)/(1 + x0*x1)``, and alone it correlates best with
the target as ``a`` approaches the pole. When the caller passes *partners*
(expressions the result may be multiplied by), the fit is therefore also run
on the product with each, and the constant best for the best partner is
returned beside the one best alone. :class:`~beamfeat.search.BeamSearch` proposes the node with each
constant, and the product with its partner, as candidates to be scored like
any other.

**Rejection.** A fit whose best grid point is the end of a grid segment ran
into a bound -- the margin a shift keeps between its pole and the data (see
:func:`~beamfeat.expression.pole_margins`), the smallest positive power
when the operand comes within that margin of 0, the pole of a negative
power, the linear limit of a small scale, or the step limit of a large
one -- and is discarded, as is one whose values are not finite, constant,
or outside the operator's domain. Snapping and rounding stay within the
same bounds, so a simple constant never moves a pole closer to the data
than the margin. And since each
operator tends to an affine function of its operand at one end of its range
(a small scale, a distant shift, a power of 1), a fit must also improve
clearly on the operand itself, or for a product on the partner alone, by
the likelihood-ratio test below; otherwise the constant only fits noise.

**Snapping.** The data rarely pin a constant to more digits than they
support, and ``exp(-0.49991*x^2)`` is a worse answer than ``exp(-x^2/2)``.
After the refinement, the simple values of :data:`SNAP_VALUES` (0, +-1,
+-2, +-3, +-1/2, +-3/2, +-1/3, +-1/4) are tried, and the simplest one the
data cannot tell apart from the optimum is used: the smallest denominator,
then the best fit. "Cannot tell apart" is a likelihood-ratio test for a
least-squares fit of the target on the feature: with ``L(a) = 1 - r(a)^2``
the unexplained fraction and ``n`` the rows, the simple constant is accepted
when ``n * log(L(simple) / L(optimum))`` is at most :data:`SNAP_LR`, the
99.9% point of a chi-squared variable with one degree of freedom. A
tolerance stated this way scales with the noise and the sample: on
near-noiseless data it admits only a value that fits essentially as well
as the optimum, on noisy data a wider range, as it should. A fixed
tolerance on ``|r|`` cannot do both; at 0.1% noise, ``r`` is within 1e-6 of
1 and a tolerance of 1e-4 would accept a constant that leaves a hundred
times the noise unexplained. The 99.9% point rather than the 95% one
makes the simpler constant the default: the true constant of a law is
rejected in one fit in a thousand rather than one in twenty. Failing a
snap, the constant is rounded to the fewest significant digits (up to six)
that cost at most :data:`ROUND_LR`, so that the printed constant is the
stored one.

Everything here is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from beamfeat.expression import OperatorSpec

__all__ = ["ROUND_LR", "SNAP_LR", "SNAP_VALUES", "FittedConstant", "fit_constants"]

#: Simple constants tried after a fit, simplest first: zero, the integers to
#: 3, halves, thirds and quarters. Among the values the data accept, one with
#: a smaller denominator is preferred, then the one that fits best.
SNAP_VALUES: tuple[Fraction, ...] = tuple(
    Fraction(sign * numerator, denominator)
    for denominator, numerators in ((1, (0, 1, 2, 3)), (2, (1, 3)), (3, (1,)), (4, (1,)))
    for numerator in numerators
    for sign in ((1,) if numerator == 0 else (1, -1))
)

#: Largest likelihood-ratio statistic, ``n * log(L(simple) / L(optimum))``,
#: at which a simple constant replaces the fitted optimum: the 99.9% point of
#: a chi-squared variable with one degree of freedom.
SNAP_LR = 10.83

#: Largest likelihood-ratio statistic at which the fitted constant is rounded
#: to fewer significant digits. A hundredth of a nat is far below what one
#: fitted parameter can resolve, so rounding never changes the fit visibly.
ROUND_LR = 0.01

# Unexplained fractions below this are equal: Pearson correlation is not
# resolvable to more than about twelve digits of 1 - r^2 in float64.
_LOSS_FLOOR = 1e-12

# Rows used for the coarse grid. The grid only brackets the optimum, which is
# then refined on every row, so a subsample of large data is enough.
_GRID_ROWS = 2000

# Golden-section steps stop once the bracket is this small relative to the
# constant, well past the six significant digits a constant is rounded to.
_REFINE_RTOL = 1e-10
_INVERSE_PHI = (math.sqrt(5.0) - 1.0) / 2.0


class FittedConstant(NamedTuple):
    """A constant returned by :func:`fit_constants`.

    Attributes:
        value: The constant.
        partner: Column of ``partners`` whose product with the operator the
            constant was fitted for, or ``None`` for the operator alone.
    """

    value: float
    partner: int | None


def fit_constants(
    spec: OperatorSpec,
    operand: np.ndarray,
    target: np.ndarray,
    partners: np.ndarray | None = None,
) -> list[FittedConstant]:
    """Fit the constant of a parametric operator to one operand.

    Args:
        spec: A parametric operator (one with a ``constant_grid``).
        operand: The operand's values on the search rows.
        target: The target, centred and scaled to unit variance, as
            :meth:`~beamfeat.scoring.CorrelationScorer.prepare` returns it.
        partners: Optional ``(n_rows, n_partners)`` matrix of expressions the
            result may be multiplied by later. When given, the constant that
            best fits the product with the best partner is returned as well.

    Returns:
        At most two fitted constants: the one best for the operator alone,
        then the one best in a product, each omitted when its fit ran into
        a bound or produced no valid values. The two may have equal values.
    """
    if not spec.parametric:
        raise ValueError(f"operator {spec.name!r} has no constant to fit")
    operand = np.asarray(operand, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    segments = [np.asarray(s, dtype=np.float64) for s in spec.constant_grid(operand) if len(s) >= 3]
    if not segments:
        return []

    sample = _grid_rows(len(operand))
    grid = np.concatenate(segments)
    # Each grid point's position in its segment, to recognise a segment's ends.
    ends = np.concatenate([np.r_[True, np.zeros(len(s) - 2, dtype=bool), True] for s in segments])
    columns = _grid_columns(spec, operand[sample], grid)
    # The segments' ends bound the constant, snapped or rounded too.
    bounds = [(float(np.min(s)), float(np.max(s))) for s in segments]

    def admissible(constant: float) -> bool:
        return any(low <= constant <= high for low, high in bounds)

    found: list[FittedConstant] = []
    solo = _correlations(columns, target[sample], None)
    constant = _fit_one(
        spec, operand, target, None, grid, ends, solo[0], _correlation(operand, target), admissible
    )
    if constant is not None:
        found.append(FittedConstant(constant, None))

    if partners is not None and partners.shape[1]:
        partners = np.asarray(partners, dtype=np.float64)
        scores = _correlations(columns, target[sample], partners[sample])
        if np.isfinite(scores).any():
            best = int(np.argmax(np.where(np.isfinite(scores), scores, -1.0).max(axis=1)))
            partner = partners[:, best]
            alone = _correlation(partner, target)
            constant = _fit_one(spec, operand, target, partner, grid, ends, scores[best], alone, admissible)
            if constant is not None:
                found.append(FittedConstant(constant, best))
    return found


def _grid_rows(n_rows: int) -> np.ndarray:
    """Evenly spaced rows for the coarse grid: all of them, up to a limit."""
    if n_rows <= _GRID_ROWS:
        return np.arange(n_rows)
    return np.linspace(0, n_rows - 1, _GRID_ROWS).round().astype(np.intp)


def _grid_columns(spec: OperatorSpec, operand: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The operator applied with each grid constant, one column per constant.

    A column is NaN where the evaluator would reject the result: outside the
    operator's domain, or on a floating-point error, underflow included.
    """
    columns = np.full((len(operand), len(grid)), np.nan)
    for index, constant in enumerate(grid):
        values = _apply(spec, operand, float(constant))
        if values is not None:
            columns[:, index] = values
    return columns


def _apply(spec: OperatorSpec, operand: np.ndarray, constant: float) -> np.ndarray | None:
    """The operator's values with ``constant``, or None where the evaluator rejects them."""
    if spec.domain is not None and not spec.domain(operand, constant):
        return None
    with np.errstate(all="raise"):
        try:
            values = np.asarray(spec.fn(operand, constant), dtype=np.float64)
        except FloatingPointError:
            return None
    if not np.all(np.isfinite(values)):
        return None
    return values


def _correlations(columns: np.ndarray, target: np.ndarray, partners: np.ndarray | None) -> np.ndarray:
    """|corr| of each column, or of each partner times each column, with the target.

    Returns a ``(n_partners, n_columns)`` matrix, one row when ``partners``
    is None. Invalid and constant columns score NaN. The products are never
    formed: their moments come from three matrix products, which is accurate
    enough to bracket an optimum, and the refinement recomputes directly.
    """
    n_rows = columns.shape[0]
    valid = np.isfinite(columns).all(axis=0)
    filled = np.where(valid, columns, 0.0)
    factors = np.ones((n_rows, 1)) if partners is None else partners
    with np.errstate(all="ignore"):
        mean = factors.T @ filled / n_rows
        second = (factors**2).T @ filled**2 / n_rows
        covariance = (factors * target[:, None]).T @ filled / n_rows
        variance = second - mean**2
        correlation = np.abs(covariance) / np.sqrt(variance)
    usable = valid[None, :] & np.isfinite(correlation) & (variance > 1e-12 * second)
    return np.where(usable, np.minimum(correlation, 1.0), np.nan)


def _correlation(values: np.ndarray, target: np.ndarray) -> float:
    """|corr| of a column with the standardised target, NaN for a constant column."""
    centred = values - values.mean()
    spread = math.sqrt(float(np.mean(centred * centred)))
    if not math.isfinite(spread) or spread <= 1e-12 * float(np.mean(np.abs(values))) or spread == 0.0:
        return math.nan
    return min(abs(float(centred @ target)) / (len(values) * spread), 1.0)


def _fit_one(
    spec: OperatorSpec,
    operand: np.ndarray,
    target: np.ndarray,
    partner: np.ndarray | None,
    grid: np.ndarray,
    ends: np.ndarray,
    grid_scores: np.ndarray,
    baseline: float,
    admissible: Callable[[float], bool],
) -> float | None:
    """Refine, then snap or round, the best grid constant for one objective.

    ``admissible`` says whether a constant lies within the grid's bounds;
    snapping and rounding never leave them.

    ``baseline`` is the correlation the fit has to beat: the operand's own,
    or the partner's for a product. Every operator here tends to an affine
    function of its operand at one end of its range (a small scale, a far
    shift, a power of 1), so a constant that does not improve on the operand
    clearly, by the likelihood-ratio test of the snap, only fits noise; the
    fit is then rejected. Returns the constant, or ``None``.
    """
    if not np.isfinite(grid_scores).any():
        return None
    best = int(np.nanargmax(grid_scores))
    if ends[best]:
        return None

    def objective(constant: float) -> float:
        values = _apply(spec, operand, constant)
        if values is None:
            return math.nan
        if partner is not None:
            values = values * partner
        return _correlation(values, target)

    # An interior grid point's neighbours lie in its own segment.
    optimum, score = _golden_max(objective, float(grid[best - 1]), float(grid[best + 1]))
    best_grid_score = objective(float(grid[best]))
    if not math.isfinite(score) or (math.isfinite(best_grid_score) and best_grid_score > score):
        optimum, score = float(grid[best]), best_grid_score
    if not math.isfinite(score):
        return None
    constant = _simplify(objective, optimum, score, len(operand), admissible)
    if _statistic(baseline, objective(constant), len(operand)) <= SNAP_LR:
        return None
    return constant


def _golden_max(objective: Callable[[float], float], low: float, high: float) -> tuple[float, float]:
    """Golden-section search for the maximum of ``objective`` on ``[low, high]``.

    NaN counts as worse than any value. Returns the best point evaluated and
    its value.
    """

    def value(point: float) -> float:
        result = objective(point)
        return result if math.isfinite(result) else -math.inf

    inner_low = high - _INVERSE_PHI * (high - low)
    inner_high = low + _INVERSE_PHI * (high - low)
    value_low, value_high = value(inner_low), value(inner_high)
    tolerance = _REFINE_RTOL * max(abs(low), abs(high), 1e-300)
    while high - low > tolerance:
        if value_low >= value_high:
            high, inner_high, value_high = inner_high, inner_low, value_low
            inner_low = high - _INVERSE_PHI * (high - low)
            value_low = value(inner_low)
        else:
            low, inner_low, value_low = inner_low, inner_high, value_high
            inner_high = low + _INVERSE_PHI * (high - low)
            value_high = value(inner_high)
    if value_low >= value_high:
        return inner_low, value_low
    return inner_high, value_high


def _statistic(simple: float, better: float, n_rows: int) -> float:
    """Likelihood-ratio statistic of a fit with correlation ``better`` over ``simple``.

    ``n * log(L(simple) / L(better))``, with ``L = 1 - r^2`` floored at
    :data:`_LOSS_FLOOR`; infinite when ``simple`` is not a valid fit.
    """
    if not math.isfinite(simple):
        return math.inf
    loss = max(1.0 - simple * simple, _LOSS_FLOOR)
    best = max(1.0 - better * better, _LOSS_FLOOR)
    return n_rows * math.log(loss / best)


def _simplify(
    objective: Callable[[float], float],
    optimum: float,
    score: float,
    n_rows: int,
    admissible: Callable[[float], bool] = lambda constant: True,
) -> float:
    """The simplest constant the data cannot tell apart from ``optimum``.

    Tries :data:`SNAP_VALUES` against :data:`SNAP_LR`, then rounding to one to
    six significant digits against :data:`ROUND_LR`; see the module notes.
    Only constants ``admissible`` accepts are tried: a snap must not carry a
    pole past its margin into the data, or a scale past the grid's bound.
    """

    def statistic(constant: float) -> float:
        return _statistic(objective(constant), score, n_rows)

    accepted: list[tuple[int, float, float]] = []
    for fraction in SNAP_VALUES:
        if accepted and fraction.denominator > accepted[0][0]:
            break
        constant = float(fraction)
        if not admissible(constant):
            continue
        lr = statistic(constant)
        if lr <= SNAP_LR:
            accepted.append((fraction.denominator, lr, constant))
    if accepted:
        return min(accepted)[2]

    for digits in range(1, 7):
        constant = float(f"{optimum:.{digits}g}")
        if admissible(constant) and statistic(constant) <= ROUND_LR:
            return constant
    return optimum
