"""Fitting the constant of a parametric operator.

A parametric operator, such as ``exp_scaled`` (``exp(a*x)``) or ``inv_shift``
(``1/(x + a)``), carries one constant. :func:`fit_constants` chooses it for a
given operand by a one-dimensional search: a coarse grid from the operator's
:attr:`~beamfeat.expression.OperatorSpec.constant_grid`, then a golden-section
refinement between the grid neighbours of the best point.
:func:`fit_constants_many` fits many operands at once: the refinements and
the snapping run side by side, one array operation per step for all of them,
which is what a search fitting hundreds of operands needs.

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

**Discarded fits.** A fit whose best grid point is the end of a grid
segment ran into a bound -- the margin a shift keeps between its pole and
the data (see :func:`~beamfeat.expression.pole_margins`), the smallest
positive power when the operand comes within that margin of 0 (the pole of
a negative power), the linear limit of a small scale, or the step limit of
a large one -- and is discarded, as is one whose values are not finite,
constant, or outside the operator's domain. Snapping and rounding stay
within the same bounds, so a simple constant never moves a pole closer to
the data than the margin. And since each operator tends to an affine
function of its operand at one end of its range (a small scale, a distant
shift, a power of 1), a fit is also discarded unless the likelihood-ratio
test below rejects the null that the operand itself, or for a product the
partner alone, fits as well; otherwise the constant only fits noise.

**Snapping.** The data rarely pin a constant to more digits than they
support, and ``exp(-0.49991*x^2)`` is a worse answer than ``exp(-x^2/2)``.
After the refinement, the simple values of :data:`SNAP_VALUES` (0, +-1,
+-2, +-3, +-1/2, +-3/2, +-1/3, +-1/4) are tried. For each, a
likelihood-ratio test asks whether the data reject the null that the
constant takes that value, and the simplest value it fails to reject is
used: the smallest denominator, then the best fit. The test is for a
least-squares fit of the target on the feature: with
``L(a) = 1 - r(a)^2`` the unexplained fraction and ``n`` the rows, it fails
to reject when ``n * log(L(simple) / L(optimum))`` is at most
:data:`SNAP_LR`, the 99.9% point of a chi-squared variable with one degree
of freedom, so it runs at level 0.001. A tolerance stated this way scales
with the noise and the sample: on near-noiseless data the test fails to
reject only a value that fits essentially as well as the optimum, on noisy
data a wider range, as it should. A fixed tolerance on ``|r|`` cannot do
both; at 0.1% noise, ``r`` is within 1e-6 of 1 and a tolerance of 1e-4
would snap to a constant that leaves a hundred times the noise
unexplained. The 99.9% point rather than the 95% one makes the simpler
constant the default: the null for a law's true constant is falsely
rejected, a Type I error, in one fit in a thousand rather than one in
twenty. If the test rejects every simple value, the constant is rounded to
the fewest significant digits (up to six) that cost at most
:data:`ROUND_LR`, so that the printed constant is the stored one.

Everything here is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from beamfeat.expression import OperatorSpec

__all__ = ["ROUND_LR", "SNAP_LR", "SNAP_VALUES", "FittedConstant", "fit_constants", "fit_constants_many"]

#: Simple constants tried after a fit, simplest first: zero, the integers to
#: 3, halves, thirds and quarters. Among the values whose null the snapping
#: test fails to reject, one with a smaller denominator is preferred, then the
#: one that fits best.
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

# Values computed at once when many fits are evaluated together, so that a
# block stays near 8 MB however many rows there are.
_BLOCK_ELEMENTS = 1 << 20


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
    operand = np.asarray(operand, dtype=np.float64)
    return fit_constants_many(spec, operand.reshape(-1, 1), target, partners)[0]


def fit_constants_many(
    spec: OperatorSpec,
    operands: np.ndarray,
    target: np.ndarray,
    partners: np.ndarray | None = None,
    with_partners: Sequence[bool] | None = None,
) -> list[list[FittedConstant]]:
    """:func:`fit_constants` for each column of ``operands``, fitted together.

    The result for a column is the one :func:`fit_constants` gives it, with
    ``partners`` when ``with_partners`` is ``None`` or true for that column
    and without them otherwise.

    Args:
        spec: A parametric operator.
        operands: ``(n_rows, n_operands)``, one operand per column.
        target: The standardised target.
        partners: Optional ``(n_rows, n_partners)`` matrix, as for
            :func:`fit_constants`.
        with_partners: Per operand, whether to fit the product with a
            partner too.

    Returns:
        One list of fitted constants per operand.
    """
    if not spec.parametric:
        raise ValueError(f"operator {spec.name!r} has no constant to fit")
    operands = np.asarray(operands, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_rows, n_operands = operands.shape
    sample = _grid_rows(n_rows)
    if partners is not None:
        partners = np.asarray(partners, dtype=np.float64)
        if not partners.shape[1]:
            partners = None
    sampled_partners = None if partners is None else partners[sample]
    # The correlations each fit has to beat, all at once: the operands' own,
    # and the partners' for the products.
    baselines = _correlation_rows(np.ascontiguousarray(operands.T), target)
    partner_baselines = None if partners is None else _correlation_rows(np.ascontiguousarray(partners.T), target)

    jobs: list[_Job] = []
    for column in range(n_operands):
        operand = operands[:, column]
        segments = [np.asarray(s, dtype=np.float64) for s in spec.constant_grid(operand) if len(s) >= 3]
        if not segments:
            continue
        grid = np.concatenate(segments)
        # Each grid point's position in its segment, to recognise a segment's ends.
        ends = np.concatenate([np.r_[True, np.zeros(len(s) - 2, dtype=bool), True] for s in segments])
        columns = _grid_columns(spec, operand[sample], grid)
        # The segments' ends bound the constant, snapped or rounded too.
        bounds = tuple((float(np.min(s)), float(np.max(s))) for s in segments)
        solo = _correlations(columns, target[sample], None)[0]
        _add_job(jobs, column, None, grid, ends, solo, float(baselines[column]), bounds)
        if sampled_partners is not None and (with_partners is None or with_partners[column]):
            scores = _correlations(columns, target[sample], sampled_partners)
            if np.isfinite(scores).any():
                best = int(np.argmax(np.where(np.isfinite(scores), scores, -1.0).max(axis=1)))
                _add_job(jobs, column, best, grid, ends, scores[best], float(partner_baselines[best]), bounds)

    found: list[list[FittedConstant]] = [[] for _ in range(n_operands)]
    for job, constant in zip(jobs, _fit_jobs(spec, operands, target, partners, jobs)):
        if constant is not None:
            found[job.column].append(FittedConstant(constant, job.partner))
    return found


class _Job(NamedTuple):
    """One constant to fit: an operand, alone or times a partner."""

    column: int
    partner: int | None
    low: float
    high: float
    point: float
    baseline: float
    bounds: tuple[tuple[float, float], ...]


def _add_job(jobs, column, partner, grid, ends, grid_scores, baseline, bounds) -> None:
    """Queue a fit around the best grid point, unless it is invalid or on a bound.

    A best point at the end of a grid segment ran into a bound (see the
    module notes), and is not refined.
    """
    if not np.isfinite(grid_scores).any():
        return
    best = int(np.nanargmax(grid_scores))
    if ends[best]:
        return
    # An interior grid point's neighbours lie in its own segment.
    jobs.append(
        _Job(column, partner, float(grid[best - 1]), float(grid[best + 1]), float(grid[best]), baseline, bounds)
    )


def _within(bounds: tuple[tuple[float, float], ...], constant: float) -> bool:
    return any(low <= constant <= high for low, high in bounds)


def _fit_jobs(
    spec: OperatorSpec,
    operands: np.ndarray,
    target: np.ndarray,
    partners: np.ndarray | None,
    jobs: list[_Job],
) -> list[float | None]:
    """Refine, then snap or round, the best grid constant of every job.

    Each job's objective is the |corr| of the operator's values, times the
    job's partner if it has one, with the target. The refinement is
    golden-section search between the grid neighbours of the best grid
    point; the grid point itself is kept if it scores better. A job's
    ``baseline`` is the correlation the fit has to beat: the operand's own,
    or the partner's for a product. Every operator here tends to an affine
    function of its operand at one end of its range (a small scale, a far
    shift, a power of 1), so unless the likelihood-ratio test of the snap
    rejects the null that the baseline fits as well, the constant only fits
    noise and the fit is discarded. Returns each job's constant, or ``None``.
    """
    if not jobs:
        return []
    n_rows = operands.shape[0]
    # Rows of these are contiguous, so a row's sums are taken exactly as a
    # single column's would be.
    operand_rows = np.ascontiguousarray(operands.T)
    partner_rows = None if partners is None else np.ascontiguousarray(partners.T)
    columns = np.array([job.column for job in jobs])
    partner_of = np.array([-1 if job.partner is None else job.partner for job in jobs])

    def evaluate(which: np.ndarray, constants: np.ndarray) -> np.ndarray:
        scores = np.full(len(which), np.nan)
        step = max(1, _BLOCK_ELEMENTS // n_rows)
        for start in range(0, len(which), step):
            part = slice(start, start + step)
            values = _apply_rows(spec, operand_rows[columns[which[part]]], constants[part])
            paired = partner_of[which[part]]
            if partner_rows is not None and (paired >= 0).any():
                values[paired >= 0] *= partner_rows[paired[paired >= 0]]
            scores[part] = _correlation_rows(values, target)
        return scores

    everyone = np.arange(len(jobs))
    optima, scores = _golden_max_many(
        evaluate, np.array([job.low for job in jobs]), np.array([job.high for job in jobs])
    )
    points = np.array([job.point for job in jobs])
    point_scores = evaluate(everyone, points)
    keep_point = ~np.isfinite(scores) | (np.isfinite(point_scores) & (point_scores > scores))
    optima = np.where(keep_point, points, optima)
    scores = np.where(keep_point, point_scores, scores)

    live = np.flatnonzero(np.isfinite(scores))
    constants, constant_scores = _simplify_many(
        evaluate, live, optima[live], scores[live], n_rows,
        [lambda constant, bounds=jobs[i].bounds: _within(bounds, constant) for i in live],
    )
    result: list[float | None] = [None] * len(jobs)
    for i, constant, score in zip(live, constants, constant_scores):
        if _statistic(jobs[i].baseline, float(score), n_rows) > SNAP_LR:
            result[i] = float(constant)
    return result


def _apply_rows(spec: OperatorSpec, operands: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """:func:`_apply` for each row of ``operands`` with its constant; NaN rows where it rejects.

    All rows are computed at once; if that raises a floating-point error, or
    does not broadcast, they are computed one at a time, which tells the
    failing ones apart.
    """
    try:
        with np.errstate(all="raise"):
            values = np.asarray(spec.fn(operands, constants[:, None]), dtype=np.float64)
            inside = spec.domain is None or spec.domain(operands, constants[:, None])
    except (FloatingPointError, ValueError, TypeError):
        values = None
    if values is not None and values.shape == operands.shape:
        if not inside:
            for row, constant in enumerate(constants):
                if not spec.domain(operands[row], float(constant)):
                    values[row] = np.nan
        values[~np.isfinite(values).all(axis=1)] = np.nan
        return values
    values = np.full(operands.shape, np.nan)
    for row, constant in enumerate(constants):
        computed = _apply(spec, operands[row], float(constant))
        if computed is not None:
            values[row] = computed
    return values


def _grid_rows(n_rows: int) -> np.ndarray:
    """Evenly spaced rows for the coarse grid: all of them, up to a limit."""
    if n_rows <= _GRID_ROWS:
        return np.arange(n_rows)
    return np.linspace(0, n_rows - 1, _GRID_ROWS).round().astype(np.intp)


def _grid_columns(spec: OperatorSpec, operand: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The operator applied with each grid constant, one column per constant.

    A column is NaN where the evaluator would reject the result: outside the
    operator's domain, or on a floating-point error, underflow included.

    The registered operators are elementwise, so every constant is applied
    at once, and their domain predicates hold for every element or fail, so
    one check over all the constants settles the usual case where all are
    inside. If the joint pass raises a floating-point error anywhere, or does
    not broadcast, the columns are computed one at a time instead, which
    tells the failing ones apart.
    """
    shape = (len(operand), len(grid))
    try:
        with np.errstate(all="raise"):
            columns = np.asarray(spec.fn(operand[:, None], grid[None, :]), dtype=np.float64)
            inside = spec.domain is None or spec.domain(operand[:, None], grid[None, :])
    except (FloatingPointError, ValueError, TypeError):
        columns = None
    if columns is not None and columns.shape == shape and np.isfinite(columns).all():
        if not inside:
            for index, constant in enumerate(grid):
                if not spec.domain(operand, float(constant)):
                    columns[:, index] = np.nan
        return columns
    columns = np.full(shape, np.nan)
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


def _correlation_rows(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """|corr| of each row with the standardised target, NaN for an invalid or constant row."""
    n_rows = values.shape[1]
    with np.errstate(all="ignore"):
        centred = values - values.sum(axis=1, keepdims=True) / n_rows
        spread = np.sqrt((centred * centred).sum(axis=1) / n_rows)
        scale = np.abs(values).sum(axis=1) / n_rows
        correlation = np.minimum(np.abs((centred * target).sum(axis=1)) / (n_rows * spread), 1.0)
    valid = np.isfinite(spread) & (spread > 1e-12 * scale) & (spread != 0.0)
    return np.where(valid, correlation, np.nan)


def _correlation(values: np.ndarray, target: np.ndarray) -> float:
    """|corr| of a column with the standardised target, NaN for a constant column."""
    return float(_correlation_rows(np.asarray(values, dtype=np.float64).reshape(1, -1), target)[0])


def _golden_max_many(
    evaluate: Callable[[np.ndarray, np.ndarray], np.ndarray], lows: np.ndarray, highs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Golden-section search for the maximum of each job's objective on its bracket.

    The jobs step side by side, each exactly as it would alone: one new
    point per unfinished job per step, all evaluated by one call of
    ``evaluate(jobs, points)``. NaN counts as worse than any value. Returns
    each job's best point evaluated and its value.
    """

    def value(which: np.ndarray, points: np.ndarray) -> np.ndarray:
        result = evaluate(which, points)
        return np.where(np.isfinite(result), result, -np.inf)

    low, high = lows.astype(np.float64), highs.astype(np.float64)
    inner_low = high - _INVERSE_PHI * (high - low)
    inner_high = low + _INVERSE_PHI * (high - low)
    everyone = np.arange(len(low))
    value_low, value_high = value(everyone, inner_low), value(everyone, inner_high)
    tolerance = _REFINE_RTOL * np.maximum(np.maximum(np.abs(low), np.abs(high)), 1e-300)
    active = np.flatnonzero(high - low > tolerance)
    while active.size:
        left = value_low[active] >= value_high[active]
        down, up = active[left], active[~left]
        high[down], inner_high[down], value_high[down] = inner_high[down], inner_low[down], value_low[down]
        inner_low[down] = high[down] - _INVERSE_PHI * (high[down] - low[down])
        low[up], inner_low[up], value_low[up] = inner_low[up], inner_high[up], value_high[up]
        inner_high[up] = low[up] + _INVERSE_PHI * (high[up] - low[up])
        values = value(active, np.where(left, inner_low[active], inner_high[active]))
        value_low[down], value_high[up] = values[left], values[~left]
        active = active[high[active] - low[active] > tolerance[active]]
    pick = value_low >= value_high
    return np.where(pick, inner_low, inner_high), np.where(pick, value_low, value_high)


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
    """The simplest constant ``c`` for which the snapping test fails to reject ``a = c``.

    Tries :data:`SNAP_VALUES` against :data:`SNAP_LR`, then rounding to one to
    six significant digits against :data:`ROUND_LR`; see the module notes.
    Only constants ``admissible`` accepts are tried: a snap must not carry a
    pole past its margin into the data, or a scale past the grid's bound.
    """

    def evaluate(which: np.ndarray, constants: np.ndarray) -> np.ndarray:
        return np.array([objective(float(constant)) for constant in constants])

    constants, _ = _simplify_many(
        evaluate, np.zeros(1, dtype=np.intp), np.array([optimum]), np.array([score]), n_rows, [admissible]
    )
    return float(constants[0])


def _simplify_many(
    evaluate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    which: np.ndarray,
    optima: np.ndarray,
    scores: np.ndarray,
    n_rows: int,
    admissible: Sequence[Callable[[float], bool]],
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_simplify` for jobs ``which``, with every candidate constant evaluated at once.

    Among the snap values whose null the test fails to reject for a job, the
    ones with the smallest denominator compete and the best fit wins (the
    smaller constant on a tie). If the test rejects every snap value, the
    fewest significant digits within :data:`ROUND_LR` win, and otherwise the
    optimum stays. Returns the constants and their scores.
    """
    constants, chosen = optima.astype(np.float64), scores.astype(np.float64)

    def candidates(make) -> tuple[list[int], list[float], list]:
        jobs, values, keys = [], [], []
        for position in range(len(which)):
            for key, constant in make(position):
                if admissible[position](constant):
                    jobs.append(position)
                    values.append(constant)
                    keys.append(key)
        return jobs, values, keys

    def tested(jobs, values):
        if not jobs:
            return np.empty(0), []
        found = evaluate(which[np.array(jobs)], np.array(values, dtype=np.float64))
        return found, [_statistic(float(s), float(scores[j]), n_rows) for j, s in zip(jobs, found)]

    jobs, values, denominators = candidates(
        lambda position: [(fraction.denominator, float(fraction)) for fraction in SNAP_VALUES]
    )
    found, statistics = tested(jobs, values)
    best: dict[int, tuple[int, float, float, float]] = {}
    for job, constant, denominator, score, lr in zip(jobs, values, denominators, found, statistics):
        if lr <= SNAP_LR:
            entry = (denominator, lr, constant, float(score))
            if job not in best or entry[:3] < best[job][:3]:
                best[job] = entry
    for job, (_, _, constant, score) in best.items():
        constants[job], chosen[job] = constant, score

    rest = [position for position in range(len(which)) if position not in best]
    jobs, values, digits = candidates(
        lambda position: (
            [(d, float(f"{optima[position]:.{d}g}")) for d in range(1, 7)] if position not in best else []
        )
    )
    found, statistics = tested(jobs, values)
    rounded: dict[int, tuple[int, float, float]] = {}
    for job, constant, digit, score, lr in zip(jobs, values, digits, found, statistics):
        if lr <= ROUND_LR and (job not in rounded or digit < rounded[job][0]):
            rounded[job] = (digit, constant, float(score))
    for job in rest:
        if job in rounded:
            _, constants[job], chosen[job] = rounded[job]
    return constants, chosen
