"""The forward step: a greedy path through the screened set, and its stopping rules.

The estimators' default ``parsimony="forward"`` adds screened terms one at a
time, each the one that most increases the in-sample R^2, and stops at the
first gain below ``parsimony_tol`` or, with ``parsimony_alpha``, the first
that fails to reject the null that chance reaches it.
"""

from __future__ import annotations

import numpy as np


def _project_out(basis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Remove from ``values`` its component in the span of ``basis``.

    ``basis`` has orthonormal columns. The projection is applied twice,
    which keeps the result orthogonal to the basis in floating point where
    a single pass loses it on nearly collinear columns.
    """
    for _ in range(2):
        values = values - basis @ (basis.T @ values)
    return values


def _forward_basis(matrix: np.ndarray, chosen: list[int]) -> np.ndarray:
    """Orthonormal basis of the intercept and the columns ``chosen``.

    A column within rounding of the span of those before it adds no
    direction.
    """
    n_rows = matrix.shape[0]
    basis = np.full((n_rows, 1), 1.0 / np.sqrt(n_rows))
    for index in chosen:
        column = matrix[:, index]
        reduced = _project_out(basis, column)
        norm = float(np.linalg.norm(reduced))
        if norm > 1e-10 * float(np.linalg.norm(column - column.mean())):
            basis = np.column_stack([basis, reduced / norm])
    return basis


# Noise draws behind each forward step's chance check (``parsimony_alpha``).
_CHANCE_DRAWS = 200


def _chance_p_value(
    matrix: np.ndarray,
    numeric: np.ndarray,
    chosen: list[int],
    remaining: list[int],
    rng: np.random.Generator,
) -> float:
    """P-value of the best gain the next forward step can add, against noise.

    The R^2 a remaining column adds is the share of variance left times the
    squared correlation of the current residual with the column net of the
    columns ``chosen``; the statistic is the largest of those correlations.
    Its null distribution is the same maximum over the same columns for
    wild-bootstrap draws of the residual (each row's residual times an
    independent standard normal), net of the chosen columns: the draws keep
    the residual's scale row by row and the correlation among the columns,
    and are associated with none of them. This is the best fit the remaining
    columns reach against noise, the multiplier bootstrap of Fan & Zhou
    (2016, *JMLR* 17). The p-value is the share of draws at least as large, counting
    the observed one among them.
    """
    basis = _forward_basis(matrix, chosen)
    residual = _project_out(basis, numeric)
    columns = matrix[:, remaining]
    reduced = _project_out(basis, columns)
    norms = np.linalg.norm(reduced, axis=0)
    live = norms > 1e-10 * np.linalg.norm(columns - columns.mean(axis=0), axis=0)
    left = float(np.linalg.norm(residual))
    if not live.any() or left <= 1e-12 * float(np.linalg.norm(numeric - numeric.mean())):
        return 1.0
    directions = reduced[:, live] / norms[live]
    draws = residual[:, None] * rng.standard_normal((residual.size, _CHANCE_DRAWS))
    draws = _project_out(basis, draws)
    draws /= np.where(np.linalg.norm(draws, axis=0) > 0.0, np.linalg.norm(draws, axis=0), 1.0)
    observed = float(np.max((directions.T @ residual / left) ** 2))
    null = np.max((directions.T @ draws) ** 2, axis=0)
    return float((1 + np.sum(null >= observed)) / (1 + _CHANCE_DRAWS))


def _extend_forward_path(
    matrix: np.ndarray,
    numeric: np.ndarray,
    chosen: list[int],
    remaining: list[int],
    total_variance: float,
) -> tuple[list[int], list[float]]:
    """Continue a greedy forward path from ``chosen`` through ``remaining``.

    Each step adds the column that most increases the in-sample R^2 of a
    least-squares fit with intercept, as the forward step does, but the
    columns already added are kept as an orthonormal basis and every other
    column as its residual against that basis. A column's R^2 gain is then
    the squared projection of the target's residual on it, so a step costs
    one pass over the remaining columns rather than one least-squares solve
    per column: of order ``n * k^2`` operations for a path through ``k``
    columns on ``n`` rows, against ``n * k^4`` for re-solving.

    A column within rounding of the span of those already added gains
    nothing, so it comes after every column that gains something.

    Returns the order in which ``remaining`` is added and the R^2 after
    each addition.
    """
    basis = _forward_basis(matrix, chosen)
    residual = _project_out(basis, numeric)

    remaining = list(remaining)
    candidates = matrix[:, remaining]
    floor = 1e-10 * np.linalg.norm(candidates - candidates.mean(axis=0), axis=0)
    candidates = _project_out(basis, candidates)

    order: list[int] = []
    scores: list[float] = []
    while remaining:
        norms = np.linalg.norm(candidates, axis=0)
        live = norms > floor
        safe = np.where(live, norms, 1.0)
        gains = np.where(live, (residual @ candidates / safe) ** 2, 0.0)
        position = int(np.argmax(gains))
        if live[position]:
            direction = candidates[:, position] / norms[position]
            residual = _project_out(direction[:, None], residual)
            candidates = _project_out(direction[:, None], candidates)
        order.append(remaining.pop(position))
        candidates = np.delete(candidates, position, axis=1)
        floor = np.delete(floor, position)
        scores.append(1.0 - float(np.var(residual)) / total_variance)
    return order, scores
