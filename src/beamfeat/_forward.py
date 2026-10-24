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


def forward_path(
    matrix: np.ndarray,
    target: np.ndarray,
    screened: list[int],
    *,
    classification: bool,
    parsimony_tol: float,
    parsimony_alpha: float | None,
    random_state: int | None,
) -> tuple[list[int], list[float], int]:
    """The greedy forward path through the whole screened set.

    Returns the screened indices in the order the forward step adds them,
    the in-sample R^2 after each addition, and the number of them the step
    keeps at ``parsimony_tol`` (and ``parsimony_alpha``). The path runs past
    that point to the full screened set, so the fit at every size is on
    record; the steps up to it are the forward step itself, computed exactly
    as it computes them, so the subset it keeps does not depend on how far
    the path is followed.
    """
    # Labels are encoded before any numeric cast: the transformer passes
    # its target through unencoded, and string labels cannot be cast.
    if classification:
        _, numeric = np.unique(target, return_inverse=True)
        numeric = numeric.astype(np.float64)
    else:
        numeric = np.asarray(target, dtype=np.float64)
    total_variance = float(np.var(numeric))
    if total_variance < 1e-12:  # pragma: no cover - defensive
        return list(screened), [float("nan")] * len(screened), len(screened)

    # The second stream of the seed: the first is the knockoff selector's.
    rng = (
        None if parsimony_alpha is None
        else np.random.default_rng(np.random.SeedSequence(random_state).spawn(2)[1])
    )
    chosen: list[int] = []
    scores_kept: list[float] = []
    remaining = list(screened)
    best_r2 = 0.0
    intercept = np.ones((matrix.shape[0], 1))
    while remaining:
        scores = []
        for index in remaining:
            design = np.column_stack([intercept] + [matrix[:, j].reshape(-1, 1) for j in chosen + [index]])
            coefficients, *_ = np.linalg.lstsq(design, numeric, rcond=None)
            residual = numeric - design @ coefficients
            scores.append(1.0 - float(np.var(residual)) / total_variance)
        best_position = int(np.argmax(scores))
        if chosen and (
            scores[best_position] - best_r2 < parsimony_tol
            or (
                rng is not None
                and _chance_p_value(matrix, numeric, chosen, remaining, rng) > parsimony_alpha
            )
        ):
            break
        best_r2 = scores[best_position]
        chosen.append(remaining.pop(best_position))
        scores_kept.append(best_r2)
    stop = len(chosen)
    if remaining:
        rest, rest_scores = _extend_forward_path(
            matrix, numeric, chosen, remaining, total_variance
        )
        chosen, scores_kept = chosen + rest, scores_kept + rest_scores
    return chosen, scores_kept, stop
