"""Evaluation of fitted expressions on new rows, one row's worth at a time.

A fitted expression can be undefined on some rows of new data, such as the
logarithm of a negative value. These helpers find those rows and give every
other row exactly the value it would get if evaluated alone, so a
prediction never depends on what else is in the batch.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from beamfeat.expression import OPERATORS, Evaluator, Node


def _undefined_rows(node: Node, data: Mapping[str, np.ndarray], n_rows: int) -> np.ndarray:
    """Mask of the rows on which ``node`` or any subexpression is non-finite.

    Operators are elementwise, so a row is undefined here exactly when it
    would make the evaluation of ``node`` fail on its own. Underflow is the
    one failure this pass cannot see, since it leaves finite values behind;
    :func:`_evaluate_by_row` checks the remaining rows for it.
    """
    return _walk_undefined(node, data, n_rows)[1]


def _walk_undefined(
    current: Node, data: Mapping[str, np.ndarray], n_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Values of ``current`` and its undefined-row mask, for :func:`_undefined_rows`.

    Module-level, not a closure that calls itself, which would be a
    reference cycle holding ``data`` until the garbage collector ran.
    """
    if current.is_leaf:
        values = np.asarray(data[str(current.column)], dtype=np.float64)
        return values, ~np.isfinite(values)
    operands, undefined = [], np.zeros(n_rows, dtype=bool)
    for child in current.children:
        values, child_undefined = _walk_undefined(child, data, n_rows)
        operands.append(values)
        undefined |= child_undefined
    if current.const is not None:
        operands.append(current.const)
    with np.errstate(all="ignore"):
        values = np.asarray(OPERATORS[current.op].fn(*operands), dtype=np.float64)
    return values, undefined | ~np.isfinite(values)


def _evaluate_by_row(node: Node, data: Mapping[str, np.ndarray], units, n_rows: int) -> np.ndarray:
    """Evaluate ``node`` on every row it is defined on, with NaN on the rest.

    Each defined row gets the value it would get if evaluated alone, so a
    prediction never depends on what else is in the batch. Rows found
    undefined by :func:`_undefined_rows` are set aside first; the rest are
    evaluated as blocks, and a block that still fails is halved until the
    failing rows are isolated.
    """
    result = np.full(n_rows, np.nan)
    rows = np.flatnonzero(~_undefined_rows(node, data, n_rows))
    pending = [rows] if len(rows) else []
    while pending:
        block = pending.pop()
        values = Evaluator(
            {name: column[block] for name, column in data.items()}, units=units
        ).transform_values(node)
        if values is not None:
            result[block] = values
        elif len(block) > 1:
            half = len(block) // 2
            pending.extend([block[:half], block[half:]])
    return result
