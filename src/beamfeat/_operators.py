"""The built-in operators: their functions, units, formulas and constant grids.

Each is registered in :data:`beamfeat.expression.OPERATORS` when this module
is imported, which :mod:`beamfeat.expression` does once it has defined what
the operators are built from. A new built-in operator is one more
:func:`~beamfeat.expression.register_operator` call here; user code registers
its own the same way, with an :class:`~beamfeat.expression.OperatorSpec`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from beamfeat.expression import (
    POLE_MARGIN_FLOOR,
    POLE_MARGIN_ROWS,
    POLE_MARGIN_SPREADS,
    OperatorSpec,
    UnitRegistryError,
    _add_unit,
    _const_factor,
    _const_text,
    _dimensionless_required,
    _same_dimension_required,
    _span,
    pole_margins,
    register_operator,
)


def _unit_preserving(unit: Any) -> Any:
    return unit


def _reciprocal_unit(unit: Any) -> Any:
    return None if unit is None else 1 / unit


def _clear_of_pole(denominator: np.ndarray) -> bool:
    """Whether ``denominator`` keeps 0 at least the pole margin from its values.

    The margin is the one a fitted shift keeps (:func:`pole_margins`), on the
    side of the values that faces 0. A denominator that comes closer to 0 on
    the search rows, or crosses it, can cross it on a new row a little beyond
    them, where the quotient takes any size and either sign. Applied in the
    search only, like every domain.
    """
    low, high = float(np.min(denominator)), float(np.max(denominator))
    if low <= 0.0 <= high:
        return False
    # The distance to 0 and the values nearest it, mirrored when all are negative.
    distance, nearest = (low, denominator) if low > 0.0 else (-high, -denominator)
    rows = min(POLE_MARGIN_ROWS, len(nearest))
    margin = POLE_MARGIN_FLOOR * _span(denominator)
    if rows >= 2:
        closest = np.partition(nearest, rows - 1)[:rows]
        margin = max(margin, POLE_MARGIN_SPREADS * float(np.max(closest) - np.min(closest)))
    return distance >= margin


def _square_unit(unit: Any) -> Any:
    return None if unit is None else unit**2


def _cube_unit(unit: Any) -> Any:
    return None if unit is None else unit**3


def _sqrt_unit(unit: Any) -> Any:
    return None if unit is None else unit**0.5


def _log_unit(unit: Any) -> Any:
    _dimensionless_required(unit, "log")
    return unit


def _exp_unit(unit: Any) -> Any:
    _dimensionless_required(unit, "exp")
    return unit


def _dimensionless_unit(op_name: str) -> Callable[[Any], Any]:
    """Unit function of an operator that needs, and returns, a dimensionless value."""

    def unit_fn(unit: Any) -> Any:
        _dimensionless_required(unit, op_name)
        return unit

    return unit_fn


def _sub_unit(left: Any, right: Any) -> Any:
    _same_dimension_required(left, right, "subtraction")
    return left if left is not None else right


def _same_registry_required(left: Any, right: Any, op_name: str) -> None:
    """Raise if ``left`` and ``right`` come from different pint registries."""
    if getattr(left, "_REGISTRY", None) is not getattr(right, "_REGISTRY", None):
        raise UnitRegistryError(
            f"{op_name} of units from different pint registries ({left.units} and {right.units}); "
            "build every unit from one UnitRegistry"
        )


def _mul_unit(left: Any, right: Any) -> Any:
    if left is None or right is None:
        return None
    _same_registry_required(left, right, "multiplication")
    return left * right


def _div_unit(left: Any, right: Any) -> Any:
    if left is None or right is None:
        return None
    _same_registry_required(left, right, "division")
    return left / right


# Unary operators ----------------------------------------------------------- #

register_operator(
    OperatorSpec(
        name="reciprocal",
        arity=1,
        fn=lambda x: np.reciprocal(x),
        formula=lambda x: f"1/({x})",
        unit_fn=_reciprocal_unit,
        domain=_clear_of_pole,
    )
)


register_operator(
    OperatorSpec(
        name="log",
        arity=1,
        fn=lambda x: np.log(x),
        formula=lambda x: f"log({x})",
        unit_fn=_log_unit,
        domain=lambda x: bool(np.all(x > 0)),
    )
)


register_operator(
    OperatorSpec(
        name="sqrt",
        arity=1,
        fn=lambda x: np.sqrt(x),
        formula=lambda x: f"sqrt({x})",
        unit_fn=_sqrt_unit,
        domain=lambda x: bool(np.all(x >= 0)),
    )
)


register_operator(
    OperatorSpec(
        name="abs",
        arity=1,
        fn=lambda x: np.abs(x),
        formula=lambda x: f"abs({x})",
        unit_fn=_unit_preserving,
        redundant=lambda x: bool(np.all(x >= 0)),
    )
)


def _sign_unit(unit: Any) -> Any:
    """A sign carries no dimension, whatever its operand's."""
    return None if unit is None else unit / unit


register_operator(
    OperatorSpec(
        name="sign",
        arity=1,
        fn=lambda x: np.sign(x),
        formula=lambda x: f"sign({x})",
        unit_fn=_sign_unit,
    )
)


register_operator(
    OperatorSpec(
        name="square",
        arity=1,
        fn=lambda x: np.square(x),
        formula=lambda x: f"({x})^2",
        unit_fn=_square_unit,
    )
)


register_operator(
    OperatorSpec(
        name="cube",
        arity=1,
        fn=lambda x: x**3,
        formula=lambda x: f"({x})^3",
        unit_fn=_cube_unit,
    )
)


register_operator(
    OperatorSpec(
        name="exp",
        arity=1,
        fn=lambda x: np.exp(x),
        formula=lambda x: f"exp({x})",
        unit_fn=_exp_unit,
        domain=lambda x: bool(np.all(x < 50.0)),
    )
)


# Parametric unary operators ------------------------------------------------ #
#
# Each carries one constant, fitted to the data by beamfeat.parametric when
# the operator is listed in a search's ``fitted_ops``. The grids are the
# coarse search over the constant, scaled to the operand so that the same
# grid serves a column in metres and one in kilometres:
#
# - exp(a*x): |a| from 1e-3 to 10 over max|x|, each sign, so that |a*x| stays
#   at most 10. Small |a| makes exp(a*x) a linear function of x, and large
#   |a| a step; a fit that runs to either end has found neither shape.
# - x^a: a from -4 to 4 in steps of 0.1, on positive operands. A negative
#   power has its pole at 0, so the negative half is used only when the
#   smallest operand lies more than the margin (below) above 0; otherwise a
#   runs from 0.1 to 4.
# - 1/(x + a) and log(x + a): the shift places the pole a distance d below
#   the smallest operand (or, for 1/(x + a), above the largest), log-spaced
#   at ten points a decade from a margin (below) up to 100 times the
#   operand's range. Far from the pole, the function is linear in x.
#
# The margin keeps a fitted pole away from the data. Close to the pole the
# function is dominated by the few rows next to it, and new rows reach past
# the training extremes: a pole fitted just below the smallest value is
# crossed by the first new row a little further out, which then evaluates to
# NaN (the logarithm) or to a value of the wrong sign and any size (the
# reciprocal). The margin is the larger of POLE_MARGIN_SPREADS times
# the spread of the POLE_MARGIN_ROWS most extreme operand values at that end
# and POLE_MARGIN_FLOOR times the range, so it widens where the data thin
# out towards the edge (a long tail, or few rows) and narrows where they are
# dense. It was chosen on the solo fits to every column of 18 real datasets,
# on five train/test splits each with half the training rows searched. Of
# the 1,556 fits whose pole was at least 0.01 ranges out, the 343 inside
# this margin were crossed by a row the fit had not seen in 14% of cases,
# and lost their gain over the bare column on those rows in 42%; the 1,213
# outside it in 0.3% and 3%. A margin of a fixed fraction of the range does
# worse: 0.1 left 1.3% crossed and 9% losing, and 0.2 left 0.3% crossed and
# 6% losing while keeping 1,086 fits instead of 1,213.
# A fit that wants its pole inside the margin runs into the grid's bound and
# is discarded.
#
# The same margin decides whether x^a may have a negative power, whose pole
# is fixed at 0: a new row at 0 evaluates to an infinity, and to NaN where it
# is multiplied by a zero. Where it may not, a fit that runs down to the
# smallest positive power, 0.1, is discarded as well: as a approaches 0, x^a
# takes the shape of log(x), which has the same pole.


_SCALE_MAGNITUDES = np.logspace(-3.0, 1.0, 41)


# Upper end of the shift grids, in operand ranges, and their density.
_SHIFT_REACH = 100.0


_SHIFT_POINTS_PER_DECADE = 10


def _scale_grid(x: np.ndarray) -> list[np.ndarray]:
    largest = float(np.max(np.abs(x)))
    if not math.isfinite(largest) or largest == 0.0:
        return []
    magnitudes = _SCALE_MAGNITUDES / largest
    return [-magnitudes[::-1], magnitudes]


def _power_grid(x: np.ndarray) -> list[np.ndarray]:
    below, _ = pole_margins(x)
    steps = np.arange(-40, 41) if float(np.min(x)) > below else np.arange(1, 41)
    return [np.round(steps * 0.1, 10)]


def _shift_distances(margin: float, span: float) -> np.ndarray:
    """Pole distances from ``margin`` to the grid's reach, log-spaced."""
    reach = _SHIFT_REACH * span
    count = max(3, int(math.ceil(_SHIFT_POINTS_PER_DECADE * math.log10(reach / margin))) + 1)
    return np.geomspace(margin, reach, count)


def _pole_grid(x: np.ndarray) -> list[np.ndarray]:
    span = _span(x)
    below, above = pole_margins(x)
    return [
        -float(np.max(x)) - _shift_distances(above, span)[::-1],
        _shift_distances(below, span) - float(np.min(x)),
    ]


def _log_shift_grid(x: np.ndarray) -> list[np.ndarray]:
    below, _ = pole_margins(x)
    return [_shift_distances(below, _span(x)) - float(np.min(x))]


def _scaled_formula(x: str, a: float) -> str:
    return f"exp(-{x})" if a == -1.0 else f"exp({_const_factor(a)}*{x})"


def _power_formula(x: str, a: float) -> str:
    text = _const_text(a)
    return f"({x})^({text})" if a < 0 or "/" in text else f"({x})^{text}"


def _shift_text(x: str, a: float) -> str:
    return f"{x} - {_const_text(-a)}" if a < 0 else f"{x} + {_const_text(a)}"


register_operator(
    OperatorSpec(
        name="exp_scaled",
        arity=1,
        fn=lambda x, a: np.exp(a * x),
        formula=_scaled_formula,
        unit_fn=_dimensionless_unit("exp_scaled"),
        domain=lambda x, a: bool(np.all(a * x < 50.0)),
        constant_grid=_scale_grid,
    )
)


register_operator(
    OperatorSpec(
        name="pow",
        arity=1,
        fn=lambda x, a: np.power(x, a),
        formula=_power_formula,
        unit_fn=_dimensionless_unit("pow"),
        domain=lambda x, a: bool(np.all(x > 0)),
        constant_grid=_power_grid,
    )
)


register_operator(
    OperatorSpec(
        name="inv_shift",
        arity=1,
        fn=lambda x, a: np.reciprocal(x + a),
        formula=lambda x, a: f"1/({_shift_text(x, a)})",
        unit_fn=_reciprocal_unit,
        domain=lambda x, a: bool(np.all(x + a != 0)),
        constant_grid=_pole_grid,
    )
)


register_operator(
    OperatorSpec(
        name="log_shift",
        arity=1,
        fn=lambda x, a: np.log(x + a),
        formula=lambda x, a: f"log({_shift_text(x, a)})",
        unit_fn=_dimensionless_unit("log_shift"),
        domain=lambda x, a: bool(np.all(x + a > 0)),
        constant_grid=_log_shift_grid,
    )
)


# The thresholds sign_shift's grid tries: these quantiles of the operand, so
# that each side of a threshold holds at least a twentieth of the rows.
_THRESHOLD_LEVELS = np.linspace(0.05, 0.95, 37)


def _threshold_grid(x: np.ndarray) -> list[np.ndarray]:
    shifts = np.unique(-np.quantile(x, _THRESHOLD_LEVELS))
    return [shifts] if len(shifts) >= 3 else []


register_operator(
    OperatorSpec(
        name="sign_shift",
        arity=1,
        fn=lambda x, a: np.sign(x + a),
        formula=lambda x, a: f"sign({_shift_text(x, a)})",
        unit_fn=_sign_unit,
        domain=lambda x, a: bool(np.any(x + a > 0) and np.any(x + a < 0)),
        constant_grid=_threshold_grid,
    )
)


register_operator(
    OperatorSpec(
        name="add",
        arity=2,
        fn=lambda x, y: x + y,
        formula=lambda x, y: f"({x} + {y})",
        unit_fn=_add_unit,
        commutative=True,
    )
)


register_operator(
    OperatorSpec(
        name="sub",
        arity=2,
        fn=lambda x, y: x - y,
        formula=lambda x, y: f"({x} - {y})",
        unit_fn=_sub_unit,
        commutative=False,
    )
)


register_operator(
    OperatorSpec(
        name="mul",
        arity=2,
        fn=lambda x, y: x * y,
        formula=lambda x, y: f"({x} * {y})",
        unit_fn=_mul_unit,
        commutative=True,
    )
)


def _divisor_text(y: str) -> str:
    """A divisor spelled ``1/(...)`` in brackets, so ``a / (1/(b))`` cannot read as ``(a/1)/b``."""
    return f"({y})" if y.startswith("1/(") else y


register_operator(
    OperatorSpec(
        name="div",
        arity=2,
        fn=lambda x, y: x / y,
        formula=lambda x, y: f"({x} / {_divisor_text(y)})",
        unit_fn=_div_unit,
        commutative=False,
        domain=lambda x, y: _clear_of_pole(y),
    )
)
