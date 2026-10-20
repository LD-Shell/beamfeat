"""Expression DAG with structural hashing, lazy evaluation, and unit propagation.

The central abstraction is :class:`Node`, an immutable description of how to
compute a feature from input columns. Nodes are *descriptions*, not data: they
carry no arrays. Evaluation happens on demand via :class:`Evaluator`, which
caches intermediate results keyed by structural hash so that shared
subexpressions are computed once.

Two design decisions distinguish this from eager approaches:

1. **Structural hashing.** Two nodes describing the same computation compare
   equal and hash identically, regardless of construction order. This makes
   deduplication automatic and lets the evaluator cache across a whole search.

2. **Fail-fast unit checking.** When dimensional metadata is supplied, unit
   compatibility is validated at *construction* time, before any array work.
   Invalid combinations raise :class:`UnitError` immediately.

Numerical failures during evaluation are never silent. Nodes that overflow,
produce non-finite values, or degenerate to near-zero variance are recorded in
an :class:`EvaluationLog` with a reason, and excluded from results. The log is
part of the public API so users can audit what was rejected.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

import numpy as np

from beamfeat.scoring import ColumnStats, column_stats

__all__ = [
    "EvaluationLog",
    "Evaluator",
    "ExclusionReason",
    "Node",
    "NodeError",
    "OperatorSpec",
    "RejectedNode",
    "UnitError",
    "UnitRegistryError",
    "combine",
    "leaf",
    "transform",
]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class NodeError(ValueError):
    """Raised when a node cannot be constructed from the given arguments."""


class UnitError(NodeError):
    """Raised when an operation is dimensionally invalid.

    Raised at construction time, before any numerical work is performed.
    """


class UnitRegistryError(UnitError):
    """Raised when operands carry units from different pint registries.

    pint cannot multiply or divide quantities from two ``UnitRegistry``
    instances, so this is a configuration problem rather than a dimensional
    verdict on one candidate. :class:`Evaluator` converts the units it is given
    into a single registry, so it only arises for nodes built directly with
    :func:`leaf` from mixed registries.
    """


# --------------------------------------------------------------------------- #
# Exclusion reasons
# --------------------------------------------------------------------------- #


class ExclusionReason(str, Enum):
    """Why an evaluated node was excluded from the candidate pool.

    Recorded rather than silently dropped, so that rejections are auditable.
    """

    NON_FINITE = "non_finite"
    """Evaluation produced NaN or infinite values."""

    OVERFLOW = "overflow"
    """Evaluation, or the variance of its result, overflowed."""

    UNDERFLOW = "underflow"
    """Evaluation triggered a floating-point underflow."""

    DOMAIN_ERROR = "domain_error"
    """Input fell outside the operator's valid domain (e.g. log of a negative)."""

    ZERO_VARIANCE = "zero_variance"
    """Result is constant, or within tolerance of constant."""

    DUPLICATE = "duplicate"
    """Result is numerically identical to an already-accepted feature, such as
    ``abs`` of an operand that is never negative."""


@dataclass(frozen=True, slots=True)
class RejectedNode:
    """A node that was evaluated but excluded, with the reason why."""

    node: Node
    reason: ExclusionReason
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        suffix = f": {self.detail}" if self.detail else ""
        return f"{self.node.name} [{self.reason.value}]{suffix}"


class EvaluationLog:
    """Record of nodes excluded during evaluation.

    Exposed on :class:`Evaluator` so callers can inspect why candidates were
    dropped. Supports iteration, ``len``, and filtering by reason.
    """

    __slots__ = ("_generation", "_records")

    def __init__(self) -> None:
        self._records: list[RejectedNode] = []
        self._generation = 0

    def record(self, node: Node, reason: ExclusionReason, detail: str = "") -> None:
        """Append a rejection record."""
        self._records.append(RejectedNode(node=node, reason=reason, detail=detail))

    def by_reason(self, reason: ExclusionReason) -> list[RejectedNode]:
        """Return all records matching ``reason``."""
        return [r for r in self._records if r.reason is reason]

    def counts(self) -> dict[ExclusionReason, int]:
        """Return a count of rejections per reason."""
        out: dict[ExclusionReason, int] = {}
        for record in self._records:
            out[record.reason] = out.get(record.reason, 0) + 1
        return out

    def clear(self) -> None:
        """Discard all records."""
        self._records.clear()
        self._generation += 1

    def __iter__(self) -> Iterator[RejectedNode]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"EvaluationLog({len(self._records)} rejections: {self.counts()})"


# --------------------------------------------------------------------------- #
# Operator specifications
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Description of a feature-construction operator.

    Attributes:
        name: Identifier used in node keys and formulas, e.g. ``"log"``.
        arity: Number of operands (1 for transforms, 2 for combinations).
        fn: Callable applied to numpy arrays.
        formula: Callable producing a display string from operand names.
        unit_fn: Callable propagating units. Receives operand units and returns
            the result unit. Raises to signal dimensional invalidity. If None,
            the operator is treated as unit-preserving for arity 1 and
            unit-requiring-match for arity 2.
        domain: Optional predicate on input arrays; returning False marks the
            operand as outside the operator's valid domain.
        commutative: Whether operand order is irrelevant. Commutative operands
            are sorted during hashing so ``x+y`` and ``y+x`` unify.
        redundant: Optional predicate on input arrays; returning True marks the
            result as identical to its operand (``abs`` of a column that is
            never negative), so it is excluded as a duplicate.
        interval: Optional bound on the result. Receives one ``(lo, hi)``
            interval per operand and returns an interval containing every
            value the operator yields, as computed in float64, for operands in
            those intervals. Used by the ``safe_domain`` search; see
            :func:`interval_of`.
        interval_domain: Optional predicate on operand intervals; returning
            True means the operator is defined on every value in them. An
            operator with a ``domain`` needs one to be admitted under
            ``safe_domain``.
    """

    name: str
    arity: int
    fn: Callable[..., np.ndarray]
    formula: Callable[..., str]
    unit_fn: Callable[..., Any] | None = None
    domain: Callable[..., bool] | None = None
    commutative: bool = False
    redundant: Callable[..., bool] | None = None
    interval: Callable[..., tuple[float, float]] | None = None
    interval_domain: Callable[..., bool] | None = None

    def __post_init__(self) -> None:
        if self.arity not in (1, 2):
            raise NodeError(f"operator {self.name!r} has unsupported arity {self.arity}")
        if self.commutative and self.arity != 2:
            raise NodeError(f"operator {self.name!r} cannot be commutative with arity {self.arity}")


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Node:
    """An immutable description of how to compute a feature.

    Nodes are compared and hashed *structurally*: two nodes describing the same
    computation are equal even if constructed independently. This is what makes
    subexpression sharing and deduplication automatic.

    Nodes are not constructed directly in normal use. Use :func:`leaf`,
    :func:`transform`, and :func:`combine`, which handle canonicalisation and
    unit propagation.

    Attributes:
        op: Operator name, or ``"leaf"`` for input columns.
        children: Operand nodes. Empty for leaves.
        column: Source column name. Set only for leaves.
        unit: Optional pint quantity describing the result's dimensions.
        depth: Longest path to a leaf. Leaves have depth 0.
        _key: Cached canonical structural key. Used for hashing and equality.
    """

    op: str
    children: tuple[Node, ...] = ()
    column: str | None = None
    unit: Any = None
    depth: int = 0
    _key: str = field(default="", compare=True, repr=False)

    # -- structural identity ------------------------------------------------ #

    @staticmethod
    def _canonical_key(op: str, children: Sequence[Node], column: str | None, commutative: bool) -> str:
        """Build a canonical string key for structural hashing.

        Commutative operands are sorted so that ``x+y`` and ``y+x`` produce the
        same key and therefore unify to a single node. Backslashes, parentheses
        and commas in column and operator names are escaped, so a column named
        ``"a,leaf:b"`` cannot make two different expressions share a key.
        """
        if not children:
            return f"leaf:{_escape_key(str(column))}"
        keys = [c._key for c in children]
        if commutative:
            keys = sorted(keys)
        return f"{_escape_key(op)}({','.join(keys)})"

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self._key == other._key

    # -- introspection ------------------------------------------------------ #

    @property
    def key(self) -> str:
        """Canonical structural key. Stable across processes."""
        return self._key

    @property
    def is_leaf(self) -> bool:
        """Whether this node is an input column."""
        return not self.children

    @property
    def name(self) -> str:
        """Human-readable formula for this node."""
        if self.is_leaf:
            return str(self.column)
        spec = OPERATORS.get(self.op)
        if spec is None:  # pragma: no cover - defensive
            child_names = ", ".join(c.name for c in self.children)
            return f"{self.op}({child_names})"
        return spec.formula(*(c.name for c in self.children))

    @property
    def size(self) -> int:
        """Number of nodes in the DAG rooted here, counting shared nodes once."""
        return len(self.subnodes())

    def subnodes(self) -> set[Node]:
        """Return every distinct node in the DAG rooted at this node."""
        seen: set[Node] = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(node.children)
        return seen

    def leaves(self) -> set[Node]:
        """Return the distinct leaf nodes this expression depends on."""
        return {n for n in self.subnodes() if n.is_leaf}

    def columns(self) -> set[str]:
        """Return the names of source columns this expression depends on."""
        return {str(n.column) for n in self.leaves()}

    def to_sympy(self, symbols: Mapping[str, Any] | None = None) -> Any:
        """Convert to a sympy expression for display or simplification.

        Imported lazily: sympy is only needed when a user asks for a symbolic
        form, so it stays off the hot path during search.

        Args:
            symbols: Optional mapping of column name to the sympy symbol used
                for it. By default each column gets a real symbol named after
                it, sanitised into an identifier; columns whose sanitised names
                coincide are given distinct numeric suffixes.
        """
        if symbols is None:
            symbols = _sympy_symbols(self.columns())
        return self._to_sympy(symbols)

    def _to_sympy(self, symbols: Mapping[str, Any]) -> Any:
        if self.is_leaf:
            return symbols[str(self.column)]
        args = [c._to_sympy(symbols) for c in self.children]
        builder = _SYMPY_BUILDERS.get(self.op)
        if builder is None:  # pragma: no cover - defensive
            import sympy

            return sympy.Function(self.op)(*args)
        return builder(*args)

    @property
    def n_operators(self) -> int:
        """Number of operator applications in the expression (leaves count 0).

        Distinct from :attr:`size`, which counts DAG nodes including leaves
        with shared subexpressions counted once; this counts operator
        applications in the expression tree, the quantity used as the
        parsimony tie-break in beam ranking.
        """
        if self.op == "leaf":
            return 0
        return 1 + sum(child.n_operators for child in self.children)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Node({self.name!r}, depth={self.depth})"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name


_KEY_ESCAPES = str.maketrans({"\\": "\\\\", "(": "\\(", ")": "\\)", ",": "\\,"})


def _escape_key(name: str) -> str:
    """Escape the characters that delimit structural keys."""
    return name.translate(_KEY_ESCAPES)


def _plain_key(node: Node) -> str:
    """The structural key with names left unescaped."""
    if node.is_leaf:
        return f"leaf:{node.column}"
    keys = [_plain_key(c) for c in node.children]
    spec = OPERATORS.get(node.op)
    if spec is not None and spec.commutative:
        keys = sorted(keys)
    return f"{node.op}({','.join(keys)})"


def _operand_order(node: Node) -> tuple[str, str]:
    """Sort key placing the operands of a commutative operator.

    Ordering by the unescaped key keeps operand order, and with it every
    printed formula, the same as before keys were escaped; the escaped key
    breaks the ties between names the unescaped form cannot tell apart.
    """
    if "\\" not in node._key:
        return node._key, node._key
    return _plain_key(node), node._key


def _sympy_safe(name: str) -> str:
    """Sanitise a column name into a valid sympy symbol name."""
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"x_{cleaned}"
    return cleaned


def _sympy_symbols(columns: Iterable[str]) -> dict[str, Any]:
    """Map column names to distinct real sympy symbols.

    Names that are already valid identifiers keep them. Others are sanitised,
    and a sanitised name that is already taken gets a numeric suffix, so two
    columns never share a symbol.
    """
    import sympy

    ordered = sorted(set(columns))
    names = {column: column for column in ordered if _sympy_safe(column) == column}
    taken = set(names.values())
    for column in ordered:
        if column in names:
            continue
        base = candidate = _sympy_safe(column)
        suffix = 1
        while candidate in taken:
            suffix += 1
            candidate = f"{base}_{suffix}"
        names[column] = candidate
        taken.add(candidate)
    return {column: sympy.Symbol(name, real=True) for column, name in names.items()}


# --------------------------------------------------------------------------- #
# Unit propagation
# --------------------------------------------------------------------------- #


def _dimensionless_required(unit: Any, op_name: str) -> None:
    """Raise if ``unit`` is not dimensionless."""
    if unit is None:
        return
    try:
        if not unit.dimensionless:
            raise UnitError(f"{op_name} requires a dimensionless argument, got {unit.units}")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise UnitError(f"cannot check dimensionality for {op_name}: {unit!r}") from exc


def _same_dimension_required(left: Any, right: Any, op_name: str) -> None:
    """Raise if ``left`` and ``right`` have differing dimensionality."""
    if left is None or right is None:
        return
    try:
        compatible = left.dimensionality == right.dimensionality
    except AttributeError as exc:  # pragma: no cover - defensive
        raise UnitError(f"cannot compare dimensionality for {op_name}") from exc
    if not compatible:
        raise UnitError(f"{op_name} requires matching dimensions, got {left.units} and {right.units}")


def _coerce_unit(value: Any, registry: Any = None) -> Any:
    """Coerce a user-supplied unit into a pint-Quantity-compatible object.

    Accepted forms, in order of checking:

    - ``None`` — dimensionless, passed through.
    - A Quantity-like object (has ``units`` and ``dimensionality``) — passed
      through.
    - A pint ``Unit`` (has ``dimensionality`` but no magnitude) — promoted to
      a Quantity by multiplying with 1.
    - A string such as ``"kg"`` or ``"m / s**2"`` — parsed with ``registry``
      when given (so strings stay consistent with any Quantities supplied
      alongside them), otherwise with pint's application registry. Requires
      pint, which is an optional dependency; a missing install raises a
      ValueError naming the fix rather than an ImportError from deep inside
      unit propagation.

    When ``registry`` is given, Quantities and Units built in a different pint
    registry are re-expressed in it, since pint cannot multiply or divide
    across registries.

    Anything else raises ``ValueError`` naming the offending value, so a typo
    fails at fit time with a readable message instead of a ``TypeError`` deep
    in an operator's unit function.
    """
    if value is None:
        return None
    if hasattr(value, "units") and hasattr(value, "dimensionality"):
        return _in_registry(value, registry)
    if hasattr(value, "dimensionality"):
        try:
            quantity = 1 * value
        except TypeError:  # pragma: no cover - defensive
            pass
        else:
            return _in_registry(quantity, registry)
    if isinstance(value, str):
        try:
            import pint
        except ImportError:
            raise ValueError(
                f"unit {value!r} was given as a string, which requires pint to parse. "
                "Install it (pip install 'beamfeat[units]') or pass pint Quantities "
                "directly."
            ) from None
        active_registry = registry if registry is not None else pint.get_application_registry()
        try:
            quantity = active_registry.parse_expression(value)
        except Exception as exc:
            raise ValueError(f"could not parse unit string {value!r}: {exc}") from None
        return quantity if hasattr(quantity, "units") else quantity * active_registry.dimensionless
    raise ValueError(
        f"unsupported unit specification {value!r} of type {type(value).__name__}; "
        "pass a pint Quantity (1 * ureg.kg), a pint Unit, a unit string ('kg'), or None"
    )


def _in_registry(quantity: Any, registry: Any) -> Any:
    """Re-express ``quantity`` in ``registry`` if it was built in another one."""
    own = getattr(quantity, "_REGISTRY", None)
    if registry is None or own is None or own is registry:
        return quantity
    try:
        converted = registry.Quantity(quantity.magnitude, quantity.units)
        converted.dimensionality  # raises for a unit this registry does not define
        return converted
    except Exception as exc:
        raise UnitRegistryError(
            f"unit {quantity!r} comes from a different pint registry and could not be converted: {exc}"
        ) from None


def _registry_of(units: dict[str, Any] | None) -> Any:
    """Registry of the first Quantity in ``units``, so strings parse consistently."""
    if not units:
        return None
    for value in units.values():
        registry = getattr(value, "_REGISTRY", None)
        if registry is not None:
            return registry
    return None


def _normalise_unit(unit: Any) -> Any:
    """Strip magnitude so units carry dimensionality only."""
    if unit is None:
        return None
    try:
        return unit / unit.magnitude if unit.magnitude else unit
    except (AttributeError, ZeroDivisionError):  # pragma: no cover - defensive
        return unit


# --------------------------------------------------------------------------- #
# Operator registry
# --------------------------------------------------------------------------- #


def _unit_preserving(unit: Any) -> Any:
    return unit


# Interval bounds -------------------------------------------------------------- #
#
# Each function maps operand intervals to an interval containing every value
# the operator yields on them. Addition, subtraction, multiplication, division,
# reciprocal, square root and squaring are correctly rounded in IEEE
# arithmetic, and rounding is monotone, so the operator applied to the
# endpoints bounds the computed values exactly. log, exp and cube go through
# implementations that may be off by a few units in the last place, so their
# bounds are widened by INTERVAL_ULPS. Outside an operator's domain the
# functions return the unbounded interval; the domain is checked separately.

Interval = tuple[float, float]

INTERVAL_ULPS = 8

_UNBOUNDED: Interval = (-math.inf, math.inf)


def _widen(lo: float, hi: float) -> Interval:
    """Widen an interval outward by :data:`INTERVAL_ULPS` units in the last place."""
    if math.isfinite(lo):
        lo -= INTERVAL_ULPS * math.ulp(lo)
    if math.isfinite(hi):
        hi += INTERVAL_ULPS * math.ulp(hi)
    return lo, hi


def _exp_bound(x: float) -> float:
    return math.inf if x > 709.0 else math.exp(x)


def _interval_reciprocal(a: Interval) -> Interval:
    lo, hi = a
    if lo > 0 or hi < 0:
        return 1.0 / hi, 1.0 / lo
    return _UNBOUNDED


def _interval_log(a: Interval) -> Interval:
    lo, hi = a
    if lo <= 0:
        return _UNBOUNDED
    return _widen(math.log(lo), math.log(hi))


def _interval_sqrt(a: Interval) -> Interval:
    lo, hi = a
    if lo < 0:
        return _UNBOUNDED
    return math.sqrt(lo), math.sqrt(hi)


def _interval_abs(a: Interval) -> Interval:
    lo, hi = a
    if lo >= 0:
        return lo, hi
    if hi <= 0:
        return -hi, -lo
    return 0.0, max(-lo, hi)


def _interval_square(a: Interval) -> Interval:
    lo, hi = _interval_abs(a)
    return lo * lo, hi * hi


def _interval_cube(a: Interval) -> Interval:
    # Widening must not move a bound across zero: the cube has its operand's
    # sign exactly, and sqrt((x)^3) of a non-negative x is defined.
    lo, hi = _widen(a[0] * a[0] * a[0], a[1] * a[1] * a[1])
    return (max(lo, 0.0) if a[0] >= 0 else lo), (min(hi, 0.0) if a[1] <= 0 else hi)


def _interval_exp(a: Interval) -> Interval:
    lo, hi = _widen(_exp_bound(a[0]), _exp_bound(a[1]))
    return max(lo, 0.0), hi


def _interval_add(a: Interval, b: Interval) -> Interval:
    lo, hi = a[0] + b[0], a[1] + b[1]
    return (-math.inf if math.isnan(lo) else lo), (math.inf if math.isnan(hi) else hi)


def _interval_sub(a: Interval, b: Interval) -> Interval:
    lo, hi = a[0] - b[1], a[1] - b[0]
    return (-math.inf if math.isnan(lo) else lo), (math.inf if math.isnan(hi) else hi)


def _interval_mul(a: Interval, b: Interval) -> Interval:
    # 0 * inf bounds a product of zero with an unbounded factor, which is 0.
    corners = [0.0 if x == 0 or y == 0 else x * y for x in a for y in b]
    return min(corners), max(corners)


def _interval_div(a: Interval, b: Interval) -> Interval:
    if not (b[0] > 0 or b[1] < 0):
        return _UNBOUNDED
    corners: list[float] = []
    for x in a:
        for y in b:
            if math.isinf(x) and math.isinf(y):
                # Both unbounded: the quotient ranges over (0, inf) in the
                # sign of the corner.
                corners.extend((0.0, math.copysign(math.inf, x) * math.copysign(1.0, y)))
            else:
                corners.append(x / y)
    return min(corners), max(corners)


def _excludes_zero(a: Interval) -> bool:
    return a[0] > 0 or a[1] < 0


OPERATORS: dict[str, OperatorSpec] = {}


def register_operator(spec: OperatorSpec) -> OperatorSpec:
    """Add an operator to the global registry.

    Raises:
        NodeError: If an operator with the same name is already registered.
    """
    if spec.name in OPERATORS:
        raise NodeError(f"operator {spec.name!r} is already registered")
    OPERATORS[spec.name] = spec
    return spec


def _reciprocal_unit(unit: Any) -> Any:
    return None if unit is None else 1 / unit


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


def _add_unit(left: Any, right: Any) -> Any:
    _same_dimension_required(left, right, "addition")
    return left if left is not None else right


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
        domain=lambda x: bool(np.all(x != 0)),
        interval=_interval_reciprocal,
        interval_domain=_excludes_zero,
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
        interval=_interval_log,
        interval_domain=lambda a: a[0] > 0,
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
        interval=_interval_sqrt,
        interval_domain=lambda a: a[0] >= 0,
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
        interval=_interval_abs,
    )
)

register_operator(
    OperatorSpec(
        name="square",
        arity=1,
        fn=lambda x: np.square(x),
        formula=lambda x: f"({x})^2",
        unit_fn=_square_unit,
        interval=_interval_square,
    )
)

register_operator(
    OperatorSpec(
        name="cube",
        arity=1,
        fn=lambda x: x**3,
        formula=lambda x: f"({x})^3",
        unit_fn=_cube_unit,
        interval=_interval_cube,
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
        interval=_interval_exp,
        interval_domain=lambda a: a[1] < 50.0,
    )
)

# Binary operators ---------------------------------------------------------- #

register_operator(
    OperatorSpec(
        name="add",
        arity=2,
        fn=lambda x, y: x + y,
        formula=lambda x, y: f"({x} + {y})",
        unit_fn=_add_unit,
        commutative=True,
        interval=_interval_add,
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
        interval=_interval_sub,
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
        interval=_interval_mul,
    )
)

register_operator(
    OperatorSpec(
        name="div",
        arity=2,
        fn=lambda x, y: x / y,
        formula=lambda x, y: f"({x} / {y})",
        unit_fn=_div_unit,
        commutative=False,
        interval=_interval_div,
        interval_domain=lambda a, b: _excludes_zero(b),
    )
)


def _read_only(values: np.ndarray) -> np.ndarray:
    """A read-only view of ``values``, leaving the array itself writable."""
    view = values.view()
    view.setflags(write=False)
    return view


def _sympy_builders() -> dict[str, Callable[..., Any]]:
    """Lazily construct sympy builders so sympy is imported only on demand."""
    import sympy

    return {
        "reciprocal": lambda x: 1 / x,
        "log": sympy.log,
        "sqrt": sympy.sqrt,
        "abs": sympy.Abs,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "exp": sympy.exp,
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
    }


class _LazySympyBuilders(Mapping):
    """Mapping that materialises sympy builders on first access."""

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[str, Callable[..., Any]] | None = None

    def _ensure(self) -> dict[str, Callable[..., Any]]:
        if self._cache is None:
            self._cache = _sympy_builders()
        return self._cache

    def __getitem__(self, key: str) -> Callable[..., Any]:
        return self._ensure()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ensure())

    def __len__(self) -> int:
        return len(self._ensure())


_SYMPY_BUILDERS: Mapping[str, Callable[..., Any]] = _LazySympyBuilders()


# --------------------------------------------------------------------------- #
# Node constructors
# --------------------------------------------------------------------------- #


def leaf(column: str, unit: Any = None) -> Node:
    """Create a leaf node referring to an input column.

    Args:
        column: Name of the source column.
        unit: Optional pint quantity describing the column's dimensions.

    Returns:
        A depth-0 :class:`Node`.
    """
    if not isinstance(column, str) or not column:
        raise NodeError(f"leaf column must be a non-empty string, got {column!r}")
    key = Node._canonical_key("leaf", (), column, commutative=False)
    return Node(op="leaf", children=(), column=column, unit=_normalise_unit(_coerce_unit(unit)), depth=0, _key=key)


def transform(op: str, child: Node) -> Node:
    """Apply a unary operator to a node.

    Unit compatibility is checked here, before any numerical work.

    Args:
        op: Registered unary operator name.
        child: Operand node.

    Returns:
        A new :class:`Node` one level deeper than ``child``.

    Raises:
        NodeError: If ``op`` is unknown or has the wrong arity.
        UnitError: If the operation is dimensionally invalid.
    """
    spec = OPERATORS.get(op)
    if spec is None:
        raise NodeError(f"unknown operator {op!r}")
    if spec.arity != 1:
        raise NodeError(f"operator {op!r} expects {spec.arity} operands, got 1")
    if not isinstance(child, Node):
        raise NodeError(f"transform operand must be a Node, got {type(child).__name__}")

    # Local algebraic normalisation. Only domain-preserving-or-widening
    # rewrites are applied: the rewritten expression is defined wherever the
    # original was (so the record-and-exclude evaluation semantics can only
    # become more permissive, never silently stricter, with the one exception
    # noted at sqrt(x^2)), and the two are equal everywhere the original is
    # defined. Full algebraic canonicalisation is deliberately out of scope;
    # these rules target the spellings the search actually produces, so that
    # ``a / (1/b)`` and ``a * b`` unify to one node instead of wasting a beam
    # slot and uglifying output.
    if op == "reciprocal":
        if child.op == "reciprocal":  # 1/(1/x) -> x  (domain equal: x != 0)
            return child.children[0]
        if child.op == "div":  # 1/(a/b) -> b/a  (domain equal: a != 0, b != 0)
            return combine("div", child.children[1], child.children[0])
    if op == "sqrt" and child.op == "square":
        # sqrt(x^2) -> |x|. Defined wherever the original is, and also where
        # x^2 overflows. Stricter in one case: when x is never negative, |x| is
        # excluded as a duplicate of x, which sqrt(x^2) would not have been.
        return transform("abs", child.children[0])
    if op == "abs" and child.op == "abs":  # ||x|| -> |x|
        return child

    unit = spec.unit_fn(child.unit) if spec.unit_fn is not None else child.unit
    key = Node._canonical_key(op, (child,), None, commutative=False)
    return Node(op=op, children=(child,), column=None, unit=_normalise_unit(unit), depth=child.depth + 1, _key=key)


def combine(op: str, left: Node, right: Node) -> Node:
    """Apply a binary operator to two nodes.

    Unit compatibility is checked here, before any numerical work. For
    commutative operators, operands are canonicalised so that ``combine("add",
    x, y)`` and ``combine("add", y, x)`` return equal nodes.

    Args:
        op: Registered binary operator name.
        left: First operand.
        right: Second operand.

    Returns:
        A new :class:`Node`.

    Raises:
        NodeError: If ``op`` is unknown, has the wrong arity, or the operands
            are identical.
        UnitError: If the operation is dimensionally invalid.
    """
    spec = OPERATORS.get(op)
    if spec is None:
        raise NodeError(f"unknown operator {op!r}")
    if spec.arity != 2:
        raise NodeError(f"operator {op!r} expects {spec.arity} operands, got 2")
    if not isinstance(left, Node) or not isinstance(right, Node):
        raise NodeError("combine operands must both be Node instances")
    if left == right:
        raise NodeError(f"refusing to combine a node with itself under {op!r}: {left.name}")

    # Local algebraic normalisation (see `transform` for the soundness rule:
    # rewrites never shrink the domain). Each rewrite delegates back to the
    # factories, so unit propagation and commutative canonicalisation are
    # recomputed rather than assumed. A rewrite that would combine a node with
    # itself falls through to the original spelling instead of failing.
    try:
        if op == "div" and right.op == "reciprocal":
            inner = right.children[0]
            #  a / (1/b) -> a*b ; x / (1/x) -> x^2
            return transform("square", left) if left == inner else combine("mul", left, inner)
        if op == "div" and left.op == "reciprocal":
            #  (1/a) / b -> 1/(a*b)   (domain equal: a != 0, b != 0)
            # Only when b is no deeper than a: otherwise the folded form is a
            # level deeper than the original and can exceed the search depth.
            inner = left.children[0]
            if inner != right and right.depth <= inner.depth:
                return transform("reciprocal", combine("mul", inner, right))
        if op == "mul":
            for first, second in ((left, right), (right, left)):
                if first.op == "reciprocal" and first.children[0] != second:
                    #  (1/a) * b -> b/a  (domain equal: a != 0)
                    return combine("div", second, first.children[0])
    except NodeError:
        pass

    # Canonicalise operand order for commutative operators so that structurally
    # identical expressions unify to one node.
    if spec.commutative and _operand_order(right) < _operand_order(left):
        left, right = right, left

    unit = spec.unit_fn(left.unit, right.unit) if spec.unit_fn is not None else _add_unit(left.unit, right.unit)
    key = Node._canonical_key(op, (left, right), None, commutative=spec.commutative)
    return Node(
        op=op,
        children=(left, right),
        column=None,
        unit=_normalise_unit(unit),
        depth=max(left.depth, right.depth) + 1,
        _key=key,
    )


def interval_of(node: Node, operands: Sequence[Interval]) -> Interval | None:
    """Bound the values ``node`` takes when each operand lies in its interval.

    Args:
        node: A non-leaf node.
        operands: One ``(lo, hi)`` interval per child of ``node``.

    Returns:
        An interval containing every value ``node`` can take on such
        operands, or ``None`` when no such bound is available: the operator
        may be undefined somewhere in the operand intervals, it has no
        :attr:`~OperatorSpec.interval`, or it has a ``domain`` but no
        :attr:`~OperatorSpec.interval_domain`. ``None`` is the conservative
        answer, so an operator registered without interval functions is never
        admitted under ``safe_domain``.
    """
    # Operand intervals are passed positionally. An operator whose node
    # carries parameters of its own, such as a fitted constant, gets them
    # from ``node`` here, after the operands.
    spec = OPERATORS.get(node.op)
    if spec is None or spec.interval is None:
        return None
    if spec.interval_domain is not None:
        if not spec.interval_domain(*operands):
            return None
    elif spec.domain is not None:
        return None
    lo, hi = spec.interval(*operands)
    if math.isnan(lo) or math.isnan(hi):
        return None
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


class Evaluator:
    """Lazily evaluates nodes against a data matrix, caching by structural key.

    The cache is bounded and least-recently-used. Leaf columns are pinned and
    never evicted, since they are the base of every expression. Intermediate
    results are evicted under memory pressure and recomputed if needed again.

    Numerical failures are recorded in :attr:`log` and reported by returning
    ``None`` from :meth:`evaluate`; they never propagate as exceptions and are
    never silently discarded. Each failing node is recorded once: evaluating it
    again returns ``None`` without a second record, until the log is cleared.

    Returned arrays are read-only, since they are shared with the cache.

    Args:
        data: Mapping of column name to 1-D array, or a 2-D array with
            ``columns`` supplied separately.
        columns: Column names, required when ``data`` is a 2-D array.
        units: Optional mapping of column name to pint quantity.
        dtype: Working dtype. Defaults to float64; float32 halves memory but
            makes overflow substantially more likely in deep expressions.
        cache_size: Maximum number of non-leaf arrays held in the cache.
        variance_tol: Relative constancy threshold. A column is treated as
            constant when its variance falls below ``variance_tol`` times the
            square of its mean magnitude, so the verdict does not depend on
            the unit the caller happened to choose. Defaults to 1e-24 for
            float64, a spread twelve orders below the magnitude: the cut
            :func:`~beamfeat.scoring.is_constant` applies, so the columns the
            search drops as constant are the ones the fit-time warning names.
            Narrower dtypes default to 1e-10, as their rounding noise alone
            exceeds that cut.
    """

    __slots__ = (
        "_cache",
        "_columns",
        "_dtype",
        "_failed",
        "_failed_stamp",
        "_leaves",
        "_n_rows",
        "_units",
        "cache_size",
        "log",
        "variance_tol",
    )

    def __init__(
        self,
        data: Mapping[str, np.ndarray] | np.ndarray,
        columns: Sequence[str] | None = None,
        units: Mapping[str, Any] | None = None,
        dtype: np.dtype | type = np.float64,
        cache_size: int = 4096,
        variance_tol: float | None = None,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self._dtype = np.dtype(dtype)
        self.cache_size = int(cache_size)
        if variance_tol is None:
            variance_tol = 1e-24 if self._dtype.itemsize >= 8 else 1e-10
        self.variance_tol = float(variance_tol)
        self.log = EvaluationLog()
        # Cached arrays, each with whether it passed the search-time filters.
        self._cache: OrderedDict[str, tuple[np.ndarray, bool]] = OrderedDict()
        # Keys of nodes already recorded as failed, each with whether the
        # failure came from a search-time filter (so it does not hold when the
        # filters are off). Tied to one generation of one log.
        self._failed: dict[str, bool] = {}
        self._failed_stamp: tuple[EvaluationLog, int] | None = None

        self._leaves, self._columns = self._ingest(data, columns)
        self._n_rows = len(next(iter(self._leaves.values()))) if self._leaves else 0
        registry = _registry_of(units)
        self._units = (
            {name: _coerce_unit(value, registry) for name, value in units.items()} if units else {}
        )

    def _ingest(
        self, data: Mapping[str, np.ndarray] | np.ndarray, columns: Sequence[str] | None
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Normalise input into a dict of contiguous 1-D arrays."""
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(f"array data must be 2-D, got {data.ndim}-D")
            if columns is None:
                columns = [f"x{i}" for i in range(data.shape[1])]
            if len(columns) != data.shape[1]:
                raise ValueError(f"got {len(columns)} column names for {data.shape[1]} columns")
            leaves = {
                str(c): _read_only(np.ascontiguousarray(data[:, i], dtype=self._dtype))
                for i, c in enumerate(columns)
            }
            return leaves, [str(c) for c in columns]

        if hasattr(data, "items"):
            leaves = {}
            lengths = set()
            for name, values in data.items():
                arr = np.ascontiguousarray(np.asarray(values), dtype=self._dtype)
                if arr.ndim != 1:
                    raise ValueError(f"column {name!r} must be 1-D, got {arr.ndim}-D")
                lengths.add(arr.shape[0])
                leaves[str(name)] = _read_only(arr)
            if len(lengths) > 1:
                raise ValueError(f"columns have differing lengths: {sorted(lengths)}")
            return leaves, list(leaves)

        raise TypeError(f"data must be a mapping or 2-D array, got {type(data).__name__}")

    # -- introspection ------------------------------------------------------ #

    @property
    def columns(self) -> list[str]:
        """Names of the available input columns."""
        return list(self._columns)

    @property
    def n_rows(self) -> int:
        """Number of rows in the data."""
        return self._n_rows

    @property
    def dtype(self) -> np.dtype:
        """Working dtype for evaluation."""
        return self._dtype

    def column(self, name: str) -> np.ndarray:
        """Values of an input column, read-only, in the working dtype."""
        return self._leaves[name]

    def leaf_nodes(self) -> list[Node]:
        """Return leaf nodes for every input column, with units attached."""
        return [leaf(name, self._units.get(name)) for name in self._columns]

    def cache_info(self) -> dict[str, int]:
        """Return current cache occupancy and capacity."""
        return {"size": len(self._cache), "capacity": self.cache_size, "pinned": len(self._leaves)}

    def clear_cache(self) -> None:
        """Evict all cached intermediate results. Leaves are retained."""
        self._cache.clear()

    # -- evaluation --------------------------------------------------------- #

    def evaluate(self, node: Node, apply_filters: bool = True) -> np.ndarray | None:
        """Compute the values of ``node``, using and updating the cache.

        Args:
            node: The expression to evaluate.
            apply_filters: Whether to apply the search-time admissibility
                filters — zero variance, and the operators' domain predicates.
                These express whether a candidate is *worth exploring*, and
                they are data-dependent: a column that varies across a training
                set may be constant within some subset of it. At transform time
                that must not change the result, so :meth:`transform_values`
                passes ``False`` and computes the expression unconditionally.
                Genuine numerical failures (overflow, non-finite results) are
                still caught either way.

        Returns:
            A 1-D array of length :attr:`n_rows`, or ``None`` if the node was
            excluded. When ``None`` is returned, the reason is appended to
            :attr:`log`.
        """
        if node.is_leaf:
            column = str(node.column)
            values = self._leaves.get(column)
            if values is None:
                raise KeyError(f"column {column!r} is not present in the data")
            if self._known_failure(node, apply_filters):
                return None
            if not np.all(np.isfinite(values)):
                n_bad = int(np.count_nonzero(~np.isfinite(values)))
                self._reject(node, ExclusionReason.NON_FINITE, f"{n_bad} non-finite value(s) in input column")
                return None
            if apply_filters:
                # Extreme-magnitude columns can overflow while computing
                # variance; treat that as an input-scale problem, not a crash.
                variance = self._variance(values)
                if not math.isfinite(variance):
                    self._reject(
                        node, ExclusionReason.OVERFLOW, "input column variance overflowed", filtered=True
                    )
                    return None
                if self._is_constant(values, variance):
                    self._reject(
                        node,
                        ExclusionReason.ZERO_VARIANCE,
                        f"input column variance={variance:.3g}",
                        filtered=True,
                    )
                    return None
            return values

        cached = self._cache.get(node.key)
        if cached is not None and (cached[1] or not apply_filters):
            # Values do not depend on the filters, but whether they were
            # checked does: an entry stored unfiltered is rechecked below.
            self._cache.move_to_end(node.key)
            return cached[0]
        if self._known_failure(node, apply_filters):
            return None

        spec = OPERATORS.get(node.op)
        if spec is None:  # pragma: no cover - defensive
            raise NodeError(f"unknown operator {node.op!r}")

        operands: list[np.ndarray] = []
        for child in node.children:
            child_values = self.evaluate(child, apply_filters=apply_filters)
            if child_values is None:
                # The child was already logged; propagate exclusion without
                # double-recording the same root cause.
                return None
            operands.append(child_values)

        if apply_filters and spec.domain is not None and not spec.domain(*operands):
            self._reject(
                node, ExclusionReason.DOMAIN_ERROR, f"input outside domain of {spec.name!r}", filtered=True
            )
            return None
        if apply_filters and spec.redundant is not None and spec.redundant(*operands):
            self._reject(
                node, ExclusionReason.DUPLICATE, f"{spec.name!r} leaves its operand unchanged", filtered=True
            )
            return None

        values = self._compute(node, spec, operands, apply_filters=apply_filters)
        if values is None:
            return None

        self._store(node.key, values, apply_filters)
        return values

    def transform_values(self, node: Node) -> np.ndarray | None:
        """Compute ``node`` without applying search-time admissibility filters.

        Use this when replaying an already-selected expression on new data.
        Applying the search filters here would make the output depend on which
        rows are present — a subset in which some column happens to be constant
        would silently yield a different answer than the full data.

        Returns:
            The computed values, or ``None`` if the expression genuinely cannot
            be evaluated (overflow, or a non-finite result).
        """
        return self.evaluate(node, apply_filters=False)

    def _compute(
        self, node: Node, spec: OperatorSpec, operands: list[np.ndarray], apply_filters: bool = True
    ) -> np.ndarray | None:
        """Apply the operator, converting numerical failures into log records."""
        with np.errstate(all="raise"):
            try:
                values = spec.fn(*operands)
            except FloatingPointError as exc:
                self._reject(node, _floating_point_reason(exc), str(exc))
                return None

        values = np.asarray(values, dtype=self._dtype)
        if values.ndim != 1 or values.shape[0] != self._n_rows:  # pragma: no cover - defensive
            raise NodeError(f"operator {spec.name!r} produced an array of shape {values.shape}")

        if not np.all(np.isfinite(values)):
            n_bad = int(np.count_nonzero(~np.isfinite(values)))
            self._reject(node, ExclusionReason.NON_FINITE, f"{n_bad} non-finite value(s)")
            return None

        if apply_filters:
            variance = self._variance(values)
            if not math.isfinite(variance):
                self._reject(node, ExclusionReason.OVERFLOW, "variance overflowed", filtered=True)
                return None
            if self._is_constant(values, variance):
                self._reject(node, ExclusionReason.ZERO_VARIANCE, f"variance={variance:.3g}", filtered=True)
                return None

        return _read_only(values)

    @staticmethod
    def _variance(values: np.ndarray) -> float:
        """Variance of ``values``; an overflow comes back as inf, without a warning."""
        with np.errstate(all="ignore"):
            return float(np.var(values))

    def _reject(self, node: Node, reason: ExclusionReason, detail: str, filtered: bool = False) -> None:
        """Record ``node`` as excluded, and remember it so it is recorded once.

        ``filtered`` marks a rejection by a search-time filter, which does not
        hold when :meth:`evaluate` is called with ``apply_filters=False``.
        """
        self.log.record(node, reason, detail)
        self._sync_failures()
        self._failed[node.key] = filtered

    def _known_failure(self, node: Node, apply_filters: bool) -> bool:
        """Whether ``node`` already failed in a way that holds for this call."""
        self._sync_failures()
        filtered = self._failed.get(node.key)
        return filtered is not None and (apply_filters or not filtered)

    def _sync_failures(self) -> None:
        """Forget remembered failures once their records leave the log."""
        stamp = (self.log, self.log._generation)
        if self._failed_stamp != stamp:
            self._failed.clear()
            self._failed_stamp = stamp

    def _is_constant(self, values: np.ndarray, variance: float) -> bool:
        """Scale-free constancy test.

        ``variance_tol`` is applied relative to the column's own magnitude.
        An absolute floor makes the verdict depend on the unit: concentrations
        in kmol/m^3 sit near 1e-6 and have variance ~1e-12, so they were
        rejected as constant while the identical data expressed in mol/m^3 was
        accepted. Worse, when only some columns fall below the floor the fit
        succeeds with the important column silently removed.

        A non-finite ``variance`` is an overflow, not constancy; callers
        reject it as such before asking.
        """
        with np.errstate(all="ignore"):
            scale = float(np.mean(np.abs(values)))
        return _is_constant(variance, scale, self.variance_tol)

    def _store(self, key: str, values: np.ndarray, filtered: bool) -> None:
        """Insert into the LRU cache, evicting the oldest entry if full."""
        self._cache[key] = (values, filtered)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def evaluate_many(self, nodes: Sequence[Node]) -> tuple[list[Node], np.ndarray]:
        """Evaluate several nodes, returning only those that survived.

        Args:
            nodes: Expressions to evaluate.

        Returns:
            A tuple of (surviving nodes, matrix with one column per surviving
            node in the same order). If nothing survives, the matrix has shape
            ``(n_rows, 0)``.
        """
        kept: list[Node] = []
        arrays: list[np.ndarray] = []
        for node in nodes:
            values = self.evaluate(node)
            if values is not None:
                kept.append(node)
                arrays.append(values)
        if not arrays:
            return [], np.empty((self._n_rows, 0), dtype=self._dtype)
        return kept, np.column_stack(arrays)

    # -- batches ------------------------------------------------------------ #

    def plan_batch(self, nodes: Sequence[Node]) -> EvaluationBatch:
        """Prepare distinct nodes for evaluation away from this evaluator.

        Evaluating a wide round of candidates one :meth:`evaluate` call at a
        time spends most of its time on per-column overhead, and cannot be
        shared between workers because the cache and the log are mutable. A
        batch splits the work in three: this method resolves each node's
        operands (and settles leaves, cached values and known failures) here;
        :meth:`EvaluationBatch.run` then applies the search-time filters and
        computes the values with no access to the evaluator, so batches can
        run concurrently; :meth:`commit_batch` records the rejections, in node
        order.

        Operands are resolved through :meth:`evaluate`, so a failing operand
        is recorded when the batch is planned, ahead of the batch's own
        rejections. Computed values are not cached: a round's candidates are
        each evaluated once, and the few kept are cached when next evaluated.

        Args:
            nodes: Expressions to evaluate. Each should appear once.

        Returns:
            An :class:`EvaluationBatch` for :meth:`EvaluationBatch.run`.
        """
        entries: list[Any] = []
        resolved: dict[str, np.ndarray | None] = {}
        self._sync_failures()
        for node in nodes:
            if node.is_leaf:
                entries.append(self.evaluate(node))
                continue
            key = node.key
            cached = self._cache.get(key)
            if cached is not None and cached[1]:
                self._cache.move_to_end(key)
                entries.append(cached[0])
                continue
            # With the filters on, every remembered failure holds.
            if key in self._failed:
                entries.append(None)
                continue
            spec = OPERATORS.get(node.op)
            if spec is None:  # pragma: no cover - defensive
                raise NodeError(f"unknown operator {node.op!r}")
            operands: list[np.ndarray] = []
            for child in node.children:
                if child.key not in resolved:
                    resolved[child.key] = self.evaluate(child)
                child_values = resolved[child.key]
                if child_values is None:
                    break
                operands.append(child_values)
            else:
                entries.append((spec, tuple(operands)))
                continue
            entries.append(None)
        return EvaluationBatch(entries, self._n_rows, self._dtype, self.variance_tol)

    def commit_batch(self, nodes: Sequence[Node], outcome: BatchOutcome) -> list[Node]:
        """Record a batch's rejections and return its surviving nodes.

        Args:
            nodes: The nodes passed to :meth:`plan_batch`, in the same order.
            outcome: What :meth:`EvaluationBatch.run` returned for them.

        Returns:
            The surviving nodes, in order, matching the columns of
            ``outcome.values``.
        """
        for position, reason, detail, filtered in outcome.rejections:
            self._reject(nodes[position], reason, detail, filtered=filtered)
        return [nodes[position] for position in outcome.kept]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Evaluator(n_rows={self._n_rows}, n_columns={len(self._columns)}, cache={len(self._cache)}/{self.cache_size})"


def _floating_point_reason(exc: FloatingPointError) -> ExclusionReason:
    """Classify a floating-point error raised while applying an operator."""
    message = str(exc).lower()
    if "overflow" in message:
        return ExclusionReason.OVERFLOW
    if "underflow" in message:
        return ExclusionReason.UNDERFLOW
    if "divide" in message or "invalid" in message:
        return ExclusionReason.DOMAIN_ERROR
    return ExclusionReason.NON_FINITE  # pragma: no cover - rare numpy conditions


def _is_constant(variance: float, magnitude: float, variance_tol: float) -> bool:
    """The :meth:`Evaluator._is_constant` verdict, from precomputed statistics."""
    if math.isfinite(magnitude) and magnitude > 0.0:
        return variance <= variance_tol * magnitude * magnitude
    return variance <= variance_tol


class BatchOutcome(NamedTuple):
    """Result of :meth:`EvaluationBatch.run`.

    Attributes:
        kept: Positions, within the batch, of the nodes that survived.
        values: ``(n_rows, len(kept))`` column-major matrix of their values,
            or ``None`` once a caller has dropped it.
        stats: :class:`~beamfeat.scoring.ColumnStats` of those columns.
        rejections: ``(position, reason, detail, filtered)`` for each
            rejected node, in position order.
    """

    kept: list[int]
    values: np.ndarray | None
    stats: ColumnStats
    rejections: list[tuple[int, ExclusionReason, str, bool]]


@dataclass(slots=True)
class EvaluationBatch:
    """Nodes planned by :meth:`Evaluator.plan_batch`, ready to compute.

    Holds no reference to the evaluator, so it can be run in another thread
    or process.

    Attributes:
        entries: Per node, either ``None`` (excluded at planning, with any
            record already made), an array (a value known at planning), or
            an ``(OperatorSpec, operands)`` pair to compute.
        n_rows: Rows per column.
        dtype: Working dtype.
        variance_tol: Relative constancy threshold, as on :class:`Evaluator`.
    """

    entries: list[Any]
    n_rows: int
    dtype: np.dtype
    variance_tol: float

    def run(self) -> BatchOutcome:
        """Apply the search-time filters and compute the surviving columns.

        The checks, their order and the rejection details are those of
        :meth:`Evaluator.evaluate`. The variance and magnitude behind the
        constancy check are kept as :class:`~beamfeat.scoring.ColumnStats`,
        so that a scorer can standardise the columns without computing
        them again.
        """
        matrix = np.empty((self.n_rows, len(self.entries)), dtype=self.dtype, order="F")
        positions: list[int] = []
        rejections: list[tuple[int, ExclusionReason, str, bool]] = []
        for position, entry in enumerate(self.entries):
            if entry is None:
                continue
            if isinstance(entry, np.ndarray):
                matrix[:, len(positions)] = entry
                positions.append(position)
                continue
            spec, operands = entry
            if spec.domain is not None and not spec.domain(*operands):
                rejections.append(
                    (position, ExclusionReason.DOMAIN_ERROR, f"input outside domain of {spec.name!r}", True)
                )
                continue
            if spec.redundant is not None and spec.redundant(*operands):
                rejections.append(
                    (position, ExclusionReason.DUPLICATE, f"{spec.name!r} leaves its operand unchanged", True)
                )
                continue
            with np.errstate(all="raise"):
                try:
                    values = spec.fn(*operands)
                except FloatingPointError as exc:
                    rejections.append((position, _floating_point_reason(exc), str(exc), False))
                    continue
            values = np.asarray(values, dtype=self.dtype)
            if values.ndim != 1 or values.shape[0] != self.n_rows:  # pragma: no cover - defensive
                raise NodeError(f"operator {spec.name!r} produced an array of shape {values.shape}")
            matrix[:, len(positions)] = values
            positions.append(position)

        matrix = matrix[:, : len(positions)]
        finite = np.isfinite(matrix).all(axis=0).tolist()
        stats = column_stats(matrix)
        variances = stats.variance.tolist()
        magnitudes = stats.magnitude.tolist()

        survivors: list[int] = []
        for column, position in enumerate(positions):
            variance = variances[column]
            if not finite[column]:
                n_bad = int(np.count_nonzero(~np.isfinite(matrix[:, column])))
                rejections.append((position, ExclusionReason.NON_FINITE, f"{n_bad} non-finite value(s)", False))
            elif not math.isfinite(variance):
                rejections.append((position, ExclusionReason.OVERFLOW, "variance overflowed", True))
            elif _is_constant(variance, magnitudes[column], self.variance_tol):
                rejections.append((position, ExclusionReason.ZERO_VARIANCE, f"variance={variance:.3g}", True))
            else:
                survivors.append(column)

        if len(survivors) < len(positions):
            matrix = matrix[:, survivors]
            stats = stats.take(survivors)
        rejections.sort(key=lambda rejection: rejection[0])
        return BatchOutcome([positions[column] for column in survivors], matrix, stats, rejections)
