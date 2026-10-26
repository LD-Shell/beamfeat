"""Screen another tool's equation on rows it did not see.

A symbolic regression tool (PySR, Operon, gplearn, PySINDy and the like)
reports an equation fitted on some rows, with no statement about whether its
terms are real. :func:`screen_equation` tests the equation's terms on other
rows, which the tool never saw. There the terms are fixed functions, so a
permutation test of each term against the target is valid in finite samples,
whatever the tool did to find them.

**The marginal test.** Each top-level additive term of the equation, as
written, is one candidate. Its null is that the term is independent of the
target. The test is the one beamfeat's own fits use,
:class:`~beamfeat.selection.PermutationSelector`: the term's absolute Pearson
correlation with the target against the same statistic on permuted targets,
with the add-one p-value. The Benjamini-Yekutieli step-up (``correction="by"``,
the default) controls the false discovery rate among the kept terms under
any dependence between their p-values; Benjamini-Hochberg (``"bh"``) needs
positive dependence.

**The input test** (``input_test=True``). The marginal test keeps a term such
as ``x0*x7`` when ``x0`` carries it, whatever ``x7`` is. The input test asks,
of each variable of each term that uses more than one, whether the term needs
it: the variable's rows are permuted inside the term alone and the term's
correlation with the target is recomputed, every pair against the same
permutations. A variable's null is that it is independent of the target and
of the term's other variables together. The p-value is the add-one estimator
over the fewest draws that let the strictest step-up threshold be reached (at
least 1,000), except that a pair stops at its tenth exceedance, after ``L``
draws, with ``p = 10 / L`` (Besag and Clifford's sequential estimator, valid
in the same sense). The step-up runs over every (term, variable) pair of
every term, kept by the marginal test or not, so which pairs are tested does
not depend on the marginal outcome. A term is kept when the marginal test
rejects its null and, with the input test, the input test rejects the null
of each of its variables. A variable correlated with another input is not
null in this sense, so a stand-in for a true input can have its null
rejected: no test of association can tell the two apart.

**What is never kept.** A term undefined (not finite, or not real) or constant
on any of the rows cannot be tested. It is reported as untestable and never
kept, and so is every pair of it in the input test.

**Coefficients.** The kept terms are reported as the tool wrote them. To use
them, refit their coefficients on the rows the tool searched, never on the
rows that screened them: a refit on the screen rows would reuse them.

**Few rows make a weak test.** A term the screen does not keep has no support
on these rows, which is not a finding that it is false. With a few dozen rows
a real term of modest effect is often not kept.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from beamfeat.selection import PermutationSelector

__all__ = ["EquationScreen", "screen_equation"]

EXCEEDANCES = 10  # Besag and Clifford's h: a pair stops at its tenth exceedance


def _import_sympy() -> Any:
    try:
        import sympy
    except ImportError:
        raise ImportError(
            "screen_equation() reads the equation with sympy, which is not installed. "
            "Install it with pip install 'beamfeat[display]'."
        ) from None
    return sympy


@dataclass(frozen=True)
class EquationScreen:
    """What :func:`screen_equation` found: each term, its p-value, and what was kept.

    Attributes:
        terms: The equation's top-level additive terms that use a variable, each
            as written, in sympy's order of the sum's terms. A constant term
            carries no variable and is left out.
        p_values: The marginal test's p-value of each term; NaN for a term that
            cannot be tested.
        marginal: Indices of the terms whose marginal null is rejected.
        kept: Indices of the kept terms: those of :attr:`marginal` and, with
            the input test, whose every variable's null is rejected as well.
        untestable: Indices of the terms undefined or constant on the rows.
        input_pairs: The input test's (term index, variable name) pairs, empty
            without the input test.
        input_p_values: The p-value of each pair.
        input_rejected: Positions in :attr:`input_pairs` of the pairs whose
            null is rejected.
        q: The false discovery rate level.
        correction: ``"by"`` (Benjamini-Yekutieli) or ``"bh"``
            (Benjamini-Hochberg).
        input_test: Whether the input test ran.
        n_rows: The number of rows the terms were tested on.
    """

    terms: tuple
    p_values: np.ndarray
    marginal: tuple[int, ...]
    kept: tuple[int, ...]
    untestable: tuple[int, ...]
    input_pairs: tuple[tuple[int, str], ...] = ()
    input_p_values: np.ndarray = field(default_factory=lambda: np.empty(0))
    input_rejected: tuple[int, ...] = ()
    q: float = 0.1
    correction: str = "by"
    input_test: bool = False
    n_rows: int = 0

    @property
    def kept_terms(self) -> list:
        """The kept terms, as the tool wrote them."""
        return [self.terms[i] for i in self.kept]

    @property
    def removed(self) -> tuple[int, ...]:
        """Indices of the terms not kept, untestable ones included."""
        return tuple(i for i in range(len(self.terms)) if i not in self.kept)

    def summary(self) -> str:
        """One line per term: its p-value, whether its nulls are rejected, and whether it is kept."""
        method = f"marginal {self.correction.upper()} at q = {self.q:g}" + (", input test" if self.input_test else "")
        lines = [f"screen_equation: {len(self.kept)} of {len(self.terms)} terms kept ({method}, {self.n_rows} rows)"]
        pairs: dict[int, list[str]] = {}
        rejected = set(self.input_rejected)
        for k, (index, name) in enumerate(self.input_pairs):
            pairs.setdefault(index, []).append(f"{name} {'rejected' if k in rejected else 'not rejected'}")
        for i, term in enumerate(self.terms):
            if i in self.untestable:
                verdict = "untestable"
            else:
                null = "rejected" if i in self.marginal else "not rejected"
                verdict = f"p = {self.p_values[i]:.4g}, null {null}"
                if i in pairs:
                    verdict += "; inputs: " + ", ".join(pairs[i])
            lines.append(f"  [{'kept' if i in self.kept else 'removed'}] {term}: {verdict}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"EquationScreen(kept={len(self.kept)}/{len(self.terms)}, correction={self.correction!r}, q={self.q})"


def screen_equation(
    equation: Any,
    X: Any,
    y: Any,
    *,
    q: float = 0.1,
    correction: str = "by",
    input_test: bool = False,
    feature_names: Any = None,
    random_state: int | None = 0,
) -> EquationScreen:
    """Test another tool's equation, term by term, on rows the tool did not see.

    See the module documentation for the tests, their nulls and what they
    guarantee. The guarantee needs rows the tool never saw: pass rows held out
    from its fit, not the rows it searched.

    Args:
        equation: The equation, as a sympy expression or a string sympy can
            read, in the names of ``X``'s columns (for instance a PySR model's
            ``sympy()``, which names the columns ``x0, x1, ...`` when it was
            fitted on an array); or a list of its terms, each tested whole. A
            list keeps a term such as ``2*(x0 + x1)`` as one, where reading
            the sum splits it in two.
        X: The held-out rows, one column per input: an array of shape
            ``(n_rows, n_inputs)`` or a pandas DataFrame.
        y: The target on those rows, finite, shape ``(n_rows,)``.
        q: The false discovery rate level, in (0, 1).
        correction: ``"by"`` (Benjamini-Yekutieli, any dependence) or ``"bh"``
            (Benjamini-Hochberg, positive dependence), for the terms and for
            the input test's pairs.
        input_test: Also test each variable inside each term of more than one.
        feature_names: The columns' names. Default: a DataFrame's column names,
            else ``x0, x1, ...``. Each name must be a Python identifier.
        random_state: Seed for the permutations.

    Returns:
        An :class:`EquationScreen`.

    Raises:
        ValueError: If the arguments are invalid, or the equation uses a name
            that is not one of the columns.
        ImportError: If sympy is not installed.
    """
    sympy = _import_sympy()
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    if correction not in ("by", "bh"):
        raise ValueError(f"correction must be 'by' or 'bh', got {correction!r}")
    names = _names(X, feature_names)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError(f"X must be 2-D with one column per name; got shape {X.shape} and {len(names)} names")
    if y.shape != (X.shape[0],) or not np.all(np.isfinite(y)):
        raise ValueError(f"y must be finite, of shape ({X.shape[0]},); got shape {y.shape}")
    columns = {name: j for j, name in enumerate(names)}
    terms = _terms(sympy, equation, names)
    unknown = sorted({str(s) for term in terms for s in term.free_symbols} - set(columns))
    if unknown:
        raise ValueError(f"the equation uses {', '.join(unknown)}, which {'is' if len(unknown) == 1 else 'are'} "
                         f"not among the columns ({', '.join(names)})")
    compiled = [_compiled(sympy, term, columns) for term in terms]
    values = [_values(function, X, indices, flag) for indices, function, flag in compiled]
    usable = [bool(np.all(np.isfinite(v)) and np.ptp(v) > 0) for v in values]
    p_values, marginal = _marginal(values, usable, y, q, correction, random_state)
    kept = marginal
    pairs: list[tuple[int, int]] = []
    pair_p, rejected = np.empty(0), []
    if input_test:
        pairs = [(i, j) for i, (indices, _, _) in enumerate(compiled) if len(indices) > 1 for j in indices]
        pair_p = _input_p_values(pairs, compiled, usable, values, X, y, q, correction, random_state)
        rejected = [int(k) for k in _step_up(pair_p, q, correction)]
        failed = {pairs[k][0] for k in range(len(pairs)) if k not in set(rejected)}
        kept = [i for i in marginal if i not in failed]
    return EquationScreen(
        terms=tuple(terms), p_values=p_values, marginal=tuple(marginal), kept=tuple(kept),
        untestable=tuple(i for i, u in enumerate(usable) if not u),
        input_pairs=tuple((i, names[j]) for i, j in pairs), input_p_values=pair_p, input_rejected=tuple(rejected),
        q=float(q), correction=correction, input_test=bool(input_test), n_rows=int(X.shape[0]),
    )


def _names(X: Any, feature_names: Any) -> list[str]:
    if feature_names is not None:
        names = [str(name) for name in feature_names]
    elif hasattr(X, "columns"):
        names = [str(name) for name in X.columns]
    else:
        names = [f"x{j}" for j in range(np.shape(X)[1] if np.ndim(X) == 2 else 0)]
    bad = [name for name in names if not name.isidentifier()]
    if bad:
        raise ValueError(f"column names must be Python identifiers to be read in an equation; got {bad}")
    if len(set(names)) < len(names):
        raise ValueError("column names must be distinct")
    return names


def _terms(sympy: Any, equation: Any, names: list[str]) -> list:
    """The terms to test, those that use a variable: the equation's top-level terms, or the items of a list.

    Each column name is read as a plain symbol, so that ``E`` is not Euler's number.
    """
    local = {name: sympy.Symbol(name) for name in names}

    def read(item: Any) -> Any:
        return sympy.sympify(item, locals=local) if isinstance(item, str) else sympy.sympify(item)

    if isinstance(equation, (list, tuple)):
        candidates = [read(item) for item in equation]
    else:
        candidates = list(sympy.Add.make_args(read(equation)))
    return [term for term in candidates if term.free_symbols]


def _compiled(sympy: Any, term: Any, columns: dict[str, int]) -> tuple[list[int], Any, bool]:
    """The columns a term reads, a NumPy function of them in that order, and whether it can be complex."""
    # Complex infinity (gplearn prints a protected division by zero as zoo) has no NumPy
    # spelling; it is undefined, so it becomes NaN like any other undefined value.
    term = term.xreplace({sympy.zoo: sympy.nan})
    symbols = sorted(term.free_symbols, key=lambda s: columns[str(s)])
    return [columns[str(s)] for s in symbols], sympy.lambdify(symbols, term, "numpy"), bool(term.has(sympy.I))


def _call(function: Any, arguments: list, complex_values: bool) -> np.ndarray:
    """The function on the given columns, NaN where it is undefined or not real."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = np.asarray(function(*arguments), dtype=np.complex128 if complex_values else np.float64)
    if np.iscomplexobj(values):
        values = np.where(np.abs(values.imag) > 0, np.nan, values.real)
    return np.where(np.isfinite(values), values, np.nan)


def _values(function: Any, X: np.ndarray, indices: list[int], complex_values: bool) -> np.ndarray:
    return np.broadcast_to(_call(function, [X[:, j] for j in indices], complex_values), (X.shape[0],)).astype(
        np.float64)


def _marginal(values: list, usable: list, y: np.ndarray, q: float, correction: str,
              random_state: int | None) -> tuple[np.ndarray, list[int]]:
    """Each term's marginal p-value (NaN where it cannot be tested) and the indices whose null is rejected."""
    p_values = np.full(len(values), np.nan)
    tested = [i for i, u in enumerate(usable) if u]
    if not tested:
        return p_values, []
    selector = PermutationSelector(target_fdr=q, correction=correction, random_state=random_state)
    result = selector.select(np.column_stack([values[i] for i in tested]), y)
    p_values[tested] = result.p_values
    return p_values, [tested[int(j)] for j in result.selected]


def _step_up(p_values: np.ndarray, q: float, correction: str) -> np.ndarray:
    """Positions rejected by the Benjamini-Hochberg or Benjamini-Yekutieli step-up."""
    m = len(p_values)
    if m == 0:
        return np.empty(0, dtype=int)
    c = sum(1.0 / k for k in range(1, m + 1)) if correction == "by" else 1.0
    order = np.argsort(p_values, kind="stable")
    passed = p_values[order] <= q * np.arange(1, m + 1) / (m * c)
    if not passed.any():
        return np.empty(0, dtype=int)
    return np.sort(order[: int(np.flatnonzero(passed).max()) + 1])


def _required_permutations(m: int, q: float, correction: str) -> int:
    c = sum(1.0 / k for k in range(1, m + 1)) if correction == "by" else 1.0
    # The smallest attainable p-value, 1 / (B + 1), must clear the strictest threshold q / (m c).
    return max(1000, math.ceil(2 * m * c / q))


def _correlations(values: np.ndarray, centred_y: np.ndarray, norm_y: float) -> np.ndarray:
    """|Pearson correlation| of each row of ``values`` with y; inf where a row is not finite everywhere."""
    finite = np.all(np.isfinite(values), axis=-1)
    safe = np.where(finite[..., None], values, 0.0)
    centred = safe - safe.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(centred, axis=-1) * norm_y
    out = np.divide(np.abs(centred @ centred_y), norms, out=np.zeros(values.shape[:-1]), where=norms > 0)
    return np.where(finite, out, np.inf)  # a draw on which the term is undefined counts as an exceedance


def _draw(function: Any, arguments: list, complex_values: bool, block: int, n: int) -> np.ndarray:
    """A term's values on a block of draws, shape (block, n)."""
    try:
        return np.broadcast_to(_call(function, arguments, complex_values), (block, n))
    except (ValueError, TypeError):  # a function NumPy cannot broadcast (Max, Min): one draw at a time
        return np.vstack([np.broadcast_to(_call(function, [a[r] if a.ndim == 2 else a for a in arguments],
                                                complex_values), (n,)) for r in range(block)])


def _input_p_values(pairs: list[tuple[int, int]], compiled: list, usable: list, values: list, X: np.ndarray,
                    y: np.ndarray, q: float, correction: str, random_state: int | None) -> np.ndarray:
    """The input test's p-value of each (term, column) pair; 1 for a pair of a term that cannot be tested."""
    if not pairs:
        return np.empty(0)
    n = len(y)
    B = _required_permutations(len(pairs), q, correction)
    centred_y = y - y.mean()
    norm_y = float(np.linalg.norm(centred_y))
    observed = {i: float(_correlations(values[i][None, :], centred_y, norm_y)[0]) for i, _ in pairs if usable[i]}
    exceed = np.zeros(len(pairs), dtype=np.int64)
    stopped_at = np.zeros(len(pairs), dtype=np.int64)  # draws used when the pair stopped; 0 while running
    active = np.array([usable[i] for i, _ in pairs])
    rng = np.random.default_rng(random_state)
    drawn = 0
    while drawn < B and active.any():
        block = min(256, B - drawn)
        order = np.argsort(rng.random((block, n)), axis=1)
        for k in np.flatnonzero(active):
            index, column = pairs[k]
            indices, function, complex_values = compiled[index]
            arguments = [X[order, j] if j == column else X[:, j] for j in indices]
            draws = _draw(function, arguments, complex_values, block, n)
            hits = np.cumsum(_correlations(draws, centred_y, norm_y) >= observed[index] - 1e-12)
            if exceed[k] + hits[-1] >= EXCEEDANCES:
                stopped_at[k] = drawn + int(np.searchsorted(hits, EXCEEDANCES - exceed[k])) + 1
                exceed[k] = EXCEEDANCES
                active[k] = False
            else:
                exceed[k] += hits[-1]
        drawn += block
    p_values = np.ones(len(pairs))
    for k, (index, _) in enumerate(pairs):
        if usable[index]:
            p_values[k] = EXCEEDANCES / stopped_at[k] if stopped_at[k] else (1 + exceed[k]) / (1 + B)
    return p_values
