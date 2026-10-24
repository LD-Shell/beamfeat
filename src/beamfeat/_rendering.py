"""Text and LaTeX renderings of a fitted model, and the records behind them.

The regressor's and classifier's ``equation`` and ``latex`` methods, and the
equation list of :meth:`~beamfeat.estimators.BeamFeatRegressor.equations`,
format their output here, so the renderings cannot drift apart in their
arithmetic or their formatting.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from beamfeat.expression import Node


def _label(value: Any) -> str:
    """Display a class label as the plain value, not a numpy scalar repr."""
    if isinstance(value, np.generic):
        value = value.item()
    return repr(value)


_EQUATION_STYLES = ("significant", "fixed", "scientific")


def _check_equation_format(precision: int, style: str) -> None:
    """Validate the shared formatting options of the ``equation`` methods."""
    if precision < 1:
        raise ValueError(f"precision must be a positive integer, got {precision!r}")
    if style not in _EQUATION_STYLES:
        raise ValueError(f"style must be one of {_EQUATION_STYLES}, got {style!r}")


def _format_coefficient(value: float, precision: int, style: str) -> str:
    """Format one number for equation output without ever displaying it as zero.

    ``significant`` and ``scientific`` are the plain ``g`` and ``e`` formats.
    ``fixed`` gives aligned decimal places, except that a nonzero value
    which would round to ``0.0000`` falls back to significant figures: a
    printed zero on a live coefficient misstates the model, which is worse
    than a change of notation mid-equation.
    """
    if style == "significant":
        return f"{value:.{precision}g}"
    if style == "scientific":
        return f"{value:.{precision}e}"
    text = f"{value:.{precision}f}"
    if value != 0.0 and float(text) == 0.0:
        return f"{value:.{precision}g}"
    return text


def _linear_terms(
    coefficients: np.ndarray, intercept: float, scaler, max_terms: int | None
) -> tuple[list[tuple[int, float]], float]:
    """The terms of a fitted standardised linear model, on raw features.

    Undoes the scaler so the coefficients apply to the raw feature values.
    Terms are ordered by standardised coefficient magnitude -- the
    scale-free measure of importance -- since raw magnitude reflects the
    feature's units: a raw coefficient of 3e-07 on a large-valued feature
    can carry the whole model. For the same reason only exactly-zero terms
    are omitted. Shared by the text and LaTeX renderings of the regressor
    and the classifier so they cannot drift in the de-standardisation
    arithmetic.

    Returns ``(index, raw coefficient)`` pairs in display order and the
    constant.
    """
    standardised = np.asarray(coefficients, dtype=np.float64)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    raw_coefficients = standardised / scale

    order = np.argsort(-np.abs(standardised))
    if max_terms is not None:
        order = order[:max_terms]
    # Terms left out by max_terms are held at their training mean, so the
    # constant carries their average contribution rather than dropping it:
    # a term such as 5*log(x) with mean 8 would otherwise shift every value
    # the shortened equation gives by 40.
    offset = float(intercept - np.sum(raw_coefficients[order] * scaler.mean_[order]))
    terms = [
        (int(index), float(raw_coefficients[index]))
        for index in order
        if raw_coefficients[index] != 0.0
    ]
    return terms, offset


def _render_linear_equation(
    lhs: str,
    coefficients: np.ndarray,
    intercept: float,
    scaler,
    formulas: list[str],
    precision: int,
    max_terms: int | None,
    style: str,
) -> str:
    """Render a fitted standardised linear model as an equation on raw features.

    The terms and constant are those of :func:`_linear_terms`, and no
    ``style`` displays a nonzero coefficient as zero (see
    ``_format_coefficient``). Shared by the regressor's ``equation`` and the
    classifier's log-odds ``equation`` so the two cannot drift in
    formatting.
    """
    terms, offset = _linear_terms(coefficients, intercept, scaler, max_terms)
    parts = []
    for index, coefficient in terms:
        sign = "-" if coefficient < 0 else "+"
        parts.append(
            f" {sign} {_format_coefficient(abs(coefficient), precision, style)}"
            f"*{formulas[index]}"
        )

    body = "".join(parts).lstrip()
    body = body.removeprefix("+ ")
    offset_sign = "-" if offset < 0 else "+"
    constant = f" {offset_sign} {_format_coefficient(abs(offset), precision, style)}"
    return (
        f"{lhs} = {body}{constant}"
        if body
        else f"{lhs} = {_format_coefficient(offset, precision, style)}"
    )


_LATEX_TEXT_ESCAPES = str.maketrans({
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
    "#": r"\#", "%": r"\%", "_": r"\_", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
})


def _latex_text(text: str) -> str:
    """Set plain text in LaTeX math, escaping the characters LaTeX reserves."""
    return r"\text{" + text.translate(_LATEX_TEXT_ESCAPES) + "}"


def _latex_number(text: str) -> str:
    """Rewrite a number formatted by ``_format_coefficient`` as LaTeX.

    E-notation becomes a power of ten, so ``3.2e-07`` reads
    ``3.2 \\times 10^{-7}``; other numbers are already valid LaTeX.
    """
    mantissa, marker, exponent = text.partition("e")
    if not marker or not exponent.lstrip("+-").isdigit():
        return text
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def _latex_label(value: Any) -> str:
    """A class label in LaTeX: numbers as they are, anything else as text."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        return repr(value)
    return _latex_text(str(value))


def _render_linear_latex(
    coefficients: np.ndarray,
    intercept: float,
    scaler,
    expressions: list[Any],
    precision: int,
    max_terms: int | None,
    style: str,
) -> str:
    """The right-hand side of :func:`_render_linear_equation`, in LaTeX.

    The terms, their order and the formatted coefficients are the same as
    in the text equation; each feature is rendered by ``sympy.latex`` from
    its sympy form in ``expressions``, and parenthesised when it is a sum or
    starts with a sign, so the coefficient multiplies all of it.
    """
    import sympy

    terms, offset = _linear_terms(coefficients, intercept, scaler, max_terms)
    parts = []
    for index, coefficient in terms:
        sign = "-" if coefficient < 0 else "+"
        feature = sympy.latex(expressions[index])
        if isinstance(expressions[index], sympy.Add) or feature.startswith("-"):
            feature = rf"\left({feature}\right)"
        # A feature that starts with a digit would run into the coefficient.
        joiner = r" \cdot " if feature[:1].isdigit() else " "
        number = _latex_number(_format_coefficient(abs(coefficient), precision, style))
        parts.append(f" {sign} {number}{joiner}{feature}")

    body = "".join(parts).lstrip()
    body = body.removeprefix("+ ")
    if not body:
        return _latex_number(_format_coefficient(offset, precision, style))
    offset_sign = "-" if offset < 0 else "+"
    return f"{body} {offset_sign} {_latex_number(_format_coefficient(abs(offset), precision, style))}"


def _equation_list(
    nodes: list[Node], path: tuple[list[int], list[float], int], fitted: list[Node]
) -> list[dict]:
    """Build the records of :attr:`equations_` from a forward path.

    ``nodes`` are the screening candidates the path indexes, and ``fitted``
    the features the model was fitted on; the record whose terms are exactly
    those is marked as the model, and no record is when none matches.
    """
    order, scores, _ = path
    fitted_keys = {node.key for node in fitted}
    terms = [nodes[index] for index in order]
    names = [node.name for node in terms]
    records: list[dict] = []
    complexity = 0
    for size, (node, score) in enumerate(zip(terms, scores), start=1):
        complexity += _tree_size(node) + 1
        records.append({
            "n_terms": size,
            "complexity": complexity,
            "r2": float(score),
            "formulas": names[:size],
            "features": tuple(terms[:size]),
            "model": size == len(fitted_keys)
            and {term.key for term in terms[:size]} == fitted_keys,
        })
    return records


def _tree_size(node: Node) -> int:
    """Number of nodes in the expression tree of ``node``.

    Operators, input columns and fitted constants each count one, and a
    subexpression counts each time it appears, as it does in the printed
    formula. Unlike :attr:`~beamfeat.expression.Node.size`, which counts a
    shared node once.
    """
    own = 1 if node.const is None else 2
    return own + sum(_tree_size(child) for child in node.children)
