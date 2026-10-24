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
from sklearn.utils.validation import check_is_fitted

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


class _EquationsMixin:
    """The equation list, and the notes the renderings of a fit share.

    Mixed into the estimators' base class, whose fitted attributes it
    reads: ``equations_``, ``selection_result_``, ``certification_result_``,
    ``fdr_controlled_``, ``fdr_scope_`` and ``n_features_out_``.
    """

    def _parsimony_note(self) -> str:
        """Suffix for ``equation()`` when the printed terms are a strict subset.

        Empty whenever the printed terms *are* the certified ones, which is
        what :attr:`fdr_scope_` already decides.
        """
        if getattr(self, "fdr_scope_", None) != "screened set":
            return ""
        screened, kept = int(self.selection_result_.n_selected), int(self.n_features_out_)
        # A subset can also arise without the parsimony step, when a screened
        # expression cannot be evaluated on the full training data; pointing
        # that caller at parsimony=None would be advice they have taken.
        dropped = int(getattr(self, "_inputs_dropped_", 0))
        remedy = (
            f"; parsimony=None prints all {screened - dropped}" if self.parsimony is not None else ""
        )
        if dropped:
            remedy += f"; input_test dropped {dropped}"
        return f"   [{kept} of {screened} certified terms{remedy}]"

    def _latex_parsimony_note(self) -> str:
        """:meth:`_parsimony_note` as LaTeX text, or empty when it is."""
        note = self._parsimony_note().strip()
        return rf" \quad {_latex_text(note)}" if note else ""

    def _path_lhs(self) -> str:
        """Left-hand side naming the modelled quantity in :meth:`equations`."""
        return "y"

    def equations(self, precision: int = 4, max_n_terms: int | None = None) -> str:
        """Return the forward path through the screened set, one equation per size.

        Line ``k`` holds the ``k`` terms the greedy forward step of
        ``parsimony="forward"`` would fit if it stopped there, with its
        complexity, the in-sample R^2 of a least-squares fit of those terms
        on the rows screening used, and the R^2 its last term added. The
        line holding the returned features is marked ``*``; at the default
        settings it is the first line whose successor adds less than
        ``parsimony_tol``. The records behind the text are
        :attr:`equations_`.

        Each line is written ``y ~ a + b``: a linear model in those terms,
        without coefficients, since only the marked line has fitted ones,
        which the regressor's and classifier's ``equation()`` prints. For classification the R^2 is that of
        the integer-coded labels, the ranking heuristic the forward step
        uses, and not a classification metric.

        The FDR guarantee covers the screened set, the last line, and not an
        equation chosen from this list: every line is a subset of the
        screened set chosen on the rows screening used, as the features
        returned at the default settings are. The output says so in its last
        line.

        Args:
            precision: Decimal places for R^2, and significant figures for
                its gain.
            max_n_terms: If set, show only the equations with at most this
                many terms, and the marked line whatever its size. The
                remaining lines are elided as ``...``.

        Returns:
            The list as text, or a one-line explanation when nothing was
            screened.
        """
        check_is_fitted(self, ["equations_"])
        _check_equation_format(precision, "significant")
        if self.equations_ is None:
            return "no equation list: no selector was configured, so no set was screened"
        if not self.equations_:
            return "no equation list: the screen rejected no null"

        screened = len(self.equations_)
        rows = [
            (
                "*" if record["model"] else "",
                str(record["n_terms"]),
                str(record["complexity"]),
                f"{record['r2']:.{precision}f}",
                f"{record['r2'] - previous:.{precision}g}",
                f"{self._path_lhs()} ~ " + " + ".join(record["formulas"]),
            )
            for record, previous in zip(
                self.equations_,
                [0.0] + [record["r2"] for record in self.equations_[:-1]],
            )
            if max_n_terms is None
            or record["n_terms"] <= max_n_terms
            or record["model"]
        ]
        header = ("", "k", "complexity", "R^2", "gain", "equation")
        widths = [max(len(row[column]) for row in [header, *rows]) for column in range(5)]
        measure = (
            "R^2 is of the integer-coded labels, a ranking heuristic rather than a "
            "classification metric"
            if self._problem_type == "classification"
            else "R^2 is in-sample"
        )

        def line(row: tuple[str, ...]) -> str:
            cells = [cell.rjust(width) for cell, width in zip(row[1:5], widths[1:])]
            return " ".join([row[0].ljust(widths[0]), *cells, row[5]]).rstrip()

        lines = [
            f"Forward path through the {screened} screened term{'' if screened == 1 else 's'}. "
            f"{measure}, on the rows screening used; gain is the R^2 the line's last "
            "term adds.",
            line(header),
        ]
        elided = " " * widths[0] + " " + "...".rjust(widths[1])
        previous = 0
        for row in rows:
            if int(row[1]) > previous + 1:
                lines.append(elided)
            previous = int(row[1])
            lines.append(line(row))
        if previous < screened:
            lines.append(elided)
        lines.append(self._path_model_note())
        lines.append(self._path_scope_note(screened))
        return "\n".join(lines)

    def _path_model_note(self) -> str:
        """Footer line of :meth:`equations` saying which line is the model."""
        returned = "fitted model" if hasattr(self, "equation") else "returned features"
        marked = [record["n_terms"] for record in self.equations_ if record["model"]]
        retested = self.certification_result_ is not None
        if not marked:
            reason = (
                "its terms were re-tested on held-back rows (parsimony_holdout)"
                if retested
                else "some screened terms could not be evaluated on the full training data"
            )
            return f"No line is the {returned}: {reason}."
        if self.parsimony is None:
            how = "parsimony=None keeps the whole screened set"
        elif retested and self.certification_result_.n_selected:
            how = "the forward subset, as re-tested on held-back rows"
        elif retested:
            how = "the re-test on held-back rows rejected no null, so the screened set is kept"
        elif marked[0] == self._forward_stop_ == len(self.equations_):
            how = "the forward step keeps every screened term"
        elif marked[0] == self._forward_stop_:
            how = (
                "the forward step stops before the first gain below "
                f"parsimony_tol={self.parsimony_tol:g}"
            )
            if self.parsimony_alpha is not None:
                how += (
                    ", or the first that fails to reject the null of chance at "
                    f"parsimony_alpha={self.parsimony_alpha:g}"
                )
        else:
            how = "parsimony_holdout could not split the selection rows, so the screened set is kept"
        return f"* the {returned} ({how})."

    def _path_scope_note(self, screened: int) -> str:
        """Footer line of :meth:`equations` stating what the guarantee covers."""
        if not self.fdr_controlled_:
            return (
                "No line carries the FDR guarantee: fdr_controlled_ is False, since "
                "selection did not run on rows held out of the search."
            )
        last = f"the screened set of {screened} term{'' if screened == 1 else 's'} (line {screened})"
        if self.certification_result_ is not None and self.certification_result_.n_selected:
            return (
                f"The FDR guarantee covers {last} and, after its re-test, the "
                f"{'fitted model' if hasattr(self, 'equation') else 'returned features'}; "
                "it does not cover another equation chosen from this list."
            )
        return f"The FDR guarantee covers {last}, not an equation chosen from this list."
