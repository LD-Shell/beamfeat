"""Scikit-learn compatible estimators.

Three classes are provided:

:class:`BeamFeatTransformer`
    Constructs features and returns them. Composes into any scikit-learn
    pipeline, which is the usual way to combine beamfeat with an arbitrary
    downstream model.

:class:`BeamFeatRegressor` / :class:`BeamFeatClassifier`
    Construct features and fit a linear model on them. The point is not the
    linear model's accuracy but that its coefficients are readable: the fitted
    object exposes a closed-form expression for what it learned, which a
    gradient-boosted model on raw features cannot.

All three fit in two stages. Features are constructed and scored by
:class:`~beamfeat.search.BeamSearch`, then filtered by a
:class:`~beamfeat.selection.Selector` so the retained set carries an FDR
guarantee rather than a heuristic cut. Both stages see only the training data
passed to :meth:`fit`; :meth:`transform` replays the selected expressions
without re-searching, so a pipeline cross-validated in the usual way does not
leak.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from beamfeat.expression import Evaluator, Node
from beamfeat.scoring import Scorer, is_constant
from beamfeat.search import DEFAULT_BINARY, DEFAULT_UNARY, BeamSearch, report
from beamfeat.selection import Selector, make_selector  # noqa: F401

__all__ = ["BeamFeatClassifier", "BeamFeatRegressor", "BeamFeatTransformer"]

logger = logging.getLogger(__name__)


_UNSET = object()


def _validate_input(estimator, X, y=_UNSET, **kwargs):
    """``validate_data`` with actionable guidance appended to common failures.

    beamfeat deliberately does not impute or encode internally — embedding
    those modelling choices silently would contradict the library's
    explicitness — so the errors point at the standard scikit-learn
    composition instead.
    """
    from sklearn.utils.validation import validate_data

    try:
        if y is _UNSET:
            return validate_data(estimator, X, **kwargs)
        return validate_data(estimator, X, y, **kwargs)
    except ValueError as exc:
        message = str(exc)
        lowered = message.lower()
        if "nan" in lowered or "infinity" in lowered or "infinite" in lowered:
            raise ValueError(
                message
                + "\n\nbeamfeat does not impute missing values internally. Handle them "
                "explicitly first, e.g.:\n"
                "    make_pipeline(SimpleImputer(strategy='median'), BeamFeatRegressor(...))"
            ) from None
        if "could not convert" in lowered or "string" in lowered:
            raise ValueError(
                message
                + "\n\nbeamfeat constructs arithmetic expressions and requires numeric "
                "columns. Encode categoricals explicitly first, e.g. with "
                "ColumnTransformer + OneHotEncoder."
            ) from None
        raise



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

    Undoes the scaler so the printed coefficients apply to the raw feature
    values. Terms are ordered by standardised coefficient magnitude — the
    scale-free measure of importance — since raw magnitude reflects the
    feature's units: a raw coefficient of 3e-07 on a large-valued feature
    can carry the whole model. For the same reason only exactly-zero terms
    are omitted, and no ``style`` displays a nonzero coefficient as zero
    (see ``_format_coefficient``). Shared by the regressor's ``equation``
    and the classifier's log-odds ``equation`` so the two cannot drift in
    formatting or in the de-standardisation arithmetic.
    """
    standardised = np.asarray(coefficients, dtype=np.float64)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    raw_coefficients = standardised / scale
    offset = float(intercept - np.sum(raw_coefficients * scaler.mean_))

    order = np.argsort(-np.abs(standardised))
    if max_terms is not None:
        order = order[:max_terms]

    parts = []
    for index in order:
        coefficient = raw_coefficients[index]
        if coefficient == 0.0:
            continue
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


class NoDiscoveriesError(RuntimeError):
    """Raised in ``on_no_discoveries="raise"`` mode when selection passes nothing."""


class DegenerateFitWarning(UserWarning):
    """Raised when the assembled model scores poorly on the selection holdout.

    FDR screening certifies that the kept features are genuinely associated
    with the target; it does not certify that the linear model built on them
    is well conditioned — a feature can be a true discovery and still carry
    extreme leverage that destabilises the fit. These are different claims,
    and this warning marks the gap when it opens: negative R^2 on the
    selection rows for regression, or accuracy below the majority-class rate
    for classification. The check is a diagnostic — the selection rows also
    enter the final fit — not an independent evaluation.
    """


class NoDiscoveriesWarning(UserWarning):
    """Raised when no feature passed FDR-controlled selection.

    Emitted through :mod:`warnings` (not only logging) so it is visible in
    notebooks and default configurations — the scikit-learn convention for
    fit-time conditions the user must act on.
    """


class _BeamFeatBase(BaseEstimator):
    """Shared fitting machinery for the beamfeat estimators.

    Not intended for direct use; instantiate one of the concrete classes.
    """

    _problem_type: str = "regression"

    def __init__(
        self,
        scorer: str | Scorer = "correlation",
        selector: str | Selector | None = "permutation",
        max_depth: int = 2,
        beam_width: int = 50,
        max_features: int | None = None,
        target_fdr: float = 0.1,
        unary_ops: tuple[str, ...] = DEFAULT_UNARY,
        binary_ops: tuple[str, ...] = DEFAULT_BINARY,
        redundancy_threshold: float = 0.95,
        include_originals: bool = True,
        units: dict[str, Any] | None = None,
        selection_holdout: float | None = 0.5,
        parsimony_holdout: float | None = None,
        selection_correction: str = "by",
        on_no_discoveries: str = "empty",
        parsimony: str | None = "forward",
        parsimony_tol: float = 1e-3,
        random_state: int | None = 0,
        verbose: int = 0,
    ) -> None:
        self.scorer = scorer
        self.selector = selector
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_features = max_features
        self.target_fdr = target_fdr
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.redundancy_threshold = redundancy_threshold
        self.include_originals = include_originals
        self.units = units
        self.selection_holdout = selection_holdout
        self.parsimony_holdout = parsimony_holdout
        self.selection_correction = selection_correction
        self.on_no_discoveries = on_no_discoveries
        self.parsimony = parsimony
        self.parsimony_tol = parsimony_tol
        self.random_state = random_state
        self.verbose = verbose

    # -- fitting ------------------------------------------------------------ #

    def _parsimonious_subset(
        self, matrix: np.ndarray, target: np.ndarray, screened: list[int]
    ) -> list[int]:
        """Greedy forward selection among the FDR-screened features.

        On by default (``parsimony="forward"``). Purpose and guarantee
        semantics, stated precisely: the q-level FDR guarantee applies to the
        *screening* step (the full screened set); this step then chooses a
        predictive subset of it for the fitted model, on the same rows
        selection used. The subset is not re-certified at level q, so the
        guarantee covers the set it came from and not the terms that survive
        — which is why :meth:`equation` says so in its own output whenever
        this step has dropped anything, rather than leaving the caller to
        find it in the documentation. Every kept feature did pass screening,
        and the full screened set with p- and q-values remains available in
        ``selection_report_``.

        The rationale: on a strong signal the marginal null correctly passes
        many mutually redundant true discoveries, and an equation of dozens of
        near-duplicate terms defeats the interpretability the library exists
        for. What compactness costs is measured rather than assumed — see
        ``benchmarks/PARSIMONY_COST.md``, which puts it at roughly a
        thousandth of held-out R^2 for an equation some twenty-five times
        shorter, with no observed inflation of the realised false discovery
        proportion.

        ``parsimony=None`` keeps the entire screened set, so the printed
        equation *is* the certified object; ``parsimony_holdout`` keeps it
        compact and certified both, at a cost in rows.

        Features are added while the incremental in-sample R^2 (on integer-
        coded labels for classification, as a ranking heuristic) improves by
        at least ``parsimony_tol``; at least one feature is always kept.
        """
        if self.parsimony is None or len(screened) <= 1:
            return list(screened)
        if self.parsimony != "forward":
            raise ValueError(f"parsimony must be 'forward' or None, got {self.parsimony!r}")

        numeric = np.asarray(target, dtype=np.float64)
        if self._problem_type == "classification":
            _, numeric = np.unique(target, return_inverse=True)
            numeric = numeric.astype(np.float64)
        total_variance = float(np.var(numeric))
        if total_variance < 1e-12:  # pragma: no cover - defensive
            return list(screened)

        chosen: list[int] = []
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
            if scores[best_position] - best_r2 < self.parsimony_tol and chosen:
                break
            best_r2 = scores[best_position]
            chosen.append(remaining.pop(best_position))
        return sorted(chosen)

    def _resolve_fdr_scope(self) -> None:
        """Record which set :attr:`fdr_controlled_` is a statement about.

        ``True`` on the flag means the returned features came from the
        FDR-screened set. That is not the same as the guarantee covering the
        returned features: when fewer terms are returned than were screened,
        the q-level statement is about the larger set they were drawn from.
        :meth:`equation` prints that distinction for a reader, but a caller
        branching on the flag in a script never sees the string, so it is
        recorded here as well.

        ``"printed equation"`` when the guarantee covers exactly the returned
        features -- ``parsimony=None``, a greedy pass that kept everything, or
        a ``parsimony_holdout`` re-test, which certifies the subset on rows of
        its own. ``"screened set"`` when it covers a larger set the returned
        features were drawn from. ``None`` when nothing is controlled at all.
        """
        if not self.fdr_controlled_:
            self.fdr_scope_ = None
            return
        if self.certification_result_ is not None:
            self.fdr_scope_ = "printed equation"
            return
        screened = (
            0 if self.selection_result_ is None
            else int(self.selection_result_.n_selected)
        )
        self.fdr_scope_ = (
            "screened set" if screened > int(self.n_features_out_)
            else "printed equation"
        )

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
        remedy = (
            f"; parsimony=None prints all {screened}" if self.parsimony is not None else ""
        )
        return f"   [{kept} of {screened} certified terms{remedy}]"

    def _certify_subset(self, X, y, rows, feature_names, units, nodes, keep_order):
        """Re-screen the terms kept after parsimony on rows not yet used for them.

        The set of terms is fixed once the screening rows have been used, so
        this is an ordinary fixed-candidate screen at the same level, and the
        guarantee it returns is over those terms themselves rather than over
        the larger set they were drawn from. With ``parsimony=None`` the terms
        are the whole screened set, and the re-test then trades length for a
        guarantee stated on rows of its own.
        """
        data = {name: X[rows, index] for index, name in enumerate(feature_names)}
        evaluator = Evaluator(data, units=units)
        columns, order = [], []
        for node, index in zip(nodes, keep_order):
            values = evaluator.transform_values(node)
            if values is not None:
                columns.append(values)
                order.append(index)
        if not columns:
            return order
        # A lone survivor still has to earn its place on rows it has not seen;
        # only the multiplicity correction is trivial in that case, not the
        # test itself.
        selector_kwargs = {}
        if isinstance(self.selector, str) and self.selector.lower() in ("permutation", "perm"):
            selector_kwargs["correction"] = self.selection_correction
        selector = make_selector(
            self.selector,
            target_fdr=self.target_fdr,
            problem_type=self._problem_type,
            random_state=self.random_state,
            **selector_kwargs,
        )
        result = selector.select(np.column_stack(columns), y[rows])
        self.certification_result_ = result
        for diagnostic in result.warnings_raised:
            warnings.warn(f"beamfeat certification: {diagnostic}", UserWarning, stacklevel=2)
        if result.n_selected == 0:
            warnings.warn(
                "beamfeat: no term of the parsimonious equation survived re-testing on the "
                f"held-back rows at target FDR {self.target_fdr:.3g}, so no compact "
                "certified equation is available on this sample. The whole screened set is "
                "returned instead: it does carry the guarantee, but it is not compact.",
                NoDiscoveriesWarning,
                stacklevel=2,
            )
        return [order[int(i)] for i in result.selected]

    def _holdout_fit_check(self, X: np.ndarray, y: np.ndarray) -> None:
        """Warn when the assembled model is degenerate on the selection rows.

        A diagnostic, not an independent evaluation: the selection rows also
        enter the final fit and the parsimony step, so this check exists to
        catch numerically pathological assemblies loudly, not to estimate
        generalisation. It changes nothing about the fitted model.
        """
        rows = getattr(self, "_holdout_rows_", None)
        if rows is None or len(rows) == 0 or self.n_features_out_ == 0:
            return
        if getattr(self, "model_", None) is None:
            return
        with warnings.catch_warnings():
            # X is this estimator's own validated array, so it carries no
            # column names by construction. Without this filter scikit-learn's
            # feature-name check fires on every fit that was given a
            # DataFrame, pointing the caller at a mismatch of our own making.
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            holdout_score = float(self.score(X[rows], y[rows]))
        if self._problem_type == "classification":
            _, counts = np.unique(y[rows], return_counts=True)
            floor = float(np.max(counts)) / len(rows)
            degenerate = holdout_score < floor - 1e-9
            description = f"accuracy {holdout_score:.3f} on the selection rows is below the majority-class rate {floor:.3f}"
        else:
            degenerate = holdout_score < 0.0
            description = f"R^2 {holdout_score:.3f} on the selection rows is negative"
        if degenerate:
            warnings.warn(
                "beamfeat: the selected features passed FDR screening (genuine "
                f"associations), but the assembled model is degenerate: {description}. "
                "This is a diagnostic rather than an independent evaluation — the "
                "selection rows also enter the final fit. "
                "Association does not guarantee a well-conditioned fit; inspect "
                "selection_report_ for extreme-valued features, or reduce max_depth "
                "or the feature count.",
                DegenerateFitWarning,
                stacklevel=2,
            )

    def _run_pipeline(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """Search, select, and store the resulting expressions.

        When a selector is configured and ``selection_holdout`` is set, the
        training rows are split: the search sees one part, and selection is
        performed on the other. This is not an optimisation but a validity
        requirement — the search retains candidates partly *because* they
        correlate with the target in its sample, so p-values computed on that
        same sample are optimistically biased and the nominal FDR would not be
        guaranteed. Testing on rows the search never saw restores the fixed-
        candidate-set premise the guarantees are stated under.

        Sets :attr:`fdr_controlled_`: ``True`` when every returned feature
        comes from the selector's FDR-screened set on held-out data. The
        set-level q guarantee of the screening applies to that full screened
        set, available with per-candidate p- and q-values in
        :attr:`selection_report_`. At the default ``parsimony="forward"`` the
        returned features are a data-dependent subset of that set, which is
        not re-certified at level q; under ``parsimony=None`` they *are* the
        set. ``False`` when selection returned nothing and the estimator fell
        back to the search output (in which case the features carry no FDR
        guarantee, and a warning says so); ``None`` when no selector was
        configured.

        Sets :attr:`fdr_scope_` alongside it, naming which set the flag is a
        statement about: ``"printed equation"`` when the guarantee covers the
        returned features themselves, ``"screened set"`` when it covers a
        larger set they were drawn from, ``None`` when the flag is not
        ``True``. The flag says whether there is a guarantee; the scope says
        what it is over.

        Returns the transformed training matrix on the full data.
        """
        # Checked here rather than in BeamSearch so the message follows
        # scikit-learn's convention, which its estimator checks match against.
        if X.shape[0] < 2:
            raise ValueError(
                f"Found array with {X.shape[0]} sample(s) while a minimum of 2 is required "
                f"by {type(self).__name__}."
            )

        units = self._resolve_units(feature_names)
        n_samples = X.shape[0]

        # A column with no variation cannot be selected: it standardises to
        # zeros, so its association with the target is zero at every depth it
        # appears in. That is the correct outcome and costs nothing, but it is
        # indistinguishable from an uninformative column in the output, and a
        # stuck sensor or a column emptied by a bad join should not look the
        # same as a variable that simply does not matter. The test consults no
        # response values, so reporting it here cannot bias what follows.
        flat = [name for index, name in enumerate(feature_names) if is_constant(X[:, index])]
        if flat:
            shown = ", ".join(str(name) for name in flat[:5])
            warnings.warn(
                f"beamfeat: {len(flat)} of {len(feature_names)} columns are constant "
                f"({shown}{', ...' if len(flat) > 5 else ''}) and cannot be selected. "
                "They are ignored and cost nothing, but check whether they are meant "
                "to carry data.",
                UserWarning,
                stacklevel=2,
            )

        use_holdout = (
            self.selector is not None
            and self.selection_holdout is not None
            and 0.0 < float(self.selection_holdout) < 1.0
        )
        if use_holdout:
            n_holdout = int(round(float(self.selection_holdout) * n_samples))
            # Both halves need enough rows to be meaningful; below that the
            # split would cost more validity than it buys, so fall back to
            # same-data selection with the caveat recorded on the instance.
            if n_holdout < 10 or n_samples - n_holdout < 10:
                use_holdout = False

        split_rng = np.random.default_rng(self.random_state)
        if use_holdout:
            order = split_rng.permutation(n_samples)
            holdout_index = np.sort(order[: int(round(float(self.selection_holdout) * n_samples))])
            search_index = np.sort(order[int(round(float(self.selection_holdout) * n_samples)) :])
        else:
            holdout_index = np.empty(0, dtype=int)
            search_index = np.arange(n_samples)
        self._holdout_rows_ = holdout_index if use_holdout else None

        # Optional second split of the selection rows. Screening, and parsimony
        # if it is enabled, run on the first part; the surviving terms are then
        # re-tested on the second, which they have not seen. Because that set is
        # fixed once the first part has been used, the re-test is an ordinary
        # fixed-candidate screen and its guarantee therefore covers the printed
        # equation, not merely the set it was drawn from. The price is rows:
        # both parts have to be large enough to test on.
        certify_index = np.empty(0, dtype=int)
        fell_back_to_screened = False
        use_certify = (
            use_holdout
            and self.parsimony_holdout is not None
            and 0.0 < float(self.parsimony_holdout) < 1.0
        )
        if use_certify:
            n_certify = int(round(float(self.parsimony_holdout) * len(holdout_index)))
            if n_certify < 10 or len(holdout_index) - n_certify < 10:
                use_certify = False
                # The caller asked for a printed equation that carries the
                # guarantee. If the rows cannot supply a compact one, the honest
                # degradation is the screened set itself -- long, but certified --
                # not a compact subset of it, which is the one property they
                # explicitly did not ask for.
                fell_back_to_screened = True
                warnings.warn(
                    "beamfeat: parsimony_holdout was requested but the selection rows "
                    f"({len(holdout_index)}) cannot be split so that both parts hold at "
                    "least 10 rows. Returning the whole screened set instead, which does "
                    "carry the guarantee but is not compact; pass "
                    "parsimony_holdout=None to get the compact uncertified equation, "
                    "or add rows.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                inner = split_rng.permutation(len(holdout_index))
                certify_index = np.sort(holdout_index[inner[:n_certify]])
                holdout_index = np.sort(holdout_index[inner[n_certify:]])
        self._certify_rows_ = certify_index if use_certify else None

        search_data = {name: X[search_index, index] for index, name in enumerate(feature_names)}

        report(
            self.verbose,
            1,
            f"fit: {n_samples} rows x {len(feature_names)} columns"
            + (
                f", split {len(search_index)} search / {len(holdout_index)} selection"
                if use_holdout
                else ", no holdout (selection shares the search rows)"
            ),
        )

        search = BeamSearch(
            scorer=self.scorer,
            problem_type=self._problem_type,
            max_depth=self.max_depth,
            beam_width=self.beam_width,
            max_features=self.max_features,
            unary_ops=tuple(self.unary_ops),
            binary_ops=tuple(self.binary_ops),
            redundancy_threshold=self.redundancy_threshold,
            include_originals=self.include_originals,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        search_result = search.run(search_data, y[search_index], units=units)
        self.search_result_ = search_result

        nodes = list(search_result.nodes)
        if not nodes:
            nodes = Evaluator(search_data, units=units).leaf_nodes()
            logger.warning("[beamfeat] search produced no features; falling back to input columns")

        self.selection_result_ = None
        self.selection_report_: list[dict] | None = None
        self.fdr_controlled_: bool | None = None
        self.fdr_scope_: str | None = None
        self.certification_result_ = None
        self.fdp_inflation_: float | None = None

        if self.selector is not None:
            if use_holdout:
                holdout_data = {name: X[holdout_index, index] for index, name in enumerate(feature_names)}
                evaluator = Evaluator(holdout_data, units=units)
            else:
                evaluator = Evaluator(
                    {name: X[:, index] for index, name in enumerate(feature_names)}, units=units
                )
            selection_target = y[holdout_index] if use_holdout else y

            # Candidates are a fixed set here; expressions that cannot be
            # evaluated on the selection rows (overflow, non-finite) drop out
            # of candidacy rather than entering as zero columns.
            candidate_nodes: list[Node] = []
            candidate_columns: list[np.ndarray] = []
            for node in nodes:
                values = evaluator.transform_values(node)
                if values is not None:
                    candidate_nodes.append(node)
                    candidate_columns.append(values)

            if len(candidate_nodes) > 1:
                matrix = np.column_stack(candidate_columns)
                selector_kwargs = {}
                if isinstance(self.selector, str) and self.selector.lower() in ("permutation", "perm"):
                    selector_kwargs["correction"] = self.selection_correction
                selector = make_selector(
                    self.selector,
                    target_fdr=self.target_fdr,
                    problem_type=self._problem_type,
                    random_state=self.random_state,
                    **selector_kwargs,
                )
                selection_result = selector.select(matrix, selection_target)
                self.selection_result_ = selection_result
                report(
                    self.verbose,
                    1,
                    f"screen: {selection_result.n_candidates} candidates, "
                    f"{selection_result.method}"
                    + (
                        f" ({self.selection_correction.upper()})"
                        if selection_result.method == "permutation"
                        else ""
                    )
                    + f" at q={self.target_fdr:g} -> "
                    f"{selection_result.n_selected} certified",
                )
                if self.verbose >= 2 and selection_result.n_selected:
                    ranked = sorted(
                        selection_result.selected,
                        key=lambda index: float(selection_result.p_values[index]),
                    )
                    for index in ranked[:5]:
                        report(
                            self.verbose,
                            2,
                            f"  {candidate_nodes[index].name}  "
                            f"p={float(selection_result.p_values[index]):.2e} "
                            f"q={float(selection_result.q_values[index]):.2e}",
                        )
                    if len(ranked) > 5:
                        report(self.verbose, 2, f"  ... and {len(ranked) - 5} more")
                for diagnostic in selection_result.warnings_raised:
                    warnings.warn(f"beamfeat selection: {diagnostic}", UserWarning, stacklevel=2)
                screened = set(int(i) for i in selection_result.selected)

                if selection_result.n_selected > 0:
                    keep_order = self._parsimonious_subset(
                        matrix, selection_target, list(selection_result.selected)
                    )
                    self.fdp_inflation_ = float(selection_result.n_selected) / max(
                        len(keep_order), 1
                    )
                    report(
                        self.verbose,
                        1,
                        f"parsimony: {selection_result.n_selected} certified -> "
                        f"{len(keep_order)} term{'' if len(keep_order) == 1 else 's'}"
                        f" (|S|/|S'| = {self.fdp_inflation_:.2f})"
                        if self.parsimony is not None
                        else f"parsimony: off, keeping all {selection_result.n_selected} "
                        f"certified term{'' if selection_result.n_selected == 1 else 's'}",
                    )
                    if fell_back_to_screened:
                        keep_order = sorted(screened)
                    elif use_certify:
                        certified = self._certify_subset(
                            X, y, certify_index, feature_names, units,
                            [candidate_nodes[index] for index in keep_order],
                            keep_order,
                        )
                        # Nothing compact could be certified, but the screened
                        # set still is; return that rather than nothing.
                        keep_order = certified if certified else sorted(screened)
                    nodes = [candidate_nodes[index] for index in keep_order]
                    kept = set(keep_order)
                    self.fdr_controlled_ = bool(nodes)
                    if use_certify:
                        scope = (
                            " re-certified on held-back rows; the guarantee covers the"
                            " fitted equation"
                        )
                    elif self.parsimony is None:
                        scope = (
                            " in the fitted equation, the screened set entire; the"
                            " guarantee covers it"
                        )
                    else:
                        scope = (
                            " in the fitted equation (the guarantee covers the"
                            " certified set, not this subset)"
                        )
                    report(
                        self.verbose,
                        1,
                        f"result set: {len(nodes)} term{'' if len(nodes) == 1 else 's'}" + scope,
                    )
                else:
                    kept = set()
                    self.fdr_controlled_ = False
                    message = (
                        f"beamfeat: no feature passed FDR-controlled selection at target "
                        f"FDR {self.target_fdr:.3g} on the held-out split "
                        f"(on_no_discoveries={self.on_no_discoveries!r}). "
                    )
                    if selection_result.warnings_raised:
                        message += (
                            "Selector notes: " + "; ".join(selection_result.warnings_raised) + ". "
                        )
                    if self.on_no_discoveries == "raise":
                        raise NoDiscoveriesError(
                            message + "Raise mode requested; refit with a higher target_fdr, "
                            "more data, or on_no_discoveries='empty'/'fallback'."
                        )
                    if self.on_no_discoveries == "fallback":
                        warnings.warn(
                            message + "Falling back to the UNFILTERED search output: the "
                            "returned features carry no false-discovery-rate guarantee.",
                            NoDiscoveriesWarning,
                            stacklevel=2,
                        )
                    else:  # "empty" (default)
                        nodes = []
                        warnings.warn(
                            message + "Returning no constructed features; the model will "
                            "predict the training mean (regression) or class prior "
                            "(classification). Set on_no_discoveries='fallback' to keep "
                            "the unfiltered search output instead.",
                            NoDiscoveriesWarning,
                            stacklevel=2,
                        )

                self.selection_report_ = [
                    {
                        "formula": candidate_nodes[i].name,
                        "p_value": (
                            float(selection_result.p_values[i])
                            if selection_result.p_values is not None else None
                        ),
                        "q_value": (
                            float(selection_result.q_values[i])
                            if selection_result.q_values is not None else None
                        ),
                        "statistic": float(selection_result.statistics[i]),
                        "screened": i in screened,
                        "kept": i in kept if selection_result.n_selected > 0 else False,
                    }
                    for i in range(len(candidate_nodes))
                ]
            else:
                self.fdr_controlled_ = False

            if not use_holdout and self.fdr_controlled_:
                # Same-data selection: the guarantee is stated for a fixed
                # candidate set, which this is not. Record honestly.
                self.fdr_controlled_ = False
                warnings.warn(
                    "beamfeat: selection ran on the same rows the search used to choose "
                    "candidates (selection_holdout disabled or too few rows); p-values are "
                    "optimistically biased and fdr_controlled_ is therefore False",
                    UserWarning,
                    stacklevel=2,
                )

        # Evaluate the final expressions on the full training data for the
        # downstream model.
        full_evaluator = Evaluator(
            {name: X[:, index] for index, name in enumerate(feature_names)}, units=units
        )
        kept_nodes: list[Node] = []
        kept_columns: list[np.ndarray] = []
        for node in nodes:
            values = full_evaluator.transform_values(node)
            if values is not None:
                kept_nodes.append(node)
                kept_columns.append(values)

        self.features_ = kept_nodes
        self.feature_formulas_ = [node.name for node in kept_nodes]
        self.n_features_out_ = len(kept_nodes)
        self._resolve_fdr_scope()
        if not kept_nodes:
            # The honest empty outcome: zero constructed columns. Downstream
            # models degrade to an intercept-only fit rather than pretending.
            return np.empty((n_samples, 0), dtype=np.float64)
        return np.column_stack(kept_columns)

    def _resolve_units(self, feature_names: list[str], warn: bool = True) -> dict[str, Any] | None:
        """Map user-supplied units onto the internal column names.

        Accepts either a mapping of column name to unit, or a positional
        sequence with one entry per input column.

        A sequence of the wrong length, an object that is neither, or a
        mapping whose keys match no column is an error rather than a silent
        no-op. A unit constraint that is quietly ignored is worse than one
        that raises: the caller goes on to report results as dimensionally
        validated when no dimensional check ever ran.

        A mapping that matches some columns but not others is the same
        failure in weaker form, and warns. Columns without a unit are
        dimensionally unconstrained, so they combine freely with the
        labelled ones and the check does not bind where it is most needed.
        """
        if self.units is None:
            return None

        units = self.units
        if not isinstance(units, Mapping):
            if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
                raise TypeError(
                    "units must be a mapping of column name to unit, or a sequence "
                    f"with one entry per column; got {type(units).__name__}"
                )
            if len(units) != len(feature_names):
                raise ValueError(
                    f"units has {len(units)} entries but X has {len(feature_names)} "
                    "columns; pass one unit per column, or use a mapping"
                )
            units = dict(zip(feature_names, units))

        if not units:
            return None

        resolved = {name: units[name] for name in feature_names if name in units}
        if not resolved:
            raise ValueError(
                f"none of the units keys {sorted(map(str, units))[:5]} match the "
                f"column names {feature_names[:5]}; the units would have been "
                "ignored. Pass a DataFrame, or key the mapping by column name."
            )
        missing = [name for name in feature_names if name not in units]
        if missing and warn:
            # An unlabelled column is dimensionally free: it combines with
            # anything, so partial coverage leaves the gate open on exactly
            # the columns the caller did not vouch for. That is the case
            # worth naming, since labelling the known columns and leaving the
            # rest blank looks like the careful thing to do.
            warnings.warn(
                f"beamfeat: units cover {len(resolved)} of {len(feature_names)} columns; "
                f"{len(missing)} are unlabelled ({', '.join(map(str, missing[:5]))}"
                f"{', ...' if len(missing) > 5 else ''}) and are treated as "
                "dimensionally unconstrained, so expressions combining them with "
                "labelled columns are not rejected. Give every column a unit "
                "('dimensionless' for the genuinely unitless ones) for the check "
                "to bind across the whole table.",
                UserWarning,
                stacklevel=2,
            )
        return resolved

    def _input_names(self, X, n_features: int) -> list[str]:
        """Derive column names, preferring those carried by a DataFrame."""
        names = getattr(self, "feature_names_in_", None)
        if names is not None:
            return [str(name) for name in names]
        return [f"x{index}" for index in range(n_features)]

    # -- transform ---------------------------------------------------------- #

    def _apply(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the stored expressions on new data.

        Expressions that fail numerically on unseen data — overflow, or a
        non-finite result — yield a column of zeros rather than aborting the
        transform, and the substitution is logged.

        Search-time admissibility filters are deliberately not applied here.
        Those criteria are data-dependent, so applying them would make the
        output depend on which rows are present: a subset in which some column
        happens to be constant would silently produce different values than the
        full data. See :meth:`~beamfeat.expression.Evaluator.transform_values`.

        This method does not mutate the estimator. scikit-learn requires
        ``transform`` and ``predict`` to leave ``__dict__`` untouched, so the
        count of failed expressions is logged rather than stored.
        """
        if not self.features_:
            return np.empty((X.shape[0], 0), dtype=np.float64)

        check_is_fitted(self, ["features_"])
        feature_names = self._input_names(X, X.shape[1])
        data = {name: X[:, index] for index, name in enumerate(feature_names)}
        # warn=False: coverage is a fit-time concern, and this path runs on
        # every transform and predict.
        evaluator = Evaluator(data, units=self._resolve_units(feature_names, warn=False))

        columns = []
        n_failed = 0
        for node in self.features_:
            values = evaluator.transform_values(node)
            if values is None:
                n_failed += 1
                values = np.zeros(X.shape[0])
            columns.append(values)

        if n_failed:
            logger.warning(
                "[beamfeat] %d of %d expressions could not be evaluated on this data "
                "and were replaced with zeros",
                n_failed,
                len(self.features_),
            )

        if not columns:  # pragma: no cover - defensive
            return np.empty((X.shape[0], 0))
        return np.column_stack(columns)

    # -- introspection ------------------------------------------------------ #

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the formula strings of the constructed features."""
        check_is_fitted(self, ["feature_formulas_"])
        return np.asarray(self.feature_formulas_, dtype=object)

    def formulas(self) -> list[str]:
        """Return the selected expressions as readable formula strings."""
        check_is_fitted(self, ["feature_formulas_"])
        return list(self.feature_formulas_)

    def to_sympy(self) -> list[Any]:
        """Return the selected expressions as sympy objects."""
        check_is_fitted(self, ["features_"])
        return [node.to_sympy() for node in self.features_]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.non_deterministic = False
        return tags


class BeamFeatTransformer(TransformerMixin, _BeamFeatBase):
    """Constructs and selects features, for use inside a pipeline.

    Args:
        scorer: Scoring strategy for the search. A name
            (``"correlation"``, ``"mutual_information"``,
            ``"gradient_boosting"``) or a
            :class:`~beamfeat.scoring.Scorer` instance.
        selector: FDR-controlled selector, or ``None`` to keep the search
            output unfiltered. A name (``"permutation"``, ``"knockoff"``) or a
            :class:`~beamfeat.selection.Selector` instance.
        problem_type: ``"regression"`` or ``"classification"``. Determines how
            the scorer and selector treat the target.
        max_depth: Maximum expression depth.
        beam_width: Expressions retained at each search depth.
        max_features: Cap on features returned by the search, before selection.
        target_fdr: Nominal false discovery rate for the selector.
        unary_ops: Unary operators to apply.
        binary_ops: Binary operators to apply.
        redundancy_threshold: Absolute correlation above which a candidate is
            treated as a duplicate.
        include_originals: Whether input columns compete alongside constructed
            expressions.
        units: Optional mapping of input column name to pint quantity.
            Supplying units restricts the search to dimensionally valid
            expressions.
        parsimony: ``"forward"`` (default) fits a compact subset of the
            screened set, chosen by greedy forward selection on the selection
            rows. That subset is not re-certified, so the FDR guarantee covers
            the set it was drawn from and not the terms returned, and
            :meth:`equation` marks this whenever terms were dropped. ``None``
            returns the screened set entire, so the returned features are
            exactly the set the guarantee covers; ``parsimony_holdout`` buys
            both, for rows.
        parsimony_tol: Minimum incremental R^2 a feature must add to be kept by
            ``parsimony="forward"``. Ignored when ``parsimony`` is ``None``.
        random_state: Seed. Fitting is deterministic given this.
        verbose: Progress reporting to stdout. ``0`` (default) is silent;
            ``1`` prints one line per stage -- the split, the search, the
            screening, the parsimony step and the fitted result; ``2`` adds
            per-depth search detail and the strongest certified candidates
            with their p- and q-values.

    Attributes:
        features_: Selected :class:`~beamfeat.expression.Node` expressions.
        feature_formulas_: Their formula strings.
        search_result_: The :class:`~beamfeat.search.SearchResult`.
        selection_result_: The :class:`~beamfeat.selection.SelectionResult`,
            or ``None`` if no selector was used.
        fdr_controlled_: Whether the returned features came from the
            FDR-screened set. ``None`` if no selector was used.
        fdr_scope_: Which set :attr:`fdr_controlled_` is a statement about --
            ``"printed equation"`` when the guarantee covers the returned
            features, ``"screened set"`` when it covers a larger set they were
            drawn from, ``None`` when the flag is not ``True``.
        n_features_out_: Number of features produced.
    """

    def __init__(
        self,
        scorer: str | Scorer = "correlation",
        selector: str | Selector | None = "permutation",
        problem_type: str = "regression",
        max_depth: int = 2,
        beam_width: int = 50,
        max_features: int | None = None,
        target_fdr: float = 0.1,
        unary_ops: tuple[str, ...] = DEFAULT_UNARY,
        binary_ops: tuple[str, ...] = DEFAULT_BINARY,
        redundancy_threshold: float = 0.95,
        include_originals: bool = True,
        units: dict[str, Any] | None = None,
        selection_holdout: float | None = 0.5,
        parsimony_holdout: float | None = None,
        selection_correction: str = "by",
        on_no_discoveries: str = "empty",
        parsimony: str | None = "forward",
        parsimony_tol: float = 1e-3,
        random_state: int | None = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            scorer=scorer,
            selector=selector,
            max_depth=max_depth,
            beam_width=beam_width,
            max_features=max_features,
            target_fdr=target_fdr,
            unary_ops=unary_ops,
            binary_ops=binary_ops,
            redundancy_threshold=redundancy_threshold,
            include_originals=include_originals,
            units=units,
            selection_holdout=selection_holdout,
            parsimony_holdout=parsimony_holdout,
            selection_correction=selection_correction,
            on_no_discoveries=on_no_discoveries,
            parsimony=parsimony,
            parsimony_tol=parsimony_tol,
            random_state=random_state,
            verbose=verbose,
        )
        self.problem_type = problem_type

    def fit(self, X, y=None):
        """Construct and select features from the training data.

        Args:
            X: ``(n_samples, n_features)`` training data.
            y: Target. Required, since both search and selection are supervised.

        Returns:
            ``self``.
        """
        if y is None:
            raise ValueError("BeamFeatTransformer requires y; feature construction is supervised")
        X, y = _validate_input(self, X, y, dtype=np.float64, y_numeric=self.problem_type == "regression")
        self._problem_type = self.problem_type
        self._run_pipeline(X, np.asarray(y), self._input_names(X, X.shape[1]))
        return self

    def transform(self, X) -> np.ndarray:
        """Evaluate the selected expressions on new data."""
        check_is_fitted(self, ["features_"])
        X = _validate_input(self, X, dtype=np.float64, reset=False)
        return self._apply(X)


class BeamFeatRegressor(RegressorMixin, _BeamFeatBase):
    """Constructs features, then fits a linear model on them.

    The linear model is the point: its coefficients combine with the feature
    formulas to give a closed-form expression for the fitted relationship,
    available via :meth:`equation`.

    Args:
        alpha: Ridge regularisation strength for the downstream model. The
            default ``"auto"`` selects the strength by efficient leave-one-out
            cross-validation over a logarithmic grid, which stays stable when
            many correlated features are selected from few rows; pass a float
            to fix the strength instead.

    Other arguments match :class:`BeamFeatTransformer`.

    Attributes:
        features_: Selected expressions.
        feature_formulas_: Their formula strings.
        model_: The fitted downstream linear model.
        coef_: Coefficients of the downstream model.
        intercept_: Intercept of the downstream model.
        fdr_scope_: Which set :attr:`fdr_controlled_` is a statement about;
            see :class:`BeamFeatTransformer`. ``"screened set"`` is what
            :meth:`equation` marks in its output.
    """

    _problem_type = "regression"

    def __init__(
        self,
        scorer: str | Scorer = "correlation",
        selector: str | Selector | None = "permutation",
        max_depth: int = 2,
        beam_width: int = 50,
        max_features: int | None = None,
        target_fdr: float = 0.1,
        alpha: float | str = "auto",
        unary_ops: tuple[str, ...] = DEFAULT_UNARY,
        binary_ops: tuple[str, ...] = DEFAULT_BINARY,
        redundancy_threshold: float = 0.95,
        include_originals: bool = True,
        units: dict[str, Any] | None = None,
        selection_holdout: float | None = 0.5,
        parsimony_holdout: float | None = None,
        selection_correction: str = "by",
        on_no_discoveries: str = "empty",
        parsimony: str | None = "forward",
        parsimony_tol: float = 1e-3,
        random_state: int | None = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            scorer=scorer,
            selector=selector,
            max_depth=max_depth,
            beam_width=beam_width,
            max_features=max_features,
            target_fdr=target_fdr,
            unary_ops=unary_ops,
            binary_ops=binary_ops,
            redundancy_threshold=redundancy_threshold,
            include_originals=include_originals,
            units=units,
            selection_holdout=selection_holdout,
            parsimony_holdout=parsimony_holdout,
            selection_correction=selection_correction,
            on_no_discoveries=on_no_discoveries,
            parsimony=parsimony,
            parsimony_tol=parsimony_tol,
            random_state=random_state,
            verbose=verbose,
        )
        self.alpha = alpha

    def fit(self, X, y):
        """Construct features and fit the downstream model."""
        from sklearn.linear_model import Ridge, RidgeCV
        from sklearn.preprocessing import StandardScaler
        X, y = _validate_input(self, X, y, dtype=np.float64, y_numeric=True)
        matrix = self._run_pipeline(X, np.asarray(y, dtype=np.float64), self._input_names(X, X.shape[1]))

        if matrix.shape[1] == 0:
            # Empty outcome of on_no_discoveries="empty": the honest model is
            # the intercept — predict the training mean, claim nothing more.
            self.scaler_ = None
            self.model_ = None
            self.coef_ = np.empty(0)
            self.intercept_ = float(np.mean(y))
            return self

        self.scaler_ = StandardScaler().fit(matrix)
        if isinstance(self.alpha, str):
            if self.alpha != "auto":
                raise ValueError(f"alpha must be a float or 'auto', got {self.alpha!r}")
            # Leave-one-out cross-validation over a wide grid: deterministic,
            # and safe when many correlated features are selected at small n,
            # where any fixed strength is either too weak or too strong.
            model = RidgeCV(alphas=np.logspace(-4, 4, 41))
        else:
            model = Ridge(alpha=self.alpha)
        self.model_ = model.fit(self.scaler_.transform(matrix), y)
        self.alpha_ = float(getattr(self.model_, "alpha_", self.alpha))
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        self._holdout_fit_check(X, y)
        report(
            self.verbose,
            1,
            f"result: {self.equation()}"
            + (
                ""
                if self.fdr_controlled_ is None
                else f"  [FDR controlled over the {self.fdr_scope_}]"
                if self.fdr_controlled_
                else "  [NOT FDR controlled]"
            ),
        )
        return self

    def predict(self, X) -> np.ndarray:
        """Predict using the constructed features and fitted model."""
        check_is_fitted(self, ["model_"])
        X = _validate_input(self, X, dtype=np.float64, reset=False)
        if self.model_ is None:
            return np.full(X.shape[0], self.intercept_, dtype=np.float64)
        return self.model_.predict(self.scaler_.transform(self._apply(X)))

    def equation(
        self,
        precision: int = 4,
        max_terms: int | None = None,
        style: str = "significant",
    ) -> str:  # noqa: D401
        """Return the fitted model as a readable closed-form equation.

        Args:
            precision: Significant figures for the coefficients, or decimal
                places when ``style="fixed"``.
            max_terms: If set, include only the terms with the largest
                standardised coefficients.
            style: ``"significant"`` adapts to each coefficient's magnitude,
                ``"fixed"`` aligns decimal places, ``"scientific"`` uses
                uniform e-notation. ``"fixed"`` falls back to significant
                figures for any value that would otherwise display as zero.

        Returns:
            A string such as ``y = 2.104*(a * b) - 0.512*log(c) + 3.991``.
            When the parsimony step dropped terms from the screened set, a
            suffix records how many of the certified terms are printed and
            how to print them all, since the guarantee covers the screened
            set rather than this subset of it.
        """
        check_is_fitted(self, ["model_"])
        _check_equation_format(precision, style)
        if self.model_ is None:
            intercept = _format_coefficient(self.intercept_, precision, style)
            return f"y = {intercept}  (no feature passed selection)"
        return _render_linear_equation(
            "y", self.coef_, float(self.intercept_), self.scaler_,
            self.feature_formulas_, precision, max_terms, style,
        ) + self._parsimony_note()


class BeamFeatClassifier(ClassifierMixin, _BeamFeatBase):
    """Constructs features, then fits a logistic model on them.

    Args:
        C: Inverse regularisation strength for the downstream model.

    Other arguments match :class:`BeamFeatTransformer`.

    Attributes:
        features_: Selected expressions.
        classes_: Class labels seen during fit.
        model_: The fitted downstream logistic model.
        fdr_scope_: Which set :attr:`fdr_controlled_` is a statement about;
            see :class:`BeamFeatTransformer`. ``"screened set"`` is what
            :meth:`equation` marks in its output.
    """

    _problem_type = "classification"

    def __init__(
        self,
        scorer: str | Scorer = "correlation",
        selector: str | Selector | None = "permutation",
        max_depth: int = 2,
        beam_width: int = 50,
        max_features: int | None = None,
        target_fdr: float = 0.1,
        C: float = 1.0,
        unary_ops: tuple[str, ...] = DEFAULT_UNARY,
        binary_ops: tuple[str, ...] = DEFAULT_BINARY,
        redundancy_threshold: float = 0.95,
        include_originals: bool = True,
        units: dict[str, Any] | None = None,
        selection_holdout: float | None = 0.5,
        parsimony_holdout: float | None = None,
        selection_correction: str = "by",
        on_no_discoveries: str = "empty",
        parsimony: str | None = "forward",
        parsimony_tol: float = 1e-3,
        random_state: int | None = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            scorer=scorer,
            selector=selector,
            max_depth=max_depth,
            beam_width=beam_width,
            max_features=max_features,
            target_fdr=target_fdr,
            unary_ops=unary_ops,
            binary_ops=binary_ops,
            redundancy_threshold=redundancy_threshold,
            include_originals=include_originals,
            units=units,
            selection_holdout=selection_holdout,
            parsimony_holdout=parsimony_holdout,
            selection_correction=selection_correction,
            on_no_discoveries=on_no_discoveries,
            parsimony=parsimony,
            parsimony_tol=parsimony_tol,
            random_state=random_state,
            verbose=verbose,
        )
        self.C = C

    def fit(self, X, y):
        """Construct features and fit the downstream classifier."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X, y = _validate_input(self, X, y, dtype=np.float64, y_numeric=False)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        # Scorers and selectors operate numerically, so labels are encoded to
        # integer codes here. Encoding at the boundary keeps the rest of the
        # library free of label-type handling, and predictions are mapped back
        # through classes_ by the downstream model, which is fitted on the
        # original labels.
        encoded = np.searchsorted(self.classes_, y)

        matrix = self._run_pipeline(X, encoded, self._input_names(X, X.shape[1]))

        if matrix.shape[1] == 0:
            # Empty outcome of on_no_discoveries="empty": no features were
            # kept, so the honest model is the class prior.
            self.scaler_ = None
            self.model_ = None
            counts = np.array([np.sum(y == klass) for klass in self.classes_], dtype=np.float64)
            self.class_prior_ = counts / counts.sum()
            self.coef_ = np.empty((0, 0))
            self.intercept_ = np.empty(0)
            return self

        self.scaler_ = StandardScaler().fit(matrix)

        if len(self.classes_) < 2:
            # A single-class training set has nothing to discriminate; store
            # the constant so predict still returns something coherent.
            self.model_ = None
            return self

        self.model_ = LogisticRegression(C=self.C, max_iter=2000, random_state=self.random_state).fit(
            self.scaler_.transform(matrix), y
        )
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        report(
            self.verbose,
            1,
            f"result: {self.n_features_out_} constructed feature"
            f"{'' if self.n_features_out_ == 1 else 's'}, "
            f"{len(self.classes_)} classes"
            + (
                ""
                if self.fdr_controlled_ is None
                else f"  [FDR controlled over the {self.fdr_scope_}]"
                if self.fdr_controlled_
                else "  [NOT FDR controlled]"
            ),
        )
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        check_is_fitted(self, ["features_"])
        X = _validate_input(self, X, dtype=np.float64, reset=False)
        if self.model_ is None:
            if hasattr(self, "class_prior_"):
                return np.full(X.shape[0], self.classes_[int(np.argmax(self.class_prior_))])
            return np.full(X.shape[0], self.classes_[0])
        return self.model_.predict(self.scaler_.transform(self._apply(X)))

    def equation(
        self,
        precision: int = 4,
        max_terms: int | None = None,
        style: str = "significant",
    ) -> str:
        """Return the fitted classifier as a readable log-odds equation.

        For binary problems the single line gives the log-odds of the
        positive class (``classes_[1]``): ``logit P(y = c1) = ...``, so the
        decision boundary is the zero level set of the right-hand side. For
        multiclass problems one line is returned per class; these are the
        softmax scores of the fitted multinomial model, and the predicted
        class is the argmax across lines.

        Args:
            precision: Significant figures for the coefficients, or decimal
                places when ``style="fixed"``.
            max_terms: If set, include only the terms with the largest
                standardised coefficients per line.
            style: ``"significant"`` adapts to each coefficient's magnitude,
                ``"fixed"`` aligns decimal places, ``"scientific"`` uses
                uniform e-notation. ``"fixed"`` falls back to significant
                figures for any value that would otherwise display as zero.

        When the parsimony step dropped terms from the screened set, a suffix
        records how many of the certified terms are printed and how to print
        them all. The terms are shared across the lines of a multiclass
        equation, so the suffix appears once at the end of the block.
        """
        check_is_fitted(self, ["model_"])
        _check_equation_format(precision, style)
        if self.model_ is None:
            if hasattr(self, "class_prior_"):
                prior = ", ".join(
                    f"P({klass}) = {_format_coefficient(p, precision, style)}"
                    for klass, p in zip(self.classes_, self.class_prior_, strict=True)
                )
                return f"class priors only (no feature passed selection): {prior}"
            return f"constant prediction: {self.classes_[0]!r}"

        coefficients = np.atleast_2d(self.coef_)
        intercepts = np.atleast_1d(self.intercept_)
        note = self._parsimony_note()
        if coefficients.shape[0] == 1:
            return _render_linear_equation(
                f"logit P(y = {self.classes_[1]!r})",
                coefficients[0], float(intercepts[0]), self.scaler_,
                self.feature_formulas_, precision, max_terms, style,
            ) + note
        lines = [
            _render_linear_equation(
                f"score({klass!r})", coefficients[row], float(intercepts[row]),
                self.scaler_, self.feature_formulas_, precision, max_terms, style,
            )
            for row, klass in enumerate(self.classes_)
        ]
        # One note for the whole block: the terms are shared across the lines.
        return ("\n".join(lines)
                + "\n(predicted class = argmax of the softmax scores)" + note)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self, ["features_"])
        X = _validate_input(self, X, dtype=np.float64, reset=False)
        if self.model_ is None:
            if hasattr(self, "class_prior_"):
                return np.tile(self.class_prior_, (X.shape[0], 1))
            return np.ones((X.shape[0], 1))
        return self.model_.predict_proba(self.scaler_.transform(self._apply(X)))

    def decision_function(self, X) -> np.ndarray:
        """Return the model's decision scores.

        With no feature selected the scores are the log prior odds (binary)
        or the log priors (multiclass), so they agree with :meth:`predict`
        and :meth:`predict_proba`.
        """
        check_is_fitted(self, ["model_"])
        X = _validate_input(self, X, dtype=np.float64, reset=False)
        if self.model_ is None:
            if not hasattr(self, "class_prior_"):
                return np.zeros(X.shape[0])
            log_prior = np.log(self.class_prior_)
            if len(self.classes_) == 2:
                return np.full(X.shape[0], log_prior[1] - log_prior[0])
            return np.tile(log_prior, (X.shape[0], 1))
        return self.model_.decision_function(self.scaler_.transform(self._apply(X)))
