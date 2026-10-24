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
import numbers
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from beamfeat._compat import logistic_regression
from beamfeat._forward import _chance_p_value, _extend_forward_path
from beamfeat._prediction import _evaluate_by_row
from beamfeat._rendering import (
    _check_equation_format,
    _equation_list,
    _format_coefficient,
    _label,
    _latex_label,
    _latex_number,
    _latex_text,
    _render_linear_equation,
    _render_linear_latex,
)
from beamfeat.expression import Evaluator, Node, _sympy_symbols
from beamfeat.scoring import Scorer, is_constant
from beamfeat.search import DEFAULT_BINARY, DEFAULT_FITTED, DEFAULT_UNARY, BeamSearch, report
from beamfeat.selection import (  # noqa: F401
    Selector,
    _ConditionalPermutationSelector,
    _InputPermutationTest,
    make_selector,
)

__all__ = [
    "BeamFeatClassifier",
    "BeamFeatRegressor",
    "BeamFeatTransformer",
    "DegenerateFitWarning",
    "NoDiscoveriesError",
    "NoDiscoveriesWarning",
]

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


class NoDiscoveriesError(RuntimeError):
    """Raised in ``on_no_discoveries="raise"`` mode when the screen rejects no null."""


class DegenerateFitWarning(UserWarning):
    """Raised when the assembled model scores poorly on the selection holdout.

    FDR screening bounds the expected share of selected features that are not
    associated with the target; it says nothing about whether the linear
    model built on them is well conditioned — a feature can be a true
    discovery and still carry extreme leverage that destabilises the fit.
    These are different claims, and this warning marks the gap when it opens:
    negative R^2 on the selection rows for regression, or accuracy below the
    majority-class rate for classification. The check is a diagnostic — the
    selection rows also enter the final fit — not an independent evaluation.
    """


class NoDiscoveriesWarning(UserWarning):
    """Raised when FDR-controlled testing leaves no feature to return.

    The screen rejected no null, or a later test (the ``parsimony_holdout``
    re-test, or ``input_test``) would drop every term the screen selected;
    the message says which, and what the estimator returns instead.

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
        n_jobs: int | None = 1,
        max_inputs: int | None = None,
        fitted_ops: tuple[str, ...] = DEFAULT_FITTED,
        certify_test: str = "marginal",
        magnitude_slots: int = 0,
        input_test: bool = False,
        parsimony_alpha: float | None = None,
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
        self.n_jobs = n_jobs
        self.max_inputs = max_inputs
        self.fitted_ops = fitted_ops
        self.certify_test = certify_test
        self.magnitude_slots = magnitude_slots
        self.input_test = input_test
        self.parsimony_alpha = parsimony_alpha

    # -- fitting ------------------------------------------------------------ #

    def _parsimonious_subset(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        screened: list[int],
        path: tuple[list[int], list[float], int] | None = None,
    ) -> list[int]:
        """Greedy forward selection among the FDR-screened features.

        On by default (``parsimony="forward"``). Purpose and guarantee
        semantics, stated precisely: the q-level FDR guarantee applies to the
        *screening* step (the full screened set); this step then chooses a
        predictive subset of it for the fitted model, on the same rows
        selection used. The subset is not re-tested at level q, so the
        guarantee covers the set it came from and not the terms it retains
        — which is why :meth:`equation` says so in its own output whenever
        this step has dropped anything, rather than leaving the caller to
        find it in the documentation. The screen rejected the null of every
        kept feature, and the full screened set with p- and q-values remains
        available in ``selection_report_``.

        The rationale: on a strong signal the screen correctly rejects the
        nulls of many mutually redundant true discoveries, and an equation of
        dozens of near-duplicate terms defeats the interpretability the library
        exists for. What compactness costs is measured rather than assumed — see
        ``benchmarks/PARSIMONY_COST.md``, which puts it at roughly a
        thousandth of R^2, held out in every study but one scored in-sample,
        for an equation some twenty-five times shorter, with no observed inflation of the realised false discovery
        proportion.

        ``parsimony=None`` keeps the entire screened set, so the printed
        equation *is* the certified object; ``parsimony_holdout`` keeps it
        compact and certified both, at a cost in rows.

        Features are added while the incremental in-sample R^2 (on integer-
        coded labels for classification, as a ranking heuristic) improves by
        at least ``parsimony_tol`` and, with ``parsimony_alpha``, while the
        gain rejects the null that it is chance (:func:`_chance_p_value`); at
        least one feature is always kept.

        ``path`` is the result of :meth:`_forward_path` on the same
        arguments, when the caller has already computed it.
        """
        if self.parsimony is None or len(screened) <= 1:
            return list(screened)
        if path is None:
            path = self._forward_path(matrix, target, screened)
        order, _, stop = path
        return sorted(order[:stop])

    def _forward_path(
        self, matrix: np.ndarray, target: np.ndarray, screened: list[int]
    ) -> tuple[list[int], list[float], int]:
        """The greedy forward path through the whole screened set.

        Returns the screened indices in the order the forward step adds
        them, the in-sample R^2 after each addition, and the number of them
        :meth:`_parsimonious_subset` keeps at ``parsimony_tol`` (and
        ``parsimony_alpha``). The path
        runs past that point to the full screened set, so the fit at every
        size is on record; the steps up to it are the forward step itself,
        computed exactly as it computes them, so the subset it keeps does not
        depend on how far the path is followed.
        """
        # Labels are encoded before any numeric cast: the transformer passes
        # its target through unencoded, and string labels cannot be cast.
        if self._problem_type == "classification":
            _, numeric = np.unique(target, return_inverse=True)
            numeric = numeric.astype(np.float64)
        else:
            numeric = np.asarray(target, dtype=np.float64)
        total_variance = float(np.var(numeric))
        if total_variance < 1e-12:  # pragma: no cover - defensive
            return list(screened), [float("nan")] * len(screened), len(screened)

        # The second stream of the seed: the first is the knockoff selector's.
        rng = (
            None if self.parsimony_alpha is None
            else np.random.default_rng(np.random.SeedSequence(self.random_state).spawn(2)[1])
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
                scores[best_position] - best_r2 < self.parsimony_tol
                or (
                    rng is not None
                    and _chance_p_value(matrix, numeric, chosen, remaining, rng) > self.parsimony_alpha
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

    def _validate_params(self) -> None:
        """Reject option values that would otherwise be ignored or misread.

        An out-of-range holdout fraction used to switch the split off rather
        than fail, which removes the guarantee the split exists for; an
        unrecognised ``on_no_discoveries`` fell through to the default.
        """
        if self.on_no_discoveries not in ("empty", "fallback", "raise"):
            raise ValueError(
                "on_no_discoveries must be 'empty', 'fallback' or 'raise', "
                f"got {self.on_no_discoveries!r}"
            )
        for name in ("selection_holdout", "parsimony_holdout"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not 0.0 < float(value) < 1.0
            ):
                raise ValueError(
                    f"{name} must be None or a fraction strictly between 0 and 1, got {value!r}"
                )
        if self.selection_correction not in ("bh", "by"):
            raise ValueError(
                f"selection_correction must be 'bh' or 'by', got {self.selection_correction!r}"
            )
        if self.parsimony not in ("forward", None):
            raise ValueError(f"parsimony must be 'forward' or None, got {self.parsimony!r}")
        if self.parsimony_alpha is not None and (
            isinstance(self.parsimony_alpha, bool)
            or not isinstance(self.parsimony_alpha, numbers.Real)
            or not 0.0 < float(self.parsimony_alpha) < 1.0
        ):
            raise ValueError(
                "parsimony_alpha must be None or a level strictly between 0 and 1, "
                f"got {self.parsimony_alpha!r}"
            )
        if not isinstance(self.certify_test, str) or self.certify_test not in ("marginal", "conditional"):
            raise ValueError(
                f"certify_test must be 'marginal' or 'conditional', got {self.certify_test!r}"
            )
        if self.certify_test == "conditional" and self._problem_type == "classification":
            raise ValueError(
                "certify_test='conditional' is for regression only: its test residualises "
                "the target on a linear model of the other terms, which a class label does "
                "not support. Use certify_test='marginal' for classification."
            )
        if not isinstance(self.input_test, (bool, np.bool_)):
            raise ValueError(f"input_test must be True or False, got {self.input_test!r}")
        if self.input_test and self.selector is None:
            raise ValueError(
                "input_test=True tests the inputs of the screened terms, and selector=None "
                "runs no screen; pass a selector or input_test=False"
            )

    def _make_selector(self, warn: bool = False) -> Selector:
        """Build the configured selector.

        A :class:`~beamfeat.selection.Selector` instance is used as it is, so
        its own ``target_fdr``, ``problem_type`` and (for
        :class:`~beamfeat.selection.PermutationSelector`) ``correction``
        apply, and the estimator's values of those settings do not. With
        ``warn``, each estimator setting that disagrees with the instance is
        named in a warning, since a ``PermutationSelector()`` passed for its
        other options would otherwise run Benjamini-Hochberg in silence where
        the estimator's default is Benjamini-Yekutieli.
        """
        if isinstance(self.selector, Selector):
            if warn:
                conflicts = [
                    f"{own}={self._setting(own)!r} (the instance has {theirs}="
                    f"{getattr(self.selector, theirs)!r})"
                    for own, theirs in (
                        ("target_fdr", "target_fdr"),
                        ("selection_correction", "correction"),
                        ("problem_type", "problem_type"),
                    )
                    if hasattr(self.selector, theirs)
                    and getattr(self.selector, theirs) != self._setting(own)
                ]
                if conflicts:
                    warnings.warn(
                        f"beamfeat: selector is a {type(self.selector).__name__} instance, "
                        "whose own settings apply; the estimator's "
                        + ", ".join(conflicts)
                        + (" is" if len(conflicts) == 1 else " are")
                        + " not used. Configure the instance, or pass the selector "
                        "by name to use the estimator's settings.",
                        UserWarning,
                        stacklevel=3,
                    )
            return self.selector
        selector_kwargs = {}
        if isinstance(self.selector, str) and self.selector.lower() in ("permutation", "perm"):
            selector_kwargs["correction"] = self.selection_correction
        return make_selector(
            self.selector,
            target_fdr=self.target_fdr,
            problem_type=self._problem_type,
            random_state=self.random_state,
            **selector_kwargs,
        )

    def _make_certify_selector(self) -> Selector:
        """The test the ``parsimony_holdout`` re-test applies.

        ``certify_test="marginal"`` re-runs the configured selector, exactly
        as the screen ran it. ``"conditional"`` tests each term given the
        others with a Freedman-Lane permutation test, at the screen's level,
        seed and permutation budget, with its step-up correction: the
        estimator's ``selection_correction``, or a permutation selector
        instance's own ``correction``.
        """
        screen = self._make_selector()
        if self.certify_test != "conditional":
            return screen
        budget = {
            name: getattr(screen, name)
            for name in ("n_permutations", "auto_permutations", "max_permutations")
            if hasattr(screen, name)
        }
        return _ConditionalPermutationSelector(
            target_fdr=screen.target_fdr,
            random_state=screen.random_state,
            correction=getattr(screen, "correction", self.selection_correction),
            **budget,
        )

    def _make_input_test(self) -> _InputPermutationTest:
        """The test ``input_test=True`` applies, at the screen's level and seed.

        Its permutation budget and step-up correction are the screen's when
        the screen is a permutation selector, and otherwise the defaults and
        the estimator's ``selection_correction``. The estimator's ``n_jobs``
        threads share the work; the p-values are the same for every value.
        """
        screen = self._make_selector()
        budget = {
            name: getattr(screen, name)
            for name in ("n_permutations", "auto_permutations", "max_permutations")
            if hasattr(screen, name)
        }
        return _InputPermutationTest(
            target_fdr=screen.target_fdr,
            problem_type=self._problem_type,
            random_state=screen.random_state,
            correction=getattr(screen, "correction", self.selection_correction),
            n_jobs=self.n_jobs,
            **budget,
        )

    def _test_inputs(self, nodes, data, target, screened):
        """The screened candidates for which the input test rejects every column's null.

        Every candidate is tested, not only the screened ones, so which
        pairs are tested does not depend on the screen's outcome on these
        rows. Fills :attr:`input_report_`. If that leaves no screened
        candidate, the screened set is returned with a warning, as the
        re-test of ``parsimony_holdout`` does.
        """
        pairs, result = self._make_input_test().test_inputs(nodes, data, target)
        for diagnostic in result.warnings_raised:
            warnings.warn(f"beamfeat input test: {diagnostic}", UserWarning, stacklevel=3)
        rejected = set(int(i) for i in result.selected)
        self.input_report_ = [
            {
                "formula": nodes[index].name,
                "column": column,
                "p_value": float(result.p_values[position]),
                "q_value": float(result.q_values[position]),
                "null_rejected": position in rejected,
            }
            for position, (index, column) in enumerate(pairs)
        ]
        dropped = {pairs[position][0] for position in range(len(pairs)) if position not in rejected}
        eligible = [index for index in screened if index not in dropped]
        self._inputs_dropped_ = len(screened) - len(eligible)
        report(
            self.verbose,
            1,
            f"input test: {len(pairs)} (term, column) pairs, null rejected for {len(rejected)}"
            f" -> {len(eligible)} of {len(screened)} certified terms kept",
        )
        if not eligible:
            self._inputs_dropped_ = 0
            warnings.warn(
                "beamfeat: for every certified term, the input test failed to reject the "
                f"null for at least one of its columns at target FDR {result.target_fdr:.3g}. "
                "The certified terms are used as if input_test were False.",
                NoDiscoveriesWarning,
                stacklevel=3,
            )
            return list(screened)
        return eligible

    def _setting(self, name: str) -> Any:
        """The estimator's value for a selector setting it shares."""
        if name == "problem_type":
            return self._problem_type
        return getattr(self, name)

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
        a ``parsimony_holdout`` re-test that rejected at least one null, which
        certifies the subset on rows of its own. ``"screened set"`` when it
        covers a larger set the returned features were drawn from, as after a
        re-test that rejected none when ``input_test`` dropped terms. ``None``
        when nothing is controlled at all.
        """
        if not self.fdr_controlled_:
            self.fdr_scope_ = None
            return
        if self.certification_result_ is not None and self.certification_result_.n_selected:
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
        # A lone term is still tested on rows it has not seen; only the
        # multiplicity correction is trivial in that case, not the test itself.
        selector = self._make_certify_selector()
        result = selector.select(np.column_stack(columns), y[rows])
        self.certification_result_ = result
        for diagnostic in result.warnings_raised:
            warnings.warn(f"beamfeat certification: {diagnostic}", UserWarning, stacklevel=2)
        if result.n_selected == 0:
            carries = (
                "it carries the screen's marginal guarantee, not a conditional one, and it "
                "is not compact."
                if self.certify_test == "conditional"
                else "it does carry the guarantee, but it is not compact."
            )
            warnings.warn(
                "beamfeat: the re-test of the parsimonious equation on the held-back rows "
                f"rejected no term's null at target FDR {selector.target_fdr:.3g}, so no "
                "compact certified equation is available on this sample. The whole screened "
                f"set is returned instead: {carries}",
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
            # Only a fit whose features came from the screened set can claim
            # the associations were certified; a fallback to the unfiltered
            # search output, or same-data selection, cannot.
            premise = (
                "the selected features come from the FDR-screened set (the screen rejected "
                "their nulls of no association), but "
                if self.fdr_controlled_
                else "the selected features carry no FDR guarantee, and "
            )
            warnings.warn(
                f"beamfeat: {premise}the assembled model is degenerate: {description}. "
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
        not re-tested at level q; under ``parsimony=None`` they *are* the
        set. ``False`` when the screen rejected no null (the estimator then
        returns no features, or under ``on_no_discoveries="fallback"`` the
        unfiltered search output, which carries no FDR guarantee, and a
        warning says so) or when selection ran on the search rows; ``None``
        when no selector was configured.

        Sets :attr:`fdr_scope_` alongside it, naming which set the flag is a
        statement about: ``"printed equation"`` when the guarantee covers the
        returned features themselves, ``"screened set"`` when it covers a
        larger set they were drawn from, ``None`` when the flag is not
        ``True``. The flag says whether there is a guarantee; the scope says
        what it is over.

        Returns the transformed training matrix on the full data.
        """
        self._validate_params()
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
        # if it is enabled, run on the first part; the terms parsimony retains
        # are then re-tested on the second, which they have not seen. Because
        # that set is fixed once the first part has been used, the re-test is
        # an ordinary fixed-candidate screen and its guarantee therefore covers
        # the printed equation, not merely the set it was drawn from. The price
        # is rows: both parts have to be large enough to test on.
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
            n_jobs=self.n_jobs,
            max_inputs=self.max_inputs,
            fitted_ops=self.fitted_ops,
            magnitude_slots=self.magnitude_slots,
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
        self.equations_: list[dict] | None = None
        self.input_report_: list[dict] | None = None
        self._inputs_dropped_ = 0
        forward_path = None

        if self.selector is not None:
            selection_rows = holdout_index if use_holdout else np.arange(n_samples)
            selection_data = {name: X[selection_rows, index] for index, name in enumerate(feature_names)}
            evaluator = Evaluator(selection_data, units=units)
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

            selector = self._make_selector(warn=True)
            self.equations_ = []
            # A single candidate is screened like any other set: only the
            # multiplicity correction is trivial for it, not the test.
            if candidate_nodes:
                matrix = np.column_stack(candidate_columns)
                selection_result = selector.select(matrix, selection_target)
                self.selection_result_ = selection_result
                correction = getattr(selector, "correction", None)
                report(
                    self.verbose,
                    1,
                    f"screen: {selection_result.n_candidates} candidates, "
                    f"{selection_result.method}"
                    + (f" ({correction.upper()})" if correction else "")
                    + f" at q={selector.target_fdr:g} -> "
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
                    # The terms parsimony and its fallbacks draw from: the
                    # screened set, less any term the input test removes.
                    eligible = sorted(screened)
                    if self.input_test:
                        eligible = self._test_inputs(
                            candidate_nodes, selection_data, selection_target, eligible
                        )
                    forward_path = self._forward_path(matrix, selection_target, eligible)
                    keep_order = self._parsimonious_subset(
                        matrix, selection_target, eligible, forward_path
                    )
                    report(
                        self.verbose,
                        1,
                        f"parsimony: {len(eligible)} certified -> "
                        f"{len(keep_order)} term{'' if len(keep_order) == 1 else 's'}"
                        f" (|S|/|S'| = {len(eligible) / len(keep_order):.2f})"
                        if self.parsimony is not None
                        else f"parsimony: off, keeping all {len(eligible)} "
                        f"certified term{'' if len(eligible) == 1 else 's'}",
                    )
                    retest = None
                    if fell_back_to_screened:
                        keep_order = list(eligible)
                    elif use_certify:
                        certified = self._certify_subset(
                            X, y, certify_index, feature_names, units,
                            [candidate_nodes[index] for index in keep_order],
                            keep_order,
                        )
                        retest = (len(certified), len(keep_order))
                        # Nothing compact could be certified, but the screened
                        # set still is; return that (less any term the input
                        # test removed) rather than nothing.
                        keep_order = certified if certified else list(eligible)
                    nodes = [candidate_nodes[index] for index in keep_order]
                    kept = set(keep_order)
                    self.fdr_controlled_ = bool(nodes)
                    if retest is not None and retest[0]:
                        given = ", each given the others" if self.certify_test == "conditional" else ""
                        scope = (
                            f" after the re-test on held-back rows rejected the null for {retest[0]}"
                            f" of {retest[1]} re-tested terms{given}; the guarantee covers the"
                            " fitted equation"
                        )
                    elif retest is not None:
                        scope = (
                            " from the screened set, since the re-test on held-back rows rejected"
                            " no null; the screen's guarantee covers the screened set"
                        )
                    elif len(keep_order) == len(screened):
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
                    nodes = self._no_discoveries(
                        nodes, selector.target_fdr, use_holdout, selection_result.warnings_raised
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
                # Nothing could even be evaluated on the selection rows, so no
                # null was tested; returning the search output here unannounced
                # would skip the screening the caller configured.
                nodes = self._no_discoveries(
                    nodes, selector.target_fdr, use_holdout,
                    ["no candidate could be evaluated on the selection rows"],
                )

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
        self._forward_stop_ = None
        if forward_path is not None:
            self.equations_ = _equation_list(candidate_nodes, forward_path, kept_nodes)
            self._forward_stop_ = forward_path[2]
        # |S|/|S'| for the set actually returned, so it is measured after the
        # parsimony re-test or its fallback rather than on the forward subset.
        # A set the re-test certified is its own S.
        if kept_nodes and self.selection_result_ is not None and self.selection_result_.n_selected:
            certified = self.certification_result_
            base = (
                certified.n_selected
                if certified is not None and certified.n_selected
                else self.selection_result_.n_selected
            )
            self.fdp_inflation_ = float(base) / len(kept_nodes)
        if not kept_nodes:
            # The honest empty outcome: zero constructed columns. Downstream
            # models degrade to an intercept-only fit rather than pretending.
            return np.empty((n_samples, 0), dtype=np.float64)
        return np.column_stack(kept_columns)

    def _no_discoveries(
        self, nodes: list[Node], target_fdr: float, use_holdout: bool, notes: Sequence[str]
    ) -> list[Node]:
        """Apply ``on_no_discoveries`` when the screen rejected no null.

        Returns the expressions to keep: the unfiltered search output under
        ``"fallback"``, none under ``"empty"``. ``"raise"`` raises.
        """
        self.fdr_controlled_ = False
        message = (
            f"beamfeat: the FDR-controlled screen rejected no null at target "
            f"FDR {target_fdr:.3g} "
            + ("on the held-out split " if use_holdout else "on the search rows ")
            + f"(on_no_discoveries={self.on_no_discoveries!r}). "
        )
        if notes:
            message += "Selector notes: " + "; ".join(notes) + ". "
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
                stacklevel=3,
            )
            return nodes
        warnings.warn(
            message + "Returning no constructed features; the model will "
            "predict the training mean (regression) or class prior "
            "(classification). Set on_no_discoveries='fallback' to keep "
            "the unfiltered search output instead.",
            NoDiscoveriesWarning,
            stacklevel=3,
        )
        return []

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

    def _apply(self, X: np.ndarray, outcome: str = "are NaN in the output") -> np.ndarray:
        """Evaluate the stored expressions on new data.

        An expression that cannot be evaluated on some row — a logarithm of a
        negative value, overflow, or any other non-finite result — is NaN on
        that row and only that row. Every other row gets exactly the value it
        would get if evaluated alone, so a prediction never depends on what
        else is in the batch. A :class:`UserWarning` names each such
        expression and the number of rows it failed on; ``outcome`` completes
        its last sentence with what the caller does with those rows.

        Search-time admissibility filters are deliberately not applied here.
        Those criteria are data-dependent, so applying them would make the
        output depend on which rows are present: a subset in which some column
        happens to be constant would silently produce different values than the
        full data. See :meth:`~beamfeat.expression.Evaluator.transform_values`.

        This method does not mutate the estimator. scikit-learn requires
        ``transform`` and ``predict`` to leave ``__dict__`` untouched, so the
        failures are reported rather than stored.
        """
        if not self.features_:
            return np.empty((X.shape[0], 0), dtype=np.float64)

        check_is_fitted(self, ["features_"])
        feature_names = self._input_names(X, X.shape[1])
        data = {name: X[:, index] for index, name in enumerate(feature_names)}
        # warn=False: coverage is a fit-time concern, and this path runs on
        # every transform and predict.
        units = self._resolve_units(feature_names, warn=False)
        evaluator = Evaluator(data, units=units)

        columns = []
        failures = []
        for node in self.features_:
            values = evaluator.transform_values(node)
            if values is None:
                values = _evaluate_by_row(node, data, units, X.shape[0])
                n_failed = int(np.count_nonzero(np.isnan(values)))
                failures.append(f"{node.name!r} on {n_failed} row{'' if n_failed == 1 else 's'}")
            columns.append(values)

        matrix = np.column_stack(columns)
        if failures:
            n_rows = int(np.count_nonzero(np.isnan(matrix).any(axis=1)))
            warnings.warn(
                f"beamfeat: could not evaluate {', '.join(failures)} (outside the "
                f"expression's domain, or not finite). {n_rows} of {X.shape[0]} rows are "
                f"affected and {outcome}; the other rows are unaffected.",
                UserWarning,
                stacklevel=4,
            )
        return matrix

    def _model_output(self, X: np.ndarray, method: str, outcome: str) -> tuple[np.ndarray, np.ndarray]:
        """Call ``model_.<method>`` on the rows every expression is defined on.

        Returns the output with NaN on the other rows, and the mask of rows
        that were defined. Undefined rows are passed to the model at the
        training mean only so the output keeps its shape, and then
        overwritten.
        """
        matrix = self._apply(X, outcome)
        defined = ~np.isnan(matrix).any(axis=1)
        if not defined.all():
            matrix = np.where(defined[:, None], matrix, self.scaler_.mean_)
        output = getattr(self.model_, method)(self.scaler_.transform(matrix))
        if not defined.all() and output.dtype.kind == "f":
            output[~defined] = np.nan
        return output, defined

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
        # One symbol map for every feature, so a column keeps its symbol
        # across features even when its sanitised name needs a suffix.
        symbols = _sympy_symbols(set().union(*(node.columns() for node in self.features_)))
        return [node.to_sympy(symbols) for node in self.features_]

    def latex_formulas(self) -> list[str]:
        """Return the selected expressions as LaTeX strings.

        Rendered by ``sympy.latex`` from :meth:`to_sympy`, so sympy is
        required, as it is there; the formulas are in the order of
        :attr:`feature_formulas_`.
        """
        import sympy

        return [sympy.latex(expression) for expression in self.to_sympy()]

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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.non_deterministic = False
        return tags


class BeamFeatTransformer(TransformerMixin, _BeamFeatBase):
    """Constructs and selects features, for use inside a pipeline.

    Args:
        scorer: Scoring strategy for the search. A name
            (``"correlation"``, ``"spearman"``, ``"mutual_information"``,
            ``"gradient_boosting"``, or an alias such as ``"mi"``) or a
            :class:`~beamfeat.scoring.Scorer` instance.
        selector: FDR-controlled selector, or ``None`` to keep the search
            output unfiltered. A name (``"permutation"``, ``"knockoff"``) or a
            :class:`~beamfeat.selection.Selector` instance, whose own settings
            apply in place of ``target_fdr`` and ``selection_correction``.
        problem_type: ``"regression"`` or ``"classification"``. Determines how
            the scorer and selector treat the target.
        max_depth: Maximum expression depth.
        beam_width: Expressions retained at each search depth.
        max_features: Cap on features returned by the search, before selection.
        target_fdr: Nominal false discovery rate for the selector.
        unary_ops: Unary operators to apply.
        binary_ops: Binary operators to apply.
        redundancy_threshold: Absolute correlation at or above which a
            candidate is treated as a duplicate.
        include_originals: Whether input columns compete alongside constructed
            expressions.
        units: Optional units of the input columns: a mapping of column name
            to unit, or a sequence with one unit per column. A unit is a pint
            quantity or unit, or a string pint parses, such as ``"kg"`` or
            ``"m / s**2"``. Supplying units restricts the search to
            dimensionally valid expressions; cover every column, since one
            without a unit combines freely with the rest.
        selection_holdout: Fraction of the training rows held out of the
            search for selection, strictly between 0 and 1, or ``None`` to
            select on the search rows (which forfeits the guarantee, so
            :attr:`fdr_controlled_` is then ``False``).
        parsimony_holdout: Fraction of the selection rows held back to
            re-test the returned terms, strictly between 0 and 1, or ``None``
            (default) for no re-test. ``certify_test`` sets the test.
        selection_correction: ``"by"`` (Benjamini-Yekutieli, default) or
            ``"bh"`` (Benjamini-Hochberg), used when ``selector`` is the name
            ``"permutation"``. A selector instance keeps its own correction
            and ``target_fdr``, and a warning names any estimator setting it
            overrides; ``PermutationSelector()`` defaults to ``"bh"``.
        on_no_discoveries: What to return when the screen rejects no null:
            ``"empty"`` (default) keeps no constructed features,
            ``"fallback"`` keeps the unfiltered search output with
            :attr:`fdr_controlled_` ``False``, ``"raise"`` raises
            :class:`NoDiscoveriesError`.
        parsimony: ``"forward"`` (default) fits a compact subset of the
            screened set, chosen by greedy forward selection on the selection
            rows. That subset is not re-tested, so the FDR guarantee covers
            the set it was drawn from and not the terms returned, and
            :attr:`fdr_scope_` (and the regressor's and classifier's
            ``equation()``) marks this whenever terms were dropped. ``None``
            returns the screened set entire, so the returned features are
            exactly the set the guarantee covers; ``parsimony_holdout`` buys
            both, for rows. Either way :attr:`equations_` records the forward
            path through the whole screened set.
        parsimony_tol: Minimum incremental R^2 a feature must add to be kept by
            ``parsimony="forward"``. Ignored when ``parsimony`` is ``None``.
        random_state: Seed. Fitting is deterministic given this.
        verbose: Progress reporting to stdout. ``0`` (default) is silent;
            ``1`` prints one line per stage -- the split, the search, the
            screening, the parsimony step and the fitted result; ``2`` adds
            per-depth search detail and the strongest certified candidates
            with their p- and q-values.
        n_jobs: Workers that evaluate and score the search's candidates:
            threads, or processes for ``"gradient_boosting"`` (see
            :class:`~beamfeat.search.BeamSearch`). ``1`` (default) uses the
            calling thread only, ``-1`` every core, and ``None`` follows an
            enclosing joblib configuration (``joblib.parallel_config``, or
            ``parallel_backend`` before joblib 1.3). Small searches run in the
            calling thread regardless. With ``input_test``, as many threads
            share its tests. The fitted result is the same for every value.
        max_inputs: Most distinct input columns one feature may use, or
            ``None`` (default) for no limit. Candidates over it are ruled out
            during the search, before evaluation. A feature of depth d uses
            at most 2**d columns, so at the default ``max_depth`` of 2 a value
            of 4 or more has no effect. A law that needs more inputs than
            the limit cannot be found.
        fitted_ops: Parametric operators whose one constant is fitted
            inside the expression: ``"exp_scaled"`` (``exp(a*x)``),
            ``"pow"`` (``x^a``), ``"inv_shift"`` (``1/(x + a)``) and
            ``"log_shift"`` (``log(x + a)``). All four by default; ``()``
            keeps every expression free of constants, as the default did
            before 0.6.0, and fits about twice as fast on small data. The
            constants are fitted during the search, on the search rows only
            (see :class:`~beamfeat.search.BeamSearch`), so the screen on the
            selection rows sees fixed expressions and its guarantee is
            unchanged; with ``selection_holdout=None`` the rows are shared,
            as they are for the search itself. A fitted constant is part of
            the expression: ``transform`` and ``predict`` reuse it, and the
            formulas and :meth:`to_sympy` show it, as an exact fraction when
            it was snapped to one. A law the plain operators can write keeps
            its plain form unless the snapping test, on the search rows,
            rejects the null that the plain form fits as well: ``log(x0)``,
            not a fitted near copy such as ``(x0)^(-0.02)``.
        certify_test: The test the ``parsimony_holdout`` re-test applies to
            each returned term on the held-back rows. ``"marginal"``
            (default) re-runs the configured selector, whose null is that the
            term is independent of the target, so it rejects the null of a
            term that only rides on a true one, ``x0*x7`` when the target
            depends on ``x0`` alone. ``"conditional"`` tests each term given
            the other returned terms, with a Freedman-Lane permutation test:
            the target and the term are residualised on the other terms and
            an intercept, the target's residuals are permuted, the statistic
            is the partial correlation, and the screen's step-up correction
            is applied at the screen's level. Its null is that the term's
            coefficient is zero in a linear model of the target on all the
            returned terms. A term standing in for a true one the others
            leave out is not null in that sense, so rejecting its null is not
            a false rejection. Its p-values are approximate: they rest on the
            errors of a linear model of the target on the other terms being
            exchangeable (independent and identically distributed, say, so of
            constant variance), and are close to exact rather than exact even
            then, except for a single term, where the test is the marginal
            one. If the re-test rejects no null, the screened set is
            returned, which carries the screen's marginal guarantee only.
            Regression only; ignored when ``parsimony_holdout`` is ``None``.
            See the guarantees page for its measured calibration.
        magnitude_slots: Beam slots added at each search depth for
            candidates ranked by how well their magnitude tracks the
            target's, for laws whose parts do not correlate with the target,
            such as ``x0*x1*x2`` with centred inputs, whose pairwise products
            the default ranking leaves among noise. ``0`` (default) adds none
            and leaves the fit unchanged. On wide tables, where the first
            depth cannot propose every pair of inputs, the inputs are also
            proposed in an order that interleaves the two rankings. The
            ranking is part of the search, on the search rows only (see
            :class:`~beamfeat.search.BeamSearch`), so the screen on the
            selection rows sees fixed expressions and its guarantee is
            unchanged. Regression only.
        input_test: Whether to test, on the selection rows, each column
            inside each candidate that reads more than one, and drop a
            screened term before ``parsimony`` runs when the test fails to
            reject the null for one of its columns. ``False`` (default)
            leaves the fit unchanged. A column is tested by permuting its
            rows alone, wherever it occurs in the expression, and recomputing
            the screen's statistic; the screen's step-up correction is
            applied across every (candidate, column) pair of every
            candidate, screened or not. The null is that the column is
            independent of the target and of the candidate's other columns,
            under which the p-values are exact: it removes
            ``(x0*x1)/(x1 + x8)`` when ``x8`` is unrelated to everything,
            but a junk column correlated with a real one is not null in that
            sense, so its null can be rejected and the term kept. The
            screen's guarantee is unchanged and still covers the screened
            set. If the test would drop every screened term, the screened
            set is used with a warning. Each draw re-evaluates the
            expression, so the time grows with the rows: a few seconds at a
            few hundred rows, minutes at ten thousand, which ``n_jobs``
            threads share.
        parsimony_alpha: Level of a chance check on each term the forward
            step of ``parsimony="forward"`` adds after the first, or ``None``
            (default) for none, which leaves the fit unchanged. On a noisy
            target the best of many remaining terms adds more than
            ``parsimony_tol`` to R^2 by fitting noise alone, so the step keeps
            riders. With a level, the step also stops at the first term
            whose gain fails to reject the null that it is no larger than
            the best gain the remaining terms reach against noise: 200
            wild-bootstrap draws of the current residual (each row's
            residual times an independent standard normal), net of the
            terms already added. Like ``parsimony_tol`` it shapes the
            returned subset, on the selection rows, and not the screen,
            whose guarantee is unchanged; it is a stopping rule, not a
            guarantee on the subset. It suits a target of a few strong
            terms: on real data, whose many modest effects each add little,
            it stopped after one to four terms and lowered held-out score
            on 8 of 12 datasets in ``benchmarks/release_comparison/``.
            Ignored when ``parsimony`` is ``None``.

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
        equations_: The greedy forward path through the screened set, one
            record per size ``k`` from 1 to the size of the screened set,
            in order of ``k``. Each record is a dict with ``"n_terms"``
            (``k``); ``"formulas"`` and ``"features"``, the ``k`` terms as
            formula strings and :class:`~beamfeat.expression.Node`
            expressions in the order the path adds them; ``"r2"``, the
            in-sample R^2 of a least-squares fit of them on the rows
            screening used (of the integer-coded labels for
            classification, a ranking heuristic); ``"complexity"``, the
            number of nodes in their expression trees plus one per term for
            its coefficient and one per fitted constant; and ``"model"``, whether they are the returned
            features. At most one record is marked, and none is when the
            returned features are not on the path, after a
            ``parsimony_holdout`` re-test for instance. The path is recorded
            whatever ``parsimony`` is, and the steps up to the returned
            subset are those the forward step takes, so recording it changes
            nothing that is returned. ``None`` without a selector, and empty
            when the screen rejected no null. The FDR guarantee covers the
            screened set, the last record, and not a record chosen from the
            list; :meth:`equations` renders it and says so.
        selection_report_: One dict per candidate screened, in the search's
            order: ``"formula"``; ``"p_value"`` and ``"q_value"``, ``None``
            for a knockoff selector, which computes neither; ``"statistic"``;
            ``"screened"``, whether the screen rejected its null, which marks
            the set the guarantee covers; and ``"kept"``, whether it is among
            the returned features. ``None`` without a selector.
        fdp_inflation_: The size of the certified set divided by the number
            of returned features, |S|/|S'|: the factor by which returning a
            subset can raise the realised false discovery proportion. ``1.0``
            when the returned features are the certified set, ``None`` when
            nothing was certified.
        input_report_: One dict per (candidate, column) pair the input test
            ran on: ``"formula"``, ``"column"``, ``"p_value"``,
            ``"q_value"`` and ``"null_rejected"``, whether the step-up
            correction rejects the column's null, that it is independent of
            the target and of the candidate's other columns. ``None`` when
            ``input_test`` is ``False``.
        certification_result_: The :class:`~beamfeat.selection.SelectionResult`
            of the ``parsimony_holdout`` re-test, or ``None`` when it did not
            run. Its ``method`` is ``"conditional_permutation"`` under
            ``certify_test="conditional"``.
        n_features_out_: Number of features produced.
        n_features_in_: Number of input columns seen during fit.
        feature_names_in_: Names of the input columns, when ``X`` had string
            column names.

    A row on which a fitted expression cannot be evaluated at transform time
    (outside its domain, such as the logarithm of a negative value) is NaN in
    that expression's column, with a warning naming the expression; the other
    rows are unaffected. Impute or drop those rows before a downstream model
    that does not accept NaN.
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
        n_jobs: int | None = 1,
        max_inputs: int | None = None,
        fitted_ops: tuple[str, ...] = DEFAULT_FITTED,
        certify_test: str = "marginal",
        magnitude_slots: int = 0,
        input_test: bool = False,
        parsimony_alpha: float | None = None,
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
            n_jobs=n_jobs,
            max_inputs=max_inputs,
            fitted_ops=fitted_ops,
            certify_test=certify_test,
            magnitude_slots=magnitude_slots,
            input_test=input_test,
            parsimony_alpha=parsimony_alpha,
        )
        self.problem_type = problem_type

    def fit(self, X, y=None) -> BeamFeatTransformer:
        """Construct and select features from the training data.

        Args:
            X (array-like): ``(n_samples, n_features)`` training data.
            y (array-like): Target. Required, since both search and selection
                are supervised.

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
        model_: The fitted downstream linear model, ``None`` when no feature
            was kept.
        scaler_: The ``StandardScaler`` applied to the features before the
            linear model, ``None`` when no feature was kept. ``coef_`` divided
            by its ``scale_`` gives the coefficients on the raw features.
        coef_: Coefficients of the downstream model, on the standardised
            features.
        intercept_: Intercept of the downstream model; the training mean of
            the target when no feature was kept.
        alpha_: The ridge penalty used, chosen by cross-validation when
            ``alpha="auto"``; ``None`` when no feature was kept.
        fdr_scope_: Which set :attr:`fdr_controlled_` is a statement about;
            see :class:`BeamFeatTransformer`. ``"screened set"`` is what
            :meth:`equation` marks in its output.
        equations_: The forward path through the screened set, one record
            per number of terms, with the record of the fitted model marked;
            see :class:`BeamFeatTransformer` and :meth:`equations`.

    The other fitted attributes, such as :attr:`selection_report_`, are those
    of :class:`BeamFeatTransformer`.
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
        n_jobs: int | None = 1,
        max_inputs: int | None = None,
        fitted_ops: tuple[str, ...] = DEFAULT_FITTED,
        certify_test: str = "marginal",
        magnitude_slots: int = 0,
        input_test: bool = False,
        parsimony_alpha: float | None = None,
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
            n_jobs=n_jobs,
            max_inputs=max_inputs,
            fitted_ops=fitted_ops,
            certify_test=certify_test,
            magnitude_slots=magnitude_slots,
            input_test=input_test,
            parsimony_alpha=parsimony_alpha,
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
            self.alpha_ = None
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
        """Predict using the constructed features and fitted model.

        A row on which a fitted expression cannot be evaluated (outside its
        domain, such as the logarithm of a negative value) is predicted as
        NaN, with a warning; the other rows are unaffected. :meth:`score`
        raises on such rows rather than scoring the others in silence.
        """
        check_is_fitted(self, ["model_"])
        X = _validate_input(self, X, dtype=np.float64, reset=False)
        if self.model_ is None:
            return np.full(X.shape[0], self.intercept_, dtype=np.float64)
        return self._model_output(X, "predict", "are predicted as NaN")[0]

    def score(self, X, y, sample_weight=None) -> float:
        """Return the R^2 of :meth:`predict` on ``X``.

        Raises:
            ValueError: If a fitted expression cannot be evaluated on some
                row. R^2 over the remaining rows would be a different
                quantity, so it is not returned in its place.
        """
        from sklearn.metrics import r2_score

        predictions = self.predict(X)
        undefined = int(np.count_nonzero(np.isnan(predictions)))
        if undefined:
            raise ValueError(
                f"{undefined} of {len(predictions)} rows are outside the domain of a fitted "
                "expression and have no prediction, so R^2 is undefined on this data; "
                "drop or impute those rows before scoring."
            )
        return float(r2_score(y, predictions, sample_weight=sample_weight))

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
                standardised coefficients. The omitted terms are held at their
                training mean, whose contribution moves into the constant, so
                the shortened equation agrees with the model on average
                rather than treating the omitted terms as zero.
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
            return f"y = {intercept}  (no feature selected)"
        return _render_linear_equation(
            "y", self.coef_, float(self.intercept_), self.scaler_,
            self.feature_formulas_, precision, max_terms, style,
        ) + self._parsimony_note()

    def latex(
        self,
        precision: int = 4,
        max_terms: int | None = None,
        style: str = "significant",
        note: bool = True,
    ) -> str:
        """Return the fitted model as a LaTeX equation.

        The same equation as :meth:`equation`, with the same terms, order and
        coefficients, and the same note when the parsimony step dropped terms
        from the screened set, set as ``\\text``. Each feature is rendered
        by ``sympy.latex`` from :meth:`to_sympy`, so sympy is required, as it
        is there. The string has no math delimiters; in a notebook,
        ``IPython.display.Math(model.latex())`` renders it.

        Args:
            precision: As in :meth:`equation`.
            max_terms: As in :meth:`equation`.
            style: As in :meth:`equation`. E-notation is written as a power
                of ten.
            note: Whether to append the bracketed note. ``False`` returns the
                bare equation, for pasting into a document; the guarantee
                still covers the screened set, not the printed subset.

        Returns:
            A string such as ``y = 2.104 a b - 0.512 \\log{\\left(c \\right)} + 3.991``.
        """
        check_is_fitted(self, ["model_"])
        _check_equation_format(precision, style)
        expressions = self.to_sympy()
        if self.model_ is None:
            intercept = _latex_number(_format_coefficient(self.intercept_, precision, style))
            if not note:
                return f"y = {intercept}"
            return rf"y = {intercept} \quad {_latex_text('(no feature selected)')}"
        return "y = " + _render_linear_latex(
            self.coef_, float(self.intercept_), self.scaler_,
            expressions, precision, max_terms, style,
        ) + (self._latex_parsimony_note() if note else "")


class BeamFeatClassifier(ClassifierMixin, _BeamFeatBase):
    """Constructs features, then fits a logistic model on them.

    Args:
        C: Inverse regularisation strength for the downstream model.

    Other arguments match :class:`BeamFeatTransformer`.

    Attributes:
        features_: Selected expressions.
        classes_: Class labels seen during fit.
        model_: The fitted downstream logistic model, ``None`` when no
            feature was kept or the training data had a single class.
        scaler_: The ``StandardScaler`` applied to the features before the
            logistic model, ``None`` when no feature was kept.
        coef_: Coefficients of the logistic model, on the standardised
            features; empty when no feature was kept.
        intercept_: Intercepts of the logistic model; empty when no feature
            was kept.
        class_prior_: Class frequencies of the training labels, set only when
            no feature was kept, in which case the model predicts them.
        fdr_scope_: Which set :attr:`fdr_controlled_` is a statement about;
            see :class:`BeamFeatTransformer`. ``"screened set"`` is what
            :meth:`equation` marks in its output.
        equations_: The forward path through the screened set, one record
            per number of terms, with the record of the fitted model marked;
            see :class:`BeamFeatTransformer` and :meth:`equations`. Its R^2
            is of the integer-coded labels, the forward step's ranking
            heuristic.

    The other fitted attributes, such as :attr:`selection_report_`, are those
    of :class:`BeamFeatTransformer`.
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
        n_jobs: int | None = 1,
        max_inputs: int | None = None,
        fitted_ops: tuple[str, ...] = DEFAULT_FITTED,
        certify_test: str = "marginal",
        magnitude_slots: int = 0,
        input_test: bool = False,
        parsimony_alpha: float | None = None,
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
            n_jobs=n_jobs,
            max_inputs=max_inputs,
            fitted_ops=fitted_ops,
            certify_test=certify_test,
            magnitude_slots=magnitude_slots,
            input_test=input_test,
            parsimony_alpha=parsimony_alpha,
        )
        self.C = C

    def fit(self, X, y):
        """Construct features and fit the downstream classifier."""
        from sklearn.preprocessing import StandardScaler
        X, y = _validate_input(self, X, y, dtype=np.float64, y_numeric=False)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        # Set only by a fit that keeps no feature, and read by predict when
        # there is no model, so a refit must not inherit it.
        if hasattr(self, "class_prior_"):
            del self.class_prior_

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
        # predict cannot return NaN as a label, so a row it cannot evaluate is
        # given the prediction of a model with no features at all.
        counts = np.array([np.sum(y == klass) for klass in self.classes_])
        self._majority_class_ = self.classes_[int(np.argmax(counts))]

        if len(self.classes_) < 2:
            # A single-class training set has nothing to discriminate; store
            # the constant so predict still returns something coherent.
            self.model_ = None
            self.coef_ = np.empty((0, 0))
            self.intercept_ = np.empty(0)
            return self

        self.model_ = logistic_regression(C=self.C, max_iter=2000, random_state=self.random_state).fit(
            self.scaler_.transform(matrix), y
        )
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        self._holdout_fit_check(X, y)
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
        """Predict class labels.

        A row on which a fitted expression cannot be evaluated (outside its
        domain, such as the logarithm of a negative value) cannot be given a
        NaN label, so it is assigned the most frequent training class -- the
        prediction of a model with no features -- with a warning. The other
        rows are unaffected. :meth:`predict_proba` and
        :meth:`decision_function` return NaN for such rows, which is how to
        find them.
        """
        check_is_fitted(self, ["features_"])
        X = _validate_input(self, X, dtype=np.float64, reset=False)
        if self.model_ is None:
            if hasattr(self, "class_prior_"):
                return np.full(X.shape[0], self.classes_[int(np.argmax(self.class_prior_))])
            return np.full(X.shape[0], self.classes_[0])
        labels, defined = self._model_output(
            X, "predict", "are assigned the most frequent training class"
        )
        labels[~defined] = self._majority_class_
        return labels

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
                standardised coefficients per line. The omitted terms are held
                at their training mean, whose contribution moves into the
                constant.
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
                    f"P({_label(klass)}) = {_format_coefficient(p, precision, style)}"
                    for klass, p in zip(self.classes_, self.class_prior_, strict=True)
                )
                return f"class priors only (no feature selected): {prior}"
            return f"constant prediction: {_label(self.classes_[0])}"

        coefficients = np.atleast_2d(self.coef_)
        intercepts = np.atleast_1d(self.intercept_)
        note = self._parsimony_note()
        if coefficients.shape[0] == 1:
            return _render_linear_equation(
                f"logit P(y = {_label(self.classes_[1])})",
                coefficients[0], float(intercepts[0]), self.scaler_,
                self.feature_formulas_, precision, max_terms, style,
            ) + note
        lines = [
            _render_linear_equation(
                f"score({_label(klass)})", coefficients[row], float(intercepts[row]),
                self.scaler_, self.feature_formulas_, precision, max_terms, style,
            )
            for row, klass in enumerate(self.classes_)
        ]
        # One note for the whole block: the terms are shared across the lines.
        return ("\n".join(lines)
                + "\n(predicted class = argmax of the softmax scores)" + note)

    def latex(
        self,
        precision: int = 4,
        max_terms: int | None = None,
        style: str = "significant",
        note: bool = True,
    ) -> str:
        """Return the fitted classifier as a LaTeX log-odds equation.

        The same equations as :meth:`equation`: for a binary problem the
        log-odds of ``classes_[1]``, for a multiclass problem one softmax
        score per class in an ``aligned`` block, followed by the same note
        when the parsimony step dropped terms from the screened set. Each
        feature is rendered by ``sympy.latex`` from :meth:`to_sympy`, so
        sympy is required, as it is there. The string has no math
        delimiters.

        Args:
            precision: As in :meth:`equation`.
            max_terms: As in :meth:`equation`.
            style: As in :meth:`equation`. E-notation is written as a power
                of ten.
            note: Whether to append the bracketed notes. ``False`` returns the
                bare equations, for pasting into a document; the guarantee
                still covers the screened set, not the printed subset.
        """
        check_is_fitted(self, ["model_"])
        _check_equation_format(precision, style)
        expressions = self.to_sympy()
        if self.model_ is None:
            if hasattr(self, "class_prior_"):
                prior = r",\; ".join(
                    f"P(y = {_latex_label(klass)}) = "
                    + _latex_number(_format_coefficient(p, precision, style))
                    for klass, p in zip(self.classes_, self.class_prior_, strict=True)
                )
                if not note:
                    return prior
                return rf"{prior} \quad {_latex_text('(class priors only; no feature selected)')}"
            constant = f"y = {_latex_label(self.classes_[0])}"
            return rf"{constant} \quad {_latex_text('(constant prediction)')}" if note else constant

        coefficients = np.atleast_2d(self.coef_)
        intercepts = np.atleast_1d(self.intercept_)
        suffix = self._latex_parsimony_note() if note else ""
        if coefficients.shape[0] == 1:
            return (
                rf"\operatorname{{logit}} P(y = {_latex_label(self.classes_[1])}) = "
                + _render_linear_latex(
                    coefficients[0], float(intercepts[0]), self.scaler_,
                    expressions, precision, max_terms, style,
                )
                + suffix
            )
        lines = [
            rf"\operatorname{{score}}({_latex_label(klass)}) &= "
            + _render_linear_latex(
                coefficients[row], float(intercepts[row]), self.scaler_,
                expressions, precision, max_terms, style,
            )
            for row, klass in enumerate(self.classes_)
        ]
        return (
            "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}"
            + (rf" \quad {_latex_text('(predicted class = argmax of the softmax scores)')}" if note else "")
            + suffix
        )

    def _path_lhs(self) -> str:
        """Left-hand side naming the modelled quantity in :meth:`equations`.

        The log-odds of ``classes_[1]`` for a binary problem, as in
        :meth:`equation`; the per-class scores, which share their terms,
        otherwise.
        """
        if len(self.classes_) == 2:
            return f"logit P(y = {_label(self.classes_[1])})"
        return "scores"

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        NaN on a row where a fitted expression cannot be evaluated; see
        :meth:`predict`.
        """
        check_is_fitted(self, ["features_"])
        X = _validate_input(self, X, dtype=np.float64, reset=False)
        if self.model_ is None:
            if hasattr(self, "class_prior_"):
                return np.tile(self.class_prior_, (X.shape[0], 1))
            return np.ones((X.shape[0], 1))
        return self._model_output(X, "predict_proba", "have NaN probabilities")[0]

    def decision_function(self, X) -> np.ndarray:
        """Return the model's decision scores.

        With no feature selected the scores are the log prior odds (binary)
        or the log priors (multiclass), so they agree with :meth:`predict`
        and :meth:`predict_proba`. NaN on a row where a fitted expression
        cannot be evaluated; see :meth:`predict`.
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
        return self._model_output(X, "decision_function", "have NaN scores")[0]
