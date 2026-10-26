"""The fitting machinery the three estimators share.

:class:`_BeamFeatBase` runs a fit in stages: split the rows, search, screen
the search's expressions on held-out rows, choose the terms to fit, and
record the result. The public classes in :mod:`beamfeat.estimators` add the
downstream model, prediction and the readable equation.
"""

from __future__ import annotations

import logging
import numbers
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from beamfeat._forward import alpha_stop, forward_path
from beamfeat._prediction import _evaluate_by_row
from beamfeat._rendering import _equation_list, _EquationsMixin
from beamfeat.expression import Evaluator, Node, _import_sympy, _sympy_symbols
from beamfeat.scoring import Scorer, is_constant
from beamfeat.search import DEFAULT_BINARY, DEFAULT_FITTED, DEFAULT_UNARY, BeamSearch, report
from beamfeat.selection import (
    Selector,
    _ConditionalPermutationSelector,
    _InputPermutationTest,
    make_selector,
)

# Named for the public module, where the estimators are, so logging
# configured for beamfeat.estimators covers the fit.
logger = logging.getLogger("beamfeat.estimators")

# Magnitude slots under magnitude_slots="auto" for regression: the setting of
# benchmarks/search_stress/ and benchmarks/release_comparison/.
_AUTO_MAGNITUDE_SLOTS = 10

# The level whose stopping point the equation shows beside the default
# forward step's (see _EquationsMixin._alpha_hint).
_HINT_ALPHA = 0.05

# Residual slots under residual_slots="auto" for regression.
_AUTO_RESIDUAL_SLOTS = 10


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


class ExtrapolationWarning(UserWarning):
    """Raised when new rows take a feature far beyond its range on the training rows.

    A formula is fitted where the training data lie, and beyond them it can
    behave unlike anything the fit saw: a ratio whose denominator approaches
    zero there takes any size and either sign. ``predict``, ``transform`` and
    the probability methods warn when a feature's values on the new rows lie
    more than one width of its training range beyond that range, and say on
    how many rows, which features and how far. Predictions on
    those rows extrapolate the equation; check them before relying on them.
    """


# How far beyond its range on the training rows, in widths of that range, a
# feature may go on new rows before ExtrapolationWarning is raised.
_EXTRAPOLATION_WIDTHS = 1.0


# Raised here and imported, documented and caught as beamfeat.estimators.<name>;
# the module name keeps tracebacks and warnings filters on that path.
for _public in (NoDiscoveriesError, DegenerateFitWarning, NoDiscoveriesWarning, ExtrapolationWarning):
    _public.__module__ = "beamfeat.estimators"


class _Rows(NamedTuple):
    """The training rows each stage of a fit uses (row indices into ``X``)."""

    search: np.ndarray
    selection: np.ndarray
    certify: np.ndarray
    use_holdout: bool
    use_certify: bool
    fell_back_to_screened: bool


class _BeamFeatBase(_EquationsMixin, BaseEstimator):
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
        magnitude_slots: int | str = "auto",
        input_test: bool = False,
        parsimony_alpha: float | None = None,
        residual_slots: int | str = "auto",
        max_search_rows: int | None = None,
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
        self.residual_slots = residual_slots
        self.max_search_rows = max_search_rows
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
        return forward_path(
            matrix,
            target,
            screened,
            classification=self._problem_type == "classification",
            parsimony_tol=self.parsimony_tol,
            parsimony_alpha=self.parsimony_alpha,
            random_state=self.random_state,
        )

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
        if self.max_search_rows is not None and (
            isinstance(self.max_search_rows, bool)
            or not isinstance(self.max_search_rows, numbers.Integral)
            or self.max_search_rows < 1
        ):
            raise ValueError(
                f"max_search_rows must be None or a positive integer, got {self.max_search_rows!r}"
            )
        for name in ("magnitude_slots", "residual_slots"):
            value = getattr(self, name)
            if isinstance(value, str) and value != "auto":
                raise ValueError(f"{name} must be 'auto' or a non-negative integer, got {value!r}")
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
        if (
            self.certify_test == "conditional"
            and self.parsimony_holdout is not None
            and self.parsimony is not None
            and self.parsimony_alpha is None
        ):
            raise ValueError(
                "certify_test='conditional' after the forward step needs parsimony_alpha. "
                "On a noisy target the forward step hands the re-test riders it cannot "
                "certify, and the whole screened set is then returned: 40 terms on diabetes "
                "and mpg in benchmarks/release_comparison. Pass parsimony_alpha=0.05, which "
                "stops the step at the first gain chance can reach, or parsimony=None to "
                "re-test the whole screened set."
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
                        stacklevel=4,
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
            warnings.warn(f"beamfeat input test: {diagnostic}", UserWarning, stacklevel=5)
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
                stacklevel=5,
            )
            return list(screened)
        return eligible

    def _magnitude_slots(self) -> int:
        """``magnitude_slots``, with ``"auto"`` resolved: 10 for regression, none for classification."""
        if isinstance(self.magnitude_slots, str):
            return 0 if self._problem_type == "classification" else _AUTO_MAGNITUDE_SLOTS
        return self.magnitude_slots

    def _residual_slots(self) -> int:
        """``residual_slots``, with ``"auto"`` resolved: 10 for regression, none for classification."""
        if isinstance(self.residual_slots, str):
            return 0 if self._problem_type == "classification" else _AUTO_RESIDUAL_SLOTS
        return self.residual_slots

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
            warnings.warn(f"beamfeat certification: {diagnostic}", UserWarning, stacklevel=5)
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
                stacklevel=5,
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

        self._warn_constant_columns(X, feature_names)
        rows = self._split_rows(n_samples)
        nodes = self._search(X, y, rows, feature_names, units)

        self.selection_result_ = None
        self.selection_report_: list[dict] | None = None
        self.fdr_controlled_: bool | None = None
        self.fdr_scope_: str | None = None
        self.certification_result_ = None
        self.fdp_inflation_: float | None = None
        self.equations_: list[dict] | None = None
        self.input_report_: list[dict] | None = None
        self._inputs_dropped_ = 0

        candidate_nodes: list[Node] = []
        forward_path = None
        if self.selector is not None:
            nodes, candidate_nodes, forward_path = self._select(X, y, rows, feature_names, units, nodes)
        return self._finalise(X, nodes, candidate_nodes, forward_path, feature_names, units)

    def _warn_constant_columns(self, X: np.ndarray, feature_names: list[str]) -> None:
        """Warn about input columns with no variation, which cannot be selected."""
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
                stacklevel=3,
            )

    def _split_rows(self, n_samples: int) -> _Rows:
        """Split the training rows into search, selection and re-test rows.

        The selection rows are held out of the search when a selector is
        configured and ``selection_holdout`` is set; with ``parsimony_holdout``
        they are split again, the second part kept for the re-test. Sets
        :attr:`_holdout_rows_` and :attr:`_certify_rows_`.
        """
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
                    stacklevel=3,
                )
            else:
                inner = split_rng.permutation(len(holdout_index))
                certify_index = np.sort(holdout_index[inner[:n_certify]])
                holdout_index = np.sort(holdout_index[inner[n_certify:]])
        self._certify_rows_ = certify_index if use_certify else None
        return _Rows(
            search=search_index,
            selection=holdout_index,
            certify=certify_index,
            use_holdout=use_holdout,
            use_certify=use_certify,
            fell_back_to_screened=fell_back_to_screened,
        )

    def _search(
        self, X: np.ndarray, y: np.ndarray, rows: _Rows, feature_names: list[str], units
    ) -> list[Node]:
        """Run the beam search on the search rows; sets :attr:`search_result_`.

        Returns its expressions, or the input columns when it produced none.
        With ``max_search_rows``, the search runs on a sample of the search
        rows (:meth:`_search_sample`), and its expressions are then checked on
        every search row (:meth:`_check_on_search_rows`).
        """
        sample = self._search_sample(rows.search)
        search_data = {name: X[sample, index] for index, name in enumerate(feature_names)}

        report(
            self.verbose,
            1,
            f"fit: {X.shape[0]} rows x {len(feature_names)} columns"
            + (
                f", split {len(rows.search)} search / {len(rows.selection)} selection"
                + (f" (the search on {len(sample)})" if len(sample) < len(rows.search) else "")
                if rows.use_holdout
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
            magnitude_slots=self._magnitude_slots(),
            residual_slots=self._residual_slots(),
        )
        search_result = search.run(search_data, y[sample], units=units)
        self.search_result_ = search_result

        nodes = list(search_result.nodes)
        if nodes and len(sample) < len(rows.search):
            nodes = self._check_on_search_rows(nodes, X, rows.search, feature_names, units)
        if not nodes:
            nodes = Evaluator(search_data, units=units).leaf_nodes()
            logger.warning("[beamfeat] search produced no features; falling back to input columns")
        return nodes

    def _search_sample(self, search: np.ndarray) -> np.ndarray:
        """The rows the search runs on: every search row, or ``max_search_rows`` of them.

        The sample is drawn without replacement from the third stream of
        ``random_state`` (the first two are the knockoff selector's and the
        chance check's) and kept in row order.
        """
        if self.max_search_rows is None or len(search) <= self.max_search_rows:
            return search
        rng = np.random.default_rng(np.random.SeedSequence(self.random_state).spawn(3)[2])
        return np.sort(rng.choice(search, size=self.max_search_rows, replace=False))

    def _check_on_search_rows(
        self, nodes: list[Node], X: np.ndarray, search: np.ndarray, feature_names: list[str], units
    ) -> list[Node]:
        """The expressions a sampled search returned that pass its filters on every search row.

        The search excludes an expression outside an operator's domain, or
        whose denominator comes within the pole margin of zero, on the rows
        it sees; on a sample, a row it did not see can still take one there.
        Each expression is evaluated on every search row, with the same
        filters, and dropped if it fails them. Only the columns the
        expressions read are copied, and the evaluator caches few arrays.
        """
        used = set().union(*(node.columns() for node in nodes))
        data = {name: X[search, index] for index, name in enumerate(feature_names) if name in used}
        evaluator = Evaluator(data, units=units, cache_size=8)
        kept = [node for node in nodes if evaluator.evaluate(node) is not None]
        if len(kept) < len(nodes):
            report(
                self.verbose,
                1,
                f"search: {len(nodes) - len(kept)} of {len(nodes)} expressions found on the sample "
                "failed the search's filters on the other search rows and were dropped",
            )
        return kept

    def _select(
        self, X: np.ndarray, y: np.ndarray, rows: _Rows, feature_names: list[str], units, nodes: list[Node]
    ) -> tuple[list[Node], list[Node], tuple[list[int], list[float], int] | None]:
        """Screen the search's expressions on the selection rows and choose the terms to fit.

        Sets :attr:`selection_result_`, :attr:`selection_report_` and
        :attr:`fdr_controlled_`. Returns the expressions to fit, the
        candidates screened, and the forward path when one was taken.
        """
        forward_path = None
        self._alpha_stop_ = None
        selection_rows = rows.selection if rows.use_holdout else np.arange(X.shape[0])
        selection_data = {name: X[selection_rows, index] for index, name in enumerate(feature_names)}
        evaluator = Evaluator(selection_data, units=units)
        selection_target = y[rows.selection] if rows.use_holdout else y

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
            self._report_screen(selection_result, selector, candidate_nodes)
            for diagnostic in selection_result.warnings_raised:
                warnings.warn(f"beamfeat selection: {diagnostic}", UserWarning, stacklevel=3)
            screened = set(int(i) for i in selection_result.selected)

            if selection_result.n_selected > 0:
                keep_order, retest, forward_path = self._choose_terms(
                    X, y, rows, feature_names, units, matrix, selection_data, selection_target,
                    candidate_nodes, screened,
                )
                nodes = [candidate_nodes[index] for index in keep_order]
                kept = set(keep_order)
                self.fdr_controlled_ = bool(nodes)
                self._report_result(nodes, keep_order, screened, retest)
            else:
                kept = set()
                nodes = self._no_discoveries(
                    nodes, selector.target_fdr, rows.use_holdout, selection_result.warnings_raised
                )

            self.selection_report_ = self._selection_records(
                candidate_nodes, selection_result, screened, kept
            )
        else:
            # Nothing could even be evaluated on the selection rows, so no
            # null was tested; returning the search output here unannounced
            # would skip the screening the caller configured.
            nodes = self._no_discoveries(
                nodes, selector.target_fdr, rows.use_holdout,
                ["no candidate could be evaluated on the selection rows"],
            )

        if not rows.use_holdout and self.fdr_controlled_:
            # Same-data selection: the guarantee is stated for a fixed
            # candidate set, which this is not. Record honestly.
            self.fdr_controlled_ = False
            warnings.warn(
                "beamfeat: selection ran on the same rows the search used to choose "
                "candidates (selection_holdout disabled or too few rows); p-values are "
                "optimistically biased and fdr_controlled_ is therefore False",
                UserWarning,
                stacklevel=3,
            )
        return nodes, candidate_nodes, forward_path

    def _report_screen(self, selection_result, selector: Selector, candidate_nodes: list[Node]) -> None:
        """The verbose report of the screen: its outcome, and at level 2 the strongest terms."""
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

    def _choose_terms(
        self, X, y, rows: _Rows, feature_names, units, matrix, selection_data, selection_target,
        candidate_nodes: list[Node], screened: set[int],
    ) -> tuple[list[int], tuple[int, int] | None, tuple[list[int], list[float], int]]:
        """The terms to fit, from a screen that rejected at least one null.

        The screened set, less any term the input test removes; then the
        forward step's subset of it; then, with ``parsimony_holdout``, the
        terms the re-test certifies, or the screened set when it certifies
        none. Returns the kept indices, the re-test's (certified, re-tested)
        counts or ``None``, and the forward path.
        """
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
        if self.parsimony is not None and self.parsimony_alpha is None and not rows.use_certify:
            self._alpha_stop_ = alpha_stop(
                matrix, selection_target, forward_path[0], forward_path[2],
                classification=self._problem_type == "classification",
                parsimony_alpha=_HINT_ALPHA, random_state=self.random_state,
            )
        hint = (
            f"; parsimony_alpha={_HINT_ALPHA:g} would keep {self._alpha_stop_}"
            if self._alpha_stop_ is not None and self._alpha_stop_ < len(keep_order)
            else ""
        )
        report(
            self.verbose,
            1,
            f"parsimony: {len(eligible)} certified -> "
            f"{len(keep_order)} term{'' if len(keep_order) == 1 else 's'}"
            f" (|S|/|S'| = {len(eligible) / len(keep_order):.2f}){hint}"
            if self.parsimony is not None
            else f"parsimony: off, keeping all {len(eligible)} "
            f"certified term{'' if len(eligible) == 1 else 's'}",
        )
        retest = None
        if rows.fell_back_to_screened:
            keep_order = list(eligible)
        elif rows.use_certify:
            certified = self._certify_subset(
                X, y, rows.certify, feature_names, units,
                [candidate_nodes[index] for index in keep_order],
                keep_order,
            )
            retest = (len(certified), len(keep_order))
            # Nothing compact could be certified, but the screened
            # set still is; return that (less any term the input
            # test removed) rather than nothing.
            keep_order = certified if certified else list(eligible)
        return keep_order, retest, forward_path

    def _report_result(
        self, nodes: list[Node], keep_order: list[int], screened: set[int], retest: tuple[int, int] | None
    ) -> None:
        """The verbose report of the result set and what the guarantee covers."""
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

    def _selection_records(
        self, candidate_nodes: list[Node], selection_result, screened: set[int], kept: set[int]
    ) -> list[dict]:
        """The rows of :attr:`selection_report_`, one per screened candidate."""
        return [
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

    def _finalise(
        self, X: np.ndarray, nodes: list[Node], candidate_nodes: list[Node], forward_path,
        feature_names: list[str], units,
    ) -> np.ndarray:
        """Evaluate the kept expressions on the full training data and record the fit.

        Sets :attr:`features_`, :attr:`feature_formulas_`, :attr:`n_features_out_`,
        :attr:`fdr_scope_`, :attr:`equations_` and :attr:`fdp_inflation_`.
        Returns the transformed training matrix.
        """
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
        # Each feature's smallest and largest value on the training rows, which
        # predict and transform compare new rows with.
        self.feature_ranges_ = np.array(
            [[float(np.min(values)), float(np.max(values))] for values in kept_columns], dtype=np.float64
        ).reshape(-1, 2)
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
            return np.empty((X.shape[0], 0), dtype=np.float64)
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
                stacklevel=4,
            )
            return nodes
        warnings.warn(
            message + "Returning no constructed features; the model will "
            "predict the training mean (regression) or class prior "
            "(classification). Set on_no_discoveries='fallback' to keep "
            "the unfiltered search output instead.",
            NoDiscoveriesWarning,
            stacklevel=4,
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
        extrapolating = self._extrapolation_message(matrix)
        if extrapolating:
            warnings.warn(extrapolating, ExtrapolationWarning, stacklevel=4)
        return matrix

    def _extrapolation_message(self, matrix: np.ndarray) -> str | None:
        """The text of :class:`ExtrapolationWarning` for ``matrix``, or ``None`` when no feature goes that far."""
        ranges = getattr(self, "feature_ranges_", None)  # absent from a model saved before 0.7.0
        if ranges is None or not len(ranges):
            return None
        low, high = ranges[:, 0], ranges[:, 1]
        width = np.where(high > low, high - low, np.inf)
        with np.errstate(invalid="ignore"):
            beyond = np.nan_to_num(np.maximum(matrix - high, low - matrix) / width, nan=-np.inf)
        far = beyond > _EXTRAPOLATION_WIDTHS
        if not far.any():
            return None
        columns = np.flatnonzero(far.any(axis=0))
        named = ", ".join(
            f"{self.feature_formulas_[column]!r} by up to {beyond[:, column].max():.3g} widths"
            for column in columns[:3]
        )
        if len(columns) > 3:
            named += f" and {len(columns) - 3} more"
        which = "a feature lies beyond its" if len(columns) == 1 else f"{len(columns)} features lie beyond their"
        n_rows = int(np.count_nonzero(far.any(axis=1)))
        return (
            f"beamfeat: on {n_rows} of {matrix.shape[0]} rows, {which} training range by more than "
            f"{_EXTRAPOLATION_WIDTHS:g} width of that range: {named}. The predictions on those rows "
            "extrapolate the equation, where a formula can behave unlike anything the fit saw, such as a "
            "ratio whose denominator nears zero; check them."
        )

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
        sympy = _import_sympy()

        return [sympy.latex(expression) for expression in self.to_sympy()]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.non_deterministic = False
        return tags
