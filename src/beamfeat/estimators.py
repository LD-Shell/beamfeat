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
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from beamfeat._base import (
    DegenerateFitWarning,
    ExtrapolationWarning,
    NoDiscoveriesError,
    NoDiscoveriesWarning,
    _BeamFeatBase,
)
from beamfeat._compat import logistic_regression
from beamfeat._rendering import (
    _check_equation_format,
    _format_coefficient,
    _label,
    _latex_label,
    _latex_number,
    _latex_text,
    _render_linear_equation,
    _render_linear_latex,
)
from beamfeat.scoring import Scorer
from beamfeat.search import DEFAULT_BINARY, DEFAULT_FITTED, DEFAULT_UNARY, report
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
    "ExtrapolationWarning",
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
            After the forward step ``"conditional"`` needs ``parsimony_alpha``:
            without it the step hands the re-test riders it cannot certify,
            and on noisy data the screened set comes back in most fits, so
            that combination raises.
            With ``parsimony=None`` the re-test sees the whole screened set
            and needs no ``parsimony_alpha``. Regression only; ignored when
            ``parsimony_holdout`` is ``None``. See the guarantees page for its
            measured calibration.
        magnitude_slots: Beam slots added at each search depth for
            candidates ranked by how well their magnitude tracks the
            target's, for laws whose parts do not correlate with the target,
            such as ``x0*x1*x2`` with centred inputs, whose pairwise products
            the score's ranking leaves among noise. ``"auto"`` (default) adds
            10 for regression and none for classification, whose labels carry
            no magnitude; ``0`` adds none, the default before 0.7.0. On wide
            tables, where the first depth cannot propose every pair of inputs,
            the inputs are also proposed in an order that interleaves the two
            rankings. The ranking is part of the search, on the search rows
            only (see :class:`~beamfeat.search.BeamSearch`), so the screen on
            the selection rows sees fixed expressions and its guarantee is
            unchanged. In ``benchmarks/release_comparison/`` the slots
            recovered every ``x0*x1*x2`` stress law (exact recovery 0.40 and
            0.55 without them, 1.00 with), left the held-out score of 10 of 12
            real datasets within 0.005 (riboflavin fell by 0.011, eyedata rose
            by 0.009), and cost a fifth more fit time. A
            positive value is for regression only.
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
            it stopped after one to five terms on 10 of 12 datasets in
            ``benchmarks/three_settings/`` and lowered held-out score on 9.
            Ignored when ``parsimony`` is ``None``.
        residual_slots: Places of the search's final ranking kept for terms
            that explain what the leading terms leave. The ranking keeps the
            ``max_features`` candidates (``beam_width`` by default) that
            score best against the target, and a dominant term fills it with
            restatements of itself, so that a weaker term beside it is cut,
            such as ``log(x3)`` in ``x0*x1/x2 + 0.5*log(x3)``. ``"auto"``
            (default) gives 10 places for regression to the first terms of a
            greedy forward path through the search's candidates, each the
            one that adds most to a least-squares fit of those before it; a
            path term is first compared with the other transforms of its
            operand, which the search may have dropped as near copies
            (``log(x3)`` beside ``1/x3``). ``0`` gives none, the default
            before 0.8.0. The path is part of the search, on the search rows
            only (see :class:`~beamfeat.search.BeamSearch`), so the screen's
            guarantee is unchanged. A positive value is for regression only.
        max_search_rows: Most rows the search runs on, or ``None``
            (default) for every search row. The search's time and memory
            grow with its rows, while a strong term stands out on a few
            thousand. With a value, the search runs on that many search
            rows, drawn at random (``random_state``), and its expressions are
            then checked on every search row against the search's filters
            (an operator's domain, a denominator near zero). The screen still
            runs on every selection row, none of which the search sees, so
            its guarantee is unchanged. What a sample can cost is a weak
            term: ranked on fewer rows, it can be cut before the screen
            tests it.

    Attributes:
        features_: Selected :class:`~beamfeat.expression.Node` expressions.
        feature_formulas_: Their formula strings.
        feature_ranges_: Each feature's smallest and largest value on the
            training rows, one row per feature. ``transform`` and ``predict``
            raise :class:`ExtrapolationWarning` when new rows take a feature
            more than one width of this range beyond it.
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
        magnitude_slots: int | str = "auto",
        input_test: bool = False,
        parsimony_alpha: float | None = None,
        residual_slots: int | str = "auto",
        max_search_rows: int | None = None,
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
            residual_slots=residual_slots,
            max_search_rows=max_search_rows,
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
        magnitude_slots: int | str = "auto",
        input_test: bool = False,
        parsimony_alpha: float | None = None,
        residual_slots: int | str = "auto",
        max_search_rows: int | None = None,
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
            residual_slots=residual_slots,
            max_search_rows=max_search_rows,
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
        magnitude_slots: int | str = "auto",
        input_test: bool = False,
        parsimony_alpha: float | None = None,
        residual_slots: int | str = "auto",
        max_search_rows: int | None = None,
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
            residual_slots=residual_slots,
            max_search_rows=max_search_rows,
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
