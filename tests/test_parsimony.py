"""Tests for the forward step and its chance check, the parsimony_holdout re-test,
and what the guarantee covers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from beamfeat import (
    BeamFeatClassifier,
    BeamFeatRegressor,
    BeamFeatTransformer,
    NoDiscoveriesWarning,
    PermutationSelector,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260723)


@pytest.fixture
def regression_data(rng):
    """y = (a * b) / c, so the true expression needs depth 2."""
    n = 400
    X = rng.uniform(1.0, 6.0, (n, 4))
    y = (X[:, 0] * X[:, 1]) / X[:, 2] + rng.normal(0, 0.05, n)
    return X, y


@pytest.fixture
def classification_data(rng):
    n = 400
    X = rng.uniform(1.0, 6.0, (n, 4))
    score = X[:, 0] * X[:, 1]
    return X, (score > np.median(score)).astype(int)


def _fast(cls, **kwargs):
    """Estimator with cheap settings so the suite stays quick."""
    kwargs.setdefault("max_depth", 2)
    kwargs.setdefault("beam_width", 20)
    kwargs.setdefault("selector", None)
    return cls(**kwargs)


class TestParsimonyHoldout:
    """The screened set is certified; the parsimonious subset drawn from it is
    not, because it is chosen on the same rows.

    Splitting the selection rows once more restores control over the printed
    equation: the subset is fixed once the first part has been used, so
    re-testing it on the second is an ordinary fixed-candidate screen. What it
    costs is rows, which is why it is off by default.
    """

    def test_off_by_default(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        assert model.parsimony_holdout is None
        assert model.certification_result_ is None

    def test_inflation_factor_is_reported(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        report = model.selection_report_
        screened = sum(bool(row["screened"]) for row in report)
        kept = sum(bool(row["kept"]) for row in report)
        if kept:
            assert model.fdp_inflation_ == pytest.approx(screened / kept)
            assert model.fdp_inflation_ >= 1.0

    def test_second_split_certifies_the_returned_set(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", parsimony_holdout=0.5).fit(X, y)
        if model.n_features_out_:
            assert model.certification_result_ is not None
            # on these data the re-test rejects the null of every returned
            # feature, so the count cannot exceed the number it tested
            assert model.n_features_out_ <= model.certification_result_.n_candidates
            assert model.fdr_controlled_ is True

    def test_second_split_never_grows_the_returned_set(self, regression_data):
        X, y = regression_data
        plain = _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        split = _fast(BeamFeatRegressor, selector="permutation", parsimony_holdout=0.5).fit(X, y)
        assert split.n_features_out_ <= plain.n_features_out_

    def test_warns_when_rows_are_too_few_to_split(self, rng):
        X = rng.uniform(1.0, 6.0, (30, 4))
        y = (X[:, 0] * X[:, 1]) / X[:, 2] + rng.normal(0, 0.05, 30)
        with pytest.warns(UserWarning, match="cannot be split"):
            _fast(BeamFeatRegressor, selector="permutation", parsimony_holdout=0.5).fit(X, y)

    def test_falls_back_to_the_screened_set_not_a_pruned_one(self, rng):
        """A caller who asked for a certified equation and cannot have a compact
        one should get the long certified set, not the compact uncertified
        subset --- that subset is the single property they ruled out."""
        X = rng.uniform(1.0, 6.0, (30, 4))
        y = (X[:, 0] * X[:, 1]) / X[:, 2] + rng.normal(0, 0.05, 30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fallback = _fast(
                BeamFeatRegressor, selector="permutation", parsimony_holdout=0.5
            ).fit(X, y)
            unpruned = _fast(BeamFeatRegressor, selector="permutation", parsimony=None).fit(X, y)
            pruned = _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        assert fallback.n_features_out_ == unpruned.n_features_out_
        if pruned.n_features_out_ < unpruned.n_features_out_:
            assert fallback.n_features_out_ > pruned.n_features_out_

    def test_unpruned_equation_is_the_certified_set(self, regression_data):
        """`parsimony=None` gives a printed equation that carries the
        guarantee, at no cost in rows. It is the free corner of the trade-off
        and the fallback the split degrades to."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", parsimony=None).fit(X, y)
        report = model.selection_report_
        screened = sum(bool(row["screened"]) for row in report)
        assert model.n_features_out_ == screened
        assert model.fdp_inflation_ == pytest.approx(1.0)

    def test_predictions_still_work_after_the_second_split(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", parsimony_holdout=0.5).fit(X, y)
        assert model.predict(X).shape == y.shape


class TestCertifyTest:
    """``certify_test`` sets the test the ``parsimony_holdout`` re-test runs.

    ``"marginal"``, the default, re-runs the screen, whose null is
    independence from the target, so it rejects the null of a term that only
    rides on a true one. ``"conditional"`` tests each term given the others,
    fails to reject that term's null, and drops it.
    """

    @staticmethod
    def _rider_data(seed=0, n=600):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.5, 3.0, (n, 8))
        return X, 2 * X[:, 0] + rng.normal(0, 0.5, n)

    @staticmethod
    def _products(**kwargs):
        """Depth one with products only: the screened set is x0 and its products."""
        return BeamFeatRegressor(
            max_depth=1, binary_ops=("mul",), unary_ops=(), parsimony=None,
            parsimony_holdout=0.5, random_state=0, **kwargs,
        )

    def test_conditional_drops_x0_x7_and_keeps_x0(self):
        X, y = self._rider_data()
        marginal = self._products().fit(X, y)
        conditional = self._products(certify_test="conditional").fit(X, y)
        assert "x0" in marginal.formulas() and "(x0 * x7)" in marginal.formulas()
        assert conditional.formulas() == ["x0"]
        assert conditional.certification_result_.method == "conditional_permutation"
        assert conditional.fdr_scope_ == "printed equation"

    def test_conditional_keeps_true_terms(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 3.0, (600, 5))
        y = X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(0, 0.5, 600)
        fits = {
            test: BeamFeatTransformer(
                max_depth=1, parsimony_holdout=0.5, certify_test=test, random_state=0
            ).fit(X, y)
            for test in ("marginal", "conditional")
        }
        conditional = fits["conditional"]
        columns = [node.columns() for node in conditional.features_]
        assert {"x0"} in columns
        product = conditional.transform(X)[:, columns.index({"x1", "x2"})]
        assert abs(np.corrcoef(product, X[:, 1] * X[:, 2])[0, 1]) > 0.999
        assert sorted(map(sorted, columns)) == [["x0"], ["x1", "x2"]]
        assert fits["marginal"].n_features_out_ > conditional.n_features_out_

    def test_conditional_keeps_the_planted_formula(self, regression_data):
        X, y = regression_data
        model = BeamFeatTransformer(
            max_depth=2, beam_width=20, parsimony_holdout=0.5, certify_test="conditional",
            random_state=0,
        ).fit(X, y)
        assert model.certification_result_.n_selected == model.n_features_out_ >= 1
        planted = (X[:, 0] * X[:, 1]) / X[:, 2]
        assert any(abs(np.corrcoef(column, planted)[0, 1]) > 0.999 for column in model.transform(X).T)

    def test_marginal_is_the_default_and_unchanged(self, regression_data):
        X, y = regression_data
        assert BeamFeatRegressor().get_params()["certify_test"] == "marginal"
        implicit = _fast(BeamFeatRegressor, selector="permutation", parsimony_holdout=0.5).fit(X, y)
        explicit = _fast(
            BeamFeatRegressor, selector="permutation", parsimony_holdout=0.5, certify_test="marginal"
        ).fit(X, y)
        assert implicit.certification_result_.method == "permutation"
        assert explicit.formulas() == implicit.formulas()
        np.testing.assert_array_equal(
            explicit.certification_result_.p_values, implicit.certification_result_.p_values
        )
        np.testing.assert_array_equal(explicit.predict(X), implicit.predict(X))

    def test_ignored_without_parsimony_holdout(self, regression_data):
        X, y = regression_data
        plain = _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        conditional = _fast(BeamFeatRegressor, selector="permutation", certify_test="conditional").fit(X, y)
        assert conditional.certification_result_ is None
        assert conditional.formulas() == plain.formulas()
        np.testing.assert_array_equal(conditional.predict(X), plain.predict(X))

    def test_runs_at_the_screens_level_and_correction(self, regression_data):
        from beamfeat.selection import _adjusted_p_values

        X, y = regression_data
        selector = PermutationSelector(target_fdr=0.2, correction="bh", random_state=3)
        with warnings.catch_warnings():
            # the instance's settings override the estimator's, which it says
            warnings.simplefilter("ignore", UserWarning)
            model = _fast(
                BeamFeatRegressor, selector=selector, parsimony_holdout=0.5, certify_test="conditional"
            ).fit(X, y)
        result = model.certification_result_
        assert result.target_fdr == 0.2
        np.testing.assert_allclose(result.q_values, _adjusted_p_values(result.p_values, "bh"))

    def test_nothing_testable_returns_the_screened_set_and_says_what_it_carries(self):
        """x0, x1, x0 + x1 and x0 - x1 are each a combination of the others,
        so none can be tested given the rest."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 3.0, (400, 2))
        y = X[:, 0] + rng.normal(0, 0.3, 400)
        with (
            pytest.warns(UserWarning, match="could not be tested"),
            pytest.warns(NoDiscoveriesWarning, match="marginal guarantee"),
        ):
            model = BeamFeatRegressor(
                max_depth=1, binary_ops=("add", "sub"), unary_ops=(), parsimony=None,
                parsimony_holdout=0.5, certify_test="conditional", random_state=0,
            ).fit(X, y)
        assert model.certification_result_.n_selected == 0
        assert model.n_features_out_ == model.selection_result_.n_selected

    def test_verbose_says_what_the_re_test_rejected(self, capsys):
        X, y = self._rider_data()
        self._products(certify_test="conditional", verbose=1).fit(X, y)
        out = capsys.readouterr().out
        assert "re-test on held-back rows rejected the null for 1 of 8 re-tested terms" in out
        assert "each given the others" in out

        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 3.0, (400, 2))
        y = X[:, 0] + rng.normal(0, 0.3, 400)
        with warnings.catch_warnings():
            # the fallback's warnings are the previous test's subject
            warnings.simplefilter("ignore")
            BeamFeatRegressor(
                max_depth=1, binary_ops=("add", "sub"), unary_ops=(), parsimony=None,
                parsimony_holdout=0.5, certify_test="conditional", random_state=0, verbose=1,
            ).fit(X, y)
        out = capsys.readouterr().out
        assert "the re-test on held-back rows rejected no null" in out
        assert "the screen's guarantee covers the screened set" in out

    @pytest.mark.parametrize("cls", [BeamFeatClassifier, BeamFeatTransformer])
    def test_classification_is_refused(self, cls, classification_data):
        X, y = classification_data
        kwargs = {"problem_type": "classification"} if cls is BeamFeatTransformer else {}
        with pytest.raises(ValueError, match="regression only"):
            _fast(cls, selector="permutation", parsimony_holdout=0.5, certify_test="conditional",
                  **kwargs).fit(X, y)


class TestParsimonyAlpha:
    """The forward step's chance check: on a noisy target the best of many
    remaining terms clears ``parsimony_tol`` by fitting noise, and
    ``parsimony_alpha`` stops the step there."""

    @staticmethod
    def _noisy(seed, n=600, k=10):
        rng = np.random.default_rng(seed)
        X = rng.uniform(1.0, 3.0, (n, 3 + k))
        signal = X[:, 0] * X[:, 1] / X[:, 2]
        return X, signal + signal.std() * rng.standard_normal(n)

    def test_p_value_is_calibrated_under_the_null(self):
        """Columns that add nothing beyond the chosen one, five of them
        correlated with it: the check rejects at about its level."""
        from beamfeat._forward import _chance_p_value

        rejected = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            matrix = rng.standard_normal((150, 12))
            matrix[:, 1:6] += matrix[:, [0]]
            target = 2.0 * matrix[:, 0] + rng.standard_normal(150)
            rejected.append(_chance_p_value(matrix, target, [0], list(range(1, 12)), rng) <= 0.1)
        assert 0.05 <= np.mean(rejected) <= 0.16

    def test_p_value_rejects_a_remaining_term(self):
        from beamfeat._forward import _CHANCE_DRAWS, _chance_p_value

        rng = np.random.default_rng(1)
        matrix = rng.standard_normal((150, 12))
        target = 2.0 * matrix[:, 0] + 0.5 * matrix[:, 7] + rng.standard_normal(150)
        p_value = _chance_p_value(matrix, target, [0], list(range(1, 12)), rng)
        assert p_value == pytest.approx(1 / (1 + _CHANCE_DRAWS))

    def test_default_leaves_the_fit_unchanged(self, regression_data):
        X, y = regression_data
        plain = _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        explicit = _fast(BeamFeatRegressor, selector="permutation", parsimony_alpha=None).fit(X, y)
        assert explicit.feature_formulas_ == plain.feature_formulas_
        np.testing.assert_array_equal(explicit.coef_, plain.coef_)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_stops_riders_on_a_noisy_target(self, seed):
        """Same path, an earlier stop, and the text names the check."""
        X, y = self._noisy(seed)
        # The search without magnitude slots, whose path keeps riders here, so
        # the test is of the forward step alone.
        plain = _fast(BeamFeatRegressor, selector="permutation", magnitude_slots=0).fit(X, y)
        checked = _fast(
            BeamFeatRegressor, selector="permutation", magnitude_slots=0, parsimony_alpha=0.05
        ).fit(X, y)
        assert len(plain.feature_formulas_) > 3
        assert len(checked.feature_formulas_) < len(plain.feature_formulas_)
        assert [r["formulas"] for r in checked.equations_] == [r["formulas"] for r in plain.equations_]
        assert "parsimony_alpha=0.05" in checked.equations()
        assert "parsimony_alpha" not in plain.equations()

    def test_seeded(self):
        X, y = self._noisy(0)
        first = _fast(BeamFeatRegressor, selector="permutation", parsimony_alpha=0.5).fit(X, y)
        second = _fast(BeamFeatRegressor, selector="permutation", parsimony_alpha=0.5).fit(X, y)
        assert first.feature_formulas_ == second.feature_formulas_


class TestInflationOfTheReturnedSet:
    """``fdp_inflation_`` is |S|/|S'| for the set the model actually returns,
    measured after the parsimony re-test or its fallback."""

    def test_fallback_to_the_screened_set_has_no_inflation(self, rng):
        X = rng.uniform(1.0, 6.0, (38, 4))
        y = (X[:, 0] * X[:, 1]) / X[:, 2] + rng.normal(0, 0.05, 38)
        with pytest.warns(UserWarning, match="cannot be split"):
            model = _fast(BeamFeatRegressor, selector="permutation", parsimony_holdout=0.5).fit(X, y)
        assert model.n_features_out_ == model.selection_result_.n_selected
        assert model.fdp_inflation_ == pytest.approx(1.0)

    def test_re_certified_set_has_no_inflation(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", parsimony_holdout=0.5).fit(X, y)
        assert model.certification_result_.n_selected == model.n_features_out_
        assert model.fdp_inflation_ == pytest.approx(1.0)

    def test_forward_subset_reports_its_ratio(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        assert model.fdp_inflation_ == pytest.approx(
            model.selection_result_.n_selected / model.n_features_out_
        )


class TestFdrScope:
    """``fdr_controlled_`` says whether there is a guarantee; ``fdr_scope_``
    says what it is over.

    The flag is ``True`` whenever the returned features came from the screened
    set, which is not the same as the guarantee covering those features. A
    caller branching on the flag in a script never sees the note
    ``equation()`` prints, so the distinction is an attribute too."""

    def test_screened_set_when_terms_were_dropped(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        if model.n_features_out_ >= model.selection_result_.n_selected:
            pytest.skip("this fit pruned nothing, so the scope is the printed set")
        assert model.fdr_controlled_ is True
        assert model.fdr_scope_ == "screened set"

    def test_printed_equation_when_parsimony_is_off(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation",
                      beam_width=25, parsimony=None).fit(X, y)
        assert model.fdr_controlled_ is True
        assert model.fdr_scope_ == "printed equation"

    def test_printed_equation_when_re_certified(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation",
                      parsimony_holdout=0.5).fit(X, y)
        if model.certification_result_ is None:
            pytest.skip("the rows did not support a second split")
        assert model.fdr_scope_ == "printed equation"

    def test_none_without_a_selector(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector=None).fit(X, y)
        assert model.fdr_controlled_ is None
        assert model.fdr_scope_ is None

    def test_none_when_no_null_is_rejected(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, rng.standard_normal(400))
        assert model.fdr_controlled_ is False
        assert model.fdr_scope_ is None

    def test_transformer_reports_it_too(self, regression_data):
        """The transformer has no equation() to carry the note, so the
        attribute is the only signal it can give."""
        X, y = regression_data
        model = _fast(BeamFeatTransformer, selector="permutation", beam_width=25).fit(X, y)
        assert model.fdr_scope_ in ("screened set", "printed equation")

    def test_scope_and_note_agree(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        marked = "certified terms" in model.equation()
        assert marked == (model.fdr_scope_ == "screened set")
