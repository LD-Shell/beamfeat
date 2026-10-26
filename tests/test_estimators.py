"""Tests for the scikit-learn compatible estimators."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from beamfeat import (
    BeamFeatClassifier,
    BeamFeatRegressor,
    BeamFeatTransformer,
    DegenerateFitWarning,
    ExtrapolationWarning,
    NoDiscoveriesWarning,
    PermutationSelector,
)

ESTIMATORS = [BeamFeatTransformer, BeamFeatRegressor, BeamFeatClassifier]


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


# --------------------------------------------------------------------------- #
# scikit-learn API compliance
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize("cls", ESTIMATORS)
def test_check_estimator(cls):
    """The full scikit-learn conformance suite."""
    check_estimator(cls(max_depth=1, beam_width=6, selector=None))


@pytest.mark.slow
@pytest.mark.parametrize("cls", ESTIMATORS)
def test_check_estimator_with_selection(cls):
    """The same suite with FDR selection on, so the empty-selection paths run too."""
    check_estimator(cls(max_depth=1, beam_width=6))


@pytest.mark.parametrize("cls", ESTIMATORS)
class TestApiConventions:
    def test_get_params_roundtrip(self, cls):
        estimator = cls(max_depth=3, beam_width=17)
        params = estimator.get_params()
        assert params["max_depth"] == 3
        assert params["beam_width"] == 17

    def test_set_params(self, cls):
        estimator = cls().set_params(max_depth=4)
        assert estimator.max_depth == 4

    def test_n_jobs_is_a_parameter(self, cls):
        assert cls().get_params()["n_jobs"] == 1
        assert clone(cls(n_jobs=-1)).n_jobs == -1

    def test_max_inputs_is_a_parameter(self, cls):
        assert cls().get_params()["max_inputs"] is None
        assert clone(cls(max_inputs=2)).max_inputs == 2
        assert cls(max_inputs=0).max_inputs == 0

    def test_magnitude_slots_is_a_parameter(self, cls):
        assert cls().get_params()["magnitude_slots"] == "auto"
        assert clone(cls(magnitude_slots=5)).magnitude_slots == 5

    def test_residual_slots_is_a_parameter(self, cls):
        assert cls().get_params()["residual_slots"] == 0
        assert clone(cls(residual_slots=5)).residual_slots == 5

    def test_max_search_rows_is_a_parameter(self, cls):
        assert cls().get_params()["max_search_rows"] is None
        assert clone(cls(max_search_rows=500)).max_search_rows == 500

    def test_clone(self, cls):
        estimator = cls(max_depth=3, beam_width=17)
        cloned = clone(estimator)
        assert cloned.get_params() == estimator.get_params()

    def test_repr(self, cls):
        assert cls.__name__ in repr(cls())

    def test_init_does_not_validate(self, cls):
        """scikit-learn requires __init__ to only store parameters."""
        estimator = cls(max_depth=-5, beam_width=-1)
        assert estimator.max_depth == -5

    def test_unfitted_raises(self, cls, regression_data):
        from sklearn.exceptions import NotFittedError

        X, _ = regression_data
        estimator = cls()
        method = estimator.transform if cls is BeamFeatTransformer else estimator.predict
        with pytest.raises(NotFittedError):
            method(X)


# --------------------------------------------------------------------------- #
# Regressor
# --------------------------------------------------------------------------- #


class TestRegressor:
    def test_recovers_generating_formula(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        assert any(
            all(token in formula for token in ("x0", "x1", "x2")) for formula in model.formulas()
        ), f"true formula not among {model.formulas()[:5]}"

    def test_fits_well_on_recoverable_signal(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_beats_linear_model_on_nonlinear_signal(self, regression_data):
        """The reason to construct features at all."""
        from sklearn.linear_model import Ridge

        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        beamfeat = _fast(BeamFeatRegressor).fit(X_train, y_train).score(X_test, y_test)
        baseline = Ridge().fit(X_train, y_train).score(X_test, y_test)
        assert beamfeat > baseline

    def test_generalises_to_held_out_data(self, regression_data):
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        model = _fast(BeamFeatRegressor).fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.9

    def test_equation_is_readable(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        equation = model.equation()
        assert equation.startswith("y = ")
        assert any(formula in equation for formula in model.formulas())

    def test_equation_respects_max_terms(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, beam_width=30).fit(X, y)
        full = model.equation()
        short = model.equation(max_terms=2)

        # Count coefficient markers ("<number>*") rather than testing whether
        # each formula appears: formulas can be substrings of one another, so
        # substring matching over-counts.
        def n_terms(equation: str) -> int:
            return len(re.findall(r"\d\*", equation))

        assert n_terms(short) <= 2
        assert n_terms(short) < n_terms(full)

    def test_max_terms_holds_omitted_terms_at_their_mean(self, regression_data):
        """The shortened equation's constant absorbs the omitted terms' mean
        contribution, so on the training rows it agrees with the model on
        average instead of being off by the omitted terms' whole mean."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, beam_width=30).fit(X, y)
        assert model.n_features_out_ > 1
        short = model.equation(precision=12, max_terms=1).split("   [")[0]
        sign, constant = re.search(r"([+-]) ([0-9.e+-]+)$", short).groups()
        constant = float(constant) * (-1 if sign == "-" else 1)

        top = int(np.argmax(np.abs(model.coef_)))
        scale = np.where(model.scaler_.scale_ == 0, 1.0, model.scaler_.scale_)
        raw = model.coef_[top] / scale[top]
        features = model._apply(X)
        shortened = raw * features[:, top] + constant
        assert np.mean(shortened) == pytest.approx(np.mean(model.predict(X)), rel=1e-9)

    def test_equation_reproduces_predictions(self, rng):
        """The printed equation must be the model, not a decoration.

        Coefficients are de-standardised for display, so this checks that
        evaluating the printed form by hand matches ``predict``.
        """
        n = 300
        X = rng.uniform(1.0, 5.0, (n, 2))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)
        model = BeamFeatRegressor(max_depth=1, beam_width=8, selector=None, include_originals=False)
        model.fit(X, y)

        from beamfeat.expression import Evaluator

        evaluator = Evaluator({f"x{i}": X[:, i] for i in range(X.shape[1])})
        matrix = np.column_stack([evaluator.transform_values(node) for node in model.features_])

        scale = np.where(model.scaler_.scale_ == 0, 1.0, model.scaler_.scale_)
        raw_coefficients = model.coef_ / scale
        offset = model.intercept_ - np.sum(raw_coefficients * model.scaler_.mean_)
        manual = matrix @ raw_coefficients + offset

        np.testing.assert_allclose(manual, model.predict(X), rtol=1e-8, atol=1e-8)

    def test_to_sympy(self, regression_data):
        sympy = pytest.importorskip("sympy")
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        expressions = model.to_sympy()
        assert len(expressions) == len(model.formulas())
        assert all(isinstance(expression, sympy.Basic) for expression in expressions)

    def test_deterministic(self, regression_data):
        X, y = regression_data
        first = _fast(BeamFeatRegressor).fit(X, y)
        second = _fast(BeamFeatRegressor).fit(X, y)
        assert first.formulas() == second.formulas()
        np.testing.assert_allclose(first.predict(X), second.predict(X))


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #


class TestClassifier:
    def test_fits_and_predicts(self, classification_data):
        X, y = classification_data
        model = _fast(BeamFeatClassifier).fit(X, y)
        assert model.score(X, y) > 0.85

    def test_refit_does_not_keep_the_prior_of_an_empty_fit(self, classification_data):
        X, y = classification_data
        # Twenty permutations cannot reach the p-values this level needs, so
        # the first fit keeps nothing.
        unreachable = PermutationSelector(target_fdr=1e-6, n_permutations=20, max_permutations=20)
        model = _fast(BeamFeatClassifier, selector=unreachable)
        with pytest.warns(NoDiscoveriesWarning):
            model.fit(X, y)
        assert hasattr(model, "class_prior_")
        model.set_params(selector="permutation").fit(X, y)
        assert model.model_ is not None
        assert not hasattr(model, "class_prior_")

    def test_predict_proba_is_valid(self, classification_data):
        X, y = classification_data
        probabilities = _fast(BeamFeatClassifier).fit(X, y).predict_proba(X)
        assert probabilities.shape == (len(y), 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_decision_function(self, classification_data):
        X, y = classification_data
        assert _fast(BeamFeatClassifier).fit(X, y).decision_function(X).shape == (len(y),)

    @pytest.mark.parametrize("n_classes", [2, 3])
    def test_decision_function_without_discoveries(self, n_classes):
        """With nothing selected the scores are the log priors and agree with
        predict and predict_proba."""
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.integers(0, n_classes, 400)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatClassifier(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, y)
        scores = model.decision_function(X)
        log_prior = np.log(model.class_prior_)
        if n_classes == 2:
            assert scores.shape == (len(y),)
            np.testing.assert_allclose(scores, log_prior[1] - log_prior[0])
            predicted = model.classes_[(scores > 0).astype(int)]
        else:
            assert scores.shape == (len(y), n_classes)
            np.testing.assert_allclose(scores, np.tile(log_prior, (len(y), 1)))
            predicted = model.classes_[np.argmax(scores, axis=1)]
            softmax = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
            np.testing.assert_allclose(softmax, model.predict_proba(X))
        np.testing.assert_array_equal(predicted, model.predict(X))

    def test_string_labels(self, rng):
        """Labels are encoded internally; string classes must round-trip."""
        n = 300
        X = rng.uniform(1.0, 5.0, (n, 3))
        score = X[:, 0] * X[:, 1]
        y = np.where(score > np.median(score), "high", "low")
        model = _fast(BeamFeatClassifier).fit(X, y)
        assert set(model.classes_) == {"high", "low"}
        assert set(np.unique(model.predict(X))) <= {"high", "low"}

    def test_multiclass(self, rng):
        n = 400
        X = rng.uniform(1.0, 5.0, (n, 3))
        score = X[:, 0] * X[:, 1]
        y = np.digitize(score, np.quantile(score, [0.33, 0.66]))
        model = _fast(BeamFeatClassifier).fit(X, y)
        assert len(model.classes_) == 3
        assert model.score(X, y) > 0.6

    def test_recovers_discriminative_formula(self, classification_data):
        X, y = classification_data
        model = _fast(BeamFeatClassifier).fit(X, y)
        assert any("x0" in formula and "x1" in formula for formula in model.formulas())


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #


class TestTransformer:
    def test_output_shape(self, regression_data):
        X, y = regression_data
        transformer = _fast(BeamFeatTransformer).fit(X, y)
        assert transformer.transform(X).shape == (len(y), transformer.n_features_out_)

    def test_feature_names_out(self, regression_data):
        X, y = regression_data
        transformer = _fast(BeamFeatTransformer).fit(X, y)
        names = transformer.get_feature_names_out()
        assert len(names) == transformer.n_features_out_
        assert list(names) == transformer.formulas()

    def test_requires_y(self, regression_data):
        X, _ = regression_data
        with pytest.raises(ValueError, match="requires y"):
            BeamFeatTransformer().fit(X)

    def test_works_in_pipeline(self, regression_data):
        from sklearn.linear_model import Ridge

        X, y = regression_data
        pipeline = Pipeline(
            [("features", _fast(BeamFeatTransformer)), ("model", Ridge())]
        )
        assert pipeline.fit(X, y).score(X, y) > 0.9

    def test_cross_validates_without_leakage(self, regression_data):
        """Feature construction must happen inside each fold.

        If it leaked, cross-validated scores would be implausibly close to the
        in-sample score.
        """
        from sklearn.linear_model import Ridge

        X, y = regression_data
        pipeline = Pipeline([("features", _fast(BeamFeatTransformer)), ("model", Ridge())])
        scores = cross_val_score(pipeline, X, y, cv=3)
        assert np.all(scores > 0.8)
        assert np.all(scores <= 1.0)

    def test_classification_problem_type(self, classification_data):
        X, y = classification_data
        transformer = _fast(BeamFeatTransformer, problem_type="classification").fit(X, y)
        assert transformer.n_features_out_ > 0

    def test_parsimony_accepts_string_labels(self, rng):
        """The transformer passes its labels through unencoded, so the forward
        step must encode them before casting to float."""
        X = rng.uniform(1.0, 5.0, (200, 3))
        labels = np.where(X[:, 0] * X[:, 1] > 9.0, "high", "low")
        transformer = BeamFeatTransformer(problem_type="classification")
        transformer._problem_type = "classification"
        matrix = np.column_stack([X[:, 0] * X[:, 1], X[:, 0], X[:, 2]])
        kept = transformer._parsimonious_subset(matrix, labels, [0, 1, 2])
        assert 0 in kept
        encoded = (labels == "low").astype(int)
        assert kept == transformer._parsimonious_subset(matrix, encoded, [0, 1, 2])

    def test_string_labels_at_default_parsimony(self, rng):
        X = rng.uniform(1.0, 5.0, (300, 3))
        labels = np.where(X[:, 0] * X[:, 1] > 9.0, "high", "low")
        transformer = _fast(
            BeamFeatTransformer, problem_type="classification", selector="permutation"
        ).fit(X, labels)
        assert transformer.n_features_out_ > 0
        assert transformer.transform(X).shape == (300, transformer.n_features_out_)


# --------------------------------------------------------------------------- #
# Transform-time behaviour
# --------------------------------------------------------------------------- #


class TestTransformSemantics:
    def test_prediction_invariant_to_row_subset(self, rng):
        """Search-time filters must not apply at transform time.

        A subset in which some column happens to be constant would otherwise
        silently zero out a feature and change the prediction.
        """
        n = 200
        constant_prefix = np.r_[np.ones(30), np.arange(n - 30) + 2.0]
        X = np.column_stack([np.arange(float(n)), constant_prefix, np.arange(float(n)) * 2])
        y = X[:, 0] + rng.normal(0, 0.1, n)

        model = _fast(BeamFeatRegressor, max_depth=1, beam_width=8).fit(X, y)
        np.testing.assert_allclose(model.predict(X)[:30], model.predict(X[:30]))

    def test_transform_does_not_mutate_estimator(self, regression_data):
        """scikit-learn forbids state changes during transform or predict."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        before = dict(model.__dict__)
        model.predict(X)
        assert set(model.__dict__) == set(before)

    def test_out_of_domain_data_is_handled(self, rng):
        """Training data admits log; half the test rows do not. Those rows
        have no prediction, and the others keep theirs."""
        n = 300
        X_train = rng.uniform(1.0, 5.0, (n, 2))
        y_train = np.log(X_train[:, 0]) + rng.normal(0, 0.05, n)
        # Without fitted constants, which would give log(x0 + a) instead.
        model = _fast(BeamFeatRegressor, max_depth=1, beam_width=10, fitted_ops=()).fit(X_train, y_train)
        assert any("log(x0)" in formula for formula in model.formulas())

        X_test = rng.uniform(1.0, 5.0, (20, 2))
        X_test[::2, 0] = -5.0
        with pytest.warns(UserWarning, match=r"log\(x0\).* 10 of 20 rows"):
            predictions = model.predict(X_test)
        assert predictions.shape == (20,)
        assert np.all(np.isnan(predictions[::2]))
        np.testing.assert_allclose(predictions[1::2], model.predict(X_test[1::2]), rtol=1e-12)

    @staticmethod
    def _log_fit():
        rng = np.random.default_rng(0)
        X = rng.uniform(1.0, 3.0, (400, 2))
        y = 2 * np.log(X[:, 0]) + X[:, 1] + rng.normal(0, 0.01, 400)
        return X, y

    def test_valid_row_does_not_depend_on_the_batch(self):
        """An out-of-domain row used to zero a whole feature column, which moved
        the prediction of every other row in the batch (2.88 alone, -6.6 next
        to the bad row)."""
        X, y = self._log_fit()
        model = BeamFeatRegressor(random_state=0).fit(X, y)
        alone = model.predict(np.array([[2.0, 1.5]]))
        with pytest.warns(UserWarning, match="1 of 2 rows"):
            batch = model.predict(np.array([[2.0, 1.5], [-3.0, 1.5]]))
        np.testing.assert_allclose(batch[0], alone[0], rtol=1e-12)
        assert np.isnan(batch[1])

    def test_score_refuses_rows_without_a_prediction(self):
        X, y = self._log_fit()
        model = BeamFeatRegressor(random_state=0).fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="outside the domain"):
                model.score(np.array([[2.0, 1.5], [-3.0, 1.5]]), np.array([2.9, 0.0]))

    def test_transformer_marks_only_the_failing_cells(self):
        X, y = self._log_fit()
        model = BeamFeatTransformer(random_state=0).fit(X, y)
        good = np.array([[2.0, 1.5], [1.5, 2.5]])
        batch = np.vstack([good, [[-3.0, 1.5]]])
        with pytest.warns(UserWarning, match="are NaN in the output"):
            output = model.transform(batch)
        np.testing.assert_allclose(output[:2], model.transform(good), rtol=1e-12)
        assert np.isnan(output[2]).any()
        # the bad row is NaN only in the columns that fail on it alone, and
        # keeps its value in the others
        with pytest.warns(UserWarning):
            alone = model.transform(batch[2:3])[0]
        np.testing.assert_allclose(output[2], alone, rtol=1e-12)

    def test_classifier_rows_outside_the_domain(self):
        X, y = self._log_fit()
        labels = (y > np.median(y)).astype(int)
        model = BeamFeatClassifier(random_state=0).fit(X, labels)
        good = np.array([[2.0, 1.5], [2.8, 2.9]])
        batch = np.vstack([good, [[-3.0, 0.0]]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            undefined = np.isnan(model._apply(batch[2:3])).any()
        assert undefined
        majority = np.bincount(labels).argmax()
        with pytest.warns(UserWarning, match="most frequent training class"):
            predicted = model.predict(batch)
        np.testing.assert_array_equal(predicted[:2], model.predict(good))
        assert predicted[2] == majority
        with pytest.warns(UserWarning, match="NaN probabilities"):
            probabilities = model.predict_proba(batch)
        np.testing.assert_allclose(probabilities[:2], model.predict_proba(good), rtol=1e-12)
        assert np.all(np.isnan(probabilities[2]))
        with pytest.warns(UserWarning, match="NaN scores"):
            scores = model.decision_function(batch)
        np.testing.assert_allclose(scores[:2], model.decision_function(good), rtol=1e-12)
        assert np.isnan(scores[2])

    def test_underflow_is_isolated_to_its_row(self):
        """Underflow fails an evaluation but leaves finite values behind, so it
        is found by splitting the rows rather than by the non-finite mask."""
        from beamfeat import leaf, transform
        from beamfeat._prediction import _evaluate_by_row

        node = transform("exp", leaf("x0"))
        column = np.array([1.0, -800.0, 2.0, 0.5])
        values = _evaluate_by_row(node, {"x0": column}, None, 4)
        assert np.isnan(values[1])
        np.testing.assert_allclose(values[[0, 2, 3]], np.exp(column[[0, 2, 3]]))

    def test_single_sample_message_follows_convention(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor)
        with pytest.raises(ValueError, match="sample"):
            model.fit(X[:1], y[:1])


# --------------------------------------------------------------------------- #
# Selection integration
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


class TestExtrapolation:
    """New rows that take a feature far beyond its training range raise
    ExtrapolationWarning, naming the feature; rows inside it do not."""

    @staticmethod
    def _ratio_fit(cls=BeamFeatRegressor):
        rng = np.random.default_rng(0)
        X = rng.uniform(1.0, 3.0, (400, 3))
        y = X[:, 0] / X[:, 1] + rng.normal(0, 0.01, 400)
        model = cls(max_depth=1, beam_width=8, random_state=0).fit(X, y)
        assert model.formulas() == ["(x0 / x1)"]
        return X, model

    # x0/x1 is 20 on the second row, against a training range of about 0.36 to 2.74.
    FAR = np.array([[2.0, 2.0, 2.0], [2.0, 0.1, 2.0]])

    def test_ranges_are_those_of_the_training_rows(self):
        X, model = self._ratio_fit()
        values = X[:, 0] / X[:, 1]
        np.testing.assert_allclose(model.feature_ranges_, [[values.min(), values.max()]])

    def test_rows_inside_the_range_do_not_warn(self):
        X, model = self._ratio_fit()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExtrapolationWarning)
            model.predict(X)
            model.predict(self.FAR[:1])

    def test_rows_far_beyond_warn_and_name_the_feature(self):
        _, model = self._ratio_fit()
        low, high = model.feature_ranges_[0]
        widths = f"{(20.0 - high) / (high - low):.3g}"
        expected = rf"on 1 of 2 rows, a feature lies .*'\(x0 / x1\)' by up to {re.escape(widths)} widths"
        with pytest.warns(ExtrapolationWarning, match=expected):
            predictions = model.predict(self.FAR)
        # The warning changes nothing that is predicted.
        assert np.all(np.isfinite(predictions))

    def test_transform_warns_too(self):
        _, model = self._ratio_fit(BeamFeatTransformer)
        with pytest.warns(ExtrapolationWarning, match="on 1 of 2 rows"):
            model.transform(self.FAR)

    def test_a_model_saved_without_ranges_does_not_warn(self):
        """A model fitted before 0.7.0 has no feature_ranges_ and predicts as it did."""
        _, model = self._ratio_fit()
        del model.feature_ranges_
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExtrapolationWarning)
            model.predict(self.FAR)

    def test_a_model_without_features_does_not_warn(self, rng):
        X = rng.uniform(1.0, 3.0, (300, 3))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(random_state=0).fit(X, rng.standard_normal(300))
        assert model.feature_ranges_.shape == (0, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExtrapolationWarning)
            model.predict(X * 100)


class TestUnitsIntegration:
    def test_units_constrain_constructed_features(self, rng):
        pint = pytest.importorskip("pint")
        ureg = pint.UnitRegistry()

        n = 300
        X = rng.uniform(1.0, 5.0, (n, 2))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)
        units = {"x0": 1.0 * ureg.kilogram, "x1": 1.0 * ureg.meter}

        # A fitted shift, 1/(x0 + a), adds a constant in x0's own unit and
        # would print a "+"; the check is for sums of different columns.
        model = _fast(BeamFeatRegressor, units=units, fitted_ops=()).fit(X, y)
        formulas = model.formulas()
        assert not any("+" in formula or "-" in formula for formula in formulas)
        assert any("*" in formula for formula in formulas)
        fitted = _fast(BeamFeatRegressor, units=units).fit(X, y)
        assert not any("x0 + x1" in f or "x0 - x1" in f or "x1 - x0" in f for f in fitted.formulas())

    def test_units_are_optional(self, regression_data):
        X, y = regression_data
        assert _fast(BeamFeatRegressor, units=None).fit(X, y).n_features_out_ > 0

    def test_partial_coverage_warns(self, regression_data):
        """Labelling some columns and leaving the rest blank looks like the
        careful thing to do, and is the case where the gate quietly stops
        binding: an unlabelled column is dimensionally free and combines
        with anything."""
        pytest.importorskip("pint")
        X, y = regression_data
        with pytest.warns(UserWarning, match="units cover 2 of 4 columns"):
            _fast(BeamFeatRegressor, units={"x0": "kilogram", "x1": "meter"}).fit(X, y)

    def test_full_coverage_is_silent(self, regression_data):
        pytest.importorskip("pint")
        X, y = regression_data
        units = {f"x{index}": "dimensionless" for index in range(X.shape[1])}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _fast(BeamFeatRegressor, units=units).fit(X, y)

    def test_coverage_warning_does_not_repeat_on_predict(self, regression_data):
        pytest.importorskip("pint")
        X, y = regression_data
        model = _fast(BeamFeatRegressor, units={"x0": "kilogram"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y)
            model.predict(X)
            model.predict(X)
        coverage = [entry for entry in caught if "units cover" in str(entry.message)]
        assert len(coverage) == 1


class TestFeatureNames:
    def test_dataframe_columns_reach_the_equation(self, regression_data):
        pd = pytest.importorskip("pandas")
        X, y = regression_data
        frame = pd.DataFrame(X, columns=["P", "V", "n", "spare"])
        model = _fast(BeamFeatRegressor, selector="permutation").fit(frame, y)
        assert list(model.feature_names_in_) == ["P", "V", "n", "spare"]
        assert all(not re.search(r"\bx\d+\b", formula) for formula in model.formulas())

    def test_fitting_a_dataframe_raises_no_feature_name_warning(self, regression_data):
        """The post-fit diagnostic scores the estimator's own validated
        array, which carries no column names. Left unfiltered, scikit-learn's
        feature-name check fires on every fit given a DataFrame and points
        the caller at a mismatch of our own making."""
        pd = pytest.importorskip("pandas")
        X, y = regression_data
        frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fast(BeamFeatRegressor, selector="permutation").fit(frame, y)
        assert not [
            entry for entry in caught if "does not have valid feature names" in str(entry.message)
        ]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestConfiguration:
    @pytest.mark.parametrize("scorer", ["correlation", "mutual_information"])
    def test_scorer_choice(self, regression_data, scorer):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, scorer=scorer).fit(X, y)
        assert model.score(X, y) > 0.9

    def test_max_features_caps_output(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, beam_width=30, max_features=5, residual_slots=0).fit(X, y)
        assert len(model.search_result_.nodes) <= 5
        assert model.n_features_out_ <= 5
        slots = _fast(BeamFeatRegressor, beam_width=30, max_features=5, residual_slots=3).fit(X, y)
        assert len(slots.search_result_.nodes) <= 5 + 3

    def test_restricted_operators(self, regression_data):
        X, y = regression_data
        # The fitted operators are a list of their own: log(x + a) needs
        # fitted_ops=() to be excluded too.
        model = _fast(BeamFeatRegressor, unary_ops=(), binary_ops=("mul",), fitted_ops=()).fit(X, y)
        assert not any("log" in formula or "sqrt" in formula for formula in model.formulas())

    def test_include_originals_false(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, include_originals=False).fit(X, y)
        assert not any(formula.startswith("x") and len(formula) <= 3 for formula in model.formulas())

    def test_depth_zero_uses_originals_only(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, max_depth=0).fit(X, y)
        assert all(formula.startswith("x") for formula in model.formulas())

    @pytest.mark.parametrize("cls", [BeamFeatRegressor, BeamFeatClassifier])
    def test_n_jobs_does_not_change_the_fit(self, cls, regression_data, classification_data, monkeypatch):
        import beamfeat.search

        monkeypatch.setattr(beamfeat.search, "MIN_PARALLEL_CANDIDATES", 0)
        X, y = regression_data if cls is BeamFeatRegressor else classification_data
        fits = [cls(max_depth=2, beam_width=20, n_jobs=n_jobs).fit(X, y) for n_jobs in (1, 2, -1)]
        for fit in fits[1:]:
            assert fit.feature_formulas_ == fits[0].feature_formulas_
            assert fit.selection_report_ == fits[0].selection_report_
            np.testing.assert_array_equal(fit.predict(X), fits[0].predict(X))

    def test_max_inputs_reaches_the_search(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, max_inputs=2).fit(X, y)
        assert all(len(node.columns()) <= 2 for node in model.search_result_.nodes)
        assert sum(record.n_rejected_constraints for record in model.search_result_.trace) > 0

    def test_bad_max_inputs_raises_at_fit(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="max_inputs"):
            _fast(BeamFeatRegressor, max_inputs=0).fit(X, y)

    def test_recovers_within_max_inputs(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(max_depth=2, beam_width=30, max_inputs=3).fit(X, y)
        assert model.score(X, y) > 0.99

    def test_magnitude_slots_find_a_centred_product(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(-1.0, 1.0, (400, 10))
        law = X[:, 0] * X[:, 1] * X[:, 2]
        y = law + rng.normal(0, 0.05 * law.std(), 400)
        with warnings.catch_warnings():
            # Without the slots the search finds nothing here, and says so.
            warnings.simplefilter("ignore", NoDiscoveriesWarning)
            plain = BeamFeatRegressor(max_depth=2, beam_width=10, magnitude_slots=0).fit(X, y)
        assert not any(set(node.columns()) == {"x0", "x1", "x2"} for node in plain.features_)
        model = BeamFeatRegressor(max_depth=2, beam_width=10, magnitude_slots=3).fit(X, y)
        assert [set(node.columns()) for node in model.features_] == [{"x0", "x1", "x2"}]
        assert model.fdr_controlled_
        assert model.search_result_.trace[1].n_magnitude == 3
        default = BeamFeatRegressor(max_depth=2, beam_width=10).fit(X, y)
        assert [set(node.columns()) for node in default.features_] == [{"x0", "x1", "x2"}]
        assert default.search_result_.trace[1].n_magnitude == 10

    def test_magnitude_slots_auto_is_ten_for_regression_and_none_for_classification(self):
        assert BeamFeatRegressor()._magnitude_slots() == 10
        assert BeamFeatTransformer()._magnitude_slots() == 10
        assert BeamFeatClassifier()._magnitude_slots() == 0
        classifying = BeamFeatTransformer(problem_type="classification")
        classifying._problem_type = classifying.problem_type  # as fit sets it
        assert classifying._magnitude_slots() == 0
        assert BeamFeatRegressor(magnitude_slots=4)._magnitude_slots() == 4

    def test_magnitude_slots_auto_runs_for_classification(self, classification_data):
        X, y = classification_data
        model = _fast(BeamFeatClassifier).fit(X, y)
        assert all(entry.n_magnitude == 0 for entry in model.search_result_.trace)

    def test_magnitude_slots_rejects_other_strings(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="magnitude_slots must be 'auto'"):
            _fast(BeamFeatRegressor, magnitude_slots="many").fit(X, y)

    def test_magnitude_slots_raise_for_classification(self, classification_data):
        X, y = classification_data
        with pytest.raises(ValueError, match="magnitude_slots"):
            _fast(BeamFeatClassifier, magnitude_slots=3).fit(X, y)

    def test_residual_slots_find_a_weak_term_beside_a_dominant_one(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 5, (2000, 4))
        signal = X[:, 0] * X[:, 1] / X[:, 2] + 0.5 * np.log(X[:, 3])
        y = signal + rng.normal(0, 0.1 * np.std(signal), 2000)
        plain = BeamFeatRegressor(random_state=0).fit(X, y)
        assert "log(x3)" not in plain.formulas()
        for options in ({}, {"parsimony_alpha": 0.05}):
            model = BeamFeatRegressor(random_state=0, residual_slots=10, **options).fit(X, y)
            assert sorted(model.formulas()) == ["((x0 / x2) * x1)", "log(x3)"]
            assert model.fdr_controlled_

    def test_residual_slots_default_changes_nothing(self, regression_data):
        X, y = regression_data
        plain = _fast(BeamFeatRegressor).fit(X, y)
        explicit = _fast(BeamFeatRegressor, residual_slots=0).fit(X, y)
        assert explicit.formulas() == plain.formulas()
        np.testing.assert_array_equal(explicit.predict(X), plain.predict(X))

    @pytest.mark.parametrize("value", ["auto", -1, 1.5, True])
    def test_residual_slots_rejects_non_integers(self, regression_data, value):
        X, y = regression_data
        with pytest.raises(ValueError, match="residual_slots must be a non-negative integer"):
            _fast(BeamFeatRegressor, residual_slots=value).fit(X, y)

    @pytest.mark.parametrize("value", [0, -5, 2.5, True, "all"])
    def test_max_search_rows_rejects_other_values(self, regression_data, value):
        X, y = regression_data
        with pytest.raises(ValueError, match="max_search_rows must be None or a positive integer"):
            _fast(BeamFeatRegressor, max_search_rows=value).fit(X, y)

    def test_max_search_rows_samples_the_search_rows_only(self, capsys):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 5, (2000, 5))
        y = X[:, 0] * X[:, 1] / X[:, 2] + rng.normal(0, 0.1, 2000)
        full = BeamFeatRegressor(max_depth=2, beam_width=10, random_state=0).fit(X, y)
        sampled = BeamFeatRegressor(max_depth=2, beam_width=10, random_state=0, max_search_rows=300, verbose=1).fit(X, y)
        assert "split 1000 search / 1000 selection (the search on 300)" in capsys.readouterr().out
        assert sampled.fdr_controlled_
        assert [node.columns() for node in sampled.features_] == [{"x0", "x1", "x2"}]
        assert sampled.score(X, X[:, 0] * X[:, 1] / X[:, 2]) > 0.99
        assert sampled.search_result_.nodes != full.search_result_.nodes

    def test_max_search_rows_at_or_above_the_search_rows_changes_nothing(self, regression_data):
        X, y = regression_data
        plain = _fast(BeamFeatRegressor).fit(X, y)
        roomy = _fast(BeamFeatRegressor, max_search_rows=10**6).fit(X, y)
        assert roomy.formulas() == plain.formulas()
        np.testing.assert_array_equal(roomy.predict(X), plain.predict(X))

    def test_max_search_rows_sample_is_reproducible(self):
        search = np.arange(0, 5000, 2)
        first = BeamFeatRegressor(max_search_rows=100, random_state=3)._search_sample(search)
        again = BeamFeatRegressor(max_search_rows=100, random_state=3)._search_sample(search)
        other = BeamFeatRegressor(max_search_rows=100, random_state=4)._search_sample(search)
        np.testing.assert_array_equal(first, again)
        assert len(first) == 100 and np.all(np.diff(first) > 0) and set(first) <= set(search)
        assert not np.array_equal(first, other)

    def test_expressions_found_on_a_sample_are_checked_on_every_search_row(self):
        from beamfeat.expression import leaf, transform

        X = np.column_stack([np.linspace(-0.5, 4.0, 200), np.linspace(1.0, 2.0, 200)])
        nodes = [transform("log", leaf("x0")), transform("log", leaf("x1")), leaf("x0")]
        model = BeamFeatRegressor()
        kept = model._check_on_search_rows(nodes, X, np.arange(200), ["x0", "x1"], None)
        assert kept == nodes[1:]
        assert model._check_on_search_rows(nodes, X, np.arange(50, 200), ["x0", "x1"], None) == nodes

    def test_residual_slots_raise_for_classification(self, classification_data):
        X, y = classification_data
        with pytest.raises(ValueError, match="residual_slots"):
            _fast(BeamFeatClassifier, residual_slots=3).fit(X, y)

    def test_alpha_affects_regressor(self, regression_data):
        X, y = regression_data
        weak = _fast(BeamFeatRegressor, alpha=0.001).fit(X, y)
        strong = _fast(BeamFeatRegressor, alpha=1000.0).fit(X, y)
        assert np.max(np.abs(strong.coef_)) < np.max(np.abs(weak.coef_))


class TestHoldoutFitCheck:
    def test_degenerate_fit_warns(self, monkeypatch):
        """A model whose holdout score is negative must warn: FDR-vetted
        association is not a generalisation certificate, and the gap must be
        loud when it opens."""
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        monkeypatch.setattr(BeamFeatRegressor, "score", lambda self, X, y: -1.7)
        with pytest.warns(DegenerateFitWarning):
            BeamFeatRegressor(max_depth=2, beam_width=20, random_state=0).fit(X, y)

    def test_degenerate_classifier_warns(self, monkeypatch):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = (X[:, 0] * X[:, 1] > 12).astype(int)
        monkeypatch.setattr(BeamFeatClassifier, "score", lambda self, X, y: 0.0)
        with pytest.warns(DegenerateFitWarning, match="majority-class rate"):
            BeamFeatClassifier(max_depth=2, beam_width=20, random_state=0).fit(X, y)

    def test_message_does_not_claim_screening_it_did_not_have(self, monkeypatch):
        """Under the fallback the screen rejected no null for the features, so
        the warning must not say it did."""
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        monkeypatch.setattr(BeamFeatRegressor, "score", lambda self, X, y: -1.7)
        with pytest.warns(DegenerateFitWarning) as record:
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3,
                on_no_discoveries="fallback",
            ).fit(X, y)
        assert model.fdr_controlled_ is False
        message = str([w for w in record if w.category is DegenerateFitWarning][0].message)
        assert "FDR-screened set" not in message
        assert "no FDR guarantee" in message

    def test_healthy_fit_does_not_warn(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateFitWarning)
            BeamFeatRegressor(max_depth=2, beam_width=20, random_state=0).fit(X, y)

    def test_no_holdout_no_check(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(1, 6, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.05, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateFitWarning)
            BeamFeatRegressor(
                max_depth=1, beam_width=10, selector=None, random_state=0
            ).fit(X, y)


class TestAlphaAuto:
    def test_default_selects_alpha_by_cv(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        assert type(model.model_).__name__ == "RidgeCV"
        assert model.alpha_ > 0

    def test_float_alpha_keeps_fixed_ridge(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, alpha=0.5).fit(X, y)
        assert type(model.model_).__name__ == "Ridge"
        assert model.alpha_ == 0.5

    def test_bad_alpha_string_raises(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="auto"):
            _fast(BeamFeatRegressor, alpha="invalid").fit(X, y)

    def test_empty_refit_does_not_keep_the_previous_alpha(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        assert model.alpha_ > 0
        # Twenty permutations cannot reach the p-values this level needs.
        unreachable = PermutationSelector(target_fdr=1e-6, n_permutations=20, max_permutations=20)
        with pytest.warns(NoDiscoveriesWarning):
            model.set_params(selector=unreachable).fit(X, y)
        assert model.alpha_ is None

    def test_wide_selection_small_n_stays_bounded(self):
        # Regression case: many correlated screened features on few rows,
        # where a fixed penalty is effectively no penalty. Predictions from
        # the cross-validated default must stay on the target's scale.
        rng = np.random.default_rng(0)
        n, p = 140, 40
        z = rng.standard_normal((n, 1))
        X = 0.8 * z + 0.6 * rng.standard_normal((n, p))
        y = (z[:, 0] + 0.3 * rng.standard_normal(n))
        Xtr, Xte, ytr, yte = X[:100], X[100:], y[:100], y[100:]
        model = BeamFeatRegressor(random_state=0, max_depth=1).fit(Xtr, ytr)
        pred = model.predict(Xte)
        assert np.all(np.isfinite(pred))
        assert float(np.mean((yte - pred) ** 2)) < 4.0 * float(np.var(yte))


class TestVerbose:
    """`verbose` is a level, not a flag, and it reports to stdout.

    Progress the caller explicitly asked for should appear without their
    having to attach a logging handler first, which is the scikit-learn
    convention; warnings keep going through the logger.
    """

    def test_zero_is_silent(self, regression_data, capsys):
        X, y = regression_data
        _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        assert capsys.readouterr().out == ""

    def test_one_reports_each_stage(self, regression_data, capsys):
        X, y = regression_data
        _fast(BeamFeatRegressor, selector="permutation", verbose=1).fit(X, y)
        out = capsys.readouterr().out
        for stage in ("fit:", "search:", "screen:", "result:"):
            assert stage in out
        assert "depth 0:" not in out

    def test_two_adds_depth_and_candidate_detail(self, regression_data, capsys):
        X, y = regression_data
        _fast(BeamFeatRegressor, selector="permutation", verbose=2).fit(X, y)
        out = capsys.readouterr().out
        assert "depth 0:" in out
        assert "p=" in out and "q=" in out

    def test_result_line_records_the_flag_and_its_scope(self, regression_data, capsys):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", verbose=1).fit(X, y)
        out = capsys.readouterr().out
        if model.fdr_controlled_:
            assert f"[FDR controlled over the {model.fdr_scope_}]" in out
        else:
            assert "[NOT FDR controlled]" in out

    def test_a_forward_step_that_keeps_everything_is_not_called_a_subset(self, capsys):
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 3.0, (600, 8))
        y = 2 * X[:, 0] + rng.normal(0, 0.5, 600)
        BeamFeatRegressor(max_depth=0, random_state=0, verbose=1).fit(X, y)
        out = capsys.readouterr().out
        assert "result set: 1 term in the fitted equation, the screened set entire" in out
        assert "not this subset" not in out

    def test_transformer_and_classifier_accept_levels(self, regression_data, capsys):
        X, y = regression_data
        _fast(BeamFeatTransformer, selector="permutation", verbose=1).fit(X, y)
        assert "fit:" in capsys.readouterr().out
        _fast(BeamFeatClassifier, selector="permutation", verbose=1).fit(X, (y > np.median(y)).astype(int))
        assert "result:" in capsys.readouterr().out


class TestConstantColumns:
    """A column with no variation standardises to zeros, so it can never be
    selected. That is correct and costs nothing, but in the output it is
    indistinguishable from a column that simply does not matter, so it is
    reported."""

    @staticmethod
    def _with_constants(_rng, n_constant):
        # Seeded here rather than drawn from the shared generator so that the
        # padded and unpadded designs share their real columns exactly; drawing
        # twice from a shared generator would compare two different problems.
        rng = np.random.default_rng(20260723)
        n = 300
        columns = [rng.uniform(1.0, 6.0, n) for _ in range(3)]
        target = columns[0] * columns[1] + rng.normal(0, 0.05, n)
        columns += [np.full(n, float(k + 1)) for k in range(n_constant)]
        return np.column_stack(columns), target

    def test_reports_constant_columns(self, rng):
        X, y = self._with_constants(rng, 2)
        with pytest.warns(UserWarning, match=r"2 of 5 columns are constant"):
            _fast(BeamFeatRegressor, selector="permutation").fit(X, y)

    def test_silent_when_every_column_varies(self, regression_data):
        X, y = regression_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        assert not [entry for entry in caught if "columns are constant" in str(entry.message)]

    def test_small_magnitude_columns_are_not_reported(self, rng):
        """The case an absolute floor on the variance would get wrong: values
        around 1e-9 vary genuinely but have a smaller variance than a stuck
        sensor reading 9.81."""
        n = 300
        tiny = rng.normal(0, 1e-9, n)
        X = np.column_stack([rng.uniform(1.0, 6.0, (n, 3)), tiny])
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fast(BeamFeatRegressor, selector="permutation").fit(X, y)
        assert not [entry for entry in caught if "columns are constant" in str(entry.message)]

    def test_long_lists_are_truncated(self, rng):
        X, y = self._with_constants(rng, 8)
        with pytest.warns(UserWarning, match=r"8 of 11 columns are constant.*\.\.\."):
            _fast(BeamFeatRegressor, selector="permutation").fit(X, y)

    def test_constant_columns_change_nothing(self, rng):
        """They are filtered by the beam before screening, so they cost neither
        accuracy nor multiplicity."""
        plain, y = self._with_constants(rng, 0)
        padded, _ = self._with_constants(rng, 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = _fast(BeamFeatRegressor, selector="permutation").fit(plain, y)
            b = _fast(BeamFeatRegressor, selector="permutation").fit(padded, y)
        assert a.n_features_out_ == b.n_features_out_
        assert len(a.selection_report_) == len(b.selection_report_)


class TestParameterValidation:
    """Out-of-range values used to be ignored: a holdout of 1.5 switched the
    split off, and an unknown on_no_discoveries behaved like "empty"."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("on_no_discoveries", "ignore"),
            ("selection_holdout", 1.5),
            ("selection_holdout", 0.0),
            ("selection_holdout", 1),
            ("selection_holdout", -0.2),
            ("selection_holdout", "0.5"),
            ("parsimony_holdout", 2.0),
            ("parsimony_holdout", True),
            ("selection_correction", "holm"),
            ("parsimony", "backward"),
            ("certify_test", "partial"),
            ("certify_test", None),
            ("parsimony_alpha", 0.0),
            ("parsimony_alpha", 1.0),
            ("parsimony_alpha", True),
            ("parsimony_alpha", "0.05"),
        ],
    )
    @pytest.mark.parametrize("cls", ESTIMATORS)
    def test_bad_values_raise(self, cls, name, value, regression_data):
        X, y = regression_data
        y = (y > np.median(y)).astype(int) if cls is BeamFeatClassifier else y
        with pytest.raises(ValueError, match=name):
            _fast(cls, selector="permutation", **{name: value}).fit(X, y)

    def test_none_disables_the_holdout(self, regression_data):
        X, y = regression_data
        with pytest.warns(UserWarning, match="same rows"):
            model = _fast(BeamFeatRegressor, selector="permutation", selection_holdout=None).fit(X, y)
        assert model.fdr_controlled_ is False


def test_estimator_exports_are_in_the_module_all():
    import beamfeat
    from beamfeat import estimators

    exported = {
        name for name in beamfeat.__all__
        if getattr(getattr(beamfeat, name), "__module__", None) == estimators.__name__
    }
    assert exported <= set(estimators.__all__)


def test_every_package_export_is_in_its_module_all():
    import importlib

    import beamfeat

    for name in beamfeat.__all__:
        module = importlib.import_module(getattr(beamfeat, name).__module__)
        assert name in module.__all__, name
