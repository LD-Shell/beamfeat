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
    NoDiscoveriesError,
    NoDiscoveriesWarning,
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
        model = _fast(BeamFeatRegressor, max_depth=1, beam_width=10).fit(X_train, y_train)
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
        from beamfeat.estimators import _evaluate_by_row

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


class TestSelectionIntegration:
    def test_selection_result_is_exposed(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(max_depth=2, beam_width=20, selector="permutation").fit(X, y)
        assert model.selection_result_ is not None
        assert model.selection_result_.method == "permutation"

    def test_search_result_is_exposed(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        assert model.search_result_ is not None
        assert model.search_result_.n_proposed_total > 0

    def test_selector_none_skips_selection(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector=None).fit(X, y)
        assert model.selection_result_ is None
        assert model.fdr_controlled_ is None

    def test_flag_true_when_selection_passes(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", target_fdr=0.2, random_state=0
        ).fit(X, y)
        if model.fdr_controlled_:
            # Parsimony keeps a subset of the screened set, never more.
            assert 0 < model.n_features_out_ <= model.selection_result_.n_selected

    def test_empty_default_on_pure_noise(self):
        """Default on_no_discoveries="empty": on a noise target where nothing
        passes, the model keeps no features, warns visibly, and predicts the
        training mean — the honest null model."""
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, selector="permutation", target_fdr=0.05, random_state=3
            ).fit(X, y)
        assert model.n_features_out_ == 0
        assert model.fdr_controlled_ is False
        np.testing.assert_allclose(model.predict(X[:7]), np.full(7, y.mean()), rtol=1e-9)
        assert "no feature passed selection" in model.equation()

    def test_fallback_mode_keeps_search_output_flagged(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3,
                on_no_discoveries="fallback",
            ).fit(X, y)
        assert model.n_features_out_ > 0
        assert model.fdr_controlled_ is False

    def test_raise_mode_raises(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        with pytest.raises(NoDiscoveriesError):
            BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3,
                on_no_discoveries="raise",
            ).fit(X, y)

    @pytest.mark.parametrize("cls", [BeamFeatRegressor, BeamFeatClassifier, BeamFeatTransformer])
    def test_parsimony_none_returns_the_whole_screened_set(self, cls, regression_data,
                                                           classification_data):
        """``parsimony=None`` fits the screened set itself rather than a subset
        chosen from it, which is what makes the printed equation the object the
        guarantee covers."""
        X, y = (classification_data if cls is BeamFeatClassifier else regression_data)
        model = cls(max_depth=2, beam_width=20, selector="permutation",
                    random_state=0, parsimony=None).fit(X, y)
        screened = {row["formula"] for row in model.selection_report_ if row["screened"]}
        assert screened, "the fixture is meant to produce discoveries"
        assert set(model.feature_formulas_) == screened
        assert model.n_features_out_ == model.selection_result_.n_selected
        assert model.fdp_inflation_ == pytest.approx(1.0)

    @pytest.mark.parametrize("cls", [BeamFeatRegressor, BeamFeatClassifier, BeamFeatTransformer])
    def test_default_prunes_within_the_screened_set(self, cls, regression_data,
                                                    classification_data):
        """The default is ``parsimony="forward"``: what is fitted is a subset of
        what was screened, and every member of it passed screening."""
        X, y = (classification_data if cls is BeamFeatClassifier else regression_data)
        model = cls(max_depth=2, beam_width=20, selector="permutation",
                    random_state=0).fit(X, y)
        assert model.parsimony == "forward"
        screened = {row["formula"] for row in model.selection_report_ if row["screened"]}
        assert set(model.feature_formulas_) <= screened
        assert model.n_features_out_ <= model.selection_result_.n_selected
        assert model.fdp_inflation_ >= 1.0

    def test_parsimony_compacts_without_losing_recovery(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        compact = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0).fit(X, y)
        full = BeamFeatRegressor(
            max_depth=2, beam_width=25, random_state=0, parsimony=None
        ).fit(X, y)
        assert compact.n_features_out_ < full.n_features_out_
        assert any("x0" in f and "x1" in f for f in compact.formulas())
        assert compact.score(X, y) > 0.99
        # every kept feature passed screening
        report = {r["formula"]: r for r in compact.selection_report_}
        assert all(report[f]["screened"] for f in compact.formulas())

    def test_selection_report_is_consistent(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        model = BeamFeatRegressor(max_depth=2, beam_width=20, random_state=1).fit(X, y)
        report = model.selection_report_
        assert report and all(
            set(row) == {"formula", "p_value", "q_value", "statistic", "screened", "kept"}
            for row in report
        )
        for row in report:
            if row["kept"]:
                assert row["screened"]
                assert row["q_value"] <= model.target_fdr + 1e-9

    def test_nan_error_is_actionable(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (50, 3))
        X[3, 1] = np.nan
        with pytest.raises(ValueError, match="SimpleImputer"):
            BeamFeatRegressor(max_depth=1, beam_width=5).fit(X, rng.standard_normal(50))

    def test_same_data_selection_never_claims_control(self, regression_data):
        """With the holdout disabled, candidates are tested on the rows that
        chose them; the estimator must not claim the FDR guarantee."""
        X, y = regression_data
        model = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", selection_holdout=None, random_state=0
        ).fit(X, y)
        assert model.fdr_controlled_ is False

    def test_holdout_split_is_disjoint_and_seed_stable(self, regression_data, monkeypatch):
        """The rows the search sees, the rows screening tests on and the rows
        the re-test uses must not overlap. Each target value is unique, so the
        rows a stage saw are identified by the targets it was given."""
        from beamfeat.search import BeamSearch
        from beamfeat.selection import PermutationSelector

        X, y = regression_data
        assert len(np.unique(y)) == len(y)
        seen = {"search": [], "select": []}
        run, select = BeamSearch.run, PermutationSelector.select

        def recording_run(self, data, target, *args, **kwargs):
            seen["search"].append(np.asarray(target))
            return run(self, data, target, *args, **kwargs)

        def recording_select(self, features, target):
            seen["select"].append(np.asarray(target))
            return select(self, features, target)

        monkeypatch.setattr(BeamSearch, "run", recording_run)
        monkeypatch.setattr(PermutationSelector, "select", recording_select)

        first = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", random_state=7,
            parsimony_holdout=0.5,
        ).fit(X, y)
        stages = [seen["search"][0], *seen["select"]]
        assert len(stages) == 3
        for i, earlier in enumerate(stages):
            for later in stages[i + 1:]:
                assert not set(earlier) & set(later)
        assert sum(len(stage) for stage in stages) == len(y)
        np.testing.assert_array_equal(np.sort(y[first._holdout_rows_]),
                                      np.sort(np.concatenate(seen["select"])))

        second = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", random_state=7,
            parsimony_holdout=0.5,
        ).fit(X, y)
        np.testing.assert_array_equal(first._holdout_rows_, second._holdout_rows_)
        assert first.formulas() == second.formulas()

    def test_selected_features_still_predict(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", target_fdr=0.2
        ).fit(X, y)
        assert model.score(X, y) > 0.9

    def test_pipeline_level_fdr_calibration(self):
        """End-to-end control: search on one split, selection on the other.

        A feature is a false discovery iff its formula references only
        irrelevant columns. FDR is an expectation, so the mean FDP over
        trials is what must respect the nominal level (with Monte Carlo
        slack) — not each individual trial.
        """
        nominal = 0.2
        fdps = []
        recovered = 0
        n_trials = 12
        for trial in range(n_trials):
            rng = np.random.default_rng(100 + trial)
            X = rng.uniform(1, 6, (400, 6))
            signal = X[:, 0] * X[:, 1]
            y = signal + rng.normal(0, 0.05 * np.std(signal), 400)
            model = BeamFeatRegressor(
                max_depth=2, beam_width=20, selector="permutation", target_fdr=nominal,
                selection_holdout=0.5, random_state=trial,
            ).fit(X, y)
            if not model.fdr_controlled_:
                continue
            relevant = {"x0", "x1"}
            false = sum(1 for node in model.features_ if not (node.columns() & relevant))
            fdps.append(false / len(model.features_))
            recovered += any("x0" in f and "x1" in f for f in model.formulas())
        assert fdps, "selection never passed; pipeline has no power"
        assert float(np.mean(fdps)) <= nominal + 0.15
        assert recovered >= len(fdps) - 2


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


class TestUnitsIntegration:
    def test_units_constrain_constructed_features(self, rng):
        pint = pytest.importorskip("pint")
        ureg = pint.UnitRegistry()

        n = 300
        X = rng.uniform(1.0, 5.0, (n, 2))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)
        units = {"x0": 1.0 * ureg.kilogram, "x1": 1.0 * ureg.meter}

        model = _fast(BeamFeatRegressor, units=units).fit(X, y)
        formulas = model.formulas()
        assert not any("+" in formula or "-" in formula for formula in formulas)
        assert any("*" in formula for formula in formulas)

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
        model = _fast(BeamFeatRegressor, beam_width=30, max_features=5).fit(X, y)
        assert model.n_features_out_ <= 5

    def test_restricted_operators(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, unary_ops=(), binary_ops=("mul",)).fit(X, y)
        assert not any("log" in formula or "sqrt" in formula for formula in model.formulas())

    def test_include_originals_false(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, include_originals=False).fit(X, y)
        assert not any(formula.startswith("x") and len(formula) <= 3 for formula in model.formulas())

    def test_depth_zero_uses_originals_only(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, max_depth=0).fit(X, y)
        assert all(formula.startswith("x") for formula in model.formulas())

    def test_alpha_affects_regressor(self, regression_data):
        X, y = regression_data
        weak = _fast(BeamFeatRegressor, alpha=0.001).fit(X, y)
        strong = _fast(BeamFeatRegressor, alpha=1000.0).fit(X, y)
        assert np.max(np.abs(strong.coef_)) < np.max(np.abs(weak.coef_))


class TestSelectorDiagnosticsSurfaced:
    def test_knockoff_unsatisfiability_is_a_visible_warning(self):
        """A knockoff configuration that cannot select must warn through the
        warnings module, not only the logger, so notebook users see it."""
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (300, 4))
        y = X[:, 0] / X[:, 1] + rng.normal(0, 0.02, 300)
        with pytest.warns(UserWarning, match="offset"):
            BeamFeatRegressor(
                max_depth=2, beam_width=15, selector="knockoff", target_fdr=0.1, random_state=0
            ).fit(X, y)


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
        """Under the fallback the features never passed screening, so the
        warning must not say they did."""
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
        assert "passed FDR screening" not in message
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


class TestClassifierEquation:
    def test_binary_equation_reproduces_decision_function(self):
        """The printed log-odds equation, evaluated on raw feature values,
        must equal decision_function — the same model-equation identity the
        regressor guarantees."""
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = (X[:, 0] * X[:, 1] > X[:, 2] * X[:, 3]).astype(int)
        model = BeamFeatClassifier(max_depth=2, beam_width=20, random_state=0).fit(X, y)
        equation = model.equation(precision=10)
        assert equation.startswith("logit P(y = ")

        features = model._apply(X[:12])
        scale = np.where(model.scaler_.scale_ == 0, 1.0, model.scaler_.scale_)
        raw = np.atleast_2d(model.coef_)[0] / scale
        offset = float(model.intercept_[0] - np.sum(raw * model.scaler_.mean_))
        manual = features @ raw + offset
        np.testing.assert_allclose(manual, model.decision_function(X[:12]), rtol=1e-9)

    def test_labels_print_as_plain_values(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = (X[:, 0] * X[:, 1] > X[:, 2] * X[:, 3]).astype(np.int64)
        model = BeamFeatClassifier(max_depth=1, beam_width=10, random_state=0).fit(X, y)
        equation = model.equation()
        assert equation.startswith("logit P(y = 1) = ")
        assert "np." not in equation
        named = BeamFeatClassifier(max_depth=1, beam_width=10, random_state=0).fit(
            X, np.where(y == 1, "yes", "no")
        )
        assert named.equation().startswith("logit P(y = 'yes') = ")

    def test_multiclass_equation_has_one_line_per_class(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 6, (450, 4))
        y = np.digitize(X[:, 0] * X[:, 1], [8.0, 18.0])
        model = BeamFeatClassifier(max_depth=1, beam_width=10, random_state=0).fit(X, y)
        lines = model.equation().splitlines()
        assert sum(line.startswith("score(") for line in lines) == len(model.classes_)
        assert lines[0].startswith("score(0) = ")

    def test_null_model_equation_reports_priors(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = (rng.uniform(size=400) > 0.5).astype(int)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatClassifier(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, y)
        assert "class priors only" in model.equation()


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

    def test_transformer_and_classifier_accept_levels(self, regression_data, capsys):
        X, y = regression_data
        _fast(BeamFeatTransformer, selector="permutation", verbose=1).fit(X, y)
        assert "fit:" in capsys.readouterr().out
        _fast(BeamFeatClassifier, selector="permutation", verbose=1).fit(X, (y > np.median(y)).astype(int))
        assert "result:" in capsys.readouterr().out


class TestParsimonyNote:
    """``equation()`` marks a pruned equation as a subset of the certified set.

    The guarantee is a statement about the screened set, and the printed
    equation is normally a subset of it. A caller who reads the equation and
    the ``fdr_controlled_`` flag side by side would otherwise have no signal
    that the two do not refer to the same set of terms."""

    MARKER = "certified terms; parsimony=None prints all"

    def test_present_when_terms_were_dropped(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        if model.n_features_out_ >= model.selection_result_.n_selected:
            pytest.skip("this fit pruned nothing, so there is no gap to mark")
        equation = model.equation()
        assert self.MARKER in equation
        assert f"[{model.n_features_out_} of {model.selection_result_.n_selected}" in equation

    def test_absent_when_parsimony_is_off(self, regression_data):
        """The terms then are the certified set, so there is nothing to mark."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation",
                      beam_width=25, parsimony=None).fit(X, y)
        assert self.MARKER not in model.equation()

    def test_absent_without_a_selector(self, regression_data):
        """Nothing was certified, so no claim about a certified set is implied."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector=None).fit(X, y)
        assert self.MARKER not in model.equation()

    def test_absent_when_the_subset_was_re_certified(self, regression_data):
        """``parsimony_holdout`` re-tests the subset on rows of its own, so the
        guarantee does cover the printed terms and the note would be wrong."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation",
                      parsimony_holdout=0.5).fit(X, y)
        if model.certification_result_ is None:
            pytest.skip("the rows did not support a second split")
        assert self.MARKER not in model.equation()

    def test_absent_when_nothing_passed_selection(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, rng.standard_normal(400))
        assert self.MARKER not in model.equation()

    def test_does_not_disturb_the_equation_itself(self, regression_data):
        """The note is appended; the coefficients and formulas are untouched."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        head = model.equation().split("   [")[0]
        assert head.startswith("y = ")
        assert any(formula in head for formula in model.formulas())

    def test_classifier_marks_the_block_once(self, rng):
        """The terms are shared across a multiclass equation's lines, so one
        note belongs at the end of the block rather than on every line."""
        X = rng.uniform(1.0, 6.0, (400, 4))
        score = X[:, 0] * X[:, 1]
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
        model = _fast(BeamFeatClassifier, selector="permutation", beam_width=25).fit(X, y)
        equation = model.equation()
        if self.MARKER in equation:
            assert equation.count(self.MARKER) == 1


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

    def test_none_when_nothing_passed_selection(self):
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
            # every returned feature survived the re-test, so the count cannot
            # exceed what that screen certified
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


class TestSelectorInstance:
    """An instance keeps its own settings; a disagreement with the estimator's
    is named rather than resolved in silence."""

    def test_disagreeing_correction_is_named(self, regression_data):
        from beamfeat import PermutationSelector

        X, y = regression_data
        with pytest.warns(UserWarning, match=r"selection_correction='by'.*correction='bh'"):
            model = _fast(BeamFeatRegressor, selector=PermutationSelector()).fit(X, y)
        assert model.selection_result_ is not None

    def test_disagreeing_target_fdr_is_named(self, regression_data):
        from beamfeat import PermutationSelector

        X, y = regression_data
        with pytest.warns(UserWarning, match=r"target_fdr=0.05.*target_fdr=0.1"):
            _fast(
                BeamFeatRegressor, target_fdr=0.05, selector=PermutationSelector(correction="by")
            ).fit(X, y)

    def test_agreeing_instance_is_silent(self, regression_data):
        from beamfeat import PermutationSelector

        X, y = regression_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fast(BeamFeatRegressor, selector=PermutationSelector(correction="by")).fit(X, y)
        assert not [entry for entry in caught if "instance" in str(entry.message)]

    def test_verbose_reports_the_correction_that_ran(self, regression_data, capsys):
        from beamfeat import PermutationSelector

        X, y = regression_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _fast(BeamFeatRegressor, selector=PermutationSelector(), verbose=1).fit(X, y)
        assert "permutation (BH)" in capsys.readouterr().out


class TestSingleCandidate:
    def test_a_lone_candidate_is_screened(self, rng):
        """With one candidate, screening used to be skipped and the term
        returned unvetted."""
        X = rng.uniform(1.0, 6.0, (300, 1))
        y = 2 * X[:, 0] + rng.normal(0, 0.1, 300)
        model = BeamFeatRegressor(max_depth=0, random_state=0).fit(X, y)
        assert model.selection_result_ is not None
        assert model.selection_result_.n_candidates == 1
        assert model.fdr_controlled_ is True
        assert model.formulas() == ["x0"]

    def test_a_lone_null_candidate_can_be_screened_out(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1.0, 6.0, (300, 1))
        y = rng.standard_normal(300)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(max_depth=0, random_state=0).fit(X, y)
        assert model.selection_result_.p_values[0] > 0.1
        assert model.n_features_out_ == 0


def test_estimator_exports_are_in_the_module_all():
    import beamfeat
    from beamfeat import estimators

    exported = {
        name for name in beamfeat.__all__
        if getattr(getattr(beamfeat, name), "__module__", None) == estimators.__name__
    }
    assert exported <= set(estimators.__all__)


def _reference_forward_path(matrix, target):
    """The forward step re-solved by least squares at every step, to the end."""
    n_rows, n_columns = matrix.shape
    total_variance = float(np.var(target))
    chosen, remaining, scores = [], list(range(n_columns)), []
    while remaining:
        step = []
        for index in remaining:
            design = np.column_stack([np.ones(n_rows)] + [matrix[:, j] for j in chosen + [index]])
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            step.append(1.0 - float(np.var(target - design @ coefficients)) / total_variance)
        best = int(np.argmax(step))
        chosen.append(remaining.pop(best))
        scores.append(step[best])
    return chosen, scores


class TestEquationList:
    """``equations_`` records the forward path through the whole screened set.

    The fitted model must be exactly the one the forward step picks at
    ``parsimony_tol``; the path only adds a record of the sizes it did not
    pick."""

    SCOPE = "not an equation chosen from this list"

    @pytest.fixture
    def model(self, regression_data):
        X, y = regression_data
        return _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)

    def test_one_record_per_screened_size(self, model):
        records = model.equations_
        assert len(records) == model.selection_result_.n_selected
        assert [record["n_terms"] for record in records] == list(range(1, len(records) + 1))
        screened = {row["formula"] for row in model.selection_report_ if row["screened"]}
        assert set(records[-1]["formulas"]) == screened

    def test_records_are_nested_and_monotone(self, model):
        records = model.equations_
        for shorter, longer in zip(records, records[1:]):
            assert longer["formulas"][:-1] == shorter["formulas"]
            assert longer["complexity"] > shorter["complexity"]
            assert longer["r2"] >= shorter["r2"] - 1e-9
        assert all(0.0 <= record["r2"] <= 1.0 + 1e-12 for record in records)

    def test_complexity_counts_nodes_and_coefficients(self, model):
        def tree_size(node):
            return 1 + sum(tree_size(child) for child in node.children)

        for record in model.equations_:
            assert record["complexity"] == sum(tree_size(node) + 1 for node in record["features"])
            assert [node.name for node in record["features"]] == record["formulas"]

    def test_the_marked_record_is_the_fitted_model(self, model):
        marked = [record for record in model.equations_ if record["model"]]
        assert len(marked) == 1
        assert sorted(marked[0]["formulas"]) == sorted(model.feature_formulas_)
        assert marked[0]["n_terms"] == model._forward_stop_

    def test_the_marked_record_is_where_the_forward_step_stops(self, model):
        records = model.equations_
        stop = next(record["n_terms"] for record in records if record["model"])
        if stop < len(records):
            assert records[stop]["r2"] - records[stop - 1]["r2"] < model.parsimony_tol + 1e-9
        for record, previous in zip(records[1:stop], records[: stop - 1]):
            assert record["r2"] - previous["r2"] >= model.parsimony_tol

    def test_matches_a_path_re_solved_at_every_step(self, rng):
        """The steps past the forward step's stopping point are taken by
        projection rather than re-solving; they must agree with re-solving,
        including around an exactly collinear column."""
        latent = rng.standard_normal((150, 4))
        matrix = np.column_stack(
            [latent @ rng.standard_normal(4) + 0.3 * rng.standard_normal(150) for _ in range(12)]
        )
        matrix[:, 5] = 2.0 * matrix[:, 1] + 1.0
        target = matrix[:, :3] @ np.array([1.0, -0.5, 0.25]) + rng.standard_normal(150)
        reference, reference_scores = _reference_forward_path(matrix, target)

        estimator = BeamFeatRegressor(parsimony_tol=0.05)
        order, scores, stop = estimator._forward_path(matrix, target, list(range(12)))
        assert 1 <= stop < 12
        assert order[:stop] == reference[:stop]
        assert scores[:stop] == reference_scores[:stop]
        np.testing.assert_allclose(scores, reference_scores, atol=1e-10)
        assert sorted(order) == list(range(12))
        assert estimator._parsimonious_subset(matrix, target, list(range(12))) == sorted(order[:stop])

    def test_does_not_change_the_fitted_model(self, regression_data):
        """The path is the same whatever ``parsimony`` is, and only the
        marked record moves."""
        X, y = regression_data
        forward = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        full = _fast(BeamFeatRegressor, selector="permutation", beam_width=25,
                     parsimony=None).fit(X, y)
        assert [r["formulas"] for r in forward.equations_] == [r["formulas"] for r in full.equations_]
        assert [r["r2"] for r in forward.equations_] == [r["r2"] for r in full.equations_]
        assert full.equations_[-1]["model"]
        assert sorted(full.equations_[-1]["formulas"]) == sorted(full.feature_formulas_)

    def test_parsimony_none_marks_the_screened_set(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25,
                      parsimony=None).fit(X, y)
        text = model.equations()
        assert "parsimony=None keeps the whole screened set" in text
        assert self.SCOPE in text
        assert [r["model"] for r in model.equations_].count(True) == 1

    def test_text_marks_the_model_and_states_the_scope(self, model):
        text = model.equations()
        lines = text.splitlines()
        stop = next(record["n_terms"] for record in model.equations_ if record["model"])
        marked = [line for line in lines if line.startswith("*") and " ~ " in line]
        assert len(marked) == 1
        assert marked[0].split()[1] == str(stop)
        assert lines[-1].startswith("The FDR guarantee covers the screened set of "
                                    f"{len(model.equations_)} terms")
        assert self.SCOPE in lines[-1]
        assert f"parsimony_tol={model.parsimony_tol:g}" in text
        for formula in model.equations_[0]["formulas"]:
            assert formula in text

    def test_text_can_be_shortened(self, model):
        records = model.equations_
        if len(records) < 4:
            pytest.skip("too few screened terms to shorten the list")
        text = model.equations(max_n_terms=2)
        rows = [line for line in text.splitlines() if " ~ " in line]
        sizes = [int(line.lstrip("* ").split()[0]) for line in rows]
        stop = next(record["n_terms"] for record in records if record["model"])
        assert sizes == sorted({1, 2, stop})
        assert "..." in text
        assert text.splitlines()[-1].endswith(self.SCOPE + ".")

    def test_rejects_a_bad_precision(self, model):
        with pytest.raises(ValueError, match="precision"):
            model.equations(precision=0)

    def test_none_without_a_selector(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector=None).fit(X, y)
        assert model.equations_ is None
        assert "no selector was configured" in model.equations()

    def test_empty_when_nothing_passed_selection(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, rng.standard_normal(400))
        assert model.equations_ == []
        assert model.equations() == "no equation list: no feature passed selection"

    def test_a_single_screened_feature(self, rng):
        X = rng.uniform(1.0, 6.0, (300, 1))
        y = 2 * X[:, 0] + rng.normal(0, 0.1, 300)
        model = BeamFeatRegressor(max_depth=0, random_state=0).fit(X, y)
        assert len(model.equations_) == 1
        record = model.equations_[0]
        assert record["formulas"] == ["x0"]
        assert record["complexity"] == 2
        assert record["model"]
        assert record["r2"] > 0.99
        text = model.equations()
        assert "the screened set of 1 term (line 1)" in text
        assert "the forward step keeps every screened term" in text

    def test_classification(self, classification_data):
        X, y = classification_data
        model = _fast(BeamFeatClassifier, selector="permutation").fit(X, y)
        records = model.equations_
        assert len(records) == model.selection_result_.n_selected
        assert [record["model"] for record in records].count(True) == 1
        marked = next(record for record in records if record["model"])
        assert sorted(marked["formulas"]) == sorted(model.feature_formulas_)
        text = model.equations()
        assert "logit P(y = 1) ~ " in text
        assert "integer-coded labels" in text

    def test_multiclass_and_string_labels(self, rng):
        X = rng.uniform(1.0, 6.0, (400, 4))
        score = X[:, 0] * X[:, 1]
        labels = np.array(["low", "mid", "high"])[np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))]
        model = _fast(BeamFeatClassifier, selector="permutation", beam_width=25).fit(X, labels)
        assert model.equations_
        assert "scores ~ " in model.equations()

    def test_transformer_records_it_too(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatTransformer, selector="permutation", beam_width=25).fit(X, y)
        assert len(model.equations_) == model.selection_result_.n_selected
        assert "* the returned features" in model.equations()

    def test_same_data_selection_claims_no_guarantee(self, regression_data):
        X, y = regression_data
        with pytest.warns(UserWarning, match="same rows"):
            model = _fast(BeamFeatRegressor, selector="permutation",
                          selection_holdout=None).fit(X, y)
        text = model.equations()
        assert "No line carries the FDR guarantee" in text
        assert "The FDR guarantee covers" not in text

    def test_re_certified_model(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation",
                      parsimony_holdout=0.5).fit(X, y)
        if model.certification_result_ is None:
            pytest.skip("the rows did not support a second split")
        text = model.equations()
        marked = [record for record in model.equations_ if record["model"]]
        if marked:
            assert sorted(marked[0]["formulas"]) == sorted(model.feature_formulas_)
        else:
            assert "No line is the fitted model" in text
        assert "re-test" in text
        assert "does not cover another equation chosen from this list" in text

    def test_unfitted_raises(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            BeamFeatRegressor().equations()


class TestLatex:
    """``latex()`` is ``equation()`` typeset: the same terms, order,
    coefficients and note, with each feature rendered by sympy."""

    @pytest.fixture
    def model(self, regression_data):
        pytest.importorskip("sympy")
        X, y = regression_data
        return _fast(BeamFeatRegressor, selector="permutation", beam_width=25,
                     parsimony_tol=1e-4).fit(X, y)

    def test_same_coefficients_as_the_equation(self, model):
        equation, latex = model.equation(), model.latex()
        assert latex.startswith("y = ")
        coefficients = [
            match.group(1)
            for match in re.finditer(r"(\d+(?:\.\d+)?(?:e[+-]\d+)?)\*", equation)
        ]
        assert coefficients
        for coefficient in coefficients:
            assert coefficient in latex
        assert equation.split("   [")[0].rsplit(" ", 1)[-1] in latex

    def test_contains_every_feature(self, model):
        latex = model.latex()
        for formula in model.latex_formulas():
            assert formula in latex

    def test_latex_formulas_come_from_sympy(self, model):
        sympy = pytest.importorskip("sympy")
        assert model.latex_formulas() == [sympy.latex(e) for e in model.to_sympy()]

    def test_carries_the_parsimony_note(self, model):
        latex = model.latex()
        if model.fdr_scope_ == "screened set":
            assert r"\quad \text{[" in latex
            assert "certified terms; parsimony=None prints all" in latex
        else:
            assert r"\text{[" not in latex

    def test_note_false_returns_the_bare_equation(self, model):
        bare = model.latex(note=False)
        assert r"\quad" not in bare and r"\text{" not in bare
        assert model.latex().startswith(bare)

    def test_no_note_when_parsimony_is_off(self, regression_data):
        pytest.importorskip("sympy")
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25,
                      parsimony=None).fit(X, y)
        assert "certified terms" not in model.latex(max_terms=3)

    def test_options_follow_equation(self, model):
        scientific = model.latex(style="scientific", precision=2)
        assert r"\times 10^{" in scientific
        assert "e+" not in scientific and "e-" not in scientific
        assert model.latex(max_terms=1).count(r"\quad") <= 1
        with pytest.raises(ValueError, match="precision"):
            model.latex(precision=0)
        with pytest.raises(ValueError, match="style"):
            model.latex(style="roman")

    def test_sums_are_parenthesised(self):
        sympy = pytest.importorskip("sympy")
        from beamfeat.estimators import _render_linear_latex

        class Scaler:
            mean_ = np.array([0.0, 0.0])
            scale_ = np.array([1.0, 1.0])

        a, b = sympy.symbols("a b", real=True)
        rhs = _render_linear_latex(
            np.array([2.0, -0.5]), 1.0, Scaler(), [a + b, a * b], 4, None, "significant"
        )
        assert rhs == r"2 \left(a + b\right) - 0.5 a b + 1"

    def test_numbers_and_text(self):
        from beamfeat.estimators import _latex_label, _latex_number, _latex_text

        assert _latex_number("3.2e-07") == r"3.2 \times 10^{-7}"
        assert _latex_number("1.50e+03") == r"1.50 \times 10^{3}"
        assert _latex_number("0.003943") == "0.003943"
        assert _latex_text("a_b & 50%") == r"\text{a\_b \& 50\%}"
        assert _latex_label(np.int64(1)) == "1"
        assert _latex_label("yes") == r"\text{yes}"

    def test_null_model(self):
        pytest.importorskip("sympy")
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, rng.standard_normal(400))
        assert model.latex().endswith(r"\quad \text{(no feature passed selection)}")
        assert r"\text{" not in model.latex(note=False)

    def test_binary_classifier(self, classification_data):
        pytest.importorskip("sympy")
        X, y = classification_data
        model = _fast(BeamFeatClassifier).fit(X, y)
        latex = model.latex()
        assert latex.startswith(r"\operatorname{logit} P(y = 1) = ")
        for formula in model.latex_formulas():
            assert formula in latex

    def test_classifier_with_string_labels(self, rng):
        pytest.importorskip("sympy")
        X = rng.uniform(1.0, 6.0, (300, 3))
        labels = np.where(X[:, 0] * X[:, 1] > 9, "yes", "no")
        model = _fast(BeamFeatClassifier, max_depth=1).fit(X, labels)
        assert model.latex().startswith(r"\operatorname{logit} P(y = \text{yes}) = ")

    def test_multiclass_classifier(self, rng):
        pytest.importorskip("sympy")
        X = rng.uniform(1.0, 6.0, (400, 4))
        score = X[:, 0] * X[:, 1]
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
        model = _fast(BeamFeatClassifier, selector="permutation", beam_width=25).fit(X, y)
        latex = model.latex()
        assert latex.startswith("\\begin{aligned}\n")
        assert latex.count(r"\operatorname{score}(") == 3
        assert latex.count(" &= ") == 3
        assert "argmax of the softmax scores" in latex
        assert model.latex(note=False).endswith("\\end{aligned}")
        if model.fdr_scope_ == "screened set":
            assert latex.count("certified terms") == 1

    def test_classifier_null_model(self):
        pytest.importorskip("sympy")
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatClassifier(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, rng.integers(0, 2, 400))
        latex = model.latex()
        assert latex.startswith("P(y = 0) = ")
        assert "no feature passed selection" in latex

    def test_missing_sympy_fails_as_to_sympy_does(self, model, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "sympy", None)
        with pytest.raises(ImportError) as from_to_sympy:
            model.to_sympy()
        for method in (model.latex, model.latex_formulas):
            with pytest.raises(ImportError) as raised:
                method()
            assert type(raised.value) is type(from_to_sympy.value)
            assert "sympy" in str(raised.value)


class TestLambdifyExport:
    """The export recipe in the guide: ``sympy.lambdify`` on ``to_sympy()``
    gives functions of the input columns that reproduce the fitted features,
    in NumPy here and in JAX or PyTorch with ``modules="jax"`` or
    ``"torch"``."""

    @staticmethod
    def _lambdified(model, n_columns):
        sympy = pytest.importorskip("sympy")
        columns = [sympy.Symbol(f"x{i}", real=True) for i in range(n_columns)]
        return sympy.lambdify(columns, model.to_sympy(), modules="numpy")

    def test_features_reproduce_transform(self, regression_data, rng):
        X, y = regression_data
        model = _fast(BeamFeatTransformer, selector="permutation", beam_width=25,
                      parsimony=None).fit(X, y)
        features = self._lambdified(model, X.shape[1])
        X_new = rng.uniform(1.0, 6.0, (200, X.shape[1]))
        exported = np.column_stack(features(*X_new.T))
        np.testing.assert_allclose(exported, model.transform(X_new), rtol=1e-10, atol=1e-12)

    def test_equation_reproduces_predict(self, regression_data, rng):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector="permutation", beam_width=25).fit(X, y)
        features = self._lambdified(model, X.shape[1])
        X_new = rng.uniform(1.0, 6.0, (200, X.shape[1]))
        raw = model.coef_ / model.scaler_.scale_
        offset = model.intercept_ - raw @ model.scaler_.mean_
        exported = np.column_stack(features(*X_new.T)) @ raw + offset
        np.testing.assert_allclose(exported, model.predict(X_new), rtol=1e-10, atol=1e-10)
