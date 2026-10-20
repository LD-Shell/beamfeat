"""Tests for the helpers that build models across scikit-learn versions."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from beamfeat._compat import l1_logistic_coefficients, logistic_regression, quiet


@pytest.fixture
def rng():
    return np.random.default_rng(20261020)


class TestQuiet:
    @pytest.mark.parametrize("category", [UserWarning, RuntimeWarning])
    def test_silences_numerical_noise(self, category):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with quiet():
                warnings.warn("noise", category, stacklevel=1)
        assert not caught

    @pytest.mark.parametrize("category", [DeprecationWarning, PendingDeprecationWarning, FutureWarning])
    def test_lets_deprecations_through(self, category):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with quiet():
                warnings.warn("deprecated", category, stacklevel=1)
        assert [warning.category for warning in caught] == [category]


class TestLogisticRegression:
    def test_strength_reaches_the_model(self):
        model = logistic_regression(C=0.25, max_iter=100)
        if "alpha" in inspect.signature(LogisticRegression).parameters:
            assert model.alpha == pytest.approx(4.0)
        else:
            assert model.C == 0.25
        assert model.max_iter == 100

    def test_fits_without_deprecation_warnings(self, rng):
        X = rng.standard_normal((120, 3))
        y = (X[:, 0] > 0).astype(int)
        with warnings.catch_warnings():
            # scikit-learn deprecates with FutureWarning; a DeprecationWarning
            # counts when beamfeat raises it, not when a dependency's own
            # internals do (scikit-learn 1.6 passes scipy a deprecated option).
            warnings.simplefilter("error", FutureWarning)
            warnings.filterwarnings("error", category=DeprecationWarning, module="beamfeat")
            logistic_regression(C=2.0, max_iter=500).fit(X, y)


class TestL1LogisticCoefficients:
    def test_two_classes_give_one_row(self, rng):
        X = rng.standard_normal((200, 5))
        y = (X[:, 0] + 0.1 * rng.standard_normal(200) > 0).astype(int)
        coefficients = l1_logistic_coefficients(X, y, cv=3, random_state=0)
        assert coefficients.shape == (1, 5)
        assert np.argmax(np.abs(coefficients[0])) == 0

    def test_more_classes_give_one_row_each(self, rng):
        X = rng.standard_normal((300, 5))
        y = np.digitize(X[:, 1] + 0.1 * rng.standard_normal(300), [-0.5, 0.5])
        coefficients = l1_logistic_coefficients(X, y, cv=3, random_state=0)
        assert coefficients.shape == (3, 5)
        assert np.argmax(np.max(np.abs(coefficients), axis=0)) == 1

    def test_no_deprecation_warnings(self, rng):
        X = rng.standard_normal((200, 4))
        y = np.digitize(X[:, 0], [-0.5, 0.5])
        with warnings.catch_warnings():
            # scikit-learn deprecates with FutureWarning; a DeprecationWarning
            # counts when beamfeat raises it, not when a dependency's own
            # internals do (scikit-learn 1.6 passes scipy a deprecated option).
            warnings.simplefilter("error", FutureWarning)
            warnings.filterwarnings("error", category=DeprecationWarning, module="beamfeat")
            with quiet():
                l1_logistic_coefficients(X, y, cv=3, random_state=0)
                l1_logistic_coefficients(X, (y > 0).astype(int), cv=3, random_state=0)
