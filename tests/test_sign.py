"""The opt-in sign operators: ``sign`` among the transforms, ``sign_shift`` among the fitted ones."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from beamfeat import BeamFeatRegressor
from beamfeat.expression import OPERATORS, Evaluator, combine, leaf, transform
from beamfeat.search import DEFAULT_FITTED, DEFAULT_UNARY


@pytest.fixture
def ureg():
    pint = pytest.importorskip("pint")
    return pint.UnitRegistry()


def _switch(threshold, seed, n=1000):
    """``y = x1 + x2`` times 2 above the threshold on x0 and -1 below it."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-2, 2, (n, 2)), rng.uniform(1, 5, (n, 3))])
    signal = np.where(X[:, 0] > threshold, 2.0, -1.0) * X[:, 2] + X[:, 1]
    return X, signal, signal + rng.normal(0, 0.3, n)


class TestSign:
    def test_registered_and_off_by_default(self):
        assert OPERATORS["sign"].arity == 1
        assert "sign" not in DEFAULT_UNARY

    def test_formula_values_and_sympy(self):
        pytest.importorskip("sympy")
        node = transform("sign", leaf("x0"))
        assert node.name == "sign(x0)"
        values = Evaluator({"x0": np.array([-2.0, 0.0, 3.0, 1.0])}).evaluate(node)
        np.testing.assert_array_equal(values, [-1.0, 0.0, 1.0, 1.0])
        assert str(node.to_sympy()) == "sign(x0)"

    def test_a_column_of_one_sign_is_constant_and_excluded(self):
        assert Evaluator({"x0": np.array([1.0, 2.0, 3.0])}).evaluate(transform("sign", leaf("x0"))) is None

    def test_carries_no_dimension(self, ureg):
        speed = leaf("v", 1.0 * ureg.meter / ureg.second)
        assert transform("sign", speed).unit.dimensionless
        product = combine("mul", transform("sign", speed), leaf("d", 1.0 * ureg.meter))
        assert product.unit.dimensionality == (1.0 * ureg.meter).dimensionality

    @pytest.mark.parametrize("seed", [0, 1])
    def test_finds_a_switch_at_zero(self, seed):
        X, signal, y = _switch(0.0, seed)
        default = BeamFeatRegressor(random_state=seed).fit(X, y)
        model = BeamFeatRegressor(random_state=seed, unary_ops=DEFAULT_UNARY + ("sign",)).fit(X, y)
        assert model.formulas()[0] == "(x2 * sign(x0))"
        assert model.score(X, signal) > 0.995
        assert default.score(X, signal) < 0.8


class TestSignShift:
    def test_registered_fitted_and_off_by_default(self):
        assert OPERATORS["sign_shift"].parametric
        assert "sign_shift" not in DEFAULT_FITTED

    def test_formula(self):
        assert transform("sign_shift", leaf("x0"), const=-0.7).name == "sign(x0 - 0.7)"
        assert transform("sign_shift", leaf("x0"), const=0.5).name == "sign(x0 + 0.5)"
        assert transform("sign_shift", leaf("x0"), const=0.0) == transform("sign", leaf("x0"))

    def test_grid_keeps_a_twentieth_of_the_rows_each_side(self):
        x = np.linspace(0.0, 10.0, 201)
        (grid,) = OPERATORS["sign_shift"].constant_grid(x)
        assert np.all(np.diff(grid) > 0)
        assert grid[0] == pytest.approx(-9.5) and grid[-1] == pytest.approx(-0.5)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_finds_a_switch_away_from_zero(self, seed):
        X, signal, y = _switch(0.7, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = BeamFeatRegressor(random_state=seed, unary_ops=DEFAULT_UNARY + ("sign",)).fit(X, y)
        model = BeamFeatRegressor(random_state=seed, fitted_ops=DEFAULT_FITTED + ("sign_shift",)).fit(X, y)
        assert model.formulas()[0] == "(x2 * sign(x0 - 0.7))"
        assert model.score(X, signal) > 0.995
        assert plain.score(X, signal) < 0.9
