"""Tests for constants fitted inside expressions (``fitted_ops``).

Covers the parametric operators themselves (evaluation, domain, keys,
pickling, display), the one-dimensional fit and its snapping, their use in
the search, and recovery of laws that need a constant inside the expression.
"""

from __future__ import annotations

import importlib.util
import pathlib
import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone

from beamfeat import BeamFeatClassifier, BeamFeatRegressor, BeamFeatTransformer, BeamSearch
from beamfeat._prediction import _undefined_rows
from beamfeat._rendering import _equation_list, _tree_size
from beamfeat.expression import (
    OPERATORS,
    Evaluator,
    ExclusionReason,
    Node,
    NodeError,
    OperatorSpec,
    UnitError,
    _node_setstate,
    combine,
    leaf,
    parametric_operators,
    pole_margins,
    transform,
)
from beamfeat.parametric import SNAP_VALUES, FittedConstant, _simplify, fit_constants, fit_constants_many
from beamfeat.scoring import CorrelationScorer
from beamfeat.search import DEFAULT_FITTED

FITTED = ("exp_scaled", "pow", "inv_shift", "log_shift")
ESTIMATORS = [BeamFeatTransformer, BeamFeatRegressor, BeamFeatClassifier]


@pytest.fixture
def ureg():
    pint = pytest.importorskip("pint")
    return pint.UnitRegistry()


@pytest.fixture
def rng():
    return np.random.default_rng(20261020)


def _standardised(y: np.ndarray) -> np.ndarray:
    return CorrelationScorer().prepare(y)


def _feynman_panel():
    """The panel's data generator, imported from ``benchmarks/``."""
    path = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "feynman_panel.py"
    if not path.exists():
        pytest.skip("benchmarks/feynman_panel.py is not available")
    spec = importlib.util.spec_from_file_location("feynman_panel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proportional(expression, law) -> bool:
    """Whether a fitted sympy expression is the law up to a nonzero constant."""
    import sympy

    names = {symbol.name: symbol for symbol in expression.free_symbols}
    law = law.subs({symbol: names.get(symbol.name, symbol) for symbol in law.free_symbols})
    ratio = sympy.simplify(expression / law)
    return not ratio.free_symbols and ratio != 0


# --------------------------------------------------------------------------- #
# Constant-free nodes are unchanged
# --------------------------------------------------------------------------- #


class TestConstantFreeNodesUnchanged:
    def test_keys_are_those_of_earlier_releases(self):
        a, b = leaf("a"), leaf("b")
        assert transform("log", combine("add", b, a)).key == "log(add(leaf:a,leaf:b))"
        assert combine("div", transform("sqrt", a), b).key == "div(sqrt(leaf:a),leaf:b)"
        assert leaf("a,leaf:b").key == "leaf:a\\,leaf:b"
        assert transform("exp", a).key == "exp(leaf:a)"

    def test_carry_no_constant(self):
        node = combine("mul", transform("square", leaf("a")), leaf("b"))
        assert all(sub.const is None for sub in node.subnodes())
        assert node.n_constants == 0
        assert _tree_size(node) == 4

    def test_plain_operators_refuse_a_constant(self):
        with pytest.raises(NodeError, match="takes no constant"):
            transform("log", leaf("a"), const=1.0)

    def test_parametric_operators_are_registered(self):
        assert parametric_operators() == sorted(FITTED)
        assert not any(OPERATORS[name].parametric for name in ("log", "exp", "add", "square"))

    def test_parametric_operator_must_be_unary(self):
        with pytest.raises(NodeError, match="must be unary"):
            OperatorSpec(
                name="bad", arity=2, fn=lambda x, y, a: x, formula=lambda x, y, a: x,
                constant_grid=lambda x: [np.linspace(0, 1, 5)],
            )


# --------------------------------------------------------------------------- #
# The operators
# --------------------------------------------------------------------------- #


class TestParametricOperators:
    @pytest.mark.parametrize(
        ("op", "constant", "reference"),
        [
            ("exp_scaled", -0.5, lambda x: np.exp(-0.5 * x)),
            ("pow", 1.5, lambda x: x**1.5),
            ("inv_shift", 1.0, lambda x: 1.0 / (x + 1.0)),
            ("log_shift", 0.25, lambda x: np.log(x + 0.25)),
        ],
    )
    def test_evaluation(self, rng, op, constant, reference):
        x = rng.uniform(0.5, 3.0, 50)
        node = transform(op, leaf("x"), const=constant)
        evaluator = Evaluator({"x": x})
        np.testing.assert_allclose(evaluator.evaluate(node), reference(x), rtol=1e-15)
        np.testing.assert_allclose(evaluator.transform_values(node), reference(x), rtol=1e-15)
        assert node.depth == 1 and node.const == constant and node.n_constants == 1

    def test_batch_evaluation_matches(self, rng):
        x = rng.uniform(0.5, 3.0, 40)
        evaluator = Evaluator({"x": x})
        nodes = [transform(op, leaf("x"), const=c) for op, c in zip(FITTED, (-0.7, 1.3, 2.0, 0.4))]
        batch = evaluator.plan_batch(nodes)
        outcome = batch.run()
        kept = evaluator.commit_batch(nodes, outcome)
        assert kept == nodes
        for column, node in enumerate(nodes):
            assert np.array_equal(outcome.values[:, column], Evaluator({"x": x}).evaluate(node))

    @pytest.mark.parametrize(
        ("op", "constant", "values"),
        [
            ("pow", 1.5, [-1.0, 2.0, 3.0]),
            ("log_shift", -2.0, [1.0, 2.0, 3.0]),
            ("inv_shift", -2.0, [1.0, 2.0, 3.0]),
            ("exp_scaled", 30.0, [1.0, 2.0, 3.0]),
        ],
    )
    def test_domain(self, op, constant, values):
        evaluator = Evaluator({"x": np.array(values)})
        assert evaluator.evaluate(transform(op, leaf("x"), const=constant)) is None
        (record,) = list(evaluator.log)
        assert record.reason is ExclusionReason.DOMAIN_ERROR

    def test_key_includes_constant(self):
        node = transform("exp_scaled", transform("square", leaf("x")), const=-0.5)
        assert node.key == "exp_scaled(square(leaf:x),const:-0.5)"
        same = transform("exp_scaled", transform("square", leaf("x")), const=-1 / 2)
        other = transform("exp_scaled", transform("square", leaf("x")), const=-0.25)
        assert node == same and hash(node) == hash(same)
        assert node != other
        third = transform("pow", leaf("x"), const=1 / 3)
        assert float(third.key.rsplit("const:", 1)[1].rstrip(")")) == 1 / 3

    def test_negative_zero_shares_a_key(self):
        a = transform("log_shift", leaf("x"), const=0.0)
        b = transform("log_shift", leaf("x"), const=-0.0)
        assert a == b

    def test_commutative_operands_order_by_constant(self):
        x = leaf("x")
        first = transform("exp_scaled", x, const=0.5)
        second = transform("exp_scaled", x, const=-0.5)
        assert combine("add", first, second) == combine("add", second, first)
        assert combine("add", first, second).name == combine("add", second, first).name

    def test_pickle_round_trip(self, rng):
        node = combine("mul", leaf("a"), transform("inv_shift", leaf("b"), const=0.731))
        restored = pickle.loads(pickle.dumps(node))
        assert restored == node and restored.key == node.key
        assert [child.const for child in restored.children if child.const is not None] == [0.731]
        data = {"a": rng.uniform(1, 2, 20), "b": rng.uniform(1, 2, 20)}
        assert np.array_equal(Evaluator(data).evaluate(restored), Evaluator(data).evaluate(node))

    def test_node_pickled_without_constant_field_loads(self):
        node = transform("log", leaf("x"))
        old_state = [node.op, node.children, node.column, node.unit, node.depth, node._key]
        restored = object.__new__(Node)
        _node_setstate(restored, old_state)
        assert restored == node and restored.const is None and restored.name == "log(x)"

    @pytest.mark.parametrize(
        ("op", "constant", "name"),
        [
            ("exp_scaled", -0.5, "exp(-0.5*x)"),
            ("exp_scaled", -1.0, "exp(-x)"),
            ("exp_scaled", 1 / 3, "exp((1/3)*x)"),
            ("pow", 1.5, "(x)^1.5"),
            ("pow", -0.5, "(x)^(-0.5)"),
            ("pow", -1 / 3, "(x)^(-1/3)"),
            ("inv_shift", 1.0, "1/(x + 1)"),
            ("inv_shift", -2.5, "1/(x - 2.5)"),
            ("log_shift", 0.4237, "log(x + 0.4237)"),
        ],
    )
    def test_names(self, op, constant, name):
        assert transform(op, leaf("x"), const=constant).name == name

    def test_sympy_and_latex(self):
        sympy = pytest.importorskip("sympy")
        gaussian = transform("exp_scaled", transform("square", leaf("x0")), const=-0.5)
        x0 = sympy.Symbol("x0", real=True)
        assert sympy.simplify(gaussian.to_sympy() - sympy.exp(-(x0**2) / 2)) == 0
        assert sympy.latex(gaussian.to_sympy()) == r"e^{- \frac{x_{0}^{2}}{2}}"
        shifted = transform("inv_shift", combine("mul", leaf("x0"), leaf("x1")), const=1.0)
        assert str(shifted.to_sympy()) == "1/(x0*x1 + 1)"
        third = transform("pow", leaf("x0"), const=1 / 3).to_sympy()
        assert third == x0 ** sympy.Rational(1, 3)
        fitted = transform("log_shift", leaf("x0"), const=0.4237).to_sympy()
        assert isinstance(fitted.args[0].args[0], sympy.Float)
        assert str(fitted) == "log(x0 + 0.4237)"

    @pytest.mark.parametrize(
        ("op", "constant", "plain"),
        [
            ("pow", 2.0, "square(leaf:x)"),
            ("pow", 3.0, "cube(leaf:x)"),
            ("pow", 0.5, "sqrt(leaf:x)"),
            ("pow", -1.0, "reciprocal(leaf:x)"),
            ("pow", 1.0, "leaf:x"),
            ("inv_shift", 0.0, "reciprocal(leaf:x)"),
            ("log_shift", 0.0, "log(leaf:x)"),
            ("exp_scaled", 1.0, "exp(leaf:x)"),
        ],
    )
    def test_plain_forms(self, op, constant, plain):
        node = transform(op, leaf("x"), const=constant)
        assert node.key == plain and node.n_constants == 0

    @pytest.mark.parametrize(("op", "constant"), [("pow", 0.0), ("exp_scaled", 0.0)])
    def test_degenerate_constant_rejected(self, op, constant):
        with pytest.raises(NodeError, match="does not depend"):
            transform(op, leaf("x"), const=constant)

    @pytest.mark.parametrize("constant", [None, float("nan"), float("inf"), "a"])
    def test_constant_required_and_finite(self, constant):
        with pytest.raises(NodeError):
            transform("pow", leaf("x"), const=constant)

    @pytest.mark.parametrize("op", ["exp_scaled", "pow", "log_shift"])
    def test_units_dimensionless_required(self, ureg, op):
        with pytest.raises(UnitError):
            transform(op, leaf("x", 1 * ureg.metre), const=0.7)
        assert transform(op, leaf("x", 1 * ureg.dimensionless), const=0.7).unit.dimensionless

    def test_units_inv_shift_reciprocal(self, ureg):
        node = transform("inv_shift", leaf("x", 1 * ureg.second), const=0.7)
        assert node.unit.dimensionality == (1 / ureg.second).dimensionality

    def test_complexity_counts_constants(self):
        gaussian = transform("exp_scaled", transform("square", leaf("x0")), const=-0.5)
        assert _tree_size(gaussian) == 4
        records = _equation_list([gaussian], ([0], [1.0], 1), [gaussian])
        assert records[0]["complexity"] == 5

    def test_undefined_rows_use_the_constant(self):
        node = transform("log_shift", leaf("x"), const=-1.5)
        mask = _undefined_rows(node, {"x": np.array([1.0, 2.0, 1.4, 3.0])}, 4)
        assert mask.tolist() == [True, False, True, False]


# --------------------------------------------------------------------------- #
# The one-dimensional fit
# --------------------------------------------------------------------------- #


class TestFitConstants:
    @pytest.mark.parametrize(
        ("op", "make_target", "expected"),
        [
            ("exp_scaled", lambda x: np.exp(-0.5 * x), -0.5),
            ("pow", lambda x: x**1.5, 1.5),
            ("pow", lambda x: x ** (1 / 3), 1 / 3),
            ("inv_shift", lambda x: 1.0 / (x + 1.0), 1.0),
            ("log_shift", lambda x: np.log(x + 0.25), 0.25),
        ],
    )
    def test_snaps_to_the_simple_constant(self, rng, op, make_target, expected):
        x = rng.uniform(0.5, 4.0, 300)
        signal = make_target(x)
        y = signal + rng.normal(0, 1e-3 * np.std(signal), 300)
        found = fit_constants(OPERATORS[op], x, _standardised(y))
        assert found == [FittedConstant(expected, None)]

    def test_constant_that_is_not_simple_is_rounded(self, rng):
        x = rng.uniform(0.5, 4.0, 300)
        found = fit_constants(OPERATORS["exp_scaled"], x, _standardised(np.exp(-0.37 * x)))
        assert found == [FittedConstant(-0.37, None)]

    def test_snap_values(self):
        values = [float(v) for v in SNAP_VALUES]
        assert values[0] == 0.0
        assert {-0.25, 0.25, -1 / 3, 1 / 3, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0, -3.0, 3.0} <= set(values)
        assert [v.denominator for v in SNAP_VALUES] == sorted(v.denominator for v in SNAP_VALUES)

    def test_linear_target_gives_no_constant(self, rng):
        # Every operator approaches an affine function of x at one end of its
        # range; a linear target therefore has nothing for a constant to fit.
        x = rng.uniform(0.5, 4.0, 300)
        y = 2 * x + rng.normal(0, 0.01, 300)
        for op in FITTED:
            assert fit_constants(OPERATORS[op], x, _standardised(y)) == []

    def test_fit_running_into_a_bound_is_rejected(self, rng):
        # exp(-20*x) needs a scale beyond the grid's 10 / max|x|; the fit
        # runs into the bound and is discarded rather than clipped.
        x = rng.uniform(0.5, 4.0, 300)
        assert fit_constants(OPERATORS["exp_scaled"], x, _standardised(np.exp(-20.0 * x))) == []
        assert fit_constants(OPERATORS["exp_scaled"], x, _standardised(np.exp(-2.0 * x))) == [
            FittedConstant(-2.0, None)
        ]

    def test_deterministic(self, rng):
        x = rng.uniform(0.5, 4.0, 300)
        y = np.exp(-0.43 * x) + rng.normal(0, 0.01, 300)
        first = fit_constants(OPERATORS["exp_scaled"], x, _standardised(y))
        assert first == fit_constants(OPERATORS["exp_scaled"], x.copy(), _standardised(y.copy()))

    def test_constant_fitted_as_a_factor(self, rng):
        X = rng.uniform(0.1, 1.0, (300, 2))
        total, product = X[:, 0] + X[:, 1], X[:, 0] * X[:, 1]
        y = total / (1 + product)
        partners = np.column_stack([total, product])
        found = fit_constants(OPERATORS["inv_shift"], product, _standardised(y), partners)
        assert FittedConstant(1.0, 0) in found

    @pytest.mark.parametrize("op", FITTED)
    def test_many_at_once_match_one_at_a_time(self, rng, op):
        # Operands of different shapes and scales, with and without partners:
        # fitting them together gives each exactly its own result.
        X = rng.uniform(0.2, 3.0, (300, 4))
        operands = np.column_stack([X[:, 0], X[:, 0] * X[:, 1], 5.0 * X[:, 2], X[:, 3] - 0.1, X[:, 1]])
        signal = np.exp(-0.6 * X[:, 0]) * X[:, 1] + 1.0 / (X[:, 2] + 0.5) + 3.0 * np.log(X[:, 3] + 0.3)
        y = _standardised(signal + rng.normal(0, 0.02, 300))
        partners = X[:, :2]
        with_partners = [True, False, True, True, False]
        together = fit_constants_many(OPERATORS[op], operands, y, partners, with_partners)
        alone = [
            fit_constants(OPERATORS[op], operands[:, j], y, partners if with_partners[j] else None)
            for j in range(operands.shape[1])
        ]
        assert together == alone
        assert any(together)

    def test_many_with_no_operands(self):
        assert fit_constants_many(OPERATORS["pow"], np.empty((20, 0)), np.zeros(20)) == []

    def test_plain_operator_rejected(self, rng):
        with pytest.raises(ValueError, match="no constant"):
            fit_constants(OPERATORS["log"], rng.uniform(1, 2, 10), np.zeros(10))


class TestPoleMargin:
    """A fitted pole keeps its distance from the data it was fitted on."""

    def test_margins(self):
        x = np.linspace(0.0, 1.0, 101)
        assert pole_margins(x) == pytest.approx((0.18, 0.18))
        # Ties at the minimum give no spread there; the floor applies.
        tied = np.r_[np.zeros(20), np.linspace(0.0, 2.0, 81)]
        below, above = pole_margins(tied)
        assert below == pytest.approx(0.02) and above == pytest.approx(0.45)
        # With fewer rows than the margin counts, all of them are used.
        assert pole_margins(np.array([1.0, 2.0, 4.0])) == pytest.approx((6.0, 6.0))
        assert pole_margins(np.array([3.0])) == pytest.approx((0.03, 0.03))

    @pytest.mark.parametrize("op", ["inv_shift", "log_shift"])
    def test_grid_starts_at_the_margin(self, rng, op):
        x = rng.uniform(1.0, 3.0, 200)
        below, above = pole_margins(x)
        segments = OPERATORS[op].constant_grid(x)
        assert segments[-1][0] == pytest.approx(below - x.min())
        if op == "inv_shift":
            assert segments[0][-1] == pytest.approx(-x.max() - above)

    @pytest.mark.parametrize(("op", "law"), [("log_shift", np.log), ("inv_shift", np.reciprocal)])
    def test_pole_at_the_edge_of_the_data_is_not_fitted(self, op, law):
        # A pole 0.06 below the smallest of 200 values on [1, 3] (3% of the
        # range): the curvature next to the edge rewards it on these rows,
        # and the first new row a little further out lands on it. The ten
        # lowest values spread over about 0.1, so the margin is about 0.2
        # and the fit runs into it.
        rng = np.random.default_rng(5)
        x = rng.uniform(1.0, 3.0, 200)
        y = law(x - 0.94) + rng.normal(0, 0.01, 200)
        assert fit_constants(OPERATORS[op], x, _standardised(y)) == []

    def test_snapping_does_not_carry_a_pole_into_the_data(self):
        # A column of six levels, where 1/(x - 1.5) fits far better than any
        # pole outside the data; the snap to -3/2 used to be accepted as an
        # improvement on the optimum, putting the pole between two levels.
        rng = np.random.default_rng(3)
        x = rng.choice(np.arange(6.0), 1500, p=[0.3, 0.1, 0.1, 0.2, 0.2, 0.1])
        y = rng.uniform(0.05, 0.6) / (x - 1.5) + 0.3 * x + rng.normal(0, 1.0, 1500)
        below, above = pole_margins(x)
        for constant, _ in fit_constants(OPERATORS["inv_shift"], x, _standardised(y)):
            pole = -constant
            assert pole <= x.min() - below or pole >= x.max() + above

    def test_simplify_respects_the_bounds(self):
        def flat(constant):
            return 0.5

        # Every value fits alike: a snap value without bounds, the optimum
        # rounded to one digit when the bounds exclude every snap value.
        assert _simplify(flat, 7.3, 0.5, 100) in {float(value) for value in SNAP_VALUES}
        assert _simplify(flat, 7.3, 0.5, 100, admissible=lambda c: c >= 5.0) == 7.0

    def test_no_nan_just_below_the_training_minimum(self):
        # The estimator-level regression: a fitted log(x0 - 0.94) returned
        # NaN for rows just below the training minimum of x0. Only the two
        # operators with a fitted shift are enabled: x^a is undefined below 0
        # whatever its constant, as sqrt(x) and log(x) are, and a feature
        # such as (log(x0))^(1/3) is NaN on these rows for that reason.
        rng = np.random.default_rng(0)
        X = np.column_stack([rng.uniform(1.0, 3.0, 400), rng.uniform(0.0, 1.0, 400)])
        y = np.log(X[:, 0] - 0.94) + 0.5 * X[:, 1] + rng.normal(0, 0.01, 400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BeamFeatRegressor(random_state=0, fitted_ops=("inv_shift", "log_shift")).fit(X, y)
            probe = np.column_stack([np.full(5, X[:, 0].min() - 0.07), np.linspace(0.1, 0.9, 5)])
            assert np.all(np.isfinite(model.predict(probe)))
        for node in (sub for feature in model.features_ for sub in feature.subnodes()):
            if node.op in ("inv_shift", "log_shift"):
                values = Evaluator(X).transform_values(node.children[0])
                below, above = pole_margins(values)
                pole = -node.const
                assert pole <= values.min() - below / 2 or pole >= values.max() + above / 2


class TestNegativePowers:
    """A negative power has its pole at 0, which keeps the margin from the data too."""

    def test_grid_keeps_negative_powers_only_off_the_pole(self):
        x = np.linspace(0.0, 1.0, 101)  # a margin of 0.18 at either end
        (far,) = OPERATORS["pow"].constant_grid(x + 0.19)
        (near,) = OPERATORS["pow"].constant_grid(x + 0.17)
        assert far[0] == -4.0 and far[-1] == 4.0 and 0.0 in far
        np.testing.assert_array_equal(near, far[far > 0])

    def test_not_fitted_where_the_data_reach_the_margin(self):
        # The communities regression in miniature: y = x0 * x1^(-0.127), with
        # x1 down to 0.02 and its ten lowest values spread so that the margin
        # is about 0.07. The fit used to find x1^(-0.126) as a factor of x0,
        # and a new row at x1 = 0 was then predicted as NaN.
        rng = np.random.default_rng(1)
        x0 = rng.uniform(0.0, 1.0, 400)
        x1 = np.r_[0.02, rng.uniform(0.02, 1.0, 399)]
        y = x0 * x1**-0.127 + rng.normal(0, 0.01, 400)
        assert x1.min() < pole_margins(x1)[0]
        assert fit_constants(OPERATORS["pow"], x1, _standardised(y), x0[:, None]) == []
        # The other operators here have no pole, so only a power could put one at 0.
        model = BeamFeatRegressor(
            random_state=0, unary_ops=("sqrt", "square"), binary_ops=("mul", "add", "sub"), fitted_ops=("pow",)
        ).fit(np.column_stack([x0, x1]), y)
        assert all(node.const > 0 for feature in model.features_ for node in feature.subnodes() if node.op == "pow")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.all(np.isfinite(model.predict(np.array([[0.0, 0.0], [0.5, 0.0]]))))

    def test_fitted_where_the_data_stay_off_the_pole(self):
        # The same law on x1 in [1, 3]: 0 lies far below the smallest value,
        # and the negative power is fitted, alone and as a factor of x0.
        rng = np.random.default_rng(2)
        x0 = rng.uniform(0.0, 1.0, 400)
        x1 = rng.uniform(1.0, 3.0, 400)
        y = x0 * x1**-0.5 + rng.normal(0, 0.01, 400)
        assert FittedConstant(-0.5, 0) in fit_constants(OPERATORS["pow"], x1, _standardised(y), x0[:, None])
        alone = x1**-0.5 + rng.normal(0, 1e-3, 400)
        assert fit_constants(OPERATORS["pow"], x1, _standardised(alone)) == [FittedConstant(-0.5, None)]
        model = BeamFeatRegressor(
            random_state=0, unary_ops=("sqrt", "square"), binary_ops=("mul", "add", "sub"), fitted_ops=("pow",)
        ).fit(np.column_stack([x0, x1]), y)
        assert model.formulas() == ["(x0 * (x1)^(-0.5))"]

    def test_communities_split_2(self):
        # The case that showed the gap: communities_p100, split 2 of the
        # high-dimensional study, with every fitted operator. The fit chose
        # x44 * x12^(-0.127), where the smallest training x12 is 0.02 and its
        # margin 0.06, and the test row with x12 = 0 (and x44 = 0) was NaN.
        from sklearn.model_selection import train_test_split

        path = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "additional_experiments"
        path = path / "data" / "communities_crime_numeric.csv"
        if not path.exists():
            pytest.skip("benchmarks/additional_experiments/data is not available")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        X_train, X_test, y_train, _ = train_test_split(data[:, :-1], data[:, -1], test_size=0.25, random_state=2)
        assert X_train[:, 12].min() < pole_margins(X_train[:, 12])[0] and 0.0 in X_test[:, 12]
        model = BeamFeatRegressor(random_state=0, fitted_ops=FITTED).fit(X_train, y_train)
        assert not any("(x12)^(-" in formula for formula in model.formulas())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.all(np.isfinite(model.predict(X_test)))


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class TestSearch:
    def test_unknown_fitted_op_lists_the_available_ones(self):
        with pytest.raises(ValueError, match=r"unknown fitted operator\(s\): \['exp'\]; available: \['exp_scaled'"):
            BeamSearch(fitted_ops=("exp",))

    def test_parametric_op_refused_as_plain_unary(self):
        with pytest.raises(ValueError, match="fitted_ops rather than unary_ops"):
            BeamSearch(unary_ops=("log", "pow"))

    def test_repeated_fitted_op_refused(self):
        with pytest.raises(ValueError, match="more than once"):
            BeamSearch(fitted_ops=("pow", "pow"))

    def test_single_name_accepted(self):
        assert BeamSearch(fitted_ops="pow").fitted_ops == ("pow",)

    def test_default_fits_nothing(self, rng):
        X = rng.uniform(1, 3, (200, 3))
        result = BeamSearch(max_depth=2, beam_width=10).run(X, X[:, 0] ** 1.5)
        assert all(record.n_fitted == 0 for record in result.trace)
        assert all(node.n_constants == 0 for node in result.nodes)

    @pytest.mark.parametrize("max_depth", [1, 2, 3])
    def test_depth_limit_respected(self, rng, max_depth):
        X = rng.uniform(0.5, 3, (200, 3))
        y = (X[:, 0] + X[:, 1]) / (1 + X[:, 0] * X[:, 1]) + np.exp(-X[:, 2])
        search = BeamSearch(max_depth=max_depth, beam_width=10, fitted_ops=FITTED)
        result = search.run(X, y)
        assert sum(record.n_fitted for record in result.trace) > 0
        assert all(node.depth <= max_depth for node in result.nodes)

    def test_depth_one_fits_the_best_columns(self, rng):
        X = rng.uniform(1, 3, (200, 5))
        y = X[:, 3] ** 1.5 + 0.3 * X[:, 1] + rng.normal(0, 0.01, 200)
        seen_parents = []

        class Recording(BeamSearch):
            def _fit(self, parents, values, prepared, seen, **kwargs):
                seen_parents.append([parent.name for parent in parents])
                return super()._fit(parents, values, prepared, seen, **kwargs)

        Recording(max_depth=1, beam_width=2, fitted_ops=("pow",)).run(X, y)
        assert seen_parents == [["x3", "x1"]]

    def test_unit_rejections_counted(self, ureg):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 3, (200, 2))
        units = {"x0": 1 * ureg.metre, "x1": 1 * ureg.second}
        y = 1 / (X[:, 0] + 1.0) + 1 / (X[:, 1] + 2.0)
        result = BeamSearch(max_depth=1, beam_width=5, fitted_ops=FITTED).run(X, y, units=units)
        plain = BeamSearch(max_depth=1, beam_width=5).run(X, y, units=units)
        # exp_scaled, pow and log_shift need dimensionless operands; both
        # columns carry units, so six fits are refused before any work.
        assert result.trace[1].n_rejected_units == plain.trace[1].n_rejected_units + 6
        assert result.trace[1].n_fitted > 0
        search = BeamSearch(max_depth=1, fitted_ops=FITTED)
        leaves = Evaluator(X, units=units).leaf_nodes()
        fitted, n_rejected = search._fit(leaves, X, _standardised(y), set())
        assert n_rejected == 6
        assert fitted and all(node.op == "inv_shift" for node in fitted)
        assert {str(node.unit.units) for node in fitted} <= {"1 / meter", "1 / second"}

    def test_extra_parents_are_bounded_and_outside_the_beam(self, rng):
        X = rng.uniform(0.5, 3, (200, 6))
        y = X[:, 0] * X[:, 1] + np.exp(-X[:, 2]) + rng.normal(0, 0.01, 200)
        calls = []

        class Recording(BeamSearch):
            def _fit(self, parents, values, prepared, seen, n_partners=None):
                calls.append((list(parents), values, n_partners))
                return super()._fit(parents, values, prepared, seen, n_partners)

        Recording(max_depth=3, beam_width=8, fitted_ops=FITTED).run(X, y)
        assert calls[0][2] is None and len(calls[0][0]) == 6
        evaluator = Evaluator(X)
        for parents, values, n_beam in calls[1:]:
            beam, extras = parents[:n_beam], parents[n_beam:]
            assert len(beam) <= 8 and len(extras) <= 8
            assert not set(beam) & set(extras) and len(set(extras)) == len(extras)
            for column, node in enumerate(parents):
                assert np.array_equal(values[:, column], evaluator.evaluate(node))

    def test_constant_free_candidate_not_masked_by_a_fitted_one(self, rng):
        x = leaf("x")
        values = rng.uniform(0.1, 1.0, 200)
        nodes = [transform("pow", x, const=1.69), transform("square", x)]
        columns = np.column_stack([values**1.69, values**2])
        scores = np.array([0.9999, 0.998])
        search = BeamSearch(beam_width=5)
        kept, *_ = search._prune_redundant(nodes, columns, scores, 5)
        assert kept == nodes[:1]
        kept, *_ = search._prune_redundant(nodes, columns, scores, 5, shield=True)
        assert kept == nodes
        # The other way round, a fitted candidate is still dropped.
        kept, *_ = search._prune_redundant(nodes[::-1], columns[:, ::-1], scores, 5, shield=True)
        assert kept == nodes[::-1][:1]

    def test_fitted_nodes_respect_max_inputs(self, rng):
        X = rng.uniform(0.1, 1.0, (300, 3))
        y = (X[:, 0] + X[:, 1]) / (1.0 + X[:, 0] * X[:, 1]) + 0.3 * np.exp(-X[:, 2])
        result = BeamSearch(max_depth=3, beam_width=20, fitted_ops=FITTED, max_inputs=2).run(
            X, y, columns=["x0", "x1", "x2"]
        )
        assert sum(trace.n_rejected_constraints for trace in result.trace) > 0
        assert any(node.n_constants for node in result.nodes)
        assert all(len(node.columns()) <= 2 for node in result.nodes)

    def test_n_jobs_does_not_change_the_result(self, rng, monkeypatch):
        import beamfeat.search as search_module

        monkeypatch.setattr(search_module, "MIN_PARALLEL_CANDIDATES", 1)
        X = rng.uniform(0.5, 3, (200, 4))
        y = X[:, 0] * np.exp(-X[:, 1]) + X[:, 2] ** 1.5
        runs = [
            BeamSearch(max_depth=2, beam_width=15, fitted_ops=FITTED, n_jobs=n_jobs, chunk_size=64).run(X, y)
            for n_jobs in (1, 2)
        ]
        assert runs[0].names == runs[1].names
        assert np.array_equal(runs[0].scores, runs[1].scores)


class TestConstantFreeCopies:
    """A fitted near copy of a law gives way to the law unless the data reject it.

    The swap at the end of the search uses the snapping test: the null that
    the copy with fewer constants fits as well, on the search rows, at level
    0.001."""

    @staticmethod
    def _final(x, y, nodes, kept):
        values = np.column_stack([Evaluator({"x0": x}).evaluate(node) for node in nodes])
        search = BeamSearch(fitted_ops=FITTED)
        final, _ = search._constant_free_copies(nodes, values, np.zeros(len(nodes)), kept, _standardised(y))
        return [node.name for node in final]

    def test_a_fitted_copy_gives_way_to_the_law(self, rng):
        x = rng.uniform(1, 3, 300)
        y = np.log(x) + 0.03 * rng.standard_normal(300)
        copy, law = transform("pow", leaf("x0"), const=-0.02), transform("log", leaf("x0"))
        assert self._final(x, y, [copy, law], [copy]) == ["log(x0)"]

    def test_the_fitted_law_stays_when_the_data_reject_the_copy(self, rng):
        x = rng.uniform(1, 3, 300)
        y = np.exp(-0.5 * x) + 1e-3 * rng.standard_normal(300)
        law, copy = transform("exp_scaled", leaf("x0"), const=-0.5), transform("reciprocal", leaf("x0"))
        assert abs(np.corrcoef(np.exp(-0.5 * x), 1 / x)[0, 1]) > 0.95   # a near copy, so the test decides
        assert self._final(x, y, [law, copy], [law]) == ["exp(-0.5*x0)"]

    def test_the_simplest_copy_is_taken(self, rng):
        x = rng.uniform(1, 3, 300)
        y = np.log(x) + 0.03 * rng.standard_normal(300)
        copy = transform("pow", leaf("x0"), const=-0.02)
        nested, law = transform("log", transform("sqrt", leaf("x0"))), transform("log", leaf("x0"))
        assert self._final(x, y, [copy, nested, law], [copy]) == ["log(x0)"]

    def test_a_copy_already_kept_is_not_taken_twice(self, rng):
        x = rng.uniform(1, 3, 300)
        y = np.log(x) + 0.03 * rng.standard_normal(300)
        copy, law = transform("pow", leaf("x0"), const=-0.02), transform("log", leaf("x0"))
        assert self._final(x, y, [copy, law], [copy, law]) == ["(x0)^(-0.02)", "log(x0)"]

    def test_a_copy_that_duplicates_another_member_is_passed_over(self, rng):
        x = rng.uniform(1, 3, 300)
        y = np.log(x) + 0.03 * rng.standard_normal(300)
        copy, law = transform("pow", leaf("x0"), const=-0.02), transform("log", leaf("x0"))
        twin = transform("log", transform("sqrt", leaf("x0")))   # log(x0) / 2, kept already
        assert self._final(x, y, [copy, law, twin], [copy, twin]) == ["(x0)^(-0.02)", "log(sqrt(x0))"]

    def test_not_consulted_without_fitted_ops(self, rng, monkeypatch):
        def consulted(*args, **kwargs):
            raise AssertionError("consulted")

        monkeypatch.setattr(BeamSearch, "_constant_free_copies", consulted)
        X = rng.uniform(1, 3, (200, 3))
        BeamSearch(max_depth=2, beam_width=10).run(X, np.log(X[:, 0]))


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #


class TestEstimators:
    @pytest.mark.parametrize("cls", ESTIMATORS)
    def test_parameter(self, cls):
        assert cls().get_params()["fitted_ops"] == DEFAULT_FITTED == ("exp_scaled", "pow", "inv_shift", "log_shift")
        assert clone(cls(fitted_ops=("pow",))).fitted_ops == ("pow",)

    @pytest.mark.parametrize("cls", ESTIMATORS)
    def test_unknown_name_raises_at_fit(self, cls, rng):
        X = rng.uniform(1, 2, (60, 2))
        y = (X[:, 0] > 1.5).astype(int)
        with pytest.raises(ValueError, match="available"):
            cls(fitted_ops=("shift",)).fit(X, y)

    def test_constant_is_reused_on_new_data(self, rng):
        X = rng.uniform(1, 5, (400, 1))
        y = X[:, 0] ** 1.5 + rng.normal(0, 1e-3, 400)
        model = BeamFeatRegressor(max_depth=1, beam_width=10, fitted_ops=("pow",)).fit(X, y)
        assert model.formulas() == ["(x0)^1.5"]
        X_new = rng.uniform(1, 5, (50, 1))
        np.testing.assert_allclose(model.predict(X_new), X_new[:, 0] ** 1.5, rtol=1e-3, atol=1e-3)
        restored = pickle.loads(pickle.dumps(model))
        assert np.array_equal(restored.predict(X_new), model.predict(X_new))
        assert model.equations_[0]["complexity"] == 4

    def test_latex_shows_exact_constant(self):
        pytest.importorskip("sympy")
        rng = np.random.default_rng(11)
        X = rng.uniform(0.1, 1.0, (400, 1))
        y = np.exp(-(X[:, 0] ** 2) / 2) + rng.normal(0, 1e-4, 400)
        model = BeamFeatRegressor(max_depth=2, beam_width=20, fitted_ops=("exp_scaled",)).fit(X, y)
        assert model.formulas() == ["exp(-0.5*(x0)^2)"]
        assert r"e^{- \frac{x_{0}^{2}}{2}}" in model.latex()

    def test_classifier(self, rng):
        X = rng.uniform(0.5, 3.0, (400, 2))
        y = (np.exp(-0.5 * X[:, 0] ** 2) > 0.3 * X[:, 1]).astype(int)
        model = BeamFeatClassifier(max_depth=2, beam_width=15, fitted_ops=FITTED).fit(X, y)
        assert model.score(X, y) > 0.9

    @pytest.mark.parametrize(
        ("law", "printed"),
        [(np.log, "log(x0)"), (np.sqrt, "sqrt(x0)"), (np.reciprocal, "1/(x0)"), (np.square, "(x0)^2")],
    )
    def test_a_simple_law_prints_as_itself(self, law, printed):
        # At 10% noise a fitted near copy, such as (x0)^(-0.02) for log(x0),
        # fits the search rows a little better than the law; it used to
        # displace the law and be printed.
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 3, (300, 3))
        signal = law(X[:, 0])
        y = signal + 0.1 * signal.std() * rng.standard_normal(300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BeamFeatRegressor(random_state=0).fit(X, y)
        assert model.formulas() == [printed]


class TestRecovery:
    """Laws a constant-free search cannot write, at the Feynman panel's settings.

    Data come from ``benchmarks/feynman_panel.py`` (0.1% noise, 500 rows,
    held-out probe), fitted with ``max_depth=3, beam_width=40,
    random_state=0`` as the panel does.
    """

    def _fit(self, name, **kwargs):
        panel = _feynman_panel()
        (law,) = [entry for entry in panel.equations() if entry[0] == name]
        X, X_test, y, y_test = panel.make_data(name, law[1], law[2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BeamFeatRegressor(max_depth=3, beam_width=40, random_state=0, **kwargs).fit(X, y)
        return model, model.score(X_test, y_test)

    def test_gaussian(self):
        sympy = pytest.importorskip("sympy")
        model, r2 = self._fit("gaussian", fitted_ops=("exp_scaled",))
        assert r2 > 0.999
        x0 = sympy.Symbol("x0")
        assert _proportional(model.to_sympy()[0], sympy.exp(-(x0**2) / 2))

    def test_gaussian_with_every_fitted_operator(self):
        sympy = pytest.importorskip("sympy")
        model, r2 = self._fit("gaussian", fitted_ops=FITTED)
        assert r2 > 0.999
        x0 = sympy.Symbol("x0")
        assert _proportional(model.to_sympy()[0], sympy.exp(-(x0**2) / 2))

    @pytest.mark.parametrize("fitted_ops", [("inv_shift",), FITTED])
    def test_relativistic_velocity(self, fitted_ops):
        sympy = pytest.importorskip("sympy")
        model, r2 = self._fit("relativistic_velocity", fitted_ops=fitted_ops)
        assert r2 > 0.999
        x0, x1 = sympy.symbols("x0 x1")
        assert _proportional(model.to_sympy()[0], (x0 + x1) / (1 + x0 * x1))

    def test_relativistic_denominator_comes_from_a_dropped_candidate(self, monkeypatch):
        # On this domain x0*x1 and x0 + x1 correlate at |r| = 0.95, the
        # default redundancy_threshold, and with the panel's split x0*x1 is
        # dropped from the depth-1 beam as a near-duplicate of x0 + x1. It is
        # still among the depth's best candidates, so it is fitted on.
        product = combine("mul", leaf("x0"), leaf("x1"))
        calls = []
        fit = BeamSearch._fit

        def recording(self, parents, values, prepared, seen, n_partners=None):
            calls.append((list(parents), n_partners))
            return fit(self, parents, values, prepared, seen, n_partners)

        monkeypatch.setattr(BeamSearch, "_fit", recording)
        model, _ = self._fit("relativistic_velocity", fitted_ops=("inv_shift",))
        parents, n_beam = calls[1]
        assert product not in parents[:n_beam]
        assert product in parents[n_beam:]
        assert model.formulas()[0] == "((x0 + x1) * 1/((x0 * x1) + 1))"

    @pytest.mark.parametrize(
        ("make_target", "law"),
        [
            (lambda X: X[:, 0] ** 1.5, "x0**(3/2)"),
            (lambda X: X[:, 0] * X[:, 1] ** 1.5, "x0*x1**(3/2)"),
            (lambda X: X[:, 0] ** (1 / 3) * X[:, 1], "x0**(1/3)*x1"),
        ],
    )
    def test_power_law(self, make_target, law):
        sympy = pytest.importorskip("sympy")
        rng = np.random.default_rng(7)
        n_cols = 1 if "x1" not in law else 2
        X, X_test = rng.uniform(1, 5, (500, n_cols)), rng.uniform(1, 5, (500, n_cols))
        signal = make_target(X)
        y = signal + rng.normal(0, 1e-3 * np.std(signal), 500)
        model = BeamFeatRegressor(max_depth=3, beam_width=40, random_state=0, fitted_ops=("pow",)).fit(X, y)
        assert model.score(X_test, make_target(X_test)) > 0.999
        assert len(model.formulas()) == 1
        assert _proportional(model.to_sympy()[0], sympy.sympify(law))

    def test_additive_logarithm_is_not_found(self):
        # y = x0*x1/x2 + 0.5*log(x3): the logarithm carries 0.6% of the
        # target's variance (|corr| 0.08 with y). It is representable without
        # any constant, and survives the depth-1 beam, but the search's final
        # ranking scores every candidate against the raw target and keeps the
        # best max_features (here beam_width=50); log(x3) ranks below fifty
        # restatements of x0*x1/x2. Fitted constants do not change that:
        # log_shift(x3) has the same small correlation. Recorded as current
        # behaviour, with and without fitted_ops.
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 5, (500, 4))
        signal = X[:, 0] * X[:, 1] / X[:, 2] + 0.5 * np.log(X[:, 3])
        y = signal + rng.normal(0, 1e-3 * np.std(signal), 500)
        for fitted_ops in ((), FITTED):
            model = BeamFeatRegressor(random_state=0, fitted_ops=fitted_ops).fit(X, y)
            assert model.score(X, signal) > 0.99
            assert not any("log(x3" in name for name in model.search_result_.names)
            assert not any("log(x3" in formula for formula in model.formulas())


class TestNullTargets:
    def test_no_discoveries_on_noise(self):
        # Constants are fitted on the search rows only; the screen on the
        # held-out rows sees fixed expressions, so a pure-noise target should
        # give no more discoveries than without them.
        hits = 0
        for seed in range(4):
            rng = np.random.default_rng(1000 + seed)
            X = rng.uniform(0.5, 5.0, (200, 5))
            y = rng.normal(size=200)
            model = BeamFeatRegressor(random_state=seed, fitted_ops=FITTED).fit(X, y)
            hits += model.selection_result_.n_selected > 0
        assert hits == 0
