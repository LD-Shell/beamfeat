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
from beamfeat.estimators import _equation_list, _tree_size, _undefined_rows
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
    transform,
)
from beamfeat.parametric import SNAP_VALUES, FittedConstant, fit_constants
from beamfeat.scoring import CorrelationScorer

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

    def test_plain_operator_rejected(self, rng):
        with pytest.raises(ValueError, match="no constant"):
            fit_constants(OPERATORS["log"], rng.uniform(1, 2, 10), np.zeros(10))


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
            def _fit(self, parents, values, prepared, seen):
                seen_parents.append([parent.name for parent in parents])
                return super()._fit(parents, values, prepared, seen)

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

    @pytest.mark.parametrize(
        ("op", "constant", "box", "safe"),
        [
            ("inv_shift", 1.0, (0.1, 1.0), True),
            ("inv_shift", -0.5, (0.1, 1.0), False),
            ("log_shift", 0.5, (-0.4, 1.0), True),
            ("log_shift", 0.5, (-0.6, 1.0), False),
            ("pow", 1.5, (0.1, 5.0), True),
            ("pow", 1.5, (-0.1, 5.0), False),
            ("exp_scaled", -0.5, (-90.0, 5.0), True),
            ("exp_scaled", -0.5, (-110.0, 5.0), False),
        ],
    )
    def test_safe_domain_bounds_fitted_nodes(self, op, constant, box, safe):
        search = BeamSearch(safe_domain={"x": box})
        search._box = {"x": box}
        assert search._allowed(transform(op, leaf("x"), const=constant)) is safe

    def test_fitted_nodes_respect_the_constraints(self, rng):
        X = rng.uniform(0.1, 1.0, (300, 3))
        y = (X[:, 0] + X[:, 1]) / (1.0 + X[:, 0] * X[:, 1]) + 0.3 * np.exp(-X[:, 2])
        result = BeamSearch(
            max_depth=3, beam_width=20, fitted_ops=FITTED, max_inputs=2, safe_domain={"x0": (-1.0, 1.0)}
        ).run(X, y, columns=["x0", "x1", "x2"])
        assert sum(trace.n_rejected_constraints for trace in result.trace) > 0
        assert all(len(node.columns()) <= 2 for node in result.nodes)
        probe = {"x0": rng.uniform(-1.0, 1.0, 20_000), "x1": rng.uniform(0.1, 1.0, 20_000), "x2": rng.uniform(0.1, 1.0, 20_000)}
        evaluator = Evaluator(probe)
        for node in result.nodes:
            with np.errstate(all="ignore"):
                values = evaluator.evaluate(node, apply_filters=False)
            assert values is not None and np.all(np.isfinite(values)), node.name

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


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #


class TestEstimators:
    @pytest.mark.parametrize("cls", ESTIMATORS)
    def test_parameter(self, cls):
        assert cls().get_params()["fitted_ops"] == ()
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

    def test_relativistic_velocity(self):
        # On this domain x0*x1 and x0 + x1 correlate at |r| = 0.95, the
        # default redundancy_threshold, and with the panel's split x0*x1 is
        # dropped at depth 1 as a near-duplicate of x0 + x1: the denominator
        # is then never built. A slightly higher threshold keeps both.
        sympy = pytest.importorskip("sympy")
        model, r2 = self._fit("relativistic_velocity", fitted_ops=("inv_shift",), redundancy_threshold=0.97)
        assert r2 > 0.999
        x0, x1 = sympy.symbols("x0 x1")
        assert _proportional(model.to_sympy()[0], (x0 + x1) / (1 + x0 * x1))

    def test_relativistic_velocity_at_the_default_threshold(self):
        # The honest outcome at the panel's own settings, recorded so that a
        # change to it is noticed: not recovered, for the reason above.
        sympy = pytest.importorskip("sympy")
        model, _ = self._fit("relativistic_velocity", fitted_ops=("inv_shift",))
        x0, x1 = sympy.symbols("x0 x1")
        assert not _proportional(model.to_sympy()[0], (x0 + x1) / (1 + x0 * x1))

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
