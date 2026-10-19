"""Tests for the expression DAG: structural identity, evaluation, units, exclusion."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from beamfeat.expression import (
    EvaluationLog,
    Evaluator,
    ExclusionReason,
    Node,
    NodeError,
    OperatorSpec,
    UnitError,
    UnitRegistryError,
    combine,
    leaf,
    transform,
)

# pint is an optional dependency: only the unit-aware tests need it. A
# module-level importorskip would silently skip every test in this file when
# pint is absent, hollowing out the suite; instead the `ureg` fixture skips,
# so only tests that actually request units are affected.


@pytest.fixture
def ureg():
    pint = pytest.importorskip("pint")
    return pint.UnitRegistry()


@pytest.fixture
def rng():
    return np.random.default_rng(20260723)


@pytest.fixture
def data(rng):
    """Well-conditioned positive data, safe for log/sqrt/reciprocal."""
    n = 200
    return {
        "a": rng.uniform(1.0, 5.0, n),
        "b": rng.uniform(1.0, 5.0, n),
        "c": rng.uniform(1.0, 5.0, n),
    }


# --------------------------------------------------------------------------- #
# Structural identity
# --------------------------------------------------------------------------- #


class TestStructuralIdentity:
    def test_identical_leaves_unify(self):
        assert leaf("a") == leaf("a")
        assert hash(leaf("a")) == hash(leaf("a"))

    def test_distinct_leaves_differ(self):
        assert leaf("a") != leaf("b")

    def test_independently_built_expressions_unify(self):
        left = transform("log", leaf("a"))
        right = transform("log", leaf("a"))
        assert left == right
        assert hash(left) == hash(right)

    def test_commutative_operand_order_unifies(self):
        assert combine("add", leaf("a"), leaf("b")) == combine("add", leaf("b"), leaf("a"))
        assert combine("mul", leaf("a"), leaf("b")) == combine("mul", leaf("b"), leaf("a"))

    def test_noncommutative_operand_order_matters(self):
        assert combine("sub", leaf("a"), leaf("b")) != combine("sub", leaf("b"), leaf("a"))
        assert combine("div", leaf("a"), leaf("b")) != combine("div", leaf("b"), leaf("a"))

    def test_deep_commutative_canonicalisation(self):
        """Canonicalisation must hold when operands are themselves expressions."""
        left = combine("mul", transform("log", leaf("a")), transform("sqrt", leaf("b")))
        right = combine("mul", transform("sqrt", leaf("b")), transform("log", leaf("a")))
        assert left == right

    def test_nodes_usable_as_dict_keys(self):
        pool = {combine("add", leaf("a"), leaf("b")): "first"}
        pool[combine("add", leaf("b"), leaf("a"))] = "second"
        assert len(pool) == 1
        assert pool[combine("add", leaf("a"), leaf("b"))] == "second"

    def test_node_is_immutable(self):
        node = leaf("a")
        with pytest.raises((AttributeError, TypeError)):
            node.op = "mul"

    def test_key_is_deterministic(self):
        """Keys must be stable, not tied to object identity or address."""
        assert leaf("a").key == leaf("a").key
        expr = combine("mul", leaf("a"), leaf("b"))
        assert expr.key == combine("mul", leaf("a"), leaf("b")).key

    def test_equality_with_non_node_returns_false(self):
        assert leaf("a") != "a"
        assert leaf("a") is not None

    def test_key_separators_in_column_names_cannot_merge_nodes(self):
        """Unescaped, both keys read ``sub(leaf:a,leaf:b,leaf:c)``."""
        left = combine("sub", leaf("a,leaf:b"), leaf("c"))
        right = combine("sub", leaf("a"), leaf("b,leaf:c"))
        assert left != right
        assert left.key != right.key

    def test_key_separators_do_not_share_cache_entries(self):
        columns = {
            "a": np.array([1.0, 2.0, 3.0, 4.0]),
            "c": np.array([0.5, 0.1, 0.9, 0.3]),
            "a,leaf:b": np.array([10.0, 30.0, 20.0, 40.0]),
            "b,leaf:c": np.array([7.0, 3.0, 5.0, 1.0]),
        }
        ev = Evaluator(columns)
        left = combine("sub", leaf("a,leaf:b"), leaf("c"))
        right = combine("sub", leaf("a"), leaf("b,leaf:c"))
        np.testing.assert_allclose(ev.evaluate(left), columns["a,leaf:b"] - columns["c"])
        np.testing.assert_allclose(ev.evaluate(right), columns["a"] - columns["b,leaf:c"])

    def test_plain_keys_are_unchanged(self):
        assert combine("mul", transform("log", leaf("a")), leaf("b")).key == "mul(leaf:b,log(leaf:a))"

    def test_escaping_keeps_operand_order(self):
        """Escaped, ``f(x)`` would sort after ``fA``; operand order, and so
        the printed formula, must not depend on the escaping."""
        assert combine("add", leaf("fA"), leaf("f(x)")).name == "(f(x) + fA)"
        assert combine("add", leaf("f(x)"), leaf("fA")).name == "(f(x) + fA)"


class TestStructure:
    def test_leaf_properties(self):
        node = leaf("a")
        assert node.is_leaf
        assert node.depth == 0
        assert node.columns() == {"a"}

    def test_depth_accumulates(self):
        assert transform("log", leaf("a")).depth == 1
        assert transform("sqrt", transform("log", leaf("a"))).depth == 2

    def test_depth_is_max_of_children(self):
        deep = transform("sqrt", transform("log", leaf("a")))
        node = combine("mul", deep, leaf("b"))
        assert node.depth == 3

    def test_shared_subexpressions_counted_once(self):
        """The DAG must not double-count a node reachable by two paths."""
        shared = transform("log", leaf("a"))
        node = combine("div", transform("sqrt", shared), transform("square", shared))
        # leaf a, log(a), sqrt(log(a)), square(log(a)), div(...) == 5
        assert node.size == 5

    def test_columns_deduplicates(self):
        node = combine("div", transform("log", leaf("a")), transform("sqrt", leaf("a")))
        assert node.columns() == {"a"}

    def test_formula_rendering(self):
        assert leaf("a").name == "a"
        assert transform("log", leaf("a")).name == "log(a)"
        assert combine("mul", leaf("a"), leaf("b")).name == "(a * b)"

    def test_to_sympy_roundtrip(self):
        sympy = pytest.importorskip("sympy")
        node = combine("mul", leaf("a"), leaf("b"))
        expr = node.to_sympy()
        assert expr == sympy.Symbol("a", real=True) * sympy.Symbol("b", real=True)

    def test_to_sympy_sanitises_awkward_names(self):
        pytest.importorskip("sympy")
        expr = leaf("3 weird-name!").to_sympy()
        assert expr.name[0].isalpha() or expr.name[0] == "_"

    def test_to_sympy_keeps_columns_with_alike_names_apart(self):
        """Names that sanitise to the same identifier must stay distinct
        symbols, or a sum of two columns collapses to twice one of them."""
        pytest.importorskip("sympy")
        expr = combine("add", leaf("a b"), leaf("a-b")).to_sympy()
        assert len(expr.free_symbols) == 2
        expr = combine("sub", leaf("a b"), leaf("a_b")).to_sympy()
        assert {s.name for s in expr.free_symbols} == {"a_b", "a_b_2"}
        assert expr.subs({s: 1.0 for s in expr.free_symbols}) == 0

    def test_to_sympy_accepts_a_symbol_mapping(self):
        sympy = pytest.importorskip("sympy")
        p, q = sympy.symbols("p q", positive=True)
        expr = combine("mul", leaf("a"), leaf("b")).to_sympy({"a": p, "b": q})
        assert expr == p * q


class TestConstructorValidation:
    def test_unknown_operator_rejected(self):
        with pytest.raises(NodeError, match="unknown operator"):
            transform("nonexistent", leaf("a"))

    def test_arity_mismatch_rejected(self):
        with pytest.raises(NodeError, match="expects"):
            transform("mul", leaf("a"))
        with pytest.raises(NodeError, match="expects"):
            combine("log", leaf("a"), leaf("b"))

    def test_self_combination_rejected(self):
        """x - x and x / x are degenerate; reject at construction."""
        with pytest.raises(NodeError, match="itself"):
            combine("sub", leaf("a"), leaf("a"))

    def test_empty_column_name_rejected(self):
        with pytest.raises(NodeError):
            leaf("")

    def test_non_node_operand_rejected(self):
        with pytest.raises(NodeError):
            transform("log", "a")
        with pytest.raises(NodeError):
            combine("mul", leaf("a"), "b")

    def test_duplicate_operator_registration_rejected(self):
        from beamfeat.expression import register_operator

        with pytest.raises(NodeError, match="already registered"):
            register_operator(OperatorSpec(name="log", arity=1, fn=np.log, formula=lambda x: x))

    def test_bad_arity_spec_rejected(self):
        with pytest.raises(NodeError, match="arity"):
            OperatorSpec(name="ternary", arity=3, fn=lambda x: x, formula=lambda x: x)

    def test_commutative_unary_rejected(self):
        with pytest.raises(NodeError, match="commutative"):
            OperatorSpec(name="odd", arity=1, fn=lambda x: x, formula=lambda x: x, commutative=True)


# --------------------------------------------------------------------------- #
# Unit propagation
# --------------------------------------------------------------------------- #


class TestUnits:
    def test_multiplication_multiplies_units(self, ureg):
        force = leaf("F", 1.0 * ureg.newton)
        distance = leaf("d", 1.0 * ureg.meter)
        work = combine("mul", force, distance)
        assert work.unit.dimensionality == (1.0 * ureg.joule).dimensionality

    def test_division_divides_units(self, ureg):
        distance = leaf("d", 1.0 * ureg.meter)
        time = leaf("t", 1.0 * ureg.second)
        speed = combine("div", distance, time)
        assert speed.unit.dimensionality == (1.0 * ureg.meter / ureg.second).dimensionality

    def test_addition_requires_matching_dimensions(self, ureg):
        mass = leaf("m", 1.0 * ureg.kilogram)
        length = leaf("l", 1.0 * ureg.meter)
        with pytest.raises(UnitError, match="matching dimensions"):
            combine("add", mass, length)

    def test_addition_allows_matching_dimensions(self, ureg):
        a = leaf("a", 1.0 * ureg.meter)
        b = leaf("b", 1.0 * ureg.kilometer)
        assert combine("add", a, b).unit is not None

    def test_subtraction_requires_matching_dimensions(self, ureg):
        with pytest.raises(UnitError):
            combine("sub", leaf("m", 1.0 * ureg.kilogram), leaf("l", 1.0 * ureg.meter))

    def test_log_requires_dimensionless(self, ureg):
        with pytest.raises(UnitError, match="dimensionless"):
            transform("log", leaf("m", 1.0 * ureg.meter))

    def test_exp_requires_dimensionless(self, ureg):
        with pytest.raises(UnitError, match="dimensionless"):
            transform("exp", leaf("m", 1.0 * ureg.meter))

    def test_log_allows_dimensionless_ratio(self, ureg):
        ratio = combine("div", leaf("a", 1.0 * ureg.meter), leaf("b", 1.0 * ureg.meter))
        assert transform("log", ratio) is not None

    def test_square_squares_units(self, ureg):
        side = leaf("s", 1.0 * ureg.meter)
        area = transform("square", side)
        assert area.unit.dimensionality == (1.0 * ureg.meter**2).dimensionality

    def test_sqrt_halves_units(self, ureg):
        area = leaf("A", 1.0 * ureg.meter**2)
        assert transform("sqrt", area).unit.dimensionality == (1.0 * ureg.meter).dimensionality

    def test_reciprocal_inverts_units(self, ureg):
        time = leaf("t", 1.0 * ureg.second)
        assert transform("reciprocal", time).unit.dimensionality == (1.0 / ureg.second).dimensionality

    def test_unitless_nodes_impose_no_constraints(self):
        """Without units, every combination is permitted."""
        assert combine("add", leaf("a"), leaf("b")) is not None
        assert transform("log", leaf("a")) is not None

    def test_units_rejected_before_any_evaluation(self, ureg, data):
        """The failure must occur at construction, not at evaluate time."""
        units = {"a": 1.0 * ureg.kilogram, "b": 1.0 * ureg.meter}
        ev = Evaluator(data, units=units)
        nodes = ev.leaf_nodes()
        by_name = {n.column: n for n in nodes}
        with pytest.raises(UnitError):
            combine("add", by_name["a"], by_name["b"])
        assert len(ev.log) == 0

    def test_magnitude_stripped_from_units(self, ureg):
        """Units carry dimensionality only; magnitude must not accumulate."""
        node = leaf("a", 5.0 * ureg.meter)
        assert node.unit.magnitude == pytest.approx(1.0)

    def test_mixed_registries_raise_a_unit_registry_error(self, ureg):
        """pint's own error is a plain ValueError, which reads as an ordinary
        construction failure; the specific error names the cause."""
        other = type(ureg)()
        mass, length = leaf("m", 1.0 * ureg.kilogram), leaf("l", 1.0 * other.meter)
        with pytest.raises(UnitRegistryError, match="different pint registries"):
            combine("mul", mass, length)
        with pytest.raises(UnitRegistryError, match="different pint registries"):
            combine("div", mass, length)

    def test_evaluator_unifies_registries(self, ureg, data):
        other = type(ureg)()
        ev = Evaluator(data, units={"a": 1.0 * ureg.kilogram, "b": other.meter, "c": 2.0 * other.second})
        a, b, c = ev.leaf_nodes()
        product = combine("div", combine("mul", a, b), c)
        assert product.unit.dimensionality == (ureg.kilogram * ureg.meter / ureg.second).dimensionality

    def test_unconvertible_unit_from_another_registry_raises(self, ureg, data):
        other = type(ureg)()
        other.define("widget = 3 * meter")
        with pytest.raises(UnitRegistryError, match="widget"):
            Evaluator(data, units={"a": 1.0 * ureg.kilogram, "b": 1.0 * other.widget})


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


class TestEvaluation:
    def test_leaf_returns_source_column(self, data):
        ev = Evaluator(data)
        np.testing.assert_allclose(ev.evaluate(leaf("a")), data["a"])

    def test_transform_matches_numpy(self, data):
        ev = Evaluator(data)
        np.testing.assert_allclose(ev.evaluate(transform("log", leaf("a"))), np.log(data["a"]))

    def test_combination_matches_numpy(self, data):
        ev = Evaluator(data)
        node = combine("mul", leaf("a"), leaf("b"))
        np.testing.assert_allclose(ev.evaluate(node), data["a"] * data["b"])

    def test_nested_expression_matches_numpy(self, data):
        ev = Evaluator(data)
        node = combine("div", combine("mul", leaf("a"), leaf("b")), transform("sqrt", leaf("c")))
        expected = (data["a"] * data["b"]) / np.sqrt(data["c"])
        np.testing.assert_allclose(ev.evaluate(node), expected)

    def test_commutative_nodes_give_identical_results(self, data):
        ev = Evaluator(data)
        left = ev.evaluate(combine("add", leaf("a"), leaf("b")))
        right = ev.evaluate(combine("add", leaf("b"), leaf("a")))
        np.testing.assert_array_equal(left, right)

    def test_accepts_2d_array_input(self, rng):
        matrix = rng.uniform(1.0, 5.0, (50, 3))
        ev = Evaluator(matrix, columns=["p", "q", "r"])
        assert ev.columns == ["p", "q", "r"]
        np.testing.assert_allclose(ev.evaluate(leaf("p")), matrix[:, 0])

    def test_default_column_names_for_arrays(self, rng):
        ev = Evaluator(rng.uniform(1, 5, (10, 2)))
        assert ev.columns == ["x0", "x1"]

    def test_column_count_mismatch_rejected(self, rng):
        with pytest.raises(ValueError, match="column names"):
            Evaluator(rng.uniform(1, 5, (10, 3)), columns=["only", "two"])

    def test_ragged_columns_rejected(self):
        with pytest.raises(ValueError, match="differing lengths"):
            Evaluator({"a": np.ones(10), "b": np.ones(5)})

    def test_missing_column_raises(self, data):
        ev = Evaluator(data)
        with pytest.raises(KeyError, match="absent"):
            ev.evaluate(leaf("absent"))

    def test_evaluate_many_returns_aligned_matrix(self, data):
        ev = Evaluator(data)
        nodes = [leaf("a"), combine("mul", leaf("a"), leaf("b")), transform("log", leaf("c"))]
        kept, matrix = ev.evaluate_many(nodes)
        assert len(kept) == matrix.shape[1] == 3
        np.testing.assert_allclose(matrix[:, 1], data["a"] * data["b"])

    def test_evaluate_many_drops_failures(self, data):
        ev = Evaluator(data)
        bad = transform("log", combine("sub", leaf("a"), leaf("b")))  # negatives present
        kept, matrix = ev.evaluate_many([leaf("a"), bad])
        assert len(kept) == 1
        assert matrix.shape[1] == 1

    def test_evaluate_many_handles_total_failure(self, data):
        ev = Evaluator(data)
        bad = transform("log", combine("sub", leaf("a"), leaf("b")))
        kept, matrix = ev.evaluate_many([bad])
        assert kept == []
        assert matrix.shape == (ev.n_rows, 0)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


class TestCaching:
    def test_repeated_evaluation_hits_cache(self, data):
        ev = Evaluator(data)
        node = combine("mul", leaf("a"), leaf("b"))
        first = ev.evaluate(node)
        assert ev.evaluate(node) is first  # identical object, not just equal

    def test_shared_subexpression_computed_once(self, data):
        """The DAG's whole point: a shared child is evaluated a single time."""
        calls = {"n": 0}
        original = np.log

        def counting_log(x):
            calls["n"] += 1
            return original(x)

        from beamfeat.expression import OPERATORS

        spec = OPERATORS["log"]
        object.__setattr__(spec, "fn", counting_log)
        try:
            ev = Evaluator(data)
            shared = transform("log", leaf("a"))
            ev.evaluate(combine("mul", shared, leaf("b")))
            ev.evaluate(combine("mul", shared, leaf("c")))
            assert calls["n"] == 1
        finally:
            object.__setattr__(spec, "fn", original)

    def test_cache_respects_capacity(self, data):
        ev = Evaluator(data, cache_size=2)
        for op in ("log", "sqrt", "square"):
            ev.evaluate(transform(op, leaf("a")))
        assert ev.cache_info()["size"] <= 2

    def test_eviction_preserves_correctness(self, data):
        """An evicted node must recompute to the same values."""
        ev = Evaluator(data, cache_size=1)
        node = combine("mul", leaf("a"), leaf("b"))
        first = ev.evaluate(node).copy()
        for op in ("log", "sqrt", "square"):
            ev.evaluate(transform(op, leaf("c")))
        np.testing.assert_allclose(ev.evaluate(node), first)

    def test_clear_cache_preserves_leaves(self, data):
        ev = Evaluator(data)
        ev.evaluate(combine("mul", leaf("a"), leaf("b")))
        ev.clear_cache()
        assert ev.cache_info()["size"] == 0
        np.testing.assert_allclose(ev.evaluate(leaf("a")), data["a"])

    def test_zero_cache_size_rejected(self, data):
        with pytest.raises(ValueError, match="cache_size"):
            Evaluator(data, cache_size=0)

    def test_unfiltered_result_is_rechecked_with_filters(self):
        """A value cached by transform_values has not been through the
        filters; a later filtered evaluation must still apply them."""
        ev = Evaluator({"a": np.array([2.0, 3.0, 4.0]), "b": np.array([3.0, 4.0, 5.0])})
        constant = combine("sub", leaf("b"), leaf("a"))
        assert ev.transform_values(constant) is not None
        assert ev.evaluate(constant) is None
        assert ev.log.by_reason(ExclusionReason.ZERO_VARIANCE)

    def test_filtered_result_serves_unfiltered_requests(self, data):
        ev = Evaluator(data)
        node = combine("mul", leaf("a"), leaf("b"))
        first = ev.evaluate(node)
        assert ev.transform_values(node) is first

    def test_returned_arrays_are_read_only(self, data):
        """Callers share the cached array; writing to it would corrupt every
        later evaluation that reuses it."""
        ev = Evaluator(data)
        node = combine("mul", leaf("a"), leaf("b"))
        for values in (ev.evaluate(node), ev.evaluate(leaf("a")), ev.transform_values(node)):
            with pytest.raises(ValueError, match="read-only"):
                values[0] = 0.0
        assert data["a"].flags.writeable  # the caller's own array is untouched


# --------------------------------------------------------------------------- #
# Exclusion: recorded, never silent
# --------------------------------------------------------------------------- #


class TestExclusion:
    def test_log_of_negative_is_recorded(self, data):
        ev = Evaluator(data)
        node = transform("log", combine("sub", leaf("a"), leaf("b")))
        assert ev.evaluate(node) is None
        assert len(ev.log) == 1
        assert ev.log.by_reason(ExclusionReason.DOMAIN_ERROR)

    def test_division_by_zero_is_recorded(self):
        ev = Evaluator({"a": np.array([1.0, 2.0, 3.0]), "z": np.array([1.0, 0.0, 2.0])})
        assert ev.evaluate(combine("div", leaf("a"), leaf("z"))) is None
        assert len(ev.log) == 1

    def test_reciprocal_of_zero_is_recorded(self):
        ev = Evaluator({"z": np.array([1.0, 0.0, 2.0]), "a": np.array([1.0, 2.0, 3.0])})
        assert ev.evaluate(transform("reciprocal", leaf("z"))) is None
        assert ev.log.by_reason(ExclusionReason.DOMAIN_ERROR)

    def test_overflow_is_recorded_not_silent(self):
        """The failure mode autofeat masks: overflow must surface in the log.
        The column itself is admissible; only its cube overflows."""
        ev = Evaluator({"big": np.array([1e103, 2e103, 3e103]), "a": np.array([1.0, 2.0, 3.0])})
        node = transform("cube", leaf("big"))
        assert ev.evaluate(node) is None
        assert [(r.node, r.reason) for r in ev.log] == [(node, ExclusionReason.OVERFLOW)]

    def test_constant_result_is_recorded(self):
        ev = Evaluator({"a": np.array([2.0, 3.0, 4.0]), "b": np.array([2.0, 3.0, 4.0]) + 1.0})
        # (b - a) is constant 1.0 everywhere
        assert ev.evaluate(combine("sub", leaf("b"), leaf("a"))) is None
        assert ev.log.by_reason(ExclusionReason.ZERO_VARIANCE)

    def test_exclusion_returns_none_never_raises(self, data):
        """Numerical failure is a return value, not an exception."""
        ev = Evaluator(data)
        node = transform("log", combine("sub", leaf("a"), leaf("b")))
        result = ev.evaluate(node)  # must not raise
        assert result is None

    def test_failed_child_propagates_without_double_logging(self, data):
        """A parent of a failed child logs once, at the true root cause."""
        ev = Evaluator(data)
        bad_child = transform("log", combine("sub", leaf("a"), leaf("b")))
        parent = combine("mul", bad_child, leaf("c"))
        assert ev.evaluate(parent) is None
        assert len(ev.log) == 1

    def test_log_counts_by_reason(self, data):
        ev = Evaluator(data)
        ev.evaluate(transform("log", combine("sub", leaf("a"), leaf("b"))))
        ev.evaluate(transform("sqrt", combine("sub", leaf("b"), leaf("a"))))
        counts = ev.log.counts()
        assert counts[ExclusionReason.DOMAIN_ERROR] == 2

    def test_log_is_iterable_and_sized(self, data):
        ev = Evaluator(data)
        ev.evaluate(transform("log", combine("sub", leaf("a"), leaf("b"))))
        records = list(ev.log)
        assert len(records) == len(ev.log) == 1
        assert records[0].reason is ExclusionReason.DOMAIN_ERROR
        assert records[0].node is not None

    def test_log_clear(self, data):
        ev = Evaluator(data)
        ev.evaluate(transform("log", combine("sub", leaf("a"), leaf("b"))))
        ev.log.clear()
        assert len(ev.log) == 0

    def test_variance_overflow_is_labelled_overflow(self):
        """Finite values whose variance overflows are not constant."""
        ev = Evaluator({"a": np.array([1e100, 2e100, 3e100]), "b": np.array([1e100, 3e100, 2e100])})
        node = combine("mul", leaf("a"), leaf("b"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert ev.evaluate(node) is None
        assert [(r.node, r.reason) for r in ev.log] == [(node, ExclusionReason.OVERFLOW)]

    def test_underflow_is_labelled_underflow(self):
        ev = Evaluator({"a": np.array([-800.0, -1.0, -2.0])})
        node = transform("exp", leaf("a"))
        assert ev.evaluate(node) is None
        assert [(r.node, r.reason) for r in ev.log] == [(node, ExclusionReason.UNDERFLOW)]

    def test_abs_of_non_negative_is_a_duplicate(self, data):
        """abs(x) equals x when x is never negative: a duplicate of x, not a
        domain violation."""
        ev = Evaluator(data)
        node = transform("abs", leaf("a"))
        assert ev.evaluate(node) is None
        assert [(r.node, r.reason) for r in ev.log] == [(node, ExclusionReason.DUPLICATE)]
        assert ev.transform_values(node) is not None

    def test_failure_is_logged_once(self, data):
        ev = Evaluator(data)
        bad = transform("log", combine("sub", leaf("a"), leaf("b")))
        assert ev.evaluate(bad) is None
        assert ev.evaluate(bad) is None
        assert ev.evaluate(combine("mul", bad, leaf("c"))) is None
        assert ev.evaluate_many([bad, bad])[0] == []
        assert len(ev.log) == 1

    def test_failure_is_logged_again_after_clearing_the_log(self, data):
        ev = Evaluator(data)
        bad = transform("log", combine("sub", leaf("a"), leaf("b")))
        ev.evaluate(bad)
        ev.log.clear()
        assert ev.evaluate(bad) is None
        assert len(ev.log) == 1

    def test_filter_rejection_does_not_block_unfiltered_evaluation(self):
        ev = Evaluator({"a": np.array([2.0, 3.0, 4.0]), "b": np.array([3.0, 4.0, 5.0])})
        constant = combine("sub", leaf("b"), leaf("a"))
        assert ev.evaluate(constant) is None
        np.testing.assert_allclose(ev.transform_values(constant), 1.0)
        assert ev.evaluate(constant) is None
        assert len(ev.log) == 1

    def test_variance_tolerance_is_configurable(self, rng):
        """The tolerance is relative to the column's magnitude, so the
        verdict survives a change of units."""
        near_constant = {"a": 1.0 + rng.normal(0, 1e-4, 100), "b": rng.normal(0, 1.0, 100)}
        strict = Evaluator(near_constant, variance_tol=1e-6)
        assert strict.evaluate(transform("square", leaf("a"))) is None
        lenient = Evaluator(near_constant, variance_tol=1e-40)
        assert lenient.evaluate(transform("square", leaf("a"))) is not None
        rescaled = {name: values * 1e-6 for name, values in near_constant.items()}
        assert Evaluator(rescaled, variance_tol=1e-6).evaluate(transform("square", leaf("a"))) is None


# --------------------------------------------------------------------------- #
# Numerical hygiene
# --------------------------------------------------------------------------- #


class TestNumericalHygiene:
    def test_default_dtype_is_float64(self, data):
        assert Evaluator(data).dtype == np.float64

    def test_dtype_is_respected(self, data):
        ev = Evaluator(data, dtype=np.float32)
        assert ev.evaluate(combine("mul", leaf("a"), leaf("b"))).dtype == np.float32

    def test_float64_survives_where_float32_overflows(self):
        """Motivates the float64 default: float32 overflows on cube of 1e13,
        while the column's own variance still fits."""
        payload = {"a": np.array([1e13, 2e13, 3e13]), "b": np.array([1.0, 2.0, 3.0])}
        node = transform("cube", leaf("a"))
        narrow = Evaluator(payload, dtype=np.float32)
        assert narrow.evaluate(node) is None
        assert [(r.node, r.reason) for r in narrow.log] == [(node, ExclusionReason.OVERFLOW)]
        assert Evaluator(payload, dtype=np.float64).evaluate(node) is not None

    def test_no_nan_or_inf_ever_returned(self, rng):
        """Whatever survives evaluation must be finite everywhere."""
        payload = {
            "a": rng.normal(0, 10, 300),
            "b": rng.normal(0, 10, 300),
            "c": rng.uniform(0.1, 5, 300),
        }
        ev = Evaluator(payload)
        leaves = [leaf("a"), leaf("b"), leaf("c")]
        nodes: list[Node] = []
        for op in ("log", "sqrt", "reciprocal", "square", "cube", "exp", "abs"):
            nodes.extend(transform(op, x) for x in leaves)
        for op in ("add", "sub", "mul", "div"):
            nodes.append(combine(op, leaves[0], leaves[1]))
            nodes.append(combine(op, leaves[1], leaves[2]))
        for node in nodes:
            values = ev.evaluate(node)
            if values is not None:
                assert np.all(np.isfinite(values)), f"{node.name} returned non-finite values"

    def test_every_rejection_has_a_reason(self, rng):
        """No node may be dropped without an entry in the log."""
        payload = {"a": rng.normal(0, 10, 100), "b": rng.normal(0, 10, 100)}
        ev = Evaluator(payload)
        nodes = [transform(op, leaf("a")) for op in ("log", "sqrt", "reciprocal", "exp")]
        n_rejected = sum(1 for node in nodes if ev.evaluate(node) is None)
        assert len(ev.log) == n_rejected


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


class TestDeterminism:
    def test_keys_stable_across_construction_order(self):
        forward = combine("mul", transform("log", leaf("a")), leaf("b"))
        backward = combine("mul", leaf("b"), transform("log", leaf("a")))
        assert forward.key == backward.key

    def test_evaluation_is_reproducible(self, data):
        node = combine("div", combine("mul", leaf("a"), leaf("b")), transform("sqrt", leaf("c")))
        first = Evaluator(data).evaluate(node)
        second = Evaluator(data).evaluate(node)
        np.testing.assert_array_equal(first, second)

    def test_cache_state_does_not_affect_results(self, data):
        node = combine("mul", leaf("a"), leaf("b"))
        cold = Evaluator(data, cache_size=1).evaluate(node).copy()
        warm_ev = Evaluator(data, cache_size=4096)
        warm_ev.evaluate(transform("log", leaf("a")))
        np.testing.assert_array_equal(warm_ev.evaluate(node), cold)


class TestEvaluationLogUnit:
    def test_empty_log(self):
        log = EvaluationLog()
        assert len(log) == 0
        assert log.counts() == {}

    def test_record_and_filter(self):
        log = EvaluationLog()
        node = leaf("a")
        log.record(node, ExclusionReason.OVERFLOW, "too big")
        log.record(node, ExclusionReason.ZERO_VARIANCE)
        assert len(log) == 2
        assert len(log.by_reason(ExclusionReason.OVERFLOW)) == 1
        assert log.counts()[ExclusionReason.OVERFLOW] == 1


class TestAlgebraicRewrites:
    """Local normalisation rules: the rewritten node must be equal to the
    original everywhere the original is defined, and the domain must never
    shrink. Each rule is pinned structurally."""

    def test_reciprocal_of_reciprocal_collapses(self):
        x = leaf("x")
        assert transform("reciprocal", transform("reciprocal", x)) == x

    def test_div_by_reciprocal_becomes_mul(self):
        x, y = leaf("x"), leaf("y")
        assert combine("div", x, transform("reciprocal", y)) == combine("mul", x, y)

    def test_div_by_own_reciprocal_becomes_square(self):
        x = leaf("x")
        assert combine("div", x, transform("reciprocal", x)) == transform("square", x)

    def test_reciprocal_of_div_swaps(self):
        a, b = leaf("a"), leaf("b")
        assert transform("reciprocal", combine("div", a, b)) == combine("div", b, a)

    def test_mul_by_reciprocal_becomes_div(self):
        a, b = leaf("a"), leaf("b")
        assert combine("mul", a, transform("reciprocal", b)) == combine("div", a, b)

    def test_reciprocal_div_left_folds(self):
        a, b = leaf("a"), leaf("b")
        node = combine("div", transform("reciprocal", a), b)
        assert node == transform("reciprocal", combine("mul", a, b))

    def test_sqrt_of_square_is_abs(self):
        x = leaf("x")
        assert transform("sqrt", transform("square", x)) == transform("abs", x)

    def test_rewrites_agree_numerically(self, rng):
        values = {"x": rng.uniform(0.5, 5.0, 200), "y": rng.uniform(0.5, 5.0, 200)}
        evaluator = Evaluator(values)
        x, y = leaf("x"), leaf("y")
        rewritten = combine("div", x, transform("reciprocal", y))
        direct = evaluator.transform_values(rewritten)
        np.testing.assert_allclose(direct, values["x"] * values["y"], rtol=1e-12)

    def test_n_operators_counts_operators(self):
        x, y = leaf("x"), leaf("y")
        assert x.n_operators == 0
        assert combine("mul", x, y).n_operators == 1
        assert transform("log", combine("mul", x, y)).n_operators == 2
        # the pre-existing `size` (DAG node count) is unshadowed and distinct
        assert x.size == 1


class TestUnitCoercion:
    """User-supplied units may be Quantities, pint Units, or strings; anything
    else fails at fit time with a readable message."""

    def test_string_units_parse(self, ureg):
        node = leaf("mass", "kg")
        assert node.unit is not None
        assert node.unit.dimensionality == (1 * ureg.kilogram).dimensionality

    def test_string_and_quantity_share_a_registry(self, ureg):
        evaluator = Evaluator(
            {"m": np.ones(10), "l": np.ones(10)},
            units={"m": 1 * ureg.kilogram, "l": "meter"},
        )
        nodes = evaluator.leaf_nodes()
        product = combine("mul", nodes[0], nodes[1])
        assert product.unit.dimensionality == (1 * ureg.kilogram * ureg.meter).dimensionality

    def test_bare_pint_unit_promoted(self, ureg):
        node = leaf("mass", ureg.kilogram)
        assert node.unit.dimensionality == (1 * ureg.kilogram).dimensionality

    def test_compound_string(self, ureg):
        node = leaf("accel", "m / s**2")
        assert node.unit.dimensionality == (1 * ureg.meter / ureg.second**2).dimensionality

    def test_unparseable_string_names_the_value(self):
        pytest.importorskip("pint")
        with pytest.raises(ValueError, match="notaunit"):
            leaf("x", "notaunit_xyz".replace("_xyz", ""))

    def test_unsupported_type_names_the_type(self):
        with pytest.raises(ValueError, match="unsupported unit specification"):
            leaf("x", 3.14)

    def test_string_without_pint_gives_guidance(self, monkeypatch):
        import builtins
        import sys

        monkeypatch.delitem(sys.modules, "pint", raising=False)
        original_import = builtins.__import__

        def no_pint(name, *args, **kwargs):
            if name == "pint":
                raise ImportError("No module named 'pint'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pint)
        with pytest.raises(ValueError, match="beamfeat\\[units\\]"):
            leaf("x", "kg")


class TestStringUnits:
    """Unit strings must be full citizens: parsed through pint and enforced
    identically to Quantity objects at every depth, never silently ignored."""

    def test_string_units_reject_invalid_sums(self, rng):
        pytest.importorskip("pint")
        values = {"mass": rng.uniform(1, 5, 200), "length": rng.uniform(1, 5, 200)}
        evaluator = Evaluator(values, units={"mass": "kg", "length": "m"})
        with pytest.raises(UnitError):
            combine(
                "add",
                evaluator.leaf_nodes()[0],
                evaluator.leaf_nodes()[1],
            )

    def test_string_units_enforce_at_depth_two(self, rng):
        """kg*m and kg**2 differ dimensionally; a sum of the two products must
        be rejected even though both operands are constructed nodes."""
        pytest.importorskip("pint")
        values = {"a": rng.uniform(1, 5, 200), "b": rng.uniform(1, 5, 200)}
        evaluator = Evaluator(values, units={"a": "kg", "b": "m"})
        a, b = evaluator.leaf_nodes()
        with pytest.raises(UnitError):
            combine("add", combine("mul", a, b), transform("square", a))

    def test_string_and_quantity_mix(self, rng):
        pint = pytest.importorskip("pint")
        registry = pint.UnitRegistry()
        values = {"a": rng.uniform(1, 5, 100), "b": rng.uniform(1, 5, 100)}
        evaluator = Evaluator(values, units={"a": "kg", "b": 1.0 * registry.kilogram})
        a, b = evaluator.leaf_nodes()
        node = combine("add", a, b)  # same dimension: allowed
        assert evaluator.transform_values(node) is not None

    def test_bad_string_is_actionable(self, rng):
        pytest.importorskip("pint")
        values = {"a": rng.uniform(1, 5, 50)}
        with pytest.raises(ValueError, match="could not parse unit string"):
            Evaluator(values, units={"a": "kgg^^2"})

    def test_string_units_without_pint_name_the_fix(self, rng, monkeypatch):
        """When pint is absent, a string unit must fail at fit time with the
        install command, not with an ImportError from inside unit
        propagation."""
        import builtins

        real_import = builtins.__import__

        def no_pint(name, *args, **kwargs):
            if name == "pint":
                raise ImportError("No module named 'pint'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pint)
        values = {"a": rng.uniform(1, 5, 50)}
        with pytest.raises(ValueError, match=r"beamfeat\[units\]"):
            Evaluator(values, units={"a": "kg"})
