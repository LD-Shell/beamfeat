"""Tests for screen_equation: another tool's equation tested term by term on held-out rows."""

from __future__ import annotations

import sys

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")

from beamfeat import EquationScreen, screen_equation  # noqa: E402


@pytest.fixture
def rows():
    """250 held-out rows of y = 1.5 x0 + 0.8 x1^2 + noise, inputs positive as physical quantities often are;
    x2 and x3 are unrelated to y."""
    rng = np.random.default_rng(20261025)
    X = rng.uniform(1, 3, (250, 4))
    y = 1.5 * X[:, 0] + 0.8 * X[:, 1] ** 2 + 0.5 * rng.normal(size=250)
    return X, y


def _at(found, text):
    """The index of the term printed as ``text`` (sympy orders the terms its own way)."""
    return [str(t) for t in found.terms].index(text)


def test_true_terms_are_kept_and_a_false_one_removed(rows):
    X, y = rows
    found = screen_equation("1.5*x0 + 0.8*x1**2 + 0.02*x2 + 0.3", X, y)
    assert isinstance(found, EquationScreen)
    assert sorted(str(t) for t in found.terms) == ["0.02*x2", "0.8*x1**2", "1.5*x0"]  # the constant is not a term
    assert sorted(str(t) for t in found.kept_terms) == ["0.8*x1**2", "1.5*x0"]
    assert found.removed == (_at(found, "0.02*x2"),) and found.untestable == ()
    assert found.p_values[_at(found, "1.5*x0")] < 0.01 and found.p_values[_at(found, "0.02*x2")] > 0.05
    assert found.marginal == found.kept and found.input_pairs == ()


def test_the_input_test_removes_a_rider_the_marginal_test_keeps(rows):
    X, y = rows
    equation = "1.5*x0 + 0.8*x1**2 + 0.05*x0*x3"
    marginal = screen_equation(equation, X, y)
    rider = _at(marginal, "0.05*x0*x3")
    assert rider in marginal.kept  # x3 is positive, so x0*x3 rides on x0 and is correlated with y
    found = screen_equation(equation, X, y, input_test=True)
    assert found.input_pairs == ((rider, "x0"), (rider, "x3"))
    assert [found.input_pairs[k] for k in found.input_rejected] == [(rider, "x0")]
    assert rider not in found.kept and len(found.kept) == 2 and found.marginal == marginal.kept


def test_under_the_null_a_term_is_rarely_kept():
    rng = np.random.default_rng(7)
    kept = 0
    for _ in range(40):
        X = rng.normal(size=(120, 3))
        y = rng.normal(size=120)
        kept += bool(screen_equation("x0 + x1*x2 + sin(x2)", X, y, input_test=True).kept)
    assert kept <= 10  # every null is true, so a kept term is a false discovery; at q = 0.1 about 4 of 40 expected


def test_a_term_undefined_or_constant_on_the_rows_is_untestable_and_never_kept(rows):
    X, y = rows
    found = screen_equation("1.5*x0 + log(x1 - 2) + x2/x2 + zoo*x3", X, y, input_test=True)
    assert sorted(str(t) for t in found.terms) == ["1.5*x0", "log(x1 - 2)", "zoo*x3"]  # x2/x2 is the constant 1
    undefined = (_at(found, "log(x1 - 2)"), _at(found, "zoo*x3"))  # log of a negative on half the rows; zoo
    assert found.untestable == tuple(sorted(undefined)) and found.kept == (_at(found, "1.5*x0"),)
    assert np.isnan(found.p_values[list(undefined)]).all()


def test_column_names_from_a_dataframe_or_feature_names(rows):
    pd = pytest.importorskip("pandas")
    X, y = rows
    frame = pd.DataFrame(X, columns=["E", "I", "temp", "beta"])
    from_frame = screen_equation("1.5*E + 0.8*I**2 + 0.02*temp", frame, y)
    from_names = screen_equation("1.5*E + 0.8*I**2 + 0.02*temp", X, y, feature_names=["E", "I", "temp", "beta"])
    plain = screen_equation("1.5*x0 + 0.8*x1**2 + 0.02*x2", X, y)
    # E and I are columns here, not Euler's number and the imaginary unit
    assert len(plain.kept) == 2 and from_frame.kept == from_names.kept == plain.kept
    assert np.array_equal(from_frame.p_values, plain.p_values)
    x0 = sympy.Symbol("x0")
    given = screen_equation(1.5 * x0 + sympy.Symbol("x2"), X, y)  # a sympy expression as well
    assert given.kept == (_at(given, "1.5*x0"),)


def test_a_list_of_terms_keeps_each_term_whole(rows):
    X, y = rows
    as_sum = screen_equation("1.5*x0 + 0.4*(x1**2 + x2)", X, y)
    as_list = screen_equation(["1.5*x0", "0.4*(x1**2 + x2)", "7"], X, y)
    assert sorted(str(t) for t in as_sum.terms) == ["0.4*x1**2", "0.4*x2", "1.5*x0"]  # sympy splits the sum
    assert [str(t) for t in as_list.terms] == ["1.5*x0", "0.4*x1**2 + 0.4*x2"]  # one term, kept whole; 7 is a constant
    assert as_list.kept == (0, 1)


def test_the_same_seed_gives_the_same_screen(rows):
    X, y = rows
    a = screen_equation("1.5*x0 + 0.05*x0*x3 + x2", X, y, input_test=True, random_state=3)
    b = screen_equation("1.5*x0 + 0.05*x0*x3 + x2", X, y, input_test=True, random_state=3)
    assert a.kept == b.kept and np.array_equal(a.p_values, b.p_values)
    assert np.array_equal(a.input_p_values, b.input_p_values)


def test_bh_keeps_at_least_what_by_keeps(rows):
    X, y = rows
    equation = "1.5*x0 + 0.8*x1**2 + 0.05*x2 + 0.05*x3"
    by, bh = screen_equation(equation, X, y), screen_equation(equation, X, y, correction="bh")
    assert set(by.kept) <= set(bh.kept)


def test_an_equation_without_variables_has_nothing_to_screen(rows):
    X, y = rows
    found = screen_equation("2.5", X, y, input_test=True)
    assert found.terms == () and found.kept == () and found.input_pairs == ()


def test_the_summary_names_each_terms_outcome(rows):
    X, y = rows
    text = screen_equation("1.5*x0 + 0.05*x0*x3 + log(x1 - 2)", X, y, input_test=True).summary()
    assert text.splitlines()[0] == "screen_equation: 1 of 3 terms kept (marginal BY at q = 0.1, input test, 250 rows)"
    assert "[kept] 1.5*x0: p = " in text and "null rejected" in text
    assert "inputs: x0 rejected, x3 not rejected" in text and "[removed] log(x1 - 2): untestable" in text


@pytest.mark.parametrize(("kwargs", "message"), [
    (dict(equation="x9"), "x9, which is not among the columns"),
    (dict(q=1.0), "q must be in"),
    (dict(correction="holm"), "correction must be"),
    (dict(y=np.full(250, np.nan)), "y must be finite"),
    (dict(y=np.zeros(10)), "y must be finite"),
    (dict(feature_names=["a", "b", "c d", "e"]), "Python identifiers"),
    (dict(feature_names=["a", "a", "b", "c"]), "distinct"),
    (dict(feature_names=["a", "b"]), "one column per name"),
])
def test_invalid_arguments(rows, kwargs, message):
    X, y = rows
    arguments = dict(equation="x0", X=X, y=y) | kwargs
    with pytest.raises(ValueError, match=message):
        screen_equation(arguments.pop("equation"), arguments.pop("X"), arguments.pop("y"), **arguments)


def test_without_sympy_the_error_names_the_extra(rows, monkeypatch):
    monkeypatch.setitem(sys.modules, "sympy", None)
    with pytest.raises(ImportError, match=r"beamfeat\[display\]"):
        screen_equation("x0", *rows)
