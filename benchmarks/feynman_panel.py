"""Feynman-equation panel for beamfeat.

Twelve physics equations spanning what a depth-limited, constant-free
symbolic feature space can and cannot represent, evaluated the way the
symbolic-regression community evaluates (SRBench): a problem counts as
*solved* at held-out R^2 > 0.999 under 0.1% noise. A second, stricter
criterion is reported alongside: *exact form*, meaning the fitted equation's
kept terms match the law's terms one-to-one, each proportional to its
counterpart (or a single kept term is proportional to the whole law). The
check is algebraic: sympy simplifies the ratio of each fitted term from
``to_sympy()`` to the expected term and requires a nonzero constant, so
operand order, association and the leading constant the linear model
supplies do not matter, while any extra kept term does.

Four problems are included as expected failures, each marking a documented
boundary: the Gaussian needs ``exp`` and a scale inside the exponent,
relativistic velocity addition needs a literal constant inside the expression,
the weighted mean is a depth-3 rational, and the 2D distance formula needs
depth-4 nesting. The panel reports the default operator set and one with
``exp`` enabled.

Run with ``python benchmarks/feynman_panel.py``. Results are written to
``benchmarks/feynman_results.json``.
"""

from __future__ import annotations

import json
import pathlib
import time
import warnings
import zlib

import numpy as np

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).parent


def _proportional(model_expr, expected_expr) -> bool:
    """True when the two sympy expressions differ only by a nonzero constant.

    String comparison is too strict: the search canonicalises commutative
    operand order but not associativity, so ``x0*(x1*x2)`` and
    ``(x0*x1)*x2`` are distinct spellings of one expression, and the linear
    model supplies any leading constant (the 1/2 in kx^2/2, the 2*pi in the
    pendulum period). Algebraic equivalence up to scale is the honest
    criterion for symbolic recovery.
    """
    import sympy

    try:
        ratio = sympy.simplify(model_expr / expected_expr)
    except Exception:  # pragma: no cover - sympy edge cases
        return False
    return ratio.free_symbols == set() and ratio != 0


def _exact_form(model, expected_terms) -> bool:
    """Whether the fitted terms match the expected terms up to scale.

    ``expected_terms`` is a list of sympy expressions; the fitted equation is
    exact when its kept formulas match them as a multiset, each pair
    proportional. Single-term laws therefore require exactly one kept feature
    proportional to the law; multi-term laws (u + a t) require the same
    one-to-one correspondence.
    """
    import sympy

    fitted = model.to_sympy()
    # ``to_sympy`` mints its symbols with assumptions, and sympy treats
    # same-named symbols with different assumptions as distinct — a ratio of
    # the two never cancels. Rebuild the expected terms inside the model's
    # own symbol world by name substitution before comparing.
    model_symbols = {s.name: s for expr in fitted for s in expr.free_symbols}
    expected_terms = [
        term.subs({s: model_symbols.get(s.name, s) for s in term.free_symbols})
        for term in expected_terms
    ]
    # A single fitted feature proportional to the whole law is exact too: the
    # search may capture a multi-term law as one composed expression, with the
    # linear model supplying only the overall scale.
    if len(fitted) == 1 and len(expected_terms) > 1:
        return _proportional(fitted[0], sympy.Add(*expected_terms))
    if len(fitted) != len(expected_terms):
        return False
    remaining = list(expected_terms)
    for term in fitted:
        match = next((e for e in remaining if _proportional(term, e)), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def equations():
    """(name, generator, n_columns, expected sympy terms or None, note)."""
    import sympy

    x0, x1, x2, x3 = sympy.symbols("x0 x1 x2 x3")
    return [
        (
            "lorentz_qvB", lambda X: X[:, 0] * X[:, 1] * X[:, 2], 3,
            [x0 * x1 * x2],
            "triple product",
        ),
        (
            "spring_energy", lambda X: 0.5 * X[:, 0] * X[:, 1] ** 2, 2,
            [x0 * x1**2],
            "k x^2 / 2",
        ),
        (
            "coulomb", lambda X: X[:, 0] * X[:, 1] / X[:, 2] ** 2, 3,
            [x0 * x1 / x2**2],
            "q1 q2 / r^2",
        ),
        (
            "pendulum_period", lambda X: 2 * np.pi * np.sqrt(X[:, 0] / X[:, 1]), 2,
            [sympy.sqrt(x0 / x1)],
            "2 pi sqrt(L/g)",
        ),
        (
            "ohmic_power", lambda X: X[:, 0] ** 2 / X[:, 1], 2,
            [x0**2 / x1],
            "V^2 / R",
        ),
        (
            "uniform_accel", lambda X: X[:, 0] + X[:, 1] * X[:, 2], 3,
            [x0, x1 * x2],
            "u + a t (two-term)",
        ),
        (
            "ideal_gas_T", lambda X: X[:, 0] * X[:, 1] / X[:, 2], 3,
            [x0 * x1 / x2],
            "P V / (n R)",
        ),
        (
            "weighted_mean",
            lambda X: (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]) / (X[:, 0] + X[:, 2]), 4,
            [(x0 * x1 + x2 * x3) / (x0 + x2)],
            "(m1 r1 + m2 r2)/(m1 + m2), depth 3",
        ),
        (
            "gravitational_pe", lambda X: X[:, 0] * X[:, 1] * X[:, 2] / X[:, 3], 4,
            [x0 * x1 * x2 / x3],
            "G m1 m2 / r (as product/ratio)",
        ),
        (
            "gaussian", lambda X: np.exp(-(X[:, 0] ** 2) / 2.0), 1,
            None,
            "no exact form available: needs exp, and the exponent's 1/2 scale is not representable; "
            "smooth surrogates can still clear the numeric criterion on a bounded domain",
        ),
        (
            "relativistic_velocity",
            lambda X: (X[:, 0] + X[:, 1]) / (1.0 + X[:, 0] * X[:, 1]), 2,
            None,
            "no exact form available: a literal constant inside the expression; smooth "
            "surrogates can clear the numeric criterion on a bounded domain",
        ),
        (
            "distance_2d",
            lambda X: np.sqrt((X[:, 0] - X[:, 1]) ** 2 + (X[:, 2] - X[:, 3]) ** 2), 4,
            None,
            "no exact form at depth 3: the law needs depth-4 nesting",
        ),
    ]


def make_data(name, generator, n_cols):
    """Training and probe draws for one law: ``(X, X_test, y, y_test)``.

    Separated from the fit loop so that a comparison harness scores its method
    on the same rows rather than on a copy of this code that can drift from
    it (see ``benchmarks/pysr_panel/``).
    """
    # crc32 rather than hash(): Python randomises string hashing per
    # process, so hash(name) would reseed the data on every run and the
    # panel would not be reproducible.
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    low, high = (0.1, 1.0) if name in ("gaussian", "relativistic_velocity") else (1.0, 5.0)
    X = rng.uniform(low, high, (500, n_cols))
    X_test = rng.uniform(low, high, (500, n_cols))
    signal = generator(X)
    y = signal + rng.normal(0, 1e-3 * np.std(signal), 500)
    y_test = generator(X_test)
    return X, X_test, y, y_test


def run_panel(unary_ops=None, label="default operators", parsimony="forward"):
    from beamfeat import BeamFeatRegressor

    rows = []
    for name, generator, n_cols, expected_terms, note in equations():
        X, X_test, y, y_test = make_data(name, generator, n_cols)

        kwargs = {"unary_ops": unary_ops} if unary_ops is not None else {}
        started = time.perf_counter()
        model = BeamFeatRegressor(
            max_depth=3, beam_width=40, random_state=0, parsimony=parsimony, **kwargs
        ).fit(X, y)
        elapsed = time.perf_counter() - started
        r2 = float(model.score(X_test, y_test))

        exact = _exact_form(model, expected_terms) if expected_terms is not None else None
        rows.append({
            "equation": name, "note": note, "r2": r2, "seconds": elapsed,
            "solved": r2 > 0.999, "exact_form": exact,
            "n_terms": model.n_features_out_,
            "fdr_controlled": model.fdr_controlled_,
            "top_formula": model.formulas()[0] if model.formulas() else "",
        })
    solved = sum(r["solved"] for r in rows)
    exact = sum(1 for r in rows if r["exact_form"])
    mean_t = float(np.mean([r["seconds"] for r in rows]))
    print(f"\n[{label}]  solved {solved}/12 (R^2>0.999)  exact-form {exact}  mean {mean_t:.1f}s/eq")
    for r in rows:
        mark = "SOLVED" if r["solved"] else "  --  "
        ef = {True: " exact", False: "", None: ""}[r["exact_form"]]
        print(f"  {mark}{ef:>6}  {r['equation']:<22} R2 {r['r2']:.4f}  {r['seconds']:4.1f}s  {r['top_formula'][:44]}")
    return {"label": label, "solved": solved, "exact_form": exact,
            "parsimony": parsimony, "mean_seconds": mean_t, "rows": rows}


def main(parsimony="forward"):
    results = [
        run_panel(parsimony=parsimony),
        run_panel(unary_ops=("log", "sqrt", "reciprocal", "square", "abs", "exp"),
                  label="with exp enabled", parsimony=parsimony),
    ]
    with (HERE / "feynman_results.json").open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {HERE / 'feynman_results.json'}")


if __name__ == "__main__":
    main()
