"""A weak term beside a dominant one, and targets that switch at a threshold.

Two studies of 0.8.0's search changes against 0.7.0, each release in its own
environment (``benchmarks/release_comparison/venvs/<version>``):

``weak``: ``y = x0*x1/x2 + 0.5*log(x3)`` on [1, 5], where the logarithm
carries 0.6% of the variance, with 4 or 10 columns (six unrelated), 500,
2,000 or 5,000 rows, noise of 0.1% or 10% of the signal's standard deviation,
five seeds. Recorded: whether ``log(x3)`` is among the search's expressions,
whether a printed term is ``log(x3)``, whether the printed equation is the
law (its two terms and nothing else), R^2 against the noiseless law on
5,000 new rows.

``switch``: ``y = s(x0)*x2 + x1`` with ``s = 2`` above a threshold on ``x0``
and ``-1`` below it, the threshold at 0 or 0.7; ``x0`` and ``x1`` on [-2, 2],
``x2`` to ``x4`` on [1, 5]; 1,000 rows, noise of standard deviation 0.3 or
1.0, five seeds. Settings: each release's default, and 0.8.0 with ``sign``
among the unary operators or ``sign_shift`` among the fitted ones; a
gradient-boosted tree model (scikit-learn's ``HistGradientBoostingRegressor``
at its defaults) for reference. Recorded: R^2 against the noiseless target on
5,000 new rows, whether a printed term is the switch, terms printed.

Usage, from this folder::

    ../release_comparison/venvs/0.7.0/bin/python -I run.py --version 0.7.0
    ../release_comparison/venvs/0.8.0/bin/python -I run.py --version 0.8.0
    python run.py --report > TABLES.md
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
warnings.filterwarnings("ignore")

DEFAULT_UNARY = ("log", "sqrt", "reciprocal", "square", "abs")
DEFAULT_FITTED = ("exp_scaled", "pow", "inv_shift", "log_shift")
SETTINGS = {
    "0.7.0": {"default": {}},
    "0.8.0": {
        "default": {},
        "sign": {"unary_ops": DEFAULT_UNARY + ("sign",)},
        "sign_shift": {"fitted_ops": DEFAULT_FITTED + ("sign_shift",)},
    },
}
WEAK_CELLS = [(p, n, noise) for p in (4, 10) for n in (500, 2000, 5000) for noise in (0.001, 0.1)]
SWITCH_CELLS = [(threshold, noise) for threshold in (0.0, 0.7) for noise in (0.3, 1.0)]
SEEDS = range(5)


def weak_data(p, n, noise, seed):
    rng = np.random.default_rng(seed)
    X, X_test = rng.uniform(1, 5, (n, p)), rng.uniform(1, 5, (5000, p))
    law = lambda Z: Z[:, 0] * Z[:, 1] / Z[:, 2] + 0.5 * np.log(Z[:, 3])  # noqa: E731
    signal = law(X)
    return X, signal + rng.normal(0, noise * np.std(signal), n), X_test, law(X_test)


def switch_data(threshold, noise, seed):
    rng = np.random.default_rng(100 + seed)

    def draw(n):
        return np.column_stack([rng.uniform(-2, 2, (n, 2)), rng.uniform(1, 5, (n, 3))])

    law = lambda Z: np.where(Z[:, 0] > threshold, 2.0, -1.0) * Z[:, 2] + Z[:, 1]  # noqa: E731
    X, X_test = draw(1000), draw(5000)
    return X, law(X) + rng.normal(0, noise, 1000), X_test, law(X_test)


def _plain(expr):
    import sympy

    plain = expr.subs({symbol: sympy.Symbol(symbol.name) for symbol in expr.free_symbols})
    return sympy.nsimplify(plain, rational=True)


def _same(a, b) -> bool:
    import sympy

    ratio = sympy.simplify(a / b)
    return ratio.free_symbols == set() and ratio != 0


def r2(prediction, truth):
    return float(1 - np.mean((prediction - truth) ** 2) / np.var(truth))


def run(version, out):
    import sympy

    from beamfeat import BeamFeatRegressor

    x = sympy.symbols("x0:10")
    done = set()
    if out.exists():
        done = {(r["study"], r["item"], r["config"]) for r in map(json.loads, out.open())}
    with out.open("a") as handle:
        def write(record):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record), flush=True)

        for seed in SEEDS:
            for p, n, noise in WEAK_CELLS:
                item = f"p{p}_n{n}_noise{noise}:{seed}"
                if ("weak", item, "default") in done:
                    continue
                X, y, X_test, truth = weak_data(p, n, noise, seed)
                started = time.perf_counter()
                model = BeamFeatRegressor(random_state=seed).fit(X, y)
                terms = [_plain(t) for t in model.to_sympy()] if model.features_ else []
                logs = [t for t in terms if _same(t, sympy.log(x[3]))]
                strong = [t for t in terms if _same(t, x[0] * x[1] / x[2])]
                write({"version": version, "study": "weak", "item": item, "config": "default",
                       "searched": "log(x3)" in model.search_result_.names, "printed": bool(logs),
                       "exact": len(terms) == 2 and bool(logs) and bool(strong),
                       "terms": len(terms), "score": r2(model.predict(X_test), truth),
                       "formulas": list(model.feature_formulas_), "seconds": time.perf_counter() - started})
            for threshold, noise in SWITCH_CELLS:
                item = f"t{threshold}_noise{noise}:{seed}"
                X, y, X_test, truth = switch_data(threshold, noise, seed)
                for config, kwargs in SETTINGS[version].items():
                    if ("switch", item, config) in done:
                        continue
                    started = time.perf_counter()
                    model = BeamFeatRegressor(random_state=seed, **kwargs).fit(X, y)
                    terms = [_plain(t) for t in model.to_sympy()] if model.features_ else []
                    switch = x[2] * sympy.sign(x[0] - sympy.Rational(7, 10) if threshold else x[0])
                    write({"version": version, "study": "switch", "item": item, "config": config,
                           "switch_printed": any(_same(t, switch) for t in terms), "terms": len(terms),
                           "score": r2(model.predict(X_test), truth), "formulas": list(model.feature_formulas_),
                           "seconds": time.perf_counter() - started})
                if version == "0.8.0" and ("switch", item, "trees") not in done:
                    from sklearn.ensemble import HistGradientBoostingRegressor

                    started = time.perf_counter()
                    trees = HistGradientBoostingRegressor(random_state=seed).fit(X, y)
                    write({"version": "sklearn", "study": "switch", "item": item, "config": "trees",
                           "score": r2(trees.predict(X_test), truth), "seconds": time.perf_counter() - started})


def report(folder):
    rows = []
    for path in sorted(folder.glob("records_*.jsonl")):
        rows += [json.loads(line) for line in path.open()]
    cell = lambda r: r["item"].split(":")[0]  # noqa: E731
    print("## A weak term beside a dominant one: `x0*x1/x2 + 0.5*log(x3)`, five seeds per cell", "")
    print("| columns, rows, noise | 0.7.0: `log(x3)` searched / printed / exact | 0.8.0: searched / printed / exact "
          "| R² vs law, 0.7.0 | R² vs law, 0.8.0 |")
    print("| :-- | --: | --: | --: | --: |")
    for p, n, noise in WEAK_CELLS:
        key = f"p{p}_n{n}_noise{noise}"
        line = [f"{p}, {n:,}, {noise:.1%}"]
        scores = []
        for version in ("0.7.0", "0.8.0"):
            s = [r for r in rows if r["study"] == "weak" and r["version"] == version and cell(r) == key]
            line.append(f"{sum(r['searched'] for r in s)} / {sum(r['printed'] for r in s)} / {sum(r['exact'] for r in s)}"
                        f" of {len(s)}" if s else "—")
            scores.append(f"{np.mean([r['score'] for r in s]):.5f}" if s else "—")
        print("| " + " | ".join(line + scores) + " |")
    print()
    print("## A switch at a threshold: `y = s(x0)*x2 + x1`, five seeds per cell", "")
    settings = [("0.7.0", "default"), ("0.8.0", "default"), ("0.8.0", "sign"), ("0.8.0", "sign_shift"), ("sklearn", "trees")]
    names = {("0.7.0", "default"): "0.7.0", ("0.8.0", "default"): "0.8.0", ("0.8.0", "sign"): "0.8.0 `sign`",
             ("0.8.0", "sign_shift"): "0.8.0 `sign_shift`", ("sklearn", "trees"): "gradient-boosted trees"}
    print("| threshold, noise | " + " | ".join(names[s] for s in settings) + " |")
    print("| :-- |" + " --: |" * len(settings))
    for threshold, noise in SWITCH_CELLS:
        key = f"t{threshold}_noise{noise}"
        line = [f"{threshold}, {noise}"]
        for version, config in settings:
            s = [r for r in rows if r["study"] == "switch" and r["version"] == version and r["config"] == config
                 and cell(r) == key]
            if not s:
                line.append("—")
            elif config == "trees":
                line.append(f"{np.mean([r['score'] for r in s]):.3f}")
            else:
                line.append(f"{np.mean([r['score'] for r in s]):.3f} ({sum(r['switch_printed'] for r in s)} of {len(s)})")
        print("| " + " | ".join(line) + " |")
    print()
    print("R² against the noiseless target on 5,000 new rows, mean over seeds; in brackets, fits whose printed "
          "equation has the switch term `x2*sign(x0)` or `x2*sign(x0 - 0.7)`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    results = HERE / "results"
    if args.report:
        report(results)
    else:
        results.mkdir(exist_ok=True)
        run(args.version, results / f"records_{args.version}.jsonl")
