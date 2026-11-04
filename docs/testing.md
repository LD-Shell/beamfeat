---
description: "How beamfeat is tested: calibration of its false discovery rate, recovery of known formulas, scikit-learn conformance, and the CI matrix."
---

# Testing

The test suite lives in
[`tests/`](https://github.com/LD-Shell/beamfeat/tree/main/tests) and runs
with pytest. Besides ordinary unit tests, it checks the package's statistical
claims directly, by simulating data where the truth is known.

## Running the tests

```bash
pip install -e ".[test]"
pytest -q
```

The two scikit-learn conformance tests are marked `slow`; `pytest -q -m "not slow"`
skips them. Tests that need pint skip with a stated reason when it is not
installed (`pytest -rs` lists them).

## What the tests check

### False discovery rate

FDR is an expectation over repeated experiments, so these tests repeat a
simulation over many seeds and compare the average with the nominal level. A
single run above the level is allowed; an average above it, beyond Monte Carlo
slack, fails the test.

| check | design | passes when |
|---|---|---|
| valid p-values | 40 pure-noise datasets, 320 permutation p-values | at most 15% are ≤ 0.1 |
| FDR, permutation screen (BH and BY) | 25 planted-signal datasets, q = 0.2 | mean FDP ≤ 0.3 |
| FDR, knockoffs (`offset=1`, and `s_method="ci"` on correlated blocks) | 20 datasets each, q = 0.2 | mean FDP ≤ 0.3 |
| FDR of a whole fit | 12 fits that search on one half and screen on the other, q = 0.2 | mean FDP ≤ 0.35, and the true product found in all but at most two fits |
| pure noise | 25 noise targets against 30 noise columns, permutation screen, q = 0.1 | fewer than one column selected per run on average |
| `screen_equation` under the null | 40 noise targets, q = 0.1 | at most 10 reject any term's null (about 4 expected) |
| forward step's chance check | 200 designs where no further column matters, five of them correlated with the chosen one | rejects in 5–16% of designs at level 0.1 |

The power tests sit beside them, so a screen that never rejects anything can't
pass by default: on planted signals the permutation screen must recover more
than 80% of the true features on average.

The slack in these tests is wide enough to catch a broken procedure without
failing on Monte Carlo noise. The precise figures, from 100 trials each with
standard errors, are on [Statistical guarantees](guarantees.md) and
[Benchmarks](benchmarks.md).

### Known formulas

Fits on data generated from a known law must find that law: a product of
columns, a quotient, power laws with fractional exponents such as
`x0**(3/2)`, and laws that need a fitted constant inside the expression. Where
the test compares formulas symbolically, it uses sympy and checks that the fitted
expression is proportional to the true one.

### scikit-learn behaviour

- The estimators pass scikit-learn's `check_estimator` suite, with selection
  on and off.
- Feature construction happens inside each cross-validation fold, so
  cross-validated scores carry no leakage from the held-out fold.
- A fit with the same `random_state` gives the same result.

## Continuous integration

[`ci.yml`](https://github.com/LD-Shell/beamfeat/blob/main/.github/workflows/ci.yml)
runs on every push to `main`, on every pull request and every Monday, so a new
release of a dependency is tested even when nothing in beamfeat has changed.
It runs:

- the full suite on Python 3.10 to 3.15 with the newest dependencies, with
  line coverage of at least 90% required;
- the suite on the oldest supported stack (Python 3.10, numpy 1.26,
  scikit-learn 1.6, joblib 1.2) without pint;
- the four tutorial notebooks, executed end to end;
- ruff on the package, the tests and the benchmarks.

Deprecation warnings raised from beamfeat or its tests, and any
`FutureWarning`, are turned into errors, so a deprecated call fails the suite
while it still works.
