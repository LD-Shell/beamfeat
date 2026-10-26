---
description: "Choose between beamfeat's default, a short equation (parsimony_alpha) and every column tested (input_test), and when to use the options for tall data, switches and weak terms: what each prints, what it costs, what was measured."
---

# Choosing a setting

Every fit runs the same search and the same screen, and the screen's FDR
guarantee covers the same certified set whatever the setting. What changes is
which certified formulas the printed equation uses. The default and two
options cover what most fits need:

| you want | write | the printed equation | the cost |
| :-- | :-- | :-- | :-- |
| the best prediction | `BeamFeatRegressor()` | the formulas that predict best; on a noisy target, some extra terms | none |
| a short equation | `BeamFeatRegressor(parsimony_alpha=0.05)` | usually one to three terms | some accuracy where the target is not one compact formula |
| every column tested | `BeamFeatRegressor(input_test=True)` | no formula with a column whose null the test fails to reject | 2 to 7 times the fit time, and some accuracy on real data |

The same parameters work on `BeamFeatClassifier` and `BeamFeatTransformer`,
and the two options combine.

## The three on one dataset

The target is `x0 * x1` with 25% noise, among ten columns that play no part
in it:

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(11)
X = rng.uniform(1, 6, (500, 12))
signal = X[:, 0] * X[:, 1]
y = signal + rng.normal(0, 0.25 * np.std(signal), 500)

for options in ({}, {"parsimony_alpha": 0.05}, {"input_test": True}):
    print(BeamFeatRegressor(random_state=0, **options).fit(X, y).equation())
```

```text
y = 1.052*(x0 * x1) - 0.4088*((x0 * x1) / (x0 + x6)) - 0.1452*((x3 / x0) - x1) + 0.1295*((x5 / x0) - x1) + 0.04642   [4 of 50 certified terms; parsimony=None prints all 50; parsimony_alpha=0.05 prints 1, selection-row R^2 0.935 against 0.939]
y = 1.013*(x0 * x1) - 0.1608   [1 of 50 certified terms; parsimony=None prints all 50]
y = 1.013*(x0 * x1) - 0.1608   [1 of 50 certified terms; parsimony=None prints all 8; input_test dropped 42]
```

The default adds three terms to the product, and each brings in a column the
target does not use: `x6`, `x3` or `x5`. Each adds a little to R² on the rows the
equation was chosen on, by fitting noise. The note at the end of the line
says how much: `parsimony_alpha=0.05` would print one term, for an R² of 0.935
against 0.939 on the selection rows, and it does stop before them.
`input_test` drops 42 of the 50 certified formulas, those with a column
whose null the test fails to reject, and the forward step then stops at the
product.

## The default: the best prediction

`BeamFeatRegressor()` adds certified formulas to the equation while each adds
at least 0.001 to R² (`parsimony_tol`). That is the right rule for prediction,
and on a target with little noise the equation is short anyway. On a noisy
target, though, the best of the many remaining formulas clears 0.001 by
fitting noise alone, so the equation grows terms that each add a little.
On `y = x0/x1` among ten unrelated columns, with noise as large as the signal,
the default printed a median of 9.5 terms, and 78 of its 92 terms over ten
fits read an unrelated column.

Keep the default when prediction comes first, and read its equation with
[`equations()`](guide/reading-a-fit.md#the-equation-at-every-size), which
shows how much each term adds. When `parsimony_alpha=0.05` would print fewer
terms, `equation()` ends with a note saying how many, and the R² of both
equations on the selection rows; `equations()` marks that line `a`.

## A short equation: `parsimony_alpha=0.05`

The forward step also stops at the first term whose gain fails to reject the
null that chance reaches it: the null distribution is the best gain the
remaining formulas reach against noise. On the noisy `x0/x1` above it printed
the one true term in every fit, and its predictions were closer to the true
function than the default's (R² 0.998 against 0.981).

Use it for an equation to read or report, or to look for a law. Real targets
tend to carry many modest effects, each adding little, so the step stops
after a few terms (one to five on 10 of 12 real datasets) and costs some
accuracy: held-out score fell on 9 of the 12, by 0.01 to 0.05, and rose a
little on eyedata, where the default fits 31 terms to 84 training rows. It
is a stopping rule, not a guarantee on the terms it keeps. [A short equation](guide/printed-equation.md#a-short-equation-parsimony_alpha)
has the details.

## Every column tested: `input_test=True`

The screen tests each formula as a whole, so a formula that attaches an
unrelated column to a true one, such as `(x0 * x1) / (x0 + x6)` when
`y = x0 * x1`, is a true discovery. `input_test=True` tests each column
inside each formula that reads more than one, by permuting that column's
rows, and drops a certified formula when the test fails to reject the null
that the column is independent of the target and of the formula's other
columns.

Use it when every column in the equation must be shown to matter. It takes
2.4 to 7 times as long as the default (3.6 times on the real datasets), more
on more rows, and `n_jobs` threads share the tests. Held-out score stayed
within 0.005 of the default's on 5 of 12 real datasets and fell on 7, by
0.06 at most. It does not remove riders built only from the target's own
columns, as `parsimony_alpha` does, and a junk column correlated with a real
one is not null in its sense.
[Every column tested](guide/printed-equation.md#every-column-tested-input_test)
has the details.

## Both

`BeamFeatRegressor(parsimony_alpha=0.05, input_test=True)` gives a short
equation whose every column was tested. It is no more accurate than
`parsimony_alpha` alone and takes as long as `input_test`.

## Tall data: `max_search_rows`

The search's time and memory grow with its rows. On a tall table,
`BeamFeatRegressor(max_search_rows=10_000)` runs the search on 10,000 of its
rows, drawn at random, and the screen on every selection row as before, so the
guarantee is unchanged. On 200,000 rows, a search on 10,000 of its 100,000
rows cut the median fit from 110 to 33 seconds. A product carrying 2% of the
signal's variance, `x4*x5`, came out as `x4` and `x5` terms in every fit, as
with the full search, and R² against the signal was 0.002 lower
(`benchmarks/search_rows/`). Memory fell by a sixth only, since the screen and
the final fit still see every row. A term weak enough to need every row to
stand out can be lost.

## A target that switches: `sign` and `sign_shift`

A target that follows one formula on one side of a threshold and another on
the other side is outside what products, quotients and smooth transforms
build. The `sign_shift` operator, off by default, fits `sign(x + a)` with its
threshold `a`, so that a term such as `x2*sign(x0 - 0.7)` can be built:

```python
# The default fitted operators and sign_shift.
BeamFeatRegressor(fitted_ops=("exp_scaled", "pow", "inv_shift", "log_shift", "sign_shift"))
```

On `y = s(x0)*x2 + x1`, where `s` is 2 above a threshold on `x0` and -1
below it, it printed the switch term in all 20 fits and predicted the
target to R² 0.997 to 0.999, against 0.76 to 0.88 for the default and 0.983
to 0.996 for gradient-boosted trees; on the main suite's `threshold_step` it
raised held-out score from 0.841 to 0.999. Where the target does not switch
it changes little, though it lowered the classifier's accuracy on
breast_cancer by 0.022. `sign`, listed in `unary_ops`, builds a switch at
zero only, which `sign_shift` also finds.
[Switches at a threshold](guide/search.md#switches-at-a-threshold) has the
details.

## A law with a weak term: `residual_slots=10`

The search ranks its candidates against the target, so a term much weaker
than the leading one can be left out of the ranking, as `log(x3)` is from
`x0*x1/x2 + 0.5*log(x3)`. `BeamFeatRegressor(residual_slots=10,
parsimony_alpha=0.05)` adds to the ranking the terms that explain what the
leading ones leave, and stops the riders they would otherwise bring. On that
law it printed the exact equation in 33 of 60 fits, the default in none. On
the comparison's 11 real regression datasets its held-out score was within
0.005 of `parsimony_alpha` alone on 10, and 0.012 lower on California
housing.
[A weak term beside a dominant one](guide/search.md#a-weak-term-beside-a-dominant-one)
has the details.

## What was measured

[`benchmarks/three_settings/`](https://github.com/LD-Shell/beamfeat/tree/main/benchmarks/three_settings)
fits the default, `parsimony_alpha=0.05`, `input_test=True` and both
together on the same data with the same seeds:

| | default | `parsimony_alpha=0.05` | `input_test=True` | both |
| :-- | --: | --: | --: | --: |
| noise sweep, noise as large as the signal: median terms | 9.5 | 1 | 2 | 1 |
| … terms reading an unrelated column | 78 of 92 | 0 of 10 | 3 of 21 | 0 of 10 |
| … R² against the true function | 0.981 | 0.998 | 0.995 | 0.998 |
| 12 real datasets: mean held-out score | 0.693 | 0.675 | 0.675 | 0.667 |
| … mean terms | 13.8 | 3.5 | 10.5 | 3.1 |
| … better / same / worse than the default | — | 1 / 2 / 9 | 0 / 5 / 7 | 1 / 1 / 10 |
| 18 synthetic datasets: within 0.005 of the default | — | 15 | 17 | 16 |
| fit time, × the default's | 1 | 1.0 | 3.6 | 3.6 |
| the paper's comparison study: mean R² (paper 0.803) | 0.803 | 0.784 | 0.803 | 0.786 |

The noise sweep is ten fits of `y = x0/x1` among ten unrelated columns; the
real and synthetic datasets are those of the release comparison, five splits
each. At the paper's own settings all four recover the same formulas, solve
the same Feynman laws and print no formula built only from distractors, as
the paper reports; the benchmark's report has every table.

Whatever the setting, a single fit shows one equation from one random split
of the rows. [`stability_report`](guide/reading-a-fit.md#one-equation-or-one-of-many)
refits on many splits and shows whether that equation is the only good one or
one of many; run it before reading much into which formula was picked.

## Other options that change the equation

Two more options trade length against the guarantee:

- `parsimony=None` prints every certified formula, so the guarantee covers
  the printed equation itself; on wide data that is dozens of terms.
- `parsimony_holdout` re-tests the printed terms on rows held back for it,
  so the equation is short and the guarantee covers it; it costs those rows.

[The printed equation](guide/printed-equation.md) describes both.

## When beamfeat is the wrong tool

If a tree model beats beamfeat by a wide margin, your signal is likely
piecewise, not algebraic. A single switch is within reach of `sign_shift`
(above); for anything more piecewise, use the tree. `fitted_ops` covers one constant
per operator, fitted one operator at a time; if your law needs several
constants fitted jointly (`exp(-a·x)·cos(b·x + c)`), constants inside sums
or products (`a·x0 + b·x1` under a logarithm), or deeper nesting than the
beam reaches, use a symbolic regressor such as PySR. The
[guarantees page](guarantees.md) states every boundary with the measurement
behind it.
