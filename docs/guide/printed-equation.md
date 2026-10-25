---
description: "How beamfeat chooses the printed equation from the certified set: parsimony_alpha, input_test, parsimony=None and parsimony_holdout."
---

# The printed equation

The screen certifies a set of formulas, and the equation is chosen from that
set. By default a forward step adds the certified formula that improves the
fit most, one at a time, and stops when the next would add less than
`parsimony_tol` (0.001) to R². The options on this page change that choice.
None of them changes the search or the screen, so the FDR guarantee on the
certified set is the same under each. [Choosing a setting](../settings.md)
compares the default with the two options most fits need, `parsimony_alpha`
and `input_test`.

## A short equation (`parsimony_alpha`)

On a noisy target the default keeps riders: the best of the many remaining
formulas adds more than `parsimony_tol` to R² by fitting noise alone.
`parsimony_alpha=0.05` also stops the forward step at the first term whose
gain fails to reject the null that chance reaches it. Here the target is
`x0 * x1` with 25% noise, among ten unrelated columns:

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(11)
X = rng.uniform(1, 6, (500, 12))
signal = X[:, 0] * X[:, 1]
y = signal + rng.normal(0, 0.25 * np.std(signal), 500)
print(BeamFeatRegressor(random_state=0).fit(X, y).equation())
print(BeamFeatRegressor(random_state=0, parsimony_alpha=0.05).fit(X, y).equation())
```

```text
y = 1.052*(x0 * x1) - 0.4088*((x0 * x1) / (x0 + x6)) - 0.1452*((x3 / x0) - x1) + 0.1295*((x5 / x0) - x1) + 0.04642   [4 of 50 certified terms; parsimony=None prints all 50]
y = 1.013*(x0 * x1) - 0.1608   [1 of 50 certified terms; parsimony=None prints all 50]
```

The default's other three terms attach unrelated columns, `x6`, `x3` and
`x5`, to the product. At a signal-to-noise ratio of 1 the default printed 18
terms on average where one was true, and the check printed 1.1, with held-out
R² slightly higher on average and a little lower in a few fits.

On real data the check costs accuracy. Real targets tend to carry many
modest effects, each adding little, so the step stops after a few terms
(one to five on 10 of 12 real datasets), and held-out score fell on 9 of the
12 (diabetes 0.433 to 0.381, tips 0.472 to 0.433). Use it to look for a short
law, not for prediction. It is a stopping rule, not a guarantee on the terms
it keeps; the path in `equations()` is the same either way, and only the
marked line moves.
[Choosing a setting](../settings.md#what-was-measured) and the
[guarantees page](../guarantees.md#parsimony-within-the-screened-set) have
the figures.

## Every column tested (`input_test`)

The screen tests each term as a whole, so a term that attaches a junk column
to the true one is still a true discovery: `((x0 * x1) / (x0 + x6))` is
associated with the target through `x0 * x1`, however unrelated `x6` is.
`input_test=True` tests each column inside each term that reads more than
one. The column's rows are permuted wherever it occurs in the expression,
and its null is that it is independent of the target and of the term's
other columns. A screened term is dropped before the forward step when the
test fails to reject the null for any of its columns. On the same data:

```python
print(BeamFeatRegressor(random_state=0, input_test=True).fit(X, y).equation())
```

```text
y = 1.013*(x0 * x1) - 0.1608   [1 of 50 certified terms; parsimony=None prints all 8; input_test dropped 42]
```

`input_report_` lists every (term, column) pair with its p- and q-value, and
`null_rejected` marks the pairs whose null the test rejected. A junk column
correlated with a real one is not null in this sense, so its null can be
rejected and the term kept; the
[guarantees page](../guarantees.md#columns-inside-a-term-input_test) states
what the test covers. It is opt-in because its time grows with the rows:
here the fit took about four times as long, and `n_jobs` threads share the
tests.

## The whole certified set (`parsimony=None`)

`parsimony=None` skips the forward step and fits every certified formula.
The guarantee then covers the printed equation itself (`fdr_scope_` is
`"printed equation"`), and the equation is long: 25 terms in the
[first example](reading-a-fit.md#why-the-equation-says-1-of-25) of Reading a
fit, dozens on wide data. `equation(max_terms=...)` shortens what is
displayed without changing the model.

## Re-testing the printed terms (`parsimony_holdout`)

`parsimony_holdout` holds back that fraction of the selection rows.
Screening and the forward step run on the rest, and the terms they return,
fixed at that point, are tested again on the rows held back; terms whose
null the re-test fails to reject are dropped. By default the re-test is the
screen again (`certify_test="marginal"`), which tests each term's
association with the target on its own. It rejects the null for a term that
only rides on a true one, since such a term is associated with the target
through the true one. `certify_test="conditional"` tests each term given the
other printed terms instead:

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
X = rng.uniform(0.5, 3.0, (600, 8))
y = 2 * X[:, 0] + rng.normal(0, 0.5, 600)  # x0 alone

# Products only, and the whole screened set: every product of x0 with
# another column rides on x0, and the re-test sees them all.
options = dict(max_depth=1, binary_ops=("mul",), unary_ops=(), parsimony=None,
               parsimony_holdout=0.5, random_state=0)
marginal = BeamFeatRegressor(**options).fit(X, y)
conditional = BeamFeatRegressor(**options, certify_test="conditional").fit(X, y)
print(marginal.formulas())
print(conditional.equation())
```

```text
['x0', '(x0 * x7)', '(x0 * x6)', '(x0 * x3)', '(x0 * x2)', '(x0 * x1)', '(x0 * x5)', '(x0 * x4)']
y = 2.009*x0 - 0.003251
```

The conditional re-test is a permutation test for regression only, and its
p-values are approximate; the guarantees page states its null and what its
validity rests on. It works when the subset it receives is short. On a long
subset whose terms nearly reproduce one another, no term adds anything given
the rest, the re-test rejects no null, and the estimator returns the whole
screened set with only the screen's marginal guarantee
(`certification_result_.n_selected` is 0, and a `NoDiscoveriesWarning` says
so). After the forward step that happened in about nine fits in ten at a
signal-to-noise ratio of 1 (`benchmarks/parsimony_holdout/`). So after the
forward step the conditional re-test needs `parsimony_alpha` (above), which
brings the fallback under one fit in a hundred; without it the combination
is deprecated since 0.7.0 and will raise from 0.8. The example above
re-tests the whole screened set (`parsimony=None`) and needs no
`parsimony_alpha`. Both re-tests cost rows: hold back a fraction only when
the selection rows number a few hundred or more.
