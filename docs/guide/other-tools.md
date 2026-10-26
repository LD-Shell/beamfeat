---
description: "Screen the equation another symbolic regression tool reports (PySR, Operon, gplearn, PySINDy) on rows it never saw, with false discovery rate control over its terms."
---

# Screening another tool's equation

A symbolic regression tool such as PySR, Operon, gplearn or PySINDy reports an
equation with no statement about whether its terms are real.
[`screen_equation`](../api.md#beamfeat.screen) tests each term on rows the
tool never saw, with the permutation test beamfeat's own fits use, and keeps
the terms whose null is rejected under false discovery rate control. It needs
sympy (`pip install 'beamfeat[display]'`).

## Hold rows out, then screen

The guarantee needs rows the tool did not search. Split first, fit the tool on
one part, and screen on the other:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from beamfeat import screen_equation

rng = np.random.default_rng(0)
X = rng.uniform(1, 3, size=(600, 4))
y = 1.5 * X[:, 0] + 0.8 * X[:, 1] ** 2 + rng.normal(scale=0.5, size=600)
X_search, X_held, y_search, y_held = train_test_split(X, y, test_size=0.5, random_state=0)

# the tool searches X_search only; say it reports this equation
reported = "1.5*x0 + 0.8*x1**2 + 0.05*x0*x3 + 0.02*x2 + 0.1"

found = screen_equation(reported, X_held, y_held)
print(found.summary())
```

```text
screen_equation: 3 of 4 terms kept (marginal BY at q = 0.1, 300 rows)
  [kept] 1.5*x0: p = 0.0004998, null rejected
  [kept] 0.8*x1**2: p = 0.0004998, null rejected
  [removed] 0.02*x2: p = 0.1654, null not rejected
  [kept] 0.05*x0*x3: p = 0.0004998, null rejected
```

Each top-level term of the sum is one candidate, and the constant is not a
term. A term's null is that it is independent of `y` on these rows. The
Benjamini-Yekutieli step-up, the default, controls the false discovery rate
among the kept terms under any dependence between them; `correction="bh"`
uses Benjamini-Hochberg, which needs positive dependence. The level is `q`
(default 0.1).

With PySR, pass the model's equation directly: `screen_equation(model.sympy(),
X_held, y_held)`. A model fitted on an array names its columns `x0, x1, ...`;
for a DataFrame, the equation uses its column names and `screen_equation`
reads them from the DataFrame you pass, or from `feature_names`.

## Every column tested: the input test

Above, `0.05*x0*x3` is kept although `x3` has nothing to do with `y`: the term
rides on `x0`. `input_test=True` also tests each variable inside each term of
more than one, by permuting that variable's rows inside the term alone:

```python
found = screen_equation(reported, X_held, y_held, input_test=True)
print(found.summary())
```

```text
screen_equation: 2 of 4 terms kept (marginal BY at q = 0.1, input test, 300 rows)
  [kept] 1.5*x0: p = 0.0004998, null rejected
  [kept] 0.8*x1**2: p = 0.0004998, null rejected
  [removed] 0.02*x2: p = 0.1654, null not rejected
  [removed] 0.05*x0*x3: p = 0.0004998, null rejected; inputs: x0 rejected, x3 not rejected
```

A variable's null is that it is independent of `y` and of the term's other
variables together, and the step-up runs over the (term, variable) pairs. The
test costs power where a tool's terms mix true and false variables, since it
removes such a term whole. A variable correlated with a true input is not null
in this sense, so a stand-in for a true input can pass: no test of association
can tell the two apart.

## What to do with the result

`found.kept_terms` holds the kept terms as the tool wrote them. To use them,
refit their coefficients on the rows the tool searched, `X_search`, never on
the rows that screened them. `found.p_values`, `found.marginal`,
`found.untestable` and the input test's pairs and p-values are on the result
as well (see the [API reference](../api.md#beamfeat.screen)).

A term undefined or constant on any held-out row cannot be tested: it is
reported as untestable and never kept. A term the screen does not keep has no
support on these rows, which is not a finding that it is false; with a few
dozen rows a real term of modest effect is often not kept.

## A term the sum would split

sympy distributes a number over a sum, so reading `2*(x0 + x1)` gives two
terms. To test the terms exactly as a tool reported them, pass a list, each
item tested whole:

```python
screen_equation(["1.5*x0", "0.4*(x1**2 + x2)"], X_held, y_held)
```
