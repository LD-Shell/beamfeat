# Quickstart

Install beamfeat, fit a model and read what it prints. Every example runs
as shown, and the outputs are from real runs. The
[Getting started](notebooks/01_getting_started.ipynb) tutorial covers the
same ground in a notebook.

## Install

```bash
pip install beamfeat              # core: numpy, scikit-learn, joblib
pip install "beamfeat[units]"     # + pint, for dimensional analysis
pip install "beamfeat[display]"   # + sympy, for to_sympy() and latex()
pip install "beamfeat[all]"       # + both, pandas and the test dependencies
```

See [Installation](installation.md) for running the test suite or
reproducing the benchmarks.

## Sixty seconds to a vetted equation

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 6, (400, 4))
y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)

model = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0).fit(X, y)
print(model.equation())
print(model.fdr_controlled_)
```

```text
y = 0.9999*(x0 * x1) - 0.003943   [1 of 25 certified terms; parsimony=None prints all 25]
True
```

Three things matter. `equation()` is the fitted model itself — evaluate it on
raw feature values and you reproduce `predict()`. `fdr_controlled_` states
whether the features carry the false-discovery-rate guarantee; check it before
treating them as statistically vetted. And the suffix says that those two do
not describe the same set of terms;
[Reading a fit](guide/reading-a-fit.md#why-the-equation-says-1-of-25)
explains it.

That last point has an attribute as well as a suffix, because a script never
reads the printed string:

```python
print(model.fdr_controlled_, "|", model.fdr_scope_)
```

```text
True | screened set
```

`fdr_controlled_` says whether there is a guarantee. `fdr_scope_` says what it
is over: `"screened set"` when the printed terms were pruned from a larger
certified set, `"printed equation"` when the guarantee covers the terms you
are looking at, `None` when there is no guarantee to scope. Branch on
`fdr_scope_`, not on the equation text.

## Column names flow into formulas

Fit on a DataFrame and formulas use your names:

```python
import pandas as pd

df = pd.DataFrame({"mass": rng.uniform(1, 5, 300),
                   "vol":  rng.uniform(1, 5, 300)})
target = df["mass"] / df["vol"] + rng.normal(0, 0.02, 300)
print(BeamFeatRegressor(random_state=0).fit(df, target).formulas())
```

```text
['(mass / vol)']
```

## Classification

`BeamFeatClassifier` prints the log-odds of the positive class, whose zero
level set is the decision boundary:

```python
from beamfeat import BeamFeatClassifier

labels = (X[:, 0] * X[:, 1] > X[:, 2] * X[:, 3]).astype(int)
clf = BeamFeatClassifier(max_depth=2, random_state=0).fit(X, labels)
print(clf.equation(max_terms=3))
```

```text
logit P(y = 1) = - 0.1685*((x2 * x3) - (x0 * x1)) - 1.313*((x2 / x0) + (x3 / x1)) - 1.185*((x2 / x1) - (x0 / x3)) + 2.709   [20 of 50 certified terms; parsimony=None prints all 50]
```

The fitted equation has 20 terms; `max_terms=3` displays the three with the
largest standardised coefficients and holds the rest at their training mean.
With `parsimony_alpha=0.05` the model itself has two terms, and its accuracy
on 2,000 new rows is the same, 0.982 against 0.9815:

```python
short = BeamFeatClassifier(max_depth=2, random_state=0, parsimony_alpha=0.05).fit(X, labels)
print(short.equation())
```

```text
logit P(y = 1) = - 0.526*((x2 * x3) - (x0 * x1)) - 1.005*((x2 / x0) + (x3 / x1)) + 2.455   [2 of 50 certified terms; parsimony=None prints all 50]
```

## Next

- [Choosing a setting](settings.md): the default for prediction,
  `parsimony_alpha` for a short equation, `input_test` for an equation whose
  every column was tested.
- [User guide](guide/index.md): reading a fit, the printed equation, the
  search options, and data and prediction.
- [Tutorials](notebooks/01_getting_started.ipynb): four notebooks with real
  outputs.
- [Statistical guarantees](guarantees.md): what the guarantee covers, and
  what was measured.
