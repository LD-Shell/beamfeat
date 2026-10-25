# Data and prediction

What beamfeat expects of a table, and how a fitted model behaves on new rows.
The examples start from:

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 6, (400, 4))
y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
model = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0).fit(X, y)
```

## Columns that carry no data

A column with no variation cannot be selected: it standardises to zeros, so
its association with the target is zero wherever it appears. That costs
nothing, but it looks the same as a column that does not matter, so it is
reported. Real tables have such columns more often than one would think: in
the UJIIndoorLoc Wi-Fi fingerprints, dozens of access points never register.

```python
stuck = np.full((400, 1), 9.81)               # a sensor stuck at one reading
BeamFeatRegressor(random_state=0).fit(np.hstack([X, stuck]), y)
```

```text
UserWarning: beamfeat: 1 of 5 columns are constant (x4) and cannot be selected.
They are ignored and cost nothing, but check whether they are meant to carry
data.
```

Constancy is relative to the column's magnitude, not an absolute floor on the
spread — a column varying over 0 to 1e-9 has a smaller variance than a stuck
sensor reading 9.81, and only the second is constant. Use `is_constant` to
apply the same test when screening a table before fitting.

A constant with a genuine unit is a different matter: gravity is m/s² whether
or not it varies, and expressions built from it are unit-checked, so label it.
Its absence from a recovered formula is not a miss — a multiplicative constant
is absorbed into the fitted coefficient rather than returned as a symbol.

## Real data with missing values or categoricals

beamfeat never imputes or encodes silently — compose explicitly:

```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

pipe = make_pipeline(SimpleImputer(strategy="median"),
                     BeamFeatRegressor(random_state=0))
```

## Rows outside a formula's domain

A fitted formula can be undefined on new data that the training data never
reached: `log(x0)` at a negative `x0`, or `1/(x0)` at zero. Such a row is NaN
in that formula's column of `transform`, and `predict` returns NaN for it; every
other row gets exactly the value it would get on its own, whatever else is in
the batch. A `UserWarning` names the formula and counts the rows. The
classifier's `predict` cannot return NaN as a label, so it assigns those rows
the most frequent training class, while `predict_proba` and
`decision_function` return NaN for them, which is how to find them. The
regressor's `score` raises rather than scoring the remaining rows.

Fitted constants do not change this. `(x0)^1.5` is NaN for a negative
`x0`, as `sqrt(x0)` is. A fitted pole keeps a
[margin](search.md#constants-inside-expressions) from the training rows, but a row
far enough outside them can still reach it: `(x1)^(-0.5)`, fitted where
`x1` lies between 1 and 3, is infinite at `x1 = 0`, as `1/(x1)` would be.

## Predicting with the original columns as well

The regressor predicts from the certified features alone, so the fitted model
is the equation it prints. When accuracy matters more than a readable model,
fit the final model on the original columns plus the certified features:

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

from beamfeat import BeamFeatTransformer

model = make_pipeline(
    make_union("passthrough", BeamFeatTransformer(random_state=0)),
    SimpleImputer(strategy="median"),  # rows outside a formula's domain
    StandardScaler(),
    RidgeCV(),
).fit(X, y)
```

This helps when the signal is spread thinly over many columns, which a few
certified features cannot capture but a penalised fit on every column can,
and when the certified features add structure a linear model on the raw
columns misses. Two things change:

- **The model is no longer interpretable.** It has a coefficient for every
  input column, so it reads like ridge regression, not like an equation.
- **The guarantee is unchanged but narrower in scope.** The FDR guarantee
  covers which constructed features are reported, and those are the same as
  without the original columns. It says nothing about the coefficients the
  final model puts on the raw columns.

If the combined model scores no better than ridge on the original columns
alone, the certified features are adding nothing to prediction on that data.

## New rows beyond the training data

A formula is fitted where the training rows lie, and beyond them it can
behave unlike anything the fit saw. A ratio is the usual case: its
denominator can stay clear of zero on the training rows and reach it a little
beyond them, where the quotient takes any size and either sign. Two things
guard against it.

The search rules out a quotient whose denominator crosses zero on the
search rows, or comes closer to it than a margin: twice the spread of its
ten values nearest zero, and at least 1% of its range. On the 172 training
samples of tecator's own split, the denominator of
`(x41 - x43) / (x41 - x54)` runs from -0.75 to -0.10, its ten values nearest
zero spread over more than half their distance from it; on 986 of 1,000
random halves of those samples, the search rules the formula out.

`predict`, `transform` and the probability methods compare each feature on
the new rows with its range on the training rows, kept in `feature_ranges_`,
and raise `ExtrapolationWarning` when a row takes a feature more than one
width of that range beyond it:

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1.0, 3.0, (400, 3))
y = X[:, 0] / X[:, 1] + rng.normal(0, 0.01, 400)
model = BeamFeatRegressor(max_depth=1, beam_width=8, random_state=0).fit(X, y)
print(model.formulas(), model.feature_ranges_)
```

```text
['(x0 / x1)'] [[0.35937791 2.7401815 ]]
```

```python
new_rows = np.array([[2.0, 2.0, 2.0], [2.0, 0.1, 2.0]])
print(model.predict(new_rows))
```

```text
ExtrapolationWarning: beamfeat: on 1 of 2 rows, a feature lies beyond its
training range by more than 1 width of that range: '(x0 / x1)' by up to 7.25
widths. The predictions on those rows extrapolate the equation, where a
formula can behave unlike anything the fit saw, such as a ratio whose
denominator nears zero; check them.
[ 1.00053387 19.99928935]
```

The predictions are returned unchanged. Here the formula is the law itself,
so 20.0 is right: the warning cannot tell a law from a formula that only
fits the training range. It says how many rows to check, and
`feature_ranges_` holds the ranges to find them. A row a little beyond the
training range does not raise it, and a model saved with an earlier release
has no `feature_ranges_` and predicts without the check.
