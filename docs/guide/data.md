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

A column with no variation cannot be selected — it standardises to zeros, so
its association with the target is zero wherever it appears. That is the right
outcome and costs nothing, but in the output it looks the same as a column
that simply does not matter, so it is reported. Real tables have such columns
more often than one would think: in the UJIIndoorLoc Wi-Fi fingerprints,
dozens of access points never register at all.

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
