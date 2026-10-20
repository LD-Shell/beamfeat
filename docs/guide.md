# User guide

Everything below is runnable as-is; outputs shown are from real runs.

## Install

```bash
pip install beamfeat            # core: numpy + scikit-learn only
pip install "beamfeat[units]"   # + pint, for dimensional analysis
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
not describe the same set of terms.

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

## Why the equation says "1 of 25"

Screening certified twenty-five formulas. On a strong signal the marginal
null correctly passes every near-duplicate of the true feature — each one
really is associated with the target — so twenty-five is the honest size of
the certified set. A parsimony step then keeps the compact predictive subset
and fits that, because an equation of dozens of near-duplicate terms defeats
the interpretability the library exists for.

The subset is chosen on the rows selection already used, so the q-level
guarantee covers the twenty-five it came from and not the one that survived.
That is what the suffix records, and `fdp_inflation_` prices it:

```python
print(model.fdp_inflation_)
```

```text
25.0
```

|S|/|S'|, the factor by which pruning can inflate the realised false
discovery proportion, since the denominator shrinks faster than the numerator
can. Set `parsimony=None` to fit the screened set entire, at which point the
printed equation *is* the certified object and the suffix disappears:

```python
full = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0,
                         parsimony=None).fit(X, y)
print(full.n_features_out_)
print(full.equation(max_terms=3))
```

```text
25
y = 1.077*(x0 * x1) - 0.2264*(x0 * sqrt(x1)) - 0.5879*(sqrt(x1) - 1/(x0)) + 1.369
```

`max_terms` shortens the display, not the model: the terms it leaves out are
held at their training mean, and their average contribution moves into the
constant, which is why it differs from the full equation's.

What the default costs you in fit is small, and measured rather than asserted:
about a thousandth of held-out R² across 308 paired fits, for an equation some
twenty-five times shorter, with no case where pruning inflated the realised
false discovery proportion. The measurement is
`benchmarks/PARSIMONY_COST.md`. Short *and* certified costs rows instead of
the guarantee: see `parsimony_holdout` on the
[guarantees page](guarantees.md).

## The equation at every size

The forward step adds terms one at a time and stops when the next adds less
than `parsimony_tol` to R². `equations()` shows the step continued through the
whole screened set, one equation per number of terms, so you can see what the
stopping point traded away:

```python
print(model.equations(max_n_terms=3))
```

```text
Forward path through the 25 screened terms. R^2 is in-sample, on the rows screening used; gain is the R^2 the line's last term adds.
  k complexity    R^2      gain equation
* 1          4 0.9999    0.9999 y ~ (x0 * x1)
  2         10 0.9999 1.159e-06 y ~ (x0 * x1) + ((x0 + x3) * x1)
  3         18 0.9999 1.103e-06 y ~ (x0 * x1) + ((x0 + x3) * x1) + ((x0 * x1) * (x1 * x2))
  ...
* the fitted model (the forward step stops before the first gain below parsimony_tol=0.001).
The FDR guarantee covers the screened set of 25 terms (line 25), not an equation chosen from this list.
```

Complexity counts the nodes in each term's expression tree — operators and
input columns, `(x0 * x1)` being 3 — plus one per term for its coefficient
and one per [fitted constant](#constants-inside-expressions).
The lines are written `y ~ a + b`, without coefficients, because only the
marked one has been fitted; `equation()` prints its coefficients. For a
classifier the R² is that of the integer-coded labels, the heuristic the
forward step ranks by, not a classification metric.

The last line of the output is the one to keep in mind. Every line is a subset of the
screened set chosen on the rows screening used, so picking line 3 because it
looks better is the same kind of choice the forward step makes, and the
guarantee covers it no more than it covers the fitted line. Recording the path
changes nothing that is fitted, and it is recorded at `parsimony=None` as
well, where the marked line is the last.

The records behind the text are in `equations_`, one dict per line with
`n_terms`, `complexity`, `r2`, `formulas`, the `features` as expression
nodes, and `model`, which marks the fitted line. It is `None` without a
selector and empty when nothing passed screening.

## LaTeX

`latex()` typesets `equation()`: the same terms, coefficients and options
(`precision`, `max_terms`, `style`), and the same note when the guarantee
covers more terms than are printed. It needs sympy (`pip install
"beamfeat[display]"`), as `to_sympy()` does.

```python
print(model.latex())
print(model.latex_formulas())
```

```text
y = 0.9999 x_{0} x_{1} - 0.003943 \quad \text{[1 of 25 certified terms; parsimony=None prints all 25]}
['x_{0} x_{1}']
```

`latex(note=False)` leaves the note off, for pasting into a document; the
guarantee still covers the screened set, so say so where the equation
appears. The string has no math delimiters; wrap it in `$...$`, or in a
notebook use `IPython.display.Math(model.latex())`. The classifier's `latex()` gives the
log-odds of the positive class, or one softmax score per class in an
`aligned` block, as its `equation()` does.

## Export to NumPy, JAX or PyTorch

`to_sympy()` returns each feature as a sympy expression in real symbols named
after the input columns, so `sympy.lambdify` turns the features into a
function for any array library sympy can print to. No extra dependency is
involved beyond sympy and the target library:

```python
import sympy

columns = sympy.symbols("x0:4", real=True)  # one symbol per input column, in order
features = sympy.lambdify(columns, model.to_sympy(), modules="numpy")

raw = model.coef_ / model.scaler_.scale_
offset = model.intercept_ - raw @ model.scaler_.mean_
prediction = np.column_stack(features(*X.T)) @ raw + offset  # model.predict(X), to rounding
```

`modules="jax"` or `modules="torch"` gives the same function on JAX or
PyTorch arrays, with `jnp.stack(..., axis=1)` or `torch.stack(..., dim=1)` in
place of `np.column_stack`; the coefficients and offset are plain arrays to
convert once. Columns of a DataFrame keep their names as symbols when the
names are valid identifiers, and are sanitised otherwise; read the symbols
from `free_symbols` of the expressions in that case.

Two differences from `transform` remain. sympy may rewrite an expression into
an equivalent form, so a value can overflow in one and not the other. And a
row outside a formula's domain gives whatever the array library gives there,
with no warning from beamfeat.

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

## Dimensional analysis

Give units as pint quantities or plain strings; dimensionally invalid
expressions (metres plus kilograms) are rejected before any numerical work — so `x0 + x1` (kg plus m) is never even scored:

```python
y_phys = X[:, 0] * X[:, 1] / X[:, 2]          # kg·m/s
model = BeamFeatRegressor(
    units={"x0": "kg", "x1": "m", "x2": "s"}, random_state=0
).fit(X[:, :3], y_phys)
print(model.formulas())
```

```text
['((x1 / x2) * x0)']
```

The recovered form, kg·m/s, is exactly the target's dimension — and the
dimensionally invalid spellings never consumed a beam slot.

Cover every column. A column without a unit is dimensionally
unconstrained, so it combines freely with the labelled ones and the check
stops binding on exactly the columns you did not vouch for — labelling your
real measurements and leaving the noise blank is the case worth naming,
because it looks careful and is not. Give the genuinely unitless columns
`"dimensionless"`; the estimator warns when coverage is partial, and raises
when the keys match no column at all. Keys are column names, so a DataFrame
lets you write `{"rho": "kg/m**3"}` rather than `{"x0": ...}`.

## Constants inside expressions

Expressions are free of constants by default; the linear model supplies the
coefficients outside them. That cannot write `exp(-x²/2)`, `x^1.5` or
`(x0 + x1)/(1 + x0·x1)`, where the constant sits inside a nonlinearity.
`fitted_ops` adds operators that carry one fitted constant each:

| name | expression | operand |
|---|---|---|
| `"exp_scaled"` | `exp(a·x)` | dimensionless |
| `"pow"` | `x^a` | positive, dimensionless |
| `"inv_shift"` | `1/(x + a)` | any; `a` takes the operand's unit |
| `"log_shift"` | `log(x + a)` | `x + a` positive, dimensionless |

```python
rng = np.random.default_rng(0)
X = rng.uniform(0.1, 1.0, (500, 2))
y = np.exp(-X[:, 0] ** 2 / 2) + rng.normal(0, 1e-3, 500)
model = BeamFeatRegressor(max_depth=2, fitted_ops=("exp_scaled",), random_state=0).fit(X, y)
print(model.formulas(), model.to_sympy())
```

```text
['exp(-0.5*(x0)^2)'] [exp(-x0**2/2)]
```

Without `fitted_ops` the same data give `(x0 + (x0)^2)` and `log(x0)`, a
surrogate that scores well on this range and says nothing about the law.

How the constants are chosen:

- **Where.** At each depth, every listed operator is fitted to every
  expression in the previous depth's beam (at the first depth, the
  `beam_width` best input columns), and to the previous depth's
  `beam_width` best candidates that the beam dropped as near-duplicates:
  two expressions that correlate at 0.95 need not stay alike under
  `1/(x + a)` or `exp(a·x)`. The results join that depth's candidates, to
  be scored, pruned and kept like any other. The ceiling
  `max_candidates_per_depth` counts the operator proposals only.
- **How.** Each constant maximises the absolute correlation with the target:
  a coarse grid scaled to the operand (for `exp_scaled`, `|a|` up to
  10/max|x|; for the shifts, poles kept off the observed range), then a
  golden-section refinement. A fit that runs to the edge of its grid, or
  that does not clearly beat the operand on its own (each operator becomes
  linear in `x` at one end of its range), is discarded. Correlation is used
  whatever the `scorer`; the scorer then ranks the result.
- **Poles.** `inv_shift` and `log_shift` keep their pole away from the
  data they were fitted on: at least twice the spread of the operand's ten
  most extreme values at that end, and at least 1% of its range. A pole
  just past the edge of the data is shaped by the few rows next to it, and
  the first new row a little further out lands on it and evaluates to NaN.
  A fit that wants its pole closer runs into the bound and is discarded,
  and snapping never moves a constant past it. The margin widens where the
  data thin out towards the edge and narrows where they are dense. On 18
  real datasets, poles inside it were crossed by a row the fit had not
  seen in 14% of fits, and poles outside it in 0.3%.
- **In context.** `1/(x0·x1 + a)` on its own correlates best with the target
  as `a` runs to the pole; only as the denominator of `(x0 + x1)` is `a = 1`
  identifiable. While the depth limit leaves room, each constant is also
  fitted as a factor of a beam member, and that product joins the next
  depth's candidates.
- **Snapping.** The constant is replaced by the simplest of 0, ±1, ±2, ±3,
  ±1/2, ±3/2, ±1/3, ±1/4 that the data cannot tell apart from the optimum
  (a likelihood-ratio test at the 99.9% point), so `exp(-0.49991·x²)` is
  reported as `exp(-x²/2)`; failing that it is rounded to the fewest
  significant digits that cost nothing measurable. A snapped constant is
  exact in `to_sympy()`, `latex()` and the printed formulas.
- **Stored.** The constant is part of the expression: `transform`,
  `predict` and a pickled model reuse it on new data without refitting.

The constants are fitted inside the search, on the search rows only, so the
screen on the selection rows sees fixed expressions and its guarantee is
unchanged; see the [guarantees page](guarantees.md#selective-inference-why-the-estimators-split-their-data).

What it costs. Fitting adds a few candidates per parent and operator, and
makes a fit on the Feynman panel about five times slower (about 0.2 s to
1.1 s per equation). Fitted expressions also take beam slots, so enabling
them can change what else a depth keeps: a constant-free candidate is never
dropped as a near-duplicate of a fitted one, but it can be out-ranked. And
constants are fitted one at a time, greedily: a law needs its constant-free
parts to be among a depth's best candidates first.

## Columns that carry no data

A column with no variation cannot be selected — it standardises to zeros, so
its association with the target is zero wherever it appears. That is the right
outcome and costs nothing, but in the output it looks the same as a column
that simply does not matter, so it is reported:

```text
beamfeat: 55 of 521 columns are constant (WAP003, WAP004, WAP092, ...) and
cannot be selected. They are ignored and cost nothing, but check whether they
are meant to carry data.
```

Constancy is relative to the column's magnitude, not an absolute floor on the
spread — a column varying over 0 to 1e-9 has a smaller variance than a stuck
sensor reading 9.81, and only the second is constant. Use `is_constant` to
apply the same test when screening a table before fitting.

A constant with a genuine unit is a different matter: gravity is m/s² whether
or not it varies, and expressions built from it are unit-checked, so label it.
Its absence from a recovered formula is not a miss — a multiplicative constant
is absorbed into the fitted coefficient rather than returned as a symbol.

## Audit what selection did

Every candidate's permutation p-value and its adjusted q-value is kept:

```python
for row in model.selection_report_[:3]:
    print(row["formula"], round(row["q_value"], 4), row["kept"])
```

Features are *kept* only if they pass FDR screening at `target_fdr`
(default 0.1, Benjamini–Yekutieli) on a held-out split, then survive the
parsimony pass. `screened` marks the certified set, which is what the
guarantee covers; `kept` marks the compact subset the equation prints. Set
`parsimony=None` and the two agree.

## Honest failure, by default

On a target with no structure, nothing passes — and the model says so
instead of returning junk:

```python
noise = rng.standard_normal(400)
model = BeamFeatRegressor(random_state=0).fit(X, noise)
# NoDiscoveriesWarning: ... Returning no constructed features ...
print(model.equation())
```

```text
y = 0.03527  (no feature passed selection)
```

Not on every draw. Under a global null every selection is false, so the FDR
is the probability that a fit certifies anything at all: up to `target_fdr`
of pure-noise fits still certify features, and when one does, correlated
candidates tend to pass together. The same noise under another search seed
can certify a dozen formulas with `fdr_controlled_ = True`: that is the error
the guarantee bounds in expectation, not a breach of it.

Prefer the old behaviour? `on_no_discoveries="fallback"` keeps the
unfiltered search output (flagged `fdr_controlled_=False`); `"raise"`
raises.

## Classification

```python
from beamfeat import BeamFeatClassifier

labels = (X[:, 0] * X[:, 1] > X[:, 2] * X[:, 3]).astype(int)
clf = BeamFeatClassifier(max_depth=2, random_state=0).fit(X, labels)
print(clf.equation())          # log-odds form; boundary = zero level set
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
regressor's `score` raises rather than scoring the remaining rows. To rule
such rows out within the training ranges, fit with `safe_domain=True` (next
section).

Fitted constants do not change this. `(x12)^(-0.127)`, fitted where `x12`
is at least 0.02, is infinite at `x12 = 0`, as `1/(x12)` would be, and
`(x0)^1.5` is NaN for a negative `x0`, as `sqrt(x0)` is. A fitted
[shift](#constants-inside-expressions) keeps its pole a margin away from the
training rows, but a row far enough outside them can still reach it.
`safe_domain` with the columns' physical bounds declared, such as
`{"x12": (0, 1)}`, rules all of these out.

## Search constraints

Two options restrict what the search will propose. Both are checked as each
candidate is constructed, before any numerical work, so a rejected candidate
costs no evaluation and takes no beam slot. Rejections are counted per depth
in `search_result_.trace` (`n_rejected_constraints`).

**`safe_domain`** rules out the NaN above, inside a box. With `True`, every
feature is guaranteed defined wherever each input lies in its training range,
not only on the training rows:

```python
model = BeamFeatRegressor(safe_domain=True, random_state=0).fit(X, y)
model.input_bounds_          # {"x0": (lo, hi), ...}: where predict is defined
```

A feature such as `log(x0 - x1)`, fitted on data where `x0 > x1` on every row
but whose ranges overlap, is rejected, because the box contains points with
`x0 < x1`. Declare bounds for columns whose physical range you know, and the
training range is used for the rest: `safe_domain={"T": (0, np.inf)}` says a
temperature is never negative. Each operator's bounds come from interval
arithmetic (every built-in operator has them; an operator registered without
`interval` and, if it has a `domain`, `interval_domain` is never admitted).
Interval arithmetic treats repeated columns as independent, so it can
overestimate: `sqrt(x0 - x0 * x1)` with `x1` below 1 is defined throughout
but may be rejected. That is the price of the guarantee. See
[guarantees](guarantees.md#where-predictions-are-defined-safe_domain).

**`max_inputs`** caps the distinct input columns in one feature:
`max_inputs=2` admits `x0 * log(x0 / x1)` but not `x0 * x1 / x2`. Use it when
a feature that mixes many columns would be uninterpretable to you anyway.

Neither option makes selection more powerful. The screen's multiple-testing
correction runs over the features that reach selection, which the beam
already caps at `beam_width` and `max_features` however many expressions were
proposed; constraints change which features those are and how long the
search takes.

## Real data with missing values or categoricals

beamfeat never imputes or encodes silently — compose explicitly:

```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

pipe = make_pipeline(SimpleImputer(strategy="median"),
                     BeamFeatRegressor(random_state=0))
```

## Predicting with the original columns as well

The regressor predicts from the certified features alone, so the fitted model
is the equation it prints. When accuracy matters more than a readable model,
fit the final model on the original columns plus the certified features:

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

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

## When to reach for something else

If a tree model beats beamfeat by a wide margin, your signal is likely
piecewise, not algebraic — use the tree. `fitted_ops` covers one constant
per operator, fitted one operator at a time; if your law needs several
constants fitted jointly (`exp(-a·x)·cos(b·x + c)`), constants inside sums
or products (`a·x0 + b·x1` under a logarithm), or deeper nesting than the
beam reaches, use a symbolic regressor such as PySR. The [guarantees page](guarantees.md) states every boundary with the
measurement behind it.
