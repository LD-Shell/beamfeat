# User guide

Everything below is runnable as-is; outputs shown are from real runs.

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

The screen rejected the null for twenty-five formulas, the certified set.
Each formula's null is that it is independent of the target, and on a strong
signal every near-duplicate of the true feature really is associated with
the target, so rejecting its null is correct and twenty-five is the honest
size of the certified set. A parsimony step then selects the compact
predictive subset and fits that, because an equation of dozens of
near-duplicate terms defeats the interpretability the library exists for.

The subset is chosen on the rows selection already used, so the q-level
guarantee covers the twenty-five it came from and not the one the forward
step retained.
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
about a thousandth of R² across 308 paired fits (held out in every study but
the end-to-end calibration, which scores in-sample), for an equation some
twenty-five times shorter, with no case where pruning inflated the realised
false discovery proportion. The measurement is
`benchmarks/PARSIMONY_COST.md`. Short *and* certified costs rows instead of
the guarantee: see `parsimony_holdout` on the
[guarantees page](guarantees.md), and the next section.

## Re-testing the printed terms

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
validity rests on. It works when the subset it receives is short. When the
subset is long and its terms nearly reproduce one another, no term adds
anything given the rest, the re-test rejects no null, and the estimator
returns the whole screened set, which carries only the screen's marginal
guarantee; `certification_result_.n_selected` is then 0, and a
`NoDiscoveriesWarning` says so. In the calibration study
(`benchmarks/parsimony_holdout/`) that happened in about nine fits in ten at
a signal-to-noise ratio of 1, where the forward step retains 18 terms on
average, and in about one in five at a
ratio of 3. At that ratio, the share of printed terms using a column outside
the generating formula was 0.42 to 0.44 at the defaults, 0.53 to 0.62 with
the marginal re-test and 0.18 to 0.22 with the conditional one, which
printed about one term in the fits it did not hand back to the screened
set. Both re-tests cost rows: hold back a fraction only when the selection
rows number a few hundred or more.

## Columns that ride along inside a term

The screen tests each term as a whole, so a term that attaches a junk column
to the true one is still a true discovery: `((x0 * x1) / (x0 + x6))` is
associated with the target through `x0 * x1`, however unrelated `x6` is.
`input_test=True` tests each column inside each term that reads more than
one. The column's rows are permuted wherever it occurs in the expression,
and its null is that it is independent of the target and of the term's
other columns. A screened term is dropped before the forward step when the
test fails to reject the null for any of its columns:

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(11)
X = rng.uniform(1, 6, (500, 12))
signal = X[:, 0] * X[:, 1]
y = signal + rng.normal(0, 0.25 * np.std(signal), 500)
print(BeamFeatRegressor(random_state=0).fit(X, y).equation())
print(BeamFeatRegressor(random_state=0, input_test=True).fit(X, y).equation())
```

```text
y = 1.052*(x0 * x1) - 0.4088*((x0 * x1) / (x0 + x6)) - 0.1452*((x3 / x0) - x1) + 0.1295*((x5 / x0) - x1) + 0.04642   [4 of 50 certified terms; parsimony=None prints all 50]
y = 1.013*(x0 * x1) - 0.1608   [1 of 50 certified terms; parsimony=None prints all 8; input_test dropped 42]
```

`input_report_` lists every (term, column) pair with its p- and q-value, and
`null_rejected` marks the pairs whose null the test rejected. The screen's
guarantee is unchanged and still covers the screened set of 50. A junk
column that is correlated with a real one is not null in this sense, so its
null can be rejected and the term kept; the
[guarantees page](guarantees.md#columns-inside-a-term-input_test) states
what the test covers and what was measured. It is opt-in because it costs
time that grows with the rows: here the fit took about seven times as long,
and `n_jobs` threads share the tests.

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

On a noisy target that rule keeps riders: the best of many remaining terms
adds more than `parsimony_tol` by fitting noise alone, and at a
signal-to-noise ratio of 1 the step kept 18 terms on average where one was
true. `parsimony_alpha=0.05` (off by default) also stops the step at the
first term whose gain fails to reject the null that chance reaches it. In the
same study it printed 1.1 terms on average, held-out R² rose slightly on
average, and a few fits lost a little; the
[guarantees page](guarantees.md#parsimony-within-the-screened-set) has the
figures. The path in `equations()` is the same either way; only the marked
line moves.

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
selector and empty when the screen rejected no null.

## One equation or one of many

A fit prints one equation, found on one random split of the training rows
into the part the search sees and the part selection tests on. Where many
expressions predict about equally well, another split prints another
equation, and nothing in a single fit says which case you are in.
`stability_report` finds out. It refits clones of the estimator on random
splits of the data, scores each once on a test part its fit never saw, and
counts the distinct equations:

```python
from beamfeat import stability_report

report = stability_report(BeamFeatRegressor(max_depth=2, beam_width=25), X, y, n_splits=30)
print(report)
```

```text
BeamFeatRegressor refitted on 30 random splits (random_state=0), each fitted on 300 rows and scored on the other 100.

Held-out R^2: mean 1.000, sd 0.000, min 1.000, max 1.000.
1 distinct equation in 30 splits: every split printed it.
1 distinct feature, selected in 30 of the 30 splits.

Equations printed in more than one split, most frequent first:
  30/30  y ~ (x0 * x1)

Features, most frequent first:
  30/30  (x0 * x1)

Features whose values on the rows of X correlate above 0.999 in absolute value are counted as one; an equation is the set of its features, whatever its coefficients.
```

Every split found the planted product. The diabetes data are another matter:

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
report = stability_report(BeamFeatRegressor(), diabetes.data, diabetes.target, n_splits=30)
print(report.summary(max_rows=3))
```

```text
BeamFeatRegressor refitted on 30 random splits (random_state=0), each fitted on 331 rows and scored on the other 111.

Held-out R^2: mean 0.496, sd 0.050, min 0.389, max 0.575.
30 distinct equations in 30 splits; none recurs.
86 distinct features; the most frequent in 11 of the 30 splits.

Features, most frequent first:
  11/30  ((bmi + s5) + (bp - s3))
   8/30  ((bmi + s5) - (s3 + sex))
   8/30  ((bmi + bp) - (s3 + sex))
  ... and 83 more features

Features whose values on the rows of X correlate above 0.999 in absolute value are counted as one; an equation is the set of its features, whatever its coefficients.
```

No two of the thirty splits printed the same equation, and their held-out
R² ranges from 0.389 to 0.575, a spread that one fit's score would hide. An
equation fitted to these data is one draw among many that predict about as
well. What recurs is a
family of sums and differences of `bmi`, `s5`, `bp` and `s3`, the commonest
member in 11 of the 30 splits. Report that family, with its frequencies,
rather than one equation.

Two formulas count as one feature when their values on the rows of `X`
correlate above 0.999 in absolute value. A rewriting such as `(x1 - x0)` for
`(x0 - x1)` is then not counted twice, and neither is a rescaling or a
near-duplicate. An equation is the set of its features, whatever its
coefficients. The report keeps everything behind the summary:

- `report.splits[i]` has the rows split `i` was fitted and scored on, its
  formulas, its `equation()`, its score and the fitted clone;
- `report.features` and `report.equations` hold the classes, most frequent
  first, each with the splits it appeared in;
- `report.scores` holds the held-out scores.

The splits and the clones' seeds come from `random_state` (default 0), so a
report is reproducible, and `n_jobs` fits splits in parallel processes
without changing it. The regressor is scored by R² and the classifier by
accuracy, as their `score` methods are. A transformer has no score of its
own, so it is scored by a ridge regression on its features, or by a logistic
regression for classification. When a test row is outside a selected
formula's [domain](#rows-outside-a-formulas-domain), the classifier predicts
its most frequent training class there, as its `predict` does. A regressor or
transformer has no score on such a test part, so that split goes unscored,
and the summary counts it.

`resample` chooses what varies. The default, `"splits"`, draws new training
and test rows as well as a new seed for every split. It measures
split-to-split variability: how far the equation and its score move with
another sample of the data. `resample="seeds"` keeps the first split's rows
and varies only the seed, and with it the search/selection split inside each
fit. That design, the one in the paper's stability study (Appendix E),
measures seed-to-seed variability on fixed data: how much the printed
equation owes to one random draw, the draw the FDR guarantee is conditional
on. Run on diabetes as above with `resample="seeds"`, it still prints 30
distinct equations in 30 fits, with 84 distinct features and the commonest
in 10. The standard deviation of R² falls from 0.050 to 0.008, though, so
most of the default's spread comes from the test parts, while the turnover of
equations persists with the rows held fixed.

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

`modules="jax"` or `modules="torch"` (sympy 1.14 or newer) gives the same
function on JAX or PyTorch arrays, with `jnp.stack(..., axis=1)` or `torch.stack(..., dim=1)` in
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
expressions (metres plus kilograms) are ruled out before any numerical work —
so `x0 + x1` (kg plus m) is never even scored:

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

The linear model supplies the coefficients outside the expressions. That
alone cannot write `exp(-x²/2)`, `x^1.5` or `(x0 + x1)/(1 + x0·x1)`, where the
constant sits inside a nonlinearity, so the estimators also use operators that
carry one fitted constant each, listed in `fitted_ops`. All four are on by
default (since 0.6.0); `fitted_ops=()` keeps every expression free of
constants, as the default did before, and fits about twice as fast on small
data. They are a list of their own: restricting `unary_ops` leaves them on,
so `unary_ops=()` still allows `log(x + a)` unless `fitted_ops=()` too.

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
model = BeamFeatRegressor(max_depth=2, random_state=0).fit(X, y)
print(model.formulas(), model.to_sympy())
```

```text
['exp(-0.5*(x0)^2)'] [exp(-x0**2/2)]
```

With `fitted_ops=()` the same data give `(x0 + (x0)^2)` and `log(x0)`, a
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
  10/max|x|; for `pow`, `a` from -4 to 4; for the shifts, poles kept off the
  observed range), then a golden-section refinement. A fit that runs to the
  edge of its grid is discarded, and so is one for which the likelihood-ratio
  test of the snapping step below does not reject the null that the operand
  on its own fits as well (each operator becomes linear in `x` at one end of
  its range). Correlation is used whatever the `scorer`; the scorer then
  ranks the result.
- **Poles.** `inv_shift` and `log_shift` keep their pole away from the
  data they were fitted on: at least twice the spread of the operand's ten
  most extreme values at that end, and at least 1% of its range. A pole
  just past the edge of the data is shaped by the few rows next to it, and
  the first new row a little further out falls beyond it: NaN for the
  logarithm, a value of the wrong sign and any size for the reciprocal.
  A fit that wants its pole closer runs into the bound and is discarded,
  and snapping never moves a constant past it. The margin widens where the
  data thin out towards the edge and narrows where they are dense. On 18
  real datasets, poles inside it were crossed by a row the fit had not
  seen in 14% of fits, and poles outside it in 0.3%. A negative power has
  its pole at 0, so `pow` fits one only when the operand stays more than
  the same margin above 0. Otherwise `a` runs from 0.1 to 4, and a fit that
  wants less is discarded, since `x^a` then takes the shape of `log(x)`,
  which has the same pole.
- **In context.** `1/(x0·x1 + a)` on its own correlates best with the target
  as `a` runs to the pole; only as the denominator of `(x0 + x1)` is `a = 1`
  identifiable. While the depth limit leaves room, each constant is also
  fitted as a factor of a beam member, and that product joins the next
  depth's candidates.
- **Snapping.** The constant is replaced by the simplest of 0, ±1, ±2, ±3,
  ±1/2, ±3/2, ±1/3, ±1/4 for which a likelihood-ratio test at level 0.001
  fails to reject the null that the constant takes that value, so
  `exp(-0.49991·x²)` is reported as `exp(-x²/2)`. If the test rejects every
  one of them, the constant is rounded to the fewest significant digits that
  cost nothing measurable. A snapped constant is exact in `to_sympy()`,
  `latex()` and the printed formulas.
- **Near copies.** A fitted constant can also bend an expression into a near
  copy of a simple law. With 10% noise on `log(x0)`, `(x0)^(-0.02)`
  correlates with `log(x0)` at 0.99999, and since its constant was fitted on
  the search rows it fits them a little better. So at the end of the search,
  each kept expression with fitted constants is compared with its near copies
  among everything the search found (correlation at `redundancy_threshold`)
  that carry fewer constants and are not near copies of another kept
  expression. The snapping test decides, on the same rows: if it fails to
  reject the null that a copy fits the target as well, the simplest such copy
  takes the expression's place. Without this step, `log(x0)`, `sqrt(x0)`,
  `1/x0` and `x0²` at that noise were printed as themselves in 0, 0, 2 and 2
  of 10 fits; with it, in 10, 7, 10 and 8, against 10, 9, 10 and 8 with
  `fitted_ops=()`. A law that needs its constant keeps it while the data
  reject the simpler forms, as the Gaussian and relativistic velocity
  addition of the physics panel do at 0.1% noise.
- **Stored.** The constant is part of the expression: `transform`,
  `predict` and a pickled model reuse it on new data without refitting.

The constants are fitted inside the search, on the search rows only, so the
screen on the selection rows sees fixed expressions and its guarantee is
unchanged; see the [guarantees page](guarantees.md#selective-inference-why-the-estimators-split-their-data).

What it costs. Fitting adds a few candidates per parent and operator, and
makes a fit on the Feynman panel about five times slower (about 0.2 s to
1.1 s per equation). Fitted expressions also take beam slots, so enabling
them can change what else a depth keeps: within a depth, a constant-free
candidate is never dropped as a near-duplicate of a fitted one, but it can be
out-ranked, and at the end of the search it takes a fitted expression's place
only as described under *Near copies*. And
constants are fitted one at a time, greedily: a law needs its constant-free
parts to be among a depth's best candidates first.

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

## Audit what selection did

Every candidate's permutation p-value and its adjusted q-value is recorded:

```python
for row in model.selection_report_[:3]:
    print(row["formula"], round(row["q_value"], 4), row["kept"])
```

A feature is *kept* only if the screen rejects its null at `target_fdr`
(default 0.1, Benjamini–Yekutieli) on the held-out split and the parsimony
step then retains it. `screened` marks the certified set, the candidates
whose nulls the screen rejected, which is what the guarantee covers; `kept`
marks the compact subset the equation prints. Set `parsimony=None` and the
two agree.

## When no null is rejected

On a target with no structure the screen rejects no null, and the model says
so instead of returning junk:

```python
noise = rng.standard_normal(400)
model = BeamFeatRegressor(random_state=0).fit(X, noise)
# NoDiscoveriesWarning: ... Returning no constructed features ...
print(model.equation())
```

```text
y = 0.03527  (no feature selected)
```

Not on every draw. Under the global null every rejection is a false
rejection, so the FDR is the probability that a fit rejects any null at all:
up to `target_fdr` of pure-noise fits still certify features, and when one
does, the nulls of correlated candidates tend to be rejected together. The
same noise under another search seed can certify a dozen formulas with
`fdr_controlled_ = True`: that is the error the guarantee bounds in
expectation, not a breach of it.

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
regressor's `score` raises rather than scoring the remaining rows.

Fitted constants do not change this. `(x0)^1.5` is NaN for a negative
`x0`, as `sqrt(x0)` is. A fitted pole keeps a
[margin](#constants-inside-expressions) from the training rows, but a row
far enough outside them can still reach it: `(x1)^(-0.5)`, fitted where
`x1` lies between 1 and 3, is infinite at `x1 = 0`, as `1/(x1)` would be.

## Search constraints

`max_inputs` caps the distinct input columns in one feature: `max_inputs=2`
admits `x0 * log(x0 / x1)` but not `x0 * x1 / x2`. Use it when a feature that
mixes many columns would be uninterpretable to you anyway. It is checked as
each candidate is constructed, before any numerical work, so a candidate it
rules out costs no evaluation and takes no beam slot. `search_result_.trace`
counts them per depth (`n_rejected_constraints`).

It does not make selection more powerful. The screen's multiple-testing
correction runs over the features that reach selection, which the beam
already caps at `beam_width` and `max_features` however many expressions were
proposed; the limit changes which features those are and how long the search
takes.

It is off by default (`None`), and it only ever narrows the search. An
expression of depth d combines at most 2^d input columns, so at the default
`max_depth=2` no feature uses more than four, and `max_inputs=4` or more
changes nothing; at `max_depth=3` the limit matters up to 8. Set it at or
above the number of inputs any law you look for needs. A law that needs more
cannot be built, and the search returns the best expression within the limit
instead:

```python
import numpy as np
from beamfeat import BeamSearch

rng = np.random.default_rng(0)
X = rng.uniform(0.5, 3, (400, 8))
y = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + rng.normal(0, 0.1, 400)
for limit in (None, 3):
    search = BeamSearch(max_depth=2, beam_width=30, random_state=0, max_inputs=limit).run(X, y)
    print(limit, search.names[0])
```

```text
None ((x0 * x1) + (x2 * x3))
3 ((x0 + x2) * sqrt(x1))
```

On wide data whose signal is spread over many columns, a low limit can
still help, because it keeps the beam from filling with combinations of many
columns. Compare the held-out score with and without it rather than assuming
either way.

## Laws whose parts the score cannot see

The search keeps, at each depth, the candidates that correlate best with the
target and builds the next depth from them. A law can need a part that does
not correlate with the target at all. With inputs centred on zero, `x0` and
`x1` are each uncorrelated with `x0*x1`, and every pairwise product is
uncorrelated with `x0*x1*x2`: the score ranks those parts among the noise,
and the beam drops them before they can be combined. On positive inputs, as
in most physical tables, each factor of a product correlates with it, and
the problem does not arise.

[`benchmarks/search_stress/`](https://github.com/LD-Shell/beamfeat/tree/main/benchmarks/search_stress)
measures what it costs. At the defaults, with 500 rows and 10% noise, the
search recovered `x0*x1*x2` on centred uniform inputs in 54% of fits with 20
columns, 18% with 100 and 10% with 300, and in every fit on positive inputs;
each miss was the first depth's beam dropping the pairwise products. A beam
of 200 lifted the 20-column figure to 96% at about nine times the fit time,
but the 100-column one only to 26%. Past the proposal ceiling, from about 258
columns at the defaults, a centred pair can be lost before it is scored:
with 1,000 columns `x0*x1` was recovered in 32% of fits, because neither
input scored well enough for the pair to be proposed. A miss mostly ends in
no equation and a `NoDiscoveriesWarning`, sometimes in terms that ride on
one of the law's columns.

`magnitude_slots` adds beam slots for candidates whose magnitude tracks the
target's:

```python
rng = np.random.default_rng(3)
X = rng.uniform(-1, 1, (400, 10))
y = X[:, 0] * X[:, 1] * X[:, 2] + rng.normal(0, 0.01, 400)
for slots in (0, 3):
    model = BeamFeatRegressor(max_depth=2, beam_width=10, magnitude_slots=slots).fit(X, y)
    print(slots, model.formulas())
```

```text
0 []
3 ['(x0 * (x1 * x2))']
```

The first fit finds nothing and warns `NoDiscoveriesWarning`.

How it works:

- **The score.** A candidate `c` is scored by the absolute correlation of
  `log(1 + |c|/rms(c))` with the same transform of the centred target, where
  `rms` is the root mean square. The logarithm turns a product into a sum,
  so a factor's magnitude tracks the product's even where the factor itself
  does not correlate with it. The 1 keeps the rows where a product nears
  zero, and the noise there, from dominating, and it compresses a heavy
  tail, so that a ratio with a denominator near zero does not score by
  accident. On more than 1,000 search rows the score is computed on 1,000
  of them, evenly spaced, which is plenty to rank a few slots and keeps its
  cost from growing with the table.
- **The slots.** After the `beam_width` candidates the score keeps, up to
  `magnitude_slots` more are kept from the rest, best magnitude score first,
  under the same redundancy check. They are kept at every depth but the
  last, since they are there to be built on.
- **The proposals.** When the first depth cannot propose every pair of
  inputs, the inputs are proposed in an order that interleaves their
  magnitude ranking with their score ranking. The magnitude ranking takes
  the first place and `magnitude_slots / (beam_width + magnitude_slots)` of
  those after it.
- **A part behind a visible term.** In `x0*x1 + x2` the product does
  correlate with the target, but hundreds of sums of `x2` with another column
  can outrank it. The search then relies on scoring candidates against the
  residual left by the inputs, and with more inputs than search rows that
  residual is empty. The magnitude score is also taken against the residual
  on the best-scoring columns, at most a quarter as many as the rows, which
  keeps `x0*x1` in view.

Measured at 10 slots on the stress test, it recovered `x0*x1*x2` on
centred inputs in every fit with 20, 100 and 300 columns, the centred pair
in every fit with 1,000 columns, and `x0*x1 + x2` with 300 columns and 250
search rows in 88% of fits, where the default recovered none; in no setting
did it lose a law the default found. Fits took 1.2 to 1.7 times as long
there, and 1.4 times on the high-dimensional study's three 10,000-row
tables. Where no such law is hidden it changes little either way: the main
suite's mean held-out score over 24 regression datasets went from 0.885 to
0.884, with diabetes and california housing about 0.01 lower, and the
high-dimensional study's mean from 0.647 to 0.649. Try it when a law may
hide behind centred or signed inputs, and keep it if the held-out score
agrees.

It is for regression only, since a class label has no magnitude to track.
The ranking runs inside the search, on the search rows only, so the screen on
the selection rows sees fixed expressions and its guarantee is unchanged;
see the [guarantees page](guarantees.md#selective-inference-why-the-estimators-split-their-data).
It was measured on products and on a product beside a visible term, and a
law whose parts the magnitude score cannot see either is still out of reach.

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

## Speed and memory on wide data

`n_jobs` spreads the search over cores, and `-1` uses all of them. The
workers are threads, since numpy releases the GIL while it scores, or
processes for the gradient-boosting scorer, whose model fits do not. Apart
from that scorer, a depth with fewer than 4096 candidates is scored in the
calling thread, so the setting pays off on wide tables rather than on a
handful of columns. It never changes the fit: features, p-values and
predictions are identical for every value.

```python
model = BeamFeatRegressor(random_state=0, n_jobs=-1).fit(X, y)
```

Memory does not grow with the number of candidates either. Each depth is
evaluated and scored in chunks of about 8 MiB and only the scores are kept,
so a default fit on 1000 columns and 600 rows peaks at about 0.35 GB.
`BeamSearch(chunk_size=...)` sets the chunk size directly, again without
changing the result. The exceptions are mutual information, which draws its
noise for a whole depth, and a custom scorer that does not declare itself
`separable`: both still receive a depth at once.

## When to reach for something else

If a tree model beats beamfeat by a wide margin, your signal is likely
piecewise, not algebraic — use the tree. `fitted_ops` covers one constant
per operator, fitted one operator at a time; if your law needs several
constants fitted jointly (`exp(-a·x)·cos(b·x + c)`), constants inside sums
or products (`a·x0 + b·x1` under a logarithm), or deeper nesting than the
beam reaches, use a symbolic regressor such as PySR. The [guarantees page](guarantees.md) states every boundary with the
measurement behind it.
