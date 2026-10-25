---
description: "What a fitted beamfeat model prints, what the FDR guarantee covers, the equation at every size, and how to check whether an equation is stable."
---

# Reading a fit

What a fitted model prints, what each part means, and what to check before
relying on it. The examples continue from the
[quickstart](../quickstart.md)'s fit:

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 6, (400, 4))
y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
model = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0).fit(X, y)
```

## Why the equation says "1 of 25"

The screen rejected the null for twenty-five formulas, the certified set.
Each formula's null is that it is independent of the target. On a strong
signal every near-duplicate of the true feature really is associated with
the target, so rejecting its null is correct. A forward step then picks the
compact subset that predicts best, since an equation of dozens of
near-duplicates would be unreadable.

The subset is chosen on the rows the screen already used, so the guarantee
covers the twenty-five, not the one term printed. The suffix records that,
and `fdp_inflation_` prices it:

```python
print(model.fdp_inflation_)
```

```text
25.0
```

That is |S|/|S'|, the factor by which pruning can inflate the realised false
discovery proportion. With `parsimony=None` the model fits the whole
screened set, the printed equation *is* the certified set, and the suffix
disappears:

```python
full = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0,
                         parsimony=None).fit(X, y)
print(full.n_features_out_)
print(full.equation(max_terms=3))
```

```text
25
y = 0.077*(x1 * (x0 * x1)) + 0.4333*(x0 * x1) - 0.0842*((x0 * x1) / (1/(x1 - 8.17))) - 1.127
```

`max_terms` shortens the display, not the model: the terms it leaves out are
held at their training mean, and their average contribution moves into the
constant, which is why it differs from the full equation's.

The short equation costs little: about a thousandth of R² over 308 paired
fits, for an equation about twenty-five times shorter, and pruning never
inflated the realised false discovery proportion
(`benchmarks/PARSIMONY_COST.md`). For an equation both short and certified,
spend rows instead:
[Re-testing the printed terms](printed-equation.md#re-testing-the-printed-terms-parsimony_holdout).

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
  2         12 0.9999  1.45e-06 y ~ (x0 * x1) + (1/(x0 - 8.1) * 1/(x1 - 8.17))
  3         18 0.9999 8.674e-07 y ~ (x0 * x1) + (1/(x0 - 8.1) * 1/(x1 - 8.17)) + ((x0 + x3) * x1)
  ...
* the fitted model (the forward step stops before the first gain below parsimony_tol=0.001).
The FDR guarantee covers the screened set of 25 terms (line 25), not an equation chosen from this list.
```

On a noisy target the marked line can sit further down the list, each term
after the first adding a little by fitting noise.
[`parsimony_alpha`](printed-equation.md#a-short-equation-parsimony_alpha)
stops the step earlier; the list is the same, and only the marked line moves.

Complexity counts the nodes in each term's expression tree — operators and
input columns, `(x0 * x1)` being 3 — plus one per term for its coefficient
and one per [fitted constant](search.md#constants-inside-expressions).
The lines are written `y ~ a + b`, without coefficients, because only the
marked one has been fitted; `equation()` prints its coefficients. For a
classifier the R² is that of the integer-coded labels, the heuristic the
forward step ranks by, not a classification metric.

Keep the output's last line in mind: every line is chosen on the rows the
screen used, so picking line 3 because it looks better is the same kind of
choice the forward step makes, and the guarantee covers it no more than the
marked line. The records behind the text are in `equations_`, one dict per
line (`n_terms`, `complexity`, `r2`, `formulas`, `features`, and `model`,
which marks the fitted line).

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

Held-out R^2: mean 0.491, sd 0.052, min 0.395, max 0.575.
30 distinct equations in 30 splits; none recurs.
98 distinct features; the most frequent in 12 of the 30 splits.

Features, most frequent first:
  12/30  ((bmi + s5) + (bp - s3))
  10/30  ((bmi + s5) - (s3 + sex))
   7/30  ((bmi + s5) + (age * sex))
  ... and 95 more features

Features whose values on the rows of X correlate above 0.999 in absolute value are counted as one; an equation is the set of its features, whatever its coefficients.
```

No two of the thirty splits printed the same equation, and held-out R²
ranges from 0.395 to 0.575, a spread one fit's score would hide. What
recurs is a family of sums and differences of `bmi`, `s5`, `bp` and `s3`.
Report that family, with its frequencies, rather than one equation.

??? note "What the report holds, and how it scores"

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
    formula's [domain](data.md#rows-outside-a-formulas-domain), the classifier predicts
    its most frequent training class there, as its `predict` does. A regressor or
    transformer has no score on such a test part, so that split goes unscored,
    and the summary counts it.

`resample` chooses what varies. The default, `"splits"`, draws new
training and test rows and a new seed for every split: how far the equation
and its score move with another sample of the data. `resample="seeds"`
keeps the first split's rows and varies only the seed, as in the paper's
stability study (Appendix E): how much the equation owes to one random draw.
On diabetes it still prints 30 distinct equations in 30 fits, but the
standard deviation of R² falls from 0.052 to 0.009, so most of the score's
spread comes from the test rows, while the turnover of equations remains.

## When no null is rejected

On a target with no structure the screen rejects no null, and the model says
so instead of returning junk:

```python
noise = np.random.default_rng(1).standard_normal(400)
empty = BeamFeatRegressor(random_state=0).fit(X, noise)
# NoDiscoveriesWarning: ... Returning no constructed features ...
print(empty.equation())
```

```text
y = -0.08207  (no feature selected)
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
