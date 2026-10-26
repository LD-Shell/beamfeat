---
description: "beamfeat's search options: units and dimensional analysis, constants inside expressions, magnitude and residual slots, switches at a threshold, search constraints, and speed and memory."
---

# Search options

How the search builds its candidates, and the options that change what it
can find and what it costs. The examples that use `X` and `y` start from:

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 6, (400, 4))
y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
model = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0).fit(X, y)
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

Give every column a unit. A column without one combines freely with the
others, so the check stops binding on exactly the columns you left out. Give
unitless columns `"dimensionless"`; the estimator warns when coverage is
partial, and raises when no key matches a column. Keys are column names, so
with a DataFrame you can write `{"rho": "kg/m**3"}`.

## Constants inside expressions

The linear model supplies the coefficients outside the expressions. That alone
cannot write `exp(-x²/2)`, `x^1.5` or `(x0 + x1)/(1 + x0·x1)`, where the
constant sits inside a nonlinearity, so the estimators also use operators that
carry one fitted constant each, listed in `fitted_ops`. All but `sign_shift`
are on by default; `fitted_ops=()` keeps every expression free of constants
and fits about twice as fast on small data. They are a list of their own:
restricting `unary_ops` leaves them on, so `unary_ops=()` still allows
`log(x + a)` unless `fitted_ops=()` too.

| name | expression | operand |
|---|---|---|
| `"exp_scaled"` | `exp(a·x)` | dimensionless |
| `"pow"` | `x^a` | positive, dimensionless |
| `"inv_shift"` | `1/(x + a)` | any; `a` takes the operand's unit |
| `"log_shift"` | `log(x + a)` | `x + a` positive, dimensionless |
| `"sign_shift"`, off by default | `sign(x + a)` | any; `a` takes the operand's unit ([Switches at a threshold](#switches-at-a-threshold)) |

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

??? note "How the constants are chosen"

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
      observed range; for `sign_shift`, the operand's 5% to 95% quantiles),
      then a golden-section refinement. A fit that runs to the edge of its grid
      is discarded, and so is one for which the likelihood-ratio test of the
      snapping step below does not reject the null that the operand on its own
      fits as well (each operator but `sign_shift` becomes linear in `x` at one
      end of its range, and `sign_shift` is held to the same test).
      Correlation is used whatever the `scorer`; the scorer then ranks the
      result.
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
      which has the same pole. The denominator of every quotient and
      reciprocal in the search keeps the same margin from zero, fitted constant
      or not; see
      [New rows beyond the training data](data.md#new-rows-beyond-the-training-data).
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
unchanged; see the [guarantees page](../guarantees.md#selective-inference-why-the-estimators-split-their-data).

**What it costs.** A fit on the Feynman panel takes about five times as
long (0.2 s to 1.1 s per equation). Fitted expressions also take beam
places, so they can change what else a depth keeps: a constant-free
candidate can be out-ranked by a fitted one, though never dropped as its
near-duplicate. And constants are fitted one at a time, greedily, so a law
needs its constant-free parts among a depth's best candidates first.

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
measures the cost. Without magnitude slots, on 500 rows with 10% noise,
`x0*x1*x2` on centred inputs was recovered in 54% of fits with 20 columns,
18% with 100 and 10% with 300 (in every fit on positive inputs), and `x0*x1`
in 32% with 1,000 columns. A wider beam barely helps: 200 places lifted the
20-column figure to 96% at about nine times the fit time, and the 100-column
one only to 26%. A miss mostly ends in no equation and a
`NoDiscoveriesWarning`.

`magnitude_slots` adds beam slots for candidates whose magnitude tracks the
target's. Its default, `"auto"`, adds 10 for regression; `0` turns them off,
as the first fit here does:

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

??? note "How the magnitude slots work"

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

With 10 slots the search recovered `x0*x1*x2` on centred inputs in every fit
with 20, 100 and 300 columns, the centred pair in every fit with 1,000
columns, and `x0*x1 + x2` with 300 columns on 250 search rows in 88% of
fits, against none without them; it lost no law the search without them
found. Fits took 1.2 to 1.7 times as long. Where no such law is hidden it
changes little: the main suite's mean held-out score over 24 regression
datasets went from 0.885 to 0.884, and the high-dimensional study's from
0.647 to 0.649.

It is for regression only, since a class label has no magnitude to track:
`"auto"` adds no slot for classification, and a positive number raises. The
ranking runs inside the search, on the search rows only, so the screen's
guarantee is unchanged (see the
[guarantees page](../guarantees.md#selective-inference-why-the-estimators-split-their-data)).
A law whose parts the magnitude score cannot see either is still out of
reach.

## A weak term beside a dominant one

The search ends by ranking every candidate it kept against the target and
keeping the best `max_features` (`beam_width` by default). A dominant term
fills that ranking with restatements of itself, and a weaker term beside it
is cut although the search built it. The residual slots (`residual_slots`,
off by default) add to the ranking the first terms of a greedy forward path
through the candidates: each the one that adds most to a least-squares fit
of those before it. A restatement adds nothing to the term it restates, so
the weaker term comes second.

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(1)
X = rng.uniform(1, 5, (2000, 4))
signal = X[:, 0] * X[:, 1] / X[:, 2] + 0.5 * np.log(X[:, 3])
y = signal + rng.normal(0, 0.1 * np.std(signal), 2000)
for slots in (0, 10):
    print(BeamFeatRegressor(random_state=0, residual_slots=slots).fit(X, y).equation())
```

```text
y = 0.9929*((x0 / x2) * x1) + 0.1761*((x0 + x3) + (x1 - x2)) - 0.1693*(x0 + (x1 - x2)) - 0.02271   [3 of 50 certified terms; parsimony=None prints all 50]
y = 0.9968*((x0 / x2) * x1) + 0.475*log(x3) + 0.03165   [2 of 55 certified terms; parsimony=None prints all 55]
```

The logarithm carries 0.6% of the target's variance. Without the slots, two
sums stand in for it.

??? note "How the residual slots work"

    - **The path.** It runs over every candidate the search kept, on the
      search rows (on more than 20,000 of them, that many evenly spaced, and
      fewer on a wide table, so that the values stay within 64 MiB, though
      never fewer than 1,000). Gains within a relative 2e-6 of the best are
      tied, and the simpler expression wins, as in the beam.
    - **Near copies.** On [1, 5], `log(x3)`, `sqrt(x3)` and `1/x3` correlate
      above the redundancy threshold, so a depth keeps one of them and drops
      the others. Before the final ranking, each path term that is an input
      column, or has a single operand, adds that column or operand and its
      transforms by `unary_ops` to the candidates, so the path chooses among
      them; one of them can also take a place by score.
    - **The count.** At most `residual_slots` terms, and never more than
      `max_features` or a quarter of the search rows. A path term replaces a
      ranked candidate that is a near copy of it, and is dropped if it is a
      near copy of an earlier path term; every other candidate the score
      ranked stays, so the screen tests at most `residual_slots` more
      candidates than before.

Over 60 fits of that law, with 4 or 10 columns, 500 to 5,000 rows and two
noise levels, the slots printed the exact law in 33, the default in none
(`benchmarks/weak_terms_and_switches/`). They are off by default because the
path terms explain whatever the leading terms leave, noise included, and the
forward step prints some of them as riders. In `benchmarks/release_comparison/`
they printed a term built only from unrelated columns in 19 of 800
calibration-design fits, where the default printed none, raised the share of
riders at a signal-to-noise ratio of 1 from 0.722 to 0.729, and left the
held-out score of 8 of the 11 real regression datasets within 0.005, raising
one and lowering two. With `parsimony_alpha=0.05` beside them the riders go
(one design fit of 800, riders 0.064 against 0.061 for the check alone, the
generating term in as many fits), and the weak term prints as often. On pure
noise the two together printed a term in 4 of 200 fits, against 2 at the
default or with the check alone, within the guarantee's q = 0.1. For a law
with a weak term, use both:

```python
model = BeamFeatRegressor(residual_slots=10, parsimony_alpha=0.05).fit(X, y)
```

The slots are for regression only; a positive number raises for
classification. The path runs inside the search, on the search rows only, so
the screen's guarantee is unchanged.

## Switches at a threshold

`sign` (a unary operator; list it in `unary_ops`) and `sign_shift` (a fitted
one, `sign(x + a)`; list it in `fitted_ops`) are off by default. With `sign`,
a product with a switch at zero, such as `x2*sign(x0)`, is a candidate like
any other, and with `sign_shift` one with a switch at a fitted threshold,
such as `x2*sign(x0 - 0.7)`. `sign_shift` tries
thresholds at the operand's 5% to 95% quantiles, then refines and snaps the
threshold as the other fitted operators do their constants
([Constants inside expressions](#constants-inside-expressions)).

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
X = np.column_stack([rng.uniform(-2, 2, (1000, 2)), rng.uniform(1, 5, (1000, 3))])
y = np.where(X[:, 0] > 0.7, 2.0, -1.0) * X[:, 2] + X[:, 1] + rng.normal(0, 0.3, 1000)
fitted = ("exp_scaled", "pow", "inv_shift", "log_shift", "sign_shift")
for slots in (0, 10):
    print(BeamFeatRegressor(random_state=0, fitted_ops=fitted, residual_slots=slots).fit(X, y).equation())
```

```text
y = 1.539*(x2 * sign(x0 - 0.7)) + 1.01*(x1 + sign(x0 - 0.7)) - 1.161*(sign(x0 - 0.7) - log(x2)) + 0.3139   [3 of 50 certified terms; parsimony=None prints all 50]
y = 1.501*(x2 * sign(x0 - 0.7)) + 0.4993*(x1 + x2) + 0.4915*x1 + 0.002704   [3 of 52 certified terms; parsimony=None prints all 52]
```

The slope in `x2` is 2 above the threshold and -1 below it: 0.5 plus 1.5
times the sign. Both fits find the switch term; the rest of the law, `x1`
and `0.5*x2`, is weaker, and a surrogate stands in for it until the
[residual slots](#a-weak-term-beside-a-dominant-one) add it. Over 20 fits like
these, at thresholds 0 and 0.7 and two noise levels, `sign_shift` printed the
switch in every fit, with R² against the noiseless target of 0.997 to 0.999
(1.000 with the slots), against 0.76 to 0.88 for the default and 0.983 to
0.996 for gradient-boosted trees (`benchmarks/weak_terms_and_switches/`).
`sign` finds a switch at zero only; `sign_shift` finds that one too. On the main
suite's `threshold_step`, `3*b*1[a > 1.5]`, `sign_shift` raised the held-out
score from 0.841 to 0.999; on the other datasets of
`benchmarks/release_comparison/` it moved held-out score little, raising it
on penguins by 0.025 and lowering the classifier's accuracy on breast_cancer
by 0.022.

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

## Speed and memory

On a tall table, `max_search_rows` runs the search on that many of its rows,
drawn at random, while the screen keeps every selection row; see
[Tall data](../settings.md#tall-data-max_search_rows). The rest of this
section is about wide tables.

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
