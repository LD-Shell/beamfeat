"""Build the tutorial notebooks.

Notebooks are generated from source rather than hand-edited so that their code
stays in sync with the library and can be executed in CI. Run this script,
execute the notebooks to populate outputs, then run it again with --sync-docs
to copy the executed notebooks into docs/notebooks/, which is the copy the
documentation site publishes.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import nbformat as nbf


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def build(
    path: pathlib.Path, cells: list[nbf.NotebookNode], title: str, requirement: str = "beamfeat"
) -> None:
    notebook = nbf.v4.new_notebook()
    url = f"https://colab.research.google.com/github/LD-Shell/beamfeat/blob/main/notebooks/{path.name}"
    badge = markdown(
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})\n\n"
        f"Running in Colab? Install first: `%pip install {requirement}`."
    )
    notebook.cells = [badge] + cells
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "title": title,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(path))
    print(f"wrote {path}")


HERE = pathlib.Path(__file__).parent


def sync_docs() -> None:
    """Copy the executed notebooks over the documentation site's copies.

    Run after executing, not after building: the copies are wanted with their
    outputs. Keeping this in the build script is what stops the two trees
    drifting apart, which they have.
    """
    target = HERE.parent / "docs" / "notebooks"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(HERE.glob("[0-9][0-9]_*.ipynb")):
        shutil.copyfile(path, target / path.name)
        print(f"copied {path.name} to docs/notebooks")


# --------------------------------------------------------------------------- #
# 01: Getting started
# --------------------------------------------------------------------------- #

getting_started = [
    markdown(
        """
# Getting started with beamfeat

beamfeat constructs interpretable features from tabular data. Given columns
`a`, `b`, and `c`, it searches expressions like `log(a)`, `a * b`, and
`(a * b) / c`, keeps the ones that predict the target, and reports them as
readable formulas.

Two things distinguish it from exhaustive feature engineering:

1. **The search is guided.** Candidates are scored against the target and only
   the best are extended, so cost does not compound with expression depth.
2. **Selection is calibrated.** The retained features carry a false discovery
   rate guarantee rather than surviving a heuristic threshold.

This notebook covers the basic workflow. Later notebooks cover the search and
selection machinery in detail.
        """
    ),
    code(
        """
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
        """
    ),
    markdown(
        """
## A problem with a known answer

We generate data where the target is a known function of the inputs, so we can
check whether beamfeat recovers it. The relationship is `y = (a * b) / c`,
which no linear model on the raw columns can represent.
        """
    ),
    code(
        """
n = 500
a = rng.uniform(1.0, 6.0, n)
b = rng.uniform(1.0, 6.0, n)
c = rng.uniform(1.0, 6.0, n)
d = rng.uniform(1.0, 6.0, n)  # an irrelevant column

# A DataFrame rather than a bare array: beamfeat takes column names from it,
# so the formulas come back as `(a * b) / c` instead of `(x0 * x1) / x2`.
# A NumPy array works identically and reports columns as x0...xp.
X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
y = (a * b) / c + rng.normal(0, 0.05, n)

print(f"{X.shape[0]} rows, {X.shape[1]} columns")
print(f"target range: {y.min():.2f} to {y.max():.2f}")
        """
    ),
    markdown(
        """
## Fitting

`BeamFeatRegressor` follows the usual scikit-learn interface. The two
parameters that matter most are `max_depth`, which bounds expression
complexity, and `beam_width`, which bounds how many expressions survive each
depth.
        """
    ),
    code(
        """
model = BeamFeatRegressor(max_depth=2, beam_width=30, random_state=0)
model.fit(X, y)

print(f"R^2: {model.score(X, y):.4f}")
print(f"features constructed: {model.n_features_out_}")
        """
    ),
    markdown(
        """
## Watching it work

`verbose` is a level. `0` is silent, `1` prints a line per stage, and `2` adds
per-depth search detail and the strongest certified candidates with their
p- and q-values. It is the quickest way to see where a fit spent its effort,
and to tell a search that found nothing from a screen that certified nothing.
        """
    ),
    code(
        """
BeamFeatRegressor(max_depth=2, beam_width=30, random_state=0, verbose=1).fit(X, y)
        """
    ),
    markdown(
        """
## Reading what it found

This is the part a gradient-boosted model cannot give you. The fitted model
exposes both the individual feature formulas and the full equation.
        """
    ),
    code(
        """
for formula in model.formulas()[:5]:
    print(" ", formula)
        """
    ),
    code(
        """
print(model.equation())
        """
    ),
    markdown(
        """
The true generating expression should appear among the selected features. Note
that `d` — the irrelevant column — should be largely absent.

The bracketed suffix is the equation telling you that it is a subset. Screening
certified a larger set; the parsimony step kept the compact predictive part of
it and fitted that, so the guarantee covers the set the terms came from rather
than the terms themselves. Notebook 04 takes that apart.

## Comparing against a linear baseline

The point of constructing features is to let a simple model fit a relationship
it otherwise could not.
        """
    ),
    code(
        """
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

baseline = Ridge().fit(X_train, y_train)
beamfeat = BeamFeatRegressor(max_depth=2, beam_width=30, random_state=0).fit(X_train, y_train)

print(f"Ridge on raw columns:  R^2 = {baseline.score(X_test, y_test):.4f}")
print(f"beamfeat features:     R^2 = {beamfeat.score(X_test, y_test):.4f}")
        """
    ),
    markdown(
        """
## Using it in a pipeline

`BeamFeatTransformer` constructs features without fitting a model, so it
composes with any downstream estimator. Because construction happens inside
`fit`, cross-validating the pipeline does not leak information across folds.
        """
    ),
    code(
        """
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from beamfeat import BeamFeatTransformer

pipeline = Pipeline(
    [
        ("features", BeamFeatTransformer(max_depth=2, beam_width=20, random_state=0)),
        ("model", Ridge()),
    ]
)

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"cross-validated R^2: {scores.mean():.4f} (+/- {scores.std():.4f})")
        """
    ),
    markdown(
        """
## Classification

`BeamFeatClassifier` works the same way, with `predict_proba` and
`decision_function` available as usual.
        """
    ),
    code(
        """
from beamfeat import BeamFeatClassifier

labels = ((a * b) > np.median(a * b)).astype(int)

classifier = BeamFeatClassifier(
    max_depth=2, beam_width=20, selector="permutation", target_fdr=0.1, random_state=0
)
classifier.fit(X, labels)

print(f"accuracy: {classifier.score(X, labels):.4f}")
print(f"features retained after FDR control: {classifier.n_features_out_}")
for formula in classifier.formulas():
    print(" ", formula)
        """
    ),
    markdown(
        """
## What to read next

- **02: Search and scoring** — how the beam search works, and how the scoring
  strategies differ in what they detect and what they cost.
- **03: Selection and units** — how false discovery rate control works, why the
  default is not knockoffs, and how dimensional analysis constrains the search.
- **04: Reading a fit** — what to do when the equation contains columns you did
  not expect, and how to tell a search failure from a selection failure.
        """
    ),
]


# --------------------------------------------------------------------------- #
# 02: Search and scoring
# --------------------------------------------------------------------------- #

search_and_scoring = [
    markdown(
        """
# Search and scoring

This notebook covers how beamfeat explores the space of expressions, and how
the choice of scorer changes what it finds.

The core problem is that expression space is enormous. With 5 input columns, 5
unary operators, and 4 binary operators, exhaustive expansion to depth 3
produces well over a million candidates, and each must be evaluated before it
can be judged. Beam search avoids this by scoring candidates as they are
generated and extending only the most promising.
        """
    ),
    code(
        """
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from beamfeat import BeamSearch

rng = np.random.default_rng(0)

n = 500
data = {name: rng.uniform(1.0, 6.0, n) for name in "abcd"}
target = (data["a"] * data["b"]) / data["c"] + rng.normal(0, 0.05, n)
        """
    ),
    markdown(
        """
## Running a search directly

`BeamSearch` is the layer beneath the estimators. Using it directly is useful
when you want to inspect the search rather than fit a model.
        """
    ),
    code(
        """
search = BeamSearch(max_depth=2, beam_width=30, random_state=0)
result = search.run(data, target)

print(result.summary())
        """
    ),
    markdown(
        """
## The trace

Every search records what happened at each depth. This is how you attribute
cost, and how you notice a beam that collapsed or saturated.
        """
    ),
    code(
        """
print(f"{'depth':>6} {'proposed':>10} {'evaluated':>10} {'kept':>6} {'best':>8} {'seconds':>8}")
for record in result.trace:
    print(
        f"{record.depth:>6} {record.n_proposed:>10} {record.n_evaluated:>10} "
        f"{record.n_kept:>6} {record.best_score:>8.4f} {record.elapsed:>8.3f}"
    )
        """
    ),
    markdown(
        """
## How beam width controls cost

Beam width is the parameter that bounds work. Widening it explores more of the
space at proportionally greater cost; narrowing it risks pruning a parent whose
children would have been valuable.
        """
    ),
    code(
        """
print(f"{'beam':>6} {'proposed':>10} {'kept':>6} {'seconds':>9}  top feature")
for width in (5, 10, 25, 50, 100):
    outcome = BeamSearch(max_depth=2, beam_width=width, random_state=0).run(data, target)
    top = outcome.names[0] if outcome.names else "-"
    print(
        f"{width:>6} {outcome.n_proposed_total:>10} {len(outcome):>6} "
        f"{outcome.elapsed:>9.3f}  {top}"
    )
        """
    ),
    markdown(
        """
Note that cost grows with beam width but the *quality* of the top feature
plateaus quickly. The proposals stop growing at 50 only because this problem
runs out of parents: four columns yield 52 evaluated depth-1 candidates (the
trace above), fewer than 50 of them survive the redundancy check described
below, and a wider beam has nothing more to extend. In practice a moderate beam
is usually enough, and the remaining budget is better spent on depth.

## How depth controls expressiveness

Some relationships simply cannot be expressed below a certain depth. `(a * b) /
c` needs depth 2: one step to build `a * b`, another to divide by `c`.
        """
    ),
    code(
        """
for depth in (0, 1, 2, 3):
    outcome = BeamSearch(max_depth=depth, beam_width=25, random_state=0).run(data, target)
    found = any(all(token in name for token in "abc") for name in outcome.names)
    print(f"depth {depth}: {len(outcome):>3} features, "
          f"three-way interaction found: {found}")
        """
    ),
    markdown(
        """
## Workers and memory

At the defaults a depth may propose up to 200,000 candidates
(`max_candidates_per_depth`), so the search evaluates and scores a depth in
chunks and keeps only the scores; the candidates the redundancy check reaches,
usually a small share of the round, are evaluated again. Memory stays bounded
however many candidates a depth proposes. `chunk_size` sets how many candidates
make a chunk, by default as many as fit in 8 MiB. Mutual information is the
exception: it draws its noise for the whole batch, so it still scores a depth
at once.

`n_jobs` spreads the chunks over joblib workers: threads, since numpy releases
the GIL, or processes for gradient boosting, whose model fits hold it. `-1`
uses every core. A depth of fewer than 4096 candidates stays in the calling
thread unless the scorer fits models, since starting workers would cost more
than it saves, so on the four-column problem above `n_jobs` changes nothing.
The estimators take the same argument.

Neither setting changes the result, because no seed depends on a worker or a
chunk. On forty columns, where both depths are large enough for workers:
        """
    ),
    code(
        """
# A generator of its own, so that this cell leaves the draws of the cells
# below unchanged.
wide_rng = np.random.default_rng(1)
wide = {f"x{index}": wide_rng.uniform(1.0, 6.0, n) for index in range(40)}
wide_target = wide["x3"] * wide["x17"] / wide["x31"] + wide_rng.normal(0, 0.05, n)

serial = BeamSearch(max_depth=2, beam_width=30, random_state=0).run(wide, wide_target)
print(f"candidates at depth 1: {serial.trace[1].n_proposed}, at depth 2: {serial.trace[2].n_proposed}")
print(f"top feature: {serial.names[0]}")
print()

for name, value in (("n_jobs", 2), ("n_jobs", -1), ("chunk_size", 100)):
    outcome = BeamSearch(max_depth=2, beam_width=30, random_state=0, **{name: value}).run(
        wide, wide_target
    )
    same = outcome.names == serial.names and np.array_equal(outcome.scores, serial.scores)
    label = f"{name}={value}"
    print(f"{label:>14}: same features and scores as n_jobs=1: {same}")
        """
    ),
    markdown(
        """
Whether workers save time depends on the machine and on the size of the round,
so they are not timed here. The 0.5.0 changelog measures the faster search as a
whole: a default fit on 1000 columns and 600 rows takes 7 s rather than 29 s
and peaks at 0.35 GB of memory rather than 2.6 GB, with the same features,
p-values and predictions.

## Choosing a scorer

Four scoring strategies are available, and they differ in what they can detect
and what they cost.

- `"correlation"` — absolute Pearson correlation against the residual. One
  dot product per candidate. The default.
- `"spearman"` — absolute rank correlation against the same residual, for a
  sort per column more. Sees any monotone relationship and is insensitive to
  outlying values.
- `"mutual_information"` — nearest-neighbour mutual information. Detects
  dependence that correlation misses.
- `"gradient_boosting"` — measured out-of-fold predictive improvement. Scores
  what is actually being optimised.

### Where they agree

On a relationship whose nonlinearity is captured by the expression itself, the
first three find the same thing, and correlation is much the cheapest. Gradient
boosting is left out: it fits models for every candidate, which on a round of
this size takes far longer still.
        """
    ),
    code(
        """
import time

for name in ("correlation", "spearman", "mutual_information"):
    started = time.perf_counter()
    outcome = BeamSearch(scorer=name, max_depth=2, beam_width=20, random_state=0).run(data, target)
    elapsed = time.perf_counter() - started
    print(f"{name:>20}: {elapsed:>6.2f}s   top: {outcome.names[0]}")
        """
    ),
    markdown(
        """
### Where they differ

Correlation is blind to symmetric relationships. If the target depends on the
*magnitude* of a feature but not its sign, correlation sees nothing while
mutual information sees the dependence clearly.
        """
    ),
    code(
        """
from beamfeat.scoring import CorrelationScorer, MutualInformationScorer

symmetric_feature = rng.uniform(-3, 3, n)
symmetric_target = symmetric_feature**2 + rng.normal(0, 0.3, n)
noise = rng.normal(size=n)

candidates = np.column_stack([symmetric_feature, noise])

corr = CorrelationScorer().score_batch(candidates, symmetric_target)
mutual = MutualInformationScorer().score_batch(candidates, symmetric_target)

print(f"{'':>12} {'true feature':>14} {'noise':>10}")
print(f"{'correlation':>12} {corr[0]:>14.4f} {corr[1]:>10.4f}")
print(f"{'mutual info':>12} {mutual[0]:>14.4f} {mutual[1]:>10.4f}")
        """
    ),
    markdown(
        """
Correlation cannot distinguish the true feature from noise here. Mutual
information can.

In practice this matters less than it appears, because beamfeat applies the
scorer to *transformed* columns. Once `square(x)` has been constructed, a
linear scorer detects it easily — which is why correlation remains a sensible
default despite this weakness.
        """
    ),
    code(
        """
squared = symmetric_feature**2  # what the search would hand the scorer
print(f"correlation on the raw column:       {CorrelationScorer().score(symmetric_feature, symmetric_target):.4f}")
print(f"correlation on the squared column:   {CorrelationScorer().score(squared, symmetric_target):.4f}")
        """
    ),
    markdown(
        """
## Redundancy control

A beam scored purely on individual merit fills with variants of the same
signal. beamfeat scores each candidate against the residual left by what is
already selected as well as against the target, and ranks it on the larger of
the two, so a refinement of a selected feature survives while novelty is still
rewarded. It also drops a candidate too correlated with one already admitted at
the same depth; the final ranking applies the same check across depths.
        """
    ),
    code(
        """
from beamfeat import Evaluator

for threshold in (0.5, 0.9, 0.999):
    outcome = BeamSearch(
        max_depth=2, beam_width=30, redundancy_threshold=threshold, random_state=0
    ).run(data, target)

    evaluator = Evaluator(data)
    _, matrix = evaluator.evaluate_many(outcome.nodes)
    correlations = np.abs(np.corrcoef(matrix, rowvar=False))
    np.fill_diagonal(correlations, 0.0)

    print(
        f"threshold {threshold:<6}: {len(outcome):>3} features, "
        f"max pairwise correlation {np.nanmax(correlations):.3f}"
    )
        """
    ),
    markdown(
        """
## Auditing what was rejected

Candidates excluded during evaluation are recorded rather than silently
dropped. This matters when a search returns less than expected: the log tells
you whether candidates were numerically invalid, degenerate, copies of a column
already there — `abs` of a column that is never negative is the column itself —
or simply never proposed.
        """
    ),
    code(
        """
signed_data = dict(data)
signed_data["e"] = rng.normal(0, 2, n)  # negative values break log and sqrt

outcome = BeamSearch(max_depth=1, beam_width=20, random_state=0).run(signed_data, target)

print("rejections by reason:")
for reason, count in outcome.evaluation_log.counts().items():
    print(f"  {reason.value:>15}: {count}")

print("\\nexamples:")
for record in list(outcome.evaluation_log)[:5]:
    print(f"  {record}")
        """
    ),
]


# --------------------------------------------------------------------------- #
# 03: Selection and units
# --------------------------------------------------------------------------- #

selection_and_units = [
    markdown(
        """
# Selection and units

Feature construction proposes far more candidates than any dataset can support.
Selection is where spurious features are either excluded or silently admitted,
and it is where most automated feature engineering tools are weakest: an
importance threshold gives no guarantee about how many reported features are
noise, and the expected number of false positives grows with the size of the
candidate pool.

beamfeat controls the **false discovery rate** — the expected proportion of
selected features that are spurious.
        """
    ),
    code(
        """
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# The selectors also log each diagnostic they record. The cells below print
# them from warnings_raised where they matter, so the logged copies are muted.
logging.getLogger("beamfeat").setLevel(logging.ERROR)

from beamfeat.selection import KnockoffSelector, PermutationSelector

rng = np.random.default_rng(0)
        """
    ),
    markdown(
        """
## The problem, demonstrated

With a pure-noise target, every selected feature is by definition a false
positive. A well-calibrated selector should return almost nothing.
        """
    ),
    code(
        """
n_trials = 15
selections = []

for trial in range(n_trials):
    trial_rng = np.random.default_rng(100 + trial)
    features = trial_rng.standard_normal((200, 30))
    noise_target = trial_rng.standard_normal(200)

    result = PermutationSelector(target_fdr=0.1, n_permutations=20, random_state=trial).select(
        features, noise_target
    )
    selections.append(result.n_selected)

print(f"pure-noise target, {n_trials} trials, 30 candidate features each")
print(f"mean features selected: {np.mean(selections):.2f}")
print(f"trials selecting nothing: {sum(s == 0 for s in selections)}/{n_trials}")
        """
    ),
    markdown(
        """
## Measured calibration

More usefully: with real signal present, what fraction of selections are false?
This is the quantity the library claims to control.
        """
    ),
    code(
        """
def gaussian_design(seed, n=300, n_signal=5, n_noise=20, effect=3.0):
    trial_rng = np.random.default_rng(seed)
    n_features = n_signal + n_noise
    features = trial_rng.standard_normal((n, n_features))
    coefficients = np.zeros(n_features)
    coefficients[:n_signal] = effect
    target = features @ coefficients + trial_rng.standard_normal(n)
    truth = np.zeros(n_features, dtype=bool)
    truth[:n_signal] = True
    return features, target, truth


def realised_fdr(selector_factory, nominal, n_trials=15):
    false_rates, recalls = [], []
    for trial in range(n_trials):
        features, target, truth = gaussian_design(3000 + trial)
        result = selector_factory(nominal, trial).select(features, target)
        if result.n_selected:
            false_rates.append(np.sum(~truth[result.selected]) / result.n_selected)
        else:
            false_rates.append(0.0)
        recalls.append(np.sum(truth[result.selected]) / truth.sum())
    return np.mean(false_rates), np.mean(recalls)


print(f"{'nominal':>8} {'realised FDR':>14} {'power':>8}")
for nominal in (0.05, 0.1, 0.2):
    fdr, power = realised_fdr(
        lambda f, s: PermutationSelector(target_fdr=f, random_state=s, n_permutations=25), nominal
    )
    print(f"{nominal:>8.2f} {fdr:>14.3f} {power:>8.2f}")
        """
    ),
    markdown(
        """
Realised FDR tracks the nominal level, and power is full. Two reading notes,
both scientifically important. First, FDR is an **expectation**: individual
trials may exceed the level, and a mean over 15 trials carries visible Monte
Carlo noise, so values a little above nominal are consistent with control.
Second, a well-calibrated test *spends* its error budget — realised FDR near
nominal is the signature of a calibrated procedure, where a realised FDR
pinned at zero would signal wasted power.

## Knockoffs: two constructions, two sets of assumptions

The knockoff filter comes in two forms, and it matters which one you mean.

**Fixed-X knockoffs** (Barber & Candès, 2015) treat the design as fixed and
make *no assumption about the distribution of the features* — deterministic
engineered columns are admissible. The guarantee needs `n > 2p` and Gaussian
noise in `y = Xβ + ε`. Engineered designs strain it in two ways. Near-duplicate
columns drive the construction's `s` toward zero, the knockoffs become nearly
identical to the originals, and the filter loses the ability to tell them
apart: a cost in **power, not validity**. Exact dependence is worse. When the
design covariance has an eigenvalue below 1e-6 — a constant column, or one that
is a linear combination of others, as `a + b` is of `a` and `b` — the knockoffs
of the columns involved can only equal the originals, the lasso's tie-breaking
then favours the originals, and that costs validity. beamfeat leaves such
columns out of the construction, so they cannot be selected, and names them in
its warning.

**Model-X knockoffs** (Candès et al., 2018) instead assume the features are
jointly Gaussian — which engineered features are not: `log(a)` and `a * b`
are deterministic functions of shared parents. beamfeat only uses this
construction when `n <= 2p`, where the fixed-X construction does not exist,
and warns.

beamfeat routes between the two automatically and reports which assumptions
are under strain, rather than pretending otherwise.

One thing knockoffs will *not* do, since the conditional null invites the
inference: they cannot prune a term out of an expression. Selection operates
on whole candidate columns, so a candidate like `noise + (a * b) / c` is one
column and is accepted or rejected entire — switching the selector does not
reach inside it. Constraining what gets *built* is the lever there, and
notebook 04 uses units for exactly that.
        """
    ),
    code(
        """
def engineered_design(seed, n=300):
    trial_rng = np.random.default_rng(seed)
    a = trial_rng.uniform(1.0, 5.0, n)
    b = trial_rng.uniform(1.0, 5.0, n)
    c = trial_rng.uniform(1.0, 5.0, n)
    columns = [
        a, b, c, np.log(a), np.log(b), np.sqrt(a), 1.0 / a,
        a * b, a / b, b / a, a * c, a - b, a + b, a**2, b**2,
    ]
    return np.column_stack(columns), a * b + trial_rng.normal(0, 0.1, n)


features, target = engineered_design(5000)

knockoff_result = KnockoffSelector(target_fdr=0.1).select(features, target)
print("knockoff selector on an engineered design:")
for message in knockoff_result.warnings_raised:
    print(f"  warning: {message}")

permutation_result = PermutationSelector(target_fdr=0.1, n_permutations=25).select(features, target)
print(f"\\npermutation selector warnings: {permutation_result.warnings_raised or 'none'}")
        """
    ),
    markdown(
        """
So the practical division of labour: the permutation selector is the default
because its validity does not depend on the design at all; fixed-X knockoffs
are a strong choice when there are more than twice as many rows as candidates
and the *noise* is roughly Gaussian (the features can be anything); model-X is
a last resort for `n <= 2p`.

One further caveat, measured rather than assumed: the `offset` parameter
dominates knockoff power on narrow designs. `offset=1` (knockoff+) requires
`(1 + #negatives) / #positives <= target_fdr`, which cannot be satisfied when
there are fewer than `1 / target_fdr` features, regardless of signal strength.
        """
    ),
    code(
        """
print(f"{'offset':>8} {'power':>8}")
for offset in (0, 1):
    recalls = []
    for trial in range(10):
        features, target, truth = gaussian_design(6000 + trial)
        result = KnockoffSelector(target_fdr=0.1, random_state=trial, offset=offset).select(
            features, target
        )
        recalls.append(np.sum(truth[result.selected]) / truth.sum())
    print(f"{offset:>8} {np.mean(recalls):>8.2f}")
        """
    ),
    markdown(
        """
## End to end

Putting search and selection together: construct candidates, then keep only
those that survive FDR control.
        """
    ),
    code(
        """
from beamfeat import BeamSearch, Evaluator

n = 500
data = {name: rng.uniform(1.0, 6.0, n) for name in "abcd"}
y = (data["a"] * data["b"]) / data["c"] + rng.normal(0, 0.05, n)

search_result = BeamSearch(max_depth=2, beam_width=40, random_state=0).run(data, y)
evaluator = Evaluator(data)
nodes, matrix = evaluator.evaluate_many(search_result.nodes)

selection = PermutationSelector(target_fdr=0.1, n_permutations=30).select(matrix, y)

print(f"proposed: {search_result.n_proposed_total}")
print(f"kept by search: {len(search_result)}")
print(f"kept by selection: {selection.n_selected}")
print("\\nselected features:")
for index in selection.selected:
    print(f"  {nodes[index].name}")
        """
    ),
    markdown(
        """
## Dimensional analysis

If your columns carry physical units, supplying them restricts the search to
dimensionally valid expressions. Adding a mass to a length is rejected at
construction time, before any numerical work happens.

This is both a correctness feature and a performance one: it prunes large parts
of the search space for free.
        """
    ),
    code(
        """
import pint

ureg = pint.UnitRegistry()

n = 400
physical = {
    "mass": rng.uniform(1.0, 5.0, n),
    "length": rng.uniform(1.0, 5.0, n),
    "time": rng.uniform(1.0, 5.0, n),
}
units = {
    "mass": 1.0 * ureg.kilogram,
    "length": 1.0 * ureg.meter,
    "time": 1.0 * ureg.second,
}

momentum = physical["mass"] * physical["length"] / physical["time"]
physical_target = momentum + rng.normal(0, 0.05, n)

unconstrained = BeamSearch(max_depth=2, beam_width=25, random_state=0).run(physical, physical_target)
constrained = BeamSearch(max_depth=2, beam_width=25, random_state=0).run(
    physical, physical_target, units=units
)

print(f"without units: {unconstrained.n_proposed_total} candidates proposed")
print(f"with units:    {constrained.n_proposed_total} candidates proposed")
print(f"reduction:     {100 * (1 - constrained.n_proposed_total / unconstrained.n_proposed_total):.0f}%")
        """
    ),
    code(
        """
print("top features with units enforced:")
for name in constrained.names[:5]:
    print(f"  {name}")

invalid = [name for name in constrained.names if " + " in name or " - " in name]
print(f"\\ndimensionally invalid sums or differences: {len(invalid)}")
        """
    ),
    markdown(
        """
Mass plus length never appears, because it was never constructible.

## Units through the estimator API

The same constraint is available on the estimators via the `units` argument.
The fitted equation is compact because, after FDR screening, the estimator
keeps a parsimonious forward-selected subset of the screened features by
default (`parsimony="forward"`); the equation's suffix records how many of the
certified terms it printed, and the full screened set with per-candidate p- and
q-values is available in `selection_report_`.
        """
    ),
    code(
        """
from beamfeat import BeamFeatRegressor

# Rebuild from a fresh generator so this cell does not depend on how many
# draws earlier cells happened to consume.
cell_rng = np.random.default_rng(7)
physical = {
    "mass": cell_rng.uniform(1.0, 5.0, n),
    "length": cell_rng.uniform(1.0, 5.0, n),
    "time": cell_rng.uniform(1.0, 5.0, n),
}
physical_target = (
    physical["mass"] * physical["length"] / physical["time"] + cell_rng.normal(0, 0.05, n)
)

# Units are keyed by column name, so a DataFrame lets the mapping read as
# physics rather than as indices.
X_physical = pd.DataFrame(physical)
# Units may be pint quantities or plain strings; strings are parsed with
# pint at fit time. Cover every column: one left unlabelled is dimensionally
# unconstrained and combines freely with the rest, so partial coverage stops
# the check binding where it matters (beamfeat warns when that happens).
estimator_units = {"mass": "kg", "length": "meter", "time": "second"}

model = BeamFeatRegressor(
    max_depth=2, beam_width=25, units=estimator_units, selector="permutation",
    random_state=0,
).fit(X_physical, physical_target)

print(f"R^2: {model.score(X_physical, physical_target):.4f}")
print(f"features retained: {model.n_features_out_}")
print(model.equation())
        """
    ),
]


# --------------------------------------------------------------------------- #
# 04: Reading a fit
# --------------------------------------------------------------------------- #

reading_a_fit = [
    markdown(
        """
# Reading a fit

The three notebooks before this one plant an answer and check that beamfeat
finds it. Real fits do not arrive that way. They arrive as an equation you did
not write, containing columns you did not expect, with `fdr_controlled_`
reporting `True` and no obvious way to tell a triumph from a coincidence.

This notebook is about that moment. We take a problem with a known physical
answer, hide it among noise, and then get it wrong on purpose — because the
ways it goes wrong are more instructive than the way it goes right, and every
one of them has a fix.
        """
    ),
    code(
        """
import re
import sys
import warnings

import numpy as np
import pandas as pd


# beamfeat's warnings, in order with the printed output and without this
# machine's file paths
def show_warning(message, category, *_):
    sys.stdout.flush()
    print(f"{category.__name__}: {message}", file=sys.stderr, flush=True)


warnings.filterwarnings("default")
warnings.showwarning = show_warning

from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(99)
n = 300
        """
    ),
    markdown(
        """
## The problem

Hydraulic power of a centrifugal pump, `P = rho * g * Q * H / eta`: density,
gravity, flow rate, head, efficiency. Five real variables, buried among eight
columns of plausible-looking nonsense — the ambient noise any plant historian
is full of.

Note that `g` is a constant. Keep an eye on it.
        """
    ),
    code(
        """
columns = {
    "rho": rng.uniform(800, 1200, n),   # density, kg/m^3
    "g": np.full(n, 9.81),              # gravity, m/s^2 — constant
    "Q": rng.uniform(0.01, 0.5, n),     # flow rate, m^3/s
    "H": rng.uniform(10, 100, n),       # head, m
    "eta": rng.uniform(0.6, 0.9, n),    # efficiency, dimensionless
}
power = columns["rho"] * columns["g"] * columns["Q"] * columns["H"] / columns["eta"]
y = power + rng.normal(0, 0.02 * power.std(), n)

for index in range(8):
    columns[f"noise_{index}"] = rng.uniform(0, 50, n)

X = pd.DataFrame(columns)
print(f"{X.shape[0]} rows, {X.shape[1]} columns, {X.shape[1] - 5} of them irrelevant")
        """
    ),
    markdown(
        """
## Start at the defaults

Always. The defaults are `max_depth=2`, `beam_width=50`.
        """
    ),
    code(
        """
model = BeamFeatRegressor(random_state=99).fit(X, y)

print(model.equation())
print(f"R^2: {model.score(X, y):.4f}")
        """
    ),
    markdown(
        """
`(H / eta) * (Q * rho)` is `rho * Q * H / eta` — the physics, exactly, with
none of the eight noise columns in it.

The suffix is the equation being honest about its own scope: screening
certified fifty formulas, most of them pairing the physics core with a noise
column, and the parsimony step kept the one that carries the model. Those
fifty are the set the guarantee covers. `parsimony=None` prints them all, and
the section below on the marginal null explains why the noisy ones are there
and why certifying them is not an error.

### Where did gravity go?

It is in the coefficient. `g` never varies, so no expression containing it can
be distinguished from the same expression without it: `rho * g` is just
`9.81 * rho`, perfectly collinear with `rho`, and the redundancy pass keeps one
representative of the pair. The scaling then lands in the linear model, which
is why the fitted coefficient reads 9.80 rather than 1. Under
`parsimony=None` the same scaling is shared across fifty collinear terms and
the leading coefficient reads lower, which is why a constant is easiest to
read off the compact fit.

This generalises. A multiplicative constant is absorbed into the coefficient
rather than recovered as a symbol, so its absence from the formula is not a
miss. Check the coefficient before concluding a factor was lost.
        """
    ),
    markdown(
        """
## Searching harder makes it worse

Five variables feels like it should need a deep search. Let us give it one:
depth 3, and a wider beam. `verbose=1` shows what each stage did.
        """
    ),
    code(
        """
deep = BeamFeatRegressor(
    random_state=99, max_depth=3, beam_width=60, verbose=1
).fit(X, y)

print()
print(deep.equation())
print(f"R^2: {deep.score(X, y):.4f}")
print(f"fdr_controlled_: {deep.fdr_controlled_}")
print(
    f"candidates proposed: {deep.search_result_.n_proposed_total}, "
    f"against {model.search_result_.n_proposed_total} at the defaults"
)
        """
    ),
    markdown(
        """
The physics is still in there — `(H / eta) * (Q * rho)` — but a junk term
containing `noise_0` has been bolted onto it, and `fdr_controlled_` is still
`True`. The R^2 is unchanged to four decimals. We searched four times as many
candidates for an equation nobody can read.

### Why the flag is still True

This is the single most important thing to understand about the guarantee, and
it is not a defect.

The null being tested is **marginal**: is this expression, *as a whole*,
independent of the target? An expression that combines the true core with a
noise column is emphatically not independent of the target — the core sees to
that — so its null is false and selecting it is not an error under the
criterion. The guarantee certifies that a selected formula is not pure noise.
It does not certify that every column inside it earns its place.

So `fdr_controlled_ = True` means what it says, and it is not the property you
wanted here. Noise hitchhikes on a strong signal.
        """
    ),
    markdown(
        """
## Depth is not the number of variables

The instinct that sent us to `max_depth=3` was wrong arithmetic. Depth counts
*composition levels*, not variables, and a binary operator combines any two
surviving expressions — not a variable onto a chain.

So `rho * Q * H / eta`, four variables, is depth **two**:

| depth | built |
|---|---|
| 1 | `(Q * H)` and `(eta / rho)` |
| 2 | `(Q * H) / (eta / rho)` |

The default reached it on the first try. The extra level bought no accuracy and
gave the search room to decorate a finished answer with noise. Raise `max_depth`
when a target you expect is *not* being recovered, never speculatively: a deeper
search enlarges the candidate pool, every threshold scales as 1/m, and each
extra candidate costs power on the ones that matter.
        """
    ),
    markdown(
        """
## Units are the fix

Statistics cannot tell that `noise_0` does not belong — as we just saw, it is
genuinely associated with the target once attached to the core. Physics can.
You cannot add a dimensionless number to a quantity in kg·m/s, and beamfeat
rejects such expressions at construction, before they are ever scored.
        """
    ),
    code(
        r"""
units = {
    "rho": "kg/m**3", "g": "m/s**2", "Q": "m**3/s", "H": "m",
    "eta": "dimensionless",
    **{f"noise_{index}": "dimensionless" for index in range(8)},
}

gated = BeamFeatRegressor(
    random_state=99, max_depth=3, beam_width=60, units=units
).fit(X, y)

print(gated.equation())
print(f"R^2: {gated.score(X, y):.4f}")


# Formulas that add or subtract a noise column outright, across the whole
# certified set rather than the pruned equation.
ADDED_NOISE = re.compile(r"(noise_\d+ [+-]|[+-] noise_\d+)")


def added_noise(model):
    full = BeamFeatRegressor(**{**model.get_params(), "parsimony": None, "verbose": 0}).fit(X, y)
    return [f for f in full.formulas() if ADDED_NOISE.search(f)], full.n_features_out_


for label, m in (("no units", deep), ("units", gated)):
    hits, total = added_noise(m)
    print(f"certified formulas adding a noise column, {label:8s}: {len(hits)} of {total}")
        """
    ),
    markdown(
        """
The same over-deep search now returns the clean core. Identical R^2, no
parasites — and the effect is not just in the pruned equation: across the whole
certified set, not one formula adds a noise column to a dimensioned quantity,
against almost all of them before. That is the general lesson — where a
statistical criterion cannot separate two candidates, a structural constraint
often can, and it costs nothing because it applies before any numerical work.

Two cautions about what units check.

They enforce that an expression is internally coherent, not that it matches the
target's dimension. `rho * Q * H / eta` is kg·m/s while power is watts, and
nothing objects, because `g` was absorbed. Units reject nonsense; they will not
tell you a factor is missing.

And a *dimensionless* column is legal almost everywhere. `Q / noise_7` is still
m³/s, so noise can ride in multiplicatively even under a full labelling, and
most of the certified set still mentions one. What units removed is the whole
class of violations — adding a bare number to a physical quantity — not every
appearance of an irrelevant column.
        """
    ),
    markdown(
        """
### Cover every column

Labelling the five real measurements and leaving the noise columns blank looks
like the careful thing to do. It is not.
        """
    ),
    code(
        """
partial = {
    "rho": "kg/m**3", "g": "m/s**2", "Q": "m**3/s", "H": "m",
    "eta": "dimensionless",
}

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    half_gated = BeamFeatRegressor(
        random_state=99, max_depth=3, beam_width=60, units=partial
    ).fit(X, y)

for entry in caught:
    if "units cover" in str(entry.message):
        print(str(entry.message)[:200])
        break

print()
print(half_gated.equation())
        """
    ),
    markdown(
        """
`noise_0` is back, added directly to a kg·m/s quantity — the exact violation
the units pass exists to catch. An unlabelled column is dimensionally
*unconstrained*, so it combines freely with everything, and the gate stops
binding on precisely the columns you did not vouch for. Units are all or
nothing; give the genuinely unitless columns `"dimensionless"`.
        """
    ),
    markdown(
        """
## The equation is not the certified set

`equation()` prints the parsimony subset, and its suffix says so. The guarantee
covers the larger screened set, which `selection_report_` holds in full, with
a permutation p-value and its q-value per candidate.
        """
    ),
    code(
        """
report = deep.selection_report_
screened = [row for row in report if row["screened"]]
printed = [row for row in report if row["kept"]]

print(f"candidates screened: {len(report)}")
print(f"certified (guarantee applies here): {len(screened)}")
print(f"terms in equation(): {len(printed)}")
print(f"|S|/|S'|: {deep.fdp_inflation_:.2f}")
print(f"fdr_controlled_: {deep.fdr_controlled_}   fdr_scope_: {deep.fdr_scope_!r}")
print()
for row in sorted(screened, key=lambda r: r["p_value"])[:3]:
    formula = row["formula"]
    shown = formula if len(formula) <= 62 else formula[:59] + "..."
    print(f"  {shown:62} p={row['p_value']:.2e} q={row['q_value']:.2e}")
        """
    ),
    markdown(
        """
The gap matters. Parsimony picks a subset by greedy forward selection on the
same rows, and a data-dependent subset of an FDR-controlled set does not
inherit the guarantee: pruning cannot add false selections, but it can raise
the false discovery *proportion*, because the denominator shrinks faster than
the numerator, and `fdp_inflation_` reports by how much. This is why
`fdr_controlled_` and `fdr_scope_` are two attributes rather than one: the
flag is `True` and the scope is the screened set, not the printed equation. What that costs in
fit is small and has been measured — about a thousandth of held-out R^2 across
308 paired fits, in `benchmarks/PARSIMONY_COST.md`. Report the screened set
when you need the guarantee; use the equation when you need something to read.
        """
    ),
    markdown(
        """
## When nothing is selected

Two very different situations produce an empty selection, and they have
different fixes. Here is the honest one — a target that is pure noise.
        """
    ),
    code(
        """
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    empty = BeamFeatRegressor(random_state=0).fit(X, rng.normal(0, 1, n))

print(f"features constructed: {empty.n_features_out_}")
print(f"fdr_controlled_: {empty.fdr_controlled_}")
for entry in caught:
    print(f"  warning: {str(entry.message)[:130]}")
        """
    ),
    markdown(
        """
An intercept-only model and a visible warning, which is the correct answer:
there was nothing to find, and the constructor declined to invent it.
`fdr_controlled_` reads `False` here rather than `True` — no feature was
returned, so nothing carries the guarantee, and the flag describes what came
back rather than how hard the procedure tried.

The other case looks identical from the outside but is not. If the candidate
pool is large enough that the multiplicity threshold falls below the smallest
p-value the permutation budget can produce, the correction cannot reject
*anything*, whatever the data say. The selector detects this and says so in its
warning, naming the correction and what to change. So read the warning before
concluding there is no signal: an empty selection with a budget warning means
the test could not fire, not that the data are silent.
        """
    ),
    markdown(
        """
## A checklist

- Start at the defaults. Raise `beam_width` when an expected target is missed;
  raise `max_depth` only when the target genuinely needs the depth, and count
  composition levels, not variables.
- An equation that grew noise terms is a sign you searched too deep, not that
  you need to search deeper.
- `fdr_controlled_ = True` says the formula is not noise. It does not say every
  column in it belongs.
- Supply units when you have them, for every column.
- A missing constant is usually in the coefficient.
- Quote `selection_report_` when you need the guarantee; quote `equation()`
  when you need a sentence.
- Read the warnings. They distinguish states that look identical from outside.
        """
    ),
    markdown(
        """
## What to read next

- **02: Search and scoring** — beam width, depth and the scoring strategies,
  with the trace that shows where the cost went.
- **03: Selection and units** — the calibration behind the guarantee, why the
  default is not knockoffs, and the dimensional analysis used above.
        """
    ),
]



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the tutorial notebooks.")
    ap.add_argument("--sync-docs", action="store_true",
                    help="copy the executed notebooks into docs/notebooks/ instead of building")
    if ap.parse_args().sync_docs:
        sync_docs()
    else:
        build(HERE / "01_getting_started.ipynb", getting_started, "Getting started with beamfeat")
        build(HERE / "02_search_and_scoring.ipynb", search_and_scoring, "Search and scoring")
        # 03 and 04 need pint for their units sections
        build(HERE / "03_selection_and_units.ipynb", selection_and_units, "Selection and units",
              requirement='"beamfeat[units]"')
        build(HERE / "04_reading_a_fit.ipynb", reading_a_fit, "Reading a fit",
              requirement='"beamfeat[units]"')
