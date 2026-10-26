# Provenance

What this calibration fixed in advance, what it changed after a smoke run
and why, and what a rerun will and will not reproduce. Read it before citing
a figure from `REPORT.md`.

## What is measured

`parsimony_holdout` splits the selection rows again: screening and the
forward parsimony step use the first part, and the subset they return, now
fixed, is re-tested on the part held back. Before this study the guarantees
page said the two-stage procedure had not been FDR-calibrated. `calibration.py`
measures it, together with `certify_test="conditional"`, which re-tests each
frozen term given the others instead of on its own.

For every fit it records, about the printed equation (the returned features):

- **false terms, marginal sense**: printed terms built only from columns
  outside the generating formula;
- **false terms, conditional sense**: printed terms using any column outside
  it, so a rider such as `(a*b/c) * x7` counts;
- **the true term**: a printed term using exactly `a`, `b` and `c` and
  correlating at least 0.999 in absolute value with `a*b/c` on the test rows,
  which admits every algebraic rewriting the search produces;
- the number of printed terms, the number screened, the re-test's survivors,
  and whether the fit **fell back** to the screened set (the re-test ran, or
  was due, and certified nothing);
- **held-out R²** on 2,000 test rows drawn with the training rows but never
  passed to `fit`. No score uses a row the search, the screen, the parsimony
  step, the re-test or the final ridge fit saw.

The false discovery rate in each sense is the mean, over replicates, of the
false share of the printed terms, with 0 for an empty equation. Every figure
in the report carries its Monte Carlo standard error (the standard deviation
over replicates divided by the square root of their number; for a design
pooled over its four cells, the cell errors combined for an equally weighted
mean).

## Fixed before the first fit

- The designs: a global null (`y` independent of `X`, `n` in {300, 1000},
  `p` in {5, 20}); a planted `y = a*b/c + noise` with `k` in {5, 20}
  distractors independent of `a`, `b`, `c`; and a proxy design, the planted
  one plus a column correlated with a true input. All inputs are U(1, 3), so
  no generated formula meets a pole. The planted and proxy designs use the
  null design's `n` in {300, 1000}.
- Noise with the signal's standard deviation, signal to noise 1, population
  R² 0.5, computed exactly for U(1, 3) inputs rather than estimated.
- The proxy: `a + U(-w, w)` with `w` = 0.484, which correlates 0.9 with `a`,
  appended as the last column, drawn after everything else so that the
  planted and proxy replicates of a seed share every other array.
- The settings: `parsimony="forward"` (the default, whose printed equation is
  not certified), `parsimony=None` (the screened set), `parsimony_holdout`
  0.3 and 0.5 with the marginal re-test, and the same two with
  `certify_test="conditional"`. Everything else at the estimator defaults:
  depth 2, beam 50, Benjamini-Yekutieli, `selection_holdout=0.5`,
  `random_state` equal to the seed.
- The levels, q = 0.1 and 0.2; 200 replicates per cell; the measures and the
  0.999 equivalence threshold above; 2,000 test rows.

## Changed after a smoke run, and why

A smoke run of three replicates per cell (432 fits, not archived) showed that
at signal to noise 1 the default forward step keeps 15 to 30 terms. A term
fitting only noise adds about (1 − R²)/m to the in-sample R² on the m rows
parsimony uses, which clears `parsimony_tol = 1e-3` whenever m is below about
500 at R² 0.5. The subsets every re-test sees were therefore long and
collinear, almost all riders on the true term, and the conditional re-test
certified nothing and fell back to the screened set in 11 of the 12 planted
replicates. A probe at a third of that noise (R² 0.9, two replicates per
cell) gave subsets of 1 to 14 terms, and the conditional re-test kept a
single term whenever the subset was short.

One noise level would therefore have reported the noise level rather than
the procedure. A second, signal to noise 3 (noise standard deviation a
third of the signal's), was added to the planted and proxy designs, from the
same standard normal draws, so the two levels are paired replicate by
replicate. Nothing else changed, and the signal-to-noise-1 cells are those
fixed in advance. The choice of 3 was made with knowledge of the probe; it is
the level at which the subsets' length varies across replicates, not one
chosen to favour a setting, and every setting is reported at both levels.

## The memoised search

The search depends on the search rows and the estimator's search options,
and on neither the parsimony setting nor the level. The script therefore
runs it once per replicate and lets the twelve fits of that replicate reuse
it, by replacing the `BeamSearch` name the estimators module looks up with a
subclass that memoises `run` on a hash of its constructor arguments and every
array and option it is given. The estimator code is otherwise unmodified. The
first two replicates of every cell (480 fits) are refitted without the memo
and compared bit for bit — features, coefficients, intercept, test-set
predictions and re-test p-values — and the run stops on any difference;
`summary.json` records the count that matched.

## Environment

Python 3.12.14, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, beamfeat at
the commit `summary.json` names, imported from this checkout's `src`. One
process at `nice 10` with `OMP_NUM_THREADS=2`, on a 22-thread machine that
was running another long experiment at the same time. The study records no
timings; `wall_seconds` in `summary.json` is the run's length on that shared
machine and nothing more.

## What reproduces

Every record, given the environment: the data come from
`numpy.random.default_rng` seeded with the design, the size and the
replicate, and beamfeat is deterministic given `random_state`. Another numpy,
BLAS or scikit-learn can move a borderline permutation count or a ridge
penalty and so a few records; the tables should then agree within their
standard errors, not to the digit.

The settings that run the conditional re-test after the forward step without
`parsimony_alpha` (`conditional_0.3`, `conditional_0.5`) raise from 0.8;
rerun them with 0.7.0.

## Scope

One formula, one operator set, two noise levels, inputs on a bounded positive
range, regression only, and the screen's default Benjamini-Yekutieli
correction. The marginal-sense rate is hardly stressed: a term built only
from distractors has to pass a conservative screen first. The study says
nothing about real data, about classification (where the conditional re-test
is not available), or about `parsimony_tol` and other options left at their
defaults.

## Running it

From the repository root, with this checkout's `src` importable (the run
used `PYTHONPATH=src`):

```bash
python benchmarks/parsimony_holdout/calibration.py --out <dir>   # 97 minutes on one core here
python benchmarks/parsimony_holdout/calibration.py --tabulate     # tables from results/
```

The run checkpoints per replicate and resumes where it stopped. It writes
`records.csv.gz` (one row per fit), `summary.json` (provenance and every
aggregate), `tables.md` and `examples.json` (the printed formulas of the two
replicates per cell that were checked against the unmemoised search);
`calibration.log` is its output.

## `parsimony_alpha.py` (added for 0.6.1)

Reuses this study's designs, records and tables, at q = 0.1 only, with four
settings: the default forward step and `parsimony_holdout=0.3,
certify_test="conditional"`, each with and without `parsimony_alpha=0.05`.
Fifty replicates per cell, one per cell checked against the unmemoised
search. The level 0.05 and the settings were fixed before the run, with no
smoke run on these designs; a first run from the uncommitted code was
repeated from commit `aa9ea25` and matched it record for record. That run
is in `results_alpha/` (588 seconds on one core here):

```bash
python benchmarks/parsimony_holdout/parsimony_alpha.py --out <dir>
python benchmarks/parsimony_holdout/parsimony_alpha.py --tabulate   # tables from results_alpha/
```
