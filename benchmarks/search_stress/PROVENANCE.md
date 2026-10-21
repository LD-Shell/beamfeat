# Provenance

What this stress test fixed in advance, what it changed after a pilot and
why, and what a rerun will and will not reproduce. Read it before citing a
figure from `REPORT.md`.

## What is measured

The beam search keeps, at each depth, the candidates that correlate best
with the target on the search rows. `stress.py` measures how often that
ranking loses a law because the law is built from parts that do not
correlate with the target: with centred, symmetric inputs, `x0` and `x1` are
uncorrelated with `x0*x1`, and each pairwise product is uncorrelated with
`x0*x1*x2`. Two ways of losing such a law were named in advance:

- **pruned**: the part the law needs is proposed at depth 1 but ranks among
  noise and leaves the beam before it can be combined;
- **unproposed**: when `max_candidates_per_depth` binds at depth 1 (from 258
  inputs at the defaults), the inputs are proposed in order of their score,
  and a pair of inputs neither of which scores is never proposed.

For every fit it records whether the printed equation is the law, by the
Feynman panel's exact-form criterion (`benchmarks/feynman_panel.py`); the
held-out R² against the noiseless law on 2,000 test rows the fit never saw;
the fit's wall time; and where a miss happened, read from the search itself
through a subclass of `BeamSearch` (below). A miss is put down to the first
stage that lost the law: never proposed, pruned at depth 1, lost later in the
search, not passed by the screen, or passed but printed with other terms.

## Fixed before the first fit

- The laws: `x0*x1`, `x0*x1*x2` and `x0*x1 + x2`, with `x0`, `x1`, `x2` the
  first columns and the rest independent distractors, p columns in all.
- The inputs: U(-1, 1) and N(0, 1), centred, and U(0.5, 2.5), the control on
  which every part of each law correlates with it.
- The widths p in {5, 20, 100, 300}, on both sides of the ceiling; the rows,
  500 and 2,000; noise of 1% and 10% of the law's standard deviation,
  computed exactly from the input moments; the beams, 50 (the default) and
  200. The draws depend on the inputs, p, n and the seed, so a replicate's
  laws, noise levels and beams share every input value and one standard
  normal noise vector, and comparisons between them are paired.
- 50 replicates per cell; the measures and the miss stages above; the
  equivalence threshold for "the law is among these expressions" (the same
  input columns and an absolute correlation of at least 0.9999 with the
  law's term on the test rows); 2,000 test rows.
- Every fit is the estimator as a user calls it: `BeamFeatRegressor()` or
  `BeamFeatRegressor(beam_width=200)`, everything else at its default,
  `random_state=0` included.

## Changed after a pilot, and why

A pilot of one replicate per cell on the triple law (16 cells), and of three
on the two pair laws at p = 300 (12 fits), neither archived, timed the fits:
up to 41 s at p = 300 with 2,000 rows and a beam of 200. The full crossing of
the six factors, 288 cells, would have taken some fifteen hours on three
workers. The grid was therefore cut to a base and one factor at a time:

- the base, beam 50 with 500 rows and 10% noise, crossed with every law,
  input distribution and width;
- the wide beam on the centred inputs up to p = 100, where the triple law's
  misses lie, and at p = 300 for the triple and sum laws;
- the lower noise, 1%, at 500 and at 2,000 rows, up to p = 100.

Four cells were added after the pilot had shown something, and are reported
as such. At p = 300 the pilot found the sum law's product, `x0*x1`, proposed
but ranked about 790th at depth 1 with 500 rows: 300 inputs on 250 search
rows leave the residual the search also ranks against empty, and some 600
sums and differences with `x2` outscore `x0*x1`. The pair and sum laws were
therefore added at p = 300 with 2,000 rows (1% noise), where the residual is
not empty. And the pair law was added at p = 1,000 with its shifted control,
to see the proposals' loss past the ceiling at a width where the analysis
predicts it is large; no fit at that width had been made.

The grid then held 114 cells, 5,700 fits.

## `magnitude_slots`

The option was written after the default settings' records were complete,
and measured at 10 slots, a fifth of the default beam, a value fixed before
any of its runs:

- on the stress grid's 48 cells at beam 50 (every cell but the wide beam's),
  on the default's replicates, so that every comparison is paired;
- on the main suite and the high-dimensional study, run unchanged through
  `with_options.py`, which sets the option as a new default for the
  process;
- by `null.py` on four null and two planted designs, 200 paired replicates
  each, defined before its first fit.

The first version computed the magnitude score on every search row. Its run
of the suites took 2.2 times the default's processor time on the three
10,000-row sets, timed side by side, against 1.2 to 1.7 on the stress grid,
whose fits have at most 1,000 search rows. The score was then capped at
1,000 search rows, evenly spaced, and the suites were run again; those are
the figures the report gives, with the first run's kept in
`results/suites/magnitude_slots_uncapped/`. The cap cannot change a fit with
1,000 search rows or fewer, which covers every fit of the stress grid and of
the null study: their records were made before it, and four stress
replicates, one with exactly 1,000 search rows, gave the same fit to the bit
before and after it.

The fit times in the records were taken hours apart on a shared machine, so
the option's cost was measured side by side as well: `stress.py --timing`
fits ten replicates of ten base cells at the default and with the option in
turn, in one process.

## Following the search

`stress.py` replaces the `BeamSearch` the estimators module looks up with a
subclass that calls the package's own methods and only reads their arguments
and results: at each depth, whether the parts the law needs were proposed,
their rank among that depth's candidates (as the beam ranks them, before the
redundancy check) and whether the beam kept them, and, at depth 1 when the
ceiling binds, the order the inputs were proposed in. The package is not
modified. The first replicate of every cell is refitted with the plain
`BeamSearch` and must match the traced fit to the last bit (features,
coefficients, intercept, test predictions), or the run stops; the 114 checks
all matched.

## Environment

Python 3.12.14, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, sympy 1.14.0,
beamfeat at the commit `summary.json` names, imported from this checkout's
`src`; the option's records name their commits, which differ from the
default's only by the option and by records, documentation and the
benchmark scripts. Three worker processes at `nice 10`, one BLAS thread
each, on a 22-thread machine that was running other long jobs at the same
time. Fit
times are wall times on that shared machine: compare them within the study,
across cells, rather than with timings from elsewhere.

## What reproduces

Every record but the times, given the environment: the data come from
`numpy.random.default_rng` seeded with the inputs, p, n and the replicate,
and a fit is deterministic. Another numpy, BLAS or scikit-learn can move a
borderline rank or permutation count, and so a few records; the tables
should then agree within their standard errors.

## Scope

Three laws, built only of products and a sum, at depth 2, the default
operators and scorer, regression only. The ceiling-bound cells are few:
p = 300 and p = 1,000 for the pair law, p = 300 for the others. The study
says nothing about real data, where a law's parts rarely have exactly zero
correlation with the target.

## Running it

From the repository root, with this checkout's `src` importable (the run
used `PYTHONPATH=src` and `OMP_NUM_THREADS=1`):

```bash
python benchmarks/search_stress/stress.py --workers 3 --out <dir>   # 92 minutes here; resumable
python benchmarks/search_stress/stress.py --tabulate --out <dir>    # tables from the records
```

For the option, `--setting magnitude_slots=10 --kwargs '{"magnitude_slots": 10}'`
with `--beams 50` (the run used three invocations, which `summary.json`
lists), then `--compare results/default` and `--timing`; `null.py --out <dir>`
for the null study; and the commands in `suites.py`'s docstring for the
suites.

It writes `parts/` (one record file per cell, while running),
`records.csv.gz` (one row per fit), `summary.json` (provenance and every
aggregate) and `tables.md`; `stress.log` is its output. The default
settings' records are in `results/default/`, copied from the directory the
run wrote to, which the log shows as `<out>`; `--tabulate` rebuilds their
`summary.json` and `tables.md` from `records.csv.gz` alone, byte for byte.
