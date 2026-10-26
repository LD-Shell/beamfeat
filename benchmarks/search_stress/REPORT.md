# The search on laws whose parts do not correlate with the target

The search keeps, at each depth, the candidates that correlate best with the
target, and builds the next depth from them. With centred, symmetric inputs a
law can need parts that do not correlate with the target at all: `x0` and
`x1` in `x0*x1`, the pairwise products in `x0*x1*x2`. This study measures how
often the search then loses the law, where it loses it, and what
`magnitude_slots`, an option added after the measurement, recovers and costs.

5,700 fits at the default settings, 114 cells of 50 replicates, made by
`stress.py` at commit `6ee3b22`; later commits add the comparison and the
`magnitude_slots` records and do not change what a default fit records.
`PROVENANCE.md` says what was fixed in advance, what was added after a pilot
(four cells) and what a rerun reproduces. Every fit is the estimator as a
user calls it, `BeamFeatRegressor()` or `BeamFeatRegressor(beam_width=200)`.
**Exact** is the Feynman panel's criterion: the printed equation is the law,
term for term, up to constant factors. Means carry their Monte Carlo standard
errors (the standard deviation over replicates over the square root of their
number); with 50 replicates a rate of 0 or 1 has a one-sided 95% bound of
0.06 from it. The first replicate of every cell was refitted with the
package's own `BeamSearch` in place of the traced one, and all 114 matched it
to the bit.

## The default search

### Recovery

Exact recovery at the defaults, with 500 rows (250 for the search) and noise
of 10% of the law's standard deviation; misses by the stage that lost the
law, where there were any.

| law | inputs | p = 5 | 20 | 100 | 300 | 1,000 |
| :-- | :-- | --: | --: | --: | --: | --: |
| `x0*x1` | U(-1, 1) | 1.00 | 1.00 | 1.00 | 0.98 ± 0.02 (1 unproposed) | 0.32 ± 0.07 (34 unproposed) |
| `x0*x1` | N(0, 1) | 1.00 | 1.00 | 1.00 | 0.98 ± 0.02 (1 unproposed) | |
| `x0*x1` | U(0.5, 2.5) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `x0*x1*x2` | U(-1, 1) | 1.00 | 0.54 ± 0.07 (23 pruned) | 0.18 ± 0.05 (41 pruned) | 0.10 ± 0.04 (45 pruned) | |
| `x0*x1*x2` | N(0, 1) | 1.00 | 0.80 ± 0.06 (10 pruned) | 0.38 ± 0.07 (31 pruned) | 0.12 ± 0.05 (44 pruned) | |
| `x0*x1*x2` | U(0.5, 2.5) | 1.00 | 1.00 | 1.00 | 1.00 | |
| `x0*x1 + x2` | U(-1, 1) | 1.00 | 1.00 | 1.00 | 0.00 (44 pruned, 6 unproposed) | |
| `x0*x1 + x2` | N(0, 1) | 1.00 | 1.00 | 1.00 | 0.96 ± 0.03 (2 pruned) | |
| `x0*x1 + x2` | U(0.5, 2.5) | 1.00 | 1.00 | 1.00 | 1.00 | |

**On the shifted inputs every law was recovered in every fit, at every
width.** There each factor correlates with the product, so the parts rank
first.

**A three-way product on centred inputs is lost at depth 1, and more so as
the table widens.** Each pairwise product is uncorrelated with `x0*x1*x2`, so
the beam keeps one only when it happens to rank among the 50 best of the
round's candidates. The best of the three ranked, at the median, 49th of
1,200 candidates with 20 columns, 1,038th of 30,000 with 100 and 8,074th of
199,400 with 300 (uniform inputs; 20th, 171st and 1,711th on Gaussian ones,
whose heavier tails let an uncorrelated product's sample correlation stray
further from zero). On the shifted inputs it ranked first in every fit. With
five columns the depth has only 75 candidates and the beam holds most of
them.

**A centred pair is lost before it is scored once the proposals are capped.**
From 258 columns the first depth's candidates exceed
`max_candidates_per_depth`, and the inputs are proposed in order of their
score, each with all its partners, until the 200,000 candidates are used: up
to about the 199th input of 300 and the 49th of 1,000. A pair is proposed
only when one of its inputs comes that early, and the records agree: the pair
was proposed in every fit where `x0` or `x1` came within those places and in
none where both came later. With 1,000 columns both came later in 34 of 50
fits; on the shifted inputs `x0` and `x1` were first in every fit.

**A product beside a visible term is crowded out when the inputs outnumber
the search rows.** In `x0*x1 + x2` on U(-1, 1), `x0*x1` correlates 0.5 with
the target, but each sum or difference of `x2` with another column correlates
0.61, and with 300 inputs on 250 search rows the residual the search also
ranks against is empty. `x0*x1` was proposed in 44 of the 50 fits and ranked
797th at the median, below the sums and differences, and was kept in none. On
N(0, 1) the two terms have equal variance, `x0*x1` correlates 0.70 with the
target and the sums 0.50, and it ranked first in 47 of 50 fits. With 2,000
rows (1,000 for the search) the residual is not empty and the law was
recovered in 0.84 ± 0.05 of fits; every miss there was a pair never proposed.

All 651 misses, in every cell of the study, happened at depth 1: 591 parts
proposed and pruned, 60 never proposed. Whenever the part survived depth 1
the search found the law, the screen passed it and the printed equation was
exactly the law.

### What a miss looks like

A miss is mostly visible as one. In 481 of the 651 fits nothing passed the
screen: an empty equation with a `NoDiscoveriesWarning`. In 165 the equation
held terms that use a column of the law beside distractors, such as
`((x1 * x228) - (x1 * x236))` for `x0*x1*x2`, and for the sum law `x2` with
riders, at a held-out R² of 0.73 (median); in one it held `x2` alone. In 4
fits of the 5,700, all on the three-way product, the equation held a term
built only from distractors. On a missed product the held-out R² was at most
0.015 for the triple law; one missed pair printed riders scoring 0.51.

### A wider beam

| law, inputs | p | exact, beam 50 | exact, beam 200 | paired difference | fit s, 50 → 200 |
| :-- | --: | --: | --: | --: | :-- |
| `x0*x1*x2`, U(-1, 1) | 20 | 0.54 ± 0.07 | 0.96 ± 0.03 | +0.42 ± 0.07 | 0.7 → 6.3 |
| `x0*x1*x2`, U(-1, 1) | 100 | 0.18 ± 0.05 | 0.26 ± 0.06 | +0.08 ± 0.04 | 1.6 → 5.9 |
| `x0*x1*x2`, U(-1, 1) | 300 | 0.10 ± 0.04 | 0.10 ± 0.04 | 0.00 | 8.0 → 11.4 |
| `x0*x1*x2`, N(0, 1) | 20 | 0.80 ± 0.06 | 0.96 ± 0.03 | +0.16 ± 0.05 | 0.8 → 5.3 |
| `x0*x1*x2`, N(0, 1) | 100 | 0.38 ± 0.07 | 0.64 ± 0.07 | +0.26 ± 0.06 | 1.6 → 5.8 |
| `x0*x1 + x2`, U(-1, 1) | 300 | 0.00 | 0.00 | 0.00 | 6.2 → 12.3 |

Fit times are medians. Four times the beam closes the gap at 20 columns, at
seven to nine times the fit time, and leaves most of it at 100, where the
parts rank in the thousands; at 300 it changes nothing. The other laws it was
run at, the pair and sum laws up to 100 columns on centred inputs, were
already recovered in every fit and stayed so.

### Noise and rows

The noise level and the number of rows hardly move the three-way product's
recovery: at 20 columns 0.54, 0.64 and 0.58 (uniform) and 0.80, 0.80 and 0.74
(Gaussian) with 500 rows at 10% noise, 500 at 1% and 2,000 at 1%; at 100
columns 0.18, 0.16 and 0.22, and 0.38, 0.38 and 0.42. Only the first step, to
less noise at 20 uniform columns, exceeds its standard error (paired, +0.10 ±
0.04); four times the rows moved recovery by 0.04 to 0.06 either way, within
one standard error. The rank of an uncorrelated part among uncorrelated
candidates does not depend on how precisely the correlations are estimated.
Rows matter where the residual does: the sum law at 300 columns went from
0.00 to 0.84. Every other law, input and width up to 100 columns was
recovered in every fit at both noise levels and both row counts.

### Time

The median default fit took 0.4 to 0.5 s with 5 columns, 0.5 to 0.8 s with
20, 1.4 to 2.4 s with 100, 5.4 to 9.2 s with 300 and 12 s with 1,000 (500
rows, one core of a shared machine).

## `magnitude_slots`

`magnitude_slots`, added to the search after the measurement above, keeps a
few more candidates at each depth but the last: those whose magnitude best
tracks the target's, by the absolute correlation of `log(1 + |c|/rms(c))`
with the same transform of the centred target or of its residuals. When the
first depth cannot propose every pair of inputs, it also proposes them in an
order that interleaves that ranking with the score's. One residual is taken
on the best-scoring columns kept so far, at most a quarter as many as the
search rows, for the case the sum law exposed: the residual on every input
is empty once the inputs outnumber the rows. It was measured at 10 slots, a
fifth of the default beam, a value fixed before any of the runs below: on
the stress grid's 48 cells at beam 50, on the same replicates as the
default's; on the main suite and the high-dimensional study; and on null
and planted targets.

The score is computed on at most 1,000 search rows, evenly spaced. That cap
came after the first run of the suites, which had none: on the three
high-dimensional sets of 10,000 rows (3,750 for the search) it took 2.2 times
the default's time, against 1.2 to 1.7 on the stress grid, because the score
was computed on every row of up to 200,000 candidates. The suites were run
again with the cap, and those are the figures below. No stress-grid fit and
no fit of the null study had more than 1,000 search rows, so the cap leaves
them as they were: by construction, and checked on four replicates, including
one with exactly 1,000 rows.

### On the stress grid

Exact recovery, the default against `magnitude_slots=10`, in the 21 cells
where the default lost anything:

| law, inputs | p | 500 rows, 10% noise | 500 rows, 1% | 2,000 rows, 1% |
| :-- | --: | :-- | :-- | :-- |
| `x0*x1*x2`, U(-1, 1) | 20 | 0.54 → 1.00 | 0.64 → 1.00 | 0.58 → 1.00 |
| | 100 | 0.18 → 1.00 | 0.16 → 1.00 | 0.22 → 1.00 |
| | 300 | 0.10 → 1.00 | | |
| `x0*x1*x2`, N(0, 1) | 20 | 0.80 → 1.00 | 0.80 → 1.00 | 0.74 → 1.00 |
| | 100 | 0.38 → 1.00 | 0.38 → 1.00 | 0.42 → 1.00 |
| | 300 | 0.12 → 1.00 | | |
| `x0*x1`, U(-1, 1) | 300 | 0.98 → 1.00 | | 0.92 → 1.00 |
| | 1,000 | 0.32 → 1.00 | | |
| `x0*x1`, N(0, 1) | 300 | 0.98 → 1.00 | | |
| `x0*x1 + x2`, U(-1, 1) | 300 | 0.00 → 0.88 ± 0.05 | | 0.84 → 1.00 |
| `x0*x1 + x2`, N(0, 1) | 300 | 0.96 → 1.00 | | |

The other 27 cells, every law on the shifted inputs and every law up to 100
columns that the default already recovered, stayed at 1.00. No cell got
worse. The option's only misses were 6 fits of the sum law with 300 columns
and 500 rows, in which `x0*x1` was proposed but ranked about 800th by score
and missed the magnitude slots as well; the slots' ranking proposed the pair
in every fit, where the default had not in 6. Terms built only from
distractors were printed in 3 of these cells' default fits and in none of the
option's. Every one of the 48 cells' first replicates matched the plain
search to the bit. `results/magnitude_slots/compare.md` gives each cell's
paired difference with its standard error, its held-out R² and its times.

### Time

Fit times recorded at different hours on a shared machine are not comparable,
so the cost was measured side by side: ten replicates of ten base cells, each
fitted at the default and with the option in turn, in one process
(`stress.py --timing`; `results/magnitude_slots/timing.md`). The option's
median fit took 1.20 to 1.43 times the default's in nine cells and 1.69 times
in the tenth, the three-way product with 100 columns: 0.6 to 0.7 s became 0.8
to 0.9 s with 20 columns, 2.2 to 2.6 s became 3.3 to 3.7 s with 100, 10.4 to
11.2 s became 14.7 to 15.9 s with 300, and 14.1 s became 17.0 s with 1,000.
The time goes to the magnitude score of every candidate of the first depth
and to the ten extra parents of the second; the last depth computes no
magnitudes. A beam of 200, which closed less of the gap, cost four to nine
times the fit time.

### On the suites

The main suite (`benchmarks/run_benchmarks.py --suite all`) and the
high-dimensional study (`benchmarks/additional_experiments/bench.py highdim`,
the estimator, five splits) ran unchanged, with the option set as a new
default by `with_options.py`, and `suites.py` compares the runs
(`results/suites/`). The default runs reproduce 0.5.0's archived fits
exactly: all 25 of the main suite, and all 45 high-dimensional fits, R² and
formulas.

| | default | `magnitude_slots=10` |
| :-- | --: | --: |
| main suite, mean held-out score (24 regression sets) | 0.8851 | 0.8841 |
| main suite, fits identical to the default | | 19 of 24 |
| main suite, formula recovered | 15 of 15 | 15 of 15 |
| high-dimensional study, mean held-out R² (9 sets) | 0.647 | 0.649 |
| Feynman panel, solved and exact (both operator sets) | 10 and 8 | 10 and 8 |

On the main suite the option refused the one classification set, as it must.
Of the 24 regression sets, 19 gave the default's fit to the last digit, and
four scored lower: california housing by 0.012 (0.699 to 0.687), diabetes by
0.011 (0.382 to 0.370), and two synthetic sets by 0.002 and 0.0004; friedman1
differed in the last digits only. On the high-dimensional study no dataset's
mean moved by more than 0.012: superconduct went from 0.731 to 0.740 (four
splits up, one down), blogfeedback from 0.415 to 0.427 and eyedata from 0.416
to 0.420, riboflavin from 0.340 to 0.338, and the other five by 0.001 or
less. The three 10,000-row sets, timed side by side, took 1.42 times the
default's processor time (572 s against 402 s) with the row cap, and 2.21
times without it; peak memory was 547 MB against 540 MB. The panel's laws, on
positive inputs, were found as before, with 0.43 to 0.49 s per law against
the default's 0.28 s.

### False discoveries

Two measures. On the stress grid the option printed no term built only from
distractors in its 2,400 fits, against 3 in the default's fits of the same
replicates. And `null.py` fits 200 paired replicates of six designs at the
default and with the option, keeping the screened set (`parsimony=None`), the
set the guarantee covers, at q = 0.1:

| design | inputs | any false feature, default | with the option | discordant (option only / default only) | FDR, default → option |
| :-- | :-- | --: | --: | :-- | :-- |
| null | U(0.5, 3), 8 columns | 0.020 ± 0.010 | 0.020 ± 0.010 | 0 / 0 | 0.020 → 0.020 |
| null | U(-1, 1), 8 columns | 0.025 ± 0.011 | 0.025 ± 0.011 | 1 / 1 | 0.025 → 0.025 |
| null | N(0, 1), 8 columns | 0.015 ± 0.009 | 0.015 ± 0.009 | 0 / 0 | 0.015 → 0.015 |
| null | U(-1, 1), 200 columns | 0.025 ± 0.011 | 0.025 ± 0.011 | 0 / 0 | 0.025 → 0.025 |
| `x0*x1/x2` | U(0.5, 3), 8 columns | 0.000 | 0.000 | 0 / 0 | 0.000 → 0.000 |
| `x0*x1*x2` | U(-1, 1), 20 columns | 0.020 ± 0.010 | 0.025 ± 0.011 | 3 / 2 | 0.005 → 0.005 |

All with 300 rows, 150 of them for the search, and noise of half the law's
standard deviation on the planted designs, where a false feature is one that
uses none of the law's columns. The wide null design has more columns than
search rows. On the null designs the share of fits with any screened feature
was the same with and without the option, and in at most one replicate per
design did only one setting find something. On the planted centred design the
option screened 10.5 features on average against 3.3, because it found the
law and the screen also passes expressions built on it; one more fit in 200
held a false feature, from three discordant replicates against two.

## `residual_slots` (0.8.0)

`residual_slots`, an opt-in search option of 0.8.0, adds the terms of a
greedy forward path through the search's candidates to what the search
returns. `null.py` ran the same six designs and 200 paired replicates at
0.8.0's default and with `residual_slots=10` (commit `e1a2839`; both with
`magnitude_slots` on, as it is by default; `results/residual_slots/`):

| design | any false feature, default | with the option | discordant (option only / default only) | FDR, default → option |
| :-- | --: | --: | :-- | :-- |
| null, U(0.5, 3), 8 columns | 0.020 ± 0.010 | 0.020 ± 0.010 | 0 / 0 | 0.020 → 0.020 |
| null, U(-1, 1), 8 columns | 0.020 ± 0.010 | 0.020 ± 0.010 | 0 / 0 | 0.020 → 0.020 |
| null, N(0, 1), 8 columns | 0.015 ± 0.009 | 0.015 ± 0.009 | 0 / 0 | 0.015 → 0.015 |
| null, U(-1, 1), 200 columns | 0.020 ± 0.010 | 0.015 ± 0.009 | 0 / 1 | 0.020 → 0.015 |
| `x0*x1/x2`, U(0.5, 3), 8 columns | 0.000 | 0.020 ± 0.010 | 4 / 0 | 0.0000 → 0.0004 |
| `x0*x1*x2`, U(-1, 1), 20 columns | 0.020 ± 0.010 | 0.040 ± 0.014 | 4 / 0 | 0.0047 → 0.0055 |

On the null designs the option changed one fit in 800. On each planted
design it added one false feature in four fits, each of which also screened
the law, beside 50 and 15 screened features on average. The default here is
0.8.0's, with the fitted constants and the denominator guard that came after
the `magnitude_slots` study above, so on the same seeds it differs from that
study's rows with the option on three designs (0.020 against 0.025).

## Scope

Three laws, two products and a product beside a visible term, at depth 2
with the default operators, on independent inputs, for regression. A law's parts have exactly zero
correlation with it here only because the inputs are centred and symmetric;
on real data a part's correlation is rarely exactly zero, and the losses
measured here mark the end of a range rather than a typical case. The
ceiling-bound cells are few: p = 300 for every law and p = 1,000 for the pair
law. Nothing here measures laws deeper than two levels, interactions among
more than three inputs or classification.

## Files

`results/default/` holds the default settings' records: `records.csv.gz`, one
row per fit; `summary.json`, the provenance and every aggregate; `tables.md`,
the tables for every cell; `stress.log`, the run's output.
`results/magnitude_slots/` holds the same for `magnitude_slots=10`, made in
three invocations that `summary.json` lists with their commits, with
`compare.md`, the paired comparison with the default cell by cell, and
`timing.md`, the side-by-side times. `results/null/` holds the null study's
records, one file per design, and its table, and `results/residual_slots/`
the same for `residual_slots=10`; `results/suites/` the suites'
outputs for the default, for the option with the row cap and for the first
run without it, with `suites.md`. Every table can be rebuilt from the
records: `stress.py --tabulate` and `--compare`, `null.py --tabulate` and
`suites.py`.
