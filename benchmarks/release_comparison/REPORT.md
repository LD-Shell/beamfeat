# Releases compared: 0.4.2, 0.5.0, 0.6.0, 0.6.1 and 0.7.0

The same benchmarks, data, splits and seeds, run through five releases of
beamfeat, each in its own environment with the same numpy, scipy and
scikit-learn. 0.4.2 is the release the TMLR paper describes. Each release is
measured at its defaults, and the options 0.6.0, 0.6.1 and 0.7.0 add or
change are measured beside them. About 23,000 fits, none of which raised.
`TABLES.md` has every table and `PROVENANCE.md` how the run was made and how
to repeat it.

Releases run here at their defaults over five splits, so the figures differ
from the paper's, which come from other settings. Run with the paper
scripts' own settings, 0.4.2 reproduces the archived Feynman panel and 19 of
the 25 `run_benchmarks.py` datasets exactly (`PROVENANCE.md`, Against the
paper's numbers).

## In short

- **Fits.** 0.5.0 gives the same fits as 0.4.2 on 1,415 of 1,430 items. The
  rest are near-ties, mostly on riboflavin (49 training rows, 4,088 columns).
  0.6.0 changes fits only through `fitted_ops`, now on by default: with
  `fitted_ops=()` it reproduces 0.5.0 on all 1,430. 0.6.1 gives the same fits
  as 0.6.0 on all 1,430. 0.7.0 changes fits in two ways: `magnitude_slots`,
  now 10 for regression by default, and a guard that rules out a quotient
  whose denominator comes within the pole margin of zero on the search rows.
  With `magnitude_slots=0` it gives 0.6.1's fits on 1,243 of 1,430 items; the
  others are fits whose search met such a quotient.
- **Accuracy.** Between releases at their defaults, held-out accuracy moves
  little. `fitted_ops` solves three more of the 60 Feynman-panel fits,
  slightly lowers the conditional false rate (riders), and gains on
  friedman1, superconduct and riboflavin; it loses a little on breast_cancer
  and threshold_step. 0.7.0's magnitude slots recover every search stress law
  (0.883 → 1.000 exact). Its denominator guard raised the held-out score on 3
  of 12 real datasets and lowered it on none, most on eyedata (0.110 →
  0.312), whose default had a split ruined by a quotient near its pole.
- **Speed and memory.** 0.5.0 fits 3.2 times faster than 0.4.2, with a fifth
  of its memory. 0.6.0 and 0.6.1 take 1.5 times 0.5.0's time and about 10%
  more memory, the cost of `fitted_ops`; 0.7.0 takes 1.25 times 0.6.1's time
  with about the same memory: 1.19 times for the magnitude slots and 1.05
  times for the denominator guard.
- **Fixed along the way.** 0.4.2 replaced a formula that fails on any
  prediction row with zeros on every row (fixed in 0.5.0); on one California
  split that cost it 0.045 of R². 0.6.0's `input_test` ran out of memory on
  large data (over 7 GB on California housing); 0.6.1 needs 0.55 GB, with the
  same fits.
- **Guarantee.** No design fit printed a term outside the generating columns'
  set (marginal FDR 0 everywhere), and under the global null every release
  printed a term in 2 of 200 fits, against q = 0.1.

## The releases at their defaults

| measure | 0.4.2 | 0.5.0 | 0.6.0 | 0.6.1 | 0.7.0 |
| :-- | --: | --: | --: | --: | --: |
| Calibration designs, snr 1: terms printed | 18.6 | 18.6 | 18.2 | 18.2 | 18.5 |
| &nbsp;&nbsp;conditional FDR (riders) | 0.736 | 0.736 | 0.720 | 0.720 | 0.722 |
| &nbsp;&nbsp;power (generating term printed) | 0.895 | 0.895 | 0.892 | 0.892 | 0.890 |
| &nbsp;&nbsp;held-out R² | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 |
| Calibration designs, snr 3: terms printed | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 |
| &nbsp;&nbsp;conditional FDR | 0.437 | 0.437 | 0.424 | 0.424 | 0.428 |
| Global null: fits printing any term | 2/200 | 2/200 | 2/200 | 2/200 | 2/200 |
| Search stress laws: exact | 0.883 | 0.883 | 0.883 | 0.883 | 1.000 |
| Feynman panel: solved / exact form (of 60) | 47 / 40 | 47 / 40 | 50 / 40 | 50 / 40 | 50 / 40 |
| Suite, 18 synthetic datasets: mean score | 0.9625 | 0.9625 | 0.9627 | 0.9627 | 0.9626 |
| Suite, 7 real datasets: mean score | 0.7246 | 0.7259 | 0.7263 | 0.7263 | 0.7270 |
| High-dimensional, 5 datasets: mean R² | 0.609 | 0.585 | 0.600 | 0.600 | 0.645 |
| Junk-column equations (of 40) | 21 | 21 | 23 | 23 | 27 |
| Fit time (× 0.5.0, geometric mean of 8 items) | 3.18 | 1.00 | 1.53 | 1.54 | 1.93 |
| Peak fit memory (× 0.5.0) | 5.27 | 1.00 | 1.08 | 1.10 | 1.07 |

The high-dimensional means differ mostly through riboflavin, whose five
splits swing by up to 0.35 between near-tied fits (0.526, 0.405, 0.506, 0.506
and 0.516 by release), and, for 0.7.0, through eyedata. 0.7.0's ratios to
0.5.0 are chained through 0.6.1, which its own speed pass shares with the
main one.

Held-out score by dataset (R², accuracy for breast_cancer; mean of 5 splits):

| dataset | 0.4.2 | 0.5.0 | 0.6.0 | 0.6.1 | 0.7.0 |
| :-- | --: | --: | --: | --: | --: |
| diabetes | 0.430 | 0.430 | 0.430 | 0.430 | 0.433 |
| breast_cancer | 0.979 | 0.979 | 0.974 | 0.974 | 0.971 |
| mpg | 0.864 | 0.864 | 0.868 | 0.868 | 0.868 |
| diamonds | 0.850 | 0.850 | 0.853 | 0.853 | 0.852 |
| penguins | 0.797 | 0.797 | 0.803 | 0.803 | 0.803 |
| tips | 0.462 | 0.462 | 0.461 | 0.461 | 0.472 |
| california_housing | 0.691 | 0.699 | 0.695 | 0.695 | 0.690 |
| communities | 0.644 | 0.646 | 0.648 | 0.648 | 0.648 |
| superconduct | 0.738 | 0.738 | 0.746 | 0.746 | 0.747 |
| tecator | 0.991 | 0.991 | 0.991 | 0.991 | 0.992 |
| eyedata (84 training rows) | 0.145 | 0.145 | 0.110 | 0.110 | 0.321 |
| riboflavin (49 training rows) | 0.526 | 0.405 | 0.506 | 0.506 | 0.516 |
| friedman1 | 0.786 | 0.786 | 0.798 | 0.798 | 0.796 |
| threshold_step | 0.849 | 0.849 | 0.841 | 0.841 | 0.841 |
| distractors_noise50 | 0.811 | 0.811 | 0.811 | 0.811 | 0.808 |

The other 15 synthetic suite datasets score the same in every release.
eyedata and riboflavin have standard errors of 0.04 to 0.30 over their
splits.

## Speed and memory

Each fit in a fresh process on one core, median of 3; memory is the peak
added by the fit. 0.4.2 was timed in a second pass beside 0.5.0, whose
control matched the main pass to within 0.09 s, and 0.7.0 in a third beside
0.6.1, which ran 7 to 12% slower throughout; 0.7.0's seconds below are scaled,
item by item, by the ratio of 0.6.1's seconds in the main pass to those in
its own.

| item | 0.4.2 (s) | 0.5.0 (s) | 0.6.0 (s) | 0.6.1 (s) | 0.7.0 (s) | 0.4.2 (MB) | 0.5.0 (MB) | 0.6.0 (MB) | 0.6.1 (MB) | 0.7.0 (MB) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| planted design, n = 1,000 | 0.99 | 0.28 | 0.61 | 0.62 | 0.77 | 260 | 50 | 52 | 52 | 53 |
| stress law, p = 100 | 3.92 | 0.75 | 1.00 | 1.01 | 1.21 | 315 | 57 | 66 | 66 | 61 |
| Feynman, coulomb | 0.59 | 0.24 | 0.45 | 0.45 | 0.52 | 102 | 41 | 49 | 50 | 45 |
| diabetes | 0.55 | 0.16 | 0.24 | 0.24 | 0.33 | 75 | 35 | 35 | 35 | 31 |
| california_housing | 3.46 | 1.78 | 2.84 | 2.83 | 3.08 | 2,571 | 88 | 99 | 99 | 101 |
| tecator | 2.94 | 1.26 | 1.47 | 1.47 | 2.19 | 120 | 44 | 44 | 49 | 52 |
| superconduct | 5.00 | 1.36 | 2.44 | 2.47 | 3.02 | 1,462 | 61 | 67 | 67 | 73 |
| riboflavin | 19.57 | 4.87 | 5.45 | 5.46 | 6.92 | 338 | 167 | 172 | 173 | 173 |

## The options

Each option against its own release's default. Real datasets: the 12 above
other than friedman1 and threshold_step; "better / same / worse" means the
mean held-out score moved by more than 0.005. Time and memory are geometric
means over the 8 items, where the option was timed.

| option | real datasets: better / same / worse | worst real change (R²) | mean terms on real data | riders at snr 1 (Δ conditional FDR) | time (× default) | memory (× default) |
| :-- | --: | :-- | --: | --: | --: | --: |
| `fitted_ops=()` (0.5.0's search, in 0.6.1) | 1 / 8 / 3 | riboflavin -0.101 | 13.3 → 13.9 | +0.016 | 0.66 | 0.93 |
| `magnitude_slots=10` (0.7.0's default) | 1 / 10 / 1 | riboflavin -0.011 | 13.8 → 13.8 | +0.004 | 1.19 | 1.07 |
| denominator guard (0.7.0, always on) | 3 / 9 / 0 | california_housing -0.005 | 13.3 → 13.8 | -0.002 | 1.05 | 0.91 |
| `input_test=True` (0.6.1) | 0 / 5 / 7 | riboflavin -0.065 | 13.3 → 10.5 | -0.541 | 5.2 | 1.6 |
| conditional re-test (0.6.0) | 1 / 4 / 6 | riboflavin -1.428 | 13.3 → 23.9 | -0.025 | 1.02 | 0.99 |
| conditional re-test with `parsimony_alpha` (0.7.0) | 1 / 2 / 8 | tips -0.046 | 13.8 → 6.9 | -0.642 | — | — |
| `selection_correction="bh"` (0.6.1) | 1 / 11 / 0 | none | 13.3 → 13.3 | 0.000 | 0.97 | 0.98 |
| `parsimony_alpha=0.05` (0.6.1) | 1 / 3 / 8 | riboflavin -0.200 | 13.3 → 3.4 | -0.657 | 1.00 | 0.99 |
| `parsimony_alpha` + `input_test` (0.6.1) | 1 / 1 / 10 | riboflavin -0.226 | 13.3 → 3.4 | -0.673 | 5.2 | 1.6 |

The two 0.7.0 rows take its changes one at a time: the magnitude slots are
0.7.0 against 0.7.0 with `magnitude_slots=0`, and the guard is 0.7.0 with
`magnitude_slots=0` against 0.6.1.

What they are for, as these results show it:

- **`magnitude_slots`**, on for regression since 0.7.0, finds products of
  inputs that do not correlate with the target, which a search without them
  ranks among noise: every `x0*x1*x2` stress law is recovered (0.40 and 0.55
  → 1.00 exact), and leaves the held-out score of 10 of 12 real datasets
  within 0.005 (riboflavin −0.011, eyedata +0.009). It adds a junk column to
  two more equations of 40 (27 against 25). `magnitude_slots=0` turns it
  off.
- **The denominator guard**, new in 0.7.0 and always on, rules out a quotient
  whose denominator comes within the pole margin of zero on the search rows,
  or crosses it. It changed the fit on 187 of 1,430 items, raised held-out
  score on 3 of 12 real datasets (eyedata 0.110 → 0.312, riboflavin 0.506 →
  0.527, tips 0.461 → 0.472) and lowered none by more than 0.005, at 5% of
  the fit time.
  On tecator's own split it removed predictions of up to 507% fat on the
  high-fat extrapolation set (`benchmarks/tecator_chemometrics/`).
- **`input_test`** removes columns that ride inside a term: equations with a
  junk column fell from 10 of 20 to 1 of 20, and riders at signal to noise 1
  from 0.70 to 0.06. It costs some held-out accuracy on real data, where
  correlated columns stand in for each other, and five times the fit time. On
  more than a few thousand rows, use 0.6.1 or later.
- **The conditional re-test** (`parsimony_holdout` with
  `certify_test="conditional"`) re-tests the forward step's terms on held-back
  rows. When it rejects no null it returns the whole screened set, up to 50
  terms, which after the forward step is the common case on noisy data: 46
  terms at signal to noise 1 on the calibration designs and 40 on diabetes
  and mpg in 0.7.0, and in 0.6.0 a held-out R² of -6.6 on one riboflavin
  split. With `parsimony_alpha`, the forward step stops at the first gain
  chance can reach, and the re-test then printed 1.0 to 1.5 terms on the
  designs with riders at 0.07 to 0.09 against 0.72, keeping power at 0.82
  (0.89 by default); on real data it shortens equations as `parsimony_alpha`
  does, and costs accuracy as it does. Since 0.7.0 the re-test after the
  forward step without `parsimony_alpha` is deprecated.
- **`selection_correction="bh"`** changed almost nothing here against the
  default Benjamini-Yekutieli correction, whose guarantee holds under any
  dependence between the tests.
- **`parsimony_alpha`** stops the equation at the first term whose gain fails
  to reject the null that chance alone reaches it. It printed 1.1 terms
  instead of 18 on the calibration designs, with riders down from 0.70 to
  0.05, but real targets carry many modest effects, so it stopped early there
  and lost accuracy on 8 of 12 (9 of 12 in 0.7.0,
  `benchmarks/three_settings/`). It suits a search for a short law, not
  prediction.
- **`KnockoffSelector(s_method="ci")`** helps when some columns are near-copies
  of others: power with knockoff+ rose from 0 to 0.22 on the constructed
  design of `benchmarks/knockoff_s_method.py`, and was within a few points of
  the default on Gaussian block and factor designs (`TABLES.md`, section 7).

The knockoff statistic changed in 0.6.0 to one the fixed-X guarantee covers.
On correlated designs that costs power against 0.4.2 and 0.5.0 (block
correlation 0.9: 0.84 → 0.43 with knockoff+; factor design 1.00 → 0.28).
The permutation selector, the default, is unaffected; 0.7.0 does not change
the knockoff selector, which was not run again.
