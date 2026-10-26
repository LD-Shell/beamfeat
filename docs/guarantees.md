---
description: "What beamfeat's false discovery rate guarantee covers, the assumptions behind each selection procedure, and what was measured."
---

# Statistical guarantees

This page states, for each selection procedure, exactly what is guaranteed,
under which assumptions, and what was measured. Every number here is
reproduced by the committed test suite (`tests/test_selection.py`,
`tests/test_estimators.py`) or benchmark scripts. Calibration figures come
from 100 trials and are quoted with their standard error; read them at that
precision rather than at the last digit.

## What "FDR control at level q" means

Each candidate `j` carries its own null hypothesis H₀ⱼ. For the default
screen, H₀ⱼ is that candidate `j` is independent of the target. A selection
procedure rejects some of these nulls, and the candidates whose nulls it
rejects are its discoveries: the selected features, which beamfeat also calls
the screened or certified set. Rejecting a true null is a false discovery, a
Type I error. With R rejections of which V are false, the false discovery
proportion is FDP = V / max(R, 1), and the false discovery rate is its
**expectation** over repetitions of the experiment, FDR = E[FDP]. A single
run with FDP above q is consistent with control; a mean over trials above q
is not. All documentation and tests in beamfeat use this definition.

Under the global null, where every H₀ⱼ is true, V = R, so the FDR is the
probability of rejecting any null at all. Controlling it at q then means that
at most a fraction q of pure-noise fits select anything.

## Permutation selector (default)

**Statistic.** |Pearson correlation| (regression) or eta-squared
(classification) between each candidate and the target — a fixed function of
the data, which is the condition for a permutation test to be valid.
Statistics that re-tune themselves on the observed target (e.g. a
cross-validated lasso penalty) violate exchangeability unless the tuning is
repeated inside every permutation; beamfeat deliberately avoids them.

**P-values.** Add-one estimator `(1 + #exceedances)/(B + 1)` (Phipson &
Smyth, 2010): valid in finite samples, and never zero. The floor `1/(B+1)` interacts with the
multiplicity correction — Benjamini–Hochberg needs the leading feature to
reach `q/m`, Benjamini–Yekutieli the smaller `q/(m c(m))` — so B is
auto-raised to twice the bound belonging to the *configured* correction
(satisfiability plus headroom for one null exceedance), capped at
`max_permutations`. The requirement binds hardest when few candidates reach
the floor together: several tied signals relax the threshold by their count,
a lone one carries it alone. If the pool is large enough that even the cap
cannot meet it, the selector says so in `warnings_raised` and an empty
selection should be read as an exhausted budget rather than an absent signal.

**Null semantics.** Marginal independence: H₀ⱼ is that candidate `j` is
independent of the target. A near-duplicate of a genuine signal is not
independent of the target, so rejecting its null is a *true* discovery;
de-duplication is performed by the search's redundancy threshold, not the
error-control procedure.

**Multiplicity.** The estimators (`BeamFeatRegressor`,
`BeamFeatClassifier`, `BeamFeatTransformer`) default to Benjamini–Yekutieli,
valid under arbitrary dependence; `PermutationSelector` used directly defaults
to Benjamini–Hochberg, valid under positive regression dependence (PRDS).
Engineered candidates share parents and are mutually correlated by
construction, so PRDS cannot be taken for granted and the estimators take the
conservative option. Over 100 pure-noise pipelines BH returned features in 7
trials against BY's 1 (FDP 0.07 and 0.01,
`benchmarks/selector_calibration.py`), both under the nominal 0.10. Both
corrections saw the same 100 datasets, and the one trial on which BY selected
is among BH's seven, so the exact paired test rejects equal rates (p = 0.03):
under the global null BY selects less often, as a stricter correction
should. Neither rate exceeds the nominal level, so this is no evidence that BH
loses control here. A selector *instance* passed to an estimator
keeps its own settings, so `selector=PermutationSelector()` runs BH at the
instance's `target_fdr`; the estimator warns when its `selection_correction`
or `target_fdr` disagrees with the instance's.

**Measured calibration** (Gaussian designs, 300 rows, 25 candidates, 5 signals,
100 trials). BH controls the FDR at `q·m0/m`, here `q·20/25`, and realised
0.046 ± 0.008, 0.084 ± 0.012 and 0.161 ± 0.016 at nominal 0.05/0.10/0.20
against ceilings of 0.040/0.080/0.160. BY is stricter by a harmonic factor and
realised 0.013 ± 0.005, 0.018 ± 0.005 and 0.046 ± 0.008. Power 1.00 throughout.
`benchmarks/selector_calibration.py` derives each bound from the design and
stops with an error if a bound falls below a realised rate's whole 95%
interval, so these figures cannot drift from the behaviour.

**What the marginal null cannot see.** A feature can be jointly essential
yet nearly marginally independent of the target — a centred quadratic
`(c − 0.5)²` contributes almost no marginal correlation until its linear
complement is fitted. The marginal screen has almost no power to reject the
null for such a feature, and where the feature is exactly marginally
independent of the target, failing to reject is correct under that null. The
cost is measurable on targets built from them (Friedman #1, over six draws:
the screen never rejects the null for `c`, giving a screening-admissible
ceiling of 0.874 ± 0.006 against a representation oracle of 0.960 ± 0.003,
with the pipeline reaching 0.780 ± 0.013. Around 0.086 is lost to screening
and 0.094 to the search. The null for `c²` is rejected on some draws and not
others, since `(c − 0.5)²` expands to carry a little marginal signal through
it.)
Joint selection recovers
such features but certifies nothing; that trade is the design.

## Knockoff selector

**Fixed-X** (Barber & Candès, 2015), used when `n > 2p`: no distributional
assumption on the features — deterministic engineered columns are fine. The
guarantee requires the linear model with Gaussian, homoskedastic noise. The
construction's exchangeability identities `X̃'X̃ = X'X` and
`X'X̃ = X'X − S` are verified to numerical precision by the tests, also
after the centring the lasso statistic applies (the knockoffs are built
orthogonal to the intercept as well, which takes one row more than `2p`). On
a near-singular design (smallest covariance eigenvalue below 1e-6) valid
knockoffs of the dependent columns would equal the originals, and the lasso's
tie-breaking would then favour the originals; constant and linearly dependent
columns are therefore left out of the construction, get `W = 0`, and are
named in a runtime warning. Routing and the `n > 2p` requirement count the
retained columns.

**How far knockoffs are from the originals.** `S = diag(s)`, and the
correlation of column `j` with its knockoff is `1 − s_j`. The default,
`s_method="equicorrelated"`, uses one value for every column,
`min(1, 2λ_min)`, so one tight group of near copies, common among
constructed candidates, makes every knockoff nearly equal its original.
`s_method="ci"` (Liu & Rigollet, 2019) sets `s_j = c/(Σ⁻¹)_jj`, the share
of column `j` the other columns leave unexplained, times the largest
`c ≤ 1` that keeps `S ≤ 2Σ`; the guarantee is the same. In
`benchmarks/knockoff_s_method.py` (nominal 0.1, 50 trials) on 76
constructed columns, six transforms per input and the pairwise products,
knockoff+ power rose from 0.000 to 0.217 (realised FDR 0.000 and 0.038),
and `offset=0` power from 0.073 to 0.596 (0.096 and 0.112). On Gaussian
designs with correlated pairs or shared factors the two were within a few
points of each other.

**Statistic.** The fixed-X guarantee holds for a statistic that sees the
data only through `[X X̃]'[X X̃]` and `[X X̃]'y`. On the fixed-X path for
regression beamfeat uses the lasso-path signed maximum,
`W_j = max(Z_j, Z̃_j)·sign(Z_j − Z̃_j)`, where `Z_j` is the penalty at which
column `j` enters the path, computed from those inner products alone; a
test checks that rotating the rows, which preserves them, leaves `W`
unchanged.
On Gaussian linear models at q = 0.2 (200 trials each), knockoff+ with this
statistic realised 0.180, 0.172 and 0.135, under the level, with power 0.96,
0.98 and 0.48. A difference of cross-validated lasso coefficients, whose
folds look at individual rows and which the guarantee therefore does not
strictly cover, had power 0.97, 0.99 and 0.82: the valid statistic costs
power when columns are correlated (0.7 between neighbours in the third
design). Model-X, which admits any statistic that swaps sign with the pair,
keeps the cross-validated one.

**Model-X Gaussian** (Candès et al., 2018), used only when `n ≤ 2p`: assumes
jointly Gaussian features, which engineered features violate; a warning is
recorded, and the permutation selector is recommended in this regime.

The estimators surface every selector diagnostic — assumption strain,
unsatisfiable configurations — through the `warnings` module, so an empty
knockoff result reads as "could not have selected at this configuration"
rather than "no evidence"; when knockoff+ selects nothing but `offset=0`
would have, the warning says so and names the trade-off.

**Offsets.** `offset=1` (knockoff+) carries the finite-sample FDR guarantee
and realised 0.170 at nominal 0.20 over 100 trials; it is unsatisfiable with
fewer than `1/q` features, and a warning fires on narrow designs. `offset=0`
controls only a modified FDR and realised 0.260, above `offset=1` as expected.
Both move across numeric stacks, since knockoff construction depends on matrix
decompositions; treat the ordering and the nominal bound as the claim, not the
second decimal. `benchmarks/selector_calibration.py` regenerates both, and the
committed tests pin the ordering rather than the decimals.

## Selective inference: why the estimators split their data

The search retains candidates *because* they correlate with the target in
its sample. P-values computed on that same sample are therefore
optimistically biased, and the nominal FDR is not guaranteed for them. By
default the estimators search on one half of the training rows and run
selection on the other (`selection_holdout=0.5`), restoring the
fixed-candidate-set premise. Measured end-to-end over
200 replicates at nominal 0.10 (500 rows, 6 input columns of which 2 carry
signal through their product, noise at 5% of the signal's standard
deviation, `max_depth=2`, `beam_width=25`): not one false discovery, a 95%
upper bound of 0.015 on the true rate, power 1.000, no fallbacks; 60
global-null replicates selected nothing. That is a far stronger signal than
the selector-level design above — the generating feature correlates with the
target at 0.999 — so read it as an end-to-end check that the split does its
job, not as a power measurement at the margin.
`benchmarks/calibration_study.py` regenerates it. If the holdout is
disabled, the estimator refuses to claim the guarantee
(`fdr_controlled_ = False`).

Constants fitted inside expressions (`fitted_ops`) are part of the search.
Each is chosen inside the search, on the search rows, before selection sees
the candidates, and the selection rows receive every expression with its
constant fixed, as they receive its structure. The fixed-candidate-set
premise therefore holds as before and the guarantee is unchanged. A fitted
constant is one more way the search adapts to its sample, so with
`selection_holdout=None`, where search and screen share rows, it adds to
the optimism for which the estimator already declines the claim. On
pure-noise targets (50 seeds, 200 rows, 5 columns, defaults otherwise) the
fraction of fits with any discovery was the same with all four fitted
operators as without: 0/50 for regression, 2/50 for classification. The
four are on by default, and with them the end-to-end study above gives the
same result as without them (no false discovery in 200 replicates, power
1.000, nothing selected in 60 global-null replicates).

`magnitude_slots` changes which expressions the search keeps, not how they
are tested. Its magnitude score is computed inside the search, on the search
rows, as the score is; the expressions it keeps are built on and ranked like
any other; and the selection rows receive the search's output as a fixed
set of expressions, chosen without them. The premise of the screen holds as
before and its guarantee is unchanged. Measured on 200 paired replicates per
design at q = 0.1 (`benchmarks/search_stress/null.py`, the screened set),
the share of fits with at least one false discovery was the same with the
option as without on four null designs, 0.015 to 0.025, among them centred and
Gaussian inputs and 200 columns on 150 search rows; on a planted
`x0*x1*x2` among centred distractors it was 0.025 against 0.020, one fit in
200, with a false discovery rate of 0.005 either way. With the slots on, as
they are by default for regression, the end-to-end study gives the result it
gave without them: no false discovery in 200 replicates, power 1.000,
nothing selected in 60 global-null replicates. At the defaults, with the
slots and the denominator guard, the pure-noise pipeline stress gives BH
7/100 and BY 1/100.

`residual_slots` changes only which expressions the search returns, in the
same way: its forward path runs inside the search, on the search rows, and
the terms it adds reach the screen as fixed expressions like the rest. The
guarantee is unchanged. On the six designs of `null.py` (four null, two
planted), 200 paired replicates each, the share of null fits with any false
discovery was 0.015 to 0.020 with the option and without it; in one
replicate of the wide design the default screened three false features and
the option none. On the planted designs the option added one false feature
in 4 fits of 200, each a fit that also screened the law: the share of fits
with any went from 0 to 0.020 on `x0*x1/x2` and from 0.020 to 0.040 on
`x0*x1*x2`, and the false discovery rate from 0.0000 to 0.0004 and from
0.0047 to 0.0055 (`benchmarks/search_stress/results/residual_slots/`).

## Columns inside a term (`input_test`)

The screen's null is that a term is independent of the target, so
`(x0*x1)/(x1 + x8)` is a true discovery when `y = x0*x1`: `x0*x1` carries it,
however unrelated `x8` is. A reader of the equation still takes `x8` to
matter. `input_test=True` tests each column inside each candidate that reads
more than one: the column's rows are permuted, wherever it occurs in the
expression, and the screen's statistic is recomputed. The null for a column
is that it is independent of the target and of the candidate's other columns
together; under it, with independent rows from one distribution, permuting
the column leaves the distribution of the data unchanged, so the p-values
are exact. The screen's step-up correction runs across every
(candidate, column) pair of every candidate, so which pairs are tested does
not depend on the screen's outcome. A screened term is dropped before
parsimony when the test fails to reject the null for any of its columns;
`input_report_` lists every pair, with `null_rejected` marking the pairs
whose null the test rejected.

What it does not cover: a junk column correlated with a real one is not null
in this sense, so its null can be rejected and the term kept. Testing it
given the other columns needs a model
of the column given the others (knockoffs or the conditional randomisation
test), which this test does not assume. The screen's guarantee is unchanged
and still covers the screened set.

Measured on 20 seeds of `y = x0*x1` plus 25% noise among 10 junk columns
(500 training and 500 test rows), against the default: printed equations with
a junk column 14 → 1, printed terms with one 23/45 → 1/28, held-out R² 0.9413
→ 0.9416. With the junk `x8` correlated 0.82 with `x1`, 10 → 3. On pure-noise
targets the screen rejects no null with the option or without it, and the
physics panel is unchanged. On the release comparison's 12 real datasets
(`benchmarks/three_settings/`) held-out score fell on 7, by 0.06 at most, and
fits took 3.6 times as long.

## Parsimony within the screened set

Three things can be wanted of the fitted equation: that it be short, that the
guarantee cover the terms actually printed, and that it cost no selection
rows. Any two are available together, and the choice is one parameter.
[Choosing a setting](settings.md) sets the default beside `parsimony_alpha`
and `input_test` on the same data.

**`parsimony="forward"` (the default)** applies greedy forward selection
*within* the screened set and fits the compact subset. It is short and costs
no rows. What it gives up is the second property: the subset is chosen on the
rows selection already used, so the q-level guarantee certifies the set it was
drawn from and not the terms the forward step retained. `equation()` says so
on its own output whenever terms were dropped —

```text
y = 0.9999*(x0 * x1) - 0.003943   [1 of 25 certified terms; parsimony=None prints all 25]
```

— and `fdr_scope_` records the same thing for code, which never sees that
string: `"screened set"` here, `"printed equation"` when the guarantee covers
the returned features themselves. `fdr_controlled_` says whether there *is* a
guarantee; `fdr_scope_` says what it is over. The full screened set with exact
p- and q-values per candidate stays auditable in `selection_report_`. `fdp_inflation_` reports |S|/|S'|, the factor by which
pruning can inflate the realised false discovery proportion: the denominator
shrinks faster than the numerator, so the count of false discoveries cannot
rise but the proportion can.

**`parsimony=None`** prints the screened set entire. The guarantee covers what
is printed — the equation *is* the certified object, not a selection from it —
and it still costs no rows. The price is length: on wide data the equation runs
to dozens of terms. `fdp_inflation_` is 1.0, because the printed set and the
screened set are the same set.

**`parsimony_holdout`** buys both, and pays in rows. It splits the selection
rows again: screening and parsimony use the first part, and the resulting
subset — fixed at that point, and so an ordinary fixed-candidate set — is
re-tested on the part held back. Terms whose null the re-test fails to
reject are dropped, and because the guarantee then does cover the printed
terms, `equation()` prints
no subset note. The cost is real: comfortable above a few hundred rows,
unaffordable below roughly a hundred, where `parsimony=None` is the better
choice. If a compact certified equation cannot be produced, the whole screened
set is returned rather than a pruned subset — the pruned subset is the one
thing a caller asking for certification did not want.

**Measured.** `benchmarks/parsimony_holdout/` calibrates the procedure over
48,000 fits: a global null, and `y = a*b/c` plus noise among 5 or 20
distractors, with and without a column correlated 0.9 with `a`, at
signal-to-noise ratios 1 and 3 and with 300 and 1,000 rows, 200 replicates
per cell, at q = 0.1 and 0.2, beside the default and `parsimony=None` on the
same data. The figures below are means over replicates. Their Monte Carlo
standard errors are in the study's report, and are at most 0.012 for a
design pooled over its four cells.

- *In the marginal sense, the guarantee's,* the printed equation's realised
  FDR stayed far below q under every setting. On the global null it was 0.009
  to 0.018 at q = 0.1 and 0.021 to 0.029 at q = 0.2 (a term printed in 7 to
  14 and 17 to 23 fits of 800), and on the planted and proxy designs no
  setting printed a term built only from distractors in 3,200 fits per
  level. The re-test added nothing measurable here. On null data it failed
  to reject the null for the screen's rare false discoveries, in 20 of 22
  fits at q = 0.1, and the fallback to the screened set then printed them
  anyway, so the printed equation's error is the screen's own.
- *In the conditional sense*, counting a printed term that uses any column
  outside the generating formula, such as a rider `(a*b/c) * x7`, no setting
  controls the rate and the marginal re-test does not lower it. It was 0.67
  to 0.77 under every setting at a signal-to-noise ratio of 1, where the
  forward step retains 18 terms on average. At a ratio of 3 it was 0.42 to
  0.44 at the default and 0.53 to 0.62 with the re-test: on its smaller
  share of rows the forward step retains more riders, and each rider is
  associated with the target, so the re-test rejects its null.
- Power to print the generating term (0.85 to 0.93 at a ratio of 1, 0.99 to
  1.00 at 3) and held-out R² (within 0.0033 of the default, paired) hardly
  moved under any setting.

**`certify_test="conditional"`** re-tests each frozen term given the others
rather than on its own, with a Freedman–Lane permutation test (Freedman &
Lane, 1983): the target and the term are residualised on the other terms and
an intercept, the target's residuals are permuted, the statistic is the
partial correlation, and the screen's step-up correction is applied at the
screen's level. The null for a term is that its coefficient is zero in the
linear model of the target on all the frozen terms: given the others, it
adds nothing linear. That null is relative to the frozen set, so a term
standing in for a true one the set leaves out is not null under it, and
rejecting its null is not a false rejection. The p-values are approximate.
For a single term the test is the marginal permutation test and exact. With
others present its validity rests on the errors of a linear model of the
target on the other terms being exchangeable (independent and identically
distributed, so of constant variance), and even then the residuals it
permutes are estimated and only nearly exchangeable, so the p-values are
close to exact rather than exact (Anderson & Robinson, 2001; Winkler et al.,
2014). The test is for regression only, and the step-up correction inherits
its own condition: Benjamini–Yekutieli holds under any dependence among the
p-values, Benjamini–Hochberg needs positive dependence.

Measured in the same study, at a signal-to-noise ratio of 3 it brought the
conditional rate to 0.18 to 0.22, paired −0.21 to −0.26 against the default.
In the fits where it rejected at least one null it printed 1.1 to 1.2 terms,
with a conditional rate of 0.026 to 0.040 at q = 0.1 and 0.052 to 0.073 at
q = 0.2, and power 0.98 to 0.99. Its weakness is the fallback. When the
frozen subset is long and its terms nearly reproduce one another, no term
adds anything given the rest, the test rejects no null, and the screened set
is returned, which carries the screen's marginal guarantee only. That
happened in 87 to 92% of fits at a ratio of 1 and in 18 to 26% at 3 (37 to
52% with 300 rows, under 1% with 1,000). `certification_result_.n_selected`
is 0 on such a fit. With the chance check below, the fallback all but
disappears: under 1% of fits at a ratio of 1. The conditional re-test after
the forward step without `parsimony_alpha` raises a `ValueError`. With
`parsimony=None`, which re-tests the whole screened set, it needs no
`parsimony_alpha`.

**The forward step's chance check, `parsimony_alpha`.** The riders above
are a property of the forward step's stopping rule. The best of the many
remaining terms adds more than `parsimony_tol` to R² by fitting noise alone:
on `n` rows its chance gain is about the largest of `m` χ²₁ draws over `n`,
times the share of variance left, so around 0.01 to 0.03 at signal to noise 1
on 500 to 150 selection rows, against a tolerance of 0.001. `parsimony_alpha` (off by default) also stops the step
at the first term whose gain fails to reject the null that it is no larger
than the best gain the remaining terms reach against noise. The null
distribution is that best gain for 200 wild-bootstrap draws of the current
residual, each row's residual times an independent standard normal, net of
the terms already added: the multiplier bootstrap of Fan & Zhou (2016) for
the best spurious fit. On columns that add nothing beyond the chosen one, five
of them correlated with it, it rejected in 4% of 400 trials at level 0.05 and
10% at 0.10. It shapes the returned subset, as `parsimony_tol` does, and
leaves the screen and its guarantee alone; it is a stopping rule, not a
guarantee on the subset. `benchmarks/parsimony_holdout/parsimony_alpha.py`
reran the designs above at q = 0.1, 50 replicates per cell, with
`parsimony_alpha=0.05`:

| design | setting | terms | FDR, conditional | power | held-out R² |
| :-- | :-- | --: | --: | --: | --: |
| planted, snr 1 | default | 18.4 | 0.703 | 0.890 | 0.482 |
| planted, snr 1 | `parsimony_alpha=0.05` | 1.1 | 0.048 | 0.875 | 0.489 |
| planted, snr 3 | default | 3.8 | 0.420 | 1.000 | 0.897 |
| planted, snr 3 | `parsimony_alpha=0.05` | 1.1 | 0.022 | 1.000 | 0.899 |
| proxy, snr 1 | default | 18.1 | 0.738 | 0.895 | 0.481 |
| proxy, snr 1 | `parsimony_alpha=0.05` | 1.1 | 0.080 | 0.865 | 0.489 |
| proxy, snr 3 | default | 4.0 | 0.429 | 1.000 | 0.897 |
| proxy, snr 3 | `parsimony_alpha=0.05` | 1.1 | 0.020 | 1.000 | 0.899 |

Held-out R² rose by 0.007 to 0.008 on average at signal to noise 1, paired,
and by 0.002 at 3, but 9% of the fits at 1 lost more than 0.01 and the worst
lost 0.10. Those were fits on 300 rows whose first term only approximates
`a*b/c` with a distractor inside, such as `(x1 + x10) * (x0 / x2)`: the check
stops after it, where the default's further terms made up part of the
difference. Under the global null nothing changed. Combined with
`parsimony_holdout=0.3, certify_test="conditional"`, the re-test receives a
short subset, so its fallback to the screened set went from 92% of fits at
signal to noise 1 and 20 to 24% at 3 to under 1%, and the conditional rate was
0.07 to 0.09 at 1 and 0.000 at 3; power at 1 fell to 0.82, from 0.88 to 0.90.

On real data the check costs accuracy. These designs have one strong term
and noise; real targets tend to carry many modest effects, each of which adds
little on its own, often less than the best of the remaining terms reaches
against noise, so the step stops after a few terms: one to five on 10 of
these 12 datasets. In
`benchmarks/three_settings/`, on the datasets and splits of
`benchmarks/release_comparison/` (5 splits each, 70/30), held-out score fell
by more than 0.005 on 9 of 12 real datasets, was within 0.005 on 2, and rose
on one:

| dataset | rows (train) | default | `parsimony_alpha=0.05` |
| :-- | --: | --: | --: |
| diabetes | 309 | 0.433 (6.6 terms) | 0.381 (1.4) |
| tips | 170 | 0.472 (8.6) | 0.433 (1.0) |
| penguins | 239 | 0.803 (7.8) | 0.768 (1.4) |
| riboflavin | 49 | 0.516 (18.6) | 0.489 (1.0) |
| breast_cancer (accuracy) | 398 | 0.971 (25.8) | 0.952 (2.8) |
| communities | 1,395 | 0.647 (11.6) | 0.629 (2.4) |
| mpg | 274 | 0.868 (9.4) | 0.858 (1.6) |
| superconduct | 3,500 | 0.747 (18.8) | 0.737 (12.6) |
| california_housing | 14,448 | 0.690 (21.2) | 0.681 (13.4) |
| diamonds | 980 | 0.852 (3.4) | 0.851 (1.0) |
| tecator | 168 | 0.992 (3.0) | 0.991 (2.2) |
| eyedata | 84 | 0.321 (30.6) | 0.330 (1.0) |

The one gain is on eyedata, where the default fits 31 terms to 84 training
rows. Two synthetic targets built from many terms behaved like the real ones
(friedman1 0.796 to 0.759, threshold_step 0.841 to 0.808). The check suits a search for a short law,
where the target is a few strong terms; for prediction, leave it off.

**The equation list.** `equations()` and `equations_` record the forward path
through the whole screened set, one equation per number of terms, at every
setting above. Each line is a subset of the screened set chosen on the rows
screening used, the same kind of selection as the default fitted equation, so
the guarantee covers none of them but the last, which is the screened set
itself, and, under `parsimony_holdout`, the re-tested fitted equation. An
equation picked from the list after reading it is not covered; the list says
so in its last line.

### What the default costs

The exchange above is stated rather than assumed elsewhere in this page, so it
is measured here. `benchmarks/parsimony_cost.py` re-runs the affected studies
at both settings and pairs them fit by fit; `benchmarks/PARSIMONY_COST.md`
records which studies depend on the setting and why. Over 308 paired fits
across four studies:

| study | fits | mean ΔR² (forward − None) | worst ΔR² | terms, None ÷ forward | `fdr_controlled_` disagreements |
|---|---|---|---|---|---|
| end-to-end calibration | 260 | −0.0001 | −0.0003 | 25.0× | 0 |
| known-formula + stress suites | 18 | −0.0008 | −0.0148 | 26.0× | 0 |
| Feynman physics panel | 24 | −0.0009 | −0.0111 | 30.6× | 0 |
| Friedman #1 decomposition | 6 | −0.0095 | −0.0192 | 3.6× | 0 |

Compactness costs about a thousandth of R² on average and at most 0.019 on a
single fit, and buys an equation roughly twenty-five times shorter. R² is held
out in every study but the end-to-end calibration, which fits and scores on
the same rows.
`fdr_controlled_` never disagreed between the two settings on any of the 308
fits, and the end-to-end calibration figures above — empirical FDR, its upper
bound, power, fallbacks, null selections — are identical at both settings to
every digit reported.

Pruning was also never observed to inflate the realised false discovery
proportion. That is not a theorem and should not be read as one: |S|/|S'| is
the only bound available, and it is weak. It is an empirical observation with
a mechanism behind it — greedy forward selection ranks by predictive
contribution, and a marginally null feature contributes nothing, so it tends
to be dropped first rather than kept. On a selector-level stress where
screening was made to reject many true nulls (Gaussian design, nominal
q = 0.5, 300 trials), the screened set realised FDP 0.397 and the greedily
pruned subset realised 0.000, with the proportion rising in none of the 300
trials.

## Association is not generalisation

FDR screening bounds the expected share of selected features that are not
associated with the target. It does not certify that the linear model
assembled from them generalises: a true discovery can carry extreme leverage
— a reciprocal or logarithm evaluated near its pole — that destabilises the
least-squares fit at points close to the singularity. The estimators keep
these claims separate with a post-fit check on the selection holdout: a
negative held-out R² (regression) or held-out accuracy below the
majority-class rate (classification) raises a visible `DegenerateFitWarning`
naming the gap. Rank-based scoring deserves particular attention here, since
rank transformation hides exactly the magnitudes that cause the problem.

New rows can reach a pole the training rows stayed clear of. The search
rules out a quotient whose denominator comes within the pole margin of
zero on the search rows, and `predict` and `transform` raise
`ExtrapolationWarning` when a new row takes a feature more than one width of
its training range beyond it. Neither touches the screen or its guarantee;
[New rows beyond the training data](guide/data.md#new-rows-beyond-the-training-data)
describes both.

## When no null is rejected

When the screen rejects no null, the default (`on_no_discoveries="empty"`)
uses no constructed features: the model degrades to an intercept-only fit
(training mean, or class prior for classification), `fdr_controlled_` is
`False`, and a visible `NoDiscoveriesWarning` is raised. `"fallback"` uses
the unfiltered search output, also with `fdr_controlled_ = False`; `"raise"`
raises. In every mode, check `fdr_controlled_` before treating fitted
features as statistically vetted.
