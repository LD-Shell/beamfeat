# Statistical guarantees

This page states, for each selection procedure, exactly what is guaranteed,
under which assumptions, and what was measured. Every number here is
reproduced by the committed test suite (`tests/test_selection.py`,
`tests/test_estimators.py`) or benchmark scripts. Calibration figures come
from 100 trials and are quoted with their standard error; read them at that
precision rather than at the last digit.

## What "FDR control at level q" means

The false discovery rate is the **expectation**, over repetitions of the
experiment, of the false discovery proportion (FDP) among selected features.
A single run with FDP above q is consistent with control; a mean over trials
above q is not. All documentation and tests in beamfeat use this definition.

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

**Null semantics.** Marginal independence. A near-duplicate of a genuine
signal is a *true* discovery under this null; de-duplication is performed by
the search's redundancy threshold, not the error-control procedure.

**Multiplicity.** The estimators (`BeamFeatRegressor`,
`BeamFeatClassifier`, `BeamFeatTransformer`) default to Benjamini–Yekutieli,
valid under arbitrary dependence; `PermutationSelector` used directly defaults
to Benjamini–Hochberg, valid under positive regression dependence (PRDS).
Engineered candidates share parents and are mutually correlated by
construction, so PRDS cannot be taken for granted and the estimators take the
conservative option. Over 100 pure-noise pipelines BH returned features in 6
trials against BY's 1 (FDP 0.06 and 0.01,
`benchmarks/selector_calibration.py`), both under the nominal 0.10. The
difference is not significant at this trial count: both corrections saw the
same 100 datasets, the one trial on which BY selected is among BH's six, and
the exact paired test gives p = 0.06. Read it as a consistency check rather
than a demonstrated BH failure. A selector *instance* passed to an estimator
keeps its own settings, so `selector=PermutationSelector()` runs BH at the
instance's `target_fdr`; the estimator warns when its `selection_correction`
or `target_fdr` disagrees with the instance's.

**Measured calibration** (Gaussian designs, 300 rows, 25 candidates, 5 signals,
100 trials). BH controls the FDR at `q·m0/m`, here `q·20/25`, and realised
0.046 ± 0.008, 0.084 ± 0.012 and 0.161 ± 0.016 at nominal 0.05/0.10/0.20
against ceilings of 0.040/0.080/0.160. BY is stricter by a harmonic factor and
realised 0.013 ± 0.005, 0.018 ± 0.005 and 0.046 ± 0.008. Power 1.00 throughout.
`benchmarks/selector_calibration.py` derives each bound from the design and
fails if a bound falls below a realised rate's whole 95% interval, so these figures cannot drift from the
behaviour.

**What the marginal null cannot see.** A feature can be jointly essential
yet nearly marginally independent of the target — a centred quadratic
`(c − 0.5)²` contributes almost no marginal correlation until its linear
complement is fitted. Marginal screening correctly excludes such features
under its null, at a measurable predictive cost on targets built from them
(Friedman #1, over six draws: the marginal null never admits `c`, giving a
screening-admissible ceiling of 0.874 ± 0.006 against a representation oracle
of 0.960 ± 0.003, with the pipeline reaching 0.780 ± 0.013. Around 0.086 is
lost to screening and 0.094 to the search. `c²` is admitted on some draws and
not others, since `(c − 0.5)²` expands to carry a little marginal signal
through it.)
Joint selection recovers
such features but certifies nothing; that trade is the design.

## Knockoff selector

**Fixed-X** (Barber & Candès, 2015), used when `n > 2p`: no distributional
assumption on the features — deterministic engineered columns are fine. The
guarantee requires the linear model with Gaussian, homoskedastic noise. The
construction's exchangeability identities `X̃'X̃ = X'X` and
`X'X̃ = X'X − sI` are verified to numerical precision by the tests, also
after the centring the lasso statistic applies (the knockoffs are built
orthogonal to the intercept as well, which takes one row more than `2p`). On
near-singular designs validity holds but `s → 0` and power degrades toward
zero; a runtime warning says so.

**Model-X Gaussian** (Candès et al., 2018), used only when `n ≤ 2p`: assumes
jointly Gaussian features, which engineered features violate; a warning is
recorded, and the permutation selector is recommended in this regime.

The estimators surface every selector diagnostic — assumption strain,
unsatisfiable configurations — through the `warnings` module, so an empty
knockoff result reads as "could not have selected at this configuration"
rather than "no evidence"; when knockoff+ selects nothing but `offset=0`
would have, the warning says so and names the trade-off.

**Offsets.** `offset=1` (knockoff+) carries the finite-sample FDR guarantee
and realised 0.159 at nominal 0.20 over 100 trials; it is unsatisfiable with
fewer than `1/q` features, and a warning fires on narrow designs. `offset=0`
controls only a modified FDR and realised 0.249, above `offset=1` as expected.
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

## Parsimony within the screened set

Three things can be wanted of the fitted equation: that it be short, that the
guarantee cover the terms actually printed, and that it cost no selection
rows. Any two are available together, and the choice is one parameter.

**`parsimony="forward"` (the default)** applies greedy forward selection
*within* the screened set and fits the compact subset. It is short and costs
no rows. What it gives up is the second property: the subset is chosen on the
rows selection already used, so the q-level guarantee certifies the set it was
drawn from and not the terms that survived. `equation()` says so on its own
output whenever terms were dropped —

```text
y = 0.9999*(x0 * x1) - 0.003943   [1 of 25 certified terms; parsimony=None prints all 25]
```

— and `fdr_scope_` records the same thing for code, which never sees that
string: `"screened set"` here, `"printed equation"` when the guarantee covers
the returned features themselves. `fdr_controlled_` says whether there *is* a
guarantee; `fdr_scope_` says what it is over. The full screened set with exact
p- and q-values per candidate stays auditable in `selection_report_`. `fdp_inflation_` reports |S|/|S'|, the factor by which
pruning can inflate the realised false discovery proportion: the denominator
shrinks faster than the numerator, so the count of false selections cannot
rise but the proportion can.

**`parsimony=None`** prints the screened set entire. The guarantee covers what
is printed — the equation *is* the certified object, not a selection from it —
and it still costs no rows. The price is length: on wide data the equation runs
to dozens of terms. `fdp_inflation_` is 1.0, because the printed set and the
screened set are the same set.

**`parsimony_holdout`** buys both, and pays in rows. It splits the selection
rows again: screening and parsimony use the first part, and the resulting
subset — fixed at that point, and so an ordinary fixed-candidate set — is
re-tested on the part held back. Terms failing the re-test are dropped, and
because the guarantee then does cover the printed terms, `equation()` prints
no subset note. The cost is real: comfortable above a few hundred rows,
unaffordable below roughly a hundred, where `parsimony=None` is the better
choice. If a compact certified equation cannot be produced, the whole screened
set is returned rather than a pruned subset — the pruned subset is the one
thing a caller asking for certification did not want. The two-stage procedure
has not been FDR-calibrated; it is offered as an option, not a measured result.

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

Compactness costs about a thousandth of held-out R² on average and at most
0.019 on a single fit, and buys an equation roughly twenty-five times shorter.
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
screening was forced to admit false candidates (Gaussian design, nominal
q = 0.5, 300 trials), the screened set realised FDP 0.397 and the greedily
pruned subset realised 0.000, with the proportion rising in none of the 300
trials.

## Association is not generalisation

FDR screening certifies that kept features are genuinely associated with the
target. It does not certify that the linear model assembled from them
generalises: a true discovery can carry extreme leverage — a reciprocal or
logarithm evaluated near its pole — that destabilises the least-squares fit
at points close to the singularity. The estimators keep these claims
separate with a post-fit check on the selection holdout: a negative held-out
R² (regression) or held-out accuracy below the majority-class rate
(classification) raises a visible `DegenerateFitWarning` naming the gap.
Rank-based scoring deserves particular attention here, since rank
transformation hides exactly the magnitudes that cause the problem.

## Honest failure

When selection passes nothing, the default (`on_no_discoveries="empty"`)
keeps no constructed features: the model degrades to an intercept-only fit
(training mean, or class prior for classification) and a visible
`NoDiscoveriesWarning` is raised. `"fallback"` keeps the unfiltered search
output with `fdr_controlled_ = False`; `"raise"` raises. In every mode, check
`fdr_controlled_` before treating fitted features as statistically vetted.
