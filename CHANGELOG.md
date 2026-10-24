# Changelog

Notable changes to `beamfeat`. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Towards 0.7.0: `beamfeat.__version__` is `0.7.0.dev0` until the release.

## [0.6.1] - 2026-10-24

Default fits are those of 0.6.0, feature for feature. This release fixes the
memory of `input_test`, adds two options that are off by default
(`parsimony_alpha` and `KnockoffSelector(s_method="ci")`), and adds a
comparison of the releases since the paper's.

### Added

- `s_method` on `KnockoffSelector`: `"equicorrelated"` (the default, as
  before) or `"ci"`, the conditional-independence knockoffs of Liu &
  Rigollet (2019), which give each column its own `s_j = c/(Σ⁻¹)_jj`
  instead of one value set by the smallest eigenvalue. The guarantee is the
  same. In `benchmarks/knockoff_s_method.py` (fixed-X, nominal FDR 0.1, 50
  trials) on 76 constructed columns, power rose from 0.000 to 0.217 with
  knockoff+ (realised FDR 0.000 and 0.038) and from 0.073 to 0.596 with
  `offset=0` (0.096 and 0.112); on Gaussian block and factor designs the two
  were within a few points.
- `parsimony_alpha` on `BeamFeatTransformer`, `BeamFeatRegressor` and
  `BeamFeatClassifier`, a chance check on the forward step (default `None`,
  which leaves every fit unchanged). On a noisy target the best of many
  remaining terms adds more than `parsimony_tol` to R² by fitting noise, so
  the step kept riders, 18 terms on average at signal to noise 1 in
  `benchmarks/parsimony_holdout/`. With a level, the step also stops at the
  first term whose gain fails to reject the null that it is no larger than
  the best gain the remaining terms reach against noise (200 wild-bootstrap
  draws of the residual, net of the terms already added; Fan & Zhou, 2016).
  In `benchmarks/parsimony_holdout/parsimony_alpha.py` (its designs at
  q = 0.1, 50 replicates per cell), `parsimony_alpha=0.05` printed 1.1 terms
  instead of 18.4 at signal to noise 1 and 3.8 at 3; the conditional false
  rate fell from 0.70 to 0.05 and from 0.42 to 0.02, power to print the
  generating term went from 0.89 to 0.88 at 1 (1.00 at 3), and held-out R²
  rose by 0.007 on average at 1, though 9% of those fits lost more than
  0.01. With `parsimony_holdout` and `certify_test="conditional"` it ends the
  re-test's fallback to the screened set (92% of fits at signal to noise 1,
  under 1% with the check). On real data it costs accuracy: in
  `benchmarks/release_comparison/` (5 splits each) held-out score fell on 8
  of 12 real datasets (tips 0.461 to 0.397, diabetes 0.430 to 0.381,
  riboflavin 0.506 to 0.306) and rose on one, eyedata, where the default's 32
  terms overfit 84 training rows. Real targets carry many modest effects,
  each adding little, so the step stops after one to four terms. It suits a
  search for a short law, not prediction.

### Fixed

- Memory of `input_test`. The expression evaluation behind it was a nested
  function that called itself, a reference cycle that kept the
  intermediates of every evaluation alive until Python's garbage collector
  happened to run. On California housing (7,224 selection rows, 133
  term-column pairs) a fit passed 7 GB without finishing; it now finishes in
  80 s using 0.55 GB, with the same terms. The undefined-row pass behind
  `predict` had the same cycle, holding its input until collected.

### Benchmarks

- `release_comparison/` runs the same benchmarks, data, splits and seeds
  through 0.4.2 (the paper's release), 0.5.0, 0.6.0 and 0.6.1 on one
  software stack, at their defaults and with the options of 0.6.0 and 0.6.1:
  about 16,000 fits, with speed and memory. 0.5.0 gives 0.4.2's fits on 1,415
  of 1,430 items, 3.2 times faster with a fifth of the memory; 0.6.0 with
  `fitted_ops=()` gives 0.5.0's on all 1,430, and 0.6.1's defaults give
  0.6.0's. `archive_check.py` refits 0.4.2 with the paper scripts' own
  settings: it reproduces the archived Feynman panel and 19 of the 25
  `run_benchmarks.py` datasets exactly.

### Documentation

- The Benchmarks page's "Corrections since the paper" is now "Reading the
  paper today", with the two bugs, the figures from runs made under an
  earlier release (riboflavin, the independent comparison and, newly noted,
  threshold_step: 0.808 in the paper, 0.843 from 0.4.2) and speed kept apart.
  The old link still leads there.
- The Benchmarks page and the README link the release comparison.

## [0.6.0] - 2026-10-24

0.5.1 was prepared but not released; its options are part of this release
and listed under "Added". One default changes: the estimators fit constants
inside expressions, `fitted_ops` (under "Changed"). The other new options
are off by default (`max_inputs=None`, `certify_test="marginal"`,
`magnitude_slots=0`, `input_test=False`), and a fit that leaves them there
and passes `fitted_ops=()` gives the features, p-values and predictions of
0.5.0, to the last bit, except with `selector="knockoff"`, whose fixed-X
statistic is fixed (under "Fixed").

The re-test's figures below are from `benchmarks/parsimony_holdout/`:
48,000 fits of a global null and of
`y = a*b/c` plus noise among 5 or 20 distractors, with and without a column
correlated 0.9 with `a`, at signal-to-noise ratios 1 and 3, 300 and 1,000
rows, 200 replicates per cell, q = 0.1 and 0.2. A printed term counts as
false in the conditional sense when it uses any column outside the formula,
as a rider such as `(a*b/c) * x7` does.

### Added

- `fitted_ops` on `BeamSearch`, `BeamFeatTransformer`, `BeamFeatRegressor`
  and `BeamFeatClassifier` (on by default in the three estimators, under
  "Changed"; `BeamSearch` defaults to `()`, which leaves every result
  unchanged): operators with one constant fitted inside the expression,
  `exp_scaled` (`exp(a*x)`), `pow` (`x^a`), `inv_shift` (`1/(x + a)`) and
  `log_shift` (`log(x + a)`). At each depth each is fitted to each member of
  the previous depth's beam and to that depth's `beam_width` best candidates
  the redundancy check kept out of the beam (a linear near-duplicate need not
  stay one under `1/(x + a)` or `exp(a*x)`), and as a factor of a beam member
  where the depth limit allows, by maximising |corr| with the target on the
  search rows. A fitted pole keeps a margin from the data, twice the spread
  of the operand's ten most extreme values and at least 1% of its range:
  the shifts' grids start at it, and `pow` fits a negative power, whose pole
  is at 0, only where the operand stays more than the margin above 0. The
  constant snaps to a simple value (0, ±1, ±2, ±3, ±1/2, ±3/2, ±1/3, ±1/4)
  when a likelihood-ratio test at level 0.001 fails to reject the null that
  it takes that value, never past the margin, and is exact in `to_sympy()`
  and `latex()`. At the end of the search, a kept feature with fitted
  constants gives way to a near copy with fewer (correlation at
  `redundancy_threshold`, and no near copy of another kept feature) when the
  same test, on the same rows, fails to reject the null that the copy fits
  the target as well; the simplest such copy is taken. Without that step, a
  constant fitted on the search rows bends an expression into a near copy of
  a simple law that outscores the law there, and the equation prints
  `(x0)^(-0.02)` for `log(x0)`: at 10% noise, `log(x0)`, `sqrt(x0)`, `1/x0`
  and `x0^2` were printed as themselves in 0, 0, 2 and 2 of 10 fits, and
  with it in 10, 7, 10 and 8 (10, 9, 10 and 8 with `fitted_ops=()`). The
  screen sees fixed expressions, so its guarantee is unchanged. `Node.const`,
  `Node.n_constants`, `transform(..., const=...)`,
  `OperatorSpec.constant_grid`, `pole_margins`, `SearchTrace.n_fitted` and
  `beamfeat.parametric` support it.

- `max_inputs` on the same four classes (default `None`): the most distinct
  input columns one feature may use, checked before evaluation. An
  expression of depth d uses at most 2^d columns, so at the default depth a
  value of 4 or more changes nothing. Measured at `max_inputs=3` against
  0.5.0, mean held-out R² over 5 splits:
  - gains on wide data with diffuse signal: eyedata 0.416 to 0.488,
    ct_slices 0.664 to 0.684, riboflavin 0.340 to 0.378;
  - losses: the twelve-law panel loses the gravitational law, which needs
    four inputs, and diabetes falls from 0.382 to 0.360;
  - one blogfeedback split falls from 0.533 to −2.14, where a three-input
    product extrapolates far past its training range on one test row.

- `SearchTrace.n_rejected_constraints`, the candidates `max_inputs` ruled
  out at each depth (0 when it is off).

- `.zenodo.json`, so that the Zenodo record of each release links the TMLR
  paper as the work the software supplements. Zenodo read the title, author
  and keywords from `CITATION.cff` but not its `preferred-citation`, so the
  records up to 0.5.0 do not mention the paper.

- `certify_test` on `BeamFeatTransformer`, `BeamFeatRegressor` and
  `BeamFeatClassifier`: the test the `parsimony_holdout` re-test applies.
  `"marginal"` re-runs the screen, as before, and so rejects the null of a
  term that only rides on a true one, `x0*x7` when the target depends on
  `x0` alone.
  `"conditional"` tests each re-tested term given the others with a
  Freedman-Lane permutation test: the target and the term are residualised
  on the other terms and an intercept, the target's residuals are permuted,
  the statistic is the partial correlation, and the screen's step-up
  correction is applied at the screen's level. Its null is a zero
  coefficient in a linear model of the target on the re-tested terms, and
  its p-values are approximate: exact for a single term, otherwise resting
  on exchangeable errors of a linear model of the target on the other terms.
  Regression only; classification raises. At a signal-to-noise ratio of 3 it
  brought the conditional rate of the printed equation to 0.18 to 0.22,
  against 0.42 to 0.44 at the defaults and 0.53 to 0.62 with the marginal
  re-test, and in the fits where it rejected at least one null it printed
  1.1 to 1.2 terms at a rate of 0.026 to 0.040 (q = 0.1), with power 0.98 to
  0.99. When the re-tested subset is long and its terms nearly reproduce one
  another the test rejects no null, and the screened set, which carries the
  marginal guarantee only, is returned: in 87 to 92% of fits at a ratio of
  1, where the forward step retains 18 terms on average, and in 18 to 26%
  at 3.

- `beamfeat.stability`: `stability_report(estimator, X, y)`, also exported
  from `beamfeat`, makes the paper's stability study (Appendix E) a package
  feature; by default `n_splits=30`, `test_size=0.25`, `random_state=0` and
  `n_jobs=1`. It refits clones of a regressor, classifier or transformer on
  random splits. Each split's test part is kept out of every step of its fit
  and scored once after it, and the clones' seeds are spawned from
  `random_state` by `SeedSequence`, so the report is the same for every
  `n_jobs`. The `StabilityReport` holds:
  - each split's rows, formulas, equation and held-out score;
  - the distinct features and equations, with how often each recurs;
  - a summary that states the ambiguity plainly. On diabetes it reads "30
    distinct equations in 30 splits; none recurs", with held-out R² from
    0.389 to 0.575.

  Two features are one when their values on the rows of `X` correlate above
  0.999 in absolute value, as in the paper, and an equation is the set of
  its features. The regressor and classifier are scored by their own
  `score`. A transformer is scored by a ridge regression (R²) on its
  features, or by a logistic regression (accuracy) for classification. The
  paper held one test part fixed and varied only the seed, so its spread is
  narrower: on diabetes, an R² standard deviation of 0.015 against 0.050
  here.

- `resample` on `stability_report`, which picks what varies. `"splits"`,
  the default and the behaviour above, measures split-to-split variability.
  `"seeds"` keeps the first split's partition for every split and varies
  only the clones' seeds, as the paper's study did, and measures
  seed-to-seed variability. With the paper's transformer on diabetes it
  gives an R² standard deviation of 0.008 to 0.019 over five partitions,
  with 75 to 84 classes, the commonest in 8 to 18 of 30 splits. The paper
  reports 0.015, 79 and 22 for its own partition. On that partition, its
  own code with other lists of 30 seeds gives 79 to 88 classes, the
  commonest in 18 to 23.

- `magnitude_slots` on `BeamSearch`, `BeamFeatTransformer`,
  `BeamFeatRegressor` and `BeamFeatClassifier` (default 0, which leaves
  every result unchanged; regression only, and a classifier raises): beam
  slots for laws whose parts do not correlate with the target, which the
  score ranks among noise, such as `x0` and `x1` in `x0*x1` or the pairwise
  products in `x0*x1*x2` when the inputs are centred. At every depth but the
  last, after the `beam_width` candidates the score keeps, up to that many
  more are kept, ranked by the absolute correlation of `log(1 + |c|/rms(c))`
  with the same transform of the centred target or of its residuals; one
  residual is on the best-scoring columns kept so far, at most a quarter as
  many as the rows, since the residual on every input is empty once they
  outnumber the rows. When `max_candidates_per_depth` can bind at the first
  depth, the inputs are proposed in an order that interleaves that ranking
  with the score's. Measured at 10 slots on `benchmarks/search_stress/`, the
  three-way product on centred inputs went from 0.54, 0.18 and 0.10 to 1.00
  with 20, 100 and 300 columns, the centred pair with 1,000 columns from
  0.32 to 1.00, and `x0*x1 + x2` with 300 columns and 250 search rows from
  0.00 to 0.88; no cell got worse. The score is computed on at most 1,000
  search rows, evenly spaced. Fits took 1.2 to 1.7 times as long on the
  stress grid and 1.4 times on the three 10,000-row high-dimensional sets;
  the main suite's mean held-out score over its 24 regression sets went from
  0.8851 to 0.8841 (diabetes and california housing about 0.01 lower) and
  the high-dimensional study's from 0.647 to 0.649. The ranking uses the
  search rows only, so the screen sees fixed expressions and its guarantee
  is unchanged; on four null and two planted designs, 200 paired replicates
  each, the share of fits with a false discovery moved by at most one in
  200.
  `magnitude_scores` and `SearchTrace.n_magnitude` support it.

- `benchmarks/search_stress/`: a stress test of the search on `x0*x1`,
  `x0*x1*x2` and `x0*x1 + x2` among distractors, on centred and on positive
  inputs, with 5 to 1,000 columns, 500 and 2,000 rows, 1% and 10% noise and
  beams of 50 and 200: 5,700 fits at the defaults, each followed through the
  search to the stage that lost the law. On positive inputs every law was
  recovered in every fit. On centred ones with 500 rows, `x0*x1*x2` was
  recovered in 54%, 18% and 10% of fits with 20, 100 and 300 columns, every
  miss the first depth's beam dropping the pairwise products, and a beam of
  200 lifted 54% to 96% and 18% to 26%; a centred pair was never proposed in
  68% of fits with 1,000 columns, and `x0*x1 + x2` was lost in every fit
  with 300 columns and 250 search rows, crowded out by sums with `x2`. A
  paired null study and a launcher that runs the other suites with new
  option defaults measure `magnitude_slots` as well.

- `input_test` on `BeamFeatTransformer`, `BeamFeatRegressor` and
  `BeamFeatClassifier` (default `False`): after the screen, each column
  inside each candidate that reads more than one is tested by permuting
  that column alone and recomputing the screen's statistic, with the
  screen's step-up correction across every (candidate, column) pair, and a
  screened term is dropped before parsimony when the test fails to reject
  the null for one of its columns. It removes riders such as
  `(x0*x1)/(x1 + x8)`, whose null the screen rejects because `x0*x1`
  carries them. The null for a column is that it is independent of the
  target and of the candidate's other columns, under which the p-values are
  exact; a junk column correlated with a real one is not null in that sense,
  so its null can be rejected and the term kept. A pair stops at its tenth
  exceedance (Besag & Clifford, 1991). `input_report_` lists every pair
  with its p- and q-value and `null_rejected`, and `equation()` counts the
  terms it dropped. If the test would drop every screened term, the screened
  set is used with a warning. Measured
  against the default, 20 seeds of `y = x0*x1` plus 25% noise among 10
  junk columns, 500 training and 500 test rows: printed equations with a
  junk column 14 → 1, printed terms with one 23/45 → 1/28, held-out R²
  0.9413 → 0.9416 (true formula 0.9425), 0.25 → 2.6 s a fit. With the junk
  `x8` correlated 0.82 with `x1`: 10 → 3. The physics panel is unchanged at
  10/12 solved, 8 exact; with five junk columns appended, equations with
  one 2 → 0, solved and exact unchanged. On pure-noise targets the screen
  rejects no null either way. Real data, mean over 5 splits: diabetes R² 0.429 →
  0.415 (6.2 → 5.0 terms), breast cancer accuracy 0.969 → 0.969 (24.2 →
  13.4 terms). The cost grows with the rows: california housing (15,480
  training rows) took 135 s against 2.6 s, R² 0.655 → 0.664 on one split.
  `n_jobs` threads share the tests, with the same p-values for every value:
  on california housing, 77 s with 8 threads against 202 s with one, in a
  later run on the same machine.

### Changed

- **`fitted_ops` is on by default** in `BeamFeatTransformer`,
  `BeamFeatRegressor` and `BeamFeatClassifier`: all four operators,
  `DEFAULT_FITTED` in `beamfeat.search` (`BeamSearch` itself still defaults
  to none). `fitted_ops=()` restores the previous default. It is a list of
  its own: restricting `unary_ops` leaves it on, so `log(x + a)` can appear
  unless `fitted_ops=()` too. Measured at the new default against
  `fitted_ops=()`: the physics panel solves the same 10/12 and gives the
  exact form of the Gaussian, `exp(-0.5*(x0)^2)`, and of relativistic
  velocity addition, `(x0 + x1)/((x0*x1) + 1)`, which it previously
  approximated; the core suite recovers the same 10/10, Friedman #1 rises
  from 0.745 to 0.765 and the other problems move by at most 0.0004; the
  robustness suite recovers the same 5/5 with R² lower by at most 0.004; the
  end-to-end calibration gives no false discovery in 200 replicates (power
  1.000) and no selection in 60 global-null replicates, as before; the
  pure-noise pipeline stress is unchanged (BH 6/100, BY 1/100). On five
  high-dimensional sets (communities, superconductivity, tecator, eyedata,
  riboflavin; 3 splits) the mean R² is 0.669 against 0.670, within noise.
  Fits take about 1.9 times as long on small data and 1.3 times on
  california housing. The benchmark scripts build their estimators at the
  defaults; `benchmarks/search_stress/with_options.py` with
  `BEAMFEAT_OPTIONS='{"fitted_ops": []}'` reproduces the archived results,
  and the Benchmarks page says which results predate the change. The
  tutorials are re-executed at the new default.
- `fitted_ops` fits run in batches: each operator is fitted to all of a
  depth's parents at once, with the golden-section refinements, the
  snapping and the rounding run side by side as array operations
  (`fit_constants_many`, which `fit_constants` calls), and the coarse grid
  is applied to every constant at once. Against the first implementation,
  prepared for 0.5.1, this halves their cost, from 3.4 to 1.9 times a plain
  fit's time on 18 fits (five distractor sets, the physics panel and
  diabetes), whose formulas, screened candidates and predictions are the
  same to the last bit. The search also counts each subexpression's
  operators and constants once per ranking instead of once per candidate,
  which speeds up fits without `fitted_ops` slightly too.
- `parsimony_holdout` is calibrated, and the guarantees page reports the
  figures in place of the sentence saying it had not been. In the marginal
  sense, the guarantee's, the printed equation's realised FDR stays far
  below q under every setting: 0.009 to 0.018 at q = 0.1 and 0.021 to 0.029
  at q = 0.2 on the null, and no term built only from distractors printed in
  the 3,200 planted and proxy fits per setting and level. The re-test adds
  nothing measurable there: on null data it failed to reject the null for
  the screen's false discoveries in 20 of 22 fits at q = 0.1, and the
  fallback to the screened set printed them anyway. In the conditional
  sense no setting controls the rate: 0.67 to 0.77 under every setting at a
  signal-to-noise ratio of 1, and at 3, 0.42 to 0.44 at the defaults and
  0.53 to 0.62 with the marginal re-test. Power and held-out R² hardly move
  (R² within 0.0033 of the default, paired).
- The guide has a section on laws whose parts the score cannot see, with
  `magnitude_slots`, and the guarantees page says why the option leaves the
  guarantee unchanged.
- The front page, which is also the README and the PyPI description, is
  shorter and leads with one result, tecator, in a figure that
  `benchmarks/figures/tecator.py` draws from the archived run. The measured
  results, the comparison with autofeat and the corrections to the paper's
  results move to a new Benchmarks page in the documentation, where the
  autofeat core-suite mean reads 0.9970, as in the paper, rather than 0.9968
  from a superseded run. The tutorials open with an install cell,
  `%pip install -q "beamfeat[all]"`, so that Run all works in Colab with
  units, LaTeX and export included, and the first notes which scikit-learn
  versions show the parameters and fitted attributes in the estimator's
  summary box.
- Hypothesis-testing wording throughout the documentation, docstrings and
  messages. Each candidate has its own null hypothesis, and a test rejects
  it or fails to reject it, where the text used to say that a feature
  passes, fails, survives or is kept. The guarantees page opens with these
  definitions, and the guide has a section on `input_test`. Messages change
  with the wording: an empty model's `equation()` and `latex()` end in
  `(no feature selected)` rather than `(no feature passed selection)`,
  `equations()` on it says that the screen rejected no null, and
  `NoDiscoveriesWarning`, `NoDiscoveriesError` and `DegenerateFitWarning`
  say what the screen rejected. The migration guide no longer says that
  beamfeat fits no constants inside expressions.

### Fixed

- The fixed-X knockoff statistic. The fixed-X guarantee (Barber & Candès,
  2015) needs a statistic that sees the data only through the Gram matrix
  of originals and knockoffs and their products with the target; the
  difference of cross-validated lasso coefficients used until now does not,
  since its folds look at individual rows. Regression on the fixed-X path
  now uses the lasso-path signed maximum, `max(Z_j, Z~_j) * sign(Z_j -
  Z~_j)` with `Z_j` the penalty at which column `j` enters the path,
  computed from those inner products (`lars_path_gram`). Model-X and
  classification keep the cross-validated statistic, which model-X allows.
  On Gaussian linear models at q = 0.2, 200 trials each, knockoff+ realised
  FDR 0.180, 0.172 and 0.135 (previously 0.151, 0.165 and 0.155) with power
  0.96, 0.98 and 0.48 (previously 0.97, 0.99 and 0.82) at n = 300, p = 25;
  n = 500, p = 50; and the same with correlation 0.7 between neighbouring
  columns: the guarantee now covers the statistic, at a cost in power on
  correlated designs, and a fit is 2–4 times faster. The permutation
  selector, the default, is unaffected.
- The `verbose` summary of a fit with `parsimony_holdout` said that the
  returned terms were "re-certified on held-back rows" even when the
  re-test rejected no null and the screened set was returned. It now says
  which happened, and for how many of the re-tested terms the re-test
  rejected the null. It also no longer calls the fitted terms a subset of
  the certified set when they are the whole of it.
- The permutation selector's warning for a permutation budget too small to
  reject any null suggested `correction='bh'` to a selector already using
  it; the suggestion is now made under Benjamini-Yekutieli only.

## [0.5.0] - 2026-10-20

0.4.5 was prepared but not released; its fixes are part of this release and
listed under "Fixed". The default path (correlation scoring, permutation
screening with Benjamini-Yekutieli) gives the same results as 0.4.4 except
where a fix applies; "Changed" lists the cases that differ. The new options
leave a default fit unchanged, and the search is faster and needs far less
memory on wide data: the high-dimensional study runs 2 to 28 times faster per
dataset, and the speed-up changes no feature, p-value or prediction.

### Added

- `equations()` and `equations_` on all three estimators: the forward
  parsimony step continued through the whole screened set, one equation per
  number of terms, with its complexity (expression-tree nodes plus one per
  coefficient), in-sample R² and the R² each term adds, and the fitted one
  marked. The model fitted is unchanged, and the path is recorded at
  `parsimony=None` too. The output states that the FDR guarantee covers the
  screened set, not an equation chosen from the list.
- `latex()` on `BeamFeatRegressor` and `BeamFeatClassifier`, the equation of
  `equation()` typeset through sympy with the same options and note
  (`note=False` drops it), and `latex_formulas()` for the features on all
  three estimators.
- The guide shows how to export the features to NumPy, JAX or PyTorch with
  `sympy.lambdify` on `to_sympy()` (PyTorch needs sympy 1.14 or newer).
- `n_jobs` on `BeamSearch`, `BeamFeatTransformer`, `BeamFeatRegressor` and
  `BeamFeatClassifier` (default 1). A depth's candidates are evaluated and
  scored in chunks on joblib threads; `GradientBoostingScorer`, whose fits
  hold the GIL, scores on worker processes, and `MutualInformationScorer`
  receives `n_jobs` through the joblib backend. Depths under 4096 candidates
  stay in the calling thread unless the scorer fits models. The result is
  identical for every value, for the scorers that draw random numbers too: no
  seed depends on a worker or a chunk.
- `BeamSearch(chunk_size=...)`, the candidates evaluated and scored together;
  by default as many as fit in 8 MiB.
- `Scorer.prepare`, `Scorer.score_prepared`, `Scorer.scores_by_column` and the
  `separable` flag, set on the correlation, Spearman and gradient-boosting
  scorers, so that a round can be scored in pieces; `scoring.ColumnStats` and
  `column_stats`; `Evaluator.plan_batch`, `EvaluationBatch.run` and
  `Evaluator.commit_batch`.
- `UnitRegistryError`, `ExclusionReason.UNDERFLOW`, an optional
  `OperatorSpec.redundant` predicate, a `symbols` argument to
  `Node.to_sympy`, and `SearchResult.n_rejected_redundant_final`.

### Changed

- The search evaluates and scores each depth in chunks and keeps only the
  scores; the candidates the redundancy pass reaches are evaluated again.
  Peak memory no longer grows with `max_candidates_per_depth`: a default fit
  on 1000 columns and 600 rows peaked at 2.6 GB and now at 0.35 GB.
  `MutualInformationScorer`, whose noise is drawn for the whole batch, and
  custom scorers that do not declare `separable` still receive a whole depth.
- A candidate's mean, variance and magnitude are computed once, for the
  constancy filter and the scorer, and columns are standardised in blocks
  rather than one at a time. The same fit takes 7 s rather than 29 s, and one
  on 300 columns 3.3 s rather than 19 s. Features, p-values and predictions
  are unchanged, as are search scores and the evaluation log, to the last
  bit.
- Results that differ from 0.4.4: knockoff selections on candidate pools
  with dependent columns, which are common among constructed features;
  searches whose first round reaches `max_candidates_per_depth` (about 258
  inputs or more at the defaults); `SpearmanScorer` scores with ties;
  predictions for batches containing out-of-domain rows; the constant printed
  by `equation(max_terms=...)`; fits with a single candidate, which may now
  return nothing; fits on columns whose spread is between 1e-12 and 1e-5 of
  their magnitude, which the constancy cut no longer drops; and the spelling,
  not the value, of some formulas, since `(1/a)/b` is folded into `1/(a*b)`
  only when that keeps the depth (`1/((x1 * (x0 * x2)))` now reads
  `(1/(x1) / (x0 * x2))`, with the same p-values and predictions).
- `SearchResult.n_evaluated_total` no longer counts the input columns,
  matching `n_proposed_total`.
- `joblib` is a declared dependency (`>=1.2`, scikit-learn 1.6's own
  minimum), since the search imports it directly; scikit-learn already
  installs it, and the oldest-stack CI job tests 1.2.0.
- The API reference documents every public module and the methods the three
  estimators share. `ColumnStats`, `column_stats`, `EvaluationBatch` and
  `BatchOutcome` are listed in their modules' `__all__`.
- The tutorials cover the equation list, LaTeX, export, `n_jobs` and the
  chunked search, describe knockoff routing and the scorers as they now are,
  and are executed on 0.5.0. The guide has a section on speed and memory.
- The release workflow runs the full CI suite and publishes only after it
  passes; a tag push previously published before the tests finished.
  `CONTRIBUTING.md` describes the release flow.
- The logistic models fitted internally are built with the arguments the
  installed scikit-learn expects: `alpha = 1/C` from 1.10 rather than the
  deprecated `C`, `l1_ratios` rather than `penalty` from 1.8, accuracy as the
  cross-validation score explicitly (1.11 changes the default), and
  `newton-cd` rather than the deprecated `liblinear` from 1.10. From 1.8,
  where `liblinear` fits two classes only, each class of a multiclass target
  is fitted on its own; the warning 1.7 gives ahead of that change is
  silenced. Fits on scikit-learn 1.6 to 1.9 are unchanged. Their other
  warnings are silenced by category, user and runtime warnings only, so a
  deprecation of an argument beamfeat passes is no longer hidden with the
  convergence notices.
- CI also runs weekly, a weekly workflow tests the pre-releases and nightly
  builds of numpy, SciPy, scikit-learn and pandas, and the test suite fails on
  any `FutureWarning` and on a `DeprecationWarning` raised from beamfeat or its
  tests, so that an upstream deprecation is dealt with before the removal.
- Docstrings describe what the code does: ranking on the larger of the
  absolute and marginal scores, redundancy pruning within a depth with
  cross-depth duplicates removed in the final ranking, and correlation scoring
  as one dot product per candidate. `SpearmanScorer`, `is_constant` and the
  estimator warnings are listed in their modules' `__all__`, and
  `SpearmanScorer` is exported from the package, as the other scorers are.
  The estimators document all their fitted attributes.
- 230 tests added (686 in all).

### Fixed

- `KnockoffSelector` on a classification target with three or more classes,
  and `BeamFeatClassifier(selector="knockoff")` on one, raised on
  scikit-learn 1.8 and later, where the `liblinear` solver fits two classes
  only. The statistic is computed one class against the rest, as it was.
- A refit no longer keeps fitted attributes of an earlier fit: the
  classifier's `class_prior_` after a fit that kept no feature, the
  regressor's `alpha_` when the new fit keeps none (it is now `None`), and
  the classifier's coefficients when the new training set has one class.
  Predictions were not affected.
- The model-X warning of `KnockoffSelector` no longer reports a fallback from
  fixed-X when model-X was requested with `construction="gaussian"`. It is
  recorded whenever model-X runs, as the docstrings now say.
- A row on which a fitted formula is undefined, such as the logarithm of a
  negative value, no longer changes the predictions for the other rows of the
  batch. The formula's whole column was replaced with zeros for every row
  whenever any row failed, so a valid row predicted 2.88 alone and -6.63 next
  to an out-of-domain row, with only a log message. The failing rows are now
  NaN in `transform`, `predict`, `predict_proba` and `decision_function`, and
  a `UserWarning` names the formula and counts the rows. The classifier's
  `predict` assigns those rows the most frequent training class, since a
  label cannot be NaN; the regressor's `score` raises.
- `KnockoffSelector` leaves constant and linearly dependent columns out of the
  knockoff construction when the design covariance has an eigenvalue below
  1e-6. Valid knockoffs of such columns equal the originals, so the lasso's
  tie-breaking decided the statistics in favour of the originals: with one
  duplicated null column, knockoff+ reached a mean FDP of 0.31 at q = 0.1,
  under fixed-X and model-X alike. Dropped columns get W = 0 and are named in
  the warning; routing and the fixed-X row requirement count the retained
  columns.
- `BeamSearch` reaches late input columns when `max_candidates_per_depth`
  binds at depth 1. The first round proposed parents in column order, so from
  about 258 inputs (default ceiling) pairs among the last columns were never
  proposed. When the ceiling can bind, the first round now proposes the
  inputs in order of their score, as documented; rounds the ceiling cannot
  reach are proposed as before and give identical results. Three datasets of
  the high-dimensional study reach the ceiling. Rerunning 0.4.4 and 0.5.0 on
  one machine, mean R² moves from 0.523 to 0.340 on riboflavin (the archived
  study, made with an earlier version, reads 0.544), from 0.687 to 0.664 on
  ct_slices and from 0.812 to 0.816 on ujiindoorloc.
  With the ceiling raised so that every input of ct_slices and ujiindoorloc
  is proposed, the search returns the result above exactly; the 0.4.4 figures
  depended on which columns came first in the file.
- `SpearmanScorer` gives tied values their average rank. Ordinal ranks broke
  ties by row order, so a score depended on how the rows were arranged and a
  constant candidate scored above zero. An incumbent set that explains the
  target exactly now zeroes every score, a 1-D incumbent is accepted, and a
  candidate with a non-finite value scores zero, as under `CorrelationScorer`.
  For classification only the candidates are ranked: ranking the logistic
  residual let a copy of an incumbent score about 0.5.
- String and object class labels work in every scorer and in the parsimony
  step; both cast the target to float, so `BeamFeatTransformer(problem_type=
  "classification")` failed on string labels.
- `BeamFeatClassifier` runs the post-fit check on the selection rows, so
  `DegenerateFitWarning` fires for classification as documented.
- A lone candidate is screened like any other set; screening was skipped and
  the term returned unvetted. When no candidate can be evaluated on the
  selection rows, `on_no_discoveries` applies. The `DegenerateFitWarning`
  message no longer says features passed FDR screening when they did not.
- `on_no_discoveries`, `selection_holdout`, `parsimony_holdout`,
  `selection_correction` and `parsimony` are validated at fit. A holdout
  fraction outside (0, 1), such as 1.5, switched the split off silently, and
  an unrecognised `on_no_discoveries` behaved like `"empty"`.
- A `Selector` instance keeps its own `correction` and `target_fdr`, and the
  estimator warns when its own settings disagree: `selector=
  PermutationSelector()` ran Benjamini-Hochberg where the estimators default
  to Benjamini-Yekutieli, without saying so. Verbose output and warnings
  report the correction and level that ran.
- `fdp_inflation_` describes the set the model returns, after the
  `parsimony_holdout` re-test or its fallback.
- `equation(max_terms=...)` holds the omitted terms at their training mean,
  so the printed constant absorbs their average contribution; it kept the
  full model's constant, as if the omitted terms were zero.
- Class labels print as plain values (`logit P(y = 1)`, not `np.int64(1)`).
- `make_scorer` warns when given a scorer instance whose `problem_type`
  differs from the one requested.
- Node keys escape backslashes, parentheses and commas in column names, so a
  column named like `a,leaf:b` can no longer give two expressions one key and
  one cache entry. Keys and formulas of ordinary names are unchanged.
- Units from different pint registries no longer drop candidates silently:
  `Evaluator` re-expresses every unit in the first one's registry, and nodes
  built directly from mixed registries raise `UnitRegistryError`.
- Exclusion reasons name the cause: an overflowing variance is `overflow`
  (was `zero_variance`), an underflow is `underflow` (was `non_finite`), and
  `abs` of a never-negative operand is `duplicate` (was `domain_error`).
  Which candidates are excluded is unchanged; the overflow's `RuntimeWarning`
  no longer escapes.
- The evaluation cache re-checks a value first computed without the search
  filters, and returns read-only arrays. A failing node is logged once per
  evaluator.
- `(1/a)/b` is folded into `1/(a*b)` only when the fold does not deepen the
  node; it could exceed `max_depth`.
- `Node.to_sympy` and `to_sympy()` on the estimators give distinct symbols to
  columns whose sanitised names coincide (`a b` and `a-b`).
- The evaluator's default constancy cut matches `is_constant` (spread below
  1e-12 of the magnitude; was 1e-5), so a column varying by a millionth of its
  value is no longer dropped from the search without a warning.
- `redundancy_threshold=1.0` drops exact affine and sign-flipped copies
  consistently; their correlation landed a rounding error either side of 1.
  The redundancy check judges constancy relative to a column's magnitude.
- `BeamSearch` rejects `max_candidates_per_depth` below 1 (0 returned the
  inputs), and its final `verbose` line reports the depth reached.
- The guide's "honest failure" example used a seed on which pure noise
  certifies 12 formulas, which happens on about 1 fit in 100; it now uses one
  where nothing passes and says that some pure-noise fits certify features.

### Benchmarks

- The comparison harnesses (`independent/bench.py` and its copy in
  `additional_experiments/`) filled non-finite test cells of constructed
  features with 0 while training cells got the training median; both now use
  the training medians. The archived results are unchanged;
  `independent/PROVENANCE.md` lists the affected `featuretools` and `OpenFE`
  fits and relates them to the paper's comparison table.
- `run_benchmarks.py` scores knockpy recovery with the shared criterion
  instead of recording `False`.
- `calibration_study.py` bounds the global-null rate by trials selecting
  anything rather than features selected, and says its scores are in-sample.
- `depth_ladder.py` seeds with `zlib.crc32` instead of the per-process
  `hash()`; the archived ladder used the old seeding.
- `additional_experiments/aggregate.py` reads the `highdim_*.json` records,
  the README run order matches the archived run, and `selector_comparison.py`
  reports selector errors instead of counting them as skips.
- Stale docstrings corrected in `selector_calibration.py`, `feynman_panel.py`,
  `bench.py` and `selector_comparison.py`.

## [0.4.4] - 2026-10-18

### Fixed

- `Selector.select` now rejects a target containing NaN or infinity, and a
  missing class label. A NaN made every permutation statistic NaN, so no null
  draw counted as an exceedance and `PermutationSelector` selected every
  candidate at the p-value floor; as a class label it formed a class of its
  own. The estimators were not affected: scikit-learn's validation rejects
  such a target before selection.
- Fixed-X knockoffs are built orthogonal to the intercept as well as to the
  design, so they are centred like the standardised features. The lasso
  statistic fits an intercept, which centres every column; uncentred
  knockoffs lost part of their norm to it, and the exchangeability identities
  `X~'X~ = X'X` and `X'X~ = X'X - S` held for the constructed columns but not
  for what the lasso saw (deviation up to about 0.03 at n close to 2p).
  Simulations at q = 0.1 showed no measurable change in realised FDR, which
  stayed far below nominal either way. The complement now needs one more
  row: fixed-X is used when n > 2p (was n >= 2p), and at n = 2p the selector
  falls back to model-X.
- Model-X knockoff noise is drawn from a seed spawned from `random_state`, as
  the fixed-X basis already was. Drawn from `random_state` itself, it equalled
  a data matrix generated from the same seed and shape, which made the
  knockoffs a fixed linear image of the features.
- `PermutationSelector` with `auto_permutations=False` warns when
  `n_permutations` is below what the correction needs to select anything;
  an empty selection was previously returned in silence.
- The `"fixed_x"` and `"model_x"` selector names select that knockoff
  construction; both previously meant `construction="auto"`. An explicit
  `construction` argument still takes precedence.
- The knockoff+ warning states the floor as "at least ceil(1/q)" features;
  it said "more than 1/q". The condition itself was correct.

### Changed

- Knockoff selections can differ from 0.4.3 for the same `random_state`,
  since both constructions draw differently. The permutation selector's
  results are unchanged.
- Docstrings, `docs/guarantees.md` and the README describe permutation
  p-values as valid in finite samples rather than exact, as the paper does.
- A Benjamini-Hochberg test now uses p-values on which Bonferroni selects
  fewer, so it distinguishes the two procedures. 20 tests added (456 in all).

## [0.4.3] - 2026-10-18

### Added

- The TMLR paper, "Symbolic feature construction with false discovery rate
  control", is the preferred citation in `CITATION.cff` and the README.

### Fixed

- Package metadata links point to the lowercase `LD-Shell/beamfeat` repository
  and documentation site; the 0.4.2 Documentation link returned 404.

## [0.4.2] - 2026-10-16

### Fixed

- `BeamFeatClassifier.decision_function` raised `AttributeError` when no
  feature passed selection. It now returns the log prior odds (binary) or log
  priors (multiclass), consistent with `predict` and `predict_proba`, which
  already fell back to the class priors. scikit-learn's `check_estimator` is
  now also run with FDR selection on, which is what exposed it.

### Changed

- The tutorial notebooks no longer suggest installing from GitHub; `pip install
  beamfeat` is the install route.
- CI now also tests Python 3.14 and 3.15, and the package metadata lists them.

## [0.4.1] - 2026-09-20

### Fixed

- `CITATION.cff` shipped in 0.4.0 with unresolved merge-conflict markers, so it
  was not valid YAML and Zenodo could not archive the release. The 0.4.0 source
  distribution carries the invalid file; the wheel never contained it. No code
  changed.

## [0.4.0] - 2026-09-19

`equation()` now says when the equation it prints is a subset of the certified
set. The default is unchanged -- `parsimony="forward"` still keeps the compact
predictive subset -- but a caller reading the equation next to
`fdr_controlled_ = True` previously had no signal that the two do not describe
the same terms. They now do:

```text
y = 0.9999*(x0 * x1) - 0.003943   [1 of 25 certified terms; parsimony=None prints all 25]
```

What the default costs is also measured rather than asserted. Across 308
paired fits the compact equation is worse by 0.001 of held-out R^2 on average
and 0.019 at worst, roughly twenty-five times shorter, and `fdr_controlled_`
never disagreed between the two settings. The end-to-end calibration figures
are identical at both. `benchmarks/PARSIMONY_COST.md` records the numbers, the
per-study dependency map behind them, and what pruning does to the realised
false discovery proportion, which in every regime tested was nothing or an
improvement. Seventeen tests take the suite to 431.

The printed string is the only behavioural change, and it affects anything
that parses `equation()` output. `equation()` splits cleanly on `"   ["`, and
the suffix is absent whenever the printed terms *are* the certified ones.

### Added

- `equation()` appends `[k of m certified terms; parsimony=None prints all m]`
  when the parsimony step dropped terms. It is absent with no selector, with
  nothing screened, at `parsimony=None`, when the greedy pass happened to keep
  everything, and after a `parsimony_holdout` re-test, which certifies the
  subset on rows of its own and so leaves nothing to warn about. On a
  multiclass classifier the terms are shared across the lines, so the suffix
  appears once at the end of the block.
- `fdr_scope_` names which set `fdr_controlled_` is a statement about:
  `"printed equation"` when the guarantee covers the returned features,
  `"screened set"` when it covers a larger set they were drawn from, `None`
  when the flag is not `True`. The flag has always meant "the returned
  features came from the screened set", which is not the same claim, and a
  caller branching on it in a script cannot see the note `equation()` prints.
  At `verbose=1` the result line names the scope too.
- `benchmarks/PARSIMONY_COST.md` and `benchmarks/parsimony_cost.py`: the
  measurement above, and the runner that produces it. The runner drives each
  study through its own module rather than a copy of its protocol, and writes
  only to `benchmarks/parsimony_cost/`; no archived result is read or written.
- `parsimony` is selectable from the benchmark harnesses -- a keyword argument
  on `feynman_panel.run_panel`, `calibration_study.main`,
  `friedman_decomposition.main` and `run_benchmarks.evaluate`, a `--parsimony`
  flag on the three `additional_experiments` studies, and the
  `BEAMFEAT_PARSIMONY` environment variable for the two `bench.py` harnesses.
  Unset or omitted reproduces the library default exactly, and the value in
  force is recorded on every row.
- `feynman_panel.make_data` exposes the panel's data generation, so a
  comparison harness scores its method on the same rows rather than on a copy
  of that code. It is the original code moved, not rewritten: the arrays it
  returns are identical to those the previous version produced.
- `benchmarks/pysr_panel/`: PySR on the same twelve-law panel, in its own
  pinned environment with its own Julia runtime, scored by the panel's own
  data generators and proportionality test imported unmodified. The budget,
  the operator sets and the Pareto-front selection rule are fixed in advance
  and stated in its README. `pysr` is not a dependency of this package.

### Documentation

- The guarantees page states the three-way trade-off with the measurement
  attached, rather than asserting the exchange rate.
- The user guide's opening example explains the suffix where a reader first
  meets it, and shows what `parsimony=None` returns instead.

## [0.3.1] - 2026-08-22

Correctness release for the automatic permutation budget, plus two warning
fixes. Selections are unchanged everywhere the previous budget was already
adequate: the two widest problems in the high-dimensional study (riboflavin
and superconductivity) return identical formulas, feature counts, flags and
held-out scores, with fit times differing by less than the run-to-run spread.
The budget binds in the selector calibration studies, where a small pool at a
strict level is where the harmonic factor matters: rerun with the fix,
the Benjamini-Yekutieli figures move in the third decimal (realised FDR at
q = 0.05 from 0.008 to 0.013; power at that level from 0.988 to 0.989), and
the figures in the README, the docs and `benchmarks/additional_experiments/REPORT.md`
come from runs with the fix. Thirty-seven tests take the suite to 411 at 95%
statement coverage.

### Added

- `verbose` is now a level rather than a flag, and reports to stdout instead
  of through the module logger -- progress a caller explicitly asked for
  should not require them to attach a handler first, which is the
  scikit-learn convention. `0` (default) is silent and unchanged; `1` prints
  one line per stage, covering the split, the search, the screening, the
  parsimony step and the fitted equation, with the FDR flag on the result
  line; `2` adds per-depth search detail -- proposed, evaluated, kept, what
  was rejected and why -- and the strongest certified candidates with their
  p- and q-values. Warnings and diagnostics continue to use the logger,
  because those are events the caller did not ask for. Reporting now covers
  selection as well as search; previously any positive `verbose` logged
  search progress only, and the screening step, which is where the guarantee
  is actually applied, was invisible.

### Added (continued)

- `parsimony_holdout` divides the selection rows a second time, so that the
  printed equation carries a guarantee of its own rather than being a subset
  of a certified set. Screening and parsimony run on the first part; the
  resulting subset is fixed at that point, so re-testing it on the part held
  back is an ordinary fixed-candidate screen and its guarantee covers the
  subset itself. Terms that fail are dropped; if none survives, an
  intercept-only model is returned with a visible warning rather than an
  uncertified equation. Off by default, because what it costs is rows: on
  240- and 5000-row problems it returns the same terms at the same accuracy,
  and on a 71-row problem it loses twelve of nineteen terms. The estimator
  warns rather than proceeding when the selection rows cannot be split so
  that both parts hold at least ten. When a compact certified equation cannot
  be produced -- because the rows will not split, or because nothing survives
  the re-test -- the whole screened set is returned rather than a pruned
  subset of it. The caller asked for a guarantee over what is printed; the
  screened set supplies one and a pruned subset does not, so that is the
  honest way to degrade. Note that `parsimony=None` reaches the same corner
  for free, and on short samples is the better choice: a long certified
  equation beats a compact one whose terms were dropped for want of rows.
  The two-stage procedure has not been FDR-calibrated and is offered as an
  option rather than a measured result.
- `fdp_inflation_` reports |S|/|S'|, the factor by which pruning the screened
  set down to the printed equation can inflate the realised false discovery
  proportion. It was derivable from `selection_report_` before and is now
  surfaced directly, since it is the number to quote alongside `target_fdr`.

### Added (continued)

- Constant input columns are reported at fit time. Such a column standardises
  to zeros, so its association with the target is zero at every depth it can
  appear in and it can never be selected; that is the correct outcome and
  costs neither accuracy nor multiplicity, since the beam filters it before
  screening. What it does cost is legibility: in the output a stuck sensor or
  a column emptied by a bad join looks exactly like a variable that does not
  matter. The test consults no response values, so reporting it cannot bias
  what follows. It fires on real data -- 55 of 521 columns in ujiindoorloc,
  five of 385 in ct_slices, four of 281 in blogfeedback.
- `is_constant` is exported. Constancy is judged relative to the column's
  magnitude rather than against an absolute floor on the spread, because an
  absolute floor makes the verdict depend on the column's units: a column
  varying over the whole range 0 to 1e-9 has a smaller variance than a stuck
  sensor reading 9.81, and any floor that keeps the first also keeps the
  second. The estimator and the diagnostic now share the one predicate, so
  the warning cannot come to describe behaviour the fit no longer has.

### Documentation

- New tutorial, `notebooks/04_reading_a_fit.ipynb`. The first three notebooks
  plant an answer and confirm it is recovered; this one is about the fit you
  did not plant. On a pump-power problem hidden among noise columns it walks
  through the cases an applied user actually hits: why a deeper search returns
  a *worse* equation, why `fdr_controlled_` stays `True` when noise columns
  appear inside a formula (the null is marginal, so a formula combining the
  true core with noise has a false null and selecting it is not an error),
  why depth counts composition levels rather than variables, how units remove
  the hitchhikers where statistics cannot, why unit coverage is all or
  nothing, where a constant factor goes, how to read `selection_report_`
  against `equation()`, and how to tell an exhausted permutation budget from
  an absent signal. Runs in about 14 s.
- Notebook 03 notes that knockoffs cannot prune a term out of an expression:
  selection operates on whole candidate columns, so switching the selector
  does not reach inside a composite. The conditional null invites that
  inference and it does not hold.
- Notebook 01 builds its design as a DataFrame, so formulas read `(a * b) / c`
  rather than `(x0 * x1) / x2`, and demonstrates the verbosity levels.

### Fixed

- The permutation count chosen when `auto_permutations` is on was sized to
  the Benjamini-Hochberg resolution bound `2m/q`, while `"by"` is the
  estimator's default correction. Benjamini-Yekutieli requires the leading
  p-value to clear `q/(m c(m))`, so its bound is larger by the harmonic
  factor -- above five once `m` reaches the low hundreds. Below it the
  correction cannot reject at all, and the failure was silent: an empty
  selection reads as "no signal" rather than "budget too small". It bites
  hardest when few candidates reach the p-value floor together, since several
  tied signals relax the threshold by their count while a lone one carries it
  alone, which is why a sparse true set in a large pool was the exposed case.
  `_required_permutations` now takes its bound from the configured
  correction, and `max_permutations` rises from 100,000 to 1,000,000 so the
  larger requirement is reachable. The warning raised when it is not names
  the correction and offers `correction="bh"` as the cheaper alternative.
- Fitting on a `DataFrame` no longer emits scikit-learn's "X does not have
  valid feature names" warning. The post-fit degeneracy diagnostic scores the
  estimator's own validated array, which carries no column names by
  construction; the warning fired on every named fit and pointed the caller
  at a mismatch of our own making.
- `units` covering some columns but not all now warns. An unlabelled column
  is dimensionally unconstrained and combines freely with the labelled ones,
  so partial coverage leaves the check not binding on exactly the columns the
  caller did not vouch for -- while a mapping matching *no* column already
  raised. Labelling the known columns and leaving the rest blank looks like
  the careful thing to do, which is what made the silence worth removing.
  Pass `"dimensionless"` for genuinely unitless columns. The warning is
  raised once at fit, not on every `transform` or `predict`.

## [0.3.0] - 2026-08-12

Behavioural release: the downstream ridge penalty is now selected by
leave-one-out cross-validation. Fitted coefficients and held-out scores move
against 0.2.0 -- in trailing digits when rows comfortably exceed the selected
features, and materially at p >> n, where the fixed default could produce an
effectively unregularised fit. Selections are unchanged: the search,
permutation tests and FDR control sit upstream of this step. Every benchmark
artifact ships regenerated under the new default; the Friedman #1
decomposition reads 0.780 ± 0.013 achieved with 0.094 to the search (from
0.776 and 0.098), and four estimator-check tests take the suite to 374 at
94% statement coverage.

### Changed

- `alpha` on `BeamFeatRegressor` defaults to `"auto"`: the downstream ridge
  penalty is chosen by efficient leave-one-out cross-validation over a
  logarithmic grid, deterministically. Pass a float for the previous fixed
  behaviour. The chosen strength is exposed as `alpha_`.
- The `fdr_controlled_` documentation states the guarantee's object
  precisely: the set-level q guarantee applies to the full screened set in
  `selection_report_`; the parsimony subset is not re-certified at level q.
- The post-fit check is documented as what it is -- a degeneracy diagnostic
  on the selection rows, which also enter the final fit -- rather than an
  independent generalisation estimate, and its warning text says so.

### Fixed

- With many correlated certified features selected from few rows, the fixed
  default `alpha=1.0` amounted to almost no regularisation and could produce
  catastrophically negative held-out scores on features the screening had
  correctly certified. The cross-validated default keeps such fits on the
  scale of the target.

## [0.2.0] - 2026-08-08

Behavioural release: scale-free numerics throughout, strict unit validation,
and a faithful `equation()`. Results are not bit-reproducible against 0.1.1,
and selections on small- or mixed-unit data change deliberately.

### Added

- `style` parameter on `equation()` for both supervised estimators:
  `"significant"` (default), `"fixed"`, and `"scientific"`. `"fixed"` falls
  back to significant figures for any value that would otherwise display as
  zero.
- `units` accepts a positional sequence with one entry per column, alongside
  the existing mapping form.

### Changed

- `variance_tol` is judged relative to a column's squared mean magnitude, so
  the constant/not-constant verdict no longer depends on the caller's units.
  Anyone who tuned it against an absolute variance should re-check the value.
- Candidate ranking treats scores within 1e-6 as tied and admits the smaller
  expression; the redundancy pass then removes the complex near-duplicate.
- Invalid `units` fail loudly: a mapping matching no columns, a wrong-length
  sequence, or an unsupported type raises instead of silently skipping
  dimensional analysis.
- `equation()` orders terms by standardised coefficient magnitude and prints
  significant figures; `precision` counts decimal places only under
  `style="fixed"`.

### Fixed

- `equation()` dropped any term whose raw coefficient fell below the printed
  precision, deleting dominant terms on large-unit features from their own
  equation. Only exactly-zero terms are omitted now.
- Scoring zeroed any column whose absolute spread fell below 1e-12, which
  made products of small-unit columns undiscoverable; the guard is now
  relative to the column's magnitude.
- A saturated incumbent yields zero candidate scores at every target scale;
  previously the check held only for targets near unit magnitude.

## [0.1.1] - 2026-07-28

Documentation, benchmarks and packaging. No library code changed; the public
API, defaults and behaviour are identical to 0.1.0.

### Added

- `benchmarks/selector_calibration.py`: realised FDR and power for both
  multiplicity corrections, the two fixed-X knockoff offsets, and the
  global-null stress behind the Benjamini-Yekutieli default.
- `benchmarks/make_figures.py`: seven vector PDFs in `paper/figures/`, sized to
  the ACS column widths, including mean held-out R² against fit time from the
  comparison study.
- `benchmarks/independent/setup_env.sh`: builds the comparison environment,
  registers a Jupyter kernel, and smoke-tests `autofeat`, `knockpy` and
  `matplotlib`.
- `beamfeat_ridge` in the comparison study: `beamfeat` as a transformer into
  the shared `RidgeCV`, giving one row strictly like-for-like with the other
  feature builders.
- `docs/installation.md`: three setups, from `pip install beamfeat` to the
  pinned benchmark environment.
- `benchmarks/independent/autofeat_repeatability.json`: six runs of one split.

### Changed

- Friedman #1 decomposition is reported over six draws rather than one:
  oracle 0.960 ± 0.003, screening ceiling 0.874 ± 0.006, achieved
  0.776 ± 0.013, with about 0.086 lost to screening and 0.098 to the search.
  The previous single-draw figures came from a different split for the achieved
  value and overstated the gap. `friedman_decomposition.py` now checks the
  ordering the argument rests on rather than comparing against recorded
  numbers, and computes the ceiling on the set actually admitted: the marginal
  null never admits `c`, but admits `c²` on some draws, so the earlier
  "admits only four of the six" was true of one draw rather than in general.
- knockpy's stress false-feature rate was averaged over six datasets where
  only five declare their generating columns. Reported as a count of affected
  datasets instead, since a mean over five carries a standard error of order
  0.1; knockpy is also unseeded, so the count moves between runs.
- Selector calibration runs 100 trials rather than 25 and reports a standard
  error. The published 25-trial figures were noisy point estimates: BH at
  nominal 0.10 read 0.120, above nominal, where 100 trials give 0.084 ± 0.012
  against a ceiling of 0.080. `selector_calibration.py` now derives each
  procedure's bound from the design and fails if a realised rate exceeds it,
  so no calibration constant is written by hand anywhere.
- Fixed-X knockoff+ 0.201 to 0.159 and `offset=0` 0.289 to 0.249 at 100 trials.
- The stress-suite false-feature rate is reported as a count of affected
  datasets rather than a mean over five, which carried a standard error of
  about 0.15 for autofeat and 0.08 for knockpy.
- The end-to-end result is stated as no false discovery in 200 replicates with
  a 95% upper bound of 0.015, rather than as an empirical rate of 0.0000.
- Pure-noise stress at 100 trials: BH returned features in 6 trials against
  BY's 1, both under the nominal 0.10 and not significantly different (the
  one BY trial is among BH's six; exact paired test p = 0.06, which replaces
  an unpaired Fisher p = 0.12 reported earlier). BY is the default because it
  is valid under arbitrary dependence, not because a BH failure was observed.
- `autofeat` reproducibility: it exposes no `random_state` and draws decoy
  features from the global NumPy generator, so six runs of one identical split
  returned R² from +0.952 to -109.8 with 17 to 26 selected features. Thread
  pinning does not help; the previous "seeds no internal subsampling" was wrong.
- Real-data panel is seven datasets; California housing uses all 20,640 rows
  rather than a 2,000-row subsample.
- Every measured figure now comes from one machine (Dell Inspiron 16 Plus 7640,
  22 logical cores, Linux 7.0, Python 3.11.15, scikit-learn 1.7.2, numpy
  1.26.4), the pinned environment the compared tools require.
- Reproducibility is stated from two measured machines: deterministic given a
  seed within a stack, with marginal fits moving in the third decimal across
  scikit-learn 1.6 to 1.9.
- `pytest-cov` moved into the `test` extra, so `floor-versions` in CI and a
  local `pip install -e ".[test]"` both get it without naming it separately.
- The comparison notebook calls `bench.py` rather than duplicating it, dropping
  from 523 lines to 136.
- Removed `benchmarks/independent/run_all.sh`: it re-implemented the notebook's
  full-run path, so a method added in one place was missing from the other.
- The sdist ships the benchmark scripts and archived results, which
  `benchmarks/README.md` already documented.

### Fixed

- `feynman_panel.py` seeded each equation's data with `hash(name)`. Python
  randomises string hashing per process, so the panel drew different data on
  every run and reported 9/12 or 10/12 at random; the earlier claim that this
  varied "across numeric stacks" mistook the cause. Seeded with `zlib.crc32`,
  it now reproduces exactly at 10/12 solved and 8/12 exact form.

- Both OpenFE patchers imported the package before rewriting it, so the kernel
  kept running unpatched code while reporting success.
- `benchmarks/independent/requirements.txt` installs `beamfeat[all]`, so the
  test suite runs there; pins `setuptools<82`, without which every
  `featuretools` fit fails on the removed `pkg_resources`; and adds `knockpy`
  under the existing numpy pin rather than letting it pull numpy 2.
- `aggregate.py` takes the results directory as an argument and writes its CSV
  beside the files it read.
- Independent study regenerated in a single pass on one machine: 360 fits,
  nine datasets, eight methods. It replaces an archive stitched from several
  partial runs that contained one hand-transcribed fit.
- Removed an unsupported OpenFE figure on the diabetes dataset from the README.

## [0.1.0] - 2026-07-27

Initial release.

- Beam search over an expression DAG with commutative canonicalisation,
  local algebraic normalisation, and record-and-exclude handling of
  numerical failures. Unary operators `log`, `sqrt`, `reciprocal`,
  `square`, `abs` by default, with `cube` and `exp` available; binary
  `mul`, `div`, `add`, `sub`.
- FDR-controlled selection: exact permutation test of marginal association
  with Benjamini--Hochberg or Benjamini--Yekutieli correction (BY default),
  and the knockoff filter in fixed-X and model-X forms. Default holdout
  split separates search from selection; `fdr_controlled_` states whether
  the guarantee applies; `selection_report_` exposes per-candidate p- and
  q-values.
- Estimators `BeamFeatTransformer`, `BeamFeatRegressor`,
  `BeamFeatClassifier`, passing scikit-learn's estimator-conformance
  checks. Parsimony (forward selection within the screened set) on by
  default; `on_no_discoveries` in `"empty"` (default), `"fallback"`, and
  `"raise"` modes; a post-fit holdout check (`DegenerateFitWarning`)
  separating association from generalisation; readable `equation()` on
  both supervised estimators (log-odds form for classification).
- Optional dimensional analysis via pint quantities or unit strings,
  enforced at expression construction.
- Committed benchmark harness, physics-equation panel, calibration
  studies, and a vendored independent comparison study.
- Supported: Python >= 3.10, numpy >= 1.26, scikit-learn 1.6 through 1.9
  verified, no upper version pins.