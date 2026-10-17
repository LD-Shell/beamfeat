# Changelog

Notable changes to `beamfeat`. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

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