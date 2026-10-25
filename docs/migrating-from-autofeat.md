# Migrating from autofeat

*Examined against autofeat 2.1.3, the current release as of July 2026,
in the pinned comparison environment (scikit-learn 1.7.2); the
version-compatibility failures below were reproduced on scikit-learn 1.8
and 1.9.*

`beamfeat`'s design began from a review of
[autofeat](https://github.com/cod3licious/autofeat) (Horn et al., 2019) and
keeps its central idea — compact symbolic features feeding a linear model.
The APIs are close enough that most code ports in a few lines. This page is
a reference for doing so; it is not a claim that one tool supersedes the
other, and autofeat's exhaustive expansion still finds structure a beam
search prunes (see the *Where autofeat does better* section below).

## Version compatibility

If you arrived here from an error, this is likely the one:

```text
TypeError: check_array() got an unexpected keyword argument 'force_all_finite'
```

autofeat 2.1.3 raises this when transforming or predicting on unseen data
after engineering at least one feature, on **scikit-learn ≥ 1.8** — the
argument was renamed `ensure_all_finite` in 1.6 and removed in 1.8. `fit`
still succeeds, so the failure appears only when you apply the fitted model
to a test set. Its declared pins (`numpy<2.0`, `pandas<3.0`) will also
downgrade a current environment on installation.

`beamfeat` declares no upper version bounds and its test suite runs
unpatched from scikit-learn 1.6 through 1.9.

## Class mapping

| autofeat | beamfeat |
|---|---|
| `AutoFeatRegressor` | `BeamFeatRegressor` |
| `AutoFeatClassifier` | `BeamFeatClassifier` |
| `AutoFeatModel`, `FeatureSelector` | `BeamFeatTransformer` |

## Parameter mapping

| autofeat | beamfeat | notes |
|---|---|---|
| `feateng_steps=2` | `max_depth=2` | same meaning: composition depth |
| `featsel_runs=5` | *(no equivalent)* | selection is a single FDR-controlled test; tune `target_fdr` (default 0.1) instead of a repetition count |
| `transformations=('1/','exp','log','abs','sqrt','^2','^3')` | `unary_ops=("reciprocal","exp","log","abs","sqrt","square","cube")` | all seven exist; `beamfeat`'s default omits `exp` and `cube` |
| *(no equivalent)* | `binary_ops=("mul","div","add","sub")` | binary operators are configurable |
| `units={...}`, `apply_pi_theorem=True` | `units={...}` | accepts pint quantities or strings (`"kg"`, `"m / s**2"`); enforced at expression construction rather than as a separate theorem pass |
| `categorical_cols=[...]` | *(not supported)* | compose explicitly: `make_pipeline(ColumnTransformer(...), BeamFeatRegressor(...))` |
| `feateng_cols=[...]` | *(not supported)* | subset `X` before fitting |
| `n_jobs=1` | `n_jobs=1` | parallelises the search over cores; the fitted result is the same for every value |
| `verbose=0` | `verbose=0` | same |

## Fitted attributes

| autofeat | beamfeat |
|---|---|
| `new_feat_cols_` | `feature_formulas_`, or `formulas()` |
| `good_cols_` | `formulas()` (post-selection) |
| `prediction_model_` | `model_` |
| *(no equivalent)* | `fdr_controlled_` — whether the returned features carry the FDR guarantee |
| *(no equivalent)* | `selection_report_` — per-candidate p-values, q-values, and screening decisions |
| *(no equivalent)* | `equation()` — the fitted model as a readable equation |
| *(no equivalent)* | `equations()` — the equation at every size along the forward path |
| *(no equivalent)* | `latex()` — the fitted equation typeset for a document |

## Worked example

```python
# autofeat
from autofeat import AutoFeatRegressor
model = AutoFeatRegressor(feateng_steps=2, featsel_runs=5, verbose=0)
model.fit(X_train, y_train)
print(model.new_feat_cols_)
predictions = model.predict(X_test)

# beamfeat
from beamfeat import BeamFeatRegressor
model = BeamFeatRegressor(max_depth=2, target_fdr=0.1, random_state=0)
model.fit(X_train, y_train)
print(model.formulas())
print(model.equation())          # the fitted model, readable
print(model.fdr_controlled_)     # check before treating features as vetted
predictions = model.predict(X_test)
```

## Where autofeat does better

Exhaustive expansion does not need to *see* a signal marginally in order to
generate the right basis. On targets with a strong non-monotone interaction —
Friedman #1 is the standard example — autofeat's brute-force expansion finds
structure that correlation-guided beam search prunes. On four of five splits
of Friedman #1 it reaches R² 0.94 to 0.96 where `beamfeat` reaches 0.72 to
0.79; on the fifth it returns −103. Take the win on the typical split as real
and the mean as unusable, which is the shape of the trade: `beamfeat` accepts
a lower ceiling as the price of a guarantee and a floor. Quantified in
[Statistical guarantees](guarantees.md).

## Behavioural differences to expect

- **Fewer features.** Selection is a multiple hypothesis test rather than a
  noise-threshold heuristic: a feature is selected when the test rejects its
  null, and a parsimony step then retains a compact subset of the selected
  features. Expect single-digit feature counts where autofeat returns tens.
  `equation()` records how many of the certified terms it printed, and
  `parsimony=None` returns the whole certified set if you want it.
- **A guarantee, and an honest empty result.** If the screen rejects no
  null, `beamfeat` returns *no* constructed features, warns, and sets
  `fdr_controlled_ = False` rather than returning unvetted candidates. Check
  that flag.
- **Determinism.** `random_state` makes the whole pipeline reproducible,
  whatever `n_jobs` is set to. autofeat exposes none, and cannot be made reproducible from outside: its
  noise-injection screen draws decoy features from the global NumPy generator
  before its own internal seeding applies, so each process starts from
  different entropy. Six runs of one identical split returned R² from +0.952
  to −109.8.
- **Constants inside expressions.** autofeat fits none. `beamfeat` fits one
  constant inside an expression by default, such as
  `exp(-3.2 x)` or `1/(x + a)` (`fitted_ops`; `fitted_ops=()` turns it off).
  If a law needs several constants fitted jointly, a symbolic regressor such
  as PySR is the right instrument.
