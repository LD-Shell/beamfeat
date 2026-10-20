# beamfeat vs autofeat, OpenFE, featuretools and raw baselines

*360 fits: nine datasets, eight methods, five fixed 75/25 splits, run in one
pass on one machine in the pinned environment of `requirements.txt`. Raw
per-fit results in `results_as_reported/`, aggregated to
`independent_benchmark_results.csv`. Reproduce with
`beamfeat_benchmark.ipynb`.*

## Protocol

The four feature-construction tools hand their features to the same downstream
model, a `RidgeCV` on standardised features, so that comparison isolates the
constructed features rather than the estimator. `beamfeat` appears twice: as
the estimator as shipped, predicting through its own internal ridge, and as
`beamfeat_ridge`, a transformer feeding the shared model. The three `*_raw`
entries are reference points on unengineered columns and are not part of that
controlled comparison.

## Summary

| method | mean R² | worst fit | fits below 0 | features | seconds | recovery |
|---|---|---|---|---|---|---|
| random forest | **0.810** | 0.247 | 0 | — | 0.34 | — |
| beamfeat | 0.803 | **0.355** | 0 | 9.2 | 0.52 | **1.00** |
| beamfeat → ridge | 0.803 | 0.355 | 0 | 9.2 | 0.55 | **1.00** |
| LightGBM | 0.798 | 0.117 | 0 | — | 0.46 | — |
| OpenFE | 0.736 | 0.360 | 0 | 7.3 | 3.82 | — |
| ridge | 0.704 | 0.353 | 0 | — | 0.01 | — |
| autofeat | −1.561 | −103.245 | 2 | 16.8 | 32.32 | 0.67 |
| featuretools | −2.478 | −57.320 | 6 | 104.7 | 0.26 | — |

`recovery` is the fraction of scoreable synthetic fits in which some returned
feature referenced all of the generating columns. It is column recovery, not
symbolic recovery, and is blank for methods that return no inspectable
formulas. `beamfeat`'s `fdr_controlled_` flag was true on 45 of 45 fits.

The `featuretools` row, and its rank and paired test below, include test cells
the harness filled with 0 rather than the training median. Read with the
median fill, `featuretools` has a mean R² of about 0.461 rather than −2.478, a
worst fit of −12.882 rather than −57.320, and one fit below zero rather than
six; `PROVENANCE.md`, under "Non-finite test cells", gives the per-fit figures
and the paired test.

## Statistical comparison

Ranks and the omnibus test use one `beamfeat` entry; two would split the rank
space and make the figures incomparable with a seven-method analysis.

| method | average rank |
|---|---|
| beamfeat | **3.22** |
| random forest | 3.44 |
| LightGBM | 3.44 |
| autofeat | 3.67 |
| featuretools | 4.22 |
| OpenFE | 4.78 |
| ridge | 5.22 |

Friedman: χ² = 6.71, p = 0.348 over 9 datasets and 7 methods. With nine
datasets the omnibus is underpowered and separates nothing; the paired tests
carry the inference.

**Paired Wilcoxon, positive difference favours beamfeat, n = 45:**

| comparison | median Δ | mean Δ | p |
|---|---|---|---|
| vs ridge | +0.0878 | +0.0990 | < 0.0001 |
| vs featuretools | +0.0080 | +3.2805 | < 0.0001 |
| vs OpenFE | +0.0572 | +0.0663 | 0.0001 |
| vs autofeat | −0.0001 | +2.3641 | 0.059 |
| vs random forest | +0.0095 | −0.0076 | 0.599 |
| vs LightGBM | +0.0074 | +0.0047 | 0.858 |
| vs beamfeat → ridge | −0.0000 | −0.0006 | 0.100 |

The autofeat row is the one to read carefully: level on a typical split, with
the mean gap coming entirely from the splits where autofeat fails outright.

## Does the downstream model matter?

Across the 45 paired fits, `beamfeat` and `beamfeat_ridge` differ by a median
absolute 0.00033, a largest absolute 0.016, and only two fits differ by more
than 0.01. Mean, worst case, feature count and recovery are identical. The
result comes from the constructed features, not from the estimator.

## Was the construction worth it?

Ridge on the unengineered columns reaches 0.704. Of the four constructors,
`beamfeat` (9.2 features) reaches 0.803 and `OpenFE` (7.3) reaches 0.736;
`autofeat` (16.8) and `featuretools` (104.7) both end below that anchor. More
constructed features did not buy accuracy here, and for two methods it cost
some.

## Reproducibility of the compared tools

Seven of the eight methods reproduced to four decimals across the four
executions of this study: `beamfeat`, `beamfeat_ridge`, `ridge_raw`,
`rf_raw`, `lgbm_raw`, `openfe` and `featuretools` returned the same means,
worst cases and paired-test statistics on every execution. `autofeat` did
not, and the reason is structural: it exposes no `random_state`, and its
noise-injection screen draws decoy features from the global NumPy generator
before any internal seeding applies, so each process starts from different
entropy. Six runs of one
identical split of Friedman #1 returned R² from +0.952 to −109.8 with 17 to 26
selected features (`autofeat_repeatability.json`); pinning `OMP_NUM_THREADS`,
`NUMBA_NUM_THREADS` and `MKL_NUM_THREADS` to 1 changed nothing.

The same instability is visible at study level. Four executions of this
comparison returned autofeat mean R² of 0.746, −1.694, 0.754 and −1.561, with
worst single fits from −2.28 to −108.9. Any single `autofeat` figure here is one draw; the
instability is the citable finding.

Catastrophic held-out scores are a documented property of the method, not an
artifact of this harness. Horn et al. (2019) report test R² of −12.4 on
diabetes and 0.048 on Boston at three feature-engineering steps, attributing
it to tens of thousands of features fitted against fewer than 500 rows. This
study uses two steps, the library default, at which their Table 2 reports no
such failure — so what is new here is that the failure mode reaches the
default setting, not that it exists.

## Limits

Nine datasets leaves the Friedman test underpowered. Five splits. Default or
paper-recommended settings only, with no hyperparameter search. Real datasets
come from mirrors. PySR and tsfresh are excluded. `OpenFE` is run into a
linear model rather than the gradient-boosted setting it is designed for,
which understates it. The repository's own benchmarks, which include
calibration and robustness suites this study does not re-run, are directionally
consistent with everything measured here.
