# Provenance

What `REPORT.md` measured, on what, and how to repeat it.

## Settings and items

Four settings of the estimators, each fitted on the same 297 items with the
same seeds:

| setting | parameters |
| :-- | :-- |
| default | none |
| `parsimony_alpha` | `parsimony_alpha=0.05` |
| `input_test` | `input_test=True` |
| both | `parsimony_alpha=0.05, input_test=True` |

| study | items | measures |
| :-- | :-- | :-- |
| `sweep` | `y = x0/x1` among 12 columns, inputs uniform on [1, 3], Gaussian noise at 0.1, 0.3, 0.5 and 1.0 times the signal's standard deviation, 1,000 rows, 10 seeds | terms, terms reading a column other than `x0` and `x1`, R² against the noiseless target on 2,000 new rows |
| `suite` | the 25 datasets of `benchmarks/run_benchmarks.py` (18 synthetic, 7 real) with the splits and seeds of `benchmarks/release_comparison/`: 5 splits, 70/30 | held-out R² (accuracy for breast_cancer), terms, recovery and false-feature rate where the formula is known |
| `highdim` | communities, superconduct (5,000 rows), tecator, eyedata and riboflavin, as in `benchmarks/release_comparison/`: 5 splits, 70/30 | held-out R², terms |
| `paper_suite` | the 25 datasets of `run_benchmarks.py` with its settings (depth 2, beam 40, q = 0.1, one 70/30 split, seed 0), as `release_comparison/archive_check.py` fits them | as `suite` |
| `paper_feynman` | the 12 laws of `benchmarks/feynman_panel.py` with its settings (depth 3, beam 40) | solved (held-out R² above 0.999), exact form |
| `paper_comparison` | the 9 datasets of `benchmarks/independent/bench.py`, its five 75/25 splits, the estimator as shipped | held-out R², terms |
| `paper_highdim` | the 5 datasets of `benchmarks/additional_experiments/bench.py` that the repository stores, its five 75/25 splits | held-out R², terms |

Every record also holds the fit's wall time, `fdr_controlled_` and the
printed formulas.

## Environment

beamfeat 0.7.0, built from the commit that bracketed a divisor spelled
`1/(...)` in formula text, which is the code of the release in every respect
that affects a fit. The stack is that of `benchmarks/release_comparison/`:
Python 3.12, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.0,
joblib 1.6.0, sympy 1.14.0. Machine: Linux, 22 cores, 15 GB of memory. The
four settings ran in parallel, one process each, with one BLAS thread per
process.

The suite's four seaborn datasets are read from `benchmarks/data/`, the cache
`run_benchmarks.py` fills on its first run; the other data are in the
repository.

## To rerun

```bash
uv venv --python 3.12 venv
uv pip install --python venv -e . numpy==2.5.4 scipy==1.18.1 scikit-learn==1.9.1 \
  pandas==3.1.0 joblib==1.6.0 sympy==1.14.0
cd benchmarks/three_settings
for s in default parsimony_alpha input_test both; do
  OPENBLAS_NUM_THREADS=1 ../../venv/bin/python run.py $s &
done
wait
../../venv/bin/python run.py rescore
../../venv/bin/python report.py > REPORT.md
```

A rerun with the same stack reproduces every fit, since every one is seeded.
Fit times depend on the machine; their ratios between settings carry over
better than the seconds.

## Unscored splits

When a formula is undefined on some test rows (the logarithm of a negative
value, say), the regressor's `score` refuses, and the record's score is NaN.
`run.py rescore` refits all four settings on each such split and scores them
on the rows every setting predicts, as `release_comparison/rescore.py` does;
`results/rescore.json` holds those scores, and `REPORT.md` names the splits.

## Against the release comparison

The default and `parsimony_alpha` fits of the `suite` and `highdim` studies
are the same fits as the release comparison's 0.7.0 run, with the same
harness, splits, seeds and stack. The last section of `REPORT.md` checks
this fit by fit.
