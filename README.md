# beamfeat

[![CI](https://github.com/LD-Shell/beamfeat/actions/workflows/ci.yml/badge.svg)](https://github.com/LD-Shell/beamfeat/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-995%20passing-brightgreen)](https://github.com/LD-Shell/beamfeat/tree/main/tests)
[![PyPI](https://img.shields.io/pypi/v/beamfeat)](https://pypi.org/project/beamfeat/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.15-blue)](https://github.com/LD-Shell/beamfeat)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/LD-Shell/beamfeat/blob/main/LICENSE)
[![TMLR](https://img.shields.io/badge/TMLR-2026-blue)](https://openreview.net/forum?id=UibXoy4h3u)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21619567.svg)](https://doi.org/10.5281/zenodo.21619567)
[![Docs](https://img.shields.io/badge/docs-ld--shell.github.io-blue)](https://ld-shell.github.io/beamfeat/)

beamfeat builds readable formulas from the columns of a table, such as
`(x0 / x2) * x1` or `log(a) * log(b)`, tests each against the null
hypothesis that it is independent of the target, and keeps only those whose
null is rejected, with a stated false-discovery-rate guarantee. The result is
a short equation in your own column names, from a scikit-learn estimator you
can fit, inspect and predict with. The method is described in a
[TMLR 2026 paper](https://openreview.net/forum?id=UibXoy4h3u).

**New in 0.7.0.** The search now keeps a few beam places for building
blocks whose size tracks the target's even when they do not correlate with it
(`magnitude_slots`, on by default for regression): every `x0*x1*x2` stress law
is recovered, the held-out score of 11 of 12 real datasets stays within 0.005
of 0.6.1's, and fits take a fifth longer. The conditional re-test after the
forward step now asks for `parsimony_alpha`, and the code is split into
smaller modules, mapped in
[`ARCHITECTURE.md`](https://github.com/LD-Shell/beamfeat/blob/main/ARCHITECTURE.md).
The documentation has a new
[Choosing a setting](https://ld-shell.github.io/beamfeat/settings/) page:
when to keep the default, and when to ask for a short equation or for every
column to be tested.
How 0.4.2 (the paper's release) through 0.7.0 compare on accuracy, speed and
memory:
[release comparison](https://github.com/LD-Shell/beamfeat/blob/main/benchmarks/release_comparison/REPORT.md).

## An example: fat content from a spectrum

![Held-out R² on tecator over five splits: beamfeat 0.991 from three or four formulas; ridge 0.957, LightGBM 0.756 and random forest 0.717 on all 100 channels](https://raw.githubusercontent.com/LD-Shell/beamfeat/main/docs/assets/tecator.png)

In the tecator data, the task is to predict the fat content of meat samples
from 100 near-infrared absorbance channels. Over five random 75/25 splits,
beamfeat's equation uses three or four formulas, mostly ratios of channel
differences such as `((x35 - x42) / (x37 - x39))`, and reaches a mean
held-out R² of 0.991. On all 100 channels, ridge regression reaches 0.957, a
random forest 0.717 and LightGBM 0.756. A random forest given the formulas in
place of the raw channels goes from 0.739 to 0.992, on three of the five
splits. Against the methods of chemometrics, compared since the paper,
beamfeat's error is 44 to 59% of that of partial least squares (R² 0.951 to
0.971), and the error of kernel methods on second-derivative spectra (R²
0.996 to 0.998) is 48 to 62% of beamfeat's, from every channel and with no
formula to read.

The accuracy is stable from split to split; which ratios are picked is not.
Neighbouring channels correlate at 0.99988, so many ratios are nearly
interchangeable, and each split settles on its own. `stability_report`
measures this on any data: it refits on many random splits and counts the
distinct equations.

The numbers are the paper's (Table 6 and its appendices), from the archived
run in [`benchmarks/additional_experiments/`](https://github.com/LD-Shell/beamfeat/tree/main/benchmarks/additional_experiments),
except the chemometrics comparison, from
[`benchmarks/tecator_chemometrics/`](https://github.com/LD-Shell/beamfeat/tree/main/benchmarks/tecator_chemometrics);
[`benchmarks/figures/tecator.py`](https://github.com/LD-Shell/beamfeat/blob/main/benchmarks/figures/tecator.py)
draws the plot. The [Benchmarks](https://ld-shell.github.io/beamfeat/benchmarks/)
page has all nine datasets of that study, including those where beamfeat
trails.

## Install

```bash
pip install beamfeat              # core: numpy, scikit-learn, joblib
pip install "beamfeat[units]"     # + pint, for units
pip install "beamfeat[display]"   # + sympy, for to_sympy() and latex()
pip install "beamfeat[all]"       # + both, pandas and the test dependencies
```

Python 3.10 or newer, with no upper version pins: the 995 tests pass on
Python 3.10 to 3.15 with the newest releases each supports, and the suite runs
on the oldest supported stack too.
[Installation](https://ld-shell.github.io/beamfeat/installation/) lists the
versions, and how to run the tests or reproduce the benchmarks.

It fits like any scikit-learn estimator:

```python
from beamfeat import BeamFeatRegressor

model = BeamFeatRegressor().fit(X, y)

model.formulas()        # ['((x0 / x2) * x1)', ...]
model.equation()        # 'y = 0.9847*((x0 / x2) * x1) + ...'
model.fdr_controlled_   # True: the features carry the stated FDR guarantee
model.predict(X_new)
```

## Three settings

```python
BeamFeatRegressor()                        # the equation that predicts best
BeamFeatRegressor(parsimony_alpha=0.05)    # a short equation, to read or report
BeamFeatRegressor(input_test=True)         # every column inside each formula tested
```

On noisy data the default's equation can carry extra terms. On `y = x0/x1`
among ten unrelated columns, with noise as large as the signal, it printed a
median of 9.5 terms, most of them reading an unrelated column, and
`parsimony_alpha=0.05` printed the one true term in every fit. On real data,
which rarely follow one compact formula, `parsimony_alpha` gives up some
held-out score, and `input_test` takes 2 to 7 times as long.
[Choosing a setting](https://ld-shell.github.io/beamfeat/settings/) has the
measurements.

## What you get

- **Readable formulas.** `formulas()` lists the features, and `equation()`
  prints the fitted model in your column names.
- **The equation at every size.** `equations()` lists one equation per
  number of terms, with the R² each term adds.
- **LaTeX and SymPy.** `latex()` typesets the equation, and `to_sympy()`
  hands the features to SymPy, and from there to NumPy, JAX or PyTorch.
- **Units.** Give the columns units, as strings such as `"kg"` or as pint
  quantities, and dimensionally invalid expressions are never built.
- **A guarantee code can check.** `fdr_controlled_` says whether the
  features carry the guarantee, and `fdr_scope_` whether it covers the
  printed equation or the larger certified set the equation was drawn from.
- **Fitted constants.** Expressions can carry one constant fitted inside
  them, such as `exp(a*x)` and `1/(x + a)`, which recovers laws like
  `exp(-x^2/2)` exactly. On by default; `fitted_ops=()` turns it off.
- **Hidden building blocks.** `magnitude_slots` keeps a few beam places for
  parts whose size tracks the target's even when they do not correlate with
  it, such as `x0` and `x1` in `y = x0*x1` with centred inputs. On by default
  for regression since 0.7.0; `magnitude_slots=0` turns it off.
- **Every column in a formula tested, opt in.** `input_test=True` tests each
  column inside each formula, and drops a formula when the test fails to
  reject the null for one of its columns, such as `x6` in
  `(x0 * x1) / (x0 + x6)`.
- **Stability.** `stability_report` refits on many splits and shows whether
  the equation is the only good one or one of many.

## The guarantee

The search runs on half of the training rows. On the other half, each formula
it returns is tested against the null hypothesis that it is independent of
the target, with a permutation test and the Benjamini–Yekutieli correction.
The formulas whose nulls are rejected are the certified set, and on average
at most 10% of them are false discoveries (`target_fdr`). That is an average
over repeated fits, not a promise about any one fit, and it covers the
certified set, of which the printed equation is by default a compact subset.
A formula is a false discovery only when it is independent of the target, so
`(x0*x1)/(x1 + x8)` is a true one when `y = x0*x1`, however unrelated `x8`
is; `input_test=True` tests each column inside each formula.
If no null is rejected, the model uses no formulas and warns.
[Statistical guarantees](https://ld-shell.github.io/beamfeat/guarantees/)
gives each procedure's assumptions and what was measured.

## Documentation

- [Quickstart](https://ld-shell.github.io/beamfeat/quickstart/): a first fit
  and how to read it.
- [Choosing a setting](https://ld-shell.github.io/beamfeat/settings/): the
  default, a short equation, or every column tested.
- [User guide](https://ld-shell.github.io/beamfeat/guide/): reading a fit,
  the printed equation, the search options, and data and prediction.
- [Tutorials](https://ld-shell.github.io/beamfeat/notebooks/01_getting_started/):
  four notebooks, from a first fit to reading one in detail.
- [Statistical guarantees](https://ld-shell.github.io/beamfeat/guarantees/):
  what the guarantee covers, its assumptions, and what was measured.
- [Benchmarks](https://ld-shell.github.io/beamfeat/benchmarks/): every
  measured result, with the scripts and archived runs behind it.
- [Migrating from autofeat](https://ld-shell.github.io/beamfeat/migrating-from-autofeat/):
  the API mapping and the differences to expect.
- [API reference](https://ld-shell.github.io/beamfeat/api/) and
  [changelog](https://ld-shell.github.io/beamfeat/changelog/).

## Citing

If you use beamfeat, please cite the paper:

```bibtex
@article{
daramola2026symbolic,
title={Symbolic feature construction with false discovery rate control},
author={Olanrewaju M. Daramola},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=UibXoy4h3u},
note={}
}
```

The Zenodo DOIs archive the code, not the paper:
[10.5281/zenodo.23417132](https://doi.org/10.5281/zenodo.23417132) is 0.4.2,
the version the paper describes, with the run records behind its results, and
[10.5281/zenodo.21619567](https://doi.org/10.5281/zenodo.21619567) collects
every release. [`CITATION.cff`](https://github.com/LD-Shell/beamfeat/blob/main/CITATION.cff)
has the same details in machine-readable form.

Later releases correct two of the paper's results, detailed on the
[Benchmarks](https://ld-shell.github.io/beamfeat/benchmarks/#reading-the-paper-today)
page: a harness bug understated featuretools, and fixing a search bug takes
the datasets where beamfeat leads both tree ensembles from five to four.

If you use the selection procedures, please also cite the underlying
statistics: [Benjamini & Hochberg (1995)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x),
[Benjamini & Yekutieli (2001)](https://doi.org/10.1214/aos/1013699998),
[Barber & Candès (2015)](https://doi.org/10.1214/15-AOS1337),
[Candès et al. (2018)](https://doi.org/10.1111/rssb.12265), and
[Phipson & Smyth (2010)](https://doi.org/10.2202/1544-6115.1585).

## License

[MIT](https://github.com/LD-Shell/beamfeat/blob/main/LICENSE).
