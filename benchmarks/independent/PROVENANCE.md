# Provenance

What in this study is reproducible, and what is not. Read this before citing
any figure from it.

## What was run

360 fits: nine datasets, eight methods, five fixed 75/25 splits, executed in
one pass on one machine in the pinned environment of `requirements.txt`
(Python 3.11, numpy 1.26.4, scikit-learn 1.7.2, autofeat 2.1.3, openfe 0.0.12,
featuretools 1.31.0, lightgbm 4.7.0). Raw per-fit records are in
`results_as_reported/`; `independent_benchmark_results.csv` is their
aggregation. `beamfeat_benchmark.ipynb` reproduces the analysis in about two
minutes and the fits in 45 to 70.

An earlier version of this study was assembled from several partial runs on an
unknown machine and contained one value transcribed by hand rather than
recorded. It has been replaced entirely by the single-pass run described here.

## What reproduces

Every method except `autofeat`. Across four executions of this comparison,
`beamfeat`, `beamfeat_ridge`, `ridge_raw`, `rf_raw`, `lgbm_raw`, `openfe` and
`featuretools` returned the same means, worst cases and paired-test statistics
to four decimals. `beamfeat` is deterministic given a seed and reproduced
digit-for-digit across two machines running different Python, numpy and
scikit-learn versions.

Across releases the figures move slightly. The archived run predates 0.3.0:
it was made on 28 July 2026, between the 0.1.0 and 0.1.1 releases, when the
estimator's internal ridge penalty was fixed at `alpha=1`; from 0.3.0 it is
chosen by cross-validation. Rerun with 0.4.2, the release the paper
describes, reading the wine file as the archived run did (1,598 rows; see
*Protocol limits*), `beamfeat_ridge` matches 37 of the 45 archived fits to
four decimals, with the same worst fit, 0.3546, and a mean of 0.8034 against
0.8033. `beamfeat`, whose penalty changed, matches 21: mean 0.8034 against
0.8026, worst fit 0.3544 against 0.3553. No fit moves by more than 0.016, and
the differences come from that change and from fixes between the two
releases. Read with all 1,599 wine rows, the wine fits move with their splits
and the worst fit becomes 0.315.

## What does not

`autofeat`. It exposes no `random_state`, and its noise-injection screen draws
decoy features from the global NumPy generator before its own internal seeding
applies, so every process starts from different entropy. The decoys set the
bar each candidate feature must clear, so a different draw changes which
features survive.

Six runs of one identical split of Friedman #1, at the library defaults in
this environment, each a separate process (`autofeat_repeatability.json`):

| run | threading | R² | selected features |
|---|---|---|---|
| 1 | default | +0.952 | 17 |
| 2 | default | −100.96 | 26 |
| 3 | default | −105.31 | 19 |
| 4 | pinned to 1 | −109.79 | 21 |
| 5 | pinned to 1 | +0.950 | 18 |
| 6 | pinned to 1 | +0.939 | 20 |

Calling `np.random.seed()` before fitting does not control it, and pinning
`OMP_NUM_THREADS`, `NUMBA_NUM_THREADS` and `MKL_NUM_THREADS` to 1 does not
either. The effect appears where selection is marginal; on problems with an
unambiguous signal, repeated processes agree exactly.

The same instability shows at study level. Four executions of the full
nine-dataset comparison returned `autofeat` mean R² of 0.746, −1.694, 0.754
and −1.561, with worst single fits from −2.28 to −108.9.

**Consequence for citation.** `autofeat`'s row in `REPORT.md` is one draw from
that distribution, not a measurement. The shipped run gives mean −1.561 and a
worst fit of −103.245 (Friedman #1 split 0; its other four splits on that
dataset fall between 0.938 and 0.955). A rerun will give something else. Any
single `autofeat` figure here is one draw; the instability is the citable
finding.

## Non-finite test cells

The archived records were produced by a harness that filled non-finite
engineered cells with the training-column median on the training rows but
with 0 on the test rows. `bench.py` now fills both from the training medians.
The archive is left as run; the change touches only the `featuretools` and
`OpenFE` rows, whose ratio features divide by columns that contain zeros.

Rerunning those rows on the archived splits with each fill:

| fits | zero fill (archived) | training-median fill |
|---|---|---|
| `featuretools`, Boston (ratios over `chas`, 0 on 471 of 506 rows), splits 0-4 | −2.368, −15.680, −57.320, −27.085, −25.684 | 0.564, 0.890, 0.816, 0.870, 0.869 |
| `featuretools`, Concrete, splits 0-4 | 0.743, 0.819, 0.841, 0.871, 0.816 | 0.797, 0.829, 0.855, 0.880, 0.831 |
| `OpenFE`, Concrete, splits 1 and 4 | 0.7422, 0.7975 | 0.7420, 0.7962 |

On wine, measured on the current 1,599-row loader and so against a rerun
rather than the archive, the fill moves `featuretools` by at most 0.0003 and
`OpenFE` split 1 by 0.0022. Every other `featuretools` and `OpenFE` fit is
unchanged to four decimals. Read with the median fill, `featuretools`
has a mean R² of about 0.461 rather than −2.478, a worst fit of −12.882
(Friedman #1 split 2, which has no non-finite cells and is unaffected)
rather than −57.320, and one fit below zero rather than six; its average rank
and the paired tests against it in `REPORT.md` move with it. Every other
method's row, and every conclusion about `beamfeat`, is unaffected.

The TMLR paper's comparison-study table reports the archived zero-fill run
for `featuretools` (mean R² −2.48, worst −57.3, six fits below zero). With
the training-median fill these read 0.461, −12.882 and one. `beamfeat`
still beats it in the paired test: Wilcoxon p = 0.0005 (paper: p < 0.0001),
median difference +0.0046 rather than +0.0080, `beamfeat` ahead on 32 of 45
fits rather than 33.

## Data

`data/` holds the three CSVs that are not bundled with scikit-learn, with
MD5 checksums in `CHECKSUMS.md5`. Verify from this directory:

```bash
md5sum -c data/CHECKSUMS.md5
```

Boston housing is included for comparability with autofeat's published table,
despite its removal from scikit-learn over documented ethical concerns. The
repository's own real-data panel contains no contested dataset.

## Protocol limits

- Nine datasets leaves the Friedman omnibus underpowered; the paired tests
  carry the inference.
- Five splits per dataset.
- Default or paper-recommended settings only, with no hyperparameter search.
- `OpenFE` is run into a linear model rather than the gradient-boosted setting
  it is designed for, which understates it.
- The wine dataset here is the red-only variant (1,598 rows as archived, 11
  columns); autofeat's published table used red and white combined
  (6497 x 12). The file ships headerless and the loader formerly consumed
  UCI's first observation as a header; it now reads all 1,599 rows, so fresh
  runs will differ by one row from every archived result.
- `recovered` measures column membership, not symbolic form: it asks whether
  some returned feature references all of the generating columns, not whether
  the operators combining them are right. See the docstring in `bench.py`.
