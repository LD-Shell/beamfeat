# Provenance

What the release comparison ran, on what, what was added after the first
run and why, and what a rerun reproduces. Read it before citing a figure
from `REPORT.md` or `TABLES.md`.

## Releases and environment

| release | source |
| :-- | :-- |
| 0.4.2 | the release the TMLR paper describes, from its tag |
| 0.5.0 | from its tag |
| 0.6.0 | from its tag |
| 0.6.1 | this repository: the studies ran before the input_test memory fix and the rest after it, which changes no fit; later commits change documentation and the version only |

Each release has its own environment with the same stack: Python 3.12,
numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.0, sympy 1.14.0.
The run installed each tag's `src/` into its environment; the PyPI wheels are
built from the same tags. To rerun:

```bash
cd benchmarks/release_comparison
for v in 0.4.2 0.5.0 0.6.0; do
  uv venv --python 3.12 venvs/$v
  uv pip install --python venvs/$v "beamfeat==$v" numpy==2.5.4 scipy==1.18.1 \
    scikit-learn==1.9.1 pandas==3.1.0 sympy==1.14.0
done
uv venv --python 3.12 venvs/0.6.1
uv pip install --python venvs/0.6.1 -e ../.. numpy==2.5.4 scipy==1.18.1 \
  scikit-learn==1.9.1 pandas==3.1.0 sympy==1.14.0
./run_all.sh
```

Machine: Linux, 22 cores, 15 GB of memory, one BLAS thread per process.

## What is measured

| study | items | measures |
| :-- | :-- | :-- |
| `designs` | the 20 cells of `benchmarks/parsimony_holdout/` (null, planted and proxy designs; n = 300, 1,000), 50 seeds, q = 0.1 | terms printed, marginal and conditional false-discovery proportions, power to print the generating term, R² on 2,000 held-out rows |
| `stress` | the laws of `benchmarks/search_stress/` (`x0*x1`, `x0*x1*x2`, `x0*x1 + x2`) x three input distributions x p = 20, 100; 500 rows, 10% noise, 10 seeds | exact recovery (the Feynman panel's criterion), distractor-only terms, R² against the noiseless law |
| `feynman` | the 12 laws of `benchmarks/feynman_panel.py`, its draw and four more by its recipe; depth 3, beam 40 | solved (held-out R² > 0.999), exact form |
| `suite` | the 25 datasets of `benchmarks/run_benchmarks.py` (18 synthetic, 7 real), 5 splits, 70/30 | held-out R² (accuracy for breast_cancer), terms, recovery and false-feature rate where the formula is known |
| `highdim` | communities, superconduct (5,000 rows), tecator, eyedata, riboflavin, 5 splits, 70/30 | held-out R², terms |
| `junk` | `y = x0*x1` + 25% noise among 10 junk columns, independent or one correlated 0.82 with `x1`, 20 seeds | equations with a junk column inside a term |
| `knockoff` | the designs of `benchmarks/knockoff_s_method.py`, fixed-X, offsets 0 and 1, 50 trials | selector-level FDR and power |
| perf | 8 items, one per study and three large ones, 3 repetitions | wall time, CPU time and peak memory of the fit alone |

Settings: every release at its defaults; 0.6.0 with `fitted_ops=()`,
`magnitude_slots=10`, `input_test=True`, the conditional re-test
(`parsimony_holdout=0.3, certify_test="conditional"`) and
`selection_correction="bh"`; 0.6.1 with `parsimony_alpha=0.05`, alone and with
`input_test=True`. `magnitude_slots` and the conditional re-test are
regression only, so breast_cancer has no entry for them.

## Procedure

`harness.py` runs one (release, study, part) job in the release's
environment. The data come from the repository's benchmark scripts, which
import beamfeat only inside functions, so every release reads the same rows,
splits and seeds. Within an item every setting is fitted on the same data.
Settings that leave the beam search unchanged reuse it through a memo on
`BeamSearch.run`, keyed by its inputs. The first item of every job was
refitted without the memo and matched it in every case. Records are written
one fit at a time, so a stopped job resumes at its next fit.

Each job stops itself if its memory passes 3 GB, recording the fit that was
running, and at most three jobs ran at once. The speed and memory pass runs
every fit in a fresh process, one at a time, settings interleaved and their
order rotated between repetitions. The peak-RSS counter is reset after the
data are loaded.

## Added after the first run, and why

- **0.6.0 `input_test` on California housing** passed 7 GB on every split
  and is recorded as stopped. The cause, a reference cycle in the input test's
  expression evaluation, is fixed in 0.6.1 (CHANGELOG, 0.6.1, Fixed). The fix
  changes no fit: diabetes and friedman1 refitted with it matched the 0.6.0
  records term for term. The 0.6.1 run of the same test (setting
  `input_test` under 0.6.1) stands in for 0.6.0 on California, marked † in
  the tables.
- **Held-out rows without a prediction.** On three splits (communities 0 and
  2, California 4) some fitted expression is undefined on one or two
  held-out rows. From 0.5.0 on, those rows are predicted as NaN and `score`
  refuses. `rescore.py` refits every setting on those splits and scores all of
  them on the rows every setting predicts. 0.4.2 replaces a formula that fails
  on any row with zeros on every row (fixed in 0.5.0, CHANGELOG), which is
  what its scores on those splits show.
- **0.4.2** was added after the main run, with the same harness and data, at
  its defaults. It was timed in a second pass, alternating with 0.5.0 by the
  same protocol; its time and memory ratios are taken against that pass's
  0.5.0.
- **The shipped scripts.** The run used frozen copies of the four benchmark
  scripts above, identical to the repository's files. The shipped `harness.py`
  imports the repository's files instead. Refitted with it, 30 sampled 0.6.1
  fits, 58 for 0.6.0 and 10 for 0.5.0 matched their records exactly.

## Data

The high-dimensional datasets are the repository's copies in
`benchmarks/additional_experiments/data/`. The seaborn datasets (mpg,
diamonds, penguins, tips) are fetched into `benchmarks/data/`, as
`run_benchmarks.py` does. The run used these files (first 16 hex digits of
SHA-256):

| file | sha256 |
| :-- | :-- |
| data/diamonds.csv | 0e7164799468299f |
| data/mpg.csv | c14b8b855ea7ee86 |
| data/penguins.csv | 92da526b754b0a95 |
| data/tips.csv | 22415aaf1e56e675 |
| additional_experiments/data/communities_crime_numeric.csv | ba448939aa714824 |
| additional_experiments/data/superconductivity.csv | 66999cfe1fcaaa97 |
| additional_experiments/data/tecator.csv | f73e2b09675e82a0 |
| additional_experiments/data/eyedata.csv | 4ec902530333ca58 |
| additional_experiments/data/riboflavin.csv | 25856a332be05b22 |

## Against the paper's numbers

This comparison runs every release at its defaults over five splits, so its
figures are not the paper's. `benchmarks/run_benchmarks.py`, for example,
uses a beam of 40 and one split, where the default beam is 50. To check 0.4.2
against the paper-era archive, `archive_check.py` refits it with each
script's own settings:

- **Feynman panel**: all 12 laws reproduce `benchmarks/feynman_results.json`
  exactly (10 solved, 8 in exact form).
- **`run_benchmarks.py`**: 19 of 25 datasets reproduce `benchmarks/results.csv`
  and `results_real.csv` exactly: the 12 core synthetic problems and the 7 real
  datasets. The 6 robustness datasets print the same number of terms. Five
  are within 0.003 of R², and threshold_step is 0.843 against the archived
  0.808, the figure the paper quotes. That archive was regenerated under
  0.3.0, as the archived riboflavin run was (docs, Benchmarks, Reading the
  paper today). This run cannot tell whether the release or the software
  environment causes the difference.

## What a rerun reproduces

With the same releases and stack, the fits: every setting is seeded. A
different numpy, scipy or BLAS can change fits that are close to ties, most
often on riboflavin (49 training rows, 4,088 columns), where 0.4.2 and 0.5.0
already differ for that reason. Timings depend on the machine; the ratios
between releases carry over better than the seconds.

## Files

| file | what |
| :-- | :-- |
| `harness.py` | data, settings and measures; one job per call |
| `perf.py` | the speed and memory pass |
| `rescore.py` | scoring on common rows where some rows have no prediction |
| `report.py` | the tables, `python report.py > TABLES.md` |
| `archive_check.py` | 0.4.2 against the paper-era archive, with each script's own settings |
| `run_all.sh` | the whole run |
| `results/<study>.jsonl.gz` | one record per fit: release, setting, item, measures, printed formulas |
| `results/knockoff_<release>.json` | selector-level trials |
| `results/perf.jsonl`, `results/perf_042.jsonl` | the two speed and memory passes |
| `results/rescore.json` | the re-scored splits |
| `REPORT.md` | what the results say |
| `TABLES.md` | every table |
