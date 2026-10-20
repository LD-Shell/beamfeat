# Independent comparison study

A comparison of `beamfeat` against `autofeat`, `OpenFE`,
`featuretools`, and raw-feature baselines (ridge, random forest, LightGBM)
across nine datasets and five seeds, with average-rank and Friedman-test
analysis.

| file | contents |
|---|---|
| `REPORT.md` | the full write-up: protocol, results, statistics, conclusions |
| `PROVENANCE.md` | what is reproducible and what is not — read before citing |
| `beamfeat_benchmark.ipynb` | the study as a runnable notebook (`FULL_RUN = False` analyses the archived results in minutes; `True` regenerates them) |
| `bench.py`, `aggregate.py` | the datasets, method wrappers, fit loop and aggregation the notebook calls |
| `setup_env.sh` | one-shot environment setup: pinned dependencies, `beamfeat` from this tree, JupyterLab, and a registered kernel |
| `requirements.txt`, `patch_openfe.py` | pinned environment and the OpenFE source patch it requires |
| `data/` | inputs, with MD5 checksums |
| `results_as_reported/` | the published study, 360 per-fit records. Read-only: nothing writes here |
| `results_fresh_run/` | anything a regeneration produces. Created on demand, not committed |

## Setting up

`autofeat` cannot run on a current stack: it caps `scikit-learn` below 1.8 and,
through `numba`, `numpy` at 2.2 or lower. The study therefore needs its own
environment, which `setup_env.sh` builds in one step.

```bash
conda create -n af315 python=3.11 -y
conda activate af315
bash benchmarks/independent/setup_env.sh
```

That installs the pinned dependencies, `beamfeat` from this repository as an
editable install, and JupyterLab; registers a Jupyter kernel; and finishes by
printing the resolved versions and running an `autofeat` smoke test. It refuses
to install into a base or system interpreter, since the pins would downgrade a
general-purpose environment. A plain `python -m venv` works equally well.

## Two beamfeat rows

`beamfeat` is the estimator as shipped, predicting through its own internal
ridge. `beamfeat_ridge` is the same search used as a transformer feeding the
shared `RidgeCV`, exactly as `autofeat`, `featuretools` and `OpenFE` are run.
Comparing the two shows whether the downstream model, rather than the
constructed features, accounts for any difference.

## Two results directories

Reruns are kept apart from the published study so the two can never be mixed
or overwritten:

```
results_as_reported/    the study as published (360 fits) -- read-only
results_fresh_run/      whatever a regeneration produces
```

`FULL_RUN = False` in the notebook reads the archive; `FULL_RUN = True` writes
to and then reads from `results_fresh_run/`. `aggregate.py` takes the
directory as an argument:

```bash
python aggregate.py                     # the published study (default)
python aggregate.py results_fresh_run   # your regeneration
```

Each writes `independent_benchmark_results.csv` beside the files it read, so
comparing a rerun against the published numbers is a diff of two CSVs. To
recover a single missing cell rather than rerunning everything, point `bench.py`
at one dataset:

```bash
DATASETS=friedman1 python bench.py synthetic autofeat 5 results_fresh_run/results_syn_af3.json
```

## Headline figures

Mean out-of-sample R² across the nine datasets, and the worst of the 45
individual fits per method:

| method | mean R² | worst fit | mean seconds | mean new features |
|---|---|---|---|---|
| random forest (raw) | 0.810 | 0.247 | 0.34 | — |
| **beamfeat** | **0.803** | **0.355** | **0.52** | 9.2 |
| beamfeat → ridge | 0.803 | 0.355 | 0.55 | 9.2 |
| LightGBM (raw) | 0.798 | 0.117 | 0.46 | — |
| OpenFE | 0.736 | 0.360 | 3.82 | 7.3 |
| ridge (raw) | 0.704 | 0.353 | 0.01 | — |
| autofeat | −1.561 | −103.245 | 32.32 | 16.8 |
| featuretools | −2.478 | −57.320 | 0.26 | 104.7 |

`beamfeat` reported `fdr_controlled_ = True` on 45/45 fits, recovered the
generating columns on every ground-truth problem, and produced no fit below
R² 0.355. Two methods did: `autofeat` on two fits and `featuretools` on six.
`beamfeat`'s Friedman #1 result here, 0.746, independently reproduces the
0.744 measured by this repository's own harness.

## Caveats that travel with these numbers

Documented in full in `PROVENANCE.md`; in brief:

1. **`autofeat`'s row is one draw, not a measurement.** It exposes no
   `random_state` and draws its decoy features from the global NumPy generator
   before any internal seeding applies, so each process starts from different
   entropy. Four executions of this study returned mean R² of 0.746, −1.694,
   0.754 and −1.561, with worst fits from −2.28 to −108.9; every other method
   reproduced to four decimals across the same runs. See
   `autofeat_repeatability.json`.
2. **Dataset-overlap precision.** The report's comparability note is
   slightly imprecise against autofeat's Table 1: diabetes also overlaps
   (four of this study's four real datasets appear there), and the wine
   dataset here is the red-only variant (1598×11) where autofeat evaluated
   red and white combined (6497×12). Boston housing is included for that
   comparability despite its removal from scikit-learn over documented
   ethical concerns; the repository's own seven-dataset real panel contains
   no contested data.
3. **`featuretools`' negative R² on Boston is a harness artefact.** The
   archived run filled non-finite test cells with 0 instead of the training
   median; with the median fill its five Boston fits score 0.56 to 0.89, not
   −57.32 to −2.37. The archive is unchanged; see `PROVENANCE.md`.
4. **The protocol favours linear consumers.** Every construction tool feeds
   the same ridge model, which understates `OpenFE` in its intended
   gradient-boosted setting and `featuretools` outside its relational use
   case. `beamfeat` does not win real-tabular prediction against tuned tree
   models on individual datasets (Concrete: 0.845 against LightGBM's 0.932).
