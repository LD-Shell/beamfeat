---
description: "Install beamfeat with pip, with optional extras for units and display, run its test suite, and set up the environment that reproduces the benchmarks."
---

# Installation

Three levels, depending on what you need. Most people only need the first.

## Use the library

```bash
pip install beamfeat              # numpy, scikit-learn, joblib
pip install "beamfeat[units]"     # + pint, for dimensional analysis
pip install "beamfeat[display]"   # + sympy, for to_sympy() and latex()
pip install "beamfeat[all]"       # + both, pandas and the test dependencies
```

Python 3.10 or newer. There are no upper version pins. The test suite is run
against the newest releases on Python 3.10 to 3.15, where all 1,079 tests pass
(scikit-learn 1.9.1 from Python 3.11, and 1.7.2, the newest that supports it,
on 3.10; numpy 2.2 to 2.5), and against the oldest supported stack
(scikit-learn 1.6.1, numpy 1.26, joblib 1.2), which leaves pint and pandas
out so that the tests that need them show they skip cleanly without them.
CI repeats this every week, and runs the suite against the upcoming releases
of numpy, SciPy, scikit-learn and pandas too, so that a breaking change
upstream is seen before it ships.

Nothing below is needed to use `beamfeat`.

## Run the test suite

```bash
git clone https://github.com/LD-Shell/beamfeat
cd beamfeat
pip install -e ".[all]"
pytest
```

1,079 tests, a few minutes, including scikit-learn's `check_estimator`
conformance suite. `pytest --cov=beamfeat` reports 97% statement coverage.

The `units` extra matters here: without `pint`, every unit-aware test skips
(listed with its reason by `pytest -rs`) and coverage reads about 95%.
`[all]` includes it.

## Reproduce the benchmarks

Only needed to regenerate the published comparison figures. The competing
tools constrain the environment sharply, so this gets its own:

- `autofeat` 2.1.3 requires scikit-learn below 1.8 and, through `numba`,
  numpy 2.2 or lower
- current `knockpy` releases require numpy 2 or newer, and must therefore be
  installed under the same pins rather than added afterwards

```bash
conda create -n af315 python=3.11 -y
conda activate af315
bash benchmarks/independent/setup_env.sh
```

That installs the pinned dependencies, `beamfeat` from the working tree,
JupyterLab and a Jupyter kernel, then prints the resolved versions and runs
smoke tests for `autofeat` and `knockpy`. It refuses to install into a base or
system interpreter, since the pins would downgrade a general-purpose
environment. A plain `python -m venv` works equally well.

Then, in that kernel:

| notebook | what it does | runtime |
|---|---|---|
| `benchmarks/reproduce_all.ipynb` | regenerates every figure in the paper and on the [Benchmarks](benchmarks.md) page except those of the two studies below, and asserts the load-bearing ones | 20-30 min |
| `benchmarks/independent/beamfeat_benchmark.ipynb` | the comparison against autofeat, OpenFE, featuretools and raw baselines | 2 min from archived results, 45-70 min with `FULL_RUN = True` |

The higher-dimensional studies have their own driver,
`benchmarks/additional_experiments/run_all.ipynb`, and
`python benchmarks/figures/tecator.py` redraws the figure on the home page
from their archived run in a few seconds.

Three studies need none of the competing tools and run wherever beamfeat is
installed: the release comparison (`benchmarks/release_comparison/`), the
three settings (`benchmarks/three_settings/`) and the chemometrics baselines
on tecator (`benchmarks/tecator_chemometrics/`). Each folder's
`PROVENANCE.md` gives its commands and the versions it pinned.

The scripts build their estimators at the current defaults. To run one at
the defaults of the archived runs, set them through the options launcher:

```bash
BEAMFEAT_OPTIONS='{"fitted_ops": [], "magnitude_slots": 0}' \
    python benchmarks/search_stress/with_options.py benchmarks/run_benchmarks.py
```

The guard against quotients near zero has no switch, so an exact
reproduction of a run made before 0.7.0 needs the release that made it.

`reproduce_all.ipynb` also builds a throwaway virtual environment with the
newest numpy and scikit-learn and runs the suite there, so the pinned kernel
does not become the only evidence that the package works on current releases.
