# Installation

Three levels, depending on what you need. Most people only need the first.

## Use the library

```bash
pip install beamfeat            # numpy + scikit-learn only
pip install "beamfeat[units]"   # + pint, for dimensional analysis
pip install "beamfeat[all]"     # + sympy display and test dependencies
```

Python 3.10 or newer. There are no upper version pins. The test suite is run
against the oldest supported stack (numpy 1.26, scikit-learn 1.6) and against
current releases on Python 3.10 to 3.15: 456 tests pass from scikit-learn 1.6.1
with numpy 1.26 through scikit-learn 1.9.1 with numpy 2.5.

Nothing below is needed to use `beamfeat`.

## Run the test suite

```bash
git clone https://github.com/LD-Shell/beamfeat
cd beamfeat
pip install -e ".[all]"
pytest
```

456 tests, about a minute, including scikit-learn's `check_estimator`
conformance suite. `pytest --cov=beamfeat` reports 95% statement coverage.

The `units` extra matters here: without `pint`, every unit-aware test skips
(listed with its reason by `pytest -rs`) and coverage reads about 93%.
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
| `benchmarks/reproduce_all.ipynb` | regenerates every figure in the paper and README, and asserts the load-bearing ones | 20-30 min |
| `benchmarks/independent/beamfeat_benchmark.ipynb` | the comparison against autofeat, OpenFE, featuretools and raw baselines | 2 min from archived results, 45-70 min with `FULL_RUN = True` |

`reproduce_all.ipynb` also builds a throwaway virtual environment with the
newest numpy and scikit-learn and runs the suite there, so the pinned kernel
does not become the only evidence that the package works on current releases.
