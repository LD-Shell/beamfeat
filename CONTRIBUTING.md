# Contributing to beamfeat

Thank you for considering a contribution.

## Reporting problems and asking questions

Open an issue on the tracker. For suspected statistical problems — a
selector exceeding its nominal FDR, a guarantee claimed where an assumption
fails — please include a runnable snippet and, if possible, a trial count:
FDR statements are about expectations over repetitions, so a single run
exceeding the nominal level is not by itself a bug, but a measured mean over
trials that does is one, and we treat those as the highest-priority class of
issue.

## Development setup

```bash
git clone https://github.com/LD-Shell/beamfeat
cd beamfeat
pip install -e ".[all]"
pytest -q          # full suite
ruff check src tests benchmarks
```

## Standards for changes

- **Statistical claims must be measured, not asserted.** Any change to
  `beamfeat.selection` needs calibration evidence: realised FDR against
  nominal over enough trials, and power on a planted-signal design. The
  existing tests in `tests/test_selection.py` show the expected form.
- **Docstrings state assumptions.** Every guarantee documents the conditions
  it holds under, and the code warns at runtime when they are visibly
  violated.
- **No silent failure paths.** If a procedure cannot deliver its guarantee,
  it must say so (see `fdr_controlled_`).
- **No upper version pins.** New floors must be verified by the
  `floor-versions` CI job.
- Tests accompany code; `pytest -q` and `ruff check` must pass.

## Scope

Feature construction operators, scorers, and selectors with stated
error-control properties are in scope. Heuristic selectors without a stated
guarantee are accepted only when clearly labelled as such in their docstring
and excluded from the default configuration.

## Releasing

The order matters; each step feeds the next.

1. Confirm a clean tree locally: `ruff check src tests benchmarks`,
   `pytest -q` (the 90% coverage floor is enforced in CI with
   `--cov=beamfeat --cov-fail-under=90`), and `mkdocs build` (needs the
   `docs` extra).
2. Bump `__version__` in `src/beamfeat/__init__.py` (the package version is
   read from it), add the release section to `CHANGELOG.md`, and update
   `version` and `date-released` in `CITATION.cff`.
3. Commit, push `main`, and wait for CI to pass on all jobs, including the
   oldest-stack and notebook jobs.
4. Tag the release commit and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "beamfeat X.Y.Z"
   git push origin vX.Y.Z
   ```

   The tag triggers `release.yml`, which runs the CI suite again, builds
   and checks the distributions, and publishes them to PyPI through trusted
   publishing (the `pypi` environment). Do not upload with `twine` by hand.
5. Create the GitHub release from the tag, with the changelog section as
   its notes. Zenodo archives each GitHub release automatically under the
   concept DOI already in `CITATION.cff` and the README badge.
6. conda-forge: update `recipe/recipe.yaml` (`version`, the sdist `sha256`
   from PyPI, and the `run` requirements when `pyproject.toml`'s dependencies
   changed) for the pending `staged-recipes` submission; once the feedstock
   exists, its bot opens the version-bump pull request instead, and the
   requirements are still checked by hand.
