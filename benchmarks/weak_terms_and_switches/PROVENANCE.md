# Provenance

`run.py` fits each release in its own environment, those of
`benchmarks/release_comparison/` (PROVENANCE.md there): Python 3.12,
numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.0, sympy 1.14.0,
joblib 1.6.0; 0.7.0 from PyPI, 0.8.0 as 0.8.0.dev0 from commit e1a2839, whose
code fits as the release does. Two BLAS threads, on a 22-core machine running
other work, so the recorded seconds are not timings.

The data are drawn from `numpy.random.default_rng` seeded by the seed (the
switch study adds 100), every setting other than those named is the default,
and beamfeat is deterministic given `random_state`, so a rerun in the same
environment reproduces every record.

```bash
../release_comparison/venvs/0.7.0/bin/python -I run.py --version 0.7.0
../release_comparison/venvs/0.8.0/bin/python -I run.py --version 0.8.0
../release_comparison/venvs/0.8.0/bin/python -I run.py --report > TABLES.md
../release_comparison/venvs/0.8.0/bin/python -I alternatives.py > results/alternatives.jsonl
```

`results/records_<version>.jsonl` holds one record per fit, with the printed
formulas. A term counts as printed when sympy finds it proportional to the
expected one, its fitted constants read as fractions.
