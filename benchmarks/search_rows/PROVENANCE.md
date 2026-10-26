# Provenance

beamfeat 0.8.0 as 0.8.0.dev0 from commit e1a2839, whose code fits as the
release does, in the environment of `benchmarks/release_comparison/`
(`venvs/0.8.0`): Python 3.12, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1,
pandas 3.1.0, sympy 1.14.0, joblib 1.6.0. Two BLAS threads for the main
table and one for the memory pass, on a 22-core machine running other work,
so the seconds are indicative: compare them within the table, not across
machines.

The data are drawn from `numpy.random.default_rng(1000 + seed)` and beamfeat
is deterministic given `random_state`, so a rerun in the same environment
reproduces every record but the timings.

```bash
../release_comparison/venvs/0.8.0/bin/python -I run.py --out results/records.jsonl
../release_comparison/venvs/0.8.0/bin/python -I run.py --memory --out results/memory.jsonl
../release_comparison/venvs/0.8.0/bin/python -I run.py --report results > TABLES.md
```

A first run, before the residual slots were turned off by default, found the
0.5% term in all five full-search fits and in fewer of the sampled ones; it
is not kept, since 0.8.0's default leaves the slots off.
