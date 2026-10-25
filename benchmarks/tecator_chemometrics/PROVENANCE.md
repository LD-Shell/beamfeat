# Provenance

Added after the paper was published. The paper's high-dimensional study
compared beamfeat on tecator with ridge regression, a random forest and
LightGBM; a reader from chemometrics will ask for that field's own methods,
partial least squares on preprocessed spectra and kernel methods, and this
folder supplies them. The paper and its archived runs are unchanged.

## Data and splits

`benchmarks/additional_experiments/data/tecator.csv`: 240 samples recorded on
a Tecator Infratec Food and Feed Analyzer, 100 absorbance channels from 850
to 1048 nm in 2 nm steps, fat content in percent; it is redistributed under
the permission note in `data/TECATOR_PERMISSION.txt` there. Column `xk` of a
formula is channel `k + 1`, at 850 + 2k nm.

Two protocols:

- the random splits of `benchmarks/additional_experiments/bench.py`:
  `train_test_split(test_size=0.25, random_state=split)` for splits 0 to 4,
  which gives 180 training and 60 test rows;
- the split of the data's documentation (StatLib), whose sample order the
  CSV keeps: samples 1 to 172 (the sets C and M) to train, 173 to 215 (T) to
  test, and the extrapolation sets E1 (216 to 223) and E2 (224 to 240).
  beamfeat is fitted with seeds 0 to 4 there and quoted as their average.

## Environment

beamfeat 0.7.0, with the stack of `benchmarks/release_comparison/`: Python
3.12, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.0, one BLAS
thread. Every model is seeded or deterministic, so a rerun on the same stack
reproduces `results.jsonl`.

```bash
uv venv --python 3.12 venv
uv pip install --python venv -e . numpy==2.5.4 scipy==1.18.1 scikit-learn==1.9.1 \
  pandas==3.1.0
cd benchmarks/tecator_chemometrics
OPENBLAS_NUM_THREADS=1 ../../venv/bin/python run.py
../../venv/bin/python report.py > REPORT.md
```

## Limits

Five splits of 240 samples, and tuning grids of this folder's choosing; a
wider grid or another preprocessing can move a baseline. E1 has 8 samples, so
its errors show what can happen beyond the training data rather than measure
it precisely. The random forest
and LightGBM rows are the paper's Table 6 figures, not rerun here. beamfeat's
derivative window is fixed at 15, where the baselines' windows are tuned.
