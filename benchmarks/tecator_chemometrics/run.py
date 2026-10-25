"""tecator against the baselines of chemometrics, added after the paper.

    python run.py            # writes results.jsonl
    python report.py > REPORT.md

The data and the five 75/25 splits are those of the paper's high-dimensional
study (benchmarks/additional_experiments/bench.py, Table 6). Every baseline
is tuned by 5-fold cross-validation on the training rows of its split:

- ridge on standardised channels, as Table 6 fits it;
- partial least squares (PLS) on the raw spectra, on spectra after the
  standard normal variate (SNV), and on Savitzky-Golay first or second
  derivatives, with the window and the number of components tuned;
- kernel ridge regression, support vector regression and a Gaussian process
  on standardised second-derivative spectra, with the window and the kernel's
  parameters tuned (the Gaussian process by its marginal likelihood);
- beamfeat at its defaults, on the raw channels and on second-derivative
  spectra (window 15, not tuned).

The same methods are also fitted on the split the data's documentation
describes: the data sets C and M (samples 1 to 172) to train, T (173 to 215)
to test, and the extrapolation sets E1 (216 to 223, more fat than any
training sample) and E2 (224 to 240, more protein). beamfeat is fitted with
five seeds there, as the documentation asks of a model with an element of
randomness, beside its settings parsimony_alpha=0.05 and magnitude_slots=0
(0.6.1's default). For each beamfeat fit, "outside" records how far its terms
go beyond their training range on each test set: the largest distance
beyond the range, in widths of the range, over terms and rows.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

import beamfeat
from beamfeat import BeamFeatRegressor

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "additional_experiments" / "data" / "tecator.csv"
ALPHAS = np.logspace(-4, 4, 25)  # Table 6's ridge grid
WINDOWS = [7, 11, 15, 21]
INNER = KFold(5, shuffle=True, random_state=0)
warnings.filterwarnings("ignore")


def savgol(spectra, window_length=15, deriv=2):
    return savgol_filter(spectra, window_length, 2, deriv=deriv, axis=1)


def snv(spectra):
    return (spectra - spectra.mean(axis=1, keepdims=True)) / spectra.std(axis=1, keepdims=True)


def derivative(**kw):
    return FunctionTransformer(savgol, kw_args=kw)


def tuned(pipeline, grid):
    return GridSearchCV(pipeline, grid, cv=INNER, scoring="neg_mean_squared_error")


COMPONENTS = {"plsregression__n_components": list(range(1, 21))}
SG2 = [{"window_length": w, "deriv": 2} for w in WINDOWS]
METHODS = {
    "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
    "PLS": lambda: tuned(make_pipeline(PLSRegression(scale=False)), COMPONENTS),
    "PLS, SNV": lambda: tuned(make_pipeline(FunctionTransformer(snv), PLSRegression(scale=False)), COMPONENTS),
    "PLS, derivative": lambda: tuned(
        make_pipeline(derivative(), PLSRegression(scale=False)),
        {**COMPONENTS, "functiontransformer__kw_args": [{"window_length": w, "deriv": d}
                                                        for w in WINDOWS for d in (1, 2)]}),
    "kernel ridge, derivative": lambda: tuned(
        make_pipeline(derivative(), StandardScaler(), KernelRidge(kernel="rbf")),
        {"functiontransformer__kw_args": SG2, "kernelridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1],
         "kernelridge__gamma": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]}),
    "SVR, derivative": lambda: tuned(
        TransformedTargetRegressor(make_pipeline(derivative(), StandardScaler(), SVR()),
                                   transformer=StandardScaler()),
        {"regressor__functiontransformer__kw_args": SG2, "regressor__svr__C": [1, 10, 100, 1000],
         "regressor__svr__gamma": [1e-4, 1e-3, 1e-2], "regressor__svr__epsilon": [0.01, 0.1]}),
    "Gaussian process, derivative": lambda: tuned(
        make_pipeline(derivative(), StandardScaler(), GaussianProcessRegressor(
            ConstantKernel() * RBF(length_scale=10.0) + WhiteKernel(), normalize_y=True,
            n_restarts_optimizer=2, random_state=0)),
        {"functiontransformer__kw_args": SG2}),
    "beamfeat": lambda: BeamFeatRegressor(random_state=0),
    "beamfeat, derivative": lambda: make_pipeline(derivative(), BeamFeatRegressor(random_state=0)),
}


def size(model):
    """Terms for beamfeat, components for PLS, None for the rest."""
    est = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    last = est[-1] if hasattr(est, "steps") else est
    if hasattr(last, "features_"):
        return len(last.features_), list(last.feature_formulas_)
    if isinstance(last, PLSRegression):
        return last.n_components, None
    return None, None


DOCUMENTED = {"train": slice(0, 172), "T": slice(172, 215), "E1": slice(215, 223), "E2": slice(223, 240)}
BEAMFEAT_DOCUMENTED = {
    "beamfeat": lambda seed: BeamFeatRegressor(random_state=seed),
    "beamfeat, parsimony_alpha=0.05": lambda seed: BeamFeatRegressor(random_state=seed, parsimony_alpha=0.05),
    "beamfeat, magnitude_slots=0": lambda seed: BeamFeatRegressor(random_state=seed, magnitude_slots=0),
    "beamfeat, derivative": lambda seed: make_pipeline(derivative(), BeamFeatRegressor(random_state=seed)),
}


def outside(model, X_train, X_new):
    """The largest distance of a term's values on X_new beyond its range on X_train, in widths of that range."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(model, "steps"):
        X_train, X_new = model[:-1].transform(X_train), model[:-1].transform(X_new)
    if not est.features_:
        return 0.0
    train, new = est._apply(X_train), est._apply(X_new)
    low, high = train.min(axis=0), train.max(axis=0)
    width = np.where(high > low, high - low, 1.0)
    return float(np.max(np.maximum(new - high, low - new) / width))


def documented(X, y, out):
    parts = {name: (X[rows], y[rows]) for name, rows in DOCUMENTED.items()}
    X_train, y_train = parts["train"]
    fits = [(name, None, make) for name, make in METHODS.items() if not name.startswith("beamfeat")]
    fits += [(name, seed, make) for name, make in BEAMFEAT_DOCUMENTED.items() for seed in range(5)]
    for name, seed, make in fits:
        model = (make() if seed is None else make(seed)).fit(X_train, y_train)
        record = {"protocol": "documented", "method": name, "seed": seed, "beamfeat": beamfeat.__version__}
        for part in ("T", "E1", "E2"):
            X_part, y_part = parts[part]
            prediction = np.ravel(model.predict(X_part))
            record[f"rmse_{part}"] = float(mean_squared_error(y_part, prediction) ** 0.5)
            record[f"range_{part}"] = [float(prediction.min()), float(prediction.max())]
            if seed is not None:
                record[f"outside_{part}"] = outside(model, X_train, X_part)
        record["r2_T"] = float(r2_score(parts["T"][1], np.ravel(model.predict(parts["T"][0]))))
        record["size"], record["formulas"] = size(model)
        out.write(json.dumps(record) + "\n")
        out.flush()
        print(name, seed, f"T {record['rmse_T']:.3f}", f"E1 {record['rmse_E1']:.3f}", f"E2 {record['rmse_E2']:.3f}",
              flush=True)


def main():
    table = pd.read_csv(DATA)
    channels = [c for c in table.columns if c.startswith("_") or c.startswith("absorbance")]
    X, y = table[channels].to_numpy(float), table["fat"].to_numpy(float)
    out = (HERE / "results.jsonl").open("w")
    for name, make in METHODS.items():
        for split in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=split)
            model = make().fit(X_train, y_train)
            prediction = np.ravel(model.predict(X_test))
            n, formulas = size(model)
            record = {"protocol": "random", "method": name, "split": split, "r2": float(r2_score(y_test, prediction)),
                      "rmse": float(mean_squared_error(y_test, prediction) ** 0.5), "size": n,
                      "formulas": formulas,
                      "params": {k: str(v) for k, v in getattr(model, "best_params_", {}).items()},
                      "beamfeat": beamfeat.__version__, "rows": [len(y_train), len(y_test)]}
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(name, split, f"R2 {record['r2']:.4f}", f"RMSE {record['rmse']:.3f}", n, flush=True)
    documented(X, y, out)


if __name__ == "__main__":
    main()
