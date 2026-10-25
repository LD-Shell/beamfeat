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
            record = {"method": name, "split": split, "r2": float(r2_score(y_test, prediction)),
                      "rmse": float(mean_squared_error(y_test, prediction) ** 0.5), "size": n,
                      "formulas": formulas,
                      "params": {k: str(v) for k, v in getattr(model, "best_params_", {}).items()},
                      "beamfeat": beamfeat.__version__, "rows": [len(y_train), len(y_test)]}
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(name, split, f"R2 {record['r2']:.4f}", f"RMSE {record['rmse']:.3f}", n, flush=True)


if __name__ == "__main__":
    main()
