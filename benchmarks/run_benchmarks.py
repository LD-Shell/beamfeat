"""Benchmark beamfeat against autofeat, raw LightGBM, and a linear baseline.

The comparison is deliberately unflattering where beamfeat loses. Three
quantities are recorded for every method on every dataset:

- **Accuracy** (out-of-sample R^2), the usual headline.
- **Wall-clock fit time**, since a method that wins on accuracy at a hundred
  times the cost has not obviously won.
- **Expression complexity**, the number of operator nodes in the selected
  features. This is the quantity the library actually claims to optimise, and
  the one a gradient-boosted baseline cannot report at all.

Two dataset families are used. *Synthetic* datasets have a known generating
formula, so recovery can be measured directly rather than inferred from
accuracy. *Real* datasets come from scikit-learn and the seaborn-data
repository, where no ground truth exists and only predictive performance is
comparable.

Run with ``python benchmarks/run_benchmarks.py``. Results are written to
``benchmarks/results.csv`` and summarised on stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

HERE = pathlib.Path(__file__).parent


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Dataset:
    """A benchmark problem.

    Attributes:
        name: Identifier used in the results table.
        X: Design matrix.
        y: Target.
        formula: The generating expression, where known. ``None`` for real
            datasets, which have no ground truth.
        tokens: Column tokens that a correct recovery must reference, used to
            score recovery on synthetic problems.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    formula: str | None = None
    tokens: tuple[str, ...] = ()
    problem_type: str = "regression"

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape


def synthetic_datasets(n: int = 500, seed: int = 0) -> list[Dataset]:
    """Problems with known generating formulas, for measuring recovery."""
    rng = np.random.default_rng(seed)
    datasets: list[Dataset] = []

    def make(name: str, fn, formula: str, tokens: tuple[str, ...], n_cols: int = 4, low=1.0, high=6.0):
        X = rng.uniform(low, high, (n, n_cols))
        y = fn(X)
        y = y + rng.normal(0, 0.02 * np.std(y), n)
        datasets.append(Dataset(name=name, X=X, y=y, formula=formula, tokens=tokens))

    make("product", lambda X: X[:, 0] * X[:, 1], "a*b", ("x0", "x1"))
    make("ratio", lambda X: X[:, 0] / X[:, 1], "a/b", ("x0", "x1"))
    make("three_way", lambda X: X[:, 0] * X[:, 1] / X[:, 2], "a*b/c", ("x0", "x1", "x2"))
    make("log_linear", lambda X: 5.0 * np.log(X[:, 0]) + X[:, 1], "5*log(a)+b", ("x0",))
    make("sqrt_ratio", lambda X: np.sqrt(X[:, 0]) / X[:, 1], "sqrt(a)/b", ("x0", "x1"))
    make("quadratic", lambda X: X[:, 0] ** 2 + X[:, 1], "a^2+b", ("x0",))
    make("inverse_square", lambda X: X[:, 0] / (X[:, 1] ** 2), "a/b^2", ("x0", "x1"))
    make(
        "physics_kinetic",
        lambda X: 0.5 * X[:, 0] * X[:, 1] ** 2,
        "0.5*m*v^2",
        ("x0", "x1"),
    )
    # A harder case: signal buried among many irrelevant columns.
    make("sparse_10col", lambda X: X[:, 0] * X[:, 1], "a*b (10 cols)", ("x0", "x1"), n_cols=10)
    # A case with no constructible structure, where beamfeat should NOT win.
    make("purely_linear", lambda X: 3.0 * X[:, 0] - 2.0 * X[:, 1] + X[:, 2], "linear", ())

    # Friedman #1: a mixture with a non-monotone bump and a centred quadratic.
    # A depth-2 basis can represent it (an oracle least-squares fit on
    # {x0*x1, (x0*x1)^2, x2, x2^2, x3, x4} reaches R^2 0.96), but the centred
    # quadratic term is nearly *marginally* independent of the target, so a
    # marginal-association pipeline is expected to underperform joint
    # selection here. Included as a documented boundary case, not a win.
    X_f = rng.uniform(0, 1, (800, 10))
    y_f = (
        10 * np.sin(np.pi * X_f[:, 0] * X_f[:, 1])
        + 20 * (X_f[:, 2] - 0.5) ** 2
        + 10 * X_f[:, 3]
        + 5 * X_f[:, 4]
        + rng.normal(0, 1, 800)
    )
    datasets.append(Dataset(name="friedman1", X=X_f, y=y_f, formula="10sin(pi a b)+20(c-.5)^2+10d+5e", tokens=()))

    # Heavy-tailed noise: Student t with 2 degrees of freedom has infinite
    # variance; a distribution-free permutation test should be unaffected.
    X_t = rng.uniform(1.0, 6.0, (n, 4))
    signal_t = X_t[:, 0] * X_t[:, 1] / X_t[:, 2]
    y_t = signal_t + 0.1 * np.std(signal_t) * rng.standard_t(2, n)
    datasets.append(Dataset(name="heavy_tail_t2", X=X_t, y=y_t, formula="a*b/c + t(2) noise", tokens=("x0", "x1", "x2")))

    return datasets


def robustness_datasets(n: int = 500, seed: int = 1) -> list[Dataset]:
    """Problems that stress the library's stated claims rather than its accuracy.

    Three families:

    - **Distractors under noise.** The signal ``a*b`` sits among ten irrelevant
      columns, at noise levels from mild to severe. The quantity of interest
      is not accuracy but the *false-feature rate*: the fraction of returned
      formulas that reference only irrelevant columns. This is the metric the
      FDR machinery exists to control, and the one plain accuracy comparisons
      cannot see.
    - **Small samples.** The selection holdout halves the rows available to
      each stage; these datasets measure where that documented cost starts to
      bite.
    - **Weak signal.** Noise at the same scale as the signal, where honest
      procedures should start returning less rather than confabulating.
    """
    rng = np.random.default_rng(seed)
    datasets: list[Dataset] = []

    for noise_level in (0.05, 0.25, 0.50):
        X = rng.uniform(1.0, 6.0, (n, 12))
        signal = X[:, 0] * X[:, 1]
        y = signal + rng.normal(0, noise_level * np.std(signal), n)
        datasets.append(Dataset(
            name=f"distractors_noise{int(noise_level * 100):02d}",
            X=X, y=y, formula=f"a*b + {int(noise_level * 100)}% noise, 10 distractors",
            tokens=("x0", "x1"),
        ))

    # Piecewise/threshold target: no compact algebraic form exists, so a
    # tree learner is the right tool and a formula-based method should lose.
    X_s = rng.uniform(0.0, 3.0, (n, 6))
    y_s = 3.0 * X_s[:, 1] * (X_s[:, 0] > 1.5) + rng.normal(0, 0.1, n)
    datasets.append(Dataset(name="threshold_step", X=X_s, y=y_s, formula="3b*1[a>1.5] (piecewise)", tokens=()))

    for n_small in (120, 240):
        X = rng.uniform(1.0, 6.0, (n_small, 6))
        signal = X[:, 0] * X[:, 1]
        y = signal + rng.normal(0, 0.1 * np.std(signal), n_small)
        datasets.append(Dataset(
            name=f"small_n{n_small}", X=X, y=y,
            formula="a*b, small sample", tokens=("x0", "x1"),
        ))

    return datasets


def real_datasets() -> list[Dataset]:
    """Standard regression problems with no known generating formula."""
    from sklearn.datasets import fetch_california_housing, load_diabetes

    datasets: list[Dataset] = []

    from sklearn.datasets import load_breast_cancer

    diabetes = load_diabetes()
    datasets.append(Dataset(name="diabetes", X=diabetes.data, y=diabetes.target))

    cancer = load_breast_cancer()
    datasets.append(Dataset(
        name="breast_cancer", X=cancer.data, y=cancer.target, problem_type="classification",
    ))

    # Four small real regression sets from the seaborn-data repository,
    # restricted to numeric columns with missing rows dropped — a documented,
    # reproducible preprocessing rather than an implicit encoding choice.
    seaborn_specs = [
        ("mpg", "mpg", ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"], None),
        # diamonds is capped. On 0.1.0, when the cap was set, memory during
        # the search grew roughly linearly with rows (about 530 MB at 2,000
        # and 1.7 GB at 8,000 on one core) and the full 53,940 rows exhausted
        # a 4 GB machine. Under 0.5.0 a fit on all 53,940 rows peaks below
        # 0.4 GB; the cap stays so that results remain comparable with the
        # archived ones.
        ("diamonds", "price", ["carat", "depth", "table", "x", "y", "z"], 1400),
        ("penguins", "body_mass_g", ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"], None),
        ("tips", "tip", ["total_bill", "size"], None),
    ]
    import pandas as pd

    base = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{}.csv"
    cache_dir = pathlib.Path(__file__).resolve().parent / "data"

    for name, target, columns, cap in seaborn_specs:
        cache = cache_dir / f"{name}.csv"
        try:
            frame = pd.read_csv(base.format(name))
            cache_dir.mkdir(exist_ok=True)
            frame.to_csv(cache, index=False)
        except Exception:
            if not cache.exists():
                raise
            print(f"  ({name}: fetch failed, using cached copy)")
            frame = pd.read_csv(cache)
        frame = frame[columns + [target]].dropna()
        if cap is not None and len(frame) > cap:
            frame = frame.sample(cap, random_state=0)
        datasets.append(Dataset(
            name=name,
            X=frame[columns].to_numpy(dtype=float),
            y=frame[target].to_numpy(dtype=float),
        ))

    # Used in full: 20,640 rows, so the figures are comparable with the
    # published literature. scikit-learn caches the download after the first run.
    housing = fetch_california_housing()
    datasets.append(Dataset(name="california_housing", X=housing.data, y=housing.target))

    return datasets


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Outcome:
    """One method's result on one dataset."""

    dataset: str
    method: str
    r2: float
    fit_seconds: float
    n_features: int
    complexity: float
    recovered: bool | None = None
    false_feature_rate: float | None = None
    fdr_ok: bool | None = None
    formulas: list[str] = field(default_factory=list)
    error: str = ""


def _complexity(formulas: list[str]) -> float:
    """Mean operator count per formula, as a proxy for interpretability."""
    if not formulas:
        return 0.0
    operators = ("+", "-", "*", "/", "log", "sqrt", "abs", "^", "exp")
    counts = [sum(formula.count(token) for token in operators) for formula in formulas]
    return float(np.mean(counts))


def _normalise_columns(formula: str) -> str:
    """Map zero-padded column names (autofeat's ``x000``) onto ``x0`` style.

    Recovery is judged on which columns a formula references; a naming
    convention must not decide the comparison.
    """
    import re

    return re.sub(r"x0*(\d+)", lambda match: f"x{int(match.group(1))}", formula)


def _false_feature_rate(formulas: list[str], tokens: tuple[str, ...]) -> float | None:
    """Fraction of returned formulas referencing only irrelevant columns.

    Defined only for datasets with known relevant columns (``tokens``). Under
    the marginal-association null this is the empirical analogue of the false
    discovery proportion: a formula touching no relevant column cannot be
    marginally associated with the target except by chance.
    """
    import re

    if not tokens or not formulas:
        return None
    n_false = 0
    for formula in formulas:
        present = set(re.findall(r"x\d+", _normalise_columns(formula)))
        if not (present & set(tokens)):
            n_false += 1
    return n_false / len(formulas)


def _recovered(formulas: list[str], tokens: tuple[str, ...]) -> bool | None:
    """Whether any selected formula references every required column.

    Column references are matched as whole tokens after normalisation, so
    ``x1`` matches in ``x1*x0`` but not inside ``x12``.
    """
    import re

    if not tokens:
        return None
    for formula in formulas:
        normalised = _normalise_columns(formula)
        present = set(re.findall(r"x\d+", normalised))
        if all(token in present for token in tokens):
            return True
    return False


def run_beamfeat(dataset: Dataset, X_train, X_test, y_train, y_test, **kwargs) -> Outcome:
    from beamfeat import BeamFeatClassifier, BeamFeatRegressor

    estimator_class = BeamFeatClassifier if dataset.problem_type == "classification" else BeamFeatRegressor
    started = time.perf_counter()
    model = estimator_class(
        max_depth=kwargs.get("max_depth", 2),
        beam_width=kwargs.get("beam_width", 40),
        selector=kwargs.get("selector", "permutation"),
        parsimony=kwargs.get("parsimony", "forward"),
        target_fdr=0.1,
        random_state=0,
    )
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    formulas = model.formulas()
    return Outcome(
        dataset=dataset.name,
        method=kwargs.get("label", "beamfeat"),
        r2=float(model.score(X_test, y_test)),
        fit_seconds=elapsed,
        n_features=len(formulas),
        complexity=_complexity(formulas),
        recovered=_recovered(formulas, dataset.tokens),
        false_feature_rate=_false_feature_rate(formulas, dataset.tokens),
        fdr_ok=model.fdr_controlled_,
        formulas=formulas,
    )


def run_autofeat(dataset: Dataset, X_train, X_test, y_train, y_test) -> Outcome:
    try:
        from autofeat import AutoFeatRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        return Outcome(dataset.name, "autofeat", float("nan"), 0.0, 0, 0.0, error=str(exc))

    started = time.perf_counter()
    try:
        model = AutoFeatRegressor(feateng_steps=2, featsel_runs=3, verbose=0)
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        score = float(model.score(X_test, y_test))
        formulas = [str(f) for f in getattr(model, "new_feat_cols_", [])]
    except Exception as exc:
        return Outcome(
            dataset.name, "autofeat", float("nan"), time.perf_counter() - started, 0, 0.0,
            error=f"{type(exc).__name__}: {str(exc)[:120]}",
        )

    return Outcome(
        dataset=dataset.name,
        method="autofeat",
        r2=score,
        fit_seconds=elapsed,
        n_features=len(formulas),
        complexity=_complexity(formulas),
        recovered=_recovered(formulas, dataset.tokens),
        false_feature_rate=_false_feature_rate(formulas, dataset.tokens),
        formulas=formulas,
    )


def run_lightgbm(dataset: Dataset, X_train, X_test, y_train, y_test) -> Outcome:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        return Outcome(dataset.name, "lightgbm", float("nan"), 0.0, 0, 0.0, error=str(exc))

    started = time.perf_counter()
    if dataset.problem_type == "classification":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
    else:
        model = LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    return Outcome(
        dataset=dataset.name,
        method="lightgbm",
        r2=float(model.score(X_test, y_test)),
        fit_seconds=elapsed,
        n_features=X_train.shape[1],
        # A boosted ensemble has no closed form to report; recorded as NaN
        # rather than zero, which would falsely read as maximal simplicity.
        complexity=float("nan"),
        recovered=None,
    )


def run_knockpy(dataset: Dataset, X_train, X_test, y_train, y_test) -> Outcome:
    """knockpy (Spector & Janson): FDR-controlled selection among RAW features.

    The selection-only pole of the comparison: it constructs nothing, so its
    row isolates what feature construction contributes over error-controlled
    selection alone. Configuration disclosed: the Gaussian model-X sampler
    with the lasso statistic, thresholded at ``offset=0`` (modified-FDR
    control) because knockoff+ (``offset=1``) requires more than ``1/q``
    selectable features and is arithmetically unsatisfiable at these
    dimensionalities (p <= 30 at q = 0.1) — the same satisfiability boundary
    documented for beamfeat's own knockoff selector. Downstream model:
    RidgeCV on the selected raw columns; the "formulas" are the selected
    column names, so the false-feature metric counts selected distractors.
    """
    if dataset.problem_type == "classification":
        return Outcome(dataset.name, "knockpy", float("nan"), 0.0, 0, 0.0,
                       error="classification not configured in this harness")
    try:
        from knockpy import KnockoffFilter, knockoff_stats
        from sklearn.linear_model import RidgeCV
    except ImportError as exc:  # pragma: no cover - optional dependency
        return Outcome(dataset.name, "knockpy", float("nan"), 0.0, 0, 0.0, error=str(exc))

    started = time.perf_counter()
    try:
        kfilter = KnockoffFilter(ksampler="gaussian", fstat="lasso")
        kfilter.forward(X=X_train, y=y_train, fdr=0.1)
        threshold = knockoff_stats.data_dependent_threshhold(W=kfilter.W, fdr=0.1, offset=0)
        selected = np.flatnonzero(kfilter.W >= threshold)
        if len(selected):
            model = RidgeCV().fit(X_train[:, selected], y_train)
            r2 = float(model.score(X_test[:, selected], y_test))
        else:
            r2 = float(1.0 - np.var(y_test - np.mean(y_train)) / np.var(y_test))
        elapsed = time.perf_counter() - started
        formulas = [f"x{i}" for i in selected]
        return Outcome(
            dataset=dataset.name, method="knockpy", r2=r2, fit_seconds=elapsed,
            n_features=len(selected), complexity=0.0,
            recovered=_recovered(formulas, dataset.tokens),
            false_feature_rate=_false_feature_rate(formulas, dataset.tokens),
            formulas=formulas,
        )
    except Exception as exc:
        return Outcome(dataset.name, "knockpy", float("nan"), time.perf_counter() - started,
                       0, 0.0, error=f"{type(exc).__name__}: {str(exc)[:120]}")


def run_openfe(dataset: Dataset, X_train, X_test, y_train, y_test) -> Outcome:
    """OpenFE (Zhang et al., 2023): expansion + LightGBM-based feature boosting.

    Compared in its intended configuration — generated features augmenting a
    LightGBM model — since its features target tree learners, not linear
    equations. Formula strings are extracted with its own ``tree_to_formula``
    so the false-feature rate is measured on the same footing as the symbolic
    methods.
    """
    try:
        import openfe.openfe as openfe_module
        import pandas as pd
        from lightgbm import LGBMRegressor
        from openfe import OpenFE, transform, tree_to_formula
    except ImportError as exc:  # pragma: no cover - optional dependency
        return Outcome(dataset.name, "openfe", float("nan"), 0.0, 0, 0.0, error=str(exc))

    # Compatibility shim, disclosed in the write-up: OpenFE 0.0.12 calls
    # ``mean_squared_error(..., squared=False)``, a parameter removed in
    # scikit-learn >= 1.6, so it cannot run unpatched on a current stack —
    # the same dependency-stranding failure mode measured for autofeat.
    from sklearn.metrics import mean_squared_error as _mse

    def _mse_compat(y_true, y_pred, *, squared=True, **kwargs):
        value = _mse(y_true, y_pred, **kwargs)
        return value if squared else float(np.sqrt(value))

    openfe_module.mean_squared_error = _mse_compat

    if dataset.problem_type == "classification":
        return Outcome(dataset.name, "openfe", float("nan"), 0.0, 0, 0.0,
                       error="classification not configured in this harness")

    columns = [f"x{i}" for i in range(X_train.shape[1])]
    train = pd.DataFrame(X_train, columns=columns)
    test = pd.DataFrame(X_test, columns=columns)
    label = pd.DataFrame({"target": y_train})

    started = time.perf_counter()
    try:
        import os
        import tempfile

        # OpenFE writes a scratch file (openfe_tmp_data.feather) into the
        # working directory; run it inside a temporary directory so no
        # artifact lands in the repository.
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as scratch:
            try:
                os.chdir(scratch)
                engine = OpenFE()
                generated = engine.fit(data=train, label=label, n_jobs=2, verbose=False)
                train_aug, test_aug = transform(train, test, generated, n_jobs=2)
            finally:
                os.chdir(original_cwd)
        model = LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
        model.fit(train_aug, y_train)
        elapsed = time.perf_counter() - started
        formulas = []
        for feature in generated:
            try:
                formulas.append(tree_to_formula(feature))
            except Exception:  # pragma: no cover - formula rendering best-effort
                pass
        return Outcome(
            dataset=dataset.name,
            method="openfe",
            r2=float(model.score(test_aug, y_test)),
            fit_seconds=elapsed,
            n_features=len(generated),
            complexity=_complexity(formulas),
            recovered=_recovered(formulas, dataset.tokens),
            false_feature_rate=_false_feature_rate(formulas, dataset.tokens),
            formulas=formulas,
        )
    except Exception as exc:
        return Outcome(
            dataset.name, "openfe", float("nan"), time.perf_counter() - started, 0, 0.0,
            error=f"{type(exc).__name__}: {str(exc)[:120]}",
        )


def run_ridge(dataset: Dataset, X_train, X_test, y_train, y_test) -> Outcome:
    from sklearn.linear_model import LogisticRegression, RidgeCV

    started = time.perf_counter()
    if dataset.problem_type == "classification":
        model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
    else:
        model = RidgeCV().fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    return Outcome(
        dataset=dataset.name,
        method="ridge",
        r2=float(model.score(X_test, y_test)),
        fit_seconds=elapsed,
        n_features=X_train.shape[1],
        complexity=0.0,
        recovered=None,
    )


METHODS = {
    "knockpy": run_knockpy,
    "openfe": run_openfe,
    "ridge": run_ridge,
    "lightgbm": run_lightgbm,
    "autofeat": run_autofeat,
    "beamfeat": run_beamfeat,
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def evaluate(datasets: list[Dataset], methods: list[str], test_size: float = 0.3,
             beamfeat_kwargs: dict | None = None) -> list[Outcome]:
    """Run every method on every dataset with a common train/test split.

    ``beamfeat_kwargs`` reaches ``run_beamfeat`` only, so a caller comparing
    two beamfeat settings gets identical splits and identical baselines.
    """
    from sklearn.model_selection import train_test_split

    beamfeat_kwargs = beamfeat_kwargs or {}
    outcomes: list[Outcome] = []
    for dataset in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=test_size, random_state=0
        )
        print(f"\n{dataset.name}  ({dataset.shape[0]}x{dataset.shape[1]})", flush=True)
        if dataset.formula:
            print(f"  true formula: {dataset.formula}")

        for method in methods:
            try:
                extra = beamfeat_kwargs if method == "beamfeat" else {}
                outcome = METHODS[method](dataset, X_train, X_test, y_train, y_test, **extra)
            except Exception as exc:  # noqa: BLE001 - a failing method must not sink the run
                outcome = Outcome(
                    dataset=dataset.name, method=method, r2=float("nan"),
                    fit_seconds=0.0, n_features=0, complexity=float("nan"),
                    error=f"{type(exc).__name__}: {str(exc)[:120]}",
                )
            outcomes.append(outcome)
            marker = ""
            if outcome.recovered is True:
                marker = "  [recovered]"
            elif outcome.recovered is False:
                marker = "  [missed]"
            if outcome.fdr_ok is False:
                marker += "  [NO FDR: fallback]"
            if outcome.error:
                print(f"  {method:>10}: ERROR {outcome.error}")
            else:
                print(
                    f"  {method:>10}: R2 {outcome.r2:>7.4f}  "
                    f"{outcome.fit_seconds:>7.2f}s  "
                    f"{outcome.n_features:>3} feats{marker}",
                    flush=True,
                )
    return outcomes


def summarise(outcomes: list[Outcome]) -> None:
    """Print aggregate comparisons across datasets."""
    methods = sorted({o.method for o in outcomes})

    print("\n" + "=" * 74)
    print("AGGREGATE (mean over datasets where the method succeeded)")
    print("=" * 74)
    print("(the score column is R^2 for regression datasets, accuracy for classification)")
    print(f"{'method':>12} {'mean score':>10} {'median':>11} {'mean secs':>11} {'mean feats':>11}")
    for method in methods:
        rows = [o for o in outcomes if o.method == method and np.isfinite(o.r2)]
        if not rows:
            print(f"{method:>12}  (no successful runs)")
            continue
        scores = [o.r2 for o in rows]
        print(
            f"{method:>12} {np.mean(scores):>10.4f} {np.median(scores):>11.4f} "
            f"{np.mean([o.fit_seconds for o in rows]):>11.2f} "
            f"{np.mean([o.n_features for o in rows]):>11.1f}"
        )

    recoverable = [o for o in outcomes if o.recovered is not None]
    if recoverable:
        print("\n" + "=" * 74)
        print("FORMULA RECOVERY (synthetic problems with known ground truth)")
        print("=" * 74)
        for method in sorted({o.method for o in recoverable}):
            rows = [o for o in recoverable if o.method == method]
            n_hit = sum(1 for o in rows if o.recovered)
            print(f"{method:>12}: {n_hit}/{len(rows)} recovered")

    rate_rows = [o for o in outcomes if o.false_feature_rate is not None]
    if rate_rows:
        print("\n" + "=" * 74)
        print("FALSE-FEATURE RATE (fraction of returned formulas touching only")
        print("irrelevant columns; the empirical analogue of the FDP)")
        print("=" * 74)
        for method in sorted({o.method for o in rate_rows}):
            rows = [o for o in rate_rows if o.method == method]
            mean_rate = float(np.mean([o.false_feature_rate for o in rows]))
            worst = max(rows, key=lambda o: o.false_feature_rate)
            print(f"{method:>12}: mean {mean_rate:6.3f}   worst {worst.false_feature_rate:.3f} ({worst.dataset})")

    fdr_rows = [o for o in outcomes if o.fdr_ok is not None]
    if fdr_rows:
        n_ok = sum(1 for o in fdr_rows if o.fdr_ok)
        print("\n" + "=" * 74)
        print("FDR-CONTROL STATUS (beamfeat)")
        print("=" * 74)
        print(f"  selection passed with guarantee intact: {n_ok}/{len(fdr_rows)} datasets")
        for outcome in fdr_rows:
            if not outcome.fdr_ok:
                print(f"    fallback (no guarantee): {outcome.dataset}")

    complexity_rows = [o for o in outcomes if np.isfinite(o.complexity) and o.n_features]
    if complexity_rows:
        print("\n" + "=" * 74)
        print("EXPRESSION COMPLEXITY (mean operators per selected feature)")
        print("=" * 74)
        for method in sorted({o.method for o in complexity_rows}):
            rows = [o for o in complexity_rows if o.method == method]
            print(f"{method:>12}: {np.mean([o.complexity for o in rows]):>6.2f}")
        print("  (lightgbm omitted: an ensemble has no closed form to measure)")


def write_results(outcomes: list[Outcome], path: pathlib.Path) -> None:
    """Write results as CSV and JSON for later analysis."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "method", "r2", "fit_seconds", "n_features", "complexity", "recovered", "false_feature_rate", "fdr_ok", "error"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for outcome in outcomes:
            row = {key: value for key, value in asdict(outcome).items() if key in fields}
            writer.writerow(row)

    json_path = path.with_suffix(".json")
    with json_path.open("w") as handle:
        json.dump([asdict(o) for o in outcomes], handle, indent=2, default=str)

    print(f"\nwrote {path} and {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument(
        "--suite", choices=["core", "robustness", "all"], default="core",
        help="core: known-formula problems; robustness: distractor/noise/small-n stress",
    )
    parser.add_argument("--n", type=int, default=500, help="rows per synthetic dataset")
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "results.csv")
    args = parser.parse_args()

    datasets: list[Dataset] = []
    if args.real_only:
        datasets = real_datasets()
        outcomes = evaluate(datasets, args.methods)
        summarise(outcomes)
        write_results(outcomes, args.output)
        return
    if args.suite in ("core", "all"):
        datasets.extend(synthetic_datasets(n=args.n))
    if args.suite in ("robustness", "all"):
        datasets.extend(robustness_datasets(n=args.n))
    if not args.synthetic_only:
        datasets.extend(real_datasets())

    outcomes = evaluate(datasets, args.methods)
    summarise(outcomes)
    write_results(outcomes, args.output)


if __name__ == "__main__":
    main()
