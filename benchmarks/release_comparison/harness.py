"""Release comparison: beamfeat 0.4.2, 0.5.0, 0.6.0 and 0.6.1 and their options.

The same rows, splits and seeds go through whichever beamfeat the running
interpreter has. Data come from the repository's benchmark scripts
(run_benchmarks.py, search_stress/stress.py, feynman_panel.py,
knockoff_s_method.py), which import beamfeat only inside functions, so every
version reads identical data. PROVENANCE.md has the details.

Studies:
    designs   parsimony_holdout calibration designs: 20 cells x 50 seeds
    stress    search_stress laws (pair, triple, pair_plus) x inputs (unif,
              gauss, shifted) x p (20, 100), 500 rows, 10% noise, 10 seeds
    feynman   the 12 panel laws, 5 data draws each
    suite     run_benchmarks synthetic + robustness + real sets, 5 splits
    highdim   communities, superconduct (5,000 rows), tecator, eyedata,
              riboflavin, 5 splits
    junk      y = x0*x1 + 25% noise among 10 junk columns (independent, and
              with x8 correlated 0.82 with x1), 20 seeds
    knockoff  selector level: benchmarks/knockoff_s_method.py designs

    python harness.py --version 0.6.0 --study designs --part 1/4
    python harness.py --version 0.6.1 --perf designs/planted_snr3_n1000_k20:0 --config default
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
import traceback
import warnings
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
for path in (BENCH / "search_stress", BENCH):
    sys.path.insert(0, str(path))
HIGHDIM_DATA = BENCH / "additional_experiments" / "data"
SEABORN_DATA = BENCH / "data"  # run_benchmarks.py's cache of the seaborn-data files
warnings.filterwarnings("ignore")
logging.getLogger("beamfeat").setLevel(logging.ERROR)

CONFIGS = {
    "0.4.2": {"default": {}},
    "0.5.0": {"default": {}},
    "0.6.0": {
        "default": {},
        "no_fitted_ops": {"fitted_ops": ()},
        "magnitude_slots": {"magnitude_slots": 10},
        "input_test": {"input_test": True},
        "conditional_retest": {"parsimony_holdout": 0.3, "certify_test": "conditional"},
        "bh": {"selection_correction": "bh"},
    },
    "0.6.1": {
        "default": {},
        "parsimony_alpha": {"parsimony_alpha": 0.05},
        "alpha_input_test": {"parsimony_alpha": 0.05, "input_test": True},
        # The same test as 0.6.0's, with the memory fix; it stands in for 0.6.0
        # on California housing, where 0.6.0 passed 7 GB.
        "input_test": {"input_test": True},
    },
    "0.7.0": {
        "default": {},
        # 0.6.1's default, which 0.7.0 gives with the magnitude slots off.
        "no_magnitude_slots": {"magnitude_slots": 0},
        "parsimony_alpha": {"parsimony_alpha": 0.05},
        "conditional_retest": {"parsimony_holdout": 0.3, "certify_test": "conditional"},
        "alpha_conditional": {"parsimony_alpha": 0.05, "parsimony_holdout": 0.3, "certify_test": "conditional"},
    },
}
REGRESSION_ONLY = {"magnitude_slots", "conditional_retest", "alpha_conditional"}
COLUMN = re.compile(r"x(\d+)")


def columns_of(formula):
    return {int(j) for j in COLUMN.findall(formula)}


# --------------------------------------------------------------------------- #
# Search memo: configs that leave the search alone reuse it within one item
# --------------------------------------------------------------------------- #


def _search_module():
    """The module whose ``BeamSearch`` a fit looks up: ``beamfeat._base`` from 0.7.0, ``beamfeat.estimators`` before."""
    import importlib
    import importlib.util

    if importlib.util.find_spec("beamfeat._base") is not None:
        return importlib.import_module("beamfeat._base")
    return importlib.import_module("beamfeat.estimators")


class SearchMemo:
    def __init__(self):
        self.module = _search_module()
        self.original, self.results = self.module.BeamSearch, {}
        memo = self

        class Memoised(self.original):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._memo_kwargs = kwargs

            def run(self, data, target, *args, **kwargs):
                digest = hashlib.sha256(repr((sorted(self._memo_kwargs.items()), args,
                                              sorted(kwargs.items()))).encode())
                for name, column in data.items():
                    digest.update(str(name).encode())
                    digest.update(np.ascontiguousarray(column, dtype=np.float64).tobytes())
                digest.update(np.ascontiguousarray(target, dtype=np.float64).tobytes())
                key = digest.hexdigest()
                if key not in memo.results:
                    memo.results[key] = super().run(data, target, *args, **kwargs)
                return memo.results[key]

        self.memoised = Memoised

    def __enter__(self):
        self.results.clear()
        self.module.BeamSearch = self.memoised
        return self

    def __exit__(self, *exc):
        self.module.BeamSearch = self.original
        self.results.clear()


# --------------------------------------------------------------------------- #
# Items: (study, name) -> data and an evaluator of a fitted model
# --------------------------------------------------------------------------- #

TRUE_COLUMNS = frozenset({"x0", "x1", "x2"})
SIGNAL_SD = math.sqrt(169 / 27 - (2 * math.log(3)) ** 2)
PROXY_HALF_WIDTH = math.sqrt(1 / 0.81 - 1)
DESIGN_CELLS = [("null", n, p, None) for n in (300, 1000) for p in (5, 20)] + [
    (d, n, k, s) for d in ("planted", "proxy") for s in (1, 3) for n in (300, 1000) for k in (5, 20)]


def design_name(d, n, w, s):
    return f"null_n{n}_p{w}" if d == "null" else f"{d}_snr{s}_n{n}_k{w}"


def design_data(d, n, w, s, seed):
    total = n + 2000
    if d == "null":
        rng = np.random.default_rng([0, n, w, seed])
        X = rng.uniform(1.0, 3.0, (total, w))
        y = rng.normal(0.0, 1.0, total)
    else:
        rng = np.random.default_rng([1, n, w, seed])
        X = rng.uniform(1.0, 3.0, (total, 3 + w))
        noise = rng.standard_normal(total)
        y = X[:, 0] * X[:, 1] / X[:, 2] + (SIGNAL_SD / s) * noise
        if d == "proxy":
            X = np.column_stack([X, X[:, 0] + rng.uniform(-PROXY_HALF_WIDTH, PROXY_HALF_WIDTH, total)])
    return X[:n], y[:n], X[n:], y[n:]


def design_items():
    return [("designs", f"{design_name(*c)}:{seed}") for c in DESIGN_CELLS for seed in range(50)]


def design_measure(model, design, X_test, y_test):
    from beamfeat.expression import Evaluator

    truth = frozenset() if design == "null" else TRUE_COLUMNS
    used = [frozenset(node.columns()) for node in model.features_]
    true_term = None
    if design != "null":
        true_term = 0
        evaluator = Evaluator({f"x{j}": X_test[:, j] for j in range(X_test.shape[1])})
        signal = X_test[:, 0] * X_test[:, 1] / X_test[:, 2]
        for node, cols in zip(model.features_, used):
            values = evaluator.transform_values(node) if cols == truth else None
            if values is not None and abs(np.corrcoef(values, signal)[0, 1]) >= 0.999:
                true_term = 1
                break
    return {"printed": len(used), "marginal_false": sum(1 for c in used if not c & truth),
            "conditional_false": sum(1 for c in used if not c <= truth), "true_term": true_term,
            "score": safe_score(model, X_test, y_test)}


def safe_score(model, X, y):
    try:
        return float(model.score(X, y))
    except ValueError:
        return float("nan")


STRESS_CELLS = [(law, dist, p) for law in ("pair", "triple", "pair_plus")
                for dist in ("unif", "gauss", "shifted") for p in (20, 100)]


def stress_items():
    return [("stress", f"{law}_{dist}_p{p}:{seed}") for law, dist, p in STRESS_CELLS for seed in range(10)]


def stress_expected(law):
    import sympy

    x0, x1, x2 = sympy.symbols("x0 x1 x2")
    return {"pair": [x0 * x1], "triple": [x0 * x1 * x2], "pair_plus": [x0 * x1, x2]}[law]


def feynman_items():
    import feynman_panel as fp

    return [("feynman", f"{e[0]}:{rep}") for e in fp.equations() for rep in range(5)]


def feynman_data(name, rep):
    import feynman_panel as fp

    name_, generator, n_cols, expected, note = [e for e in fp.equations() if e[0] == name][0]
    if rep == 0:
        X, X_test, y, y_test = fp.make_data(name, generator, n_cols)
    else:  # further draws by the panel's recipe with another seed
        rng = np.random.default_rng([zlib.crc32(name.encode()), rep])
        low, high = (0.1, 1.0) if name in ("gaussian", "relativistic_velocity") else (1.0, 5.0)
        X = rng.uniform(low, high, (500, n_cols))
        X_test = rng.uniform(low, high, (500, n_cols))
        signal = generator(X)
        y = signal + rng.normal(0, 1e-3 * np.std(signal), 500)
        y_test = generator(X_test)
    return X, y, X_test, y_test, expected


SEABORN = [("mpg", "mpg", ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"], None),
           ("diamonds", "price", ["carat", "depth", "table", "x", "y", "z"], 1400),
           ("penguins", "body_mass_g", ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"], None),
           ("tips", "tip", ["total_bill", "size"], None)]


def seaborn_frame(name):
    """A seaborn-data table, from run_benchmarks.py's cache, fetched into it if absent."""
    import pandas as pd

    cache = SEABORN_DATA / f"{name}.csv"
    if not cache.exists():
        SEABORN_DATA.mkdir(exist_ok=True)
        pd.read_csv(f"https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{name}.csv").to_csv(cache, index=False)
    return pd.read_csv(cache)


def suite_datasets():
    """name -> (X, y, tokens, problem_type)."""
    import run_benchmarks as rb
    from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes

    out = {d.name: (d.X, d.y, tuple(d.tokens), d.problem_type)
           for d in rb.synthetic_datasets() + rb.robustness_datasets()}
    d = load_diabetes()
    out["diabetes"] = (d.data, d.target, (), "regression")
    c = load_breast_cancer()
    out["breast_cancer"] = (c.data, c.target, (), "classification")
    for name, target, cols, cap in SEABORN:
        frame = seaborn_frame(name)[cols + [target]].dropna()
        if cap is not None and len(frame) > cap:
            frame = frame.sample(cap, random_state=0)
        out[name] = (frame[cols].to_numpy(float), frame[target].to_numpy(float), (), "regression")
    h = fetch_california_housing()
    out["california_housing"] = (h.data, h.target, (), "regression")
    return out


SUITE_NAMES = ["product", "ratio", "three_way", "log_linear", "sqrt_ratio", "quadratic", "inverse_square",
               "physics_kinetic", "sparse_10col", "purely_linear", "friedman1", "heavy_tail_t2",
               "distractors_noise05", "distractors_noise25", "distractors_noise50", "threshold_step",
               "small_n120", "small_n240", "diabetes", "breast_cancer", "mpg", "diamonds", "penguins", "tips",
               "california_housing"]


def suite_items():
    return [("suite", f"{name}:{split}") for name in SUITE_NAMES for split in range(5)]


def highdim_datasets():
    import pandas as pd

    out = {}
    cc = pd.read_csv(HIGHDIM_DATA / "communities_crime_numeric.csv")
    out["communities"] = (cc.iloc[:, :-1].to_numpy(float), cc.iloc[:, -1].to_numpy(float))
    sc = pd.read_csv(HIGHDIM_DATA / "superconductivity.csv").sample(5000, random_state=0)
    out["superconduct"] = (sc.drop(columns="critical_temp").to_numpy(float), sc["critical_temp"].to_numpy(float))
    tc = pd.read_csv(HIGHDIM_DATA / "tecator.csv")
    chan = [c for c in tc.columns if c.startswith("_") or c.startswith("absorbance")]
    out["tecator"] = (tc[chan].to_numpy(float), tc["fat"].to_numpy(float))
    eye = pd.read_csv(HIGHDIM_DATA / "eyedata.csv")
    out["eyedata"] = (eye.drop(columns="trim32").to_numpy(float), eye["trim32"].to_numpy(float))
    rb = pd.read_csv(HIGHDIM_DATA / "riboflavin.csv")
    out["riboflavin"] = (rb.iloc[:, 1:].to_numpy(float), rb.iloc[:, 0].to_numpy(float))
    return out


def highdim_items():
    return [("highdim", f"{name}:{split}") for name in
            ("communities", "superconduct", "tecator", "eyedata", "riboflavin") for split in range(5)]


def junk_items():
    return [("junk", f"{kind}:{seed}") for kind in ("independent", "correlated") for seed in range(20)]


def junk_data(kind, seed):
    rng = np.random.default_rng([11, seed])
    X = rng.uniform(1, 6, (1000, 12))
    if kind == "correlated":
        X[:, 8] = X[:, 1] + rng.uniform(-1.745, 1.745, 1000)
    signal = X[:, 0] * X[:, 1]
    y = signal + 0.25 * signal.std() * rng.standard_normal(1000)
    return X[:500], y[:500], X[500:], y[500:]


def split_rows(n, split, test_size=0.3):
    from sklearn.model_selection import train_test_split

    return train_test_split(np.arange(n), test_size=test_size, random_state=split)


_CACHE = {}


def load_item(study, name):
    """(X, y, X_test, y_test, problem_type, estimator kwargs, measure(model) -> dict)."""
    key, rep = name.rsplit(":", 1)
    rep = int(rep)
    if study == "designs":
        cell = [c for c in DESIGN_CELLS if design_name(*c) == key][0]
        X, y, Xt, yt = design_data(*cell, rep)
        return X, y, Xt, yt, "regression", {"target_fdr": 0.1, "random_state": rep}, \
            lambda m: design_measure(m, cell[0], Xt, yt)
    if study == "stress":
        import stress as st

        head, p = key.rsplit("_p", 1)
        law, dist = head.rsplit("_", 1)
        p = int(p)
        X, y, Xt, law_t = st.make_data(law, dist, p, 500, 0.1, rep)

        def measure(m):
            import feynman_panel as fp

            law_cols = {0, 1} if law == "pair" else {0, 1, 2}
            formulas = list(m.feature_formulas_)
            try:
                r2 = float(1 - np.mean((m.predict(Xt) - law_t) ** 2) / np.var(law_t))
            except ValueError:
                r2 = float("nan")
            return {"printed": len(formulas), "exact": bool(fp._exact_form(m, stress_expected(law))),
                    "distractor_terms": sum(1 for f in formulas if not columns_of(f) & law_cols),
                    "score": r2}
        return X, y, Xt, law_t, "regression", {"random_state": 0}, measure
    if study == "feynman":
        import feynman_panel as fp

        X, y, Xt, yt, expected = feynman_data(key, rep)

        def measure(m):
            r2 = safe_score(m, Xt, yt)
            exact = fp._exact_form(m, expected) if expected is not None else None
            return {"printed": len(m.features_), "score": r2, "solved": bool(r2 > 0.999), "exact": exact}
        return X, y, Xt, yt, "regression", {"max_depth": 3, "beam_width": 40, "random_state": 0}, measure
    if study == "suite":
        import run_benchmarks as rb

        if "suite" not in _CACHE:
            _CACHE["suite"] = suite_datasets()
        X_all, y_all, tokens, problem = _CACHE["suite"][key]
        train, test = split_rows(len(y_all), rep)

        def measure(m):
            formulas = list(m.feature_formulas_)
            return {"printed": len(formulas), "score": safe_score(m, X_all[test], y_all[test]),
                    "false_feature_rate": rb._false_feature_rate(formulas, tokens),
                    "recovered": rb._recovered(formulas, tokens)}
        return X_all[train], y_all[train], X_all[test], y_all[test], problem, {"random_state": rep}, measure
    if study == "highdim":
        if "highdim" not in _CACHE:
            _CACHE["highdim"] = highdim_datasets()
        X_all, y_all = _CACHE["highdim"][key]
        train, test = split_rows(len(y_all), rep)
        return X_all[train], y_all[train], X_all[test], y_all[test], "regression", {"random_state": rep}, \
            lambda m: {"printed": len(m.features_), "score": safe_score(m, X_all[test], y_all[test])}
    if study == "junk":
        X, y, Xt, yt = junk_data(key, rep)

        def measure(m):
            formulas = list(m.feature_formulas_)
            junk = [f for f in formulas if columns_of(f) - {0, 1}]
            return {"printed": len(formulas), "junk_terms": len(junk), "junk_equation": bool(junk),
                    "score": safe_score(m, Xt, yt)}
        return X, y, Xt, yt, "regression", {"random_state": rep}, measure
    raise ValueError(study)


def estimator(problem):
    from beamfeat import BeamFeatClassifier, BeamFeatRegressor

    return BeamFeatClassifier if problem == "classification" else BeamFeatRegressor


def items_of(study):
    return {"designs": design_items, "stress": stress_items, "feynman": feynman_items, "suite": suite_items,
            "highdim": highdim_items, "junk": junk_items}[study]()


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


HEAVY = {("california_housing", "input_test"), ("california_housing", "alpha_input_test")}
CURRENT = {}


def memory_guard(path, version, study, limit_mb):
    """Stop this process, recording the fit that ran, once its RSS passes limit_mb."""
    import os
    import threading

    def watch():
        while True:
            if _status("VmRSS") > limit_mb:
                if CURRENT:
                    with path.open("a") as handle:
                        handle.write(json.dumps({"version": version, "study": study, **CURRENT,
                                                 "error": f"memory guard: RSS over {limit_mb} MB"}) + "\n")
                print(f"memory guard: RSS over {limit_mb} MB during {CURRENT}", flush=True)
                os._exit(3)
            time.sleep(0.25)

    threading.Thread(target=watch, daemon=True).start()


def run_study(version, study, part, out, heavy=False, limit_mb=3000):
    index, count = (int(x) for x in part.split("/"))
    items = items_of(study)[index - 1::count]
    configs = dict(CONFIGS[version])
    if heavy:
        items = [i for i in items_of(study) if i[1].split(":")[0] == "california_housing"]
        configs = {c: k for c, k in configs.items() if c in {"input_test", "alpha_input_test"}}
        path = out / f"{study}_{version}_heavy.jsonl"
    else:
        path = out / f"{study}_{version}_{index}of{count}.jsonl"
    done = set()
    if path.exists():
        done = {(r["item"], r["config"]) for r in map(json.loads, path.open())}
    memory_guard(path, version, study, limit_mb)
    memo = SearchMemo()
    with path.open("a") as handle:
        for position, (_, name) in enumerate(items):
            X, y, Xt, yt, problem, base, measure = load_item(study, name)
            ran = False
            with memo:
                for config, kwargs in configs.items():
                    if (name, config) in done or (not heavy and (name.split(":")[0], config) in HEAVY):
                        continue
                    ran = True
                    record = {"version": version, "study": study, "item": name, "config": config}
                    if problem == "classification" and config in REGRESSION_ONLY:
                        record["skipped"] = "regression only"
                    else:
                        CURRENT.update(item=name, config=config)
                        try:
                            started = time.perf_counter()
                            model = estimator(problem)(**base, **kwargs).fit(X, y)
                            record["seconds_memo"] = time.perf_counter() - started
                            record.update(measure(model))
                            record["formulas"] = list(model.feature_formulas_)
                        except Exception as exc:  # noqa: BLE001
                            record["error"] = f"{type(exc).__name__}: {exc}"[:300]
                        CURRENT.clear()
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
            # The memo must not change a fit: refit the first item of each
            # job without it (not repeated when a job resumes).
            if position == 0 and ran and not heavy:
                for config, kwargs in configs.items():
                    if (problem == "classification" and config in REGRESSION_ONLY) or (name.split(":")[0], config) in HEAVY:
                        continue
                    try:
                        plain = estimator(problem)(**base, **kwargs).fit(X, y)
                    except Exception:  # noqa: BLE001
                        continue
                    memo_rows = [json.loads(line) for line in path.open()]
                    row = [r for r in memo_rows if r["item"] == name and r["config"] == config][-1]
                    if "error" not in row and row.get("formulas") != list(plain.feature_formulas_):
                        raise RuntimeError(f"memo changed the fit: {study} {name} {config}")
            print(f"{version} {study} {name} done", flush=True)


def knockoff_study(version, out):
    """Selector level, on the designs of benchmarks/knockoff_s_method.py (fixed-X, q = 0.1)."""
    import knockoff_s_method as ks

    from beamfeat.selection import KnockoffSelector

    methods = ["equicorrelated"] + (["ci"] if version == "0.6.1" else [])
    designs = {"block rho=0.5": lambda r: ks.block(r, 0.5), "block rho=0.7": lambda r: ks.block(r, 0.7),
               "block rho=0.9": lambda r: ks.block(r, 0.9), "factor": ks.factor, "constructed": ks.constructed}
    rows = []
    for offset in (1, 0):
        for name, make in designs.items():
            for method in methods:
                for trial in range(50):
                    rng = np.random.default_rng(ks.SEED_BASE + trial)
                    X, y, truth = make(rng)
                    kwargs = dict(target_fdr=0.1, construction="fixed", offset=offset, random_state=trial,
                                  warn_on_violation=False)
                    if version == "0.6.1":
                        kwargs["s_method"] = method
                    started = time.perf_counter()
                    selected = KnockoffSelector(**kwargs).select(X, y).selected
                    rows.append({"version": version, "design": name, "offset": offset, "s_method": method,
                                 "trial": trial, "seconds": time.perf_counter() - started,
                                 "fdp": float(np.mean(~truth[selected])) if selected.size else 0.0,
                                 "power": float(np.mean(np.isin(np.flatnonzero(truth), selected)))})
            print(version, "knockoff", offset, name, flush=True)
    (out / f"knockoff_{version}.json").write_text(json.dumps(rows))


# --------------------------------------------------------------------------- #
# Speed and memory: one fit in this process, peak RSS reset before it
# --------------------------------------------------------------------------- #


def _status(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024.0  # MB
    return float("nan")


def perf(version, item, config, limit_mb=3000):
    import os
    import threading

    study, name = item.split("/", 1)
    X, y, Xt, yt, problem, base, measure = load_item(study, name)
    model_class = estimator(problem)
    import beamfeat  # noqa: F401  (imported before the reset)

    Path("/proc/self/clear_refs").write_text("5")  # reset VmHWM to the current RSS
    baseline = _status("VmRSS")
    started = time.perf_counter()
    cpu = time.process_time()

    def watch():
        while True:
            if _status("VmRSS") > limit_mb:
                print(json.dumps({"version": version, "item": item, "config": config,
                                  "error": f"memory guard: RSS over {limit_mb} MB",
                                  "seconds_at_stop": time.perf_counter() - started}), flush=True)
                os._exit(3)
            time.sleep(0.25)

    threading.Thread(target=watch, daemon=True).start()
    model = model_class(**base, **CONFIGS[version][config]).fit(X, y)
    seconds, cpu_seconds = time.perf_counter() - started, time.process_time() - cpu
    peak = _status("VmHWM")
    print(json.dumps({"version": version, "item": item, "config": config, "seconds": seconds,
                      "cpu_seconds": cpu_seconds, "baseline_mb": baseline, "peak_mb": peak,
                      "fit_mb": peak - baseline, "printed": len(model.features_)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--study")
    parser.add_argument("--part", default="1/1")
    parser.add_argument("--perf")
    parser.add_argument("--config", default="default")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--heavy", action="store_true", help="only the California input_test fits")
    parser.add_argument("--limit-mb", type=int, default=3000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.perf:
        perf(args.version, args.perf, args.config, args.limit_mb)
    elif args.study == "knockoff":
        knockoff_study(args.version, args.out)
    else:
        import beamfeat
        print(f"{args.version}: beamfeat {beamfeat.__version__} from {Path(beamfeat.__file__).parent}", flush=True)
        try:
            run_study(args.version, args.study, args.part, args.out, args.heavy, args.limit_mb)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
