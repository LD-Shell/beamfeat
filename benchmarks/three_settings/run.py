"""The three settings of the "Choosing a setting" page, measured side by side.

    python run.py default parsimony_alpha input_test both
    python run.py rescore

Each setting named on the command line is fitted on every item below and
written to results/<setting>.jsonl, one record per fit; a rerun skips the
items a file already holds. Run one process per setting to use more cores.
`rescore` then scores again, on the rows every setting predicts, the splits
where some setting left test rows without a prediction.

- sweep: y = x0/x1 among 12 columns, 10 of them unrelated, with Gaussian
  noise at 0.1, 0.3, 0.5 and 1.0 times the signal's standard deviation;
  1,000 training rows, 10 seeds, scored against the noiseless target on
  2,000 new rows.
- suite, highdim: the datasets, splits and seeds of
  benchmarks/release_comparison/ (five 70/30 splits each).
- paper_suite, paper_feynman, paper_comparison, paper_highdim: the paper's
  own scripts and settings: run_benchmarks.py's 25 datasets (depth 2, beam
  40, one 70/30 split, as archive_check.py fits them), the Feynman panel, the
  comparison study of benchmarks/independent/ and the high-dimensional study
  of benchmarks/additional_experiments/ (five 75/25 splits each; the five
  datasets the repository stores).
"""
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "release_comparison"))
sys.path.insert(0, str(BENCH))
warnings.filterwarnings("ignore")

import harness as h  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import beamfeat  # noqa: E402
from beamfeat import BeamFeatRegressor  # noqa: E402

SETTINGS = {
    "default": {},
    "parsimony_alpha": {"parsimony_alpha": 0.05},
    "input_test": {"input_test": True},
    "both": {"parsimony_alpha": 0.05, "input_test": True},
}
NOISE = (0.1, 0.3, 0.5, 1.0)
COMPARISON = ["concrete", "wine_red", "boston", "diabetes", "three_way:a*b/c", "kinetic:m*v^2/2",
              "sparse10:a*b", "linear_ctrl", "friedman1"]
PAPER_HIGHDIM = ["communities_p100", "superconduct_p81", "tecator_p100", "eyedata_p200", "riboflavin_p4088"]
ENV = {"beamfeat": beamfeat.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__}


def junk_terms(model, tokens):
    """Terms reading a column outside the generating ones, or None when those are unknown."""
    if not tokens:
        return None
    return sum(bool(node.columns() - set(tokens)) for node in model.features_)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def items():
    """(study, item, fit) for every fit; fit(kwargs) returns the record's measures."""
    for level in NOISE:
        for seed in range(10):
            def fit(kw, level=level, seed=seed):
                rng = np.random.default_rng(seed)
                X = rng.uniform(1, 3, (1000, 12))
                signal = X[:, 0] / X[:, 1]
                y = signal + rng.normal(0, level * signal.std(), 1000)
                X_new = rng.uniform(1, 3, (2000, 12))
                model = BeamFeatRegressor(random_state=seed, **kw).fit(X, y)
                return model, {"score": float(r2_score(X_new[:, 0] / X_new[:, 1], model.predict(X_new))),
                               "junk_terms": junk_terms(model, ("x0", "x1"))}
            yield "sweep", f"noise{level}:{seed}", fit

    for study in ("suite", "highdim"):
        for _, name in h.items_of(study):
            def fit(kw, study=study, name=name):
                X, y, _, _, problem, base, measure = h.load_item(study, name)
                model = h.estimator(problem)(**base, **kw).fit(X, y)
                record = measure(model)
                tokens = h._CACHE["suite"][name.split(":")[0]][2] if study == "suite" else ()
                record["junk_terms"] = junk_terms(model, tokens)
                return model, record
            yield study, name, fit

    for name in h.SUITE_NAMES:
        def fit(kw, name=name):
            X, y, X_test, y_test, problem, _, measure = h.load_item("suite", f"{name}:0")
            model = h.estimator(problem)(max_depth=2, beam_width=40, selector="permutation",
                                         parsimony="forward", target_fdr=0.1, random_state=0, **kw).fit(X, y)
            record = measure(model)
            record["score"] = float(model.score(X_test, y_test))
            record["junk_terms"] = junk_terms(model, h._CACHE["suite"][name][2])
            return model, record
        yield "paper_suite", name, fit

    panel = json.loads((BENCH / "feynman_results.json").read_text())[0]
    for row in panel["rows"]:
        def fit(kw, equation=row["equation"]):
            X, y, _, _, _, base, measure = h.load_item("feynman", f"{equation}:0")
            model = h.estimator("regression")(**base, **kw).fit(X, y)
            return model, measure(model)
        yield "paper_feynman", row["equation"], fit

    for study, path, which, names in (
            ("paper_comparison", BENCH / "independent" / "bench.py", "all", COMPARISON),
            ("paper_highdim", BENCH / "additional_experiments" / "bench.py", "highdim", PAPER_HIGHDIM)):
        for name in names:
            for split in range(5):
                def fit(kw, path=path, which=which, name=name, split=split):
                    if which not in _BENCH_DATA:
                        _BENCH_DATA[which] = module(path, f"bench_{which}").get_datasets(which)
                    X, y, _ = _BENCH_DATA[which][name]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=split)
                    model = BeamFeatRegressor(random_state=0, **kw).fit(X_train, y_train)
                    return model, {"score": float(r2_score(y_test, model.predict(X_test)))}
                yield study, f"{name}:{split}", fit


_BENCH_DATA = {}


def main(settings):
    for setting in settings:
        kw = SETTINGS[setting]
        path = HERE / "results" / f"{setting}.jsonl"
        path.parent.mkdir(exist_ok=True)
        done = set()
        if path.exists():
            done = {(r["study"], r["item"]) for r in map(json.loads, path.open())}
        with path.open("a") as out:
            for study, item, fit in items():
                if (study, item) in done:
                    continue
                start = time.perf_counter()
                model, record = fit(kw)
                record.update(study=study, item=item, setting=setting, seconds=time.perf_counter() - start,
                              terms=len(model.features_), fdr_controlled=bool(model.fdr_controlled_),
                              formulas=list(model.feature_formulas_), **ENV)
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(setting, study, item, f"{record['seconds']:.1f}s", flush=True)


def rescore():
    """Score again, on the rows every setting predicts, the suite and highdim splits where a setting
    left test rows without a prediction (a formula undefined there), as
    benchmarks/release_comparison/rescore.py does; writes results/rescore.json."""
    records = {s: {(r["study"], r["item"]): r for r in map(json.loads, (HERE / "results" / f"{s}.jsonl").open())}
               for s in SETTINGS}
    items = sorted({key for s in SETTINGS for key, r in records[s].items()
                    if key[0] in ("suite", "highdim") and r["score"] != r["score"]})
    out = []
    for study, item in items:
        X, y, X_test, y_test, problem, base, _ = h.load_item(study, item)
        models = {s: h.estimator(problem)(**base, **kw).fit(X, y) for s, kw in SETTINGS.items()}
        predictions = {s: np.asarray(m.predict(X_test)) for s, m in models.items()}
        defined = np.ones(len(y_test), dtype=bool)
        if problem == "regression":
            for p in predictions.values():
                defined &= np.isfinite(p)
        for s, p in predictions.items():
            assert len(models[s].features_) == records[s][(study, item)]["terms"]  # the same fit
            score = (r2_score(y_test[defined], p[defined]) if problem == "regression"
                     else float(np.mean(p[defined] == y_test[defined])))
            out.append({"study": study, "item": item, "setting": s, "score": float(score),
                        "rows": int(defined.sum()), "of": int(len(y_test))})
            print(study, item, s, f"{score:.4f}", f"{int(defined.sum())} of {len(y_test)} rows", flush=True)
    (HERE / "results" / "rescore.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["rescore"]:
        rescore()
    else:
        main(sys.argv[1:] or list(SETTINGS))
