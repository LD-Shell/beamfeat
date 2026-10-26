"""What a search on a sample of its rows (``max_search_rows``) costs a weak term.

200,000 rows of ten columns on [1, 5]; the target is ``x0*x1/x2 + w*x4*x5``
plus Gaussian noise as large as the signal (signal to noise 1). ``w`` gives
the weak term 2%, 0.5% or 0.1% of the signal's variance. The default 50/50
split leaves 100,000 search rows: the search runs on all of them, or on
30,000, 10,000 or 3,000 of them, and the screen on the 100,000 selection rows
in every case. Five seeds per cell, every other setting at its default.

Recorded per fit: whether the weak term is among the search's expressions,
whether a printed term is proportional to ``x4*x5`` and one to
``x0*x1/x2``, whether the printed terms read ``x4`` and ``x5`` in any form,
R^2 against the noiseless signal on 20,000 new rows, the printed formulas
and the fit's wall time. On [1, 5] a product of two independent columns is
close to a sum of functions of each, so ``x4**2`` beside ``x5**2`` can carry
most of the weak term; R^2 against the signal measures how much of it a fit
recovers, against ``1 - share`` for a fit that recovers none of it. ``--memory`` instead fits one seed per cell, each in a fresh
process, and records the peak resident memory of the fit.

Usage::

    python run.py --out results/records.jsonl
    python run.py --memory --out results/memory.jsonl
    python run.py --report results > REPORT_TABLES.md
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

N_ROWS, N_TEST, N_COLUMNS = 200_000, 20_000, 10
SHARES = (0.02, 0.005, 0.001)
SAMPLES = (None, 30_000, 10_000, 3_000)
SEEDS = range(5)


def _weight(share: float) -> float:
    """The weak term's coefficient for its share of the signal's variance."""
    rng = np.random.default_rng(12345)
    X = rng.uniform(1, 5, (1_000_000, 6))
    strong, weak = X[:, 0] * X[:, 1] / X[:, 2], X[:, 4] * X[:, 5]
    return float(np.sqrt(share / (1 - share) * np.var(strong) / np.var(weak)))


def make_data(share: float, seed: int):
    rng = np.random.default_rng(1000 + seed)
    X = rng.uniform(1, 5, (N_ROWS + N_TEST, N_COLUMNS))
    signal = X[:, 0] * X[:, 1] / X[:, 2] + _weight(share) * X[:, 4] * X[:, 5]
    y = signal + rng.normal(0, np.std(signal), len(signal))
    return X[:N_ROWS], y[:N_ROWS], X[N_ROWS:], signal[N_ROWS:]


def fit_one(share: float, sample: int | None, seed: int) -> dict:
    import sympy
    from feynman_panel import _proportional

    from beamfeat import BeamFeatRegressor

    X, y, X_test, signal_test = make_data(share, seed)
    started = time.perf_counter()
    model = BeamFeatRegressor(random_state=seed, max_search_rows=sample).fit(X, y)
    seconds = time.perf_counter() - started
    x = sympy.symbols("x0:10")
    weak, strong = x[4] * x[5], x[0] * x[1] / x[2]

    def plain(expr):
        """The expression in symbols without assumptions, as ``weak`` and ``strong`` are."""
        return expr.subs({symbol: sympy.Symbol(symbol.name) for symbol in expr.free_symbols})

    terms = [plain(term) for term in model.to_sympy()] if model.features_ else []
    searched = [node.columns() for node in model.search_result_.nodes]
    printed_columns = set().union(*(node.columns() for node in model.features_)) if model.features_ else set()
    return {
        "share": share, "sample": sample, "seed": seed, "seconds": seconds,
        "weak_searched": any(
            columns == {"x4", "x5"} and _proportional(plain(node.to_sympy()), weak)
            for columns, node in zip(searched, model.search_result_.nodes)
        ),
        "weak_printed": any(_proportional(term, weak) for term in terms),
        "strong_printed": any(_proportional(term, strong) for term in terms),
        "weak_columns_printed": {"x4", "x5"} <= printed_columns,
        "printed": len(model.features_),
        "formulas": list(model.feature_formulas_),
        "r2_signal": float(1 - np.mean((model.predict(X_test) - signal_test) ** 2) / np.var(signal_test)),
        "fdr_controlled": bool(model.fdr_controlled_),
    }


def run(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["share"], r["sample"], r["seed"]) for r in map(json.loads, out.open())}
    with out.open("a") as handle:
        for seed in SEEDS:
            for share in SHARES:
                for sample in SAMPLES:
                    if (share, sample, seed) in done:
                        continue
                    record = fit_one(share, sample, seed)
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    print(json.dumps(record), flush=True)


def memory(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        for share in SHARES[1:2]:
            for sample in SAMPLES:
                result = subprocess.run(
                    [sys.executable, __file__, "--one", str(share), str(sample), "0"],
                    check=True, capture_output=True, text=True,
                )
                handle.write(result.stdout.strip().splitlines()[-1] + "\n")
                handle.flush()


def one(share: str, sample: str, seed: str) -> None:
    """One fit in this process, with its peak memory above the data's."""
    share_, sample_, seed_ = float(share), None if sample == "None" else int(sample), int(seed)
    make_data(share_, seed_)  # the data alone, so that the baseline includes them
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = fit_one(share_, sample_, seed_)
    record["peak_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    record["data_mb"] = baseline / 1024
    print(json.dumps(record))


def report(folder: Path) -> None:
    records = [json.loads(line) for line in (folder / "records.jsonl").open()]
    print("| weak share | search rows | x4*x5 searched | x4*x5 printed | x4, x5 printed | x0*x1/x2 printed "
          "| R² vs signal | 1 - share | seconds |")
    print("| --: | --: | --: | --: | --: | --: | --: | --: | --: |")
    for share in SHARES:
        for sample in SAMPLES:
            cell = [r for r in records if r["share"] == share and r["sample"] == sample]
            if not cell:
                continue
            n = len(cell)
            print(f"| {share:.1%} | {sample or 100_000:,} | "
                  f"{sum(r['weak_searched'] for r in cell)} of {n} | "
                  f"{sum(r['weak_printed'] for r in cell)} of {n} | "
                  f"{sum(r['weak_columns_printed'] for r in cell)} of {n} | "
                  f"{sum(r['strong_printed'] for r in cell)} of {n} | "
                  f"{np.mean([r['r2_signal'] for r in cell]):.4f} | {1 - share:.3f} | "
                  f"{np.median([r['seconds'] for r in cell]):.0f} |")
    path = folder / "memory.jsonl"
    if path.exists():
        print()
        print("| search rows | peak memory of the fit (MB) | seconds |")
        print("| --: | --: | --: |")
        for r in map(json.loads, path.open()):
            print(f"| {r['sample'] or 100_000:,} | {r['peak_mb'] - r['data_mb']:,.0f} | {r['seconds']:.0f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results" / "records.jsonl")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--one", nargs=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.one:
        one(*args.one)
    elif args.report:
        report(args.report)
    elif args.memory:
        memory(args.out)
    else:
        run(args.out)
