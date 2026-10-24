"""Calibration of ``parsimony_holdout`` and of its conditional re-test.

What fraction of the printed terms is false under each parsimony setting,
how often the true term is printed, how long the equation is, how often the
re-test falls back to the screened set, and how well the fit predicts rows
no step of it saw. ``PROVENANCE.md`` records what was fixed in advance, what
was changed after a smoke run and why, and what a rerun reproduces;
``REPORT.md`` reads the results.

Designs (all inputs U(1, 3), independent; 2,000 test rows drawn beside the
training rows and never shown to the fit):

    null      y ~ N(0, 1) independent of X; n in {300, 1000}, p in {5, 20}
    planted   y = a*b/c + noise; a, b, c are x0, x1, x2, followed by k in
              {5, 20} distractors; n in {300, 1000}; the noise's standard
              deviation is the signal's (signal to noise 1, "snr1") or a
              third of it ("snr3"), from the same standard normal draws
    proxy     the planted replicate with one more column, a + U(-w, w),
              which correlates 0.9 with a; every other draw is shared

Settings, each at target_fdr 0.1 and 0.2, every other option at its default:

    forward          parsimony="forward" (the default)
    none             parsimony=None, the screened set
    holdout_0.3/0.5  parsimony_holdout 0.3 / 0.5, certify_test="marginal"
    conditional_...  the same with certify_test="conditional"

A printed term is false in the marginal sense when it is built only from
columns outside {a, b, c}, and in the conditional sense when it uses any
column outside them. It is the true term when it uses exactly a, b and c and
correlates at least 0.999 in absolute value with a*b/c on the test rows.

The search does not depend on the setting or the level, so it runs once per
replicate and is reused by its twelve fits (``BeamSearch.run`` memoised on its
inputs). The first ``--check-seeds`` replicates of every cell are refitted
without the memo and must match it to the last bit, or the run stops.

Run from the repository root, with this checkout's ``src`` importable:

    python benchmarks/parsimony_holdout/calibration.py              # 200 seeds, resumable
    python benchmarks/parsimony_holdout/calibration.py --tabulate   # tables from the records
    python benchmarks/parsimony_holdout/calibration.py --seeds 3 --out /tmp/smoke
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import math
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

TRUE_COLUMNS = frozenset({"x0", "x1", "x2"})
# Standard deviation of a*b/c for a, b, c independent U(1, 3):
# E[(ab/c)^2] = (13/3)^2 / 3 and E[ab/c] = 2 * 2 * log(3) / 2.
SIGNAL_SD = math.sqrt(169 / 27 - (2 * math.log(3)) ** 2)
# a + U(-w, w) correlates 1 / sqrt(1 + w^2) with a, which is 0.9 here.
PROXY_HALF_WIDTH = math.sqrt(1 / 0.81 - 1)
N_TEST = 2000
LEVELS = (0.1, 0.2)
SETTINGS = {
    "forward": {},
    "none": {"parsimony": None},
    "holdout_0.3": {"parsimony_holdout": 0.3},
    "holdout_0.5": {"parsimony_holdout": 0.5},
    "conditional_0.3": {"parsimony_holdout": 0.3, "certify_test": "conditional"},
    "conditional_0.5": {"parsimony_holdout": 0.5, "certify_test": "conditional"},
}
# (design, rows, columns or distractors, signal-to-noise ratio)
CELLS = [("null", n, p, None) for n in (300, 1000) for p in (5, 20)] + [
    (design, n, k, snr)
    for design in ("planted", "proxy")
    for snr in (1, 3)
    for n in (300, 1000)
    for k in (5, 20)
]
GROUPS = ["null", "planted_snr1", "planted_snr3", "proxy_snr1", "proxy_snr3"]
FIELDS = [
    "cell", "setting", "q", "seed", "screened", "printed", "marginal_false",
    "conditional_false", "true_term", "retest_kept", "retest_warned", "fallback", "r2_test",
]


def cell_name(design: str, n: int, width: int, snr: int | None) -> str:
    if design == "null":
        return f"null_n{n}_p{width}"
    return f"{design}_snr{snr}_n{n}_k{width}"


def group_of(cell: str) -> str:
    return cell.split("_n")[0]


def make_data(design: str, n: int, width: int, snr: int | None, seed: int):
    """Training and test rows of one replicate.

    The planted and proxy replicates of a seed share every draw, at both
    noise levels: the noise is one standard normal vector, scaled, and the
    proxy column is drawn last.
    """
    total = n + N_TEST
    if design == "null":
        rng = np.random.default_rng([0, n, width, seed])
        X = rng.uniform(1.0, 3.0, (total, width))
        y = rng.normal(0.0, 1.0, total)
    else:
        rng = np.random.default_rng([1, n, width, seed])
        X = rng.uniform(1.0, 3.0, (total, 3 + width))
        noise = rng.standard_normal(total)
        y = X[:, 0] * X[:, 1] / X[:, 2] + (SIGNAL_SD / snr) * noise
        if design == "proxy":
            proxy = X[:, 0] + rng.uniform(-PROXY_HALF_WIDTH, PROXY_HALF_WIDTH, total)
            X = np.column_stack([X, proxy])
    return X[:n], y[:n], X[n:], y[n:]


def measure(model, design: str, kwargs: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """The record of one fit: what it printed, judged against the design."""
    from beamfeat.expression import Evaluator

    truth = frozenset() if design == "null" else TRUE_COLUMNS
    columns = [frozenset(node.columns()) for node in model.features_]
    true_term: int | str = ""
    if design != "null":
        true_term = 0
        evaluator = Evaluator({f"x{j}": X_test[:, j] for j in range(X_test.shape[1])})
        signal = X_test[:, 0] * X_test[:, 1] / X_test[:, 2]
        for node, used in zip(model.features_, columns):
            values = evaluator.transform_values(node) if used == truth else None
            if values is not None and abs(np.corrcoef(values, signal)[0, 1]) >= 0.999:
                true_term = 1
                break
    screened = 0 if model.selection_result_ is None else int(model.selection_result_.n_selected)
    certification = model.certification_result_
    retest = "parsimony_holdout" in kwargs
    try:
        r2 = float(model.score(X_test, y_test))
    except ValueError:  # a printed term undefined on some test row
        r2 = float("nan")
    return {
        "screened": screened,
        "printed": len(columns),
        "marginal_false": sum(1 for used in columns if not used & truth),
        "conditional_false": sum(1 for used in columns if not used <= truth),
        "true_term": true_term,
        "retest_kept": "" if certification is None else int(certification.n_selected),
        "retest_warned": "" if certification is None else int(bool(certification.warnings_raised)),
        "fallback": "" if not retest else int(
            screened > 0 and (certification is None or certification.n_selected == 0)
        ),
        "r2_test": f"{r2:.6g}",
    }


def fit(X, y, q: float, seed: int, kwargs: dict):
    from beamfeat import BeamFeatRegressor

    return BeamFeatRegressor(target_fdr=q, random_state=seed, **kwargs).fit(X, y)


def fingerprint(model, X_test: np.ndarray) -> tuple:
    """Everything a fit returns that the records depend on, bit for bit."""
    certification = model.certification_result_
    return (
        tuple(model.feature_formulas_),
        np.asarray(model.coef_).tobytes(),
        np.asarray(model.intercept_).tobytes(),
        model.predict(X_test).tobytes(),
        None if certification is None else np.asarray(certification.p_values).tobytes(),
    )


def _search_module():
    """The module whose ``BeamSearch`` a fit looks up: ``beamfeat._base`` from 0.7.0, ``beamfeat.estimators`` before."""
    import importlib
    import importlib.util

    if importlib.util.find_spec("beamfeat._base") is not None:
        return importlib.import_module("beamfeat._base")
    return importlib.import_module("beamfeat.estimators")


class SearchMemo:
    """Memoises ``BeamSearch.run`` on its inputs, within one replicate.

    Installed by replacing the name the fit looks up (see _search_module), so the
    estimator runs unmodified otherwise. The key covers the search's
    constructor arguments and every array and option passed to ``run``.
    """

    def __init__(self) -> None:
        self.module = _search_module()
        self.original = self.module.BeamSearch
        self.results: dict[str, object] = {}
        memo = self

        class MemoisedSearch(self.original):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._memo_kwargs = kwargs

            def run(self, data, target, *args, **kwargs):
                digest = hashlib.sha256(
                    repr((sorted(self._memo_kwargs.items()), args, sorted(kwargs.items()))).encode()
                )
                for name, column in data.items():
                    digest.update(str(name).encode())
                    digest.update(np.ascontiguousarray(column, dtype=np.float64).tobytes())
                digest.update(np.ascontiguousarray(target, dtype=np.float64).tobytes())
                key = digest.hexdigest()
                if key not in memo.results:
                    memo.results[key] = super().run(data, target, *args, **kwargs)
                return memo.results[key]

        self.memoised = MemoisedSearch

    def __enter__(self) -> SearchMemo:
        self.results.clear()
        self.module.BeamSearch = self.memoised
        return self

    def __exit__(self, *exc) -> None:
        self.module.BeamSearch = self.original
        self.results.clear()


def run_cell(cell: tuple, seeds: int, check_seeds: int, parts: Path, memo: SearchMemo) -> int:
    """Fit every setting and level on each replicate of one cell; return fits checked."""
    design = cell[0]
    name = cell_name(*cell)
    path = parts / f"{name}.csv"
    per_seed = len(SETTINGS) * len(LEVELS)
    done: dict[int, list[dict]] = {}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                done.setdefault(int(row["seed"]), []).append(row)
        done = {seed: rows for seed, rows in done.items() if len(rows) == per_seed}
    with path.open("w", newline="") as handle:  # keep complete replicates only
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for seed in sorted(done):
            writer.writerows(done[seed])
    examples_path = parts / f"{name}.examples.json"
    examples = json.loads(examples_path.read_text()) if examples_path.exists() else {}

    checked, started = 0, time.perf_counter()
    for seed in range(seeds):
        if seed in done:
            continue
        X, y, X_test, y_test = make_data(*cell, seed)
        rows = []
        with memo:
            fits = {}
            for setting, kwargs in SETTINGS.items():
                for q in LEVELS:
                    model = fit(X, y, q, seed, kwargs)
                    fits[setting, q] = model
                    rows.append({"cell": name, "setting": setting, "q": q, "seed": seed,
                                 **measure(model, design, kwargs, X_test, y_test)})
        if seed < check_seeds:
            for (setting, q), model in fits.items():
                plain = fit(X, y, q, seed, SETTINGS[setting])
                if fingerprint(plain, X_test) != fingerprint(model, X_test):
                    raise RuntimeError(f"memoised search changed the fit: {name} seed {seed} {setting} q={q}")
                checked += 1
                examples.setdefault(f"{setting} q={q}", {})[str(seed)] = list(model.feature_formulas_)
            examples_path.write_text(json.dumps(examples, indent=1))
        with path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerows(rows)
        if (seed + 1) % 50 == 0:
            print(f"  {name}: seed {seed + 1}/{seeds}, {time.perf_counter() - started:.0f} s", flush=True)
    return checked


def provenance() -> dict:
    import scipy
    import sklearn

    import beamfeat

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    # Relative to the repository when imported from it, so that the record
    # names the checkout rather than the machine.
    imported = Path(beamfeat.__file__).resolve().parent
    if imported.is_relative_to(ROOT):
        imported = imported.relative_to(ROOT)
    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "beamfeat": beamfeat.__version__,
        "beamfeat_path": str(imported),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.platform(),
    }


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def load(out: Path) -> list[dict]:
    """Every record: the run's parts if present, else the archived file."""
    parts = sorted((out / "parts").glob("*.csv")) if (out / "parts").exists() else []
    if parts:
        rows = []
        for path in parts:
            with path.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
        return rows
    with gzip.open(out / "records.csv.gz", "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    se = float(np.std(array, ddof=1) / np.sqrt(array.size)) if array.size > 1 else float("nan")
    return float(np.mean(array)), se


MEASURES = ("fdr_marginal", "fdr_conditional", "any_false_conditional", "terms", "screened",
            "power", "fallback", "r2_test")


def summarise(rows: list[dict]) -> dict:
    """Per cell, setting and level: means with Monte Carlo standard errors."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["cell"], row["setting"], float(row["q"])), []).append(row)
    cells = {}
    for (cell, setting, q), members in sorted(groups.items()):
        printed = [int(r["printed"]) for r in members]
        entry = {
            "fits": len(members),
            "fdr_marginal": _mean_se([int(r["marginal_false"]) / max(p, 1) for r, p in zip(members, printed)]),
            "fdr_conditional": _mean_se(
                [int(r["conditional_false"]) / max(p, 1) for r, p in zip(members, printed)]
            ),
            "any_false_conditional": _mean_se([float(int(r["conditional_false"]) > 0) for r in members]),
            "terms": _mean_se(printed),
            "screened": _mean_se([int(r["screened"]) for r in members]),
        }
        if members[0]["true_term"] != "":
            entry["power"] = _mean_se([int(r["true_term"]) for r in members])
        if members[0]["fallback"] != "":
            entry["fallback"] = _mean_se([int(r["fallback"]) for r in members])
        r2 = [float(r["r2_test"]) for r in members]
        finite = [value for value in r2 if math.isfinite(value)]
        entry["r2_test"] = _mean_se(finite)
        entry["r2_test_median"] = float(np.median(finite)) if finite else float("nan")
        entry["r2_undefined"] = len(r2) - len(finite)
        cells[f"{cell}|{setting}|{q}"] = entry

    # Each group pooled over its four cells with equal weight: the mean of
    # the cell means, whose standard error combines theirs.
    pooled = {}
    for group in GROUPS:
        for setting in SETTINGS:
            for q in LEVELS:
                keys = [key for key in cells if group_of(key.split("|")[0]) == group
                        and key.endswith(f"|{setting}|{q}")]
                if not keys:
                    continue
                entry = {"cells": len(keys), "fits": sum(cells[key]["fits"] for key in keys)}
                for name in MEASURES:
                    if name not in cells[keys[0]]:
                        continue
                    means = [cells[key][name][0] for key in keys]
                    ses = [cells[key][name][1] for key in keys]
                    entry[name] = (float(np.mean(means)), float(np.sqrt(np.sum(np.square(ses)))) / len(keys))
                entry["r2_undefined"] = sum(cells[key]["r2_undefined"] for key in keys)
                pooled[f"{group}|{setting}|{q}"] = entry
    return {"cells": cells, "pooled": pooled}


def _cellfmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    mean, se = value
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def _rate(entry: dict) -> str:
    return "—" if "fallback" not in entry else f"{entry['fallback'][0]:.3f}"


def tables(summary: dict) -> str:
    """The report's tables, as Markdown."""
    lines = []
    for q in LEVELS:
        lines += [
            f"### Pooled over the four cells of each design, q = {q}",
            "",
            "| design | setting | FDR, marginal | FDR, conditional | power | terms | fallback | held-out R² |",
            "| :-- | :-- | --: | --: | --: | --: | --: | --: |",
        ]
        for group in GROUPS:
            for setting in SETTINGS:
                entry = summary["pooled"].get(f"{group}|{setting}|{q}")
                if entry is None:
                    continue
                lines.append(
                    f"| {group} | {setting} | {_cellfmt(entry['fdr_marginal'])} | "
                    f"{_cellfmt(entry['fdr_conditional'])} | {_cellfmt(entry.get('power'))} | "
                    f"{entry['terms'][0]:.1f} | {_rate(entry)} | {_cellfmt(entry['r2_test'])} |"
                )
        lines.append("")
    for q in LEVELS:
        lines += [
            f"### Per cell, q = {q}",
            "",
            "| cell | setting | FDR, marginal | FDR, conditional | power | terms | screened | fallback | held-out R² |",
            "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |",
        ]
        for cell in CELLS:
            for setting in SETTINGS:
                entry = summary["cells"].get(f"{cell_name(*cell)}|{setting}|{q}")
                if entry is None:
                    continue
                lines.append(
                    f"| {cell_name(*cell)} | {setting} | {_cellfmt(entry['fdr_marginal'])} | "
                    f"{_cellfmt(entry['fdr_conditional'])} | {_cellfmt(entry.get('power'))} | "
                    f"{entry['terms'][0]:.1f} | {entry['screened'][0]:.1f} | {_rate(entry)} | "
                    f"{_cellfmt(entry['r2_test'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def tabulate(out: Path, run_info: dict | None = None) -> None:
    rows = load(out)
    summary = summarise(rows)
    info_path = out / "summary.json"
    if run_info is None and info_path.exists():
        run_info = json.loads(info_path.read_text()).get("run")
    info_path.write_text(json.dumps({"run": run_info, **summary}, indent=1) + "\n")
    (out / "tables.md").write_text(tables(summary) + "\n")
    if (out / "parts").exists():
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["cell"], int(r["seed"]), r["setting"], r["q"])))
        # mtime 0, so that the archive is a function of the records alone
        with gzip.GzipFile(out / "records.csv.gz", "wb", mtime=0) as handle:
            handle.write(buffer.getvalue().encode())
        examples = {}
        for path in sorted((out / "parts").glob("*.examples.json")):
            examples[path.name.removesuffix(".examples.json")] = json.loads(path.read_text())
        (out / "examples.json").write_text(json.dumps(examples, indent=1) + "\n")
    print(f"{len(rows)} records -> {info_path}, {out / 'tables.md'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=200, help="replicates per cell (default 200)")
    parser.add_argument("--check-seeds", type=int, default=2,
                        help="replicates per cell refitted without the search memo (default 2)")
    parser.add_argument("--cells", default="", help="comma-separated cell names to run (default all)")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--tabulate", action="store_true", help="only rebuild the tables from the records")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run although tracked files differ from the commit")
    args = parser.parse_args(argv)

    if args.tabulate:
        tabulate(args.out)
        return

    warnings.filterwarnings("ignore")
    logging.getLogger("beamfeat").setLevel(logging.ERROR)
    info = provenance()
    if info["tracked_changes"] and not args.allow_dirty:
        sys.exit("tracked files differ from the commit; commit first, or pass --allow-dirty")
    wanted = {name for name in args.cells.split(",") if name}
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    memo = SearchMemo()
    started = time.time()
    print(f"calibration: {args.seeds} seeds per cell, settings {list(SETTINGS)}, levels {LEVELS}")
    print(json.dumps(info), flush=True)
    clock = parts / "wall_seconds.txt"
    earlier = float(clock.read_text()) if clock.exists() else 0.0
    for cell in CELLS:
        name = cell_name(*cell)
        if wanted and name not in wanted:
            continue
        cell_started = time.perf_counter()
        checked = run_cell(cell, args.seeds, args.check_seeds, parts, memo)
        clock.write_text(f"{earlier + time.time() - started:.0f}")
        print(f"{name}: done in {time.perf_counter() - cell_started:.0f} s, "
              f"{checked} fits matched without the memo", flush=True)
    run_info = {
        **info,
        "seeds": args.seeds,
        "check_seeds": args.check_seeds,
        "fits_checked_without_memo": sum(
            len(seeds)
            for path in parts.glob("*.examples.json")
            for seeds in json.loads(path.read_text()).values()
        ),
        "settings": SETTINGS,
        "levels": list(LEVELS),
        "cells": [cell_name(*cell) for cell in CELLS],
        "signal_sd": SIGNAL_SD,
        "proxy_half_width": PROXY_HALF_WIDTH,
        "n_test": N_TEST,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "wall_seconds": round(earlier + time.time() - started),
    }
    tabulate(args.out, run_info)


if __name__ == "__main__":
    main()
