"""Stress test of the beam search on laws whose building blocks it cannot rank.

The search keeps, at each depth, the ``beam_width`` candidates that correlate
best with the target on the search rows. A law can need an intermediate that
does not correlate with the target at all. With centred, symmetric inputs:

    pair         y = x0*x1          x0 and x1 are each uncorrelated with y
    triple       y = x0*x1*x2       every pairwise product is uncorrelated with y
    pair_plus    y = x0*x1 + x2     x0 and x1 uncorrelated; x2 visible

Two things can then go wrong. The intermediate can rank among noise and leave
the beam before it is combined (pruning), or, once ``max_candidates_per_depth``
binds at depth 1 (from about 258 inputs at the defaults), the inputs are
proposed in order of their score and a pair of zero-score inputs may never be
proposed at all. This script measures how often each happens, and what the
fitted equation looks like when it does.

Inputs (independent columns; x0, x1, x2 carry the law, the rest are
distractors, p columns in all):

    unif       U(-1, 1)        centred
    gauss      N(0, 1)         centred
    shifted    U(0.5, 2.5)     the control: every factor correlates with y

Noise is Gaussian with 1% or 10% of the law's standard deviation, computed
exactly from the input moments. Every fit is the estimator as a user calls it,
``BeamFeatRegressor()`` or ``BeamFeatRegressor(beam_width=200)``, with every
other option at its default (``random_state=0`` among them), on n training
rows; 2,000 test rows drawn beside them are never shown to the fit, and the
held-out R^2 is scored against the noiseless law. ``--kwargs`` passes further
options to every fit, under the name ``--setting``, to measure an option on
the same grid.

The grid (``GRID``) is a base, the default beam of 50 with 500 rows and 10%
noise, crossed with every law, input distribution and width p in {5, 20, 100,
300}, and one factor varied at a time around it: the beam (200, on the
centred inputs up to p = 100, and at p = 300 for two laws), the noise (1%),
the rows (2,000, at 1% noise; both up to p = 100, and at p = 300 for two
laws), and p = 1,000 for the pair law. The full crossing of the six factors
would take some fifteen hours here.

Per fit it records:

- **exact**: the printed equation is the law, by the Feynman panel's
  criterion (``benchmarks/feynman_panel.py``: the kept terms match the law's
  terms one to one up to a constant factor, or one kept term is proportional
  to the whole law; checked with sympy);
- **held-out R^2** and the fit's wall time;
- **where a miss happened**. The search is followed through a subclass of
  ``BeamSearch`` that records, round by round, whether the intermediates were
  proposed, how they ranked and whether the beam kept them; the package is
  not modified. A miss is put down to the first stage that lost the law:
  ``unproposed`` (the product the law needs at depth 1 was never proposed),
  ``pruned`` (proposed, not kept in the depth-1 beam), ``search`` (kept, yet
  the law is not among the search's output), ``screen`` (in the output, not
  passed by the screen), ``printed`` (passed, but the printed equation is not
  exact: riders, or a missing term);
- the printed terms built only from distractors, the false discoveries in the
  marginal sense of ``benchmarks/parsimony_holdout/``.

The instrumented search is checked against the plain one: the first
``--check-seeds`` replicates of every cell are refitted with the package's own
``BeamSearch`` and must match to the last bit (features, coefficients,
intercept, test predictions), or the run stops.

Run from the repository root, with this checkout's ``src`` importable:

    python benchmarks/search_stress/stress.py --workers 3        # resumable
    python benchmarks/search_stress/stress.py --tabulate         # tables from the records
    python benchmarks/search_stress/stress.py --seeds 1 --out /tmp/pilot
    python benchmarks/search_stress/stress.py --setting magnitude_slots=10 \
        --kwargs '{"magnitude_slots": 10}' --beams 50 --out <dir>   # an option on the grid
    python benchmarks/search_stress/stress.py --out <dir> --compare <default dir>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

N_TEST = 2000
# A term counts as the law's when it uses exactly the law's columns and
# correlates at least this closely with it on the test rows.
EQUIVALENT = 0.9999
CEILING = 200_000  # max_candidates_per_depth at its default

LAWS = ("pair", "triple", "pair_plus")
LAW_TEXT = {"pair": "x0*x1", "triple": "x0*x1*x2", "pair_plus": "x0*x1 + x2"}
DISTS = ("unif", "gauss", "shifted")
DIST_TEXT = {"unif": "U(-1, 1)", "gauss": "N(0, 1)", "shifted": "U(0.5, 2.5)"}
# First two raw moments of one input column.
MOMENTS = {"unif": (0.0, 1 / 3), "gauss": (0.0, 1.0), "shifted": (1.5, 1 / 3 + 2.25)}
NOISES = (0.1, 0.01)
BEAMS = (50, 200)
WIDTHS = (5, 20, 100, 300, 1000)
SIZES = (500, 2000)
BASE = {"n": 500, "noise": 0.1, "beam": 50}

# Blocks of the grid, each the crossing of its factor levels.
GRID = [
    # The base: every law, input distribution and width.
    {"law": LAWS, "dist": DISTS, "p": (5, 20, 100, 300), "n": (500,), "noise": (0.1,), "beam": (50,)},
    # The wide beam, on centred inputs; at p = 300 for the two laws the
    # depth-1 beam loses there (the pair law's loss is in the proposals).
    {"law": LAWS, "dist": ("unif", "gauss"), "p": (5, 20, 100), "n": (500,), "noise": (0.1,), "beam": (200,)},
    {"law": ("triple", "pair_plus"), "dist": ("unif",), "p": (300,), "n": (500,), "noise": (0.1,), "beam": (200,)},
    # Noise, and then rows at the lower noise.
    {"law": LAWS, "dist": DISTS, "p": (5, 20, 100), "n": (500, 2000), "noise": (0.01,), "beam": (50,)},
    {"law": ("pair", "pair_plus"), "dist": ("unif",), "p": (300,), "n": (2000,), "noise": (0.01,), "beam": (50,)},
    # Far past the ceiling.
    {"law": ("pair",), "dist": ("unif", "shifted"), "p": (1000,), "n": (500,), "noise": (0.1,), "beam": (50,)},
]

FIELDS = [
    "cell", "setting", "law", "dist", "p", "n", "noise", "beam", "seed",
    "exact", "stage", "r2_test", "fit_seconds", "search_seconds", "ceiling",
    "input_positions", "d1_candidates", "d1_proposed", "d1_best_rank", "d1_in_beam",
    "law_proposed", "law_in_output", "law_screened", "printed", "screened",
    "marginal_false", "conditional_false", "formulas",
]


def cell_name(law: str, dist: str, p: int, n: int, noise: float, beam: int) -> str:
    return f"{law}_{dist}_p{p}_n{n}_e{round(noise * 100)}_b{beam}"


def grid_cells() -> list[tuple]:
    """The cells of ``GRID``, each once: (law, inputs, p, n, noise, beam)."""
    cells = []
    for block in GRID:
        for law in block["law"]:
            for dist in block["dist"]:
                for p in block["p"]:
                    for n in block["n"]:
                        for noise in block["noise"]:
                            for beam in block["beam"]:
                                cell = (law, dist, p, n, noise, beam)
                                if cell not in cells:
                                    cells.append(cell)
    return cells


def law_sd(law: str, dist: str) -> float:
    """Standard deviation of the noiseless law, from the input moments."""
    mean, second = MOMENTS[dist]
    if law == "pair":
        variance = second**2 - mean**4
    elif law == "triple":
        variance = second**3 - mean**6
    else:
        variance = (second**2 - mean**4) + (second - mean**2)
    return math.sqrt(variance)


def signal(law: str, X: np.ndarray) -> np.ndarray:
    if law == "pair":
        return X[:, 0] * X[:, 1]
    if law == "triple":
        return X[:, 0] * X[:, 1] * X[:, 2]
    return X[:, 0] * X[:, 1] + X[:, 2]


def make_data(law: str, dist: str, p: int, n: int, noise: float, seed: int):
    """Training rows, noisy target, test rows and the noiseless test law.

    The draws depend on the inputs, the width, the rows and the seed only, so
    the three laws, both noise levels and both beams of a replicate share
    every input value and the same standard normal noise vector.
    """
    rng = np.random.default_rng([DISTS.index(dist), p, n, seed])
    total = n + N_TEST
    if dist == "unif":
        X = rng.uniform(-1.0, 1.0, (total, p))
    elif dist == "gauss":
        X = rng.standard_normal((total, p))
    else:
        X = rng.uniform(0.5, 2.5, (total, p))
    standard = rng.standard_normal(total)
    law_values = signal(law, X)
    y = law_values + noise * law_sd(law, dist) * standard
    return X[:n], y[:n], X[n:], law_values[n:]


# --------------------------------------------------------------------------- #
# The law's pieces, and following them through the search
# --------------------------------------------------------------------------- #


def law_nodes(law: str) -> dict:
    """The nodes the search must reach, by round, and the law's terms.

    ``needed`` are the depth-1 products of which the beam must keep one;
    ``whole`` are the spellings of the complete law the depth-2 round can
    propose (the search unifies operand order, not association); ``terms``
    are the printed terms the panel's criterion expects, as sympy
    expressions, with their input columns.
    """
    import sympy

    from beamfeat.expression import combine, leaf

    x = [leaf(f"x{j}") for j in range(3)]
    s = sympy.symbols("x0 x1 x2")
    if law == "pair":
        pair = combine("mul", x[0], x[1])
        return {"needed": [pair], "whole": [pair], "terms": [s[0] * s[1]],
                "columns": [{"x0", "x1"}], "hidden": {"x0", "x1"}}
    if law == "triple":
        pairs = {(i, j): combine("mul", x[i], x[j]) for i, j in ((0, 1), (0, 2), (1, 2))}
        whole = [combine("mul", pairs[0, 1], x[2]), combine("mul", pairs[0, 2], x[1]),
                 combine("mul", pairs[1, 2], x[0])]
        return {"needed": list(pairs.values()), "whole": whole, "terms": [s[0] * s[1] * s[2]],
                "columns": [{"x0", "x1", "x2"}], "hidden": {"x0", "x1", "x2"}}
    pair = combine("mul", x[0], x[1])
    return {"needed": [pair], "whole": [combine("add", pair, x[2])], "terms": [s[0] * s[1], s[2]],
            "columns": [{"x0", "x1"}, {"x2"}], "hidden": {"x0", "x1"}}


def traced_class():
    """A ``BeamSearch`` that records how the law's pieces fared in each round.

    It calls the package's own methods and only reads their arguments and
    results, so a fit through it is the fit through ``BeamSearch``; the run
    checks that on the first replicates of every cell.
    """
    from beamfeat.search import BeamSearch, _rank

    class TracedSearch(BeamSearch):
        follow: dict = {}

        def run(self, *args, **kwargs):
            # The estimator does not keep its search; the class keeps the
            # last one, which is this fit's, for the record.
            type(self).last = self
            self.rounds: list[dict] = []
            self.first_order: list[str] | None = None
            self._seen_ref = None
            return super().run(*args, **kwargs)

        def _propose(self, beam, pool, seen):
            self._seen_ref = seen
            if not self.rounds and self.first_order is None and beam and beam[0].is_leaf:
                # The first call of round 1 sees the inputs in the order they
                # are proposed: by score when the ceiling can bind.
                self.first_order = [str(node.column) for node in beam]
            return super()._propose(beam, pool, seen)

        def _prune_evaluated(self, nodes, scores, limit, evaluator, shield=False):
            kept = super()._prune_evaluated(nodes, scores, limit, evaluator, shield)
            self._record(nodes, scores, kept[0])
            return kept

        def _reserve(self, nodes, scores, magnitude, beam, beam_values, evaluator):
            # With magnitude_slots, the slots it adds to the round's beam.
            reserved = super()._reserve(nodes, scores, magnitude, beam, beam_values, evaluator)
            record = self.rounds[-1]
            tracked = self.follow.get(len(self.rounds), [])
            added = set(reserved[0])
            record["in_beam"] = [kept or node in added for kept, node in zip(record["in_beam"], tracked)]
            return reserved

        def _record(self, nodes, scores, beam) -> None:
            round_index = len(self.rounds) + 1
            tracked = self.follow.get(round_index, [])
            wanted = set(tracked)
            index = {node: i for i, node in enumerate(nodes) if node in wanted}
            position = np.empty(len(nodes), dtype=np.int64)
            position[_rank(nodes, scores)] = np.arange(len(nodes))
            in_beam = set(beam)
            self.rounds.append({
                "candidates": len(nodes),
                "proposed": [node in self._seen_ref for node in tracked],
                "rank": [int(position[index[node]]) + 1 if node in index else None for node in tracked],
                "in_beam": [node in in_beam for node in tracked],
            })

    return TracedSearch


class Instrumented:
    """Installs the traced search where the estimators module looks it up."""

    def __init__(self) -> None:
        import beamfeat.estimators as estimators

        self.module = estimators
        self.original = estimators.BeamSearch
        self.traced = traced_class()

    def __enter__(self):
        self.module.BeamSearch = self.traced
        return self.traced

    def __exit__(self, *exc) -> None:
        self.module.BeamSearch = self.original


# --------------------------------------------------------------------------- #
# One fit
# --------------------------------------------------------------------------- #


def fit(X, y, beam: int, kwargs: dict):
    """The estimator as a user calls it: defaults, but for the beam and ``kwargs``."""
    from beamfeat import BeamFeatRegressor

    options = dict(kwargs) if beam == 50 else {"beam_width": beam, **kwargs}
    return BeamFeatRegressor(**options).fit(X, y)


def fingerprint(model, X_test: np.ndarray) -> tuple:
    """Everything a fit returns that the records depend on, bit for bit."""
    return (
        tuple(model.feature_formulas_),
        np.asarray(model.coef_).tobytes(),
        np.asarray(model.intercept_).tobytes(),
        model.predict(X_test).tobytes(),
    )


def _equivalent(node, columns: set, target: np.ndarray, evaluator) -> bool:
    if set(node.columns()) != columns:
        return False
    values = evaluator.transform_values(node)
    if values is None or not np.all(np.isfinite(values)) or np.std(values) == 0:
        return False
    return abs(float(np.corrcoef(values, target)[0, 1])) >= EQUIVALENT


def measure(model, law: str, pieces: dict, search, X_test: np.ndarray, law_test: np.ndarray) -> dict:
    """The record of one fit: what the search and the screen did with the law."""
    from feynman_panel import _exact_form

    from beamfeat.expression import Evaluator

    evaluator = Evaluator({f"x{j}": X_test[:, j] for j in range(X_test.shape[1])})
    # The printed terms the law needs, as (columns, values on the test rows).
    term_values = [X_test[:, 0] * X_test[:, 1] * (X_test[:, 2] if law == "triple" else 1.0)]
    terms = [(pieces["columns"][0], term_values[0])]
    if law == "pair_plus":
        terms.append(({"x2"}, X_test[:, 2]))

    def has_law(nodes) -> bool:
        """Whether ``nodes`` hold every term of the law, or the whole law as one."""
        if all(any(_equivalent(node, cols, values, evaluator) for node in nodes) for cols, values in terms):
            return True
        return law == "pair_plus" and any(
            _equivalent(node, {"x0", "x1", "x2"}, law_test, evaluator) for node in nodes
        )

    output = list(model.search_result_.nodes)
    by_name = {node.name: node for node in output}
    report = model.selection_report_ or []
    screened_nodes = [by_name[row["formula"]] for row in report if row["screened"] and row["formula"] in by_name]
    rounds = search.rounds
    first = rounds[0] if rounds else None
    second = rounds[1] if len(rounds) > 1 else None

    d1_proposed = sum(first["proposed"]) if first else 0
    ranks = [rank for rank in (first["rank"] if first else []) if rank is not None]
    d1_in_beam = sum(first["in_beam"]) if first else 0
    if law == "triple":
        law_proposed = int(bool(second) and any(second["proposed"]))
    else:
        law_proposed = int(d1_proposed > 0)
    in_output = has_law(output)
    passed = has_law(screened_nodes)
    exact = bool(_exact_form(model, pieces["terms"]))

    if exact:
        stage = "exact"
    elif d1_proposed == 0:
        stage = "unproposed"
    elif d1_in_beam == 0:
        stage = "pruned"
    elif not in_output:
        stage = "search"
    elif not passed:
        stage = "screen"
    else:
        stage = "printed"

    truth = {"x0", "x1"} | ({"x2"} if law != "pair" else set())
    used = [set(node.columns()) for node in model.features_]
    order = search.first_order
    ceiling = int(first is not None and model.search_result_.trace[1].n_proposed >= CEILING)
    try:
        r2 = float(model.score(X_test, law_test))
    except ValueError:  # a printed term undefined on some test row
        r2 = float("nan")
    return {
        "exact": int(exact),
        "stage": stage,
        "r2_test": f"{r2:.6g}",
        "search_seconds": f"{model.search_result_.elapsed:.3f}",
        "ceiling": ceiling,
        # Where the law's columns stood in the order round 1 proposed the
        # inputs in (1 = first); recorded only when that order is by score.
        "input_positions": (
            " ".join(str(order.index(name) + 1) for name in sorted(pieces["hidden"]))
            if ceiling and order else ""
        ),
        "d1_candidates": first["candidates"] if first else 0,
        "d1_proposed": d1_proposed,
        "d1_best_rank": min(ranks) if ranks else "",
        "d1_in_beam": d1_in_beam,
        "law_proposed": law_proposed,
        "law_in_output": int(in_output),
        "law_screened": int(passed),
        "printed": len(used),
        "screened": 0 if model.selection_result_ is None else int(model.selection_result_.n_selected),
        "marginal_false": sum(1 for cols in used if not cols & truth),
        "conditional_false": sum(1 for cols in used if not cols <= truth),
        "formulas": " | ".join(model.feature_formulas_[:4]) + (" | ..." if len(used) > 4 else ""),
    }


# --------------------------------------------------------------------------- #
# Running cells
# --------------------------------------------------------------------------- #


def wait_for_memory(minimum_mb: int) -> None:
    """Block until the machine reports at least ``minimum_mb`` available."""
    while True:
        with open("/proc/meminfo") as handle:
            info = dict(line.split(":", 1) for line in handle)
        available = int(info["MemAvailable"].split()[0]) // 1024
        if available > minimum_mb:
            return
        print(f"  waiting: {available} MB available, need more than {minimum_mb}", flush=True)
        time.sleep(60)


def run_cell(task: tuple) -> tuple[str, int, float]:
    """Fit each replicate of one cell; return the cell, fits checked and seconds."""
    cell, seeds, check_seeds, parts, min_memory, setting, kwargs = task
    warnings.filterwarnings("ignore")
    logging.getLogger("beamfeat").setLevel(logging.ERROR)
    sys.path.insert(0, str(HERE.parent))  # feynman_panel, for the exact-form criterion

    law, dist, p, n, noise, beam = cell
    name = cell_name(*cell)
    path = Path(parts) / f"{name}.csv"
    done: dict[int, dict] = {}
    if path.exists():
        with path.open(newline="") as handle:
            done = {int(row["seed"]): row for row in csv.DictReader(handle)}
    with path.open("w", newline="") as handle:  # keep complete records only
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(done[seed] for seed in sorted(done))
    if all(seed in done for seed in range(seeds)):
        return name, 0, 0.0

    wait_for_memory(min_memory)
    pieces = law_nodes(law)
    instrumented = Instrumented()
    checked, started = 0, time.perf_counter()
    for seed in range(seeds):
        if seed in done:
            continue
        X, y, X_test, law_test = make_data(law, dist, p, n, noise, seed)
        with instrumented as traced:
            traced.follow = {1: pieces["needed"], 2: pieces["whole"]}
            clock = time.perf_counter()
            model = fit(X, y, beam, kwargs)
            seconds = time.perf_counter() - clock
            search = traced.last
        row = {"cell": name, "setting": setting, "law": law, "dist": dist, "p": p, "n": n, "noise": noise,
               "beam": beam, "seed": seed, "fit_seconds": f"{seconds:.3f}",
               **measure(model, law, pieces, search, X_test, law_test)}
        if seed < check_seeds:
            plain = fit(X, y, beam, kwargs)
            if fingerprint(plain, X_test) != fingerprint(model, X_test):
                raise RuntimeError(f"the traced search changed the fit: {name} seed {seed}")
            checked += 1
        with path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
    return name, checked, time.perf_counter() - started


def provenance() -> dict:
    import scipy
    import sklearn
    import sympy

    import beamfeat

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

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
        "sympy": sympy.__version__,
        "platform": platform.platform(),
    }


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

STAGES = ("exact", "unproposed", "pruned", "search", "screen", "printed")


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


def _mean_se(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    se = float(np.std(array, ddof=1) / np.sqrt(array.size)) if array.size > 1 else float("nan")
    return float(np.mean(array)), se


def summarise(rows: list[dict]) -> dict:
    """Per cell: means with Monte Carlo standard errors, and the misses by stage.

    Also the paired differences the grid was built for: each cell against the
    base cell of its law, inputs and width, on the replicates both hold (the
    two share every draw but the noise's scale).
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["cell"], []).append(row)
    cells = {}
    for cell, members in sorted(groups.items()):
        members.sort(key=lambda r: int(r["seed"]))
        r2 = [float(r["r2_test"]) for r in members]
        finite = [value for value in r2 if math.isfinite(value)]
        seconds = [float(r["fit_seconds"]) for r in members]
        first = members[0]
        cells[cell] = {
            "setting": first["setting"], "law": first["law"], "dist": first["dist"], "p": int(first["p"]),
            "n": int(first["n"]), "noise": float(first["noise"]), "beam": int(first["beam"]),
            "fits": len(members),
            "exact": _mean_se(int(r["exact"]) for r in members),
            "in_output": _mean_se(int(r["law_in_output"]) for r in members),
            "d1_proposed": _mean_se(int(r["d1_proposed"]) > 0 for r in members),
            "d1_in_beam": _mean_se(int(r["d1_in_beam"]) > 0 for r in members),
            "stages": {stage: sum(r["stage"] == stage for r in members) for stage in STAGES},
            "r2_test": _mean_se(finite),
            "r2_test_median": float(np.median(finite)) if finite else float("nan"),
            "r2_undefined": len(r2) - len(finite),
            "fit_seconds_median": float(np.median(seconds)),
            "fit_seconds_mean": float(np.mean(seconds)),
            "ceiling": float(np.mean([int(r["ceiling"]) for r in members])),
            "any_false": _mean_se(int(r["marginal_false"]) > 0 for r in members),
            "printed": float(np.mean([int(r["printed"]) for r in members])),
            "exact_by_seed": {int(r["seed"]): int(r["exact"]) for r in members},
        }
    for entry in cells.values():
        base = cells.get(cell_name(entry["law"], entry["dist"], entry["p"], **BASE))
        if base is None or base is entry:
            continue
        common = sorted(set(entry["exact_by_seed"]) & set(base["exact_by_seed"]))
        entry["exact_vs_base"] = _mean_se(
            entry["exact_by_seed"][seed] - base["exact_by_seed"][seed] for seed in common
        )
    for entry in cells.values():
        del entry["exact_by_seed"]
    return {"cells": cells}


def _pm(value, digits: int = 2) -> str:
    mean, se = value
    if not math.isfinite(mean):
        return "—"
    if not math.isfinite(se):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def _shares(entry: dict) -> str:
    """Misses by stage, as counts of the cell's fits; '·' for none."""
    return " | ".join(str(entry["stages"][s]) if entry["stages"][s] else "·" for s in STAGES[1:])


def _row(entry: dict, label: str) -> str:
    return (
        f"| {label} | {entry['fits']} | {_pm(entry['exact'])} | {_shares(entry)} | "
        f"{entry['r2_test_median']:.4f} | {_pm(entry['any_false'])} | {entry['fit_seconds_median']:.1f} |"
    )


HEADER = [
    "| {first} | fits | exact | unproposed | pruned | search | screen | printed "
    "| held-out R² (median) | distractor term printed | fit s (median) |",
    "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
]


def tables(summary: dict) -> str:
    """The report's tables, as Markdown."""
    cells = summary["cells"]

    def get(law, dist, p, n, noise, beam):
        return cells.get(cell_name(law, dist, p, n, noise, beam))

    lines = [
        "Misses are counted by the first stage that lost the law: `unproposed` (the depth-1",
        "product the law needs was never proposed), `pruned` (proposed, not kept in the",
        "depth-1 beam), `search` (kept, but the law is not in the search's output), `screen`",
        "(in the output, not passed), `printed` (passed, but the printed equation is not",
        "exact). Means carry their Monte Carlo standard errors.",
        "",
        "### The base: beam 50, 500 rows, 10% noise",
        "",
        HEADER[0].format(first="law, inputs, p"),
        HEADER[1],
    ]
    for law in LAWS:
        for dist in DISTS:
            for p in WIDTHS:
                entry = get(law, dist, p, **BASE)
                if entry is not None:
                    lines.append(_row(entry, f"{LAW_TEXT[law]}, {dist}, {p}"))
    lines += [
        "",
        "### Beam width: 200 against the base's 50 (500 rows, 10% noise)",
        "",
        "| law, inputs, p | exact, beam 50 | exact, beam 200 | paired difference "
        "| fit s, 50 | fit s, 200 | misses at 200 (unproposed, pruned, search, screen, printed) |",
        "| :-- | --: | --: | --: | --: | --: | :-- |",
    ]
    for law in LAWS:
        for dist in DISTS:
            for p in WIDTHS:
                wide = get(law, dist, p, 500, 0.1, 200)
                base = get(law, dist, p, **BASE)
                if wide is None or base is None:
                    continue
                lines.append(
                    f"| {LAW_TEXT[law]}, {dist}, {p} | {_pm(base['exact'])} | {_pm(wide['exact'])} | "
                    f"{_pm(wide.get('exact_vs_base', (float('nan'),) * 2))} | "
                    f"{base['fit_seconds_median']:.1f} | {wide['fit_seconds_median']:.1f} | "
                    + ", ".join(str(wide["stages"][s]) for s in STAGES[1:]) + " |"
                )
    lines += [
        "",
        "### Noise and rows (beam 50)",
        "",
        "| law, inputs, p | exact: 500 rows, 10% | 500 rows, 1% | 2,000 rows, 1% "
        "| held-out R² (median), same order | fit s (median), same order |",
        "| :-- | --: | --: | --: | :-- | :-- |",
    ]
    corners = [(500, 0.1), (500, 0.01), (2000, 0.01)]
    for law in LAWS:
        for dist in DISTS:
            for p in WIDTHS:
                entries = [get(law, dist, p, n, noise, 50) for n, noise in corners]
                if sum(e is not None for e in entries) < 2:
                    continue
                lines.append(
                    f"| {LAW_TEXT[law]}, {dist}, {p} | "
                    + " | ".join("—" if e is None else _pm(e["exact"]) for e in entries)
                    + " | " + ", ".join("—" if e is None else f"{e['r2_test_median']:.4f}" for e in entries)
                    + " | " + ", ".join("—" if e is None else f"{e['fit_seconds_median']:.1f}" for e in entries)
                    + " |"
                )
    lines += [
        "",
        "### Every cell",
        "",
        HEADER[0].format(first="cell"),
        HEADER[1],
    ]
    for law in LAWS:
        for dist in DISTS:
            for p in WIDTHS:
                for n in SIZES:
                    for noise in NOISES:
                        for beam in BEAMS:
                            entry = get(law, dist, p, n, noise, beam)
                            if entry is not None:
                                lines.append(_row(entry, cell_name(law, dist, p, n, noise, beam)))
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
        writer.writerows(sorted(rows, key=lambda r: (r["cell"], int(r["seed"]))))
        # mtime 0, so that the archive is a function of the records alone
        with gzip.GzipFile(out / "records.csv.gz", "wb", mtime=0) as handle:
            handle.write(buffer.getvalue().encode())
    print(f"{len(rows)} records -> {info_path}, {out / 'tables.md'}")


def compare(out: Path, baseline: Path) -> str:
    """The cells of ``out`` against the same cells of ``baseline``, replicate by replicate.

    Writes ``compare.md`` and ``compare.json`` to ``out``. Differences are of
    the paired replicates, with their Monte Carlo standard errors; time is the
    ratio of the median fit times.
    """
    def by_cell(rows):
        cells: dict[str, dict[int, dict]] = {}
        for row in rows:
            cells.setdefault(row["cell"], {})[int(row["seed"])] = row
        return cells

    new, old = by_cell(load(out)), by_cell(load(baseline))
    setting = next(iter(next(iter(new.values())).values()))["setting"] if new else ""
    lines = [
        f"`{setting}` against the default, on the replicates both hold.",
        "",
        "| cell | fits | exact, default | exact, option | paired difference "
        "| held-out R² (median), default → option | distractor term printed, default → option "
        "| fit s (median), default → option | time ratio |",
        "| :-- | --: | --: | --: | --: | :-- | :-- | :-- | --: |",
    ]
    summary = {}
    for law in LAWS:
        for dist in DISTS:
            for p in WIDTHS:
                for n in SIZES:
                    for noise in NOISES:
                        for beam in BEAMS:
                            name = cell_name(law, dist, p, n, noise, beam)
                            if name not in new or name not in old:
                                continue
                            seeds = sorted(set(new[name]) & set(old[name]))
                            a = [old[name][s] for s in seeds]
                            b = [new[name][s] for s in seeds]

                            def col(rows, key, cast=float):
                                return np.array([cast(r[key]) for r in rows])

                            exact_a, exact_b = col(a, "exact", int), col(b, "exact", int)
                            false_a = col(a, "marginal_false", int) > 0
                            false_b = col(b, "marginal_false", int) > 0
                            r2_a, r2_b = col(a, "r2_test"), col(b, "r2_test")
                            t_a, t_b = col(a, "fit_seconds"), col(b, "fit_seconds")
                            entry = {
                                "fits": len(seeds),
                                "exact_default": _mean_se(exact_a), "exact_option": _mean_se(exact_b),
                                "exact_difference": _mean_se(exact_b - exact_a),
                                "r2_median": [float(np.nanmedian(r2_a)), float(np.nanmedian(r2_b))],
                                "distractor_printed": [float(false_a.mean()), float(false_b.mean())],
                                "fit_seconds_median": [float(np.median(t_a)), float(np.median(t_b))],
                                "time_ratio": float(np.median(t_b) / np.median(t_a)),
                                "stages_option": {s: int(sum(r["stage"] == s for r in b)) for s in STAGES},
                            }
                            summary[name] = entry
                            lines.append(
                                f"| {name} | {len(seeds)} | {_pm(entry['exact_default'])} | "
                                f"{_pm(entry['exact_option'])} | {_pm(entry['exact_difference'])} | "
                                f"{entry['r2_median'][0]:.4f} → {entry['r2_median'][1]:.4f} | "
                                f"{entry['distractor_printed'][0]:.2f} → {entry['distractor_printed'][1]:.2f} | "
                                f"{entry['fit_seconds_median'][0]:.1f} → {entry['fit_seconds_median'][1]:.1f} | "
                                f"{entry['time_ratio']:.2f} |"
                            )
    text = "\n".join(lines)
    (out / "compare.md").write_text(text + "\n")
    (out / "compare.json").write_text(json.dumps(summary, indent=1) + "\n")
    return text


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def _cost(cell: tuple) -> float:
    """A rough relative cost, to start the slowest cells first."""
    _, _, p, n, _, beam = cell
    candidates = min(5 * p + 3 * p * (p - 1), CEILING) + min(beam * (p + beam) * 4, CEILING)
    return candidates * n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=50, help="replicates per cell (default 50)")
    parser.add_argument("--check-seeds", type=int, default=1,
                        help="replicates per cell refitted with the plain search (default 1)")
    parser.add_argument("--setting", default="default", help="name recorded for --kwargs (default 'default')")
    parser.add_argument("--kwargs", default="{}", help="further estimator options, as JSON (default none)")
    parser.add_argument("--laws", default=",".join(LAWS))
    parser.add_argument("--dists", default=",".join(DISTS))
    parser.add_argument("--widths", default=",".join(map(str, WIDTHS)))
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)))
    parser.add_argument("--noises", default=",".join(map(str, NOISES)))
    parser.add_argument("--beams", default=",".join(map(str, BEAMS)))
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument("--min-memory", type=int, default=2000,
                        help="MB the machine must have available before a cell starts (default 2000)")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--tabulate", action="store_true", help="only rebuild the tables from the records")
    parser.add_argument("--compare", type=Path, default=None,
                        help="compare the records in --out with those in this directory, cell by cell")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run although tracked files differ from the commit")
    args = parser.parse_args(argv)

    if args.compare is not None:
        print(compare(args.out, args.compare))
        return
    if args.tabulate:
        tabulate(args.out)
        return

    info = provenance()
    if info["tracked_changes"] and not args.allow_dirty:
        sys.exit("tracked files differ from the commit; commit first, or pass --allow-dirty")
    wanted = {
        "law": set(args.laws.split(",")), "dist": set(args.dists.split(",")),
        "p": {int(v) for v in args.widths.split(",")}, "n": {int(v) for v in args.sizes.split(",")},
        "noise": {float(v) for v in args.noises.split(",")}, "beam": {int(v) for v in args.beams.split(",")},
    }
    cells = [
        cell for cell in grid_cells()
        if all(value in wanted[key] for key, value in zip(("law", "dist", "p", "n", "noise", "beam"), cell))
    ]
    kwargs = json.loads(args.kwargs)
    cells.sort(key=_cost, reverse=True)
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    threads = {name: os.environ.get(name, "") for name in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    print(f"search stress: {len(cells)} cells, {args.seeds} seeds each, {args.workers} worker(s), "
          f"setting {args.setting} {kwargs}")
    print(json.dumps({**info, "threads": threads}), flush=True)
    clock = parts / "wall_seconds.txt"
    earlier = float(clock.read_text()) if clock.exists() else 0.0
    started = time.time()
    tasks = [(cell, args.seeds, args.check_seeds, str(parts), args.min_memory, args.setting, kwargs)
             for cell in cells]
    if args.workers == 1:
        results = map(run_cell, tasks)
    else:
        import multiprocessing

        pool_ = multiprocessing.get_context("fork").Pool(args.workers, maxtasksperchild=1)
        results = pool_.imap_unordered(run_cell, tasks)
    for name, checked, seconds in results:
        clock.write_text(f"{earlier + time.time() - started:.0f}")
        print(f"{name}: {seconds:.0f} s, {checked} fits matched the plain search", flush=True)
    if args.workers != 1:
        pool_.close()
        pool_.join()
    rows = load(args.out)
    run_info = {
        **info,
        "threads": threads,
        "setting": args.setting,
        "kwargs": kwargs,
        "seeds": args.seeds,
        "check_seeds": args.check_seeds,
        "fits_checked_against_plain_search": sum(
            1 for row in rows if int(row["seed"]) < args.check_seeds
        ),
        "grid": GRID,
        "cells": [cell_name(*cell) for cell in cells],
        "n_test": N_TEST,
        "equivalence": EQUIVALENT,
        "workers": args.workers,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "wall_seconds": round(earlier + time.time() - started),
    }
    tabulate(args.out, run_info)


if __name__ == "__main__":
    main()
