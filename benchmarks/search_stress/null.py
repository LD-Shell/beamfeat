"""False discoveries with and without a search option, on null and planted targets.

An option that changes what the search keeps must not raise the screen's
false discoveries. This script fits the same replicates under each setting,
so the comparison is paired, and records what the screen selected
(``parsimony=None``: the screened set, the candidates whose nulls the screen
rejected, which is what the guarantee covers).

Designs (300 rows; 8 columns unless stated):

    null_shifted      y ~ N(0, 1) independent of X; X ~ U(0.5, 3)
    null_centred      the same with X ~ U(-1, 1)
    null_gauss        the same with X ~ N(0, 1)
    null_wide         X ~ U(-1, 1) with 200 columns, more than the 150 rows
                      the search keeps after the selection split
    planted_shifted   y = x0*x1/x2 + noise of half the law's sd; X ~ U(0.5, 3)
    planted_centred   y = x0*x1*x2 + noise of half the law's sd; X ~ U(-1, 1),
                      20 columns

On a null design every screened feature is false. On a planted one a
screened feature is false when it uses none of the law's columns: such a
feature is independent of the target, so it is a null of the permutation
test. ``null_shifted`` and ``planted_shifted`` repeat the null study that
``fitted_ops`` and ``max_inputs`` were first measured on.

Run from the repository root, with this checkout's ``src`` importable:

    python benchmarks/search_stress/null.py --out <dir>                  # resumable
    python benchmarks/search_stress/null.py --out <dir> --designs null_wide
    python benchmarks/search_stress/null.py --out <dir> --tabulate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stress import _mean_se, provenance  # noqa: E402

DESIGNS = ("null_shifted", "null_centred", "null_gauss", "null_wide", "planted_shifted", "planted_centred")
SETTINGS = {"default": {}, "magnitude_slots=10": {"magnitude_slots": 10}}
FIELDS = ["design", "setting", "seed", "screened", "false", "seconds", "formulas"]


def make_data(design: str, seed: int):
    rng = np.random.default_rng([DESIGNS.index(design), seed])
    n = 300
    if design in ("null_shifted", "planted_shifted"):
        X = rng.uniform(0.5, 3.0, (n, 8))
    elif design == "null_gauss":
        X = rng.standard_normal((n, 8))
    else:
        X = rng.uniform(-1.0, 1.0, (n, {"null_wide": 200, "planted_centred": 20}.get(design, 8)))
    noise = rng.standard_normal(n)
    if design == "planted_shifted":
        law = X[:, 0] * X[:, 1] / X[:, 2]
        return X, law + 0.5 * law.std() * noise
    if design == "planted_centred":
        law = X[:, 0] * X[:, 1] * X[:, 2]
        return X, law + 0.5 * law.std() * noise
    return X, noise


def run(out: Path, seeds: int, settings: dict, designs: list[str]) -> None:
    """Fit each design's replicates under every setting, one record file per design."""
    from beamfeat import BeamFeatRegressor

    warnings.filterwarnings("ignore")
    for design in designs:
        path = out / f"{design}.csv"
        done = set()
        if path.exists():
            with path.open(newline="") as handle:
                done = {(r["setting"], int(r["seed"])) for r in csv.DictReader(handle)}
        else:
            with path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
        truth = {"x0", "x1", "x2"} if design.startswith("planted") else set()
        started = time.perf_counter()
        for seed in range(seeds):
            X, y = make_data(design, seed)
            for setting, kwargs in settings.items():
                if (setting, seed) in done:
                    continue
                clock = time.perf_counter()
                model = BeamFeatRegressor(random_state=seed, parsimony=None, **kwargs).fit(X, y)
                seconds = time.perf_counter() - clock
                used = [set(node.columns()) for node in model.features_]
                row = {
                    "design": design, "setting": setting, "seed": seed,
                    "screened": len(used), "false": sum(1 for cols in used if not cols & truth),
                    "seconds": f"{seconds:.3f}", "formulas": " | ".join(model.feature_formulas_[:3]),
                }
                with path.open("a", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
        print(f"{design}: {time.perf_counter() - started:.0f} s", flush=True)


def tabulate(out: Path) -> str:
    rows = []
    for design in DESIGNS:
        if (out / f"{design}.csv").exists():
            with (out / f"{design}.csv").open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
    groups: dict[tuple, dict[int, dict]] = {}
    for row in rows:
        groups.setdefault((row["design"], row["setting"]), {})[int(row["seed"])] = row
    settings = sorted({setting for _, setting in groups}, key=lambda s: (s != "default", s))
    lines = [
        "| design | setting | fits | any false, share of fits | FDR (mean FDP) | screened | fit s (median) "
        "| only this setting false / only default false |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: |",
    ]
    summary = {}
    for design in DESIGNS:
        base = groups.get((design, "default"), {})
        for setting in settings:
            members = groups.get((design, setting))
            if not members:
                continue
            any_false = [int(int(r["false"]) > 0) for r in members.values()]
            fdp = [int(r["false"]) / max(int(r["screened"]), 1) for r in members.values()]
            entry = {
                "fits": len(members),
                "any_false": _mean_se(any_false),
                "fdr": _mean_se(fdp),
                "screened": float(np.mean([int(r["screened"]) for r in members.values()])),
                "seconds_median": float(np.median([float(r["seconds"]) for r in members.values()])),
            }
            discordant = ""
            if setting != "default" and base:
                common = sorted(set(members) & set(base))
                only_here = sum(int(members[s]["false"]) > 0 and int(base[s]["false"]) == 0 for s in common)
                only_base = sum(int(base[s]["false"]) > 0 and int(members[s]["false"]) == 0 for s in common)
                entry["discordant"] = [only_here, only_base]
                discordant = f"{only_here} / {only_base}"
            summary[f"{design}|{setting}"] = entry

            def fmt(value):
                mean, se = value
                return f"{mean:.3f} ± {se:.3f}" if math.isfinite(se) else f"{mean:.3f}"

            lines.append(
                f"| {design} | {setting} | {entry['fits']} | {fmt(entry['any_false'])} | {fmt(entry['fdr'])} | "
                f"{entry['screened']:.2f} | {entry['seconds_median']:.2f} | {discordant} |"
            )
    table = "\n".join(lines)
    (out / "null.md").write_text(table + "\n")
    (out / "null.json").write_text(json.dumps(summary, indent=1) + "\n")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=200, help="replicates per design (default 200)")
    parser.add_argument("--settings", default=json.dumps(SETTINGS),
                        help="JSON object: setting name -> estimator options")
    parser.add_argument("--designs", default=",".join(DESIGNS), help="comma-separated designs to run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tabulate", action="store_true", help="only rebuild the table from the records")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.tabulate:
        settings = json.loads(args.settings)
        designs = [design for design in args.designs.split(",") if design]
        info = {**provenance(), "seeds": args.seeds, "settings": settings}
        (args.out / f"provenance_{'_'.join(designs)}.json").write_text(json.dumps(info, indent=1) + "\n")
        print(json.dumps(info), flush=True)
        run(args.out, args.seeds, settings, designs)
    print(tabulate(args.out))


if __name__ == "__main__":
    main()
