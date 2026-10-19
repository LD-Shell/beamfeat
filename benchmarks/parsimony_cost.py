"""Measure what the parsimony default costs, by running both settings.

``parsimony="forward"`` is the default: the fitted equation is a compact
subset of the FDR-screened set, which buys readability and gives up the
guarantee over the printed terms. That exchange is asserted in the
documentation and in the paper. This script measures it.

Every study whose numbers depend on the setting -- which ones, and why the
rest do not, is `PARSIMONY_COST.md` -- is run twice, once at each setting, and
for every fit three quantities are recorded:

    score            R² of the fitted model: held out, except in the
                     calibration study, which fits and scores on all rows
    n_terms          number of terms in the printed equation
    fdr_controlled   the ``fdr_controlled_`` flag

The point is the pairing. Accuracy lost, terms saved, and whether the flag
ever disagrees are what say whether compactness is cheap or expensive, and
they cannot be read off either setting alone.

Each study is driven through its own module rather than through a copy of its
protocol, so the two arms and the reported run share one definition of the
study. Output goes to ``benchmarks/parsimony_cost/``; nothing under
``independent/results_as_reported/`` or ``additional_experiments/results/`` is
read or written, so this measures alongside the reported results rather than
replacing them.

    python benchmarks/parsimony_cost.py --dry-run     # estimates only
    python benchmarks/parsimony_cost.py
    python benchmarks/parsimony_cost.py --only feynman,friedman
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT_ROOT = HERE / "parsimony_cost"
SETTINGS = (("parsimony_none", None), ("parsimony_forward", "forward"))

# Minutes per setting, measured on one core. Printed before anything runs so a
# caller can choose rather than discover the cost halfway through.
ESTIMATES = {
    "feynman": 0.5,
    "friedman": 0.2,
    "calibration": 1.7,
    "core": 0.4,
    "real": 10.0,
    "scalability": 30.0,
    "depth_ladder": 70.0,
    "split_stability": 45.0,
}
DEFAULT_STUDIES = ("feynman", "friedman", "calibration", "core", "real")
HEAVY_STUDIES = ("scalability", "depth_ladder", "split_stability")


def _fits(records: list[dict]) -> dict:
    """Summarise the three paired quantities over a study's fits."""
    scored = [r["score"] for r in records if r.get("score") is not None]
    terms = [r["n_terms"] for r in records if r.get("n_terms") is not None]
    flags = [r["fdr_controlled"] for r in records if r.get("fdr_controlled") is not None]
    return {
        "n_fits": len(records),
        "mean_score": float(np.mean(scored)) if scored else None,
        "median_score": float(np.median(scored)) if scored else None,
        "min_score": float(np.min(scored)) if scored else None,
        "mean_terms": float(np.mean(terms)) if terms else None,
        "median_terms": float(np.median(terms)) if terms else None,
        "max_terms": int(np.max(terms)) if terms else None,
        "fdr_controlled_true": int(sum(bool(f) for f in flags)),
        "fdr_controlled_total": len(flags),
    }


# --------------------------------------------------------------------------- #
# Studies driven in process
# --------------------------------------------------------------------------- #


def study_feynman(parsimony):
    """The twelve-law physics panel, both operator configurations.

    ``exact_form`` is the panel's own criterion, unmodified. It is a statement
    about the printed equation -- do the kept formulas match the expected
    terms as a multiset -- so it means what the paper says it means at the
    default and is False everywhere at ``parsimony=None``, where the printed
    equation is the whole screened set. ``contains_exact_form`` is recorded
    beside it as a separate diagnostic: whether *any* kept formula is
    proportional to the law. It is the same at both settings, which is how the
    record shows that the search finds the law either way. The two are
    different questions and PARSIMONY_COST.md says not to report the second as
    the first.
    """
    import feynman_panel as fp

    panels, records = [], []
    for unary, label in ((None, "default operators"),
                         (("log", "sqrt", "reciprocal", "square", "abs", "exp"),
                          "with exp enabled")):
        panel = fp.run_panel(unary_ops=unary, label=label, parsimony=parsimony)
        panels.append(panel)
        for row in panel["rows"]:
            records.append({
                "study": "feynman", "case": f"{label}/{row['equation']}",
                "score": row["r2"], "n_terms": row["n_terms"],
                "fdr_controlled": row["fdr_controlled"],
                "solved": row["solved"], "exact_form": row["exact_form"],
                "seconds": row["seconds"], "top_formula": row["top_formula"],
            })
    _add_contains_exact_form(records, parsimony)
    return {"panels": panels}, records


def _add_contains_exact_form(records, parsimony):
    """Refit each law once more to ask the weaker question the panel does not.

    Kept separate from the panel's own scoring so that ``exact_form`` in the
    record is exactly what ``feynman_panel`` computed, and this is visibly a
    second measurement rather than a redefinition of the first.
    """
    import feynman_panel as fp

    from beamfeat import BeamFeatRegressor

    lookup = {}
    for unary, label in ((None, "default operators"),
                         (("log", "sqrt", "reciprocal", "square", "abs", "exp"),
                          "with exp enabled")):
        kwargs = {"unary_ops": unary} if unary is not None else {}
        for name, generator, n_cols, expected, _note in fp.equations():
            if expected is None:
                lookup[f"{label}/{name}"] = None
                continue
            X, _X_test, y, _y_test = fp.make_data(name, generator, n_cols)
            model = BeamFeatRegressor(
                max_depth=3, beam_width=40, random_state=0, parsimony=parsimony, **kwargs
            ).fit(X, y)
            terms = model.to_sympy()
            target = expected[0] if len(expected) == 1 else sum(expected[1:], expected[0])
            # to_sympy mints its symbols with assumptions and sympy treats
            # same-named symbols with different assumptions as distinct, so a
            # ratio of the two never cancels. This is the rebinding
            # _exact_form does; without it the answer is False everywhere.
            model_symbols = {s.name: s for term in terms for s in term.free_symbols}
            target = target.subs(
                {s: model_symbols.get(s.name, s) for s in target.free_symbols}
            )
            lookup[f"{label}/{name}"] = any(
                fp._proportional(term, target) for term in terms
            )
    for record in records:
        record["contains_exact_form"] = lookup.get(record["case"])


def study_friedman(parsimony):
    """Friedman #1 decomposition. Only ``achieved`` depends on the setting."""
    import friedman_decomposition as fd

    summary = fd.main(parsimony=parsimony)
    records = [
        {"study": "friedman", "case": f"draw{k}", "score": draw["achieved"],
         "n_terms": draw["n_terms"], "fdr_controlled": draw["fdr_controlled"],
         "oracle": draw["oracle"], "ceiling": draw["ceiling"]}
        for k, draw in enumerate(summary["draws"])
    ]
    summary.pop("draws", None)
    return summary, records


def study_calibration(parsimony):
    """End-to-end FDR calibration: 200 signal replicates and 60 global-null."""
    import calibration_study as cs

    summary = cs.main(parsimony=parsimony)
    records = [
        {"study": "calibration", "case": f"{fit['arm']}{fit['trial']}",
         "score": fit["score"], "n_terms": fit["n_terms"],
         "fdr_controlled": fit["fdr_controlled"], "n_screened": fit["n_screened"]}
        for fit in summary["per_fit"]
    ]
    summary.pop("per_fit", None)
    return summary, records


def _run_benchmarks(parsimony, datasets, label):
    import run_benchmarks as rb

    outcomes = rb.evaluate(datasets, ["beamfeat"],
                           beamfeat_kwargs={"parsimony": parsimony})
    records = [
        {"study": label, "case": o.dataset,
         "score": None if not np.isfinite(o.r2) else float(o.r2),
         "n_terms": o.n_features, "fdr_controlled": o.fdr_ok,
         "complexity": o.complexity, "recovered": o.recovered,
         "false_feature_rate": o.false_feature_rate, "seconds": o.fit_seconds,
         "error": o.error}
        for o in outcomes
    ]
    return {"outcomes": [r["case"] for r in records]}, records


def study_core(parsimony):
    """The known-formula and stress suites of run_benchmarks.py."""
    import run_benchmarks as rb

    datasets = rb.synthetic_datasets() + rb.robustness_datasets()
    return _run_benchmarks(parsimony, datasets, "core")


def study_real(parsimony):
    """The real-data panel. Fetches on first run; fails loudly without network."""
    import run_benchmarks as rb

    return _run_benchmarks(parsimony, rb.real_datasets(), "real")


IN_PROCESS = {
    "feynman": study_feynman,
    "friedman": study_friedman,
    "calibration": study_calibration,
    "core": study_core,
    "real": study_real,
}

# Studies with their own CLI, their own checkpointing and their own runtimes.
# Driven as subprocesses so an interrupted one can be resumed the way its own
# README documents rather than through this script.
SUBPROCESS = {
    "scalability": ("additional_experiments/scalability.py", []),
    "depth_ladder": ("additional_experiments/depth_ladder.py", []),
    "split_stability": ("additional_experiments/split_stability.py", []),
}


def run_subprocess(name, parsimony, out_dir):
    relative, extra = SUBPROCESS[name]
    script = HERE / relative
    destination = out_dir / f"{name}.json"
    command = [sys.executable, script.name, "--out", str(destination), *extra]
    if parsimony is not None:
        command += ["--parsimony", parsimony]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=script.parent, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    (out_dir / f"{name}.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{relative} exited {completed.returncode}; see {out_dir / (name + '.log')}\n"
            + completed.stderr.strip()[-600:]
        )
    rows = json.loads(destination.read_text())
    return {"seconds": elapsed, "output": str(destination)}, _subprocess_records(name, rows)


def _subprocess_records(name, rows):
    """Lift the three paired quantities out of each script's own record shape."""
    records = []
    if name == "scalability":
        for row in rows:
            records.append({
                "study": name,
                "case": f"{row.get('target')}/{row.get('regime')}/p{row.get('p')}/s{row.get('seed')}",
                "score": None, "n_terms": row.get("n_selected"),
                "fdr_controlled": row.get("fdr"), "recovered": row.get("recovered"),
                "false_features": row.get("false_features"),
                "seconds": row.get("seconds"), "error": row.get("error"),
            })
    elif name == "depth_ladder":
        for row in rows:
            records.append({
                "study": name,
                "case": f"{row.get('problem')}/{row.get('scorer')}/beam{row.get('beam')}/s{row.get('seed')}",
                "score": None, "n_terms": row.get("n_selected"),
                "fdr_controlled": row.get("fdr"),
                "recovered_all": row.get("recovered_all"),
                "recovered_any": row.get("recovered_any"),
                "seconds": row.get("seconds"), "error": row.get("error"),
            })
    elif name == "split_stability":
        for dataset, row in rows.items():
            scores = row.get("r2_per_split") or []
            counts = row.get("n_selected") or []
            flags = row.get("fdr_controlled") or [None] * len(counts)
            for index, count in enumerate(counts):
                records.append({
                    "study": name, "case": f"{dataset}/split{index}",
                    "score": scores[index] if index < len(scores) else None,
                    "n_terms": count,
                    "fdr_controlled": flags[index] if index < len(flags) else None,
                    "jaccard_mean": row.get("jaccard_mean"),
                })
    return records


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", default="",
        help="comma list of studies to run; default is "
             + ",".join(DEFAULT_STUDIES),
    )
    parser.add_argument(
        "--heavy", action="store_true",
        help="also run " + ", ".join(HEAVY_STUDIES) + " (hours)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the estimates, run nothing")
    parser.add_argument("--out", type=pathlib.Path, default=OUT_ROOT)
    args = parser.parse_args()

    known = {**IN_PROCESS, **SUBPROCESS}
    if args.only:
        studies = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in studies if s not in known]
        if unknown:
            parser.error(f"unknown studies {unknown}; known: {sorted(known)}")
    else:
        studies = list(DEFAULT_STUDIES) + (list(HEAVY_STUDIES) if args.heavy else [])

    total = sum(ESTIMATES.get(s, 0.0) for s in studies) * len(SETTINGS)
    print("Paired measurement of what the parsimony default costs.")
    print("Determinations and justifications: benchmarks/PARSIMONY_COST.md\n")
    print(f"{'study':<18} {'per setting':>12} {'both settings':>14}")
    for name in studies:
        each = ESTIMATES.get(name, 0.0)
        print(f"{name:<18} {each:>10.1f} m {2 * each:>12.1f} m")
    print(f"{'TOTAL':<18} {'':>12} {total:>12.1f} m")
    if not args.heavy and not args.only:
        print("\nNot included (hours each): " + ", ".join(HEAVY_STUDIES)
              + ". Add --heavy to run them.")
    print(f"\nWriting to {args.out}/<setting>/  "
          "(archived results are neither read nor written)")
    if args.dry_run:
        return 0

    started = time.perf_counter()
    collected: dict[str, dict[str, list[dict]]] = {}
    failures = []
    for setting_name, parsimony in SETTINGS:
        out_dir = args.out / setting_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in studies:
            print(f"\n=== {name} @ parsimony={parsimony!r} "
                  f"({time.perf_counter() - started:.0f}s elapsed) ===", flush=True)
            try:
                if name in IN_PROCESS:
                    summary, records = IN_PROCESS[name](parsimony)
                else:
                    summary, records = run_subprocess(name, parsimony, out_dir)
            except Exception:
                detail = traceback.format_exc(limit=6)
                failures.append((name, setting_name, detail))
                print(f"FAILED {name} @ {setting_name}", flush=True)
                print(detail.strip().splitlines()[-1], flush=True)
                (out_dir / f"{name}.error.txt").write_text(detail)
                continue
            payload = {"study": name, "parsimony": parsimony,
                       "summary": summary, "fits": records,
                       "aggregate": _fits(records)}
            (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
            collected.setdefault(name, {})[setting_name] = records
            print(f"  {_fits(records)}", flush=True)

    # Rebuilt from every study present on disk, not only the ones this
    # invocation ran, so running the studies in several sessions leaves one
    # summary rather than the last session's fragment.
    paired = _pair(_collect_from_disk(args.out, collected))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paired_summary.json").write_text(json.dumps(paired, indent=2))
    _print_table(paired)

    if failures:
        print(f"\n{len(failures)} study/setting combination(s) failed:", file=sys.stderr)
        for name, setting_name, detail in failures:
            print(f"  {name} @ {setting_name}: "
                  f"{detail.strip().splitlines()[-1]}", file=sys.stderr)
        print("Tracebacks are beside the outputs as <study>.error.txt. A pair "
              "with one arm missing measures nothing, so this exits non-zero.",
              file=sys.stderr)
        return 1
    return 0


def _collect_from_disk(out_root: pathlib.Path, collected: dict) -> dict:
    """Merge this session's records with every pair already written."""
    merged: dict[str, dict[str, list[dict]]] = {}
    for setting_name, _ in SETTINGS:
        directory = out_root / setting_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict) and "fits" in payload:
                merged.setdefault(path.stem, {})[setting_name] = payload["fits"]
    for name, arms in collected.items():
        merged.setdefault(name, {}).update(arms)
    return merged


def _pair(collected: dict) -> dict:
    """Join the two arms per study and per fit, and difference them."""
    paired = {}
    for name, arms in collected.items():
        if set(arms) != {setting for setting, _ in SETTINGS}:
            paired[name] = {"complete": False, "arms": sorted(arms)}
            continue
        none_by_case = {r["case"]: r for r in arms["parsimony_none"]}
        forward_by_case = {r["case"]: r for r in arms["parsimony_forward"]}
        rows = []
        for case in none_by_case:
            if case not in forward_by_case:
                continue
            a, b = none_by_case[case], forward_by_case[case]
            rows.append({
                "case": case,
                "score_none": a.get("score"), "score_forward": b.get("score"),
                "score_delta": (
                    None if a.get("score") is None or b.get("score") is None
                    else b["score"] - a["score"]
                ),
                "terms_none": a.get("n_terms"), "terms_forward": b.get("n_terms"),
                "fdr_none": a.get("fdr_controlled"), "fdr_forward": b.get("fdr_controlled"),
            })
        deltas = [r["score_delta"] for r in rows if r["score_delta"] is not None]
        ratios = [
            r["terms_none"] / r["terms_forward"]
            for r in rows
            if r.get("terms_none") and r.get("terms_forward")
        ]
        paired[name] = {
            "complete": True,
            "n_cases": len(rows),
            "mean_score_delta_forward_minus_none": (
                float(np.mean(deltas)) if deltas else None
            ),
            "worst_score_delta": float(np.min(deltas)) if deltas else None,
            "best_score_delta": float(np.max(deltas)) if deltas else None,
            "mean_term_ratio_none_over_forward": float(np.mean(ratios)) if ratios else None,
            "fdr_flag_disagreements": sum(
                1 for r in rows if r["fdr_none"] != r["fdr_forward"]
            ),
            "cases": rows,
        }
    return paired


def _print_table(paired: dict) -> None:
    print("\n" + "=" * 78)
    print("WHAT COMPACTNESS COSTS  (default minus parsimony=None; negative = the")
    print("default is worse; the terms column is how much shorter it is)")
    print("=" * 78)
    print(f"{'study':<18} {'fits':>5} {'mean dscore':>12} {'worst':>9} "
          f"{'terms None/fwd':>15} {'flag diffs':>11}")
    for name, row in sorted(paired.items()):
        if not row.get("complete"):
            print(f"{name:<18}  incomplete: only {row.get('arms')}")
            continue
        mean_delta = row["mean_score_delta_forward_minus_none"]
        worst = row["worst_score_delta"]
        ratio = row["mean_term_ratio_none_over_forward"]
        print(
            f"{name:<18} {row['n_cases']:>5} "
            f"{'n/a' if mean_delta is None else f'{mean_delta:+.4f}':>12} "
            f"{'n/a' if worst is None else f'{worst:+.4f}':>9} "
            f"{'n/a' if ratio is None else f'{ratio:.1f}x':>15} "
            f"{row['fdr_flag_disagreements']:>11}"
        )
    print("\nPer-case rows are in paired_summary.json.")


if __name__ == "__main__":
    raise SystemExit(main())
