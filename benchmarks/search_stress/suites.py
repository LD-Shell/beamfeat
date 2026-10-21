"""A search option against the default on the main suite, the high-dimensional study and the panel.

The suites run unchanged, through ``with_options.py`` for the option:

    cd benchmarks && python search_stress/with_options.py run_benchmarks.py \\
        --suite all --methods ridge beamfeat --output <default>/main.csv
    cd benchmarks/additional_experiments && python ../search_stress/with_options.py \\
        bench.py highdim beamfeat 5 <default>/highdim.json

and the same with ``BEAMFEAT_OPTIONS='{"magnitude_slots": 10}'`` into
``<option>``. This script reads both directories and writes ``suites.md`` and
``suites.json`` to ``--out``. A classification dataset the option refuses is
left out of both settings' means and named. Times are sums of the recorded
fit times, comparable only when the two settings ran side by side.

    python benchmarks/search_stress/suites.py <default> <option> --out <dir> [--panel panel.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

HIGHDIM = ["communities_p100", "superconduct_p81", "tecator_p100", "eyedata_p200", "riboflavin_p4088",
           "geomusic_p117", "blogfeedback_p280", "ct_slices_p384", "ujiindoorloc_p520"]


def main_rows(path: Path, method: str = "beamfeat") -> dict:
    with path.open(newline="") as handle:
        return {r["dataset"]: r for r in csv.DictReader(handle) if r["method"] == method}


def highdim_rows(directory: Path) -> dict:
    rows: dict = {}
    for path in sorted(directory.glob("highdim*.json")):
        for r in json.loads(path.read_text()):
            if r["method"] == "beamfeat":
                rows.setdefault(r["dataset"], {})[r["split"]] = r
    return rows


def peak_mb(directory: Path) -> float | None:
    for path in directory.glob("*.time"):
        for line in path.read_text().splitlines():
            if "Maximum resident" in line:
                return int(line.split()[-1]) / 1024
    return None


def compare(default: Path, option: Path, panel: Path | None) -> tuple[str, dict]:
    lines: list[str] = []
    out: dict = {}
    a, b = main_rows(default / "main.csv"), main_rows(option / "main.csv")
    ridge_a, ridge_b = main_rows(default / "main.csv", "ridge"), main_rows(option / "main.csv", "ridge")
    assert {k: r["r2"] for k, r in ridge_a.items()} == {k: r["r2"] for k, r in ridge_b.items()}, "the data differ"
    refused = sorted(k for k in b if b[k]["error"])
    keys = [k for k in a if k not in refused]
    changed = {k: [float(a[k]["r2"]), float(b[k]["r2"])] for k in keys
               if a[k]["r2"] != b[k]["r2"] or a[k]["n_features"] != b[k]["n_features"]}
    recovered = [f"{sum(rows[k]['recovered'] == 'True' for k in keys)}/"
                 f"{sum(rows[k]['recovered'] in ('True', 'False') for k in keys)}" for rows in (a, b)]
    out["main"] = {"datasets": len(keys), "refused": refused, "changed": changed,
                   "mean": [st.mean(float(rows[k]["r2"]) for k in keys) for rows in (a, b)],
                   "recovered": recovered}
    lines += [
        f"## Main suite ({len(keys)} datasets" + (f"; left out, refused by the option: {', '.join(refused)}" if refused
                                                  else "") + ")",
        "",
        "| | default | option |",
        "| :-- | --: | --: |",
        f"| mean held-out score | {out['main']['mean'][0]:.4f} | {out['main']['mean'][1]:.4f} |",
        f"| identical fits | | {len(keys) - len(changed)}/{len(keys)} |",
        f"| formula recovered | {recovered[0]} | {recovered[1]} |",
        "",
        "Fits that differ: " + ", ".join(f"{k} {v[0]:.4f} → {v[1]:.4f}" for k, v in changed.items()) + ".",
        "",
    ]
    ha, hb = highdim_rows(default), highdim_rows(option)
    lines += [
        "## High-dimensional study (mean held-out R² over 5 splits)",
        "",
        "| dataset | default | option | splits up / down | fit time, default → option |",
        "| :-- | --: | --: | :-- | :-- |",
    ]
    means, times, per = ([], []), [0.0, 0.0], {}
    for ds in [d for d in HIGHDIM if d in ha and d in hb]:
        x, y = ha[ds], hb[ds]
        mx = st.mean(r["r2"] for r in x.values())
        my = st.mean(r["r2"] for r in y.values() if r["r2"] is not None)
        failed = sum(r["r2"] is None for r in y.values())
        up = sum(y[s]["r2"] is not None and y[s]["r2"] > x[s]["r2"] + 1e-12 for s in x)
        down = sum(y[s]["r2"] is not None and y[s]["r2"] < x[s]["r2"] - 1e-12 for s in x)
        tx, ty = sum(r["seconds"] for r in x.values()), sum(r["seconds"] for r in y.values())
        means[0].append(mx), means[1].append(my)
        times[0] += tx
        times[1] += ty
        per[ds] = {"r2": [mx, my], "up": up, "down": down, "failed": failed, "seconds": [tx, ty],
                   "splits": {str(s): [x[s]["r2"], y[s]["r2"]] for s in x}}
        lines.append(f"| {ds} | {mx:.3f} | {my:.3f}{f' ({failed} failed)' if failed else ''} | {up} / {down} | "
                     f"{tx:.0f} s → {ty:.0f} s |")
    lines.append(f"| mean of datasets | {st.mean(means[0]):.3f} | {st.mean(means[1]):.3f} | | "
                 f"{times[0]:.0f} s → {times[1]:.0f} s |")
    out["highdim"] = per
    peaks = [peak_mb(default), peak_mb(option)]
    if all(peaks):
        lines += ["", f"Peak memory of the runs timed with `/usr/bin/time`: {peaks[0]:.0f} MB → {peaks[1]:.0f} MB."]
        out["peak_mb"] = peaks
    if panel is not None:
        runs = json.loads(panel.read_text())
        lines += ["", "## Feynman panel (12 laws)", ""]
        for label, entries in runs.items():
            for entry in entries:
                lines.append(f"- {entry['label']}: {entry['solved']} solved, {entry['exact_form']} exact, "
                             f"{entry['mean_seconds']:.2f} s per law")
        out["panel"] = {label: [[e["solved"], e["exact_form"], e["mean_seconds"]] for e in entries]
                        for label, entries in runs.items()}
    return "\n".join(lines), out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("default", type=Path)
    parser.add_argument("option", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=None)
    args = parser.parse_args()
    text, summary = compare(args.default, args.option, args.panel)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "suites.md").write_text(text + "\n")
    (args.out / "suites.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(text)


if __name__ == "__main__":
    main()
