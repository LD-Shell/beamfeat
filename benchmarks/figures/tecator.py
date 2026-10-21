"""Draw the tecator figure on the README and the documentation home page.

Tecator is the fat content of 240 meat samples, measured by 100
near-infrared absorbance channels. The figure plots the archived run behind
Table 6 of the paper, ``benchmarks/additional_experiments/results/
highdim_fast.json``: held-out R^2 over five random 75/25 splits for beamfeat
and for ridge, random forest and LightGBM on the raw channels, one dot per
split and a bar at the mean. The style is that of the paper's figures, taken
from ``benchmarks/make_figures.py``. The dots are jittered vertically by a
seeded generator, so a rerun writes the same image.

Run:  python benchmarks/figures/tecator.py [--out docs/assets/tecator.png]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from make_figures import ACCENT, INK, NEUTRAL, use_acs_style  # noqa: E402

RESULTS = ROOT / "benchmarks" / "additional_experiments" / "results" / "highdim_fast.json"
OUT = ROOT / "docs" / "assets" / "tecator.png"
DATASET = "tecator_p100"
SEED = 0

# Top to bottom: the method's key in the results, its label, and its inputs.
METHODS = [
    ("beamfeat", "beamfeat", None),
    ("ridge_raw", "ridge", "100 channels"),
    ("lgbm_raw", "LightGBM", "100 channels"),
    ("rf_raw", "random forest", "100 channels"),
]


def pretty(formula: str) -> str:
    """``((x35 - x42) / (x37 - x39))`` as ``(x35 − x42) / (x37 − x39)``."""
    if formula.startswith("(") and formula.endswith(")"):
        formula = formula[1:-1]
    return formula.replace(" - ", " − ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=pathlib.Path, default=RESULTS)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    rows = sorted((r for r in json.loads(args.results.read_text()) if r["dataset"] == DATASET),
                  key=lambda r: r["split"])
    scores = {key: np.array([r["r2"] for r in rows if r["method"] == key]) for key, _, _ in METHODS}
    fits = [r for r in rows if r["method"] == "beamfeat"]
    n_splits = len(fits)
    if any(len(values) != n_splits for values in scores.values()):
        raise SystemExit(f"{args.results}: the methods do not cover the same splits of {DATASET}")
    counts = sorted({len(r["formulas"]) for r in fits})
    example = pretty(fits[0]["formulas"][0])

    print(f"{DATASET}, {n_splits} splits, mean held-out R^2:")
    for key, label, _ in METHODS:
        print(f"  {label:<14} {scores[key].mean():.3f}   (min {scores[key].min():.3f}, max {scores[key].max():.3f})")
    print(f"  beamfeat formulas per fit: {counts}; split 0's first: {fits[0]['formulas'][0]}")

    use_acs_style()
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    ys = np.arange(len(METHODS))[::-1]
    for y, (key, label, inputs) in zip(ys, METHODS):
        values = scores[key]
        colour = ACCENT if key == "beamfeat" else NEUTRAL
        ax.scatter(values, y + rng.uniform(-0.17, 0.17, len(values)), s=13, color=colour,
                   linewidths=0, alpha=0.85, zorder=3)
        ax.plot([values.mean()] * 2, [y - 0.3, y + 0.3], color=colour, linewidth=1.8,
                solid_capstyle="butt", zorder=4)
        if inputs is None:
            inputs = f"{counts[0]}–{counts[-1]} formulas" if len(counts) > 1 else f"{counts[0]} formulas"
        ax.annotate(label, (0, y), xycoords=("axes fraction", "data"), xytext=(-6, 0.3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=INK)
        ax.annotate(inputs, (0, y), xycoords=("axes fraction", "data"), xytext=(-6, -0.3),
                    textcoords="offset points", ha="right", va="top", fontsize=6.5, color=NEUTRAL)
        ax.annotate(f"{values.mean():.3f}", (1, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=7.5, color=INK)
    beam = scores["beamfeat"]
    ax.annotate(f"e.g. {example}", (beam.min(), ys[0]), xytext=(-9, 0), textcoords="offset points",
                ha="right", va="center", fontsize=6.5, color=NEUTRAL)
    ax.annotate("mean", (1, 1), xycoords="axes fraction", xytext=(6, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.5, color=NEUTRAL)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.6, len(METHODS) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel(f"held-out $R^2$ on tecator over {n_splits} splits: a dot per split, a bar at the mean")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
