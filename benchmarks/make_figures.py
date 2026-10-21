"""Generate the paper figures as vector PDFs in ACS format.

Reads the result files written by run_benchmarks.py and the calibration
scripts, and writes to paper/figures/. Figures are sized to the ACS column
widths (3.25 in single, 7.0 in double), set in a sans-serif face at 7-8 pt,
and drawn without gridlines or chart junk.

Run:  python benchmarks/make_figures.py [--outdir paper/figures]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

# ACS single- and double-column widths, in inches.
COL_SINGLE = 3.25
COL_DOUBLE = 7.0

# Colourblind-safe; beamfeat is the accent, everything else recedes.
ACCENT = "#14496E"        # deep slate blue - beamfeat
ACCENT_LT = "#4E86B4"     # lighter tint - beamfeat -> ridge
NEUTRAL = "#8D99A3"       # mid grey for annotations and reference lines
WARN = "#B4552B"          # muted rust for failures / warnings
INK = "#22282D"           # near-black for type and axis lines
GRID = "#E3E7EA"          # very light rule for grids
COLOURS = {
    "beamfeat": ACCENT,
    "beamfeat_ridge": ACCENT_LT,
    "rf_raw": "#5F6E79",
    "lgbm_raw": "#828F98",
    "ridge_raw": "#C6CCD1",
    "featuretools": "#DBE0E4",
    "autofeat": "#828F98",
    "openfe": "#AEB7BE",
    "knockpy": "#94A6B3",
    "lightgbm": "#5F6E79",
    "ridge": "#C6CCD1",
}

LABELS = {
    "beamfeat": "beamfeat",
    "beamfeat_ridge": "beamfeat → ridge",
    "autofeat": "autofeat",
    "openfe": "OpenFE",
    "featuretools": "featuretools",
    "knockpy": "knockpy",
    "lightgbm": "LightGBM",
    "ridge": "ridge",
    "rf_raw": "random forest",
    "lgbm_raw": "LightGBM",
    "ridge_raw": "ridge",
}

CORE10 = ["product", "ratio", "three_way", "log_linear", "sqrt_ratio", "quadratic",
          "inverse_square", "physics_kinetic", "sparse_10col", "purely_linear"]
STRESS = ["distractors_noise05", "distractors_noise25", "distractors_noise50",
          "threshold_step", "small_n120", "small_n240"]


def use_acs_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans",
                            "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Boxed frame: all four spines on, as ACS figures normally set them.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        # Light horizontal rules behind the data for readable value comparisons.
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": GRID,
        "legend.facecolor": "white",
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed as TrueType so text stays editable
        "ps.fonttype": 42,
    })


WRITE_PNG = True


def save(fig: plt.Figure, path: pathlib.Path) -> None:
    """Write the figure as PDF, and as PNG for embedding in notebooks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    if WRITE_PNG:
        fig.savefig(path.with_suffix(".png"), format="png", dpi=300)
    plt.close(fig)
    print(f"  wrote {path}")


def _load(bench: pathlib.Path, *stems: str) -> pd.DataFrame:
    """Concatenate result files, preferring CSV and falling back to JSON.

    The harness writes both formats; a figure that silently skipped a missing
    one would quietly drop a method from the comparison, which is how autofeat
    once vanished from the false-feature figure. Missing sources are named.
    """
    frames, missing = [], []
    for stem in stems:
        for suffix in (".csv", ".json"):
            path = bench / f"{stem}{suffix}"
            if path.exists():
                frames.append(pd.read_csv(path) if suffix == ".csv"
                              else pd.DataFrame(json.load(open(path))))
                break
        else:
            missing.append(stem)
    if missing:
        print(f"  WARNING: no result file for {', '.join(missing)}; "
              "the figure below is missing those methods")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig_core_accuracy(bench: pathlib.Path, out: pathlib.Path) -> None:
    """Mean R^2 on the ten formula-recovery problems."""
    # results.json already carries the merged autofeat/knockpy rows; loading the
    # venv files as well would average two different stochastic draws.
    df = _load(bench, "results")
    if df.empty:
        return
    core = df[df.dataset.isin(CORE10)]
    means = core.groupby("method").r2.mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.1))
    x = np.arange(len(means))
    ax.bar(x, means.values, width=0.68, edgecolor="white", linewidth=0.5,
           color=[COLOURS.get(m, NEUTRAL) for m in means.index])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in means.index])
    for xi, v in zip(x, means.values):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.5, color=INK)
    ax.set_ylabel("mean $R^2$")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    save(fig, out / "fig_core_accuracy.pdf")


def fig_false_features(bench: pathlib.Path, out: pathlib.Path) -> None:
    """Stress datasets returning false features, as counts.

    Counts rather than a mean: with five scoreable datasets a mean carries a
    standard error of order 0.1, which is the reason the Benchmarks page
    (docs/benchmarks.md) reports the count. Drawn as markers rather than bars
    so that a result of zero is visible — a zero-height bar renders the best
    outcome as nothing.
    """
    df = _load(bench, "results_robustness", "results_autofeat_robustness",
               "results_knockpy")
    if df.empty:
        return
    s = df[df.dataset.isin(STRESS)].dropna(subset=["false_feature_rate"])
    if s.empty:
        return
    stats = s.groupby("method").false_feature_rate.agg(
        affected=lambda r: int((r > 0).sum()), n="size", worst="max"
    ).sort_values("affected")
    total = int(stats["n"].iloc[0])

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.2))
    y = np.arange(len(stats))[::-1]
    for yi, (method, row) in zip(y, stats.iterrows()):
        colour = ACCENT if method == "beamfeat" else NEUTRAL
        if row.affected:
            ax.plot([0, row.affected], [yi, yi], color=colour, linewidth=1.3,
                    solid_capstyle="round", zorder=2)
        ax.scatter([row.affected], [yi], s=26, color=colour, zorder=3)
        label = f"{int(row.affected)} of {int(total)}"
        if row.affected:
            label += f", worst {row.worst:.2f}"
        ax.annotate(label, (row.affected, yi), textcoords="offset points",
                    xytext=(9, -2.5), fontsize=6.5,
                    color=WARN if row.affected else "#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(m, m) for m in stats.index])
    ax.set_xlim(-0.15, total)
    ax.set_xticks(range(total + 1))
    ax.set_xlabel("stress datasets returning a formula that touches\n"
                  "only irrelevant columns", fontsize=7)
    save(fig, out / "fig_false_feature_rate.pdf")


def fig_calibration(out: pathlib.Path) -> None:
    """Realised against nominal FDR for both multiplicity corrections."""
    try:
        import selector_calibration
    except ImportError:
        return
    table = selector_calibration.main()["permutation"]
    nominal = [0.05, 0.10, 0.20]

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.4))
    ax.plot([0, 0.22], [0, 0.22], color=NEUTRAL, linewidth=0.7,
            linestyle=(0, (3, 2)), zorder=1)
    ax.annotate("exact control", xy=(0.176, 0.176), xytext=(0.116, 0.212),
                fontsize=6, color=NEUTRAL,
                arrowprops=dict(arrowstyle="-", color=NEUTRAL, linewidth=0.5))
    for corr, colour, marker, label in (("bh", WARN, "o", "Benjamini-Hochberg"),
                                        ("by", ACCENT, "s", "Benjamini-Yekutieli")):
        realised = [table[(corr, q)]["realised"] for q in nominal]
        ax.plot(nominal, realised, marker=marker, markersize=3.2, color=colour,
                label=label, zorder=3)
    ax.set_xlabel("nominal FDR")
    ax.set_ylabel("realised FDR")
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0, 0.24)
    ax.set_xticks(nominal)
    ax.legend(loc="upper left")
    save(fig, out / "fig_fdr_calibration.pdf")


def fig_real_panel(bench: pathlib.Path, out: pathlib.Path) -> None:
    """Held-out R^2 across the real datasets, grouped by method."""
    p = bench / "results_real.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    piv = df.pivot_table(index="dataset", columns="method", values="r2")
    order = [m for m in ("beamfeat", "lightgbm", "ridge") if m in piv.columns]
    piv = piv[order].sort_values(order[0], ascending=False)

    fig, ax = plt.subplots(figsize=(COL_DOUBLE * 0.62, 2.3))
    x = np.arange(len(piv))
    width = 0.8 / len(order)
    for i, m in enumerate(order):
        ax.bar(x + i * width - 0.4 + width / 2, piv[m], width=width * 0.92,
               color=COLOURS.get(m, NEUTRAL), edgecolor="white", linewidth=0.5, label=LABELS.get(m, m))
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=20, ha="right")
    ax.set_ylabel("held-out $R^2$")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=len(order), loc="upper right")
    save(fig, out / "fig_real_datasets.pdf")


def _study_csv(root: pathlib.Path):
    """Per-fit results of the comparison study, or None with a reason.

    Silently skipping a figure hides the common case: the study directory was
    promoted but `aggregate.py` has not been run over it, so the CSV the
    figures read does not exist yet.
    """
    p = root / "benchmarks" / "independent" / "results_as_reported" / "independent_benchmark_results.csv"
    if p.exists():
        return p
    print(f"  skipping the comparison-study figures: {p} not found."
          "\n    build it with: python benchmarks/independent/aggregate.py results_as_reported")
    return None


def fig_worst_case(root: pathlib.Path, out: pathlib.Path) -> None:
    """Mean against worst-case R^2 from the independent comparison study."""
    p = _study_csv(root)
    if p is None:
        return
    df = pd.read_csv(p)
    stats = df.groupby("method").r2.agg(["mean", "min"]).sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.4))
    y = np.arange(len(stats))[::-1]
    for yi, (method, row) in zip(y, stats.iterrows()):
        colour = ACCENT if method == "beamfeat" else NEUTRAL
        ax.plot([row["min"], row["mean"]], [yi, yi], color=colour, linewidth=1.4,
                solid_capstyle="round", zorder=2)
        ax.scatter([row["min"]], [yi], s=11, color=WARN, zorder=3)
        ax.scatter([row["mean"]], [yi], s=14, color=colour, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(m, m) for m in stats.index])
    ax.set_xlabel("held-out $R^2$")
    ax.set_xlim(-3.2, 1.15)
    ax.axvline(0, color=NEUTRAL, linewidth=0.6, zorder=1)
    ax.scatter([], [], s=14, color=NEUTRAL, label="mean")
    ax.scatter([], [], s=11, color=WARN, label="worst fit")
    ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.3)
    ax.text(0.99, 0.02, "featuretools worst fit $-57.3$, clipped",
            transform=ax.transAxes, fontsize=6, color=NEUTRAL,
            ha="right", va="bottom")
    save(fig, out / "fig_worst_case.pdf")


def fig_accuracy_vs_cost(root: pathlib.Path, out: pathlib.Path) -> None:
    """Mean held-out R^2 against mean fit time, from the comparison study.

    No error bars: the spread across the 45 fits is dominated by differences
    between the nine datasets rather than by uncertainty about a method, so an
    interval drawn from it would invite the wrong reading. The paired tests in
    the study report carry the inference.
    """
    p = _study_csv(root)
    if p is None:
        return
    df = pd.read_csv(p)
    stats = pd.DataFrame({
        "r2": df.groupby("method").r2.mean(),
        "seconds": df.groupby("method").seconds.mean(),
    }).sort_values("r2", ascending=False)

    fig, ax = plt.subplots(figsize=(COL_SINGLE * 1.42, 2.3))
    lo = 0.68
    shown, offscale = stats[stats.r2 >= lo], stats[stats.r2 < lo]

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for marker, (method, row) in zip(markers, shown.iterrows()):
        accent = method.startswith("beamfeat")
        ax.scatter(row.seconds, row.r2, marker=marker, s=34 if accent else 24,
                   color=ACCENT if accent else NEUTRAL,
                   edgecolor="white", linewidth=0.4, zorder=3 if accent else 2,
                   label=LABELS.get(method, method))

    ax.set_xscale("log")
    ax.set_xlim(1e-3, 300)
    ax.set_ylim(lo, shown.r2.max() + 0.03)
    ax.set_xlabel("mean fit time per dataset-split (s)")
    ax.set_ylabel("mean held-out $R^2$")
    # Outside the axes: the points span all four corners, so any inside
    # placement covers one of them.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
              handletextpad=0.3, labelspacing=0.5, borderaxespad=0.0,
              frameon=False)
    if len(offscale):
        names = ", ".join(LABELS.get(m, m) for m in offscale.index)
        ax.text(0.99, -0.22, f"{names} off scale at {offscale.r2.min():.2f}",
                transform=ax.transAxes, fontsize=6, color=NEUTRAL,
                ha="right", va="top")
    save(fig, out / "fig_accuracy_vs_cost.pdf")


def fig_fit_distribution(root: pathlib.Path, out: pathlib.Path) -> None:
    """Every individual fit in the comparison study, one column per method.

    A mean hides the failures that matter. Drawn on a symmetric log scale so
    that scores near 1 stay legible while catastrophic fits, which reach two
    orders of magnitude below zero, remain on the axis rather than clipped.
    """
    p = _study_csv(root)
    if p is None:
        return
    df = pd.read_csv(p)
    order = df.groupby("method").r2.median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(COL_DOUBLE * 0.62, 2.6))
    rng = np.random.default_rng(0)
    for i, method in enumerate(order):
        vals = df[df.method == method].r2.dropna().to_numpy()
        accent = method.startswith("beamfeat")
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(vals)), vals,
                   s=7, alpha=0.8, edgecolor="none",
                   color=ACCENT if accent else NEUTRAL, zorder=3 if accent else 2)
        below = int((vals < 0).sum())
        if below:
            ax.annotate(f"{below}", (i, 0.055), xycoords=("data", "axes fraction"),
                        ha="center", fontsize=6, color=WARN)

    ax.axhline(0, color=NEUTRAL, linewidth=0.6, zorder=1)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS.get(m, m) for m in order], rotation=20, ha="right")
    ax.set_ylabel("held-out $R^2$ per fit")
    ax.set_ylim(min(-200, df.r2.min() * 1.5), 1.6)
    ax.text(0.0, -0.30, "symmetric log scale below 1; red count = fits worse "
            "than predicting the mean", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color="#666666")
    save(fig, out / "fig_fit_distribution.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=None, help="destination for the PDFs")
    parser.add_argument("--no-png", action="store_true",
                        help="skip the PNG previews used for notebook display")
    args = parser.parse_args()

    global WRITE_PNG
    WRITE_PNG = not args.no_png

    root = pathlib.Path(__file__).resolve().parent.parent
    bench = root / "benchmarks"
    out = pathlib.Path(args.outdir) if args.outdir else root / "paper" / "figures"

    import sys
    sys.path.insert(0, str(bench))

    use_acs_style()
    print(f"writing figures to {out}")
    fig_core_accuracy(bench, out)
    fig_false_features(bench, out)
    fig_calibration(out)
    fig_real_panel(bench, out)
    fig_worst_case(root, out)
    fig_accuracy_vs_cost(root, out)
    fig_fit_distribution(root, out)


if __name__ == "__main__":
    main()
