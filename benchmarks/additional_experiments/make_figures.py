"""Generate the additional-experiment figures as vector PDFs.

Reads the result files written by bench.py, depth_ladder.py, scalability.py,
selector_comparison.py and split_stability.py, and writes to figures/.
Palette, sizing and axis conventions mirror benchmarks/make_figures.py so the
new figures sit beside the paper's originals without restyling. Figures whose
result files are absent are skipped with a note.

Run:  python make_figures.py [--results results] [--outdir figures]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import warnings

import matplotlib
import numpy as np
import pandas as pd

if __name__ == "__main__":          # importing this module leaves the caller's backend alone
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

COL_SINGLE = 3.25
COL_DOUBLE = 7.0

# Palette as in benchmarks/make_figures.py: beamfeat is the accent, the rest recede.
ACCENT = "#14496E"
ACCENT_LT = "#4E86B4"
NEUTRAL = "#8D99A3"
WARN = "#B4552B"
INK = "#22282D"
GRID = "#E3E7EA"
COLOURS = {
    "beamfeat": ACCENT,
    "beamfeat_ridge": ACCENT_LT,
    "rf_raw": "#5F6E79",
    "lgbm_raw": "#828F98",
    "ridge_raw": "#C6CCD1",
    "featuretools": "#DBE0E4",
    "autofeat": "#828F98",
    "openfe": "#AEB7BE",
}
LABELS = {
    "beamfeat": "beamfeat",
    "beamfeat_ridge": "beamfeat \u2192 ridge",
    "rf_raw": "random forest",
    "lgbm_raw": "LightGBM",
    "ridge_raw": "ridge",
    "featuretools": "featuretools",
    "autofeat": "autofeat",
    "openfe": "OpenFE",
}
SELECTOR_COLOURS = {"bh": WARN, "by": ACCENT,
                    "knockoff_fixed": "#5F6E79", "knockoff_modelx": "#AEB7BE"}
SELECTOR_MARKERS = {"bh": "o", "by": "s", "knockoff_fixed": "^", "knockoff_modelx": "D"}
SELECTOR_LABELS = {"bh": "Benjamini-Hochberg", "by": "Benjamini-Yekutieli",
                   "knockoff_fixed": "knockoff (fixed-X)",
                   "knockoff_modelx": "knockoff (model-X)"}


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
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
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
        "pdf.fonttype": 42,
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


def load_rows(results: pathlib.Path, *stems: str) -> pd.DataFrame:
    rows: list[dict] = []
    for stem in stems:
        for f in sorted(glob.glob(str(results / f"{stem}*.json"))):
            data = json.load(open(f))
            if isinstance(data, list):
                rows += data
    return pd.DataFrame(rows)


def fig_selector_calibration(results: pathlib.Path, outdir: pathlib.Path) -> None:
    # Plot the larger candidate pool, which is the run the paper reports; fall
    # back to the smaller one only when the m=100 file is absent.
    primary = results / "selector_comparison_m100.json"
    fallback = results / "selector_comparison.json"
    source = primary if primary.exists() else fallback
    if source.exists():
        df = pd.DataFrame(json.load(open(source)))
    else:
        df = load_rows(results, "selector_comparison")
    if df.empty:
        print("  selector_comparison: no results, skipped")
        return
    df = (df.groupby(["regime", "selector", "nominal"], as_index=False)
            .agg({"fdr_marginal": "mean", "fdr_conditional": "mean", "power": "mean"}))
    panels = [("independent", "fdr_marginal", "independent"),
              ("correlated_nulls", "fdr_marginal", "correlated nulls"),
              ("shared_factor", "fdr_conditional", "shared factor (conditional)")]
    panels = [(r, c, t) for r, c, t in panels if (df.regime == r).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(COL_DOUBLE, 2.1))
    axes = np.atleast_1d(axes)
    for ax, (regime, col, title) in zip(axes, panels):
        sub = df[df.regime == regime]
        lim = max(0.22, float(sub[col].max() or 0) * 1.1)
        ax.plot([0, lim], [0, lim], linestyle="--", linewidth=0.8,
                color=NEUTRAL, zorder=1)
        # Series can coincide exactly -- under a shared factor BH and BY select
        # identically -- so draw earlier ones larger and let them show through.
        kinds = ["bh", "by", "knockoff_fixed", "knockoff_modelx"]
        for ki, kind in enumerate(kinds):
            s = sub[sub.selector == kind].sort_values("nominal")
            if s.empty or s[col].isna().all():
                continue
            ax.plot(s.nominal, s[col], marker=SELECTOR_MARKERS[kind],
                    color=SELECTOR_COLOURS[kind], label=SELECTOR_LABELS[kind],
                    markersize=6.0 - 0.7 * ki, zorder=2 + (len(kinds) - ki))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(title)
        ax.set_xlabel("nominal FDR")
    axes[0].set_ylabel("realised FDR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(labels), frameon=False, handlelength=1.6,
               columnspacing=1.6)
    save(fig, outdir / "selector_calibration.pdf")


def fig_depth_ladder(results: pathlib.Path, outdir: pathlib.Path) -> None:
    df = load_rows(results, "depth_ladder")
    if df.empty:
        print("  depth_ladder: no results, skipped")
        return
    df = df[df.recovered_all.notna() & (df.scorer == "correlation")]
    df["kind"] = np.where(df.problem.str.startswith("adv_"),
                          "marginally invisible", "visible")
    beams = sorted(df.beam.unique())
    shades = dict(zip(beams, [ACCENT_LT, ACCENT, "#0C2C44", "#06182A"][:len(beams)]))
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.1), sharey=True,
                             gridspec_kw={"width_ratios": [3, 2]})
    for ax, kind in zip(axes, ["visible", "marginally invisible"]):
        sub = df[df.kind == kind]
        depths = sorted(sub.depth.unique())
        width = 0.8 / max(len(beams), 1)
        for bi, beam in enumerate(beams):
            vals = [sub[(sub.depth == d) & (sub.beam == beam)].recovered_all.mean()
                    for d in depths]
            xs = np.arange(len(depths)) + (bi - (len(beams) - 1) / 2) * width
            ax.bar(xs, vals, width=width * 0.92, color=shades[beam],
                   edgecolor="white", linewidth=0.5, label=f"beam {beam}")
        ax.set_xticks(np.arange(len(depths)))
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_xlabel("minimal search depth")
        ax.set_title(kind)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("recovery rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(labels), frameon=False, handlelength=1.6,
               columnspacing=1.6)
    save(fig, outdir / "depth_ladder.pdf")


def fig_scalability(results: pathlib.Path, outdir: pathlib.Path) -> None:
    df = load_rows(results, "scalability")
    if df.empty:
        print("  scalability: no results, skipped")
        return
    ok = df[df.get("error").isna()] if "error" in df else df
    if ok.empty:
        print("  scalability: only failed cells, skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.1))
    colours = {"independent": ACCENT, "equicorrelated": WARN}
    for regime, sub in ok.groupby("regime"):
        agg = sub.groupby("p").agg(sec=("seconds", "mean"), mb=("peak_mb", "mean"),
                                   rec=("recovered", "mean"))
        axes[0].plot(agg.index, agg.sec, marker="o",
                     color=colours.get(regime, NEUTRAL), label=regime)
        axes[1].plot(agg.index, agg.mb, marker="o",
                     color=colours.get(regime, NEUTRAL), label=regime)
    for ax, ylab in zip(axes, ["fit time (s)", "peak memory (MB)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("input columns p")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="both", color=GRID, linewidth=0.5)
    axes[0].legend(loc="upper left")
    save(fig, outdir / "scalability.pdf")


def fig_split_stability(results: pathlib.Path, outdir: pathlib.Path) -> None:
    merged: dict = {}
    for f in sorted(glob.glob(str(results / "split_stability*.json"))):
        data = json.load(open(f))
        if isinstance(data, dict):
            merged.update(data)
    if not merged:
        print("  split_stability: no results, skipped")
        return
    names = sorted(merged, key=lambda k: merged[k]["jaccard_mean"])
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 0.34 * len(names) + 0.9),
                             sharey=True)
    for yi, name in zip(y, names):
        v = merged[name]
        axes[0].plot([v["r2_min"], v["r2_max"]], [yi, yi], color=ACCENT, linewidth=1.3)
        axes[0].scatter([v["r2_mean"]], [yi], s=22, color=ACCENT, zorder=3)
        axes[1].scatter([v["jaccard_mean"]], [yi], s=22, color=ACCENT, zorder=3)
        # Label to the right of the point, except near the axis limit, where
        # it would straddle the frame; pluralise properly for a single class.
        label = f"{v['n_classes']} class" + ("" if v["n_classes"] == 1 else "es")
        right = v["jaccard_mean"] > 0.85
        axes[1].annotate(f"{label}  " if right else f"  {label}",
                         (v["jaccard_mean"], yi), fontsize=6, color=NEUTRAL,
                         va="center", ha="right" if right else "left")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("held-out $R^2$ across splits (min-mean-max)")
    axes[1].set_xlabel("mean Jaccard of selections, value-equivalent")
    axes[1].set_xlim(-0.03, 1.03)
    for ax in axes:
        ax.grid(True, axis="x", color=GRID, linewidth=0.5)
        ax.grid(False, axis="y")
    save(fig, outdir / "split_stability.pdf")


def fig_highdim(results: pathlib.Path, outdir: pathlib.Path) -> None:
    df = load_rows(results, "highdim_")
    if df.empty:
        print("  highdim: no results, skipped")
        return
    ok = df[df.error.isna()] if "error" in df else df
    piv = ok.pivot_table(index="dataset", columns="method", values="r2", aggfunc="mean")
    order = [m for m in ["beamfeat", "beamfeat_ridge", "rf_raw", "lgbm_raw",
                         "ridge_raw", "openfe", "featuretools", "autofeat"]
             if m in piv.columns]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.3))
    n = len(order)
    width = 0.8 / n
    for mi, m in enumerate(order):
        xs = np.arange(len(piv.index)) + (mi - (n - 1) / 2) * width
        ax.bar(xs, piv[m].values, width=width * 0.92,
               color=COLOURS.get(m, NEUTRAL), edgecolor="white", linewidth=0.5,
               label=LABELS.get(m, m))
    ax.set_xticks(np.arange(len(piv.index)))
    ax.set_xticklabels([str(d).replace("_", " ") for d in piv.index],
                       rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylabel("held-out $R^2$")
    lo = float(np.nanmin(piv.values)) if np.isfinite(piv.values).any() else 0.0
    ax.set_ylim(min(0.0, lo - 0.02), 1.02)
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=min(5, n), frameon=False, handlelength=1.4,
               columnspacing=1.4)
    save(fig, outdir / "highdim_comparison.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", type=pathlib.Path)
    ap.add_argument("--outdir", default="figures", type=pathlib.Path)
    a = ap.parse_args()
    use_acs_style()
    for fn in (fig_highdim, fig_depth_ladder, fig_scalability,
               fig_selector_calibration, fig_split_stability):
        fn(a.results, a.outdir)


if __name__ == "__main__":
    main()
