"""The tables of REPORT.md, from results.jsonl.

    python report.py > REPORT.md
"""
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALL = [json.loads(line) for line in (HERE / "results.jsonl").open()]
for path in sorted(HERE.glob("results_*.jsonl")):  # an earlier release, beamfeat alone on the documented split
    ALL += [json.loads(line) for line in path.open()]
R = [r for r in ALL if r["protocol"] == "random"]
D = [r for r in ALL if r["protocol"] == "documented"]
METHODS = list(dict.fromkeys(r["method"] for r in R))
# Table 6 of the paper, not rerun here: the tree ensembles on all 100 channels.
PAPER = {"random forest (Table 6)": 0.717, "LightGBM (Table 6)": 0.756}


def wavelength(column):
    """Channel xk of the 100-channel spectrum, 850 to 1048 nm in 2 nm steps."""
    return 850 + 2 * column


def fmt(values, digits):
    return f"{st.mean(values):.{digits}f} ± {st.stdev(values):.{digits}f}"


def size(rows):
    sizes = [r["size"] for r in rows]
    if sizes[0] is None:
        return "all 100 channels"
    unit = "terms" if rows[0]["method"].startswith("beamfeat") else "components"
    low, high = min(sizes), max(sizes)
    return f"{low} {unit}" if low == high else f"{low} to {high} {unit}"


print("# tecator and the baselines of chemometrics\n")
print(f"beamfeat {R[0]['beamfeat']}. The fat content of 240 meat samples from 100 near-infrared absorbance "
      f"channels (850 to 1048 nm), on the five 75/25 splits of the paper's high-dimensional study "
      f"({R[0]['rows'][0]} training and {R[0]['rows'][1]} test rows each). Mean ± standard deviation over "
      "the splits; RMSE is in percent fat. `run.py` describes each method and its tuning; `PROVENANCE.md` "
      "how to rerun. These baselines were added after the paper, which compared ridge, a random forest "
      "and LightGBM only.\n")
print("| method | held-out R² | RMSE | size |")
print("| :-- | --: | --: | :-- |")
for name in sorted(METHODS, key=lambda m: -st.mean(r["r2"] for r in R if r["method"] == m)):
    rows = [r for r in R if r["method"] == name]
    print(f"| {name} | {fmt([r['r2'] for r in rows], 3)} | {fmt([r['rmse'] for r in rows], 2)} | {size(rows)} |")
for name, score in PAPER.items():
    print(f"| {name} | {score:.3f} | — | all 100 channels |")
print()
print("\"derivative\" is a Savitzky-Golay derivative of the spectra (polynomial order 2), whose window, "
      "and for PLS whose order, was tuned; for beamfeat it is the second derivative with a window of 15.\n")

print("## The formulas beamfeat printed\n")
for name in ("beamfeat", "beamfeat, derivative"):
    print(f"`{name}`:\n")
    columns = Counter()
    for r in (r for r in R if r["method"] == name):
        print(f"- split {r['split']}: " + ", ".join(f"`{f}`" for f in r["formulas"]))
        columns.update({int(c) for f in r["formulas"] for c in re.findall(r"x(\d+)", f)})
    used = sorted(columns)
    print(f"\nChannels used: {len(used)}, at {wavelength(used[0])} to {wavelength(used[-1])} nm; "
          "the most frequent: " +
          ", ".join(f"x{c} ({wavelength(c)} nm, {n} of 5 splits)" for c, n in columns.most_common(4)) + ".\n")

print("## The split of the data's documentation\n")
print("The documentation of the data (StatLib) divides the samples into the sets C (129), M (43), T (43), "
      "E1 (8) and E2 (17), in file order, and asks that a model be tuned on C and M, tested on T, and that a "
      "model with an element of randomness be quoted as the average of a handful of fits, not the best on T. "
      "E1 holds more fat than any training sample (54.7 to 58.5% against at most 49.1%) and E2 more protein, "
      "to test extrapolation. RMSE is in percent fat; beamfeat's figures are means over five seeds, with the "
      "seeds' E1 errors listed. A row \"beamfeat <version>\" is that earlier release at its defaults "
      "(`run.py --release`); since 0.7.0 the search rules out a quotient whose denominator comes near zero "
      "on the search rows.\n")
print("| method | T: RMSE | T: R² | E1: RMSE | E2: RMSE | size |")
print("| :-- | --: | --: | --: | --: | :-- |")
for name in dict.fromkeys(r["method"] for r in D):
    rows = [r for r in D if r["method"] == name]
    e1 = st.mean(r["rmse_E1"] for r in rows)
    seeds = " (" + ", ".join(f"{r['rmse_E1']:.1f}" for r in rows) + ")" if len(rows) > 1 else ""
    sizes = [r["size"] for r in rows]
    if sizes[0] is None:
        shown = "all 100 channels"
    else:
        unit = "terms" if name.startswith("beamfeat") else "components"
        shown = f"{min(sizes)} {unit}" if min(sizes) == max(sizes) else f"{min(sizes)} to {max(sizes)} {unit}"
    print(f"| {name} | {st.mean(r['rmse_T'] for r in rows):.2f} | {st.mean(r['r2_T'] for r in rows):.3f} | "
          f"{e1:.2f}{seeds} | {st.mean(r['rmse_E2'] for r in rows):.2f} | {shown} |")
print("\nThe documentation quotes, for T, 0.65 to 0.36 for three neural networks of the early 1990s on the "
      "principal components (Borggaard & Thodberg, 1992; Thodberg) and 2.78 for a linear model with 10 inputs.\n")
print("### How far the formulas reach beyond the training data\n")
print("For each beamfeat fit, \"outside\" is the largest distance by which one of its terms goes beyond its "
      "range on the training samples, in widths of that range, over the rows of each set.\n")
print("| method | seed | outside on T | outside on E1 | E1: RMSE | E1: predictions |")
print("| :-- | --: | --: | --: | --: | :-- |")
for r in (r for r in D if r["seed"] is not None):
    low, high = r["range_E1"]
    print(f"| {r['method']} | {r['seed']} | {max(r['outside_T'], 0):.2f} | {max(r['outside_E1'], 0):.2f} | "
          f"{r['rmse_E1']:.2f} | {low:.0f} to {high:.0f} |")
print()
