"""The tables of REPORT.md, from results.jsonl.

    python report.py > REPORT.md
"""
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = [json.loads(line) for line in (HERE / "results.jsonl").open()]
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
