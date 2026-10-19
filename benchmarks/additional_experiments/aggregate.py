"""Aggregate benchmark results and run the statistical comparison.

Reads every highdim_*.json (the comparison records `bench.py` writes, as
named in the README run order) in a results directory, prints the tables reported
in the study (mean +/- std R^2, fit seconds, feature counts, recovery rates,
average ranks, Friedman test, paired Wilcoxon tests), and writes
independent_benchmark_results.csv beside them.

Usage:
    python aggregate.py [results_dir]

The directory defaults to results (this suite's output).
"""
import glob
import json
import os
import sys

import pandas as pd
from scipy import stats

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"
paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "highdim_*.json")))
if not paths:
    raise SystemExit(f"no highdim_*.json under {RESULTS_DIR}/")
print(f"source: {RESULTS_DIR}/  ({len(paths)} file(s))")
rows = []
for f in paths:
    rows += json.load(open(f))
df = pd.DataFrame(rows)
df = df[df.error.isna()].copy()
df = df.drop_duplicates(subset=["dataset", "method", "split"], keep="last")

REAL = ["concrete", "wine_red", "boston", "diabetes"]
ORDER = ["ridge_raw", "rf_raw", "lgbm_raw", "featuretools", "openfe", "autofeat", "beamfeat"]
ORDER = [m for m in ORDER if m in set(df.method)]

def table(datasets):
    out = {}
    for d in datasets:
        sub = df[df.dataset == d]
        out[d] = {}
        for m in ORDER:
            s = sub[sub.method == m].r2
            out[d][m] = f"{s.mean():+.3f} ± {s.std():.3f}" if len(s) else "-"
    return pd.DataFrame(out).T

real = [d for d in REAL if d in set(df.dataset)]
syn = [d for d in df.dataset.unique() if d not in REAL]
if real:
    print("=== REAL: out-of-sample R2, mean ± std ===")
    print(table(real).to_string(), "\n")
if syn:
    # the highdim group is real data outside the original study's REAL list
    print(f"=== {'SYNTHETIC' if real else 'ALL'}: out-of-sample R2, mean ± std ===")
    print(table(syn).to_string(), "\n")

print("=== Fit seconds (mean) ===")
print(df.pivot_table(index="dataset", columns="method", values="seconds",
                     aggfunc="mean")[ORDER].round(1).to_string(), "\n")

tools = [m for m in ["featuretools", "openfe", "autofeat", "beamfeat"] if m in ORDER]
print("=== N constructed features (mean) ===")
print(df[df.method.isin(tools)].pivot_table(index="dataset", columns="method",
      values="n_new", aggfunc="mean").round(1).to_string(), "\n")

rec = df[df.recovered.notna()]
if len(rec):
    print("=== Formula recovery rate ===")
    print(rec.pivot_table(index="dataset", columns="method", values="recovered",
                          aggfunc="mean").round(2).to_string(), "\n")

bf = df[df.method == "beamfeat"].copy()
if len(bf):
    bf["fdr"] = bf["fdr"].astype(float)
    print("beamfeat fdr_controlled_ rate by dataset:")
    print(bf.groupby("dataset").fdr.mean().round(2).to_string(), "\n")

mr = df.pivot_table(index="dataset", columns="method", values="r2", aggfunc="mean")[ORDER]
ranks = mr.rank(axis=1, ascending=False)
print("=== Average rank (1=best), all datasets ===")
print(ranks.mean().sort_values().round(2).to_string())
if real and syn:
    print("REAL only:")
    print(ranks.loc[real].mean().sort_values().round(2).to_string())
    print("SYN only:")
    print(ranks.loc[syn].mean().sort_values().round(2).to_string())

if len(mr) >= 3 and len(ORDER) >= 3:
    fr = stats.friedmanchisquare(*[mr[m].values for m in ORDER])
    print(f"\nFriedman chi2={fr.statistic:.2f} p={fr.pvalue:.5f} "
          f"({len(mr)} datasets, {len(ORDER)} methods)")

print("\n=== Wilcoxon signed-rank, beamfeat vs X, paired on dataset x split ===")
piv = df.pivot_table(index=["dataset", "split"], columns="method", values="r2")
for m in ORDER:
    if m == "beamfeat" or "beamfeat" not in piv:
        continue
    pair = piv[["beamfeat", m]].dropna()
    d = pair["beamfeat"] - pair[m]
    if len(d) < 6:
        continue
    w = stats.wilcoxon(d)
    print(f"beamfeat vs {m:13s} n={len(d):2d} median diff={d.median():+.4f} "
          f"mean diff={d.mean():+.4f} p={w.pvalue:.4f}")

df.to_csv(os.path.join(RESULTS_DIR, "independent_benchmark_results.csv"), index=False)
print(f"\nSaved independent_benchmark_results.csv ({len(df)} runs)")
