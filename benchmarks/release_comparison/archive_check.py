"""0.4.2 against the archived benchmark results of the paper era.

Fits the 25 datasets of benchmarks/run_benchmarks.py with that script's own
settings and split (depth 2, beam 40, q = 0.1, one 70/30 split, seed 0) and
the Feynman panel with its own, and compares each held-out R² with
benchmarks/results*.csv and benchmarks/feynman_results.json.

    venvs/0.4.2/bin/python -I archive_check.py
"""
import csv
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
import harness as h  # noqa: E402

archive = {}
for name in ("results.csv", "results_real.csv", "results_robustness.csv"):
    for row in csv.DictReader((h.BENCH / name).open()):
        if row["method"] == "beamfeat":
            archive.setdefault(row["dataset"], (float(row["r2"]), row["n_features"]))
same = 0
for name in h.SUITE_NAMES:
    X, y, Xt, yt, problem, _, _ = h.load_item("suite", f"{name}:0")
    model = h.estimator(problem)(max_depth=2, beam_width=40, selector="permutation", parsimony="forward",
                                 target_fdr=0.1, random_state=0).fit(X, y)
    got, (expected, terms) = float(model.score(Xt, yt)), archive[name]
    same += abs(got - expected) < 1e-9
    print(f"{name:20s} archive {expected:.4f} ({terms} terms)  this release {got:.4f} ({len(model.features_)})")
print(f"run_benchmarks.py: {same} of {len(h.SUITE_NAMES)} identical\n")

panel = json.loads((h.BENCH / "feynman_results.json").read_text())[0]
same = 0
for row in panel["rows"]:
    X, y, Xt, yt, _, base, measure = h.load_item("feynman", f"{row['equation']}:0")
    got = measure(h.estimator("regression")(**base).fit(X, y))["score"]
    same += abs(got - row["r2"]) < 1e-9
print(f"Feynman panel: {same} of {len(panel['rows'])} laws identical")
