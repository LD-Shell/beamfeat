"""The tables of REPORT.md, from results/<setting>.jsonl.

    python report.py > REPORT.md
"""
import csv
import gzip
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SETTINGS = ["default", "parsimony_alpha", "input_test", "both"]
LABEL = {"default": "default", "parsimony_alpha": "`parsimony_alpha=0.05`", "input_test": "`input_test=True`",
         "both": "both"}
REAL_SUITE = ["diabetes", "breast_cancer", "mpg", "diamonds", "penguins", "tips", "california_housing"]
REAL_HIGHDIM = ["communities", "superconduct", "tecator", "eyedata", "riboflavin"]
CORE = ["product", "ratio", "three_way", "log_linear", "sqrt_ratio", "quadratic", "inverse_square",
        "physics_kinetic", "sparse_10col", "purely_linear"]
RECOVERY = CORE[:-1] + ["heavy_tail_t2"]
STRESS = ["distractors_noise05", "distractors_noise25", "distractors_noise50", "small_n120", "small_n240"]
SAME = 0.005

R = {s: [json.loads(line) for line in (HERE / "results" / f"{s}.jsonl").open()] for s in SETTINGS}
BY = {s: {(r["study"], r["item"]): r for r in R[s]} for s in SETTINGS}
# Splits where some setting left test rows without a prediction, scored on the rows every setting predicts.
RESCORED = json.loads((HERE / "results" / "rescore.json").read_text())
for fix in RESCORED:
    record = BY[fix["setting"]][(fix["study"], fix["item"])]
    record["raw_score"], record["score"] = record["score"], fix["score"]


def rows(setting, study, name=None):
    return [r for r in R[setting] if r["study"] == study and (name is None or r["item"].split(":")[0] == name)]


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else float("nan")


def cell(setting, study, name):
    rs = rows(setting, study, name)
    return mean([r["score"] for r in rs]), mean([r["terms"] for r in rs])


def time_ratio(setting, keys):
    """Geometric mean over keys of this setting's fit time over the default's."""
    logs = [math.log(BY[setting][k]["seconds"] / BY["default"][k]["seconds"]) for k in keys]
    return math.exp(sum(logs) / len(logs))


def header(first):
    print(f"| {first} | " + " | ".join(LABEL[s] for s in SETTINGS) + " |")
    print("| :-- |" + " --: |" * len(SETTINGS))


def sweep():
    print("## Noise sweep\n")
    print("`y = x0/x1` among 12 columns, 10 of them unrelated, with Gaussian noise at a multiple of the "
          "signal's standard deviation; 1,000 training rows and 10 seeds per level. R² is against the "
          "noiseless target on 2,000 new rows. A term reads an unrelated column when it uses any column "
          "but `x0` and `x1`.\n")
    print("| noise | setting | median terms | terms reading an unrelated column | median R² | median fit (s) |")
    print("| --: | :-- | --: | --: | --: | --: |")
    for level in ("0.1", "0.3", "0.5", "1.0"):
        for s in SETTINGS:
            rs = [r for r in R[s] if r["study"] == "sweep" and r["item"].startswith(f"noise{level}:")]
            junk, terms = sum(r["junk_terms"] for r in rs), sum(r["terms"] for r in rs)
            print(f"| {level} | {LABEL[s]} | {st.median(r['terms'] for r in rs):g} | {junk} of {terms} | "
                  f"{st.median(r['score'] for r in rs):.3f} | {st.median(r['seconds'] for r in rs):.1f} |")
    print()


def real():
    print("## Real datasets\n")
    print("The 12 real datasets of `benchmarks/release_comparison/`, with its five 70/30 splits and seeds: "
          "mean held-out score (R², accuracy for breast_cancer), with the mean number of terms in brackets. "
          "Better or worse means a change of more than 0.005 from the default.\n")
    header("dataset")
    for study, names in (("suite", REAL_SUITE), ("highdim", REAL_HIGHDIM)):
        for name in names:
            cells = [cell(s, study, name) for s in SETTINGS]
            print(f"| {name} | " + " | ".join(f"{score:.3f} ({terms:.1f})" for score, terms in cells) + " |")
    pairs = [(study, n) for study, names in (("suite", REAL_SUITE), ("highdim", REAL_HIGHDIM)) for n in names]
    means = {s: mean([cell(s, study, n)[0] for study, n in pairs]) for s in SETTINGS}
    terms = {s: mean([cell(s, study, n)[1] for study, n in pairs]) for s in SETTINGS}
    print("| **mean** | " + " | ".join(f"**{means[s]:.3f}** ({terms[s]:.1f})" for s in SETTINGS) + " |")
    counts = {}
    for s in SETTINGS:
        diffs = [cell(s, study, n)[0] - cell("default", study, n)[0] for study, n in pairs]
        counts[s] = (sum(d > SAME for d in diffs), sum(abs(d) <= SAME for d in diffs), sum(d < -SAME for d in diffs))
    print("| better / same / worse | — | " + " | ".join("{} / {} / {}".format(*counts[s]) for s in SETTINGS[1:]) + " |")
    keys = [k for k in BY["default"] if k[0] in ("suite", "highdim") and k[1].split(":")[0] in REAL_SUITE + REAL_HIGHDIM]
    print("| fit time (× default) | 1 | " + " | ".join(f"{time_ratio(s, keys):.1f}" for s in SETTINGS[1:]) + " |")
    note = ("\nFit time is the geometric mean, over the 60 fits, of each fit's time over the default's on the "
            "same split. The settings ran in parallel processes, one per setting, on one core each.")
    splits = sorted({(f["study"], f["item"], f["rows"], f["of"]) for f in RESCORED})
    if splits:
        note += (" On " + "; ".join(f"{item} ({rows} of {of} test rows)" for _, item, rows, of in splits) +
                 ", a formula was undefined on the other test rows, and every setting is scored on the rows "
                 "all four predict, as in `benchmarks/release_comparison/`.")
    print(note + "\n")


def synthetic():
    print("## Synthetic datasets\n")
    names = sorted({r["item"].split(":")[0] for r in rows("default", "suite")} - set(REAL_SUITE))
    changed = [n for n in names if any(abs(cell(s, "suite", n)[0] - cell("default", "suite", n)[0]) > SAME
                                       for s in SETTINGS[1:])]
    print(f"The {len(names)} synthetic datasets of the same suite. On {len(names) - len(changed)} of them no "
          "setting moved the mean held-out score by more than 0.005 from the default. The others:\n")
    header("dataset")
    for name in changed:
        print(f"| {name} | " + " | ".join("{:.3f} ({:.1f})".format(*cell(s, "suite", name)) for s in SETTINGS) + " |")
    print()


def paper():
    print("## At the paper's own settings\n")
    print("Each study of the paper rerun with its own script and settings (`run.py` lists them). The paper "
          "column gives the paper's figures, or, per dataset, the archived results its tables quote "
          "(`benchmarks/results*.csv`, `feynman_results.json`).\n")
    archive = {}
    for name in ("results.csv", "results_robustness.csv", "results_real.csv"):
        for r in csv.DictReader((BENCH / name).open()):
            if r["method"] == "beamfeat":
                archive[r["dataset"]] = (float(r["r2"]), int(r["n_features"]))
    a = {s: {r["item"]: r for r in rows(s, "paper_suite")} for s in SETTINGS}
    f = {s: rows(s, "paper_feynman") for s in SETTINGS}
    print("| claim | paper | " + " | ".join(LABEL[s] for s in SETTINGS) + " |")
    print("| :-- | :-- |" + " :-- |" * len(SETTINGS))
    print("| generating columns recovered (Table 4) | 10/10 | " +
          " | ".join(f"{sum(bool(a[s][n]['recovered']) for n in RECOVERY)}/10" for s in SETTINGS) + " |")
    print("| mean R², core suite (Table 4) | 0.9990 | " +
          " | ".join(f"{mean([a[s][n]['score'] for n in CORE]):.4f}" for s in SETTINGS) + " |")
    print("| stress sets with false features (Table 4) | 0 of 5 | " +
          " | ".join(f"{sum((a[s][n]['false_feature_rate'] or 0) > 0 for n in STRESS)} of 5" for s in SETTINGS) + " |")
    print("| Feynman panel solved, exact (§4.3) | 10/12, 8/12 | " +
          " | ".join(f"{sum(bool(r['solved']) for r in f[s])}/12, {sum(bool(r['exact']) for r in f[s])}/12"
                     for s in SETTINGS) + " |")
    print("\nReal datasets of `run_benchmarks.py` (§4.4; one split):\n")
    print("| dataset | paper | " + " | ".join(LABEL[s] for s in SETTINGS) + " |")
    print("| :-- | --: |" + " --: |" * len(SETTINGS))
    for name in REAL_SUITE:
        p = archive[name]
        print(f"| {name} | {p[0]:.3f} ({p[1]}) | " +
              " | ".join(f"{a[s][name]['score']:.3f} ({a[s][name]['terms']})" for s in SETTINGS) + " |")
    print(f"| **mean** | **{mean([archive[n][0] for n in REAL_SUITE]):.3f}** | " +
          " | ".join(f"**{mean([a[s][n]['score'] for n in REAL_SUITE]):.3f}**" for s in SETTINGS) + " |")
    print("\nThe comparison study (Table 5; 9 datasets, 5 splits):\n")
    print("| | paper | " + " | ".join(LABEL[s] for s in SETTINGS) + " |")
    print("| :-- | --: |" + " --: |" * len(SETTINGS))
    c = {s: rows(s, "paper_comparison") for s in SETTINGS}
    print("| mean R² | 0.803 | " + " | ".join(f"{mean([r['score'] for r in c[s]]):.3f}" for s in SETTINGS) + " |")
    print("| worst of 45 | 0.355 | " + " | ".join(f"{min(r['score'] for r in c[s]):.3f}" for s in SETTINGS) + " |")
    print("| fits below zero | 0 | " + " | ".join(f"{sum(r['score'] < 0 for r in c[s])}" for s in SETTINGS) + " |")
    print("| features | 9.2 | " + " | ".join(f"{mean([r['terms'] for r in c[s]]):.1f}" for s in SETTINGS) + " |")
    print("\nThe high-dimensional study (Table 6; the five datasets the repository stores, 5 splits):\n")
    table6 = {"communities_p100": (0.641, 11.6), "superconduct_p81": (0.731, 16.2), "tecator_p100": (0.991, 3.6),
              "eyedata_p200": (0.416, 34.0), "riboflavin_p4088": (0.544, 17.4)}
    print("| dataset | paper | " + " | ".join(LABEL[s] for s in SETTINGS) + " |")
    print("| :-- | --: |" + " --: |" * len(SETTINGS))
    for name, (score, terms) in table6.items():
        mark = "*" if name.startswith("riboflavin") else ""
        print(f"| {name.split('_p')[0]} | {score:.3f}{mark} ({terms}) | " +
              " | ".join("{:.3f} ({:.1f})".format(*cell(s, "paper_highdim", name)) for s in SETTINGS) + " |")
    print("\n\\* 0.340 with the column-order fix of 0.5.0 (docs, Benchmarks, Reading the paper today).\n")


def same_fit(a, b):
    raw = a.get("raw_score", a["score"])
    score = raw == b["score"] or (math.isnan(raw) and math.isnan(b["score"]))
    return score and a["terms"] == b["printed"]


def cross_check():
    print("## Against the release comparison\n")
    archived = {}
    for study in ("suite", "highdim"):
        with gzip.open(BENCH / "release_comparison" / "results" / f"{study}.jsonl.gz", "rt") as handle:
            for line in handle:
                r = json.loads(line)
                if r["version"] == "0.7.0" and r["config"] in ("default", "parsimony_alpha"):
                    archived[(study, r["item"], r["config"])] = r
    for s in ("default", "parsimony_alpha"):
        keys = [(r["study"], r["item"]) for r in R[s] if r["study"] in ("suite", "highdim")]
        same = sum(same_fit(BY[s][k], archived[(k[0], k[1], s)]) for k in keys)
        print(f"- {LABEL[s]}: {same} of {len(keys)} fits give the held-out score and the number of terms "
              "of the release comparison's 0.7.0 run on the same split.")
    print()


if __name__ == "__main__":
    env = R["default"][0]
    print("# Three settings\n")
    print("The default, `parsimony_alpha=0.05`, `input_test=True` and both options together, fitted on "
          "the same data with the same seeds; the documentation's \"Choosing a setting\" page summarises "
          f"these tables. beamfeat {env['beamfeat']}, numpy {env['numpy']}, scikit-learn {env['scikit-learn']}; "
          "`PROVENANCE.md` says how to rerun.\n")
    sweep()
    real()
    synthetic()
    paper()
    cross_check()
