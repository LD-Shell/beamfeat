"""Tables of the release comparison, from results/, as Markdown on stdout.

    python report.py > TABLES.md
"""
import gzip
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
RELEASES = [("0.4.2", "default"), ("0.5.0", "default"), ("0.6.0", "default"), ("0.6.1", "default")]
OPTIONS = [("0.6.0", "no_fitted_ops"), ("0.6.0", "magnitude_slots"), ("0.6.0", "input_test"),
           ("0.6.0", "conditional_retest"), ("0.6.0", "bh"), ("0.6.1", "parsimony_alpha"),
           ("0.6.1", "alpha_input_test")]
GROUPS = ["null", "planted_snr1", "planted_snr3", "proxy_snr1", "proxy_snr3"]
SYNTHETIC = {"product", "ratio", "three_way", "log_linear", "sqrt_ratio", "quadratic", "inverse_square",
             "physics_kinetic", "sparse_10col", "purely_linear", "friedman1", "heavy_tail_t2",
             "distractors_noise05", "distractors_noise25", "distractors_noise50", "threshold_step",
             "small_n120", "small_n240"}
REAL = ["diabetes", "breast_cancer", "mpg", "diamonds", "penguins", "tips", "california_housing"]
# Splits where some model left held-out rows without a prediction, re-scored
# on the rows every setting predicts (rescore.py).
RESCORE = {}
if (R / "rescore.json").exists():
    RESCORE = {(r["study"], r["item"], r["version"], r["config"]): r for r in json.loads((R / "rescore.json").read_text())}


def load(study):
    rows = []
    if (R / f"{study}.jsonl.gz").exists():  # the archived run
        with gzip.open(R / f"{study}.jsonl.gz", "rt") as handle:
            rows += [json.loads(line) for line in handle]
    for path in sorted(R.glob(f"{study}_*.jsonl")):  # a new run's job files
        rows += [json.loads(line) for line in path.open()]
    # Keep the last record of an (item, version, config), as a resumed job may repeat one.
    keep = {}
    for r in rows:
        keep[(r["item"], r["version"], r["config"])] = r
    for (item, version, config), r in keep.items():
        fix = RESCORE.get((study, item, version, config))
        if fix is not None and "error" not in r:
            r["score"], r["rescored"] = fix["score"], True
    # 0.6.0 input_test passed 7 GB on California; the 0.6.1 run of the same
    # test, with the memory fix that leaves every fit unchanged, stands in.
    for (item, version, config), r in list(keep.items()):
        if (version, config) == ("0.6.0", "input_test") and "error" in r:
            stand_in = keep.get((item, "0.6.1", "input_test"))
            if stand_in is not None and "error" not in stand_in:
                keep[(item, version, config)] = {**stand_in, "version": "0.6.0", "via_fix": True}
    return list(keep.values())


def perf_rows(name):
    path = R / name
    return [json.loads(line) for line in path.open()] if path.exists() else []


def perf_ratio(a, b, metric):
    """Geometric mean over items of median(b) / median(a); the 0.4.2 pass for comparisons with 0.4.2."""
    rows = perf_rows("perf_042.jsonl" if "0.4.2" in (a[0], b[0]) else "perf.jsonl")
    ratios = []
    for item in dict.fromkeys(r["item"] for r in rows):
        sa = [r[metric] for r in rows if "error" not in r and r["item"] == item and (r["version"], r["config"]) == a]
        sb = [r[metric] for r in rows if "error" not in r and r["item"] == item and (r["version"], r["config"]) == b]
        if sa and sb:
            ratios.append(max(np.median(sb), 1e-3) / max(np.median(sa), 1e-3) if metric == "seconds"
                          else max(np.median(sb), 1) / max(np.median(sa), 1))
    return math.exp(np.mean(np.log(ratios))) if ratios else float("nan")


def same_score(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def ms(values):
    a = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    if a.size == 0:
        return (float("nan"), float("nan"), 0)
    return (float(a.mean()), float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else float("nan"), a.size)


def f(x, d=3, se=True):
    m, s, n = x
    if not math.isfinite(m):
        return "—"
    return f"{m:.{d}f} ± {s:.{d}f}" if se and math.isfinite(s) else f"{m:.{d}f}"


def sel(rows, version, config, pred=lambda r: True):
    return [r for r in rows if r["version"] == version and r["config"] == config and pred(r)
            and "error" not in r and "skipped" not in r]


def paired(rows, a, b, key, pred=lambda r: True):
    """Mean of key(b) - key(a) over the items both have."""
    ia = {r["item"]: r for r in sel(rows, *a, pred)}
    ib = {r["item"]: r for r in sel(rows, *b, pred)}
    common = sorted(set(ia) & set(ib))
    return ms([key(ib[i]) - key(ia[i]) for i in common if key(ib[i]) is not None and key(ia[i]) is not None])


def errors(rows):
    bad = [r for r in rows if "error" in r]
    return bad


def identical(rows, a, b):
    ia = {r["item"]: r for r in sel(rows, *a)}
    ib = {r["item"]: r for r in sel(rows, *b)}
    common = set(ia) & set(ib)
    same = sum(ia[i].get("formulas") == ib[i].get("formulas") and same_score(ia[i].get("score"), ib[i].get("score"))
               for i in common)
    return same, len(common)


def label(v, c):
    return f"{v} {c}" if c != "default" else f"**{v}**"


# --------------------------------------------------------------------------- #


def designs():
    rows = load("designs")
    g = lambda r: r["item"].split("_n")[0]  # noqa: E731
    out = ["## 1. Calibration designs (`parsimony_holdout` designs; q = 0.1; 20 cells x 50 seeds)", "",
           "FDR is the mean false-discovery proportion of the printed terms: marginal = built only from "
           "columns outside a*b/c (the guarantee's sense); conditional = uses any column outside it (riders). "
           "Power = a term equivalent to a*b/c printed. R² on 2,000 held-out rows.", "",
           "| design | setting | terms | FDR marginal | FDR conditional | power | held-out R² |",
           "| :-- | :-- | --: | --: | --: | --: | --: |"]
    for group in GROUPS:
        for v, c in RELEASES + OPTIONS:
            s = sel(rows, v, c, lambda r: g(r) == group)
            if not s:
                continue
            out.append(f"| {group} | {label(v, c)} | {ms([r['printed'] for r in s])[0]:.1f} | "
                       f"{f(ms([r['marginal_false'] / max(r['printed'], 1) for r in s]))} | "
                       f"{f(ms([r['conditional_false'] / max(r['printed'], 1) for r in s]))} | "
                       f"{f(ms([r['true_term'] for r in s])) if group != 'null' else '—'} | "
                       f"{f(ms([r['score'] for r in s]))} |")
    out.append("")
    nulls = []
    for v, c in RELEASES + OPTIONS:
        s = sel(rows, v, c, lambda r: g(r) == "null")
        if s:
            nulls.append(f"{label(v, c)}: {sum(r['printed'] > 0 for r in s)}/{len(s)}")
    out += ["Global null, fits printing any term: " + "; ".join(nulls), ""]
    return "\n".join(out), rows


def stress():
    rows = load("stress")
    out = ["## 2. Search stress laws (500 rows, 10% noise, p = 20 and 100, 10 seeds per cell)", "",
           "Exact = the printed equation is the law (Feynman-panel criterion). R² against the noiseless law "
           "on 2,000 held-out rows.", "",
           "| law, inputs | " + " | ".join(label(v, c) for v, c in RELEASES + OPTIONS) + " |",
           "| :-- |" + " --: |" * len(RELEASES + OPTIONS)]
    cells = sorted({r["item"].rsplit("_p", 1)[0] for r in rows}, key=lambda s: (s.split("_")[-1], s))
    for cell in cells:
        vals = []
        for v, c in RELEASES + OPTIONS:
            s = sel(rows, v, c, lambda r: r["item"].rsplit("_p", 1)[0] == cell)
            vals.append(f"{ms([r['exact'] for r in s])[0]:.2f}" if s else "—")
        out.append(f"| {cell} | " + " | ".join(vals) + " |")
    out.append("")
    out += ["| overall | setting | exact | distractor-only terms | held-out R² | terms |", "| :-- | :-- | --: | --: | --: | --: |"]
    for v, c in RELEASES + OPTIONS:
        s = sel(rows, v, c)
        if s:
            out.append(f"| all cells | {label(v, c)} | {f(ms([r['exact'] for r in s]))} | "
                       f"{ms([r['distractor_terms'] for r in s])[0]:.3f} | {f(ms([r['score'] for r in s]))} | "
                       f"{ms([r['printed'] for r in s])[0]:.1f} |")
    out.append("")
    return "\n".join(out), rows


def feynman():
    rows = load("feynman")
    out = ["## 3. Feynman panel (12 laws x 5 data draws, 0.1% noise; depth 3, beam 40)", "",
           "| setting | solved (R² > 0.999) | exact form | mean terms |", "| :-- | --: | --: | --: |"]
    for v, c in RELEASES + OPTIONS:
        s = sel(rows, v, c)
        if s:
            out.append(f"| {label(v, c)} | {sum(r['solved'] for r in s)}/{len(s)} | "
                       f"{sum(1 for r in s if r['exact'])}/{len(s)} | {ms([r['printed'] for r in s])[0]:.2f} |")
    out.append("")
    laws = sorted({r["item"].split(":")[0] for r in rows})
    out += ["Per law (solved / exact out of 5):", "",
            "| law | " + " | ".join(label(v, c) for v, c in RELEASES + OPTIONS) + " |",
            "| :-- |" + " --: |" * len(RELEASES + OPTIONS)]
    for law in laws:
        vals = []
        for v, c in RELEASES + OPTIONS:
            s = sel(rows, v, c, lambda r: r["item"].split(":")[0] == law)
            vals.append(f"{sum(r['solved'] for r in s)}/{sum(1 for r in s if r['exact'])}" if s else "—")
        out.append(f"| {law} | " + " | ".join(vals) + " |")
    out.append("")
    return "\n".join(out), rows


def suite():
    rows = load("suite")
    name = lambda r: r["item"].split(":")[0]  # noqa: E731
    out = ["## 4. Main suite (`run_benchmarks.py` datasets; 5 splits, 70/30)", "",
           "Score = held-out R² (accuracy for breast_cancer). Mean ± SE over splits; terms in brackets.", "",
           "| dataset | " + " | ".join(label(v, c) for v, c in RELEASES + OPTIONS) + " |",
           "| :-- |" + " --: |" * len(RELEASES + OPTIONS)]
    datasets = [d for d in sorted(SYNTHETIC) if any(name(r) == d for r in rows)] + [d for d in REAL if any(name(r) == d for r in rows)]
    for d in datasets:
        vals = []
        for v, c in RELEASES + OPTIONS:
            s = sel(rows, v, c, lambda r: name(r) == d)
            mark = "†" if any(r.get("via_fix") for r in s) else ""
            vals.append(f"{ms([r['score'] for r in s])[0]:.3f} ({ms([r['printed'] for r in s])[0]:.1f}){mark}" if s else "n/a")
        out.append(f"| {d} | " + " | ".join(vals) + " |")
    out.append("")
    out += ["n/a: breast_cancer is classification, which `magnitude_slots` and the conditional re-test do not support. "
            "† 0.6.0 `input_test` passed 7 GB on every California split and was stopped; the value shown is the same "
            "test run with the 0.6.1 memory fix (CHANGELOG, 0.6.1, Fixed), which leaves every fit unchanged and "
            "finishes in about 80 s using 0.55 GB. The 0.6.1 `alpha_input_test` California cell also runs with the fix.", ""]
    resc = sorted({(r["item"], r["rows_dropped"], r["rows"]) for r in RESCORE.values() if r["study"] == "suite"})
    if resc:
        out += ["Re-scored on the rows every setting predicts: " + "; ".join(f"{i} ({k} of {n} rows dropped)" for i, k, n in resc), ""]
    out += ["Known-formula datasets (tokens known): recovery = a printed term uses every true column; "
            "false-feature rate = share of printed terms using no true column.", "",
            "| setting | recovered | false-feature rate |", "| :-- | --: | --: |"]
    for v, c in RELEASES + OPTIONS:
        s = sel(rows, v, c, lambda r: r.get("recovered") is not None)
        if s:
            out.append(f"| {label(v, c)} | {f(ms([float(r['recovered']) for r in s]))} | "
                       f"{f(ms([r['false_feature_rate'] for r in s]))} |")
    out.append("")
    return "\n".join(out), rows


def highdim():
    rows = load("highdim")
    name = lambda r: r["item"].split(":")[0]  # noqa: E731
    out = ["## 5. High-dimensional real data (5 splits, 70/30)", "",
           "| dataset | " + " | ".join(label(v, c) for v, c in RELEASES + OPTIONS) + " |",
           "| :-- |" + " --: |" * len(RELEASES + OPTIONS)]
    for d in ("communities", "superconduct", "tecator", "eyedata", "riboflavin"):
        vals = []
        for v, c in RELEASES + OPTIONS:
            s = sel(rows, v, c, lambda r: name(r) == d)
            vals.append(f"{f(ms([r['score'] for r in s]))} ({ms([r['printed'] for r in s])[0]:.1f})" if s else "n/a")
        out.append(f"| {d} | " + " | ".join(vals) + " |")
    out.append("")
    resc = sorted({(r["item"], r["rows_dropped"], r["rows"]) for r in RESCORE.values() if r["study"] == "highdim"})
    if resc:
        out += ["Re-scored on the rows every setting predicts: " + "; ".join(f"{i} ({k} of {n} rows dropped)" for i, k, n in resc), ""]
    if RESCORE:
        settings = [(v, c) for v, c in RELEASES + OPTIONS + [("0.6.1", "input_test")]
                    if any(r["version"] == v and r["config"] == c for r in RESCORE.values())]
        out += ["### Held-out rows left without a prediction (a fitted expression undefined there)", "",
                "| split | " + " | ".join(label(v, c) for v, c in settings) + " |", "| :-- |" + " --: |" * len(settings)]
        for item in sorted({r["item"] for r in RESCORE.values()}):
            vals = []
            for v, c in settings:
                r = next((r for r in RESCORE.values() if r["item"] == item and r["version"] == v and r["config"] == c), None)
                vals.append(str(r["own_undefined"]) if r else "—")
            out.append(f"| {item} | " + " | ".join(vals) + " |")
        out.append("")
    return "\n".join(out), rows


def junk():
    rows = load("junk")
    out = ["## 6. Junk columns inside terms (`y = x0*x1` + 25% noise, 10 junk columns, 20 seeds)", "",
           "| junk | setting | equations with a junk column | junk terms | held-out R² | terms |",
           "| :-- | :-- | --: | --: | --: | --: |"]
    for kind in ("independent", "correlated"):
        for v, c in RELEASES + OPTIONS:
            s = sel(rows, v, c, lambda r: r["item"].startswith(kind))
            if s:
                out.append(f"| {kind} | {label(v, c)} | {sum(r['junk_equation'] for r in s)}/{len(s)} | "
                           f"{sum(r['junk_terms'] for r in s)} | {f(ms([r['score'] for r in s]), 4)} | "
                           f"{ms([r['printed'] for r in s])[0]:.1f} |")
    out.append("")
    return "\n".join(out), rows


def knockoff():
    rows = []
    for path in sorted(R.glob("knockoff_*.json")):
        rows += json.loads(path.read_text())
    if not rows:
        return "", rows
    out = ["## 7. Knockoff selector alone (fixed-X, nominal FDR 0.1, 50 trials; `knockoff_s_method.py` designs)", "",
           "| design | offset | 0.4.2 FDR / power | 0.5.0 FDR / power | 0.6.0 FDR / power | 0.6.1 equicorrelated | 0.6.1 `s_method=\"ci\"` |",
           "| :-- | --: | --: | --: | --: | --: | --: |"]
    for offset in (1, 0):
        for design in ("block rho=0.5", "block rho=0.7", "block rho=0.9", "factor", "constructed"):
            vals = []
            for v, m in (("0.4.2", "equicorrelated"), ("0.5.0", "equicorrelated"), ("0.6.0", "equicorrelated"),
                         ("0.6.1", "equicorrelated"), ("0.6.1", "ci")):
                s = [r for r in rows if r["version"] == v and r["s_method"] == m and r["design"] == design and r["offset"] == offset]
                vals.append(f"{np.mean([r['fdp'] for r in s]):.3f} / {np.mean([r['power'] for r in s]):.3f}" if s else "—")
            out.append(f"| {design} | {offset} | " + " | ".join(vals) + " |")
    out.append("")
    return "\n".join(out), rows


def releases_summary(d):
    """The four releases at their defaults, one column each."""
    g = lambda r: r["item"].split("_n")[0]  # noqa: E731
    name = lambda r: r["item"].split(":")[0]  # noqa: E731
    highdim = ["communities", "superconduct", "tecator", "eyedata", "riboflavin"]

    def per_dataset_mean(rows, v, names):
        means = [ms([r["score"] for r in sel(rows, v, "default", lambda r, n=n: name(r) == n)])[0] for n in names]
        return float(np.mean(means))

    def cell(v, metric):
        des, st, fe, su, hd, ju = (d[k] for k in ("designs", "stress", "feynman", "suite", "highdim", "junk"))
        snr = lambda k: (lambda r: g(r).endswith(f"snr{k}"))  # noqa: E731
        if metric == "null":
            s = sel(des, v, "default", lambda r: g(r) == "null")
            return f"{sum(r['printed'] > 0 for r in s)}/{len(s)}"
        if metric.startswith("terms") or metric.startswith("cfdr") or metric.startswith("power") or metric.startswith("r2"):
            key, k = metric[:-1], metric[-1]
            s = sel(des, v, "default", snr(k))
            fn = {"terms": lambda r: r["printed"], "cfdr": lambda r: r["conditional_false"] / max(r["printed"], 1),
                  "power": lambda r: r["true_term"], "r2": lambda r: r["score"]}[key]
            return f"{ms([fn(r) for r in s])[0]:.{1 if key == 'terms' else 3}f}"
        if metric == "mfdr":
            s = sel(des, v, "default", lambda r: g(r) != "null")
            return f"{max(r['marginal_false'] / max(r['printed'], 1) for r in s):.3f}"
        if metric == "stress":
            return f"{ms([float(r['exact']) for r in sel(st, v, 'default')])[0]:.3f}"
        if metric == "feynman":
            s = sel(fe, v, "default")
            return f"{sum(r['solved'] for r in s)} / {sum(1 for r in s if r['exact'])}"
        if metric == "synthetic":
            return f"{per_dataset_mean(su, v, sorted(SYNTHETIC)):.4f}"
        if metric == "real":
            return f"{per_dataset_mean(su, v, REAL):.4f}"
        if metric == "recovered":
            s = sel(su, v, "default", lambda r: r.get("recovered") is not None)
            return f"{ms([float(r['recovered']) for r in s])[0]:.3f} / {ms([r['false_feature_rate'] for r in s])[0]:.3f}"
        if metric == "highdim":
            return f"{per_dataset_mean(hd, v, highdim):.3f}"
        if metric == "junk":
            s = sel(ju, v, "default")
            return f"{sum(r['junk_equation'] for r in s)}/{len(s)}"
        if metric in ("seconds", "fit_mb"):
            return "1.00" if v == "0.5.0" else f"{perf_ratio(('0.5.0', 'default'), (v, 'default'), metric):.2f}"
        raise ValueError(metric)

    rows = [("Calibration designs, snr 1: terms printed", "terms1"), ("  conditional FDR (riders)", "cfdr1"),
            ("  power (generating term printed)", "power1"), ("  held-out R²", "r21"),
            ("Calibration designs, snr 3: terms printed", "terms3"), ("  conditional FDR", "cfdr3"),
            ("  power", "power3"), ("  held-out R²", "r23"),
            ("Largest marginal FDR in any design fit", "mfdr"), ("Global null: fits printing any term", "null"),
            ("Search stress laws: exact", "stress"), ("Feynman panel: solved / exact form (of 60)", "feynman"),
            ("Suite, 18 synthetic datasets: mean score", "synthetic"), ("Suite, 7 real datasets: mean score", "real"),
            ("Known formulas: recovered / false-feature rate", "recovered"),
            ("High-dimensional, 5 datasets: mean R²", "highdim"), ("Junk-column equations (of 40)", "junk"),
            ("Fit time, x 0.5.0 (geometric mean, 8 items)", "seconds"), ("Peak fit memory, x 0.5.0", "fit_mb")]
    out = ["## 0. The releases at their defaults", "",
           "Means over the items of each study (sections 1-9 below). Designs pool the planted and proxy designs, "
           "200 fits per signal-to-noise ratio, q = 0.1.", "",
           "| measure | " + " | ".join(v for v, _ in RELEASES) + " |", "| :-- |" + " --: |" * len(RELEASES)]
    for title, metric in rows:
        out.append(f"| {title} | " + " | ".join(cell(v, metric) for v, _ in RELEASES) + " |")
    out.append("")
    return "\n".join(out)


def option_summary(study_rows):
    """Paired effects of each release and option on the same items."""
    out = ["## 8. Paired effects (same items; mean difference ± SE)", "",
           "Releases are compared with the previous release's default; options with their own release's default.", ""]
    comparisons = [(("0.4.2", "default"), ("0.5.0", "default")), (("0.5.0", "default"), ("0.6.0", "default")),
                   (("0.6.0", "default"), ("0.6.1", "default"))] + \
        [((v, "default"), (v, c)) for v, c in OPTIONS]
    d = study_rows
    g = lambda r: r["item"].split("_n")[0]  # noqa: E731
    out += ["| comparison | designs Δ cond. FDR (snr 1) | designs Δ cond. FDR (snr 3) | designs Δ R² (snr 1) | "
            "stress Δ exact | feynman Δ exact | suite Δ score (synthetic) | suite Δ score (real) | highdim Δ score | junk Δ junk eqs |",
            "| :-- |" + " --: |" * 9]
    for a, b in comparisons:
        cf = lambda r: r["conditional_false"] / max(r["printed"], 1)  # noqa: E731
        cells = [
            f(paired(d["designs"], a, b, cf, lambda r: "snr1" in g(r))),
            f(paired(d["designs"], a, b, cf, lambda r: "snr3" in g(r))),
            f(paired(d["designs"], a, b, lambda r: r["score"], lambda r: "snr1" in g(r)), 4),
            f(paired(d["stress"], a, b, lambda r: float(r["exact"]))),
            f(paired(d["feynman"], a, b, lambda r: float(bool(r["exact"])))),
            f(paired(d["suite"], a, b, lambda r: r["score"], lambda r: r["item"].split(":")[0] in SYNTHETIC), 4),
            f(paired(d["suite"], a, b, lambda r: r["score"], lambda r: r["item"].split(":")[0] in REAL), 4),
            f(paired(d["highdim"], a, b, lambda r: r["score"]), 4),
            f(paired(d["junk"], a, b, lambda r: float(r["junk_equation"]))),
        ]
        out.append(f"| {b[0]} {b[1]} vs {a[0]} {a[1]} | " + " | ".join(cells) + " |")
    out.append("")
    same = []
    for study, rows in d.items():
        s, n = identical(rows, ("0.6.0", "default"), ("0.6.1", "default"))
        same.append(f"{study} {s}/{n}")
    out += ["0.6.1 default identical to 0.6.0 default (formulas and score): " + ", ".join(same), ""]
    same = []
    for study, rows in d.items():
        s, n = identical(rows, ("0.4.2", "default"), ("0.5.0", "default"))
        same.append(f"{study} {s}/{n}")
    out += ["0.5.0 default identical to 0.4.2 default: " + ", ".join(same), ""]
    same = []
    for study, rows in d.items():
        s, n = identical(rows, ("0.5.0", "default"), ("0.6.0", "no_fitted_ops"))
        same.append(f"{study} {s}/{n}")
    out += ["0.6.0 with `fitted_ops=()` identical to 0.5.0 default: " + ", ".join(same), ""]
    errs = [r for rows in d.values() for r in errors(rows)]
    out += [f"Fits that raised: {len(errs)}" + ("" if not errs else " — " + "; ".join(
        sorted({f"{r['version']} {r['config']} {r['study']}: {r['error'][:80]}" for r in errs})[:8])), ""]
    return "\n".join(out)


def scorecard(d):
    """Each option against its own release's default, dataset by dataset."""
    def per_dataset(rows, a, b, names):
        deltas = {}
        for name in names:
            pred = lambda r, n=name: r["item"].split(":")[0] == n  # noqa: E731
            m, _, k = paired(rows, a, b, lambda r: r["score"], pred)
            if k:
                deltas[name] = m
        return deltas

    out = ["## 10. Option scorecard (each option against its own release's default)", "",
           "Datasets: the 7 real suite datasets and the 5 high-dimensional ones (12), and the 18 synthetic suite "
           "datasets; better / same / worse = mean held-out score over the 5 splits differs by more than 0.005. "
           "Time and memory: geometric-mean ratio over the perf items both settings finished.", "",
           "| option | real: better / same / worse | real: median Δ | real: worst Δ | synthetic: better / same / worse | "
           "terms (real, mean) | designs Δ cond. FDR, snr 1 | time × | memory × |",
           "| :-- | --: | --: | :-- | --: | --: | --: | --: | --: |"]
    for v, c in [(("0.4.2", "default"), ("0.5.0", "default")), (("0.5.0", "default"), ("0.6.0", "default"))] + \
            [((v, "default"), (v, c)) for v, c in OPTIONS]:
        a, b = v, c
        real = {**per_dataset(d["suite"], a, b, REAL), **per_dataset(d["highdim"], a, b, ["communities", "superconduct", "tecator", "eyedata", "riboflavin"])}
        synth = per_dataset(d["suite"], a, b, sorted(SYNTHETIC))
        tally = lambda ds: f"{sum(x > 0.005 for x in ds.values())} / {sum(abs(x) <= 0.005 for x in ds.values())} / {sum(x < -0.005 for x in ds.values())}"  # noqa: E731
        worst = min(real.items(), key=lambda kv: kv[1]) if real else ("—", float("nan"))
        terms = lambda cfg: ms([r["printed"] for st in ("suite", "highdim") for r in sel(d[st], *cfg)  # noqa: E731
                                if r["item"].split(":")[0] in set(REAL) | {"communities", "superconduct", "tecator", "eyedata", "riboflavin"}])[0]
        g = lambda r: r["item"].split("_n")[0]  # noqa: E731
        cf = paired(d["designs"], a, b, lambda r: r["conditional_false"] / max(r["printed"], 1), lambda r: "snr1" in g(r))
        out.append(f"| {label(*b)} vs {label(*a)} | {tally(real)} | {np.median(list(real.values())):+.4f} | "
                   f"{worst[0]} {worst[1]:+.3f} | {tally(synth)} | {terms(a):.1f} → {terms(b):.1f} | {f(cf)} | "
                   f"{perf_ratio(a, b, 'seconds'):.2f} | {perf_ratio(a, b, 'fit_mb'):.2f} |")
    out.append("")
    return "\n".join(out)


def perf():
    path = R / "perf.jsonl"
    if not path.exists():
        return ""
    every = [json.loads(line) for line in path.open()]
    stopped = {(r["item"], r["version"], r["config"]) for r in every if "error" in r}
    rows = [r for r in every if "error" not in r]
    items = list(dict.fromkeys(r["item"] for r in every))
    order = RELEASES + OPTIONS + [("0.6.1", "input_test")]
    present = {(r["version"], r["config"]) for r in every}
    settings = [s for s in order if s in present]
    out = ["## 9. Speed and memory (each fit in a fresh process, one core, settings interleaved; median of 3)", "",
           "Seconds of wall time for the fit; memory = peak RSS during the fit minus RSS just before it (MB). "
           "\"> 3 GB\" = stopped by the 3 GB guard (0.6.0 `input_test` on California, before the fix).", ""]
    for metric, title, fmt in (("seconds", "Fit time (s)", "{:.2f}"), ("fit_mb", "Peak memory added by the fit (MB)", "{:.0f}")):
        out += [f"### {title}", "", "| item | " + " | ".join(label(v, c) for v, c in settings) + " |",
                "| :-- |" + " --: |" * len(settings)]
        for item in items:
            vals = []
            for v, c in settings:
                s = [r[metric] for r in rows if r["item"] == item and r["version"] == v and r["config"] == c]
                vals.append(fmt.format(np.median(s)) if s else ("> 3 GB" if (item, v, c) in stopped else "—"))
            out.append(f"| {item} | " + " | ".join(vals) + " |")
        out.append("")
    # Geometric-mean ratio to 0.5.0 default over items.
    out += ["### Ratio to 0.5.0 default (geometric mean over the items each setting finished)", "",
            "| setting | time × | memory × | items |", "| :-- | --: | --: | --: |"]
    for v, c in settings:
        rt, rm = [], []
        for item in items:
            base_t = [r["seconds"] for r in rows if r["item"] == item and r["version"] == "0.5.0" and r["config"] == "default"]
            base_m = [r["fit_mb"] for r in rows if r["item"] == item and r["version"] == "0.5.0" and r["config"] == "default"]
            s_t = [r["seconds"] for r in rows if r["item"] == item and r["version"] == v and r["config"] == c]
            s_m = [r["fit_mb"] for r in rows if r["item"] == item and r["version"] == v and r["config"] == c]
            if base_t and s_t:
                rt.append(np.median(s_t) / np.median(base_t))
                rm.append(max(np.median(s_m), 1) / max(np.median(base_m), 1))
        if rt:
            out.append(f"| {label(v, c)} | {math.exp(np.mean(np.log(rt))):.2f} | {math.exp(np.mean(np.log(rm))):.2f} | {len(rt)}/{len(items)} |")
    out.append("")
    second = perf_rows("perf_042.jsonl")
    if second:
        out += ["### 0.4.2 (a second pass, interleaved with 0.5.0 as a control)", "",
                "0.4.2 was added after the main pass and timed in a pass of its own, alternating with 0.5.0 by the same "
                "protocol. Its ratios are taken against the 0.5.0 of that pass; the last column checks that pass "
                "against the main one.", "",
                "| item | 0.4.2 s | 0.5.0 s | 0.4.2 MB | 0.5.0 MB | 0.5.0 s, main pass |", "| :-- | --: | --: | --: | --: | --: |"]
        for item in dict.fromkeys(r["item"] for r in second):
            med = lambda src, v, m: np.median([r[m] for r in src if r["item"] == item and r["version"] == v  # noqa: E731
                                               and r["config"] == "default" and "error" not in r])
            out.append(f"| {item} | {med(second, '0.4.2', 'seconds'):.2f} | {med(second, '0.5.0', 'seconds'):.2f} | "
                       f"{med(second, '0.4.2', 'fit_mb'):.0f} | {med(second, '0.5.0', 'fit_mb'):.0f} | "
                       f"{med(rows, '0.5.0', 'seconds'):.2f} |")
        out += ["", f"0.4.2 against 0.5.0: time x{perf_ratio(('0.5.0', 'default'), ('0.4.2', 'default'), 'seconds'):.2f}, "
                f"memory x{perf_ratio(('0.5.0', 'default'), ('0.4.2', 'default'), 'fit_mb'):.2f}.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    parts, data = [], {}
    for name, fn in (("designs", designs), ("stress", stress), ("feynman", feynman), ("suite", suite),
                     ("highdim", highdim), ("junk", junk)):
        text, rows = fn()
        parts.append(text)
        data[name] = rows
    parts.insert(0, releases_summary(data))
    parts.append(knockoff()[0])
    parts.append(option_summary(data))
    parts.append(perf())
    parts.append(scorecard(data))
    print("\n".join(parts))
