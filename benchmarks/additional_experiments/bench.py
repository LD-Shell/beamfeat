"""Independent benchmark: beamfeat vs autofeat vs featuretools-DFS vs OpenFE
vs raw-feature baselines (Ridge, RandomForest, LightGBM).

Protocol (following Horn et al. 2019 / standard tabular practice):
- Fixed 75/25 train/test splits, N_SPLITS seeds, identical across methods.
- The four feature-construction tools feed the SAME downstream model
  (RidgeCV on standardized features) so the comparison isolates the
  engineered features, not the estimator.  beamfeat appears twice: as
  `beamfeat`, the estimator as shipped, whose internal ridge chooses its
  penalty by cross-validation (alpha="auto", the default since 0.3.0); and
  as `beamfeat_ridge`, the transformer feeding the shared RidgeCV, which is
  the strict like-for-like row.
- Non-finite engineered cells are filled with the training-column median, on
  the training and the test rows alike.
- The three `*_raw` entries are reference points on unengineered features,
  not part of that controlled comparison.
- Metrics: out-of-sample R^2 (mean +/- std over splits), wall-clock fit
  time, number of constructed features, and (synthetic only) recovery of
  the generating formula.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Worker counts for the compared tools; the shipped studies never enter the
# code paths that use these. Override via the environment.
N_JOBS = int(os.environ.get("BENCH_N_JOBS", "1"))
FT_N_JOBS = int(os.environ.get("BENCH_FT_N_JOBS", "1"))

warnings.filterwarnings("ignore")

DATA = pathlib.Path(__file__).resolve().parent / "data"

N_SPLITS = 5
ALPHAS = np.logspace(-4, 4, 25)

# ---------------------------------------------------------------- datasets
def get_datasets(which):
    rng = np.random.default_rng(0)
    ds = {}
    if which in ("real", "all"):
        # These three CSVs live in the independent study's data directory, one
        # level up; the shipped studies use the "highdim" group and never touch
        # this branch.
        ind = pathlib.Path(__file__).resolve().parent.parent / "independent" / "data"
        for name, fname in (("concrete", "data_concrete.csv"),
                            ("wine_red", "data_winequality_red.csv")):
            f = ind / fname
            if not f.exists():
                print(f"  skipping {name}: {f} not found")
                continue
            t = pd.read_csv(f)
            ds[name] = (t.iloc[:, :-1].to_numpy(float), t.iloc[:, -1].to_numpy(float), None)
        f = ind / "data_housing_boston.csv"
        if f.exists():
            b = pd.read_csv(f)
            yb = b["medv"].to_numpy(float)
            Xb = b.drop(columns=["medv"]).to_numpy(float)
            ds["boston"] = (Xb, yb, None)
        else:
            print(f"  skipping boston: {f} not found")
        d = load_diabetes()
        ds["diabetes"] = (d.data, d.target, None)
    if which in ("synthetic", "all"):
        n = 500
        X = rng.uniform(1, 6, (n, 4))
        y = X[:,0]*X[:,1]/X[:,2]
        ds["three_way:a*b/c"] = (X, y + rng.normal(0, .02*np.std(y), n), ("x0","x1","x2"))
        X = rng.uniform(1, 6, (n, 4))
        y = 0.5*X[:,0]*X[:,1]**2
        ds["kinetic:m*v^2/2"] = (X, y + rng.normal(0, .02*np.std(y), n), ("x0","x1"))
        X = rng.uniform(1, 6, (n, 10))
        y = X[:,0]*X[:,1]
        ds["sparse10:a*b"] = (X, y + rng.normal(0, .05*np.std(y), n), ("x0","x1"))
        X = rng.uniform(1, 6, (n, 4))
        y = 3*X[:,0]-2*X[:,1]+X[:,2]
        ds["linear_ctrl"] = (X, y + rng.normal(0, .02*np.std(y), n), ())
        X = rng.uniform(0, 1, (800, 10))
        y = 10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-.5)**2 + 10*X[:,3] + 5*X[:,4] + rng.normal(0,1,800)
        ds["friedman1"] = (X, y, None)
    if which in ("highdim", "all"):
        # Five real datasets, p = 81..4088. Provenance and checksums: data/CHECKSUMS.md5,
        # PROVENANCE.md. Superconductivity is capped at 5,000 rows (seeded subsample),
        # following the diamonds cap of the main paper; the full 21,263-row file ships
        # unmodified for full-n runs of the scalable methods.
        SUPERCONDUCT_CAP = 5000
        cc = pd.read_csv(DATA / "communities_crime_numeric.csv")
        ds["communities_p100"] = (cc.iloc[:, :-1].to_numpy(float), cc.iloc[:, -1].to_numpy(float), None)
        sc = pd.read_csv(DATA / "superconductivity.csv").sample(SUPERCONDUCT_CAP, random_state=0)
        ds["superconduct_p81"] = (sc.drop(columns="critical_temp").to_numpy(float),
                                  sc["critical_temp"].to_numpy(float), None)
        tc = pd.read_csv(DATA / "tecator.csv")
        chan = [c for c in tc.columns
                if c.startswith("_") or c.startswith("absorbance")]
        assert len(chan) == 100, (
            f"tecator.csv has {len(chan)} channel columns, expected 100; "
            "re-run fetch_data.py to restore the canonical header")
        ds["tecator_p100"] = (tc[chan].to_numpy(float), tc["fat"].to_numpy(float), None)
        eyd = pd.read_csv(DATA / "eyedata.csv")
        ds["eyedata_p200"] = (eyd.drop(columns="trim32").to_numpy(float),
                              eyd["trim32"].to_numpy(float), None)
        rb = pd.read_csv(DATA / "riboflavin.csv")
        ds["riboflavin_p4088"] = (rb.iloc[:, 1:].to_numpy(float), rb.iloc[:, 0].to_numpy(float), None)
        # ---- optional larger sets: present only after fetch_data.py --all on an
        # unrestricted machine. Registered when the file exists, skipped silently
        # otherwise, so the harness runs on the core five out of the box.
        # Row caps (seeded subsamples) follow the diamonds-cap precedent; override
        # with FULL_N=1 for full-n runs of the scalable methods.
        full_n = os.environ.get("FULL_N") == "1"
        def _cap(df, cap):
            return df if (full_n or len(df) <= cap) else df.sample(cap, random_state=0)
        if (DATA / "ct_slices.csv").exists():          # UCI slice localization, 53,500 x 384
            ct = _cap(pd.read_csv(DATA / "ct_slices.csv"), 10_000)
            ds["ct_slices_p384"] = (ct.iloc[:, :-1].to_numpy(float), ct.iloc[:, -1].to_numpy(float), None)
        if (DATA / "blogfeedback.csv").exists():       # UCI BlogFeedback, 52,397 x 280
            bl = _cap(pd.read_csv(DATA / "blogfeedback.csv"), 10_000)
            ds["blogfeedback_p280"] = (bl.iloc[:, :-1].to_numpy(float), bl.iloc[:, -1].to_numpy(float), None)
        if (DATA / "ujiindoorloc.csv").exists():       # UCI UJIIndoorLoc, ~21k x 520 (target: longitude)
            uj = _cap(pd.read_csv(DATA / "ujiindoorloc.csv"), 10_000)
            ds["ujiindoorloc_p520"] = (uj.iloc[:, :-1].to_numpy(float), uj.iloc[:, -1].to_numpy(float), None)
        if (DATA / "geomusic.csv").exists():           # PMLB 4544, 1,059 x 117 (target: latitude)
            gm = pd.read_csv(DATA / "geomusic.csv")
            ds["geomusic_p117"] = (gm.iloc[:, :-1].to_numpy(float), gm.iloc[:, -1].to_numpy(float), None)
    return ds

def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))

def clean(M):
    """Drop all-non-finite training columns and fill the remaining non-finite
    cells with the column's training median. Returns the cleaned matrix and the
    `fill` statistics that `clean_test` applies to held-out rows."""
    M = np.asarray(M, float)
    M[~np.isfinite(M)] = np.nan
    col_ok = ~np.all(np.isnan(M), axis=0)
    M = M[:, col_ok]
    med = np.nanmedian(M, axis=0)
    idx = np.where(np.isnan(M))
    M[idx] = np.take(med, idx[1])
    return M, (col_ok, med)

def clean_test(M, fill):
    """Apply training-set cleaning to test rows: keep the training columns and
    fill non-finite cells with the training medians, never with a constant
    (a 0 in a ratio feature is far outside its training range)."""
    col_ok, med = fill
    M = np.asarray(M, float)[:, col_ok]
    idx = np.where(~np.isfinite(M))
    M[idx] = np.take(med, idx[1])
    return M

# ---------------------------------------------------------------- methods
def run_ridge(Xtr, ytr, Xte):
    m = ridge().fit(Xtr, ytr)
    return m.predict(Xte), {}

def run_rf(Xtr, ytr, Xte):
    m = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    return m.predict(Xte), {}

def run_lgbm(Xtr, ytr, Xte):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=300, random_state=0, verbose=-1).fit(Xtr, ytr)
    return m.predict(Xte), {}

def run_autofeat(Xtr, ytr, Xte):
    from autofeat import AutoFeatRegressor
    af = AutoFeatRegressor(feateng_steps=2, verbose=0, n_jobs=N_JOBS)
    Ftr = af.fit_transform(pd.DataFrame(Xtr, columns=[f"x{i:03d}" for i in range(Xtr.shape[1])]), ytr)
    Fte = af.transform(pd.DataFrame(Xte, columns=[f"x{i:03d}" for i in range(Xte.shape[1])]))
    new = [c for c in Ftr.columns if c not in {f"x{i:03d}" for i in range(Xtr.shape[1])}]
    m = ridge().fit(Ftr.to_numpy(float), ytr)
    return m.predict(Fte.to_numpy(float)), {"n_new": len(new), "formulas": new,
                                            "n_jobs": N_JOBS}

def run_featuretools(Xtr, ytr, Xte):
    import featuretools as ft
    cols = [f"x{i}" for i in range(Xtr.shape[1])]
    prim = ["multiply_numeric", "divide_numeric"]
    def dfs(X):
        df = pd.DataFrame(X, columns=cols)
        df["_id"] = range(len(df))
        es = ft.EntitySet("d")
        es = es.add_dataframe(dataframe_name="t", dataframe=df, index="_id")
        F, defs = ft.dfs(entityset=es, target_dataframe_name="t",
                         trans_primitives=prim, max_depth=1, verbose=False,
                         n_jobs=FT_N_JOBS)
        return F.reindex(sorted(F.columns), axis=1)
    Ftr, Fte = dfs(Xtr), dfs(Xte)
    Mtr, fill = clean(Ftr.to_numpy())
    Mte = clean_test(Fte.to_numpy(), fill)
    m = ridge().fit(Mtr, ytr)
    return m.predict(Mte), {"n_new": Mtr.shape[1] - Xtr.shape[1], "n_jobs": FT_N_JOBS}

def run_openfe(Xtr, ytr, Xte):
    from openfe import OpenFE, transform
    cols = [f"x{i}" for i in range(Xtr.shape[1])]
    dtr = pd.DataFrame(Xtr, columns=cols)
    dte = pd.DataFrame(Xte, columns=cols)
    ofe = OpenFE()
    feats = ofe.fit(data=dtr, label=pd.Series(ytr.astype(float)), task="regression",
                    n_jobs=N_JOBS, verbose=False)
    ttr, tte = transform(dtr, dte, feats[:20], n_jobs=N_JOBS)
    Mtr, fill = clean(ttr.to_numpy())
    Mte = clean_test(tte.to_numpy(), fill)
    m = ridge().fit(Mtr, ytr)
    return m.predict(Mte), {"n_new": ttr.shape[1] - Xtr.shape[1], "n_jobs": N_JOBS}

# Unset reproduces the library default exactly; set BEAMFEAT_PARSIMONY=none to
# keep the whole screened set instead. Recorded per row as `parsimony`.
PARSIMONY = os.environ.get("BEAMFEAT_PARSIMONY") or "forward"
PARSIMONY = None if PARSIMONY == "none" else PARSIMONY

def run_beamfeat(Xtr, ytr, Xte):
    """beamfeat as shipped: its own estimator and internal ridge, at defaults."""
    from beamfeat import BeamFeatRegressor
    m = BeamFeatRegressor(random_state=0, parsimony=PARSIMONY).fit(Xtr, ytr)
    info = {"n_new": len(m.formulas()), "formulas": m.formulas(),
            "parsimony": PARSIMONY,
            "fdr": bool(getattr(m, "fdr_controlled_", False))}
    return m.predict(Xte), info

def run_beamfeat_ridge(Xtr, ytr, Xte):
    """beamfeat as a transformer into the shared RidgeCV, exactly as autofeat,
    featuretools and OpenFE are run. This is the like-for-like row: the only
    thing that differs from those three is which features were constructed."""
    from beamfeat import BeamFeatTransformer
    t = BeamFeatTransformer(random_state=0, parsimony=PARSIMONY).fit(Xtr, ytr)
    m = ridge().fit(t.transform(Xtr), ytr)
    info = {"n_new": len(t.formulas()), "formulas": t.formulas(),
            "parsimony": PARSIMONY,
            "fdr": bool(getattr(t, "fdr_controlled_", False))}
    return m.predict(t.transform(Xte)), info

METHODS = {"ridge_raw": run_ridge, "rf_raw": run_rf, "lgbm_raw": run_lgbm,
           "autofeat": run_autofeat, "featuretools": run_featuretools,
           "openfe": run_openfe, "beamfeat": run_beamfeat,
           "beamfeat_ridge": run_beamfeat_ridge}

def recovered(formulas, tokens):
    """Did any single returned feature reference all of the generating columns?

    This is column recovery, not symbolic recovery: it checks which input
    columns a constructed feature touches, not the operators combining them.
    For a target built from x0*x1/x2 the feature x0+x1+x2 counts, and so does
    one that also drags in an irrelevant column. Read it as a necessary
    condition for finding the law rather than as proof of it.

    Returns None when the dataset has no known generating columns, or when the
    method returned no inspectable formulas at all.
    """
    if not tokens or not formulas:
        return None
    wanted = {t.lstrip("x").lstrip("0") or "0" for t in tokens}
    for f in formulas:
        # column indices are written x0, x1 ... by beamfeat and x000, x001 ...
        # by autofeat; both normalise to the bare index.
        referenced = {m.lstrip("0") or "0" for m in re.findall(r"x0*(\d+)", f)}
        if wanted <= referenced:
            return True
    return False

class BudgetExceeded(RuntimeError):
    pass

def _budget_worker(mname, Xtr, ytr, Xte, q):
    try:
        pred, info = METHODS[mname](Xtr, ytr, Xte)
        q.put(("ok", np.asarray(pred), info))
    except Exception as e:  # exceptions cross the process boundary as data
        q.put(("err", f"{type(e).__name__}: {e}", None))

def call_with_budget(mname, Xtr, ytr, Xte, budget_s):
    """Run one fit under a wall-clock budget in a spawned subprocess.

    A method that neither finishes nor crashes within the budget is recorded
    as BudgetExceeded and recorded like the crash rows of the
    original study. FIT_BUDGET_S=0 disables the wrapper and
    reproduces the original in-process behaviour exactly.
    """
    if budget_s <= 0:
        return METHODS[mname](Xtr, ytr, Xte)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_budget_worker, args=(mname, Xtr, ytr, Xte, q))
    proc.start()
    proc.join(budget_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise BudgetExceeded(f"exceeded {budget_s:.0f}s wall-clock budget")
    if proc.exitcode not in (0, None) and q.empty():
        raise RuntimeError(f"subprocess died with exit code {proc.exitcode} (likely out of memory)")
    status, payload, info = q.get(timeout=30)
    if status == "err":
        raise RuntimeError(payload)
    return payload, info

def main(which, methods, n_splits, out):
    ds = get_datasets(which)
    only = os.environ.get("DATASETS")
    if only:
        keep = only.split(",")
        ds = {k: v for k, v in ds.items() if k in keep}
    rows = []
    for name, (X, y, tokens) in ds.items():
        for split in range(n_splits):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=split)
            budget_s = float(os.environ.get("FIT_BUDGET_S", "900"))
            for mname in methods:
                t0 = time.perf_counter()
                try:
                    pred, info = call_with_budget(mname, Xtr, ytr, Xte, budget_s)
                    dt = time.perf_counter() - t0
                    r2 = float(r2_score(yte, pred))
                    rec = recovered(info.get("formulas", []), tokens) if tokens is not None else None
                    rows.append(dict(dataset=name, method=mname, split=split, r2=r2,
                                     seconds=dt, n_new=info.get("n_new"), n_jobs=info.get("n_jobs"),
                                     recovered=rec, fdr=info.get("fdr"),
                                     parsimony=info.get("parsimony"),
                                     formulas=info.get("formulas"), error=None))
                    print(f"{name:16s} {mname:13s} s{split} R2={r2:+.4f} {dt:7.1f}s n_new={info.get('n_new')}", flush=True)
                except Exception as e:
                    dt = time.perf_counter() - t0
                    rows.append(dict(dataset=name, method=mname, split=split, r2=None,
                                     seconds=dt, n_new=None, recovered=None, fdr=None,
                                     formulas=None, error=f"{type(e).__name__}: {e}"))
                    print(f"{name:16s} {mname:13s} s{split} ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
        with open(out, "w") as fh:          # checkpoint after each dataset
            json.dump(rows, fh, indent=1)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("wrote", out)

if __name__ == "__main__":
    which = sys.argv[1]
    methods = sys.argv[2].split(",")
    n_splits = int(sys.argv[3])
    out = sys.argv[4]
    main(which, methods, n_splits, out)
