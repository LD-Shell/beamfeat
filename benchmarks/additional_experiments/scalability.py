"""Scalability in the input dimension, with a planted signal among p columns.

Sweeps p over a grid under independent and equicorrelated distractors. Each
cell runs in a spawned subprocess so that wall time and peak RSS are measured
per fit in isolation. Peak memory mattered when the archived results were
recorded: the search evaluated each depth's proposals en bloc, so it scaled
as O(beam x pool x n_search). From 0.5.0 it evaluates them in bounded chunks,
and memory no longer grows with the number of proposals.
Reported per cell: fit seconds, peak RSS (MB), recovery of the planted
formula up to value-equivalence, selections, false features (selected
formulas touching only distractor columns), and the fitted FDR flag.

Run:  python scalability.py --p-grid 10,30,100,300,1000 --seeds 5
      python scalability.py --p-grid 10,30 --seeds 1 --quick
"""
from __future__ import annotations

import argparse
import json
import re
import resource
import time

import numpy as np

TARGETS = {
    "ab": (("x0", "x1"), lambda X: X[:, 0] * X[:, 1]),
    "ab_over_c": (("x0", "x1", "x2"), lambda X: X[:, 0] * X[:, 1] / X[:, 2]),
}


def draw(rng, n, p, regime):
    if regime == "independent":
        return rng.uniform(1, 6, (n, p))
    z = rng.uniform(1, 6, (n, 1))                      # equicorrelated rho~0.5
    return np.sqrt(0.5) * z + np.sqrt(0.5) * rng.uniform(1, 6, (n, p))


def _cell(target, regime, p, n, noise, seed, q, parsimony="forward"):
    """Child process: one fit, reporting metrics through the queue."""
    from beamfeat import BeamFeatTransformer

    cols, fn = TARGETS[target]
    rng = np.random.default_rng(10_000 * seed + p)
    X = draw(rng, n, p, regime)
    y = fn(X)
    y = y + rng.normal(0, noise * y.std(), n)
    t0 = time.perf_counter()
    m = BeamFeatTransformer(random_state=seed, parsimony=parsimony).fit(X, y)
    dt = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    Xp = draw(np.random.default_rng(seed + 99), 2000, p, regime)
    truth = np.asarray(fn(Xp), float)
    rec = False
    try:
        F = m.transform(Xp)
        with np.errstate(all="ignore"):
            for j in range(F.shape[1] if F is not None and F.size else 0):
                col = np.asarray(F[:, j], float)
                mk = np.isfinite(col) & np.isfinite(truth)
                if mk.sum() > 100 and np.std(col[mk]) > 0 and \
                        abs(np.corrcoef(col[mk], truth[mk])[0, 1]) > 0.999:
                    rec = True
                    break
    except Exception:
        pass

    signal_cols = {c.lstrip("x") for c in cols}
    false_feats = 0
    for f in m.formulas():
        touched = set(re.findall(r"x0*(\d+)", f))
        touched = {t.lstrip("0") or "0" for t in touched}
        if touched and touched.isdisjoint(signal_cols):
            false_feats += 1
    q.put(dict(seconds=round(dt, 2), peak_mb=round(peak_mb, 1),
               recovered=rec, n_selected=len(m.formulas()),
               false_features=false_feats, parsimony=parsimony,
               fdr=bool(getattr(m, "fdr_controlled_", False))))


def run_cell(target, regime, p, n, noise, seed, budget_s, parsimony="forward"):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_cell,
                       args=(target, regime, p, n, noise, seed, q, parsimony))
    proc.start()
    proc.join(budget_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        return dict(error=f"BudgetExceeded: >{budget_s:.0f}s")
    if q.empty():
        return dict(error=f"subprocess died, exit {proc.exitcode} (likely out of memory)")
    return q.get(timeout=30)


def main(a):
    grid = [int(x) for x in a.p_grid.split(",")]
    regimes = ["independent"] if a.quick else ["independent", "equicorrelated"]
    targets = ["ab"] if a.quick else list(TARGETS)
    rows = []
    for target in targets:
        for regime in regimes:
            for p in grid:
                for seed in range(a.seeds):
                    out = run_cell(target, regime, p, a.n, a.noise, seed, a.budget,
                                   a.parsimony)
                    out.update(target=target, regime=regime, p=p, n=a.n, seed=seed)
                    rows.append(out)
                    msg = out.get("error") or (
                        f"{out['seconds']:7.1f}s {out['peak_mb']:8.1f}MB "
                        f"rec={out['recovered']} sel={out['n_selected']} "
                        f"false={out['false_features']} fdr={out['fdr']}")
                    print(f"{target:10s} {regime:14s} p={p:5d} s{seed} {msg}", flush=True)
                with open(a.out, "w") as fh:
                    json.dump(rows, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-grid", default="10,30,100,300,1000")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--out", default="results/scalability.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parsimony", default="forward", choices=["forward", "none"],
                    help="'forward' is the library default, the compact greedy subset; "
                         "'none' keeps the whole screened set")
    a = ap.parse_args()
    a.parsimony = None if a.parsimony == "none" else a.parsimony
    main(a)
