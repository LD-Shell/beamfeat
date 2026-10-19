"""Realised FDR and power of the selection procedures under dependence.

The procedures target different nulls: permutation p-values under BH/BY
certify marginal association (Proposition 1), while knockoffs certify
conditional relevance. Every cell therefore reports realised FDR under both
definitions where they differ. Regimes:

  independent       independent candidates; the two nulls coincide.
  correlated_nulls  null candidates correlated among themselves but
                    independent of the target: dependence among the p-values
                    with the marginal null intact, the regime of Section 3.2.
  shared_factor     every candidate loads on a factor that also drives the
                    target, so all are marginally non-null and only the
                    conditional FDR separates the procedures.
  beam              candidates from a beam search over an expression DAG;
                    ground truth is marginal, defined against the noiseless
                    signal on an independent probe draw. The search and the
                    selectors use the same rows, with no split, and on this
                    design (y = x0*x1, eight columns, beam width 25) every one
                    of the 25 candidates is associated with the signal in all
                    100 trials, so there is no null candidate to select and a
                    realised FDR of zero is certain. The regime shows power on
                    searched candidates, not FDR control.

A trial whose selector raises is counted under ``errors``, with the first
message kept per cell, and is excluded from the means; ``skipped`` counts only
trials where a procedure does not apply (fixed-X knockoffs with n < 2m).

Run:  python selector_comparison.py --trials 100 --out results/selector_comparison.json
      python selector_comparison.py --trials 100 --m 100 --k 10 --out results/selector_comparison_m100.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from beamfeat.selection import KnockoffSelector, PermutationSelector


def make_independent(rng, n, m, k, amp=0.6):
    F = rng.standard_normal((n, m))
    beta = np.zeros(m)
    beta[:k] = amp
    y = F @ beta + rng.standard_normal(n)
    sig = np.arange(m) < k
    return F, y, sig, sig            # marginal truth, conditional truth


def make_correlated_nulls(rng, n, m, k, rho=0.7, amp=0.6):
    zs = rng.standard_normal((n, 1))
    zn = rng.standard_normal((n, 1))          # independent of y's drivers
    F = np.empty((n, m))
    F[:, :k] = np.sqrt(rho) * zs + np.sqrt(1 - rho) * rng.standard_normal((n, k))
    F[:, k:] = np.sqrt(rho) * zn + np.sqrt(1 - rho) * rng.standard_normal((n, m - k))
    beta = np.zeros(m)
    beta[:k] = amp
    y = F @ beta + rng.standard_normal(n)
    sig = np.arange(m) < k
    return F, y, sig, sig            # nulls independent of y: nulls agree


def make_shared_factor(rng, n, m, k, rho=0.5, amp=0.6):
    z = rng.standard_normal((n, 1))
    F = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal((n, m))
    beta = np.zeros(m)
    beta[:k] = amp
    y = F @ beta + rng.standard_normal(n)
    cond = np.arange(m) < k
    marg = np.ones(m, bool)          # every candidate marginally non-null
    return F, y, marg, cond


def make_beam(rng, n, p=8, noise=0.10):
    from beamfeat.expression import Evaluator
    from beamfeat.search import BeamSearch

    X = rng.uniform(1, 6, (n, p))
    signal = X[:, 0] * X[:, 1]
    y = signal + rng.normal(0, noise * signal.std(), n)
    bs = BeamSearch(beam_width=25, max_depth=2, random_state=int(rng.integers(1 << 31)))
    res = bs.run(X, y)
    cols = [f"x{i}" for i in range(p)]
    ev = Evaluator(dict(zip(cols, X.T)))
    kept, F = ev.evaluate_many(list(res.nodes))
    Xp = np.random.default_rng(12345).uniform(1, 6, (4000, p))
    sig_p = Xp[:, 0] * Xp[:, 1]
    evp = Evaluator(dict(zip(cols, Xp.T)))
    truth = []
    with np.errstate(all="ignore"):
        for e in kept:
            v = evp.evaluate(e, apply_filters=False)
            v = np.asarray(v, float) if v is not None else np.full(len(Xp), np.nan)
            mfin = np.isfinite(v)
            r = np.corrcoef(v[mfin], sig_p[mfin])[0, 1] if mfin.sum() > 100 and v[mfin].std() > 0 else 0.0
            truth.append(abs(r) > 0.1)
    marg = np.asarray(truth)
    return F, y, marg, None          # conditional truth not defined here


def run_selector(kind, F, y, q, rng):
    n, m = F.shape
    if kind in ("bh", "by"):
        sel = PermutationSelector(target_fdr=q, correction=kind,
                                  random_state=int(rng.integers(1 << 31)))
    elif kind == "knockoff_fixed":
        if n < 2 * m:
            return None
        sel = KnockoffSelector(target_fdr=q, construction="fixed",
                               random_state=int(rng.integers(1 << 31)))
    elif kind == "knockoff_modelx":
        sel = KnockoffSelector(target_fdr=q, construction="gaussian",
                               random_state=int(rng.integers(1 << 31)))
    else:
        raise ValueError(kind)
    res = sel.select(F, y)
    mask = res.mask() if callable(res.mask) else res.mask
    return np.asarray(mask, bool)


def fdp(mask, truth):
    if truth is None:
        return None
    R = int(mask.sum())
    V = int((mask & ~truth).sum())
    return V / max(R, 1)


def main(a):
    regimes = {
        "independent": lambda rng: make_independent(rng, a.n, a.m, a.k),
        "correlated_nulls": lambda rng: make_correlated_nulls(rng, a.n, a.m, a.k),
        "shared_factor": lambda rng: make_shared_factor(rng, a.n, a.m, a.k),
        "beam": lambda rng: make_beam(rng, a.n),
    }
    if a.regimes:
        keep = set(a.regimes.split(","))
        regimes = {k: v for k, v in regimes.items() if k in keep}
    selectors = ["bh", "by", "knockoff_fixed", "knockoff_modelx"]
    levels = [0.05, 0.10, 0.20]
    rows = []
    for regime, maker in regimes.items():
        for q in levels:
            for kind in selectors:
                fm, fc, pw, times, skipped, errors, first_error = [], [], [], [], 0, 0, None
                for t in range(a.trials):
                    rng = np.random.default_rng(100 * t + 7)
                    F, y, marg, cond = maker(rng)
                    t0 = time.perf_counter()
                    try:
                        mask = run_selector(kind, F, y, q, rng)
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        first_error = first_error or f"{type(exc).__name__}: {exc}"[:200]
                        continue
                    dt = time.perf_counter() - t0
                    if mask is None:
                        skipped += 1
                        continue
                    fm.append(fdp(mask, marg))
                    v = fdp(mask, cond)
                    if v is not None:
                        fc.append(v)
                    planted = cond if cond is not None else marg
                    pw.append((mask & planted).sum() / max(int(planted.sum()), 1))
                    times.append(dt)
                mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
                rows.append(dict(regime=regime, selector=kind, nominal=q,
                                 trials=a.trials - skipped - errors, skipped=skipped,
                                 errors=errors, first_error=first_error,
                                 fdr_marginal=mean(fm), fdr_conditional=mean(fc),
                                 power=mean(pw), seconds=mean(times)))
                r = rows[-1]
                fmt = lambda v: "--" if v is None else f"{v:.3f}"  # noqa: E731
                print(f"{regime:17s} {kind:16s} q={q:.2f} "
                      f"FDRm={fmt(r['fdr_marginal'])} FDRc={fmt(r['fdr_conditional'])} "
                      f"power={fmt(r['power'])} skipped={skipped} errors={errors}", flush=True)
                if first_error:
                    print(f"  first error: {first_error}", flush=True)
            with open(a.out, "w") as fh:
                json.dump(rows, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--m", type=int, default=25)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/selector_comparison.json")
    ap.add_argument("--regimes", default="", help="comma list, e.g. 'independent,correlated_nulls'")
    a = ap.parse_args()
    main(a)
