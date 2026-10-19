"""Recovery of planted targets across search depth, scorer, and beam width.

A ladder of generating formulas at minimal search depths 1-4, plus targets on
zero-mean inputs whose intermediates carry zero population correlation with
the target (with a, b, c independent and zero-mean, corr(ab, abc) = 0), so a
marginal heuristic must rank the needed path from finite-sample dependence
alone. Recovery is scored up to algebraic value-equivalence: a selected
feature counts when |corr| > 0.999 with the planted feature on fresh probe
rows, so ((x0/x2)*x1) and ((x1/x2)*x0) count once, and an additive target
counts when selected as the single composite.

Run:  python depth_ladder.py --seeds 10 --out results/depth_ladder.json
      python depth_ladder.py --seeds 3 --quick
"""
from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np

# depth = minimal search depth at which the target is constructible under the
# round-based proposal rule (round d pairs a depth-(d-1) node with anything in
# the pool), which is what the recovery-vs-depth figure should be plotted over.
LADDER = [
    # name, depth, n_cols, positive_inputs, target_fn, true_feature_fns
    ("d1_product", 1, 4, True, lambda X: X[:, 0] * X[:, 1], [lambda X: X[:, 0] * X[:, 1]]),
    ("d1_ratio", 1, 4, True, lambda X: X[:, 0] / X[:, 1], [lambda X: X[:, 0] / X[:, 1]]),
    ("d2_three_way", 2, 4, True, lambda X: X[:, 0] * X[:, 1] / X[:, 2],
     [lambda X: X[:, 0] * X[:, 1] / X[:, 2]]),
    ("d2_sum_times", 2, 4, True, lambda X: (X[:, 0] + X[:, 1]) * X[:, 2],
     [lambda X: (X[:, 0] + X[:, 1]) * X[:, 2]]),
    ("d2_sq_plus", 2, 4, True, lambda X: X[:, 0] ** 2 + X[:, 1],
     [lambda X: X[:, 0] ** 2 + X[:, 1]]),
    ("d3_prod_of_sums", 2, 6, True, lambda X: (X[:, 0] + X[:, 1]) * (X[:, 2] + X[:, 3]),
     [lambda X: (X[:, 0] + X[:, 1]) * (X[:, 2] + X[:, 3])]),
    ("d3_ratio_of_sum", 2, 6, True, lambda X: X[:, 0] * X[:, 1] / (X[:, 2] + X[:, 3]),
     [lambda X: X[:, 0] * X[:, 1] / (X[:, 2] + X[:, 3])]),
    ("d3_sqrt_ratio", 3, 4, True, lambda X: np.sqrt(X[:, 0] * X[:, 1]) / X[:, 2],
     [lambda X: np.sqrt(X[:, 0] * X[:, 1]) / X[:, 2]]),
    ("d4_nested_ratio", 3, 6, True,
     lambda X: (X[:, 0] + X[:, 1]) * (X[:, 2] + X[:, 3]) / X[:, 4],
     [lambda X: (X[:, 0] + X[:, 1]) * (X[:, 2] + X[:, 3]) / X[:, 4]]),
    ("d4_distance2d", 4, 4, True,
     lambda X: np.sqrt((X[:, 0] - X[:, 1]) ** 2 + (X[:, 2] - X[:, 3]) ** 2),
     [lambda X: np.sqrt((X[:, 0] - X[:, 1]) ** 2 + (X[:, 2] - X[:, 3]) ** 2)]),
    # adversarial: zero-mean inputs, marginally invisible intermediates/mains
    ("adv_ab_zeromean", 1, 4, False, lambda X: X[:, 0] * X[:, 1],
     [lambda X: X[:, 0] * X[:, 1]]),
    ("adv_abc_zeromean", 2, 4, False, lambda X: X[:, 0] * X[:, 1] * X[:, 2],
     [lambda X: X[:, 0] * X[:, 1] * X[:, 2]]),
    ("adv_ab_plus_cd", 2, 6, False, lambda X: X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3],
     [lambda X: X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]]),
    # p=10 variant: at p=4 the proposal space (~70) barely exceeds the beam, so
    # invisible intermediates survive by default; at p=10 (~400 proposals) the
    # beam is genuinely selective and the failure mode can actually bite.
    ("adv_abc_p10", 2, 10, False, lambda X: X[:, 0] * X[:, 1] * X[:, 2],
     [lambda X: X[:, 0] * X[:, 1] * X[:, 2]]),
    # two invisible hops: ab and abc are both population-uncorrelated with abcd
    ("adv_abcd_p8", 3, 8, False, lambda X: X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3],
     [lambda X: X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]]),
]


def draw(n, p, positive, rng):
    return rng.uniform(1, 6, (n, p)) if positive else rng.uniform(-3, 3, (n, p))


def equivalent(model, X_probe, true_fn) -> bool:
    """Does any selected feature match the planted one up to algebraic equivalence?"""
    truth = np.asarray(true_fn(X_probe), float)
    try:
        F = model.transform(X_probe)  # transformer view of selected features
    except Exception:
        return False
    if F is None or F.size == 0:
        return False
    with np.errstate(all="ignore"):
        for j in range(F.shape[1]):
            col = np.asarray(F[:, j], float)
            m = np.isfinite(col) & np.isfinite(truth)
            if m.sum() < 30 or np.std(col[m]) == 0 or np.std(truth[m]) == 0:
                continue
            if abs(np.corrcoef(col[m], truth[m])[0, 1]) > 0.999:
                return True
    return False


def main(a):
    from beamfeat import BeamFeatTransformer

    grid_scorers = a.scorers.split(",") if a.scorers else ["correlation", "mutual_info"]
    grid_beams = [20, 50, 100] if a.quick else [20, 50, 100, 200]
    rows = []
    ladder = LADDER
    if a.problems:
        keep = set(a.problems.split(","))
        ladder = [row for row in LADDER if row[0] in keep]
    for name, depth, p, positive, target_fn, truth_fns in ladder:
        max_depth = max(2, min(depth, 4))
        for scorer in grid_scorers:
            for beam in grid_beams:
                for seed in range(a.seeds):
                    # crc32 rather than hash(): Python randomises string
                    # hashing per process, so hash(name) reseeded the data on
                    # every run unless PYTHONHASHSEED was fixed.
                    rng = np.random.default_rng(1000 * seed + zlib.crc32(name.encode()) % 1000)
                    X = draw(a.n, p, positive, rng)
                    y = target_fn(X)
                    y = y + rng.normal(0, a.noise * (np.std(y) or 1.0), a.n)
                    X_probe = draw(2000, p, positive, np.random.default_rng(seed + 7))
                    t0 = time.perf_counter()
                    try:
                        kw = {}
                        if a.unary:
                            kw["unary_ops"] = tuple(a.unary.split(","))
                        if a.binary:
                            kw["binary_ops"] = tuple(a.binary.split(","))
                        m = BeamFeatTransformer(
                            scorer=scorer, beam_width=beam, max_depth=max_depth,
                            random_state=seed, parsimony=a.parsimony, **kw,
                        ).fit(X, y)
                        dt = time.perf_counter() - t0
                        rec = [equivalent(m, X_probe, fn) for fn in truth_fns]
                        rows.append(dict(
                            problem=name, depth=depth, scorer=scorer, beam=beam,
                            seed=seed, seconds=round(dt, 2),
                            n_selected=len(m.formulas()),
                            parsimony=a.parsimony,
                            fdr=bool(getattr(m, "fdr_controlled_", False)),
                            recovered_all=all(rec), recovered_any=any(rec),
                            error=None,
                        ))
                    except Exception as e:  # noqa: BLE001
                        rows.append(dict(problem=name, depth=depth, scorer=scorer,
                                         beam=beam, seed=seed, seconds=None,
                                         n_selected=None, fdr=None,
                                         recovered_all=None, recovered_any=None,
                                         error=f"{type(e).__name__}: {e}"))
                print(f"{name:18s} scorer={scorer:12s} beam={beam:3d} "
                      f"recovered_all="
                      f"{np.mean([r['recovered_all'] for r in rows if r['problem']==name and r['scorer']==scorer and r['beam']==beam and r['recovered_all'] is not None] or [np.nan]):.2f}",
                      flush=True)
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--out", default="results/depth_ladder.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--unary", default="", help="comma list to restrict unary ops, e.g. 'sqrt,square'")
    ap.add_argument("--binary", default="", help="comma list to restrict binary ops, e.g. 'mul,div'")
    ap.add_argument("--problems", default="", help="comma list to run a subset of the ladder")
    ap.add_argument("--scorers", default="", help="comma list to restrict scorers, e.g. 'correlation'")
    ap.add_argument("--parsimony", default="forward", choices=["forward", "none"],
                    help="'forward' is the library default, the compact greedy subset; "
                         "'none' keeps the whole screened set")
    a = ap.parse_args()
    a.parsimony = None if a.parsimony == "none" else a.parsimony
    main(a)
