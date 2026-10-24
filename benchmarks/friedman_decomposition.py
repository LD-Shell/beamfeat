"""Friedman #1 gap decomposition: representation oracle vs screening ceiling.

Splits the shortfall on Friedman #1 into two parts. A least-squares oracle on
the depth-2 basis {ab, (ab)^2, c, c^2, d, e} bounds what any model using that
representation could reach. The screening-admissible ceiling refits on only
the terms whose marginal null the screen rejects, bounding what a marginal
screen could reach. The pipeline is then run on the same draw.

Reported over several independent draws with a standard deviation, because the
oracle and ceiling are least-squares fits on a fixed basis and barely move,
while the pipeline's score shifts by an order of magnitude more as the search
makes discrete choices. Which nulls the screen rejects is itself
draw-dependent, so the ceiling is computed on the terms actually selected
rather than an assumed set. The script checks the ordering that the argument
rests on, achieved below ceiling below oracle, and fails if it does not hold.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

N_DRAWS = 6
NAMES = ("ab", "(ab)^2", "c", "c^2", "d", "e")


def friedman1(seed: int, n: int = 800, d: int = 10):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
        + rng.normal(0, 1, n)
    )
    return X, y


def _basis(A: np.ndarray) -> np.ndarray:
    u = A[:, 0] * A[:, 1]
    return np.column_stack([u, u**2, A[:, 2], A[:, 2] ** 2, A[:, 3], A[:, 4]])


def _decompose(train_seed: int, test_seed: int, parsimony: str | None = "forward") -> dict:
    from beamfeat import BeamFeatRegressor
    from beamfeat.selection import PermutationSelector

    X, y = friedman1(train_seed)
    X_test, y_test = friedman1(test_seed)
    F, F_test = _basis(X), _basis(X_test)

    def fit_r2(indices):
        indices = list(indices)
        design = np.column_stack([np.ones(len(y)), F[:, indices]])
        test = np.column_stack([np.ones(len(y_test)), F_test[:, indices]])
        weights, *_ = np.linalg.lstsq(design, y, rcond=None)
        return float(1 - np.var(y_test - test @ weights) / np.var(y_test))

    holdout = np.random.default_rng(train_seed).permutation(len(y))[: len(y) // 2]
    admitted = sorted(
        PermutationSelector(target_fdr=0.1, correction="by", random_state=0)
        .select(F[holdout], y[holdout])
        .selected.tolist()
    )
    model = BeamFeatRegressor(
        max_depth=2, beam_width=40, random_state=0, parsimony=parsimony
    ).fit(X, y)
    achieved = model.score(X_test, y_test)
    return {
        "oracle": fit_r2(range(6)),
        "ceiling": fit_r2(admitted),
        "achieved": float(achieved),
        "admitted": [NAMES[i] for i in admitted],
        "n_terms": model.n_features_out_,
        "fdr_controlled": model.fdr_controlled_,
    }


def main(n_draws: int = N_DRAWS, parsimony: str | None = "forward") -> dict:
    draws = [_decompose(2 * k, 2 * k + 1, parsimony) for k in range(n_draws)]

    print(f"Friedman #1 decomposition over {n_draws} independent draws")
    print(f"  {'draw':>5} {'oracle':>8} {'ceiling':>8} {'achieved':>9}  admitted by the marginal null")
    for k, d in enumerate(draws):
        print(f"  {k:>5} {d['oracle']:>8.3f} {d['ceiling']:>8.3f} {d['achieved']:>9.3f}"
              f"  {', '.join(d['admitted'])}")

    summary = {}
    for key in ("oracle", "ceiling", "achieved"):
        values = np.array([d[key] for d in draws])
        summary[key] = {"mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "values": values.tolist()}
    print(f"\n  {'mean':>5} {summary['oracle']['mean']:>8.3f} "
          f"{summary['ceiling']['mean']:>8.3f} {summary['achieved']['mean']:>9.3f}")
    print(f"  {'sd':>5} {summary['oracle']['sd']:>8.3f} "
          f"{summary['ceiling']['sd']:>8.3f} {summary['achieved']['sd']:>9.3f}")

    screening_gap = summary["oracle"]["mean"] - summary["ceiling"]["mean"]
    search_gap = summary["ceiling"]["mean"] - summary["achieved"]["mean"]
    print(f"\n  lost to marginal screening: {screening_gap:.3f}")
    print(f"  lost to the search:         {search_gap:.3f}")

    # The claim the decomposition makes, checked on every draw rather than
    # compared against a recorded number.
    bad = [k for k, d in enumerate(draws)
           if not d["achieved"] < d["ceiling"] < d["oracle"]]
    if bad:
        raise AssertionError(f"achieved < ceiling < oracle does not hold on draws {bad}")
    never = sorted({n for d in draws for n in d["admitted"]})
    print(f"  admitted at least once: {', '.join(never)}")

    summary["draws"] = draws
    summary["parsimony"] = parsimony
    return summary


if __name__ == "__main__":
    main()
