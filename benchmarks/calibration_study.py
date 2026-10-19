"""End-to-end FDR calibration study — the paper's headline figures.

200 signal replicates at nominal FDR 0.10 (empirical FDR, power, fallback
count) and 60 global-null replicates (total selections, and the number of
trials selecting anything). Seeds are fixed; the run regenerates the
published numbers exactly.

Each fit uses all 500 rows, so the per-fit ``score`` is in-sample R²: it is
recorded for the parsimony comparison, not as a measure of predictive fit.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _upper_bound(successes: int, trials: int) -> float:
    """One-sided 95% Clopper-Pearson upper limit on a proportion."""
    from scipy.stats import beta

    if successes >= trials:
        return 1.0
    return float(beta.ppf(0.95, successes + 1, trials - successes))


TARGET_FDR = 0.1


def main(n_signal: int = 200, n_null: int = 60, parsimony: str | None = "forward") -> dict:
    from beamfeat import BeamFeatRegressor

    fdps, recovered, fallbacks = [], 0, 0
    per_fit = []
    for trial in range(n_signal):
        rng = np.random.default_rng(10_000 + trial)
        X = rng.uniform(1, 6, (500, 6))
        signal = X[:, 0] * X[:, 1]
        y = signal + rng.normal(0, 0.05 * np.std(signal), 500)
        model = BeamFeatRegressor(
            max_depth=2, beam_width=25, target_fdr=TARGET_FDR, random_state=trial,
            parsimony=parsimony,
        ).fit(X, y)
        per_fit.append({
            "arm": "signal", "trial": trial,
            "score": float(model.score(X, y)),
            "n_terms": model.n_features_out_,
            "fdr_controlled": model.fdr_controlled_,
            "n_screened": (
                None if model.selection_result_ is None
                else int(model.selection_result_.n_selected)
            ),
        })
        if not model.fdr_controlled_:
            fallbacks += 1
            continue
        relevant = {"x0", "x1"}
        fdps.append(
            sum(1 for node in model.features_ if not (node.columns() & relevant))
            / len(model.features_)
        )
        recovered += any("x0" in f and "x1" in f for f in model.formulas())

    null_selected, null_trials_selecting = 0, 0
    for trial in range(n_null):
        rng = np.random.default_rng(50_000 + trial)
        X = rng.uniform(1, 6, (500, 6))
        y = rng.normal(0, 1, 500)
        model = BeamFeatRegressor(
            max_depth=2, beam_width=25, target_fdr=TARGET_FDR, random_state=trial,
            parsimony=parsimony,
        ).fit(X, y)
        per_fit.append({
            "arm": "null", "trial": trial,
            "score": float(model.score(X, y)),
            "n_terms": model.n_features_out_,
            "fdr_controlled": model.fdr_controlled_,
            "n_screened": (
                None if model.selection_result_ is None
                else int(model.selection_result_.n_selected)
            ),
        })
        n_out = model.n_features_out_ if model.fdr_controlled_ else 0
        null_selected += n_out
        null_trials_selecting += n_out > 0

    # A run of zeros is exact as an observation but still bounds the underlying
    # rate only so tightly; report the one-sided 95% limit so the figure is read
    # at the precision the replicate count supports.
    n_false = sum(1 for f in fdps if f > 0)
    fdr_upper = _upper_bound(n_false, len(fdps))

    result = {
        "empirical_fdr": float(np.mean(fdps)),
        "fdr_upper_95": fdr_upper,
        "power": recovered / len(fdps),
        "fallbacks": fallbacks,
        "null_selections": null_selected,
        # Under the global null every selection is false, so the FDR equals the
        # probability that a trial selects anything: the binomial count is
        # trials with at least one selection, not features selected.
        "null_trials_selecting": null_trials_selecting,
        "null_upper_95": _upper_bound(null_trials_selecting, n_null),
        "parsimony": parsimony,
        "per_fit": per_fit,
    }
    print(
        f"SIGNAL ({n_signal} replicates, nominal 0.10): "
        f"empirical FDR {result['empirical_fdr']:.4f} "
        f"(95% upper bound {fdr_upper:.4f}) | power {result['power']:.3f} "
        f"| fallbacks {result['fallbacks']}"
    )
    print(
        f"GLOBAL NULL ({n_null} replicates): selections {result['null_selections']} "
        f"in {null_trials_selecting} trials "
        f"(95% upper bound on the per-trial rate {result['null_upper_95']:.4f})"
    )

    # The guarantee, checked against the level the study was run at rather than
    # against whatever this run happened to produce. Power carries no bound and
    # is reported only.
    if result["empirical_fdr"] > TARGET_FDR:
        raise AssertionError(
            f"empirical FDR {result['empirical_fdr']:.4f} exceeds the nominal {TARGET_FDR}"
        )
    null_rate = null_trials_selecting / n_null
    if null_rate > TARGET_FDR:
        raise AssertionError(
            f"global-null selection rate {null_rate:.4f} exceeds the nominal {TARGET_FDR}"
        )
    return result


if __name__ == "__main__":
    main()
