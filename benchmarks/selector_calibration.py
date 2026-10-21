"""Selector-level calibration: realised FDR, power, and the pure-noise stress.

Measures every selector-level number quoted in docs/benchmarks.md,
docs/guarantees.md and the paper, so that none of them is asserted without a
script that produces it. Three studies; the first two on Gaussian designs
(n=300, 25 candidate features, 5 of them signal, effect 3.0, unit noise):

1. Permutation selector, Benjamini-Hochberg and Benjamini-Yekutieli, at
   nominal FDR 0.05 / 0.10 / 0.20 -- realised FDR and power over 100 trials.
2. Fixed-X knockoffs at nominal 0.20, with offset=1 (knockoff+, controls FDR)
   and offset=0 (controls only a modified FDR), over the same 100 designs.
3. A global-null stress on the full pipeline: 100 pure-noise problems (500
   rows, six uniform inputs, a Gaussian target independent of them), fitted
   by ``BeamFeatRegressor`` at target FDR 0.10, BH against BY. Under a global null every selection is false, so the mean
   false discovery proportion equals the fraction of trials selecting
   anything; this is the measurement behind the BY estimator default.

Every figure is reported with its standard error, which at 100 trials runs
0.004 to 0.016 on a realised FDR. Read them at that precision; a difference of
a few trials in study 3 is still not statistically meaningful.

Run:  python benchmarks/selector_calibration.py
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

N_TRIALS = 100
NOMINAL = (0.05, 0.10, 0.20)
SEED_BASE = 3000
NULL_SEED_BASE = 50_000

# Design constants, also used to derive the bound each procedure must respect.
N = 300
N_SIGNAL = 5
N_NOISE = 20
EFFECT = 3.0


def gaussian_design(seed: int, n: int = N, n_signal: int = N_SIGNAL,
                    n_noise: int = N_NOISE, effect: float = EFFECT):
    """Independent Gaussian candidates; the first ``n_signal`` are real."""
    rng = np.random.default_rng(seed)
    p = n_signal + n_noise
    features = rng.standard_normal((n, p))
    coefficients = np.zeros(p)
    coefficients[:n_signal] = effect
    target = features @ coefficients + rng.standard_normal(n)
    truth = np.zeros(p, dtype=bool)
    truth[:n_signal] = True
    return features, target, truth


def _fdp(selected: np.ndarray, truth: np.ndarray) -> float:
    if selected.size == 0:
        return 0.0
    return float(np.sum(~truth[selected])) / selected.size


def _power(selected: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sum(truth[selected])) / float(truth.sum())


def permutation_calibration() -> dict:
    """Realised FDR against the bound each procedure is required to respect.

    Benjamini-Hochberg controls the FDR at ``q * m0/m``, where ``m0`` of the
    ``m`` candidates are null; here that is ``q * 20/25``. Benjamini-Yekutieli
    is stricter by a harmonic factor, so the practical claim for it is simply
    that the realised rate stays at or below nominal. Both bounds are derived
    from the design constants above rather than written in by hand, and the
    Monte Carlo error is reported so the figures are read at the precision the
    trial count supports.
    """
    from beamfeat.selection import PermutationSelector

    table = {}
    print(f"Permutation selector, {N_TRIALS} trials, Gaussian design "
          f"{N}x{N_SIGNAL + N_NOISE} with {N_SIGNAL} signals")
    print(f"  {'corr':>4} {'nominal':>8} {'realised':>9} {'SE':>7} "
          f"{'95% CI':>18} {'bound':>7} {'power':>6}  ok")
    for correction in ("bh", "by"):
        for nominal in NOMINAL:
            fdps, powers = [], []
            for trial in range(N_TRIALS):
                features, target, truth = gaussian_design(SEED_BASE + trial)
                result = PermutationSelector(
                    target_fdr=nominal, random_state=trial, correction=correction
                ).select(features, target)
                fdps.append(_fdp(result.selected, truth))
                powers.append(_power(result.selected, truth))

            fdp = np.asarray(fdps)
            realised = float(fdp.mean())
            se = float(fdp.std(ddof=1) / np.sqrt(N_TRIALS))
            lo, hi = realised - 1.96 * se, realised + 1.96 * se
            bound = nominal * N_NOISE / (N_SIGNAL + N_NOISE) if correction == "bh" else nominal
            ok = lo <= bound          # bound lies at or above the interval
            table[(correction, nominal)] = {
                "realised": realised, "se": se, "ci": (lo, hi),
                "power": float(np.mean(powers)), "bound": bound, "within": ok,
            }
            print(f"  {correction.upper():>4} {nominal:>8.2f} {realised:>9.4f} {se:>7.4f} "
                  f"[{lo:>7.4f},{hi:>7.4f}] {bound:>7.3f} {np.mean(powers):>6.2f}"
                  f"  {'y' if ok else 'NO'}")

    breached = [k for k, v in table.items() if not v["within"]]
    if breached:
        raise AssertionError(f"realised FDR above the derived bound for {breached}")
    return table


def knockoff_calibration() -> dict:
    from beamfeat.selection import KnockoffSelector

    table = {}
    print(f"\nFixed-X knockoffs at nominal 0.20 ({N_TRIALS} trials, same designs)")
    print(f"  {'offset':>10} {'realised':>9} {'power':>7}   controls")
    for offset, controls in ((1, "FDR (knockoff+)"), (0, "modified FDR only")):
        fdps, powers = [], []
        for trial in range(N_TRIALS):
            features, target, truth = gaussian_design(SEED_BASE + trial)
            result = KnockoffSelector(
                target_fdr=0.20, random_state=trial, offset=offset
            ).select(features, target)
            fdps.append(_fdp(result.selected, truth))
            powers.append(_power(result.selected, truth))
        realised, power = float(np.mean(fdps)), float(np.mean(powers))
        table[offset] = (realised, power)
        print(f"  {offset:>10} {realised:>9.3f} {power:>7.2f}   {controls}")
    return table


def pure_noise_stress() -> dict:
    """Global null on the full pipeline: BH against BY, no signal at all."""
    from beamfeat import BeamFeatRegressor

    table = {}
    print(f"\nGlobal-null stress on the full pipeline ({N_TRIALS} trials, no signal)")
    print(f"  {'correction':>10} {'trials selecting':>17} {'mean FDP':>9}")
    for correction in ("bh", "by"):
        selected_any = 0
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(NULL_SEED_BASE + trial)
            X = rng.uniform(1, 6, (500, 6))
            y = rng.normal(0, 1, 500)
            model = BeamFeatRegressor(
                max_depth=2, beam_width=25, target_fdr=0.10,
                selection_correction=correction, random_state=trial,
            ).fit(X, y)
            if model.fdr_controlled_ and model.n_features_out_ > 0:
                selected_any += 1
        mean_fdp = selected_any / N_TRIALS
        table[correction] = (selected_any, mean_fdp)
        print(f"  {correction.upper():>10} {f'{selected_any}/{N_TRIALS}':>17} {mean_fdp:>9.2f}")
    return table


def main() -> dict:
    permutation = permutation_calibration()
    knockoff = knockoff_calibration()
    null = pure_noise_stress()
    return {"permutation": permutation, "knockoff": knockoff, "pure_noise": null}


if __name__ == "__main__":
    main()
