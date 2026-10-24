"""Equicorrelated against CI knockoffs: realised FDR and power of KnockoffSelector.

Fixed-X knockoffs at nominal FDR 0.1, with offset=1 (knockoff+) and offset=0,
over the same designs for both ``s_method`` values. Designs:

  block rho     1,000 rows, 100 Gaussian columns in pairs correlated rho
                (0.5, 0.7, 0.9), 20 of them signal.
  factor        1,000 rows, 100 columns loading on three shared factors plus
                independent noise, 20 of them signal.
  constructed   600 rows, eight inputs uniform on [1, 5]; per input x, x^2,
                x^3, log x, sqrt x and 1/x, and the 28 pairwise products
                (76 columns). The target is 14 of the products and 1/x4 plus
                unit noise, so the conditional nulls are well defined. Each
                input's six transforms are near copies, as constructed
                candidates are.

Signal columns and signs are drawn per trial. Power is the share of signal
columns selected; FDR is the mean false discovery proportion. With 50 trials
the standard error of a realised FDR is about 0.01-0.02.

Run:  python benchmarks/knockoff_s_method.py [--trials 50]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from beamfeat.selection import KnockoffSelector

logging.getLogger("beamfeat").setLevel(logging.ERROR)

TARGET_FDR = 0.1
SEED_BASE = 1000


def block(rng, rho, n=1000, p=100, k=20, amplitude=1.1):
    z = rng.standard_normal((n, p))
    x = z.copy()
    x[:, 1::2] = rho * z[:, 0::2] + np.sqrt(1 - rho**2) * z[:, 1::2]
    return _linear_target(rng, x, k, amplitude)


def factor(rng, n=1000, p=100, k=20, n_factors=3, amplitude=1.6):
    loadings = rng.standard_normal((n_factors, p))
    x = rng.standard_normal((n, n_factors)) @ loadings + rng.standard_normal((n, p))
    return _linear_target(rng, x, k, amplitude)


def _linear_target(rng, x, k, amplitude):
    n, p = x.shape
    beta = np.zeros(p)
    signal = rng.choice(p, k, replace=False)
    beta[signal] = amplitude * 4 * np.sqrt(np.log(p) / n) * rng.choice([-1, 1], k)
    return x, x @ beta + rng.standard_normal(n), beta != 0


def constructed(rng, n=600, n_inputs=8):
    raw = rng.uniform(1.0, 5.0, (n, n_inputs))
    columns, names = [], []
    for j in range(n_inputs):
        x = raw[:, j]
        for name, values in (("x", x), ("x^2", x**2), ("x^3", x**3),
                             ("log", np.log(x)), ("sqrt", np.sqrt(x)), ("inv", 1 / x)):
            columns.append(values)
            names.append(f"{name}({j})")
    for a in range(n_inputs):
        for b in range(a + 1, n_inputs):
            columns.append(raw[:, a] * raw[:, b])
            names.append(f"x{a}*x{b}")
    x = np.column_stack(columns)
    signal = [i for i, name in enumerate(names) if "*" in name][::2] + [names.index("inv(4)")]
    coefficients = np.full(len(signal), 0.12)
    coefficients[-1] = 3.0
    truth = np.zeros(x.shape[1], dtype=bool)
    truth[signal] = True
    return x, x[:, signal] @ coefficients + rng.standard_normal(n), truth


def run(make, s_method, offset, trials):
    fdp, power = [], []
    for trial in range(trials):
        rng = np.random.default_rng(SEED_BASE + trial)
        x, y, truth = make(rng)
        selector = KnockoffSelector(target_fdr=TARGET_FDR, construction="fixed", s_method=s_method,
                                    offset=offset, random_state=trial, warn_on_violation=False)
        selected = selector.select(x, y).selected
        fdp.append(float(np.mean(~truth[selected])) if selected.size else 0.0)
        power.append(float(np.mean(np.isin(np.flatnonzero(truth), selected))))
    return float(np.mean(fdp)), float(np.mean(power))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=50)
    trials = parser.parse_args().trials
    designs = {
        "block rho=0.5": lambda rng: block(rng, 0.5),
        "block rho=0.7": lambda rng: block(rng, 0.7),
        "block rho=0.9": lambda rng: block(rng, 0.9),
        "factor": factor,
        "constructed": constructed,
    }
    print(f"fixed-X knockoffs, nominal FDR {TARGET_FDR}, {trials} trials")
    print(f"{'design':15s} {'offset':>6s}  {'equicorrelated FDR / power':>27s}  {'ci FDR / power':>15s}")
    for offset in (1, 0):
        for name, make in designs.items():
            equi = run(make, "equicorrelated", offset, trials)
            ci = run(make, "ci", offset, trials)
            print(f"{name:15s} {offset:>6d}  {equi[0]:>13.3f} / {equi[1]:.3f}  "
                  f"{ci[0]:>7.3f} / {ci[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
