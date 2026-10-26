"""Do the simpler settings print the weak term? ``max_features=200`` (four
times the default's final list) and ``beam_width=200``, each with
``parsimony_alpha=0.05``, against ``parsimony_alpha=0.05`` alone and with
``residual_slots=10``, on the three weak-term cells where the slots did best,
five seeds each. Same data and criterion as ``run.py``.

Usage, from this folder::

    ../release_comparison/venvs/0.8.0/bin/python -I alternatives.py > results/alternatives.jsonl
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import sympy  # noqa: E402
from run import _plain, _same, r2, weak_data  # noqa: E402

from beamfeat import BeamFeatRegressor  # noqa: E402

SETTINGS = {
    "alpha": {"parsimony_alpha": 0.05},
    "max_features_200_alpha": {"max_features": 200, "parsimony_alpha": 0.05},
    "beam_200_alpha": {"beam_width": 200, "parsimony_alpha": 0.05},
    "residual_alpha": {"residual_slots": 10, "parsimony_alpha": 0.05},
}
CELLS = [(4, 2000, 0.001), (4, 5000, 0.001), (10, 5000, 0.001)]

x = sympy.symbols("x0:10")
for seed in range(5):
    for p, n, noise in CELLS:
        X, y, X_test, truth = weak_data(p, n, noise, seed)
        for name, kwargs in SETTINGS.items():
            started = time.perf_counter()
            model = BeamFeatRegressor(random_state=seed, **kwargs).fit(X, y)
            terms = [_plain(t) for t in model.to_sympy()] if model.features_ else []
            logs = [t for t in terms if _same(t, sympy.log(x[3]))]
            strong = [t for t in terms if _same(t, x[0] * x[1] / x[2])]
            print(json.dumps({"cell": f"p{p}_n{n}", "seed": seed, "setting": name,
                              "searched": "log(x3)" in model.search_result_.names,
                              "exact": len(terms) == 2 and bool(logs) and bool(strong),
                              "terms": len(terms), "r2": round(r2(model.predict(X_test), truth), 5),
                              "s": round(time.perf_counter() - started, 1),
                              "f": list(model.feature_formulas_)[:4]}), flush=True)
