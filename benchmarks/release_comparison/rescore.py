"""Re-score the splits on which some model left held-out rows without a prediction.

beamfeat's score() refuses when a fitted expression is undefined on a test row
(predict gives NaN there). On those splits every setting is refitted, its
predictions saved, and R² is then computed on the rows every setting predicts,
so the settings of a split are compared on the same rows.

    venvs/0.6.0/bin/python -I rescore.py fit 0.6.0      # each version in turn
    venvs/0.6.1/bin/python -I rescore.py combine
"""
import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "rescore"  # saved predictions; not kept in the repository
ITEMS = [("highdim", "communities:0"), ("highdim", "communities:2"), ("suite", "california_housing:4")]


def fit(version):
    import harness as h

    OUT.mkdir(parents=True, exist_ok=True)

    def watch():
        while True:
            if h._status("VmRSS") > 3000:
                print("memory guard: RSS over 3000 MB", flush=True)
                os._exit(3)
            time.sleep(0.25)

    threading.Thread(target=watch, daemon=True).start()
    for study, item in ITEMS:
        X, y, Xt, yt, problem, base, _ = h.load_item(study, item)
        np.save(OUT / f"truth__{study}__{item}.npy", yt)
        for config, kwargs in h.CONFIGS[version].items():
            path = OUT / f"{version}__{config}__{study}__{item}.npy"
            if path.exists():
                continue
            if version == "0.6.0" and (item.split(":")[0], config) in h.HEAVY:
                continue  # passes 7 GB without the 0.6.1 fix
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = h.estimator(problem)(**base, **kwargs).fit(X, y)
                np.save(path, model.predict(Xt))
            print(version, config, study, item, flush=True)


def combine():
    from sklearn.metrics import r2_score

    rows = []
    for study, item in ITEMS:
        truth = np.load(OUT / f"truth__{study}__{item}.npy")
        preds = {tuple(p.name.split("__")[:2]): np.load(p)
                 for p in OUT.glob(f"*__*__{study}__{item}.npy") if not p.name.startswith("truth__")}
        common = np.all([np.isfinite(v) for v in preds.values()], axis=0)
        for (version, config), v in sorted(preds.items()):
            rows.append({"study": study, "item": item, "version": version, "config": config,
                         "score": float(r2_score(truth[common], v[common])),
                         "own_undefined": int(np.count_nonzero(~np.isfinite(v))),
                         "rows_dropped": int(np.count_nonzero(~common)), "rows": int(len(truth))})
    (HERE / "results" / "rescore.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['item']:22s} {r['version']} {r['config']:20s} R² {r['score']:.4f}  "
              f"own undefined {r['own_undefined']}, dropped {r['rows_dropped']} of {r['rows']}")


if __name__ == "__main__":
    fit(sys.argv[2]) if sys.argv[1] == "fit" else combine()
