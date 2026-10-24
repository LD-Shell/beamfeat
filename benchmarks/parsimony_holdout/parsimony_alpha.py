"""The forward step's chance check, ``parsimony_alpha``, on the calibration designs.

Same designs, replicates, records and tables as ``calibration.py`` (whose
docstring defines them), at q = 0.1, with four settings:

    forward                 parsimony="forward" (the default)
    forward_alpha           the same with parsimony_alpha=0.05
    conditional_0.3         parsimony_holdout=0.3, certify_test="conditional"
    conditional_0.3_alpha   the same with parsimony_alpha=0.05

The search does not depend on the setting, so it runs once per replicate and
is reused by the four fits, checked against an unmemoised refit on the first
``--check-seeds`` replicates of every cell as in ``calibration.py``.

Run from the repository root, with this checkout's ``src`` importable:

    python benchmarks/parsimony_holdout/parsimony_alpha.py      # 50 seeds, resumable
    python benchmarks/parsimony_holdout/parsimony_alpha.py --tabulate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calibration  # noqa: E402

calibration.SETTINGS = {
    "forward": {},
    "forward_alpha": {"parsimony_alpha": 0.05},
    "conditional_0.3": {"parsimony_holdout": 0.3, "certify_test": "conditional"},
    "conditional_0.3_alpha": {"parsimony_holdout": 0.3, "certify_test": "conditional",
                              "parsimony_alpha": 0.05},
}
calibration.LEVELS = (0.1,)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=50, help="replicates per cell (default 50)")
    parser.add_argument("--check-seeds", type=int, default=1,
                        help="replicates per cell refitted without the search memo (default 1)")
    parser.add_argument("--out", type=Path, default=HERE / "results_alpha")
    parser.add_argument("--tabulate", action="store_true", help="only rebuild the tables from the records")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run although tracked files differ from the commit")
    args = parser.parse_args(argv)
    if args.tabulate:
        calibration.tabulate(args.out)
        return

    warnings.filterwarnings("ignore")
    logging.getLogger("beamfeat").setLevel(logging.ERROR)
    info = calibration.provenance()
    if info["tracked_changes"] and not args.allow_dirty:
        sys.exit("tracked files differ from the commit; commit first, or pass --allow-dirty")
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    memo = calibration.SearchMemo()
    started = time.time()
    print(f"parsimony_alpha: {args.seeds} seeds per cell, settings {list(calibration.SETTINGS)}")
    print(json.dumps(info), flush=True)
    for cell in calibration.CELLS:
        cell_started = time.perf_counter()
        checked = calibration.run_cell(cell, args.seeds, args.check_seeds, parts, memo)
        print(f"{calibration.cell_name(*cell)}: done in {time.perf_counter() - cell_started:.0f} s, "
              f"{checked} fits matched without the memo", flush=True)
    calibration.tabulate(args.out, {
        **info,
        "seeds": args.seeds,
        "check_seeds": args.check_seeds,
        "settings": calibration.SETTINGS,
        "levels": list(calibration.LEVELS),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "wall_seconds": round(time.time() - started),
    })


if __name__ == "__main__":
    main()
