"""Speed and peak memory: every fit in a fresh process, settings interleaved, order rotated.

Each fit runs `harness.py --perf` in the version's own environment
(venvs/<version>/bin/python, PROVENANCE.md), with one BLAS thread. The child
resets its peak-RSS counter after loading the data and reports the wall
time, CPU time and peak memory of the fit alone. Resumable: fits already in
the output file are skipped.

    python perf.py                                                   # results/perf.jsonl
    python perf.py --settings 0.4.2:default,0.5.0:default perf_042.jsonl
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSIONS = ("0.4.2", "0.5.0", "0.6.0", "0.6.1")
PY = {v: HERE / "venvs" / v / "bin" / "python" for v in VERSIONS}
SETTINGS = [("0.5.0", "default"), ("0.6.0", "default"), ("0.6.0", "no_fitted_ops"), ("0.6.0", "magnitude_slots"),
            ("0.6.0", "input_test"), ("0.6.0", "conditional_retest"), ("0.6.0", "bh"),
            ("0.6.1", "default"), ("0.6.1", "parsimony_alpha"), ("0.6.1", "alpha_input_test"), ("0.6.1", "input_test")]
ITEMS = ["designs/planted_snr3_n1000_k20:0", "stress/triple_unif_p100:0", "feynman/coulomb:0", "suite/diabetes:0",
         "suite/california_housing:0", "highdim/tecator:0", "highdim/superconduct:0", "highdim/riboflavin:0"]
out = HERE / "results" / "perf.jsonl"
if len(sys.argv) > 2 and sys.argv[1] == "--settings":
    SETTINGS = [tuple(s.split(":")) for s in sys.argv[2].split(",")]
    out = HERE / "results" / sys.argv[3]
done = set()
if out.exists():
    done = {(r["item"], r["version"], r["config"], r["rep"]) for r in map(json.loads, out.open())}
with out.open("a") as handle:
    for rep in range(3):
        for item in ITEMS:
            order = SETTINGS[rep * 3 % len(SETTINGS):] + SETTINGS[:rep * 3 % len(SETTINGS)]
            for version, config in order:
                if (item, version, config, rep) in done:
                    continue
                result = subprocess.run([str(PY[version]), "-I", str(HERE / "harness.py"), "--version", version,
                                         "--perf", item, "--config", config], capture_output=True, text=True,
                                        env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
                try:
                    row = json.loads(result.stdout.strip().splitlines()[-1])
                except Exception:
                    row = {"version": version, "item": item, "config": config, "error": result.stderr[-300:]}
                row["rep"] = rep
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(item, version, config, rep, row.get("seconds"), row.get("fit_mb"), flush=True)
