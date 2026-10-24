#!/bin/bash
# Rerun the whole comparison: about 15,000 fits, a few hours with three jobs.
# Needs venvs/<version> for 0.4.2, 0.5.0, 0.6.0 and 0.6.1 (PROVENANCE.md).
# New records go to results/<study>_<version>_<part>.jsonl and take
# precedence over the archived results/<study>.jsonl.gz in report.py.
set -e
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
JOBS=${JOBS:-3}
mkdir -p logs

# Every job stops itself, recording the fit that ran, if its memory passes
# 3 GB (harness.py --limit-mb).
{
  for v in 0.4.2 0.5.0 0.6.0 0.6.1; do
    for part in 1/4 2/4 3/4 4/4; do echo "$v designs $part"; done
    for study in stress feynman suite highdim junk knockoff; do echo "$v $study 1/1"; done
  done
} | xargs -P "$JOBS" -L 1 sh -c \
  'venvs/$0/bin/python -I harness.py --version $0 --study $1 --part $2 > "logs/$0_$1_$(echo $2 | tr / _).log" 2>&1'

# input_test on California housing, alone. 0.6.0's passes 7 GB on every split
# (fixed in 0.6.1); to record that, run it too, with
#   venvs/0.6.0/bin/python -I harness.py --version 0.6.0 --study suite --heavy --limit-mb 7000
venvs/0.6.1/bin/python -I harness.py --version 0.6.1 --study suite --heavy > logs/0.6.1_suite_heavy.log 2>&1

# Splits with held-out rows some model cannot predict, scored on the rows all predict.
for v in 0.4.2 0.5.0 0.6.0 0.6.1; do venvs/$v/bin/python -I rescore.py fit $v; done
venvs/0.6.1/bin/python -I rescore.py combine > logs/rescore.log

# Speed and memory, one fit at a time.
venvs/0.6.1/bin/python -I perf.py > logs/perf.log
venvs/0.6.1/bin/python -I perf.py --settings 0.4.2:default,0.5.0:default perf_042.jsonl > logs/perf_042.log

venvs/0.6.1/bin/python -I report.py > TABLES.md
