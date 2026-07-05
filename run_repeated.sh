#!/usr/bin/env bash
# run_repeated.sh - Step 8 answers "what's the speedup" with a single
# invocation's number, but repeated full runs (empirical sweep + best-of-5
# timing, each from a cold process) vary by several percent run-to-run on
# a laptop (thermal throttling, scheduler noise, no CPU pinning). This
# script quantifies that noise instead of reporting one run as if exact:
# it re-runs the full Step 8 report N times and computes mean +/- stddev
# of the geometric-mean speedup.
#
# Usage: ./run_repeated.sh [N] [matrix files...]
set -euo pipefail

N="${1:-5}"
shift || true
MATRICES=("$@")
if [ ${#MATRICES[@]} -eq 0 ]; then
    MATRICES=(matrices/*.mtx giant.mtx banded.mtx big.mtx huge.mtx)
fi

if [ ! -x ./step8 ]; then
    gcc -O3 -Wall -fopenmp -march=native -o step8 step8_report.c mmio.c -lm
fi

echo "Running Step 8 full report $N times over ${#MATRICES[@]} matrices..."
vals=()
for i in $(seq 1 "$N"); do
    out=$(./step8 "${MATRICES[@]}" 2>&1)
    gm=$(echo "$out" | grep -oP 'Geometric-mean speedup: \K[0-9.]+')
    ok=$(echo "$out" | grep -oP 'Correct: \K[0-9]+/[0-9]+')
    echo "  run $i: geomean=${gm}x  correct=${ok}"
    vals+=("$gm")
done

python3 -c "
import statistics as s, sys
vals = [float(x) for x in sys.argv[1:]]
mean = s.mean(vals)
sd = s.stdev(vals) if len(vals) > 1 else 0.0
print()
print(f'Over {len(vals)} full runs: mean={mean:.3f}x  stddev={sd:.3f}  ({sd/mean*100:.1f}% of mean)')
print(f'range: {min(vals):.3f}x - {max(vals):.3f}x')
" "${vals[@]}"
