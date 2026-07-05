#!/usr/bin/env python3
# gen_matrix.py - seeded, reproducible generators for every synthetic
# benchmark matrix used across Steps 3-9 (big, huge, giant, banded).
#
# Fixes two reproducibility gaps found during review:
#   - giant.mtx / banded.mtx / huge.mtx had no generator saved anywhere;
#     they existed only as pre-built .mtx blobs with no way to reproduce
#     or regenerate them.
#   - the original generator (this file) used an unseeded RNG, so even
#     re-running it produced a different matrix every time.
#
# Usage:
#   ./gen_matrix.py                # (re)generate all four
#   ./gen_matrix.py giant.mtx banded.mtx   # regenerate specific ones

import random
import sys

SPECS = {
    "big.mtx":    dict(n=1000,  nnz=5000,   kind="random", seed=1001),
    "huge.mtx":   dict(n=10000, nnz=50000,  kind="random", seed=1002),
    "giant.mtx":  dict(n=50000, nnz=500000, kind="random", seed=1003),
    "banded.mtx": dict(n=50000, nnz=500000, kind="banded", bandwidth=60, seed=1004),
}


def gen_random(n, target_nnz, rng):
    seen = set()
    out = []
    while len(out) < target_nnz:
        r = rng.randint(1, n)
        c = rng.randint(1, n)
        if (r, c) in seen:
            continue
        seen.add((r, c))
        out.append((r, c, rng.uniform(0.1, 1.0)))
    return out


def gen_banded(n, target_nnz, bandwidth, rng):
    # columns clustered within +/- bandwidth of the row index -> banded
    # structure, deduped like gen_random (the original ad-hoc generator
    # for banded.mtx did NOT dedupe, leaving 18,375 duplicate coordinate
    # entries that scipy silently sums on read and our C reader doesn't).
    seen = set()
    out = []
    while len(out) < target_nnz:
        r = rng.randint(1, n)
        lo = max(1, r - bandwidth)
        hi = min(n, r + bandwidth)
        c = rng.randint(lo, hi)
        if (r, c) in seen:
            continue
        seen.add((r, c))
        out.append((r, c, rng.uniform(0.1, 1.0)))
    return out


def write_mtx(path, n, nnz, pairs):
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{n} {n} {nnz}\n")
        for r, c, v in pairs:
            f.write(f"{r} {c} {v:.4f}\n")


def main():
    targets = sys.argv[1:] or list(SPECS.keys())
    for name in targets:
        spec = SPECS[name]
        rng = random.Random(spec["seed"])
        if spec["kind"] == "random":
            pairs = gen_random(spec["n"], spec["nnz"], rng)
        else:
            pairs = gen_banded(spec["n"], spec["nnz"], spec["bandwidth"], rng)
        write_mtx(name, spec["n"], spec["nnz"], pairs)
        print(f"[wrote] {name}  n={spec['n']}  nnz={spec['nnz']}  kind={spec['kind']}  seed={spec['seed']}")


if __name__ == "__main__":
    main()
