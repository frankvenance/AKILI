#!/usr/bin/env python3
# step9_compare.py - Step 9: honest comparison against a real production
# library (scipy.sparse), not just against our own sequential baseline.
#
# Computes the SAME operation (C = A @ A^T) with scipy's CSR sparse matmul
# (compiled C, single-threaded) and reports it next to our OpenMP kernel's
# best time from results.csv (Step 8). This answers the question our
# earlier "speedup vs our own naive baseline" numbers could not: how do we
# stack up against what engineers actually reach for?
#
# Usage: .venv/bin/python3 step9_compare.py matrices/*.mtx giant.mtx banded.mtx big.mtx huge.mtx

import sys
import csv
import time
import numpy as np
import scipy.io
import scipy.sparse as sp

RUNS = 5


def structural_nnz(A):
    # The TRUE structural sparsity pattern (Gustavson's "union of touched
    # columns" -- what our C kernel reports as nnz(C)) computed via scipy
    # itself: cast to a positive-only pattern matrix first, so every
    # accumulated product is >=1 and no entry can land on exact 0.0 by
    # cancellation. scipy's matmul then can't silently drop anything, and
    # its stored nnz IS the structural count -- this makes the manual
    # investigation into cant.mtx's discrepancy (see README) an automatic,
    # standing check across every matrix instead of a one-off.
    Apat = A.astype(bool).astype(np.float64).tocsr()
    ATpat = Apat.T.tocsr()
    Cpat = Apat @ ATpat
    return Cpat.nnz


def bench_scipy(path):
    A = scipy.io.mmread(path).tocsr()
    AT = A.T.tocsr()

    # warmup
    C = A @ AT
    struct_nnz = structural_nnz(A)

    best = float("inf")
    for _ in range(RUNS):
        t0 = time.perf_counter()
        A @ AT
        t = time.perf_counter() - t0
        best = min(best, t)

    return A.shape[0], A.nnz, C.nnz, struct_nnz, best


def load_our_results(csv_path="results.csv"):
    ours = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ours[row["matrix"]] = row
    return ours


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: step9_compare.py <matrix.mtx> [...]", file=sys.stderr)
        sys.exit(1)

    ours = load_our_results()

    rows = []
    print(f"{'Matrix':<16}{'nnz(A)':>10}{'scipy nnz(C)':>13}{'struct nnz(C)':>14}{'ours nnz(C)':>12}{'match':>7}{'scipy(s)':>12}{'ours(s)':>12}{'ratio':>8}")
    print("-" * 108)
    for path in paths:
        name = path.split("/")[-1]
        m, nnzA, nnzC_scipy, nnzC_struct, t_scipy = bench_scipy(path)

        row = ours.get(name)
        t_ours = float(row["best_s"]) if row else None
        nnzC_ours = int(row["nnz_C"]) if row else None
        ratio = (t_scipy / t_ours) if t_ours else None
        match = "OK" if (nnzC_ours is not None and nnzC_ours == nnzC_struct) else ("n/a" if nnzC_ours is None else "DIFF")

        ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"
        ours_str = f"{t_ours:.6f}" if t_ours is not None else "n/a"
        print(f"{name:<16}{nnzA:>10}{nnzC_scipy:>13}{nnzC_struct:>14}{str(nnzC_ours):>12}{match:>7}{t_scipy:>12.6f}{ours_str:>12}{ratio_str:>8}")

        rows.append({
            "matrix": name, "m": m, "nnz_A": nnzA,
            "nnz_C_scipy_valuebased": nnzC_scipy, "nnz_C_structural": nnzC_struct,
            "nnz_C_ours": nnzC_ours, "structural_match": match,
            "scipy_s": t_scipy, "ours_s": t_ours, "ratio_scipy_over_ours": ratio,
        })

    with open("compare_scipy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nratio > 1x  => our OpenMP kernel is FASTER than scipy")
    print("ratio < 1x  => scipy is FASTER than our kernel")
    print("\n'struct nnz(C)' is the true structural pattern (positive-only")
    print("matmul, so scipy can't drop cancelled-to-zero entries) -- this")
    print("should always equal 'ours nnz(C)' ('match' column). 'scipy")
    print("nnz(C)' is scipy's normal value-based result, which can be")
    print("lower when real numeric cancellation occurs (see cant.mtx).")
    print("\nNote: scipy's CSR@CSR matmul is single-threaded compiled C.")
    print("Our kernel uses up to 4 OpenMP threads on this machine - part")
    print("of any advantage comes from parallelism itself, not just the")
    print("algorithm. Wrote compare_scipy.csv")


if __name__ == "__main__":
    main()
