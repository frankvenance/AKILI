# SpGEMM Benchmark Report

Computed `C = A * A^T` with an auto-tuned, adaptive-sort,
OpenMP SpGEMM. All results verified against a sequential
reference. Machine: 4 logical threads.

| Matrix | m | nnz(A) | nnz(C) | Fill | Work(iters) | CV | Config | Seq(s) | Best(s) | Speedup | GFLOP/s | OK |
|---|--:|--:|--:|--:|--:|--:|:--:|--:|--:|--:|--:|:--:|
| test.mtx | 5 | 8 | 11 | 1.38x | 14 | 0.27 | 2T/dyn | 0.0000 | 0.0000 | 0.49x | 0.004 | OK |

## Notes

- **Speedup** is best auto-tuned parallel time vs the sequential
  insertion-sort baseline (Step 1), so it captures BOTH the
  algorithmic win (radix/adaptive sort) and parallel scaling.
- **Config** is chosen by a zero-run heuristic: use all cores
  when work >= 200000 inner iters (calibrated empirically on this
  machine, cached in .spgemm_autotune_cache), else 2 cores; dynamic schedule.
- Correctness verified elementwise (col indices + values, tol 1e-9).
