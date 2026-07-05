// autotune_calib.h - self-calibrating HT/thread-count crossover.
//
// Step 7/8's heuristic decides: use ALL logical threads only once
// there's "enough work" (inner iterations), else stick to 2 physical
// cores (thread-spawn overhead dominates below that). That crossover
// depends on real hardware (SMT efficiency, memory bandwidth,
// thread-spawn cost) - it is NOT a portable constant. It used to be
// hardcoded as 5,000,000 iters, a number calibrated by hand on one
// Intel i7-7500U and silently wrong on any other machine.
//
// This header replaces the constant with a one-time empirical
// calibration: a synthetic scatter/accumulate microbenchmark that
// mimics the real kernel's memory-access pattern (per-thread private
// workspace, indirect indexing, parallel-for over row-like chunks)
// without needing real matrix data, run at increasing work sizes until
// using all threads beats using 2. Result is cached to disk so the
// sweep only runs once per machine.
#ifndef AUTOTUNE_CALIB_H
#define AUTOTUNE_CALIB_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define CALIB_CACHE_FILE ".spgemm_autotune_cache"
#define CALIB_ACC_SIZE   65536   // per-thread working set (doubles) > L2

#define CALIB_REPS 7 // best-of-N to survive OS/scheduler jitter at ms scale

static double calib_bench_once(long work, int T, const int *idx, int need,
                                double *accs) {
    int rows = 1000;
    long per_row = work / rows;
    if (per_row < 1) per_row = 1;

    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        double *acc = accs + (size_t)tid * CALIB_ACC_SIZE;
        #pragma omp for schedule(dynamic, 16)
        for (int r = 0; r < rows; r++) {
            for (long k = 0; k < per_row; k++) {
                acc[idx[k % need]] += 1.0;
            }
        }
    }
    return omp_get_wtime() - t0;
}

// best-of-CALIB_REPS: a single timing at millisecond scale is dominated
// by OS/scheduler jitter (observed 4x swings run-to-run on this laptop
// with only 1 rep) -- min-of-N is the same noise-reduction convention
// the rest of this project uses for its own benchmarks.
static double calib_bench(long work, int T) {
    long per_row_cap = work / 1000;
    int need = (int)(per_row_cap < 100000 ? per_row_cap : 100000);
    if (need < 1) need = 1;
    int *idx = malloc((size_t)need * sizeof(int));
    unsigned seed = 12345u;
    for (int i = 0; i < need; i++) {
        seed = seed * 1103515245u + 12345u;
        idx[i] = (int)(seed % CALIB_ACC_SIZE);
    }
    double *accs = calloc((size_t)T * CALIB_ACC_SIZE, sizeof(double));

    double best = 1e18;
    for (int r = 0; r < CALIB_REPS; r++) {
        double t = calib_bench_once(work, T, idx, need, accs);
        if (t < best) best = t;
    }

    free(accs);
    free(idx);
    return best;
}

static int calib_load_cache(int max_t, long *out_threshold) {
    FILE *f = fopen(CALIB_CACHE_FILE, "r");
    if (!f) return 0;
    int found = 0, cm; long ct;
    while (fscanf(f, "max_threads=%d threshold=%ld\n", &cm, &ct) == 2)
        if (cm == max_t) { *out_threshold = ct; found = 1; }
    fclose(f);
    return found;
}

static void calib_save_cache(int max_t, long threshold) {
    FILE *f = fopen(CALIB_CACHE_FILE, "a");
    if (!f) return;
    fprintf(f, "max_threads=%d threshold=%ld\n", max_t, threshold);
    fclose(f);
}

static int long_cmp(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}

// One HT-crossover sweep: smallest work size where max_t beats 2T by
// a clear margin. best-of-CALIB_REPS per point still isn't enough to
// fully tame OS-scheduler jitter at millisecond scale (observed on a
// shared/virtualized host) -- get_ht_threshold() runs this 3x and
// takes the median, below.
static long calib_sweep_once(int max_t) {
    long sizes[] = {200000, 500000, 1000000, 2000000, 5000000,
                     10000000, 20000000, 50000000};
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));
    long threshold = sizes[n - 1]; // default: never use HT if no crossover found
    for (int i = 0; i < n; i++) {
        double t2  = calib_bench(sizes[i], 2);
        double tHT = calib_bench(sizes[i], max_t);
        fprintf(stderr, "[autotune]   work=%-10ld 2T=%.5fs  %dT=%.5fs%s\n",
                sizes[i], t2, max_t, tHT, tHT < t2 * 0.98 ? "  <- HT wins" : "");
        if (tHT < t2 * 0.98) { threshold = sizes[i]; break; }
    }
    return threshold;
}

// Returns a work-iteration threshold above which using max_t threads
// beats using 2 threads on THIS machine. Calibrated empirically on
// first call (median of 3 independent sweeps), cached to disk (keyed
// by max_t) on every call after.
static long get_ht_threshold(int max_t) {
    if (max_t < 4) return 0; // no HT tier above 2 physical cores to gate

    long cached;
    if (calib_load_cache(max_t, &cached)) return cached;

    fprintf(stderr, "[autotune] no cached calibration for max_threads=%d -- "
                     "running one-time HT crossover sweep (3 passes, median)...\n", max_t);

    long results[3];
    for (int p = 0; p < 3; p++) {
        fprintf(stderr, "[autotune] pass %d/3:\n", p + 1);
        results[p] = calib_sweep_once(max_t);
    }
    qsort(results, 3, sizeof(long), long_cmp);
    long threshold = results[1]; // median

    fprintf(stderr, "[autotune] sweep results: %ld, %ld, %ld -> median threshold=%ld "
                     "for max_threads=%d (cached in %s)\n",
            results[0], results[1], results[2], threshold, max_t, CALIB_CACHE_FILE);
    calib_save_cache(max_t, threshold);
    return threshold;
}

#endif
