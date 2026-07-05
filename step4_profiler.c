// step4_profiler.c
// Goal: Measure exactly WHERE time goes inside SpGEMM.
//       Use perf-compatible counters + manual timing breakdowns.
//       Profile: symbolic pass, numeric pass, prefix sum,
//                sort, memory allocation, cache behavior.
//
// Compile: gcc -O2 -Wall -fopenmp -pg -o step4 step4_profiler.c mmio.c -lm
// Run:     ./step4 giant.mtx
// Perf:    perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses,branch-misses,instructions,cycles ./step4 giant.mtx

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <omp.h>
#include "mmio.h"

//  PROFILING INFRASTRUCTURE
//  Fine-grained timers for every sub-phase of SpGEMM.
//  We measure each phase separately to find the bottleneck.

typedef struct {
    double t_read;          // matrix file I/O
    double t_csc;           // CSR -> CSC conversion
    double t_symbolic;      // symbolic pass (count nnz)
    double t_prefix;        // prefix sum (row_ptr build)
    double t_alloc_output;  // malloc ci[] and cv[]
    double t_numeric;       // numeric pass (compute values)
    double t_sort;          // insertion sort within rows
    double t_scatter;       // scatter acc[] -> cv[]
    double t_total;         // end-to-end
    long   n_sort_ops;      // total insertion sort comparisons
    long   n_scatter;       // total scatter operations
    long   n_inner_ops;     // total inner loop iterations
    int    nnz_input;
    int    nnz_output;
    int    m;
} Profile;

static void profile_print(const Profile *p, int nthreads) {
    double compute = p->t_symbolic + p->t_numeric;
    double overhead = p->t_prefix + p->t_alloc_output +
                      p->t_sort + p->t_scatter;

    printf("\nSPGEMM PROFILE (%d thread(s))\n", nthreads);
    printf("------------------------------------------------------\n");
    printf("  %-20s %10s %10s\n", "Phase", "Time(s)", "% total");
    printf("------------------------------------------------------\n");

    #define ROW(label, t) \
        printf("  %-20s %10.6f %9.2f%%\n", \
               label, t, 100.0*(t)/p->t_total)

    ROW("Matrix read",       p->t_read);
    ROW("CSR -> CSC",        p->t_csc);
    printf("  --\n");
    ROW("Symbolic pass",     p->t_symbolic);
    ROW("Prefix sum",        p->t_prefix);
    ROW("Output malloc",     p->t_alloc_output);
    ROW("Numeric pass",      p->t_numeric);
    ROW("  sort (inner)",    p->t_sort);
    ROW("  scatter",         p->t_scatter);
    printf("  --\n");
    ROW("TOTAL",             p->t_total);
    printf("------------------------------------------------------\n");

    printf("\n  Compute (sym+num):   %.6f sec  (%.1f%%)\n",
           compute, 100.0*compute/p->t_total);
    printf("  Overhead (rest):     %.6f sec  (%.1f%%)\n",
           overhead, 100.0*overhead/p->t_total);

    printf("\n  Work statistics:\n");
    printf("  Inner loop iters:   %ld  (%.1f per output nnz)\n",
           p->n_inner_ops,
           p->nnz_output > 0 ?
           (double)p->n_inner_ops/p->nnz_output : 0.0);
    printf("  Sort comparisons:   %ld  (%.2f per output nnz)\n",
           p->n_sort_ops,
           p->nnz_output > 0 ?
           (double)p->n_sort_ops/p->nnz_output : 0.0);
    printf("  Fill factor:        %.2fx\n",
           (double)p->nnz_output/p->nnz_input);
    printf("  Avg row density:    %.1f nnz/row (input)\n",
           (double)p->nnz_input/p->m);
    printf("  Avg output density: %.1f nnz/row (output)\n",
           (double)p->nnz_output/p->m);
}

//  CSR STRUCTURE

typedef struct {
    int m, n, nnz;
    int    *row_ptr;
    int    *col_idx;
    double *values;
} CSR;

static void csr_free(CSR *M) {
    if (!M) return;
    free(M->row_ptr); free(M->col_idx);
    free(M->values);  free(M);
}

static CSR *read_mm_csr(const char *fname, double *t_out) {
    double t0 = omp_get_wtime();
    FILE *f = fopen(fname,"r");
    if (!f) { fprintf(stderr,"Cannot open %s\n",fname); return NULL; }
    MM_typecode mc;
    if (mm_read_banner(f,&mc) != 0) { fprintf(stderr,"Bad banner\n"); fclose(f); return NULL; }
    int M,N,nz;
    if (mm_read_mtx_crd_size(f,&M,&N,&nz) != 0) { fclose(f); return NULL; }
    int sym=mm_is_symmetric(mc);
    int max_e=sym?2*nz:nz;
    int *I=malloc(max_e*sizeof(int));
    int *J=malloc(max_e*sizeof(int));
    double *V=malloc(max_e*sizeof(double));
    int cnt=0;
    for (int e=0;e<nz;e++) {
        int r,c; double v=1.0;
        if (mm_is_pattern(mc)) { if (fscanf(f,"%d %d",&r,&c)      != 2) break; }
        else                   { if (fscanf(f,"%d %d %lg",&r,&c,&v) != 3) break; }
        r--;c--;
        I[cnt]=r;J[cnt]=c;V[cnt]=v;cnt++;
        if(sym&&r!=c){I[cnt]=c;J[cnt]=r;V[cnt]=v;cnt++;}
    }
    fclose(f); nz=cnt;
    int *rp=calloc(M+1,sizeof(int));
    for(int e=0;e<nz;e++) rp[I[e]+1]++;
    for(int i=0;i<M;i++) rp[i+1]+=rp[i];
    int *ci=malloc(nz*sizeof(int));
    double *cv=malloc(nz*sizeof(double));
    int *cur=calloc(M,sizeof(int));
    for(int e=0;e<nz;e++){
        int pos=rp[I[e]]+cur[I[e]]++;
        ci[pos]=J[e]; cv[pos]=V[e];
    }
    free(I);free(J);free(V);free(cur);
    CSR *A=malloc(sizeof(CSR));
    A->m=M;A->n=N;A->nnz=nz;
    A->row_ptr=rp;A->col_idx=ci;A->values=cv;
    if (t_out) *t_out = omp_get_wtime() - t0;
    return A;
}

static CSR *csr_to_csc(const CSR *A, double *t_out) {
    double t0 = omp_get_wtime();
    int m=A->m,n=A->n,nnz=A->nnz;
    int *cp=calloc(n+1,sizeof(int));
    for(int e=0;e<nnz;e++) cp[A->col_idx[e]+1]++;
    for(int j=0;j<n;j++) cp[j+1]+=cp[j];
    int *ri=malloc(nnz*sizeof(int));
    double *rv=malloc(nnz*sizeof(double));
    int *cur=calloc(n,sizeof(int));
    for(int i=0;i<m;i++)
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            int j=A->col_idx[p];
            int pos=cp[j]+cur[j]++;
            ri[pos]=i; rv[pos]=A->values[p];
        }
    free(cur);
    CSR *B=malloc(sizeof(CSR));
    B->m=n;B->n=m;B->nnz=nnz;
    B->row_ptr=cp;B->col_idx=ri;B->values=rv;
    if (t_out) *t_out = omp_get_wtime() - t0;
    return B;
}

//  THREAD WORKSPACE

typedef struct {
    double *acc;
    int    *occupied;
    int    *cols;
} ThreadWS;

static ThreadWS *alloc_ws(int T, int m) {
    ThreadWS *ws = malloc(T * sizeof(ThreadWS));
    for (int t=0;t<T;t++) {
        ws[t].acc      = calloc(m, sizeof(double));
        ws[t].occupied = calloc(m, sizeof(int));
        ws[t].cols     = malloc(m * sizeof(int));
    }
    return ws;
}

static void free_ws(ThreadWS *ws, int T) {
    for (int t=0;t<T;t++){
        free(ws[t].acc);
        free(ws[t].occupied);
        free(ws[t].cols);
    }
    free(ws);
}

//  INSTRUMENTED SPGEMM
//  Same algorithm as step3b but with fine-grained timers
//  and operation counters inside the hot loops.
//
//  Important: timers use omp_get_wtime() per-thread and
//  we take the MAX across threads (wall time semantics).

static CSR *spgemm_instrumented(const CSR *A, const CSR *AT,
                                 int nthreads, Profile *prof) {
    int m = A->m;
    int chunk = m > 32*nthreads ? 32 : 1;

    ThreadWS *ws = alloc_ws(nthreads, m);
    int *row_nnz = calloc(m, sizeof(int));

    // Per-thread counters (avoid false sharing with padding)
    typedef struct { long val; char pad[56]; } PaddedLong;
    PaddedLong *t_sym_arr  = calloc(nthreads, sizeof(PaddedLong));
    PaddedLong *t_num_arr  = calloc(nthreads, sizeof(PaddedLong));
    PaddedLong *t_sort_arr = calloc(nthreads, sizeof(PaddedLong));
    PaddedLong *t_scat_arr = calloc(nthreads, sizeof(PaddedLong));
    PaddedLong *inner_arr  = calloc(nthreads, sizeof(PaddedLong));
    PaddedLong *sort_arr   = calloc(nthreads, sizeof(PaddedLong));
    PaddedLong *scat_arr   = calloc(nthreads, sizeof(PaddedLong));

    // Symbolic pass
    double t_sym_start = omp_get_wtime();

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int *occupied = ws[tid].occupied;
        int *cols     = ws[tid].cols;
        long my_inner = 0;

        double ts = omp_get_wtime();

        #pragma omp for schedule(dynamic, chunk)
        for (int i=0; i<m; i++) {
            int cnt=0;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    my_inner++;
                    if (!occupied[j]) {
                        occupied[j]=1;
                        cols[cnt++]=j;
                    }
                }
            }
            row_nnz[i]=cnt;
            for (int a=0;a<cnt;a++) occupied[cols[a]]=0;
        }

        t_sym_arr[tid].val  = (long)((omp_get_wtime()-ts)*1e9);
        inner_arr[tid].val  = my_inner;
    }

    prof->t_symbolic = omp_get_wtime() - t_sym_start;

    // Prefix sum
    double t_pre_start = omp_get_wtime();
    int *rp = malloc((m+1)*sizeof(int));
    rp[0]=0;
    for (int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C = rp[m];
    prof->t_prefix = omp_get_wtime() - t_pre_start;

    // Output allocation
    double t_alloc_start = omp_get_wtime();
    int    *ci = malloc(nnz_C * sizeof(int));
    double *cv = malloc(nnz_C * sizeof(double));
    prof->t_alloc_output = omp_get_wtime() - t_alloc_start;

    // Numeric pass
    double t_num_start = omp_get_wtime();

    // Sub-timers inside numeric pass (measured per thread)
    double *t_sort_per_thread = calloc(nthreads, sizeof(double));
    double *t_scat_per_thread = calloc(nthreads, sizeof(double));
    long   *sort_ops_per_thread = calloc(nthreads, sizeof(long));
    long   *scat_ops_per_thread = calloc(nthreads, sizeof(long));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        double *acc      = ws[tid].acc;
        int    *occupied = ws[tid].occupied;
        int    *cols     = ws[tid].cols;

        double my_t_sort=0, my_t_scat=0;
        long my_sort_ops=0, my_scat_ops=0;

        #pragma omp for schedule(dynamic, chunk)
        for (int i=0; i<m; i++) {
            int cnt=0;

            // accumulate
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                double a_ik=A->values[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    if (!occupied[j]){
                        occupied[j]=1;
                        acc[j]=a_ik*AT->values[q];
                        cols[cnt++]=j;
                    } else {
                        acc[j]+=a_ik*AT->values[q];
                    }
                }
            }

            // sort - timed separately
            double ts_sort = omp_get_wtime();
            for (int a=1;a<cnt;a++) {
                int key=cols[a]; int b=a-1;
                while(b>=0&&cols[b]>key){
                    cols[b+1]=cols[b];b--;
                    my_sort_ops++;
                }
                cols[b+1]=key;
            }
            my_t_sort += omp_get_wtime() - ts_sort;

            // scatter - timed separately
            double ts_scat = omp_get_wtime();
            int base=rp[i];
            for (int a=0;a<cnt;a++){
                int j=cols[a];
                ci[base+a]=j;
                cv[base+a]=acc[j];
                acc[j]=0.0;
                occupied[j]=0;
                my_scat_ops++;
            }
            my_t_scat += omp_get_wtime() - ts_scat;
        }

        t_sort_per_thread[tid]    = my_t_sort;
        t_scat_per_thread[tid]    = my_t_scat;
        sort_ops_per_thread[tid]  = my_sort_ops;
        scat_ops_per_thread[tid]  = my_scat_ops;
    }

    prof->t_numeric = omp_get_wtime() - t_num_start;

    // aggregate per-thread sub-timers (take max = wall time)
    prof->t_sort = prof->t_scatter = 0;
    prof->n_sort_ops = prof->n_scatter = 0;
    prof->n_inner_ops = 0;
    for (int t=0;t<nthreads;t++){
        if (t_sort_per_thread[t] > prof->t_sort)
            prof->t_sort = t_sort_per_thread[t];
        if (t_scat_per_thread[t] > prof->t_scatter)
            prof->t_scatter = t_scat_per_thread[t];
        prof->n_sort_ops  += sort_ops_per_thread[t];
        prof->n_scatter   += scat_ops_per_thread[t];
        prof->n_inner_ops += inner_arr[t].val;
    }

    prof->nnz_output = nnz_C;
    prof->nnz_input  = A->nnz;
    prof->m          = m;

    free_ws(ws, nthreads);
    free(row_nnz);
    free(t_sym_arr); free(t_num_arr);
    free(t_sort_arr); free(t_scat_arr);
    free(inner_arr); free(sort_arr); free(scat_arr);
    free(t_sort_per_thread); free(t_scat_per_thread);
    free(sort_ops_per_thread); free(scat_ops_per_thread);

    CSR *C=malloc(sizeof(CSR));
    C->m=m; C->n=m; C->nnz=nnz_C;
    C->row_ptr=rp; C->col_idx=ci; C->values=cv;
    return C;
}

//  ROW WORK DISTRIBUTION ANALYSIS
//  Shows how unevenly work is distributed across rows.
//  This tells us whether dynamic scheduling was the right
//  choice and how much load imbalance exists.

static void analyze_row_distribution(const CSR *A, const CSR *AT) {
    int m = A->m;
    long *row_work = malloc(m * sizeof(long));
    long total=0, max_work=0, min_work=LONG_MAX;

    for (int i=0; i<m; i++) {
        long work=0;
        for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
            int k=A->col_idx[p];
            work += AT->row_ptr[k+1] - AT->row_ptr[k];
        }
        row_work[i]=work;
        total+=work;
        if(work>max_work) max_work=work;
        if(work>0&&work<min_work) min_work=work;
    }

    double mean = (double)total/m;

    // compute variance
    double var=0;
    for(int i=0;i<m;i++){
        double d=(double)row_work[i]-mean;
        var+=d*d;
    }
    var/=m;
    double stddev=sqrt(var);

    // histogram - 8 buckets
    long buckets[8]={0};
    long bmax = max_work+1;
    for(int i=0;i<m;i++){
        int b=(int)(8.0*row_work[i]/bmax);
        if(b>=8) b=7;
        buckets[b]++;
    }

    // count empty rows
    int empty_rows=0;
    for(int i=0;i<m;i++) if(row_work[i]==0) empty_rows++;

    printf("\n=== ROW WORK DISTRIBUTION ANALYSIS ===\n");
    printf("  Total inner iterations: %ld\n", total);
    printf("  Mean work per row:      %.1f\n", mean);
    printf("  Std deviation:          %.1f\n", stddev);
    printf("  CV (stddev/mean):       %.2f  ", stddev/mean);
    if (stddev/mean > 1.0)
        printf("(HIGH imbalance -> dynamic scheduling correct)\n");
    else
        printf("(LOW imbalance -> static scheduling fine)\n");
    printf("  Min work (non-empty):   %ld\n", min_work==LONG_MAX?0:min_work);
    printf("  Max work:               %ld\n", max_work);
    printf("  Max/Mean ratio:         %.1fx\n", mean>0?max_work/mean:0);
    printf("  Empty rows:             %d (%.1f%%)\n",
           empty_rows, 100.0*empty_rows/m);

    printf("\n  Work distribution histogram:\n");
    printf("  %-20s   Count      %%rows\n", "Work range");
    printf("  %s\n", "----------------------------------------");
    for(int b=0;b<8;b++){
        long lo=(long)(b*bmax/8), hi=(long)((b+1)*bmax/8);
        int bar=(int)(40.0*buckets[b]/m);
        printf("  [%6ld - %6ld]  %7ld   %5.1f%% ",
               lo, hi-1, buckets[b], 100.0*buckets[b]/m);
        for(int x=0;x<bar;x++) printf("#");
        printf("\n");
    }

    free(row_work);
}

//  SORT ALGORITHM COMPARISON
//  Insertion sort is O(n²) worst case.
//  For large output rows, this is a bottleneck.
//  We measure actual sort work to decide if we need
//  a better sort (radix, std qsort, or no sort via hash).

static void analyze_sort_need(const CSR *C) {
    int m = C->m;
    int max_row=0, rows_gt_100=0, rows_gt_1000=0;
    double avg=0;

    for(int i=0;i<m;i++){
        int rlen=C->row_ptr[i+1]-C->row_ptr[i];
        avg+=rlen;
        if(rlen>max_row) max_row=rlen;
        if(rlen>100) rows_gt_100++;
        if(rlen>1000) rows_gt_1000++;
    }
    avg/=m;

    printf("\n=== SORT ANALYSIS ===\n");
    printf("  Avg output row length:  %.1f\n", avg);
    printf("  Max output row length:  %d\n", max_row);
    printf("  Rows with len > 100:    %d (%.1f%%)\n",
           rows_gt_100, 100.0*rows_gt_100/m);
    printf("  Rows with len > 1000:   %d (%.1f%%)\n",
           rows_gt_1000, 100.0*rows_gt_1000/m);

    printf("\n  Sort recommendation:\n");
    if(max_row < 32)
        printf("  -> Insertion sort OK (rows short, cache-friendly)\n");
    else if(max_row < 256)
        printf("  -> Consider std qsort for rows > 64\n");
    else
        printf("  -> Use radix sort or hash-based (no sort) for long rows\n");

    // Insertion sort worst case for this distribution
    long worst_sort = 0;
    for(int i=0;i<m;i++){
        long rlen=C->row_ptr[i+1]-C->row_ptr[i];
        worst_sort += rlen*rlen; // O(n²) per row
    }
    printf("  Insertion sort worst-case ops: %ld\n", worst_sort);
    printf("  Actual sort ops measured will be in profile above.\n");
}

//  MEMORY FOOTPRINT ANALYSIS

static void analyze_memory(const CSR *A, const CSR *AT,
                            const CSR *C, int nthreads) {
    long bytes_A   = (long)(A->m+1)*4 + (long)A->nnz*12;
    long bytes_AT  = (long)(AT->m+1)*4 + (long)AT->nnz*12;
    long bytes_C   = (long)(C->m+1)*4 + (long)C->nnz*12;
    long bytes_ws  = (long)nthreads * A->n * (8+4+4); // acc+occ+cols per thread
    long total     = bytes_A + bytes_AT + bytes_C + bytes_ws;

    printf("\n=== MEMORY FOOTPRINT ===\n");
    printf("  Matrix A (CSR):         %6.2f MB\n", bytes_A/1e6);
    printf("  Matrix AT (CSC):        %6.2f MB  (= A, just reordered)\n",bytes_AT/1e6);
    printf("  Output C (CSR):         %6.2f MB\n", bytes_C/1e6);
    printf("  Thread workspaces:      %6.2f MB  (%d x %.2f MB)\n",
           bytes_ws/1e6, nthreads, bytes_ws/1e6/nthreads);
    printf("  ---------------------------------\n");
    printf("  Total peak:             %6.2f MB\n", total/1e6);

    long l2  = 256*1024;
    long l3  = 3*1024*1024;
    long ws1 = (long)A->n * (8+4+4); // one thread workspace

    printf("\n  Cache analysis (assumes L1=32KB, L2=256KB, L3=3MB):\n");
    printf("  One thread workspace:   %6.2f KB  fits L%s\n",
           ws1/1024.0,
           ws1 < 32768 ? "1" :
           ws1 < l2    ? "2" :
           ws1 < l3    ? "3" : "3 (overflow)");
    printf("  A + AT together:        %6.2f MB  fits L%s\n",
           (bytes_A+bytes_AT)/1e6,
           bytes_A+bytes_AT < 32768 ? "1" :
           bytes_A+bytes_AT < l2    ? "2" :
           bytes_A+bytes_AT < l3    ? "3" : "3 (overflow, RAM access)");
}

//  THROUGHPUT METRICS (for comparison with literature)

static void print_throughput(const Profile *p, int nthreads) {
    (void)nthreads;
    // GFLOP/s: each output nnz costs 2 flops per inner iteration
    double gflops = 2.0 * p->n_inner_ops / p->t_numeric / 1e9;

    // GB/s: estimate bytes read during numeric pass
    // Reads: A values + AT values + acc writes/reads
    long bytes_read = (long)p->nnz_input * 8    // A values
                    + (long)p->n_inner_ops * 12  // AT col+val
                    + (long)p->nnz_output * 8;   // acc reads
    double gbps = bytes_read / p->t_numeric / 1e9;

    printf("\n=== THROUGHPUT METRICS ===\n");
    printf("  Numeric pass GFLOP/s:   %.4f\n", gflops);
    printf("  Estimated GB/s:         %.2f\n", gbps);
    printf("  Reference peak memory BW: ~34 GB/s (dual-channel DDR4-2400\n");
    printf("    class laptop; replace with your machine's real figure)\n");
    printf("  BW utilization:         %.1f%%\n", gbps/34.0*100.0);
    printf("\n  Roofline position:\n");
    if (gflops < 1.0)
        printf("  -> memory bound (low GFLOP/s, high bandwidth demand)\n");
    else
        printf("  -> compute bound (high GFLOP/s)\n");
    printf("  Arithmetic intensity: %.4f FLOP/byte\n",
           2.0*p->n_inner_ops / bytes_read);
}

//  MAIN

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix.mtx> [threads]\n", argv[0]);
        fprintf(stderr, "\nFor hardware counters run:\n");
        fprintf(stderr, "  perf stat -e cache-misses,L1-dcache-load-misses,"
                        "LLC-load-misses,cycles,instructions ./step4 <mtx>\n");
        return 1;
    }

    int nthreads = omp_get_max_threads();
    if (argc >= 3) nthreads = atoi(argv[2]);

    printf("=== Step 4: SpGEMM Performance Profiler ===\n");
    printf("Matrix: %s\n\n", argv[1]);

    // Load matrix
    Profile prof = {0};
    CSR *A = read_mm_csr(argv[1], &prof.t_read);
    if (!A) return 1;

    printf("Matrix: %dx%d  nnz=%d  density=%.6f\n\n",
           A->m, A->n, A->nnz,
           (double)A->nnz/(A->m*(double)A->n));

    double t_csc_start = omp_get_wtime();
    CSR *AT = csr_to_csc(A, &prof.t_csc);
    prof.t_csc = omp_get_wtime() - t_csc_start;

    // Warm-up run (not profiled)
    printf("Warming up (1 un-timed run)...\n");
    CSR *Cwarm = spgemm_instrumented(A, AT, nthreads, &prof);
    csr_free(Cwarm);

    // Profiled run
    printf("Profiling with %d thread(s)...\n\n", nthreads);
    double t_total_start = omp_get_wtime();
    CSR *C = spgemm_instrumented(A, AT, nthreads, &prof);
    prof.t_total = omp_get_wtime() - t_total_start
                 + prof.t_read + prof.t_csc;

    // Print profiles
    profile_print(&prof, nthreads);
    analyze_row_distribution(A, AT);
    analyze_sort_need(C);
    analyze_memory(A, AT, C, nthreads);
    print_throughput(&prof, nthreads);

    // Optimization opportunities summary
    printf("\n=== OPTIMIZATION OPPORTUNITIES (for Step 5) ===\n");

    double sort_pct  = 100.0*prof.t_sort/prof.t_numeric;
    double scat_pct  = 100.0*prof.t_scatter/prof.t_numeric;
    double sym_pct   = 100.0*prof.t_symbolic/prof.t_total;
    double num_pct   = 100.0*prof.t_numeric/prof.t_total;

    printf("  Symbolic pass:  %5.1f%% of total\n", sym_pct);
    printf("  Numeric pass:   %5.1f%% of total\n", num_pct);
    printf("    sort:         %5.1f%% of numeric\n", sort_pct);
    printf("    scatter:      %5.1f%% of numeric\n", scat_pct);
    printf("  --\n");

    if (sort_pct > 20.0)
        printf("  [!] Sort is a major bottleneck -> use radix sort\n");
    else
        printf("  [ok] Sort is minor -> insertion sort is fine\n");

    if (sym_pct > 40.0)
        printf("  [!] Symbolic pass dominates -> merge sym+num passes\n");
    else
        printf("  [ok] Symbolic/numeric balance is reasonable\n");

    printf("\n  Step 5 will target the largest bottleneck above.\n");

    // perf hint
    printf("\nFor hardware cache counters, run:\n");
    printf("  perf stat -e cache-misses,L1-dcache-load-misses,\\\n");
    printf("    LLC-load-misses,cycles,instructions,branch-misses \\\n");
    printf("    ./step4 %s %d\n\n", argv[1], nthreads);

    csr_free(A); csr_free(AT); csr_free(C);
    printf("SUCCESS: Step 4 complete.\n");
    return 0;
}
