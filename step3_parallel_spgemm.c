// step3_parallel_spgemm.c
// Goal: Parallelize SpGEMM using OpenMP.
//       Each thread handles a subset of output rows independently.
//       Use SPA strategy (best for our matrix sizes from Step 2).
//
// Key insight: rows of C are INDEPENDENT - C[i] depends only on
//              row i of A and all rows of A^T. No row writes to
//              another row's output. Perfect for parallelism.
//
// Compile: gcc -O2 -Wall -fopenmp -o step3 step3_parallel_spgemm.c mmio.c -lm
// Run:     ./step3 huge.mtx
// Run:     ./step3 huge.mtx 4     (force 4 threads)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "mmio.h"

//  CSR STRUCTURE

typedef struct {
    int m, n, nnz;
    int    *row_ptr;
    int    *col_idx;
    double *values;
} CSR;

static void csr_free(CSR *M) {
    if (!M) return;
    free(M->row_ptr);
    free(M->col_idx);
    free(M->values);
    free(M);
}

static CSR *read_mm_csr(const char *fname) {
    FILE *f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); return NULL; }
    MM_typecode mc;
    if (mm_read_banner(f, &mc) != 0) { fprintf(stderr,"Bad banner\n"); fclose(f); return NULL; }
    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) { fclose(f); return NULL; }
    int sym = mm_is_symmetric(mc);
    int max_e = sym ? 2*nz : nz;
    int *I = malloc(max_e*sizeof(int));
    int *J = malloc(max_e*sizeof(int));
    double *V = malloc(max_e*sizeof(double));
    int cnt=0;
    for (int e=0; e<nz; e++) {
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
        ci[pos]=J[e];cv[pos]=V[e];
    }
    free(I);free(J);free(V);free(cur);
    CSR *A=malloc(sizeof(CSR));
    A->m=M;A->n=N;A->nnz=nz;
    A->row_ptr=rp;A->col_idx=ci;A->values=cv;
    return A;
}

static CSR *csr_to_csc(const CSR *A) {
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
            ri[pos]=i;rv[pos]=A->values[p];
        }
    free(cur);
    CSR *B=malloc(sizeof(CSR));
    B->m=n;B->n=m;B->nnz=nnz;
    B->row_ptr=cp;B->col_idx=ri;B->values=rv;
    return B;
}

//  SEQUENTIAL SPA (baseline - from step 2)

static CSR *spgemm_spa_sequential(const CSR *A, const CSR *AT) {
    int m=A->m;
    double *acc    = calloc(m, sizeof(double));
    int *occupied  = calloc(m, sizeof(int));
    int *cols      = malloc(m * sizeof(int));
    int *row_nnz   = calloc(m, sizeof(int));

    // Pass 1: symbolic
    for(int i=0;i<m;i++){
        int cnt=0;
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            int k=A->col_idx[p];
            for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                int j=AT->col_idx[q];
                if(!occupied[j]){occupied[j]=1;cnt++;}
            }
        }
        row_nnz[i]=cnt;
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            int k=A->col_idx[p];
            for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++)
                occupied[AT->col_idx[q]]=0;
        }
    }
    int *rp=malloc((m+1)*sizeof(int));
    rp[0]=0;
    for(int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C=rp[m];
    int *ci=malloc(nnz_C*sizeof(int));
    double *cv=malloc(nnz_C*sizeof(double));

    // Pass 2: numeric
    for(int i=0;i<m;i++){
        int cnt=0;
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            int k=A->col_idx[p];
            double a_ik=A->values[p];
            for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                int j=AT->col_idx[q];
                if(!occupied[j]){occupied[j]=1;acc[j]=a_ik*AT->values[q];cols[cnt++]=j;}
                else acc[j]+=a_ik*AT->values[q];
            }
        }
        for(int a=1;a<cnt;a++){
            int key=cols[a];int b=a-1;
            while(b>=0&&cols[b]>key){cols[b+1]=cols[b];b--;}
            cols[b+1]=key;
        }
        int base=rp[i];
        for(int a=0;a<cnt;a++){
            int j=cols[a];ci[base+a]=j;cv[base+a]=acc[j];
            acc[j]=0.0;occupied[j]=0;
        }
    }
    free(acc);free(occupied);free(cols);free(row_nnz);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;
    C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

//  PARALLEL SPA - VERSION 1: STATIC SCHEDULING
//
//  Idea: divide rows evenly among threads.
//        Thread 0 handles rows 0..m/T-1
//        Thread 1 handles rows m/T..2m/T-1  etc.
//
//  Each thread has its OWN private SPA workspace:
//        acc[m], occupied[m], cols[m]
//  No sharing of workspace -> no locks needed.
//
//  Two OpenMP parallel regions:
//    Region 1: symbolic (count row_nnz[i] for all i)
//    Region 2: numeric  (fill col_idx and values)
//
//  row_ptr[] prefix sum is done sequentially between regions
//  (it's O(m) - very fast, not worth parallelizing here).
//
//  Scheduling: omp parallel for schedule(static)
//    -> compiler splits iterations into equal chunks
//    -> simple, low overhead, good when rows have similar work

static CSR *spgemm_parallel_static(const CSR *A, const CSR *AT,
                                    int nthreads) {
    int m = A->m;
    omp_set_num_threads(nthreads);

    int *row_nnz = calloc(m, sizeof(int));

    // Symbolic pass
    // Each thread allocates its own private workspace.
    // #pragma omp parallel creates a team of threads.
    // The block inside runs in parallel - each thread executes it.
    #pragma omp parallel
    {
        // private per-thread workspace (each thread mallocs its own)
        int    *occupied = calloc(m, sizeof(int));
        int    *cols     = malloc(m * sizeof(int));

        // schedule(static): rows divided into equal chunks automatically
        // No two threads write to the same row_nnz[i] entry.
        #pragma omp for schedule(static)
        for (int i = 0; i < m; i++) {
            int cnt = 0;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    if (!occupied[j]) { occupied[j]=1; cnt++; }
                }
            }
            row_nnz[i]=cnt;
            // reset only touched positions
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++)
                    occupied[AT->col_idx[q]]=0;
            }
        }
        // each thread frees its own workspace after its chunk is done
        free(occupied); free(cols);
    } // end parallel region 1

    // Sequential prefix sum - O(m), fast
    int *rp = malloc((m+1)*sizeof(int));
    rp[0]=0;
    for (int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C = rp[m];

    int    *ci = malloc(nnz_C * sizeof(int));
    double *cv = malloc(nnz_C * sizeof(double));

    // Numeric pass
    #pragma omp parallel
    {
        double *acc      = calloc(m, sizeof(double));
        int    *occupied = calloc(m, sizeof(int));
        int    *cols     = malloc(m * sizeof(int));

        #pragma omp for schedule(static)
        for (int i=0; i<m; i++) {
            int cnt=0;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                double a_ik=A->values[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    if (!occupied[j]){occupied[j]=1;acc[j]=a_ik*AT->values[q];cols[cnt++]=j;}
                    else acc[j]+=a_ik*AT->values[q];
                }
            }
            // insertion sort
            for (int a=1;a<cnt;a++) {
                int key=cols[a]; int b=a-1;
                while (b>=0&&cols[b]>key){cols[b+1]=cols[b];b--;}
                cols[b+1]=key;
            }
            int base=rp[i];
            for (int a=0;a<cnt;a++) {
                int j=cols[a]; ci[base+a]=j; cv[base+a]=acc[j];
                acc[j]=0.0; occupied[j]=0;
            }
        }
        free(acc); free(occupied); free(cols);
    } // end parallel region 2

    free(row_nnz);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;
    C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

//  PARALLEL SPA - VERSION 2: DYNAMIC SCHEDULING
//
//  Problem with static scheduling:
//    Rows have DIFFERENT amounts of work.
//    Row i with 50 nonzeros takes 10x more time than
//    row j with 5 nonzeros.
//    With static: thread 0 might get all heavy rows,
//    thread 1 finishes early and sits idle.
//    This is called LOAD IMBALANCE.
//
//  Solution: schedule(dynamic, chunk_size)
//    -> A pool of chunks is maintained.
//    -> When a thread finishes its current chunk,
//       it grabs the next available chunk from the pool.
//    -> Heavy rows and light rows get distributed fairly.
//
//  Tradeoff: dynamic has higher overhead (atomic counter
//    for chunk assignment) but better load balance.
//
//  chunk_size=1: maximum balance, maximum overhead
//  chunk_size=32: moderate - sweet spot for most cases
//
//  For SpGEMM: dynamic is better because row work varies
//  wildly based on the sparsity pattern.

static CSR *spgemm_parallel_dynamic(const CSR *A, const CSR *AT,
                                     int nthreads) {
    int m=A->m;
    omp_set_num_threads(nthreads);

    int *row_nnz = calloc(m, sizeof(int));

    // chunk_size = 32: each thread grabs 32 rows at a time
    // from the work pool when it becomes free
    int chunk = (m > 32*nthreads) ? 32 : 1;

    #pragma omp parallel
    {
        int    *occupied = calloc(m, sizeof(int));
        int    *cols     = malloc(m * sizeof(int));

        // schedule(dynamic, chunk): runtime decides which thread
        // gets which rows based on availability
        #pragma omp for schedule(dynamic, chunk)
        for (int i=0; i<m; i++) {
            int cnt=0;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    if (!occupied[j]) { occupied[j]=1; cnt++; }
                }
            }
            row_nnz[i]=cnt;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++)
                    occupied[AT->col_idx[q]]=0;
            }
        }
        free(occupied); free(cols);
    }

    int *rp=malloc((m+1)*sizeof(int));
    rp[0]=0;
    for(int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C=rp[m];
    int *ci=malloc(nnz_C*sizeof(int));
    double *cv=malloc(nnz_C*sizeof(double));

    #pragma omp parallel
    {
        double *acc      = calloc(m, sizeof(double));
        int    *occupied = calloc(m, sizeof(int));
        int    *cols     = malloc(m * sizeof(int));

        #pragma omp for schedule(dynamic, chunk)
        for (int i=0; i<m; i++) {
            int cnt=0;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                double a_ik=A->values[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    if (!occupied[j]){occupied[j]=1;acc[j]=a_ik*AT->values[q];cols[cnt++]=j;}
                    else acc[j]+=a_ik*AT->values[q];
                }
            }
            for (int a=1;a<cnt;a++) {
                int key=cols[a];int b=a-1;
                while(b>=0&&cols[b]>key){cols[b+1]=cols[b];b--;}
                cols[b+1]=key;
            }
            int base=rp[i];
            for (int a=0;a<cnt;a++){
                int j=cols[a];ci[base+a]=j;cv[base+a]=acc[j];
                acc[j]=0.0;occupied[j]=0;
            }
        }
        free(acc); free(occupied); free(cols);
    }

    free(row_nnz);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;
    C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

//  PARALLEL SPA - VERSION 3: GUIDED SCHEDULING
//
//  schedule(guided, chunk):
//    -> Like dynamic but chunk SIZE decreases over time.
//    -> Starts with large chunks (fast assignment, low overhead)
//    -> As end approaches, chunks shrink (better load balance)
//    -> Formula: chunk_size = max(chunk, remaining / nthreads)
//
//  Best of both worlds:
//    Early iterations: large chunks -> low scheduling overhead
//    Late iterations:  small chunks -> good load balance
//
//  Often the best choice for SpGEMM without profiling data.

static CSR *spgemm_parallel_guided(const CSR *A, const CSR *AT,
                                    int nthreads) {
    int m=A->m;
    omp_set_num_threads(nthreads);

    int *row_nnz=calloc(m,sizeof(int));

    #pragma omp parallel
    {
        int *occupied=calloc(m,sizeof(int));
        int *cols=malloc(m*sizeof(int));

        // guided: chunk size starts large, shrinks toward 1
        #pragma omp for schedule(guided, 1)
        for (int i=0; i<m; i++) {
            int cnt=0;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    if (!occupied[j]){occupied[j]=1;cnt++;}
                }
            }
            row_nnz[i]=cnt;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++)
                    occupied[AT->col_idx[q]]=0;
            }
        }
        free(occupied);free(cols);
    }

    int *rp=malloc((m+1)*sizeof(int));
    rp[0]=0;
    for(int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C=rp[m];
    int *ci=malloc(nnz_C*sizeof(int));
    double *cv=malloc(nnz_C*sizeof(double));

    #pragma omp parallel
    {
        double *acc=calloc(m,sizeof(double));
        int *occupied=calloc(m,sizeof(int));
        int *cols=malloc(m*sizeof(int));

        #pragma omp for schedule(guided, 1)
        for (int i=0; i<m; i++) {
            int cnt=0;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                double a_ik=A->values[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    if (!occupied[j]){occupied[j]=1;acc[j]=a_ik*AT->values[q];cols[cnt++]=j;}
                    else acc[j]+=a_ik*AT->values[q];
                }
            }
            for (int a=1;a<cnt;a++){
                int key=cols[a];int b=a-1;
                while(b>=0&&cols[b]>key){cols[b+1]=cols[b];b--;}
                cols[b+1]=key;
            }
            int base=rp[i];
            for (int a=0;a<cnt;a++){
                int j=cols[a];ci[base+a]=j;cv[base+a]=acc[j];
                acc[j]=0.0;occupied[j]=0;
            }
        }
        free(acc);free(occupied);free(cols);
    }

    free(row_nnz);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;
    C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

//  CORRECTNESS CHECK

static int csr_equal(const CSR *A, const CSR *B, double tol) {
    if (A->m!=B->m||A->n!=B->n||A->nnz!=B->nnz) return 0;
    for (int i=0;i<=A->m;i++) if (A->row_ptr[i]!=B->row_ptr[i]) return 0;
    for (int e=0;e<A->nnz;e++) {
        if (A->col_idx[e]!=B->col_idx[e]) return 0;
        if (fabs(A->values[e]-B->values[e])>tol) return 0;
    }
    return 1;
}

//  SPEEDUP ANALYSIS HELPER

static void print_speedup_table(double t_seq, double *t_par,
                                 int *threads, int n, const char *label) {
    printf("\n%s Speedup vs Thread Count:\n", label);
    printf("  %-8s %-12s %-10s %-12s\n",
           "Threads", "Time(sec)", "Speedup", "Efficiency");
    printf("  %-8s %-12.6f %-10.2f %-12.2f\n",
           "1(seq)", t_seq, 1.0, 100.0);
    for (int i=0; i<n; i++) {
        double speedup = t_seq / t_par[i];
        double efficiency = speedup / threads[i] * 100.0;
        printf("  %-8d %-12.6f %-10.2f %-11.1f%%\n",
               threads[i], t_par[i], speedup, efficiency);
    }
}

//  MAIN

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix.mtx> [max_threads]\n", argv[0]);
        return 1;
    }

    int max_threads = omp_get_max_threads();
    if (argc >= 3) max_threads = atoi(argv[2]);

    printf("=== Step 3: Parallel SpGEMM with OpenMP ===\n");
    printf("Matrix file:  %s\n", argv[1]);
    printf("Max threads:  %d\n", max_threads);
    printf("OMP max:      %d\n\n", omp_get_max_threads());

    CSR *A = read_mm_csr(argv[1]);
    if (!A) return 1;
    printf("Matrix A: %d x %d,  nnz = %d\n", A->m, A->n, A->nnz);
    printf("Density:  %.6f\n\n", (double)A->nnz/(A->m*(double)A->n));

    CSR *AT = csr_to_csc(A);

    // Sequential baseline
    printf("--- Sequential SPA (baseline) ---\n");
    double t0 = omp_get_wtime();
    CSR *C_seq = spgemm_spa_sequential(A, AT);
    double t_seq = omp_get_wtime() - t0;
    printf("  nnz: %d  time: %.6f sec\n\n", C_seq->nnz, t_seq);

    // Thread counts to test: 1, 2, 4, up to max_threads
    int thread_counts[8];
    int n_counts = 0;
    for (int t=1; t<=max_threads && n_counts<8; t*=2)
        thread_counts[n_counts++] = t;
    if (thread_counts[n_counts-1] != max_threads && n_counts < 8)
        thread_counts[n_counts++] = max_threads;

    double t_static[8], t_dynamic[8], t_guided[8];
    int ok_static=1, ok_dynamic=1, ok_guided=1;

    // Static scheduling
    printf("--- Parallel SPA: schedule(static) ---\n");
    printf("  %-8s %-12s %-10s %-10s\n",
           "Threads","Time(sec)","Speedup","Correct");
    for (int i=0; i<n_counts; i++) {
        int T = thread_counts[i];
        t0 = omp_get_wtime();
        CSR *C = spgemm_parallel_static(A, AT, T);
        t_static[i] = omp_get_wtime() - t0;
        int ok = csr_equal(C_seq, C, 1e-10);
        if (!ok) ok_static=0;
        printf("  %-8d %-12.6f %-10.2f %s\n",
               T, t_static[i], t_seq/t_static[i], ok?"OK":"FAIL");
        csr_free(C);
    }

    // Dynamic scheduling
    printf("\n--- Parallel SPA: schedule(dynamic) ---\n");
    printf("  %-8s %-12s %-10s %-10s\n",
           "Threads","Time(sec)","Speedup","Correct");
    for (int i=0; i<n_counts; i++) {
        int T = thread_counts[i];
        t0 = omp_get_wtime();
        CSR *C = spgemm_parallel_dynamic(A, AT, T);
        t_dynamic[i] = omp_get_wtime() - t0;
        int ok = csr_equal(C_seq, C, 1e-10);
        if (!ok) ok_dynamic=0;
        printf("  %-8d %-12.6f %-10.2f %s\n",
               T, t_dynamic[i], t_seq/t_dynamic[i], ok?"OK":"FAIL");
        csr_free(C);
    }

    // Guided scheduling
    printf("\n--- Parallel SPA: schedule(guided) ---\n");
    printf("  %-8s %-12s %-10s %-10s\n",
           "Threads","Time(sec)","Speedup","Correct");
    for (int i=0; i<n_counts; i++) {
        int T = thread_counts[i];
        t0 = omp_get_wtime();
        CSR *C = spgemm_parallel_guided(A, AT, T);
        t_guided[i] = omp_get_wtime() - t0;
        int ok = csr_equal(C_seq, C, 1e-10);
        if (!ok) ok_guided=0;
        printf("  %-8d %-12.6f %-10.2f %s\n",
               T, t_guided[i], t_seq/t_guided[i], ok?"OK":"FAIL");
        csr_free(C);
    }

    // Speedup tables
    print_speedup_table(t_seq, t_static,  thread_counts, n_counts, "Static ");
    print_speedup_table(t_seq, t_dynamic, thread_counts, n_counts, "Dynamic");
    print_speedup_table(t_seq, t_guided,  thread_counts, n_counts, "Guided ");

    // Summary
    printf("\n=== PARALLEL CONCEPTS DEMONSTRATED ===\n");
    printf("1. Thread-private workspace:\n");
    printf("   Each thread has its OWN acc[], occupied[], cols[].\n");
    printf("   No sharing -> no race conditions -> no locks needed.\n\n");
    printf("2. schedule(static):  equal chunks, low overhead.\n");
    printf("   Best when all rows have similar work.\n\n");
    printf("3. schedule(dynamic): work pool, threads grab chunks when free.\n");
    printf("   Best when row work varies widely (typical in SpGEMM).\n\n");
    printf("4. schedule(guided):  shrinking chunks, hybrid approach.\n");
    printf("   Often best overall without profiling data.\n\n");
    printf("5. Amdahl's Law: speedup limited by sequential fraction.\n");
    printf("   Prefix sum (sequential) is our bottleneck at high thread counts.\n");

    printf("\nCorrectness: static=%s dynamic=%s guided=%s\n",
           ok_static?"ALL OK":"FAILED",
           ok_dynamic?"ALL OK":"FAILED",
           ok_guided?"ALL OK":"FAILED");

    csr_free(A); csr_free(AT); csr_free(C_seq);
    printf("\nSUCCESS: Step 3 complete.\n");
    return 0;
}
