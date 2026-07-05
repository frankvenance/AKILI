// step3_optimized.c
// Fixes applied:
//   1. Allocate thread workspaces ONCE outside parallel regions
//   2. Use thread ID to index into pre-allocated workspace array
//   3. Parallel prefix sum (Blelloch scan) to eliminate sequential bottleneck
//   4. Test with much larger matrix to see real speedup
//
// Compile: gcc -O2 -Wall -fopenmp -o step3b step3_optimized.c mmio.c -lm
// Run:     ./step3b huge.mtx

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "mmio.h"

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

// read_mm_csr (same as before)
static CSR *read_mm_csr(const char *fname) {
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
    for(int e=0;e<nz;e++){int pos=rp[I[e]]+cur[I[e]]++;ci[pos]=J[e];cv[pos]=V[e];}
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
            int j=A->col_idx[p];int pos=cp[j]+cur[j]++;
            ri[pos]=i;rv[pos]=A->values[p];
        }
    free(cur);
    CSR *B=malloc(sizeof(CSR));
    B->m=n;B->n=m;B->nnz=nnz;
    B->row_ptr=cp;B->col_idx=ri;B->values=rv;
    return B;
}

//  FIX 1 - Pre-allocated thread workspaces
//
//  Instead of calloc inside parallel region (expensive),
//  allocate ALL thread workspaces BEFORE entering parallel.
//  Each thread uses workspace[omp_get_thread_num()].
//
//  Result: zero allocation overhead inside the hot loop.

typedef struct {
    double *acc;
    int    *occupied;
    int    *cols;
} ThreadWS;

static ThreadWS *alloc_workspaces(int nthreads, int m) {
    ThreadWS *ws = malloc(nthreads * sizeof(ThreadWS));
    for (int t=0; t<nthreads; t++) {
        ws[t].acc      = calloc(m, sizeof(double));
        ws[t].occupied = calloc(m, sizeof(int));
        ws[t].cols     = malloc(m * sizeof(int));
    }
    return ws;
}

static void free_workspaces(ThreadWS *ws, int nthreads) {
    for (int t=0; t<nthreads; t++) {
        free(ws[t].acc);
        free(ws[t].occupied);
        free(ws[t].cols);
    }
    free(ws);
}

//  FIX 2 - Parallel Prefix Sum (Blelloch Scan)
//
//  Sequential prefix sum was our Amdahl bottleneck.
//  Parallel version:
//
//  Phase 1 - each thread computes local sum of its chunk:
//    Thread 0: sum(row_nnz[0..m/4])
//    Thread 1: sum(row_nnz[m/4..m/2])
//    ...
//
//  Phase 2 - sequential scan over thread sums (only T ops):
//    partial[0]=0, partial[1]=partial[0]+sum[0], etc.
//
//  Phase 3 - each thread adds its partial offset:
//    Thread t: rp[i] += partial[t] for its chunk
//
//  Total work: O(m) same as sequential but T-way parallel.
//  Overhead: only T sequential ops between phases.

static int *parallel_prefix_sum(int *row_nnz, int m, int nthreads) {
    int *rp = malloc((m+1) * sizeof(int));
    int *partial = calloc(nthreads+1, sizeof(int));

    // Phase 1: each thread sums its chunk
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int chunk = (m + nthreads - 1) / nthreads;
        int start = tid * chunk;
        int end   = start + chunk < m ? start + chunk : m;

        int local_sum = 0;
        for (int i=start; i<end; i++) local_sum += row_nnz[i];
        partial[tid+1] = local_sum;
    }

    // Phase 2: sequential prefix over thread partial sums (only T ops)
    partial[0] = 0;
    for (int t=1; t<=nthreads; t++) partial[t] += partial[t-1];

    // Phase 3: each thread builds its section of rp[]
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int chunk = (m + nthreads - 1) / nthreads;
        int start = tid * chunk;
        if (start > m) start = m; // more threads than rows -> this thread does nothing
        int end   = start + chunk < m ? start + chunk : m;

        int offset = partial[tid];
        rp[start] = offset;
        for (int i=start; i<end; i++) {
            rp[i+1] = rp[i] + row_nnz[i];
        }
    }
    rp[m] = partial[nthreads]; // total nnz

    free(partial);
    return rp;
}

//  OPTIMIZED PARALLEL SPGEMM
//  Combines Fix 1 (pre-alloc workspaces) + Fix 2 (parallel prefix)
//  + schedule(dynamic) which won in step 3

static CSR *spgemm_optimized(const CSR *A, const CSR *AT,
                              int nthreads) {
    int m = A->m;

    // Pre-allocate all workspaces BEFORE parallel regions
    ThreadWS *ws = alloc_workspaces(nthreads, m);
    int *row_nnz = calloc(m, sizeof(int));
    int chunk = m > 32*nthreads ? 32 : 1;

    // Symbolic pass
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int *occupied = ws[tid].occupied;
        int *cols     = ws[tid].cols;

        #pragma omp for schedule(dynamic, chunk)
        for (int i=0; i<m; i++) {
            int cnt=0;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    if (!occupied[j]) { occupied[j]=1; cols[cnt++]=j; }
                }
            }
            row_nnz[i]=cnt;
            // reset only touched
            for (int a=0; a<cnt; a++) occupied[cols[a]]=0;
        }
    }

    // Parallel prefix sum instead of sequential
    int *rp = parallel_prefix_sum(row_nnz, m, nthreads);
    int nnz_C = rp[m];

    int    *ci = malloc(nnz_C * sizeof(int));
    double *cv = malloc(nnz_C * sizeof(double));

    // Numeric pass
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        double *acc      = ws[tid].acc;
        int    *occupied = ws[tid].occupied;
        int    *cols     = ws[tid].cols;

        #pragma omp for schedule(dynamic, chunk)
        for (int i=0; i<m; i++) {
            int cnt=0;
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
            // insertion sort
            for (int a=1;a<cnt;a++) {
                int key=cols[a]; int b=a-1;
                while(b>=0&&cols[b]>key){cols[b+1]=cols[b];b--;}
                cols[b+1]=key;
            }
            int base=rp[i];
            for (int a=0;a<cnt;a++){
                int j=cols[a];
                ci[base+a]=j;
                cv[base+a]=acc[j];
                acc[j]=0.0;
                occupied[j]=0;
            }
        }
    }

    free_workspaces(ws, nthreads);
    free(row_nnz);

    CSR *C=malloc(sizeof(CSR));
    C->m=m; C->n=m; C->nnz=nnz_C;
    C->row_ptr=rp; C->col_idx=ci; C->values=cv;
    return C;
}

// sequential baseline
static CSR *spgemm_sequential(const CSR *A, const CSR *AT) {
    int m=A->m;
    double *acc=calloc(m,sizeof(double));
    int *occupied=calloc(m,sizeof(int));
    int *cols=malloc(m*sizeof(int));
    int *row_nnz=calloc(m,sizeof(int));
    for(int i=0;i<m;i++){
        int cnt=0;
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            int k=A->col_idx[p];
            for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                int j=AT->col_idx[q];
                if(!occupied[j]){occupied[j]=1;cols[cnt++]=j;}
            }
        }
        row_nnz[i]=cnt;
        for(int a=0;a<cnt;a++) occupied[cols[a]]=0;
    }
    int *rp=malloc((m+1)*sizeof(int));
    rp[0]=0;
    for(int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C=rp[m];
    int *ci=malloc(nnz_C*sizeof(int));
    double *cv=malloc(nnz_C*sizeof(double));
    for(int i=0;i<m;i++){
        int cnt=0;
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            int k=A->col_idx[p];double a_ik=A->values[p];
            for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                int j=AT->col_idx[q];
                if(!occupied[j]){occupied[j]=1;acc[j]=a_ik*AT->values[q];cols[cnt++]=j;}
                else acc[j]+=a_ik*AT->values[q];
            }
        }
        for(int a=1;a<cnt;a++){int key=cols[a];int b=a-1;
            while(b>=0&&cols[b]>key){cols[b+1]=cols[b];b--;}cols[b+1]=key;}
        int base=rp[i];
        for(int a=0;a<cnt;a++){int j=cols[a];ci[base+a]=j;cv[base+a]=acc[j];acc[j]=0.0;occupied[j]=0;}
    }
    free(acc);free(occupied);free(cols);free(row_nnz);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

static int csr_equal(const CSR *A, const CSR *B, double tol) {
    if(A->m!=B->m||A->n!=B->n||A->nnz!=B->nnz) return 0;
    for(int i=0;i<=A->m;i++) if(A->row_ptr[i]!=B->row_ptr[i]) return 0;
    for(int e=0;e<A->nnz;e++){
        if(A->col_idx[e]!=B->col_idx[e]) return 0;
        if(fabs(A->values[e]-B->values[e])>tol) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc<2){fprintf(stderr,"Usage: %s <matrix.mtx> [threads]\n",argv[0]);return 1;}

    int max_t = omp_get_max_threads();
    if (argc>=3) max_t = atoi(argv[2]);

    printf("=== Step 3 Optimized: Pre-alloc WS + Parallel Prefix Sum ===\n");
    printf("Matrix: %s  |  Max threads: %d\n\n", argv[1], max_t);

    CSR *A = read_mm_csr(argv[1]);
    if (!A) return 1;
    CSR *AT = csr_to_csc(A);
    printf("Matrix: %dx%d  nnz=%d  density=%.6f\n\n",
           A->m, A->n, A->nnz,
           (double)A->nnz/(A->m*(double)A->n));

    // Sequential baseline
    double t0 = omp_get_wtime();
    CSR *C_seq = spgemm_sequential(A, AT);
    double t_seq = omp_get_wtime() - t0;
    printf("Sequential:  %.6f sec  (nnz=%d)\n\n", t_seq, C_seq->nnz);

    // Test thread counts
    int tcounts[] = {1, 2, 4};
    int ntc = 3;

    printf("%-8s %-12s %-10s %-12s %-8s\n",
           "Threads","Time(sec)","Speedup","Efficiency","OK");
    printf("%s\n", "------------------------------------------------------");

    for (int i=0; i<ntc; i++) {
        int T = tcounts[i];
        if (T > max_t) continue;

        // warm up
        CSR *Cw = spgemm_optimized(A, AT, T);
        csr_free(Cw);

        // timed run
        t0 = omp_get_wtime();
        CSR *C = spgemm_optimized(A, AT, T);
        double t = omp_get_wtime() - t0;

        double speedup = t_seq / t;
        double eff = speedup / T * 100.0;
        int ok = csr_equal(C_seq, C, 1e-10);

        printf("%-8d %-12.6f %-10.2f %-11.1f%%  %s\n",
               T, t, speedup, eff, ok?"OK":"FAIL");
        csr_free(C);
    }

    printf("\n=== Lessons from Step 3 ===\n");
    printf("1. Matrix too small -> parallel overhead > compute gain\n");
    printf("2. calloc inside parallel region is expensive\n");
    printf("3. Pre-allocate workspaces before parallel regions\n");
    printf("4. Parallel prefix sum removes Amdahl bottleneck\n");
    printf("5. Real speedup needs matrix with nnz > 500,000+\n\n");

    printf("Generate a larger matrix to see true speedup:\n");
    printf("  python3 -c \"\n");
    printf("import random\n");
    printf("n=50000; k=500000\n");
    printf("print('%%%%MatrixMarket matrix coordinate real general')\n");
    printf("print(f'{n} {n} {k}')\n");
    printf("s=set()\n");
    printf("while len(s)<k:\n");
    printf("    r=random.randint(1,n); c=random.randint(1,n)\n");
    printf("    if (r,c) not in s:\n");
    printf("        s.add((r,c))\n");
    printf("        print(f'{r} {c} {random.uniform(0.1,1):.4f}')\n");
    printf("\" > giant.mtx\n");
    printf("./step3b giant.mtx\n");

    csr_free(A); csr_free(AT); csr_free(C_seq);
    printf("\nSUCCESS: Step 3 optimized complete.\n");
    return 0;
}
