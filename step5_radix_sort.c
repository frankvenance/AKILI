// step5b_radix_sort.c
// Step 5 FIX. The range-scan "counting sort" in step5_optimized_sort.c
// was 6x SLOWER (0.16x) on giant.mtx. Root cause (measured & confirmed):
//
//   Range-scan loops over acc[min_col..max_col]. For a RANDOM sparse
//   matrix the ~100 nonzeros of each output row are scattered across
//   the full column space, so range ~= m (49500), not nnz (100).
//   That is 49500 * 50000 rows ~= 2.48 BILLION wasted iterations.
//
// The profiler said "sort = 51% of numeric", but replacing the sort
// with range-scan made things worse because range >> nnz for scattered
// rows. The correct O(nnz) fix is to sort the col[] array ITSELF, not
// scan a dense range.
//
// This file implements and compares FOUR strategies so the lesson is
// complete:
//   1. baseline      - insertion sort            (Step 3b)
//   2. range_scan    - dense range scan          (the failed idea)
//   3. radix         - LSD radix sort on col[]   (O(nnz), the real fix)
//   4. adaptive      - pick range_scan if the row is banded (range
//                      small), else radix. Best of both worlds.
//
// Compile: gcc -O3 -Wall -fopenmp -march=native -o step5b step5b_radix_sort.c mmio.c -lm
// Run:     ./step5b giant.mtx 2
//          ./step5b banded.mtx 2   (to see range_scan win)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "mmio.h"

//  CSR

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
    for(int e=0;e<nz;e++){
        int r,c; double v=1.0;
        if(mm_is_pattern(mc)) { if (fscanf(f,"%d %d",&r,&c)      != 2) break; }
        else                  { if (fscanf(f,"%d %d %lg",&r,&c,&v) != 3) break; }
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
            ri[pos]=i; rv[pos]=A->values[p];
        }
    free(cur);
    CSR *B=malloc(sizeof(CSR));
    B->m=n;B->n=m;B->nnz=nnz;
    B->row_ptr=cp;B->col_idx=ri;B->values=rv;
    return B;
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

//  THREAD WORKSPACE
//  acc[m]      - dense accumulator (values)
//  occupied[m] - occupancy flags
//  cols[m]     - list of touched column indices (unsorted)
//  tmp[m]      - scratch buffer for radix sort ping-pong

typedef struct {
    double *acc;
    int    *occupied;
    int    *cols;
    int    *tmp;
} WS;

static WS *alloc_ws(int T, int m) {
    WS *ws = malloc(T*sizeof(WS));
    for(int t=0;t<T;t++){
        ws[t].acc      = calloc(m,sizeof(double));
        ws[t].occupied = calloc(m,sizeof(int));
        ws[t].cols     = malloc(m*sizeof(int));
        ws[t].tmp      = malloc(m*sizeof(int));
    }
    return ws;
}

static void free_ws(WS *ws, int T) {
    for(int t=0;t<T;t++){
        free(ws[t].acc);
        free(ws[t].occupied);
        free(ws[t].cols);
        free(ws[t].tmp);
    }
    free(ws);
}

// number of 8-bit radix passes needed to cover values in [0, maxv]
static int radix_passes(int maxv) {
    int p = 0;
    unsigned v = (unsigned)maxv;
    do { p++; v >>= 8; } while (v);
    return p; // maxv=0 -> 1 pass
}

//  LSD RADIX SORT on an int array `a[0..n)`, values in [0, maxv].
//  `tmp` is a scratch buffer of length >= n. Uses radix-256.
//  After sorting, the sorted data is guaranteed to be in `a`.
//  Complexity: O(passes * (n + 256)), passes = ceil(bits/8).
//  Independent of the value RANGE - this is why it beats range-scan
//  for scattered rows.

static void radix_sort_int(int *a, int *tmp, int n, int passes) {
    int *src = a, *dst = tmp;
    for (int pass = 0; pass < passes; pass++) {
        int shift = pass * 8;
        int count[257] = {0};
        for (int i = 0; i < n; i++)
            count[((src[i] >> shift) & 0xFF) + 1]++;
        for (int b = 0; b < 256; b++)
            count[b+1] += count[b];
        for (int i = 0; i < n; i++)
            dst[count[(src[i] >> shift) & 0xFF]++] = src[i];
        int *t = src; src = dst; dst = t;
    }
    // if odd number of passes, sorted data ended up in tmp -> copy back
    if (src != a)
        memcpy(a, src, n * sizeof(int));
}

//  Shared symbolic pass: fills row_nnz[] and returns row_ptr + sizes.

static int *symbolic(const CSR *A, const CSR *AT, WS *ws, int T, int *nnz_out) {
    int m = A->m;
    int *row_nnz = calloc(m, sizeof(int));

    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        int *occ = ws[tid].occupied;
        int *col = ws[tid].cols;
        #pragma omp for schedule(static)
        for (int i=0; i<m; i++) {
            int cnt=0;
            for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
                int k=A->col_idx[p];
                for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                    int j=AT->col_idx[q];
                    if (!occ[j]) { occ[j]=1; col[cnt++]=j; }
                }
            }
            row_nnz[i]=cnt;
            for (int a=0;a<cnt;a++) occ[col[a]]=0;
        }
    }

    int *rp = malloc((m+1)*sizeof(int));
    rp[0]=0;
    for (int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    *nnz_out = rp[m];
    free(row_nnz);
    return rp;
}

//  STRATEGY 1 - BASELINE (insertion sort)

static CSR *baseline(const CSR *A, const CSR *AT, int T) {
    int m=A->m;
    WS *ws=alloc_ws(T,m);
    int nnz_C;
    int *rp=symbolic(A,AT,ws,T,&nnz_C);
    int *ci=malloc(nnz_C*sizeof(int));
    double *cv=malloc(nnz_C*sizeof(double));

    #pragma omp parallel num_threads(T)
    {
        int tid=omp_get_thread_num();
        double *acc=ws[tid].acc;
        int    *occ=ws[tid].occupied;
        int    *col=ws[tid].cols;
        #pragma omp for schedule(static)
        for(int i=0;i<m;i++){
            int cnt=0;
            for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
                int k=A->col_idx[p];double a_ik=A->values[p];
                for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                    int j=AT->col_idx[q];
                    if(!occ[j]){occ[j]=1;acc[j]=a_ik*AT->values[q];col[cnt++]=j;}
                    else acc[j]+=a_ik*AT->values[q];
                }
            }
            for(int a=1;a<cnt;a++){
                int key=col[a];int b=a-1;
                while(b>=0&&col[b]>key){col[b+1]=col[b];b--;}
                col[b+1]=key;
            }
            int base=rp[i];
            for(int a=0;a<cnt;a++){
                int j=col[a];ci[base+a]=j;cv[base+a]=acc[j];
                acc[j]=0.0;occ[j]=0;
            }
        }
    }
    free_ws(ws,T);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

//  STRATEGY 2 - RANGE SCAN (the failed idea, kept for the lesson)

static CSR *range_scan(const CSR *A, const CSR *AT, int T) {
    int m=A->m;
    WS *ws=alloc_ws(T,m);
    int nnz_C;
    int *rp=symbolic(A,AT,ws,T,&nnz_C);
    int *ci=malloc(nnz_C*sizeof(int));
    double *cv=malloc(nnz_C*sizeof(double));

    #pragma omp parallel num_threads(T)
    {
        int tid=omp_get_thread_num();
        double *acc=ws[tid].acc;
        int    *occ=ws[tid].occupied;
        #pragma omp for schedule(static)
        for(int i=0;i<m;i++){
            int min_col=m,max_col=-1;
            for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
                int k=A->col_idx[p];double a_ik=A->values[p];
                for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                    int j=AT->col_idx[q];
                    acc[j]+=a_ik*AT->values[q];
                    if(!occ[j]){occ[j]=1;if(j<min_col)min_col=j;if(j>max_col)max_col=j;}
                }
            }
            int base=rp[i],out=0;
            for(int j=min_col;j<=max_col;j++){
                if(occ[j]){ci[base+out]=j;cv[base+out]=acc[j];out++;acc[j]=0.0;occ[j]=0;}
            }
        }
    }
    free_ws(ws,T);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

//  STRATEGY 3 - LSD RADIX SORT on col[]  (THE REAL FIX)
//  O(nnz) per row, independent of column range.

static CSR *radix(const CSR *A, const CSR *AT, int T) {
    int m=A->m;
    WS *ws=alloc_ws(T,m);
    int nnz_C;
    int *rp=symbolic(A,AT,ws,T,&nnz_C);
    int *ci=malloc(nnz_C*sizeof(int));
    double *cv=malloc(nnz_C*sizeof(double));
    int passes = radix_passes(m-1);

    #pragma omp parallel num_threads(T)
    {
        int tid=omp_get_thread_num();
        double *acc=ws[tid].acc;
        int    *occ=ws[tid].occupied;
        int    *col=ws[tid].cols;
        int    *tmp=ws[tid].tmp;
        #pragma omp for schedule(static)
        for(int i=0;i<m;i++){
            int cnt=0;
            for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
                int k=A->col_idx[p];double a_ik=A->values[p];
                for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                    int j=AT->col_idx[q];
                    if(!occ[j]){occ[j]=1;acc[j]=a_ik*AT->values[q];col[cnt++]=j;}
                    else acc[j]+=a_ik*AT->values[q];
                }
            }
            // sort the col[] list itself - O(cnt), not O(range)
            radix_sort_int(col, tmp, cnt, passes);
            int base=rp[i];
            for(int a=0;a<cnt;a++){
                int j=col[a];ci[base+a]=j;cv[base+a]=acc[j];
                acc[j]=0.0;occ[j]=0;
            }
        }
    }
    free_ws(ws,T);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

//  STRATEGY 4 - ADAPTIVE
//  Per row, decide based on the actual column range:
//    range = max_col - min_col + 1
//    if range <= RANGE_FACTOR * cnt  -> row is "banded" -> range_scan
//    else                            -> scattered       -> radix sort
//  This captures the best of both: banded matrices get the cheap
//  sequential scan; scattered matrices get O(nnz) radix.

#define RANGE_FACTOR 4

static CSR *adaptive(const CSR *A, const CSR *AT, int T) {
    int m=A->m;
    WS *ws=alloc_ws(T,m);
    int nnz_C;
    int *rp=symbolic(A,AT,ws,T,&nnz_C);
    int *ci=malloc(nnz_C*sizeof(int));
    double *cv=malloc(nnz_C*sizeof(double));
    int passes = radix_passes(m-1);

    #pragma omp parallel num_threads(T)
    {
        int tid=omp_get_thread_num();
        double *acc=ws[tid].acc;
        int    *occ=ws[tid].occupied;
        int    *col=ws[tid].cols;
        int    *tmp=ws[tid].tmp;
        #pragma omp for schedule(static)
        for(int i=0;i<m;i++){
            int cnt=0, min_col=m, max_col=-1;
            for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
                int k=A->col_idx[p];double a_ik=A->values[p];
                for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                    int j=AT->col_idx[q];
                    if(!occ[j]){
                        occ[j]=1;acc[j]=a_ik*AT->values[q];col[cnt++]=j;
                        if(j<min_col)min_col=j;
                        if(j>max_col)max_col=j;
                    } else acc[j]+=a_ik*AT->values[q];
                }
            }
            int base=rp[i];
            long range = (max_col>=min_col) ? (long)max_col-min_col+1 : 0;
            if (range > 0 && range <= (long)RANGE_FACTOR*cnt) {
                // banded row -> cheap sequential range scan (already sorted)
                int out=0;
                for(int j=min_col;j<=max_col;j++){
                    if(occ[j]){ci[base+out]=j;cv[base+out]=acc[j];out++;acc[j]=0.0;occ[j]=0;}
                }
            } else {
                // scattered row -> radix sort col[]
                radix_sort_int(col, tmp, cnt, passes);
                for(int a=0;a<cnt;a++){
                    int j=col[a];ci[base+a]=j;cv[base+a]=acc[j];
                    acc[j]=0.0;occ[j]=0;
                }
            }
        }
    }
    free_ws(ws,T);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

//  BENCHMARK HARNESS

typedef CSR* (*SpGEMMFn)(const CSR*, const CSR*, int);

static double bench(SpGEMMFn fn, const CSR *A, const CSR *AT,
                    int T, int runs, CSR **keep) {
    CSR *C = fn(A, AT, T); // warmup
    if (keep) *keep = C; else csr_free(C);

    double best = 1e18;
    for (int r=0; r<runs; r++) {
        double t0=omp_get_wtime();
        CSR *tmp = fn(A, AT, T);
        double t=omp_get_wtime()-t0;
        if (t<best) best=t;
        csr_free(tmp);
    }
    return best;
}

//  MAIN

int main(int argc, char **argv) {
    if (argc<2) {
        fprintf(stderr,"Usage: %s <matrix.mtx> [threads]\n",argv[0]);
        return 1;
    }
    int T=2;
    if (argc>=3) T=atoi(argv[2]);

    printf("=== Step 5b: Radix Sort + Adaptive SpGEMM ===\n");
    printf("Fixing the failed range-scan (was 0.16x on giant.mtx)\n\n");

    CSR *A = read_mm_csr(argv[1]);
    if (!A) return 1;
    CSR *AT = csr_to_csc(A);
    printf("Matrix: %dx%d  nnz=%d  threads=%d\n", A->m, A->n, A->nnz, T);
    printf("Radix passes (for m=%d): %d\n\n", A->m, radix_passes(A->m-1));

    int RUNS=5;
    printf("Running benchmarks (%d runs each, best time)...\n\n", RUNS);

    CSR *C_ref=NULL;
    double t_base = bench(baseline, A, AT, T, RUNS, &C_ref);

    double t_rng = bench(range_scan, A, AT, T, RUNS, NULL);
    CSR *C_rng = range_scan(A, AT, T);
    int ok_rng = csr_equal(C_ref, C_rng, 1e-10); csr_free(C_rng);

    double t_rdx = bench(radix, A, AT, T, RUNS, NULL);
    CSR *C_rdx = radix(A, AT, T);
    int ok_rdx = csr_equal(C_ref, C_rdx, 1e-10); csr_free(C_rdx);

    double t_adp = bench(adaptive, A, AT, T, RUNS, NULL);
    CSR *C_adp = adaptive(A, AT, T);
    int ok_adp = csr_equal(C_ref, C_adp, 1e-10); csr_free(C_adp);

    printf("  %-22s %-11s %-9s %-8s\n","Strategy","Time(sec)","Speedup","Correct");
    printf("  %s\n","-----------------------------------------------------");
    printf("  %-22s %-11.6f %-9s %-8s\n","Baseline (ins.sort)",t_base,"1.00x","OK");
    printf("  %-22s %-11.6f %-8.2fx %-8s\n","Range scan (failed)",t_rng,t_base/t_rng,ok_rng?"OK":"FAIL");
    printf("  %-22s %-11.6f %-8.2fx %-8s\n","Radix sort (fix)",t_rdx,t_base/t_rdx,ok_rdx?"OK":"FAIL");
    printf("  %-22s %-11.6f %-8.2fx %-8s\n","Adaptive",t_adp,t_base/t_adp,ok_adp?"OK":"FAIL");

    printf("\n=== ANALYSIS ===\n");
    printf("  Range scan visits acc[min..max] -> O(range) per row.\n");
    printf("  For scattered rows range ~= m, so it wastes billions of\n");
    printf("  iterations on empty slots. Radix sorts col[] directly ->\n");
    printf("  O(nnz) per row, independent of column range.\n");
    printf("  Adaptive picks range_scan only when a row is banded.\n");

    csr_free(A); csr_free(AT); csr_free(C_ref);
    printf("\nSUCCESS: Step 5b complete.\n");
    return 0;
}
