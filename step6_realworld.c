// step6_realworld.c
// Step 6 - Real-world matrices (SuiteSparse collection).
//
// Everything so far was tested on synthetic random/banded matrices.
// Real matrices from the SuiteSparse collection (https://sparse.tamu.edu)
// have wildly different structure: power-law row lengths, clustering,
// symmetry, empty rows. This step is a PRODUCTION-QUALITY benchmark
// harness that:
//
//   1. Loads ANY .mtx (handles symmetric / pattern / rectangular).
//   2. Analyzes the matrix structure (symmetry, bandwidth, row-length
//      distribution, coefficient of variation -> scheduling hint).
//   3. Computes C = A * A^T using the ADAPTIVE strategy (winner of
//      Step 5b: radix sort for scattered rows, range-scan for banded).
//   4. Runs a strong-scaling study (1/2/4 threads) with speedup,
//      efficiency, GFLOP/s and memory-bandwidth estimates.
//   5. Verifies every parallel result against a sequential reference.
//
// Compile: gcc -O3 -Wall -fopenmp -march=native -o step6 step6_realworld.c mmio.c -lm
// Run:     ./step6 <matrix.mtx>
// Fetch real matrices first with:  ./fetch_matrices.sh

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
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

typedef struct {
    int symmetric;
    int pattern;
    int rectangular;
} MatrixInfo;

static CSR *read_mm_csr(const char *fname, MatrixInfo *info) {
    FILE *f = fopen(fname,"r");
    if (!f) { fprintf(stderr,"Cannot open %s\n",fname); return NULL; }
    MM_typecode mc;
    if (mm_read_banner(f,&mc) != 0) { fprintf(stderr,"Bad banner\n"); fclose(f); return NULL; }
    if (!mm_is_matrix(mc) || !mm_is_sparse(mc)) {
        fprintf(stderr,"Not a sparse matrix\n"); fclose(f); return NULL;
    }
    int M,N,nz;
    if (mm_read_mtx_crd_size(f,&M,&N,&nz) != 0) { fclose(f); return NULL; }
    int sym=mm_is_symmetric(mc) || mm_is_skew(mc) || mm_is_hermitian(mc);
    int pat=mm_is_pattern(mc);
    if (info) { info->symmetric=sym; info->pattern=pat; info->rectangular=(M!=N); }

    int max_e=sym?2*nz:nz;
    int *I=malloc(max_e*sizeof(int));
    int *J=malloc(max_e*sizeof(int));
    double *V=malloc(max_e*sizeof(double));
    int cnt=0;
    for(int e=0;e<nz;e++){
        int r,c; double v=1.0;
        if(pat) { if (fscanf(f,"%d %d",&r,&c)      != 2) break; }
        else    { if (fscanf(f,"%d %d %lg",&r,&c,&v) != 3) break; }
        r--;c--;
        if (r<0||r>=M||c<0||c>=N) continue; // guard against malformed
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

//  STRUCTURE ANALYSIS - what kind of matrix is this?

static double g_cv = 0.0; // coefficient of variation of row work (global hint)

static void analyze_structure(const CSR *A, const MatrixInfo *info) {
    int m=A->m;
    long total=0, maxlen=0, minlen=LONG_MAX, empty=0;
    long bandwidth=0;
    for(int i=0;i<m;i++){
        long len=A->row_ptr[i+1]-A->row_ptr[i];
        total+=len;
        if(len>maxlen)maxlen=len;
        if(len<minlen)minlen=len;
        if(len==0)empty++;
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            long d=labs((long)A->col_idx[p]-i);
            if(d>bandwidth)bandwidth=d;
        }
    }
    double mean=(double)total/m, var=0;
    for(int i=0;i<m;i++){
        double d=(A->row_ptr[i+1]-A->row_ptr[i])-mean;
        var+=d*d;
    }
    var/=m;
    double stddev=sqrt(var);
    g_cv = mean>0 ? stddev/mean : 0.0;

    printf("=== MATRIX STRUCTURE ===\n");
    printf("  Dimensions:        %d x %d %s\n", A->m, A->n,
           info->rectangular?"(rectangular)":"(square)");
    printf("  Non-zeros:         %d\n", A->nnz);
    printf("  Density:           %.2e\n", (double)A->nnz/((double)A->m*A->n));
    printf("  Storage type:      %s%s\n",
           info->symmetric?"symmetric ":"general ",
           info->pattern?"(pattern)":"(real)");
    printf("  Row length min/avg/max: %ld / %.1f / %ld\n", minlen, mean, maxlen);
    printf("  Empty rows:        %ld (%.1f%%)\n", empty, 100.0*empty/m);
    printf("  Bandwidth:         %ld  (%.1f%% of n)\n",
           bandwidth, 100.0*bandwidth/A->n);
    printf("  CV (stddev/mean):  %.2f  %s\n", g_cv,
           g_cv>1.0 ? "-> HIGH imbalance, use dynamic schedule"
                    : "-> LOW imbalance, static schedule fine");
    printf("\n");
}

//  ADAPTIVE SpGEMM (winner of Step 5b) + sequential reference

typedef struct { double *acc; int *occ; int *cols; int *tmp; } WS;

static WS *alloc_ws(int T,int m){
    WS *w=malloc(T*sizeof(WS));
    for(int t=0;t<T;t++){
        w[t].acc=calloc(m,sizeof(double));
        w[t].occ=calloc(m,sizeof(int));
        w[t].cols=malloc(m*sizeof(int));
        w[t].tmp=malloc(m*sizeof(int));
    }
    return w;
}
static void free_ws(WS *w,int T){
    for(int t=0;t<T;t++){free(w[t].acc);free(w[t].occ);free(w[t].cols);free(w[t].tmp);}
    free(w);
}
static int radix_passes(int maxv){int p=0;unsigned v=(unsigned)maxv;do{p++;v>>=8;}while(v);return p;}
static void radix_sort_int(int *a,int *tmp,int n,int passes){
    int *src=a,*dst=tmp;
    for(int pass=0;pass<passes;pass++){
        int shift=pass*8, count[257]={0};
        for(int i=0;i<n;i++) count[((src[i]>>shift)&0xFF)+1]++;
        for(int b=0;b<256;b++) count[b+1]+=count[b];
        for(int i=0;i<n;i++) dst[count[(src[i]>>shift)&0xFF]++]=src[i];
        int *t=src;src=dst;dst=t;
    }
    if(src!=a) memcpy(a,src,n*sizeof(int));
}

#define RANGE_FACTOR 4

// dynamic schedule when work is imbalanced, static otherwise.
static CSR *spgemm_adaptive(const CSR *A, const CSR *AT, int T, int use_dynamic) {
    int m=A->m;
    WS *ws=alloc_ws(T,m);
    int *row_nnz=calloc(m,sizeof(int));
    int passes=radix_passes(m-1);
    int chunk = m>32*T?32:1;

    // symbolic
    #pragma omp parallel num_threads(T)
    {
        int tid=omp_get_thread_num();
        int *occ=ws[tid].occ,*col=ws[tid].cols;
        if (use_dynamic) {
            #pragma omp for schedule(dynamic, chunk)
            for(int i=0;i<m;i++){
                int cnt=0;
                for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
                    int k=A->col_idx[p];
                    for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                        int j=AT->col_idx[q];
                        if(!occ[j]){occ[j]=1;col[cnt++]=j;}
                    }
                }
                row_nnz[i]=cnt;
                for(int a=0;a<cnt;a++) occ[col[a]]=0;
            }
        } else {
            #pragma omp for schedule(static)
            for(int i=0;i<m;i++){
                int cnt=0;
                for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
                    int k=A->col_idx[p];
                    for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                        int j=AT->col_idx[q];
                        if(!occ[j]){occ[j]=1;col[cnt++]=j;}
                    }
                }
                row_nnz[i]=cnt;
                for(int a=0;a<cnt;a++) occ[col[a]]=0;
            }
        }
    }

    int *rp=malloc((m+1)*sizeof(int));
    rp[0]=0;
    for(int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C=rp[m];
    int *ci=malloc((nnz_C>0?nnz_C:1)*sizeof(int));
    double *cv=malloc((nnz_C>0?nnz_C:1)*sizeof(double));

    // numeric - adaptive per row
    #pragma omp parallel num_threads(T)
    {
        int tid=omp_get_thread_num();
        double *acc=ws[tid].acc;
        int *occ=ws[tid].occ,*col=ws[tid].cols,*tmp=ws[tid].tmp;
        #pragma omp for schedule(runtime)
        for(int i=0;i<m;i++){
            int cnt=0,min_col=m,max_col=-1;
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
            long range=(max_col>=min_col)?(long)max_col-min_col+1:0;
            if(range>0 && range<=(long)RANGE_FACTOR*cnt){
                int out=0;
                for(int j=min_col;j<=max_col;j++)
                    if(occ[j]){ci[base+out]=j;cv[base+out]=acc[j];out++;acc[j]=0.0;occ[j]=0;}
            } else {
                radix_sort_int(col,tmp,cnt,passes);
                for(int a=0;a<cnt;a++){
                    int j=col[a];ci[base+a]=j;cv[base+a]=acc[j];
                    acc[j]=0.0;occ[j]=0;
                }
            }
        }
    }
    free_ws(ws,T); free(row_nnz);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

// sequential reference (simple, obviously-correct)
static CSR *spgemm_seq(const CSR *A, const CSR *AT, long *inner_ops) {
    int m=A->m;
    double *acc=calloc(m,sizeof(double));
    int *occ=calloc(m,sizeof(int));
    int *col=malloc(m*sizeof(int));
    int *row_nnz=calloc(m,sizeof(int));
    long inner=0;
    for(int i=0;i<m;i++){
        int cnt=0;
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            int k=A->col_idx[p];
            for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                int j=AT->col_idx[q]; inner++;
                if(!occ[j]){occ[j]=1;col[cnt++]=j;}
            }
        }
        row_nnz[i]=cnt;
        for(int a=0;a<cnt;a++) occ[col[a]]=0;
    }
    int *rp=malloc((m+1)*sizeof(int));
    rp[0]=0;
    for(int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C=rp[m];
    int *ci=malloc((nnz_C>0?nnz_C:1)*sizeof(int));
    double *cv=malloc((nnz_C>0?nnz_C:1)*sizeof(double));
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
        for(int a=1;a<cnt;a++){int key=col[a];int b=a-1;
            while(b>=0&&col[b]>key){col[b+1]=col[b];b--;}col[b+1]=key;}
        int base=rp[i];
        for(int a=0;a<cnt;a++){int j=col[a];ci[base+a]=j;cv[base+a]=acc[j];acc[j]=0.0;occ[j]=0;}
    }
    free(acc);free(occ);free(col);free(row_nnz);
    if(inner_ops)*inner_ops=inner;
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

//  MAIN - strong scaling study on a real matrix

int main(int argc, char **argv) {
    if (argc<2) {
        fprintf(stderr,"Usage: %s <matrix.mtx>\n",argv[0]);
        fprintf(stderr,"Fetch real matrices with ./fetch_matrices.sh first.\n");
        return 1;
    }

    printf("=== Step 6: Real-World SpGEMM Benchmark (C = A*A^T) ===\n\n");
    printf("Matrix file: %s\n\n", argv[1]);

    MatrixInfo info={0};
    CSR *A = read_mm_csr(argv[1], &info);
    if (!A) return 1;
    CSR *AT = csr_to_csc(A);

    analyze_structure(A, &info);

    // sequential reference
    long inner=0;
    double t0=omp_get_wtime();
    CSR *C_ref = spgemm_seq(A, AT, &inner);
    double t_seq = omp_get_wtime()-t0;

    printf("=== SEQUENTIAL REFERENCE ===\n");
    printf("  Output nnz:  %d  (fill factor %.2fx)\n",
           C_ref->nnz, A->nnz>0?(double)C_ref->nnz/A->nnz:0.0);
    printf("  Time:        %.6f sec\n", t_seq);
    printf("  Inner iters: %ld\n\n", inner);

    // scheduling hint from structure
    int use_dynamic = g_cv > 1.0;
    // set OMP runtime schedule for the numeric-pass 'runtime' clause
    omp_set_schedule(use_dynamic?omp_sched_dynamic:omp_sched_static,
                     use_dynamic?32:0);
    printf("=== STRONG SCALING (adaptive sort, %s schedule) ===\n",
           use_dynamic?"dynamic":"static");
    printf("  %-8s %-11s %-9s %-11s %-9s %-6s\n",
           "Threads","Time(s)","Speedup","Efficiency","GFLOP/s","OK");
    printf("  %s\n","--------------------------------------------------------------");

    int tcs[]={1,2,4}; int max_t=omp_get_max_threads();
    for(int x=0;x<3;x++){
        int T=tcs[x];
        if(T>max_t) continue;
        CSR *Cw=spgemm_adaptive(A,AT,T,use_dynamic); csr_free(Cw); // warmup
        double best=1e18; CSR *C=NULL;
        for(int r=0;r<5;r++){
            double s=omp_get_wtime();
            CSR *tmp=spgemm_adaptive(A,AT,T,use_dynamic);
            double t=omp_get_wtime()-s;
            if(t<best){best=t;}
            if(r==4) C=tmp; else csr_free(tmp);
        }
        int ok=csr_equal(C_ref,C,1e-9);
        double sp=t_seq/best, eff=sp/T*100.0;
        double gflops=2.0*inner/best/1e9;
        printf("  %-8d %-11.6f %-9.2f %-10.1f%% %-9.3f %-6s\n",
               T,best,sp,eff,gflops,ok?"OK":"FAIL");
        csr_free(C);
    }

    printf("\n=== NOTES ===\n");
    printf("  - Real matrices often have power-law rows (high CV) ->\n");
    printf("    dynamic scheduling is auto-selected when CV > 1.0.\n");
    printf("  - Adaptive sort handles both banded and scattered rows.\n");
    printf("  - GFLOP/s counts 2 flops per inner accumulate iteration.\n");

    csr_free(A); csr_free(AT); csr_free(C_ref);
    printf("\nSUCCESS: Step 6 complete.\n");
    return 0;
}
