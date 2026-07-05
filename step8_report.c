// step8_report.c
// Step 8 - Final benchmark report generator.
//
// Runs the full pipeline (structure model -> auto-tuned config ->
// verified SpGEMM) across every matrix passed on the command line and
// emits:
//   - a human-readable Markdown table to stdout AND to REPORT.md
//   - a machine-readable results.csv
//
// This is the capstone: one command reproduces the entire benchmark
// suite and produces artifacts ready for a GitHub README / CV.
//
// Compile: gcc -O3 -Wall -fopenmp -march=native -o step8 step8_report.c mmio.c -lm
// Run:     ./step8 matrices/*.mtx giant.mtx banded.mtx

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include "mmio.h"
#include "autotune_calib.h"

typedef struct { int m,n,nnz; int *row_ptr,*col_idx; double *values; } CSR;
static void csr_free(CSR *M){ if(!M)return; free(M->row_ptr);free(M->col_idx);free(M->values);free(M);}

typedef struct { int symmetric, pattern, rectangular; } MatrixInfo;

static CSR *read_mm_csr(const char *fname, MatrixInfo *info){
    FILE *f=fopen(fname,"r");
    if(!f){fprintf(stderr,"Cannot open %s\n",fname);return NULL;}
    MM_typecode mc;
    if(mm_read_banner(f,&mc)!=0){fclose(f);return NULL;}
    if(!mm_is_matrix(mc)||!mm_is_sparse(mc)){fclose(f);return NULL;}
    int M,N,nz;
    if(mm_read_mtx_crd_size(f,&M,&N,&nz)!=0){fclose(f);return NULL;}
    int sym=mm_is_symmetric(mc)||mm_is_skew(mc)||mm_is_hermitian(mc);
    int pat=mm_is_pattern(mc);
    if(info){info->symmetric=sym;info->pattern=pat;info->rectangular=(M!=N);}
    int max_e=sym?2*nz:nz;
    int *I=malloc(max_e*sizeof(int)),*J=malloc(max_e*sizeof(int));
    double *V=malloc(max_e*sizeof(double));
    int cnt=0;
    for(int e=0;e<nz;e++){
        int r,c; double v=1.0;
        if(pat){ if(fscanf(f,"%d %d",&r,&c)!=2) break; }
        else   { if(fscanf(f,"%d %d %lg",&r,&c,&v)!=3) break; }
        r--;c--;
        if(r<0||r>=M||c<0||c>=N) continue;
        I[cnt]=r;J[cnt]=c;V[cnt]=v;cnt++;
        if(sym&&r!=c){I[cnt]=c;J[cnt]=r;V[cnt]=v;cnt++;}
    }
    fclose(f); nz=cnt;
    int *rp=calloc(M+1,sizeof(int));
    for(int e=0;e<nz;e++) rp[I[e]+1]++;
    for(int i=0;i<M;i++) rp[i+1]+=rp[i];
    int *ci=malloc(nz*sizeof(int)); double *cv=malloc(nz*sizeof(double));
    int *cur=calloc(M,sizeof(int));
    for(int e=0;e<nz;e++){int pos=rp[I[e]]+cur[I[e]]++;ci[pos]=J[e];cv[pos]=V[e];}
    free(I);free(J);free(V);free(cur);
    CSR *A=malloc(sizeof(CSR));
    A->m=M;A->n=N;A->nnz=nz;A->row_ptr=rp;A->col_idx=ci;A->values=cv;
    return A;
}

static CSR *csr_to_csc(const CSR *A){
    int m=A->m,n=A->n,nnz=A->nnz;
    int *cp=calloc(n+1,sizeof(int));
    for(int e=0;e<nnz;e++) cp[A->col_idx[e]+1]++;
    for(int j=0;j<n;j++) cp[j+1]+=cp[j];
    int *ri=malloc(nnz*sizeof(int)); double *rv=malloc(nnz*sizeof(double));
    int *cur=calloc(n,sizeof(int));
    for(int i=0;i<m;i++)
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            int j=A->col_idx[p];int pos=cp[j]+cur[j]++;ri[pos]=i;rv[pos]=A->values[p];
        }
    free(cur);
    CSR *B=malloc(sizeof(CSR));
    B->m=n;B->n=m;B->nnz=nnz;B->row_ptr=cp;B->col_idx=ri;B->values=rv;
    return B;
}

static int csr_equal(const CSR *A,const CSR *B,double tol){
    if(A->m!=B->m||A->n!=B->n||A->nnz!=B->nnz) return 0;
    for(int i=0;i<=A->m;i++) if(A->row_ptr[i]!=B->row_ptr[i]) return 0;
    for(int e=0;e<A->nnz;e++){
        if(A->col_idx[e]!=B->col_idx[e]) return 0;
        if(fabs(A->values[e]-B->values[e])>tol) return 0;
    }
    return 1;
}

// adaptive SpGEMM (Step 5b/7)
typedef struct { double *acc; int *occ; int *cols; int *tmp; } WS;
static WS *alloc_ws(int T,int m){
    WS *w=malloc(T*sizeof(WS));
    for(int t=0;t<T;t++){w[t].acc=calloc(m,sizeof(double));w[t].occ=calloc(m,sizeof(int));
        w[t].cols=malloc(m*sizeof(int));w[t].tmp=malloc(m*sizeof(int));}
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
        int shift=pass*8,count[257]={0};
        for(int i=0;i<n;i++) count[((src[i]>>shift)&0xFF)+1]++;
        for(int b=0;b<256;b++) count[b+1]+=count[b];
        for(int i=0;i<n;i++) dst[count[(src[i]>>shift)&0xFF]++]=src[i];
        int *t=src;src=dst;dst=t;
    }
    if(src!=a) memcpy(a,src,n*sizeof(int));
}
#define RANGE_FACTOR 4
static CSR *spgemm(const CSR *A,const CSR *AT,int T){
    int m=A->m;
    WS *ws=alloc_ws(T,m);
    int *row_nnz=calloc(m,sizeof(int));
    int passes=radix_passes(m-1);
    #pragma omp parallel num_threads(T)
    {
        int tid=omp_get_thread_num();
        int *occ=ws[tid].occ,*col=ws[tid].cols;
        #pragma omp for schedule(runtime)
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
    int *rp=malloc((m+1)*sizeof(int)); rp[0]=0;
    for(int i=0;i<m;i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C=rp[m];
    int *ci=malloc((nnz_C>0?nnz_C:1)*sizeof(int));
    double *cv=malloc((nnz_C>0?nnz_C:1)*sizeof(double));
    #pragma omp parallel num_threads(T)
    {
        int tid=omp_get_thread_num();
        double *acc=ws[tid].acc; int *occ=ws[tid].occ,*col=ws[tid].cols,*tmp=ws[tid].tmp;
        #pragma omp for schedule(runtime)
        for(int i=0;i<m;i++){
            int cnt=0,min_col=m,max_col=-1;
            for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
                int k=A->col_idx[p];double a_ik=A->values[p];
                for(int q=AT->row_ptr[k];q<AT->row_ptr[k+1];q++){
                    int j=AT->col_idx[q];
                    if(!occ[j]){occ[j]=1;acc[j]=a_ik*AT->values[q];col[cnt++]=j;
                        if(j<min_col)min_col=j;
                        if(j>max_col)max_col=j;}
                    else acc[j]+=a_ik*AT->values[q];
                }
            }
            int base=rp[i];
            long range=(max_col>=min_col)?(long)max_col-min_col+1:0;
            if(range>0&&range<=(long)RANGE_FACTOR*cnt){
                int out=0;
                for(int j=min_col;j<=max_col;j++)
                    if(occ[j]){ci[base+out]=j;cv[base+out]=acc[j];out++;acc[j]=0.0;occ[j]=0;}
            } else {
                radix_sort_int(col,tmp,cnt,passes);
                for(int a=0;a<cnt;a++){int j=col[a];ci[base+a]=j;cv[base+a]=acc[j];acc[j]=0.0;occ[j]=0;}
            }
        }
    }
    free_ws(ws,T);free(row_nnz);
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}
static void set_schedule(int dynamic){
    omp_set_schedule(dynamic?omp_sched_dynamic:omp_sched_static, dynamic?32:0);
}

// sequential insertion-sort reference (Step 1 style) for total-speedup
static CSR *spgemm_seq(const CSR *A,const CSR *AT,long *inner_out){
    int m=A->m;
    double *acc=calloc(m,sizeof(double)); int *occ=calloc(m,sizeof(int));
    int *col=malloc(m*sizeof(int)); int *row_nnz=calloc(m,sizeof(int));
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
    int *rp=malloc((m+1)*sizeof(int)); rp[0]=0;
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
    if(inner_out)*inner_out=inner;
    CSR *C=malloc(sizeof(CSR));
    C->m=m;C->n=m;C->nnz=nnz_C;C->row_ptr=rp;C->col_idx=ci;C->values=cv;
    return C;
}

// cheap model: work estimate + CV
static void model(const CSR *A,const CSR *AT,long *work,double *cv){
    int m=A->m; long total=0; double *rw=malloc(m*sizeof(double));
    for(int i=0;i<m;i++){
        long w=0;
        for(int p=A->row_ptr[i];p<A->row_ptr[i+1];p++){
            int k=A->col_idx[p]; w+=AT->row_ptr[k+1]-AT->row_ptr[k];
        }
        rw[i]=(double)w; total+=w;
    }
    double mean=(double)total/m,var=0;
    for(int i=0;i<m;i++){double d=rw[i]-mean;var+=d*d;} var/=m;
    free(rw);
    *work=total; *cv=mean>0?sqrt(var)/mean:0.0;
}

static void predict(long work,double cv,int max_t,long ht_threshold,int *T,int *dyn){
    (void)cv;
    if(max_t<2){*T=1;*dyn=0;return;}
    *T=(work>=ht_threshold)?max_t:2;
    *dyn=1;
}

static const char *basename_of(const char *p){
    const char *b=strrchr(p,'/'); return b?b+1:p;
}

typedef struct {
    char name[128];
    int m,nnz,out_nnz,T,dyn,ok;
    long work;
    double cv, t_seq, t_best, total_sp, gflops;
} Row;

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"Usage: %s <matrix.mtx> [more.mtx ...]\n",argv[0]);return 1;}
    int max_t=omp_get_max_threads();
    long ht_threshold=get_ht_threshold(max_t);

    Row *rows=calloc(argc-1,sizeof(Row));
    int nr=0;

    printf("Generating benchmark report over %d matrices (max %d threads)...\n\n",
           argc-1, max_t);

    for(int a=1;a<argc;a++){
        MatrixInfo info={0};
        CSR *A=read_mm_csr(argv[a],&info);
        if(!A){fprintf(stderr,"  [skip] %s (unreadable)\n",argv[a]);continue;}
        CSR *AT=csr_to_csc(A);

        long work; double cv; model(A,AT,&work,&cv);
        int T,dyn; predict(work,cv,max_t,ht_threshold,&T,&dyn);

        // Best-of-3 for the sequential baseline too -- a single sample
        // was getting compared against the parallel side's best-of-5,
        // which on a noisy/shared host let one unlucky sequential
        // sample make "speedup" swing wildly (observed: ca-GrQc showing
        // 0.5x-1.5x run to run) even though the parallel time itself
        // was stable near-optimal. Asymmetric measurement rigor, not a
        // real regression.
        long inner=0;
        double t_seq=1e18; CSR *C_ref=NULL;
        for(int r=0;r<3;r++){
            double s=omp_get_wtime();
            CSR *tmp=spgemm_seq(A,AT,&inner);
            double t=omp_get_wtime()-s;
            if(t<t_seq)t_seq=t;
            if(r==0)C_ref=tmp; else csr_free(tmp);
        }

        set_schedule(dyn);
        CSR *w=spgemm(A,AT,T); csr_free(w); // warmup
        double best=1e18; CSR *C=NULL;
        for(int r=0;r<5;r++){
            double s2=omp_get_wtime();
            CSR *tmp=spgemm(A,AT,T);
            double t=omp_get_wtime()-s2;
            if(t<best)best=t;
            if(r==4)C=tmp; else csr_free(tmp);
        }
        int ok=csr_equal(C_ref,C,1e-9);

        Row *R=&rows[nr++];
        snprintf(R->name,sizeof(R->name),"%s",basename_of(argv[a]));
        R->m=A->m; R->nnz=A->nnz; R->out_nnz=C->nnz;
        R->T=T; R->dyn=dyn; R->ok=ok; R->work=work; R->cv=cv;
        R->t_seq=t_seq; R->t_best=best;
        R->total_sp=t_seq/best;
        R->gflops=2.0*inner/best/1e9;

        printf("  [done] %-14s nnz=%-9d out=%-9d cfg=%dT/%s  %.2fx  %s\n",
               R->name,R->nnz,R->out_nnz,R->T,R->dyn?"dyn":"sta",
               R->total_sp, ok?"OK":"FAIL");

        csr_free(A);csr_free(AT);csr_free(C_ref);csr_free(C);
    }

    // write REPORT.md
    FILE *md=fopen("REPORT.md","w");
    FILE *csv=fopen("results.csv","w");
    if(md){
        fprintf(md,"# SpGEMM Benchmark Report\n\n");
        fprintf(md,"Computed `C = A * A^T` with an auto-tuned, adaptive-sort,\n");
        fprintf(md,"OpenMP SpGEMM. All results verified against a sequential\n");
        fprintf(md,"reference. Machine: %d logical threads.\n\n", max_t);
        fprintf(md,"| Matrix | m | nnz(A) | nnz(C) | Fill | Work(iters) | CV | Config | Seq(s) | Best(s) | Speedup | GFLOP/s | OK |\n");
        fprintf(md,"|---|--:|--:|--:|--:|--:|--:|:--:|--:|--:|--:|--:|:--:|\n");
        for(int i=0;i<nr;i++){
            Row *R=&rows[i];
            fprintf(md,"| %s | %d | %d | %d | %.2fx | %ld | %.2f | %dT/%s | %.4f | %.4f | %.2fx | %.3f | %s |\n",
                R->name,R->m,R->nnz,R->out_nnz,
                R->nnz>0?(double)R->out_nnz/R->nnz:0.0,
                R->work,R->cv,R->T,R->dyn?"dyn":"sta",
                R->t_seq,R->t_best,R->total_sp,R->gflops,R->ok?"OK":"FAIL");
        }
        fprintf(md,"\n## Notes\n\n");
        fprintf(md,"- **Speedup** is best auto-tuned parallel time vs the sequential\n");
        fprintf(md,"  insertion-sort baseline (Step 1), so it captures BOTH the\n");
        fprintf(md,"  algorithmic win (radix/adaptive sort) and parallel scaling.\n");
        fprintf(md,"- **Config** is chosen by a zero-run heuristic: use all cores\n");
        fprintf(md,"  when work >= %ld inner iters (calibrated empirically on this\n",
                ht_threshold);
        fprintf(md,"  machine, cached in .spgemm_autotune_cache), else 2 cores; dynamic schedule.\n");
        fprintf(md,"- Correctness verified elementwise (col indices + values, tol 1e-9).\n");
        fclose(md);
    }
    if(csv){
        fprintf(csv,"matrix,m,nnz_A,nnz_C,fill,work_iters,cv,threads,schedule,seq_s,best_s,speedup,gflops,ok\n");
        for(int i=0;i<nr;i++){
            Row *R=&rows[i];
            fprintf(csv,"%s,%d,%d,%d,%.4f,%ld,%.4f,%d,%s,%.6f,%.6f,%.4f,%.4f,%d\n",
                R->name,R->m,R->nnz,R->out_nnz,
                R->nnz>0?(double)R->out_nnz/R->nnz:0.0,
                R->work,R->cv,R->T,R->dyn?"dynamic":"static",
                R->t_seq,R->t_best,R->total_sp,R->gflops,R->ok);
        }
        fclose(csv);
    }

    // stdout summary table
    printf("\n=== FINAL REPORT ===\n");
    printf("%-14s %-9s %-9s %-8s %-8s %-9s %-7s %-4s\n",
           "Matrix","nnz(A)","nnz(C)","Config","Seq(s)","Best(s)","Speedup","OK");
    printf("%s\n","------------------------------------------------------------------------");
    double geo=1.0; int okc=0;
    for(int i=0;i<nr;i++){
        Row *R=&rows[i];
        printf("%-14s %-9d %-9d %d%-7s %-8.4f %-9.4f %-6.2fx %-4s\n",
               R->name,R->nnz,R->out_nnz,R->T,R->dyn?"T/dyn":"T/sta",
               R->t_seq,R->t_best,R->total_sp,R->ok?"OK":"FAIL");
        geo*=R->total_sp; if(R->ok)okc++;
    }
    if(nr>0) geo=pow(geo,1.0/nr);
    printf("%s\n","------------------------------------------------------------------------");
    printf("Geometric-mean speedup: %.2fx   |   Correct: %d/%d\n", geo, okc, nr);
    printf("\nWrote REPORT.md and results.csv\n");
    printf("SUCCESS: Step 8 complete. Portfolio pipeline finished.\n");

    free(rows);
    return 0;
}
