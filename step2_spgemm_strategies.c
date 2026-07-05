// step2_spgemm_strategies.c
// Goal: Implement 3 SpGEMM strategies and benchmark them against each other.
//       Strategy 1: SPA  (Sparse Accumulator)   - dense array per row
//       Strategy 2: HASH (Hash Map Accumulator)  - hash table per row
//       Strategy 3: HEAP (Merge via min-heap)    - heap-based row merging
//
// All compute C = A * A^T and must produce identical results.
//
// Compile: gcc -O2 -Wall -o step2 step2_spgemm_strategies.c mmio.c -lm
// Run:     ./step2 test.mtx

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mmio.h"

//  CSR STRUCTURE  (same as step 1 - no changes)

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

// read Matrix Market -> CSR (identical to step 1)
static CSR *read_mm_csr(const char *fname) {
    FILE *f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); return NULL; }

    MM_typecode mc;
    if (mm_read_banner(f, &mc) != 0) {
        fprintf(stderr, "Bad banner\n"); fclose(f); return NULL;
    }
    if (!mm_is_matrix(mc) || !mm_is_sparse(mc)) {
        fprintf(stderr, "Need sparse matrix\n"); fclose(f); return NULL;
    }

    int M, N, nz;
    mm_read_mtx_crd_size(f, &M, &N, &nz);

    int sym = mm_is_symmetric(mc);
    int max_e = sym ? 2*nz : nz;
    int *I = malloc(max_e * sizeof(int));
    int *J = malloc(max_e * sizeof(int));
    double *V = malloc(max_e * sizeof(double));
    int cnt = 0;

    for (int e = 0; e < nz; e++) {
        int r, c; double v = 1.0;
        if (mm_is_pattern(mc)) { if (fscanf(f, "%d %d",     &r, &c)      != 2) break; }
        else                   { if (fscanf(f, "%d %d %lg", &r, &c, &v)  != 3) break; }
        r--; c--;
        I[cnt]=r; J[cnt]=c; V[cnt]=v; cnt++;
        if (sym && r!=c) { I[cnt]=c; J[cnt]=r; V[cnt]=v; cnt++; }
    }
    fclose(f);
    nz = cnt;

    int *rp = calloc(M+1, sizeof(int));
    for (int e=0; e<nz; e++) rp[I[e]+1]++;
    for (int i=0; i<M; i++)  rp[i+1] += rp[i];

    int *ci = malloc(nz * sizeof(int));
    double *cv = malloc(nz * sizeof(double));
    int *cur = calloc(M, sizeof(int));
    for (int e=0; e<nz; e++) {
        int pos = rp[I[e]] + cur[I[e]]++;
        ci[pos]=J[e]; cv[pos]=V[e];
    }
    free(I); free(J); free(V); free(cur);

    CSR *A = malloc(sizeof(CSR));
    A->m=M; A->n=N; A->nnz=nz;
    A->row_ptr=rp; A->col_idx=ci; A->values=cv;
    return A;
}

// CSR -> CSC (AT stored as CSR of A^T)
static CSR *csr_to_csc(const CSR *A) {
    int m=A->m, n=A->n, nnz=A->nnz;
    int *cp = calloc(n+1, sizeof(int));
    for (int e=0; e<nnz; e++) cp[A->col_idx[e]+1]++;
    for (int j=0; j<n; j++) cp[j+1]+=cp[j];

    int *ri = malloc(nnz*sizeof(int));
    double *rv = malloc(nnz*sizeof(double));
    int *cur = calloc(n, sizeof(int));
    for (int i=0; i<m; i++)
        for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
            int j=A->col_idx[p];
            int pos=cp[j]+cur[j]++;
            ri[pos]=i; rv[pos]=A->values[p];
        }
    free(cur);

    CSR *B = malloc(sizeof(CSR));
    B->m=n; B->n=m; B->nnz=nnz;
    B->row_ptr=cp; B->col_idx=ri; B->values=rv;
    return B;
}

// sort column indices within each CSR row (insertion sort)
static void sort_csr_rows(CSR *C) {
    for (int i=0; i<C->m; i++) {
        int s = C->row_ptr[i], e = C->row_ptr[i+1];
        // insertion sort on (col_idx, values) pairs
        for (int a=s+1; a<e; a++) {
            int   kc = C->col_idx[a];
            double kv = C->values[a];
            int b = a-1;
            while (b>=s && C->col_idx[b]>kc) {
                C->col_idx[b+1]=C->col_idx[b];
                C->values[b+1] =C->values[b];
                b--;
            }
            C->col_idx[b+1]=kc;
            C->values[b+1] =kv;
        }
    }
}

// verify two CSR results match (for correctness check)
static int csr_equal(const CSR *A, const CSR *B, double tol) {
    if (A->m!=B->m || A->n!=B->n || A->nnz!=B->nnz) {
        printf("  MISMATCH: dims or nnz differ (%d vs %d nnz)\n",
               A->nnz, B->nnz);
        return 0;
    }
    for (int i=0; i<=A->m; i++)
        if (A->row_ptr[i]!=B->row_ptr[i]) {
            printf("  MISMATCH: row_ptr[%d]\n", i); return 0;
        }
    for (int e=0; e<A->nnz; e++) {
        if (A->col_idx[e]!=B->col_idx[e]) {
            printf("  MISMATCH: col_idx[%d]\n",e); return 0;
        }
        if (fabs(A->values[e]-B->values[e]) > tol) {
            printf("  MISMATCH: values[%d]: %.6f vs %.6f\n",
                   e, A->values[e], B->values[e]);
            return 0;
        }
    }
    return 1;
}

//  STRATEGY 1 - SPA (Sparse Accumulator / Dense Array)
//
//  How it works:
//    For each output row i of C = A * A^T:
//      - Keep a dense array "acc" of size m (number of output rows)
//      - Keep a boolean "occupied" array of same size
//      - For each nonzero A[i,k], walk column k of A^T (= row k of A)
//        and accumulate products into acc[j]
//      - Collect all occupied j's, sort them, write to C
//      - Reset only the occupied positions (not the whole array)
//
//  Cost per row i: O(nnz_i * max_nnz_col) numeric + O(output_nnz_i * log)
//  Memory: O(m) dense working space - fine for small/medium m
//          BAD for very large m (e.g. m = 10^7 means 80 MB just for acc)
//
//  Best for: small-to-medium matrices where m fits comfortably in cache

static CSR *spgemm_spa(const CSR *A, const CSR *AT) {
    int m = A->m;

    // working space - allocated ONCE outside the row loop
    double *acc      = calloc(m, sizeof(double)); // accumulator
    int    *occupied = calloc(m, sizeof(int));    // 0 = empty
    int    *cols     = malloc(m * sizeof(int));   // column list for this row

    // Pass 1 - symbolic: count nnz per output row
    int *row_nnz = calloc(m, sizeof(int));
    for (int i=0; i<m; i++) {
        int cnt=0;
        for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
            int k = A->col_idx[p];
            for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                int j = AT->col_idx[q];
                if (!occupied[j]) { occupied[j]=1; cnt++; }
            }
        }
        row_nnz[i] = cnt;
        // reset occupied
        for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
            int k=A->col_idx[p];
            for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++)
                occupied[AT->col_idx[q]]=0;
        }
    }

    // build row_ptr
    int *rp = malloc((m+1)*sizeof(int));
    rp[0]=0;
    for (int i=0; i<m; i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C = rp[m];

    int    *ci = malloc(nnz_C * sizeof(int));
    double *cv = malloc(nnz_C * sizeof(double));

    // Pass 2 - numeric
    for (int i=0; i<m; i++) {
        int cnt=0;
        for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
            int k = A->col_idx[p];
            double a_ik = A->values[p];
            for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                int j = AT->col_idx[q];
                double prod = a_ik * AT->values[q];
                if (!occupied[j]) {
                    occupied[j]=1;
                    acc[j]=prod;
                    cols[cnt++]=j;
                } else {
                    acc[j]+=prod;
                }
            }
        }

        // sort column list for this row
        // insertion sort: rows are typically short, so this is fast
        for (int a=1; a<cnt; a++) {
            int key=cols[a]; int b=a-1;
            while (b>=0 && cols[b]>key) { cols[b+1]=cols[b]; b--; }
            cols[b+1]=key;
        }

        // write to output and reset
        int base = rp[i];
        for (int a=0; a<cnt; a++) {
            int j = cols[a];
            ci[base+a] = j;
            cv[base+a] = acc[j];
            acc[j]     = 0.0;
            occupied[j]= 0;
        }
    }

    free(acc); free(occupied); free(cols); free(row_nnz);

    CSR *C = malloc(sizeof(CSR));
    C->m=m; C->n=m; C->nnz=nnz_C;
    C->row_ptr=rp; C->col_idx=ci; C->values=cv;
    return C;
}

//  STRATEGY 2 - HASH (Open-Addressing Hash Accumulator)
//
//  How it works:
//    For each output row i:
//      - Allocate a hash table of size hash_size (next power of 2 >= 2*expected_nnz)
//      - For each nonzero A[i,k], walk column k of A^T and insert/accumulate
//        (col_j -> value) into the hash table using linear probing
//      - Collect all occupied slots, sort by key, write to C
//
//  Key insight: hash table is local to one row - allocated fresh each row
//               (or pre-allocated to max possible size and reused with a
//                generation counter to avoid clearing)
//
//  Here we use a FIXED large table + generation trick:
//    - hash_gen[slot] stores the row index when slot was last used
//    - No memset needed between rows - just compare hash_gen[slot]==i
//
//  Cost per row: O(nnz_i * avg_chain) amortized O(1) per insert
//  Memory: O(hash_size) working space - hash_size = next_pow2(2*m)
//          Independent of matrix size if hash_size bounded
//
//  Best for: medium matrices, cache-friendly when hash table fits in L2/L3
//            Faster than SPA when m is large (hash table << m)

// next power of 2 >= x
static int next_pow2(int x) {
    int p=1; while(p<x) p<<=1; return p;
}

typedef struct {
    int    key;   // column index, -1 = empty
    double val;
    int    gen;   // generation (which row last used this slot)
} HashSlot;

static CSR *spgemm_hash(const CSR *A, const CSR *AT) {
    int m = A->m;

    // Hash table size: next power of 2 >= 4*m
    // (generous so load factor stays low - fewer collisions)
    int ht_size = next_pow2(4 * m + 1);
    unsigned ht_mask = (unsigned)ht_size - 1;

    HashSlot *ht = malloc(ht_size * sizeof(HashSlot));
    for (int i=0; i<ht_size; i++) { ht[i].key=-1; ht[i].gen=-1; }

    int *cols = malloc(m * sizeof(int)); // column collector

    // We do symbolic + numeric in ONE pass (hash avoids needing 2 passes)
    // But we still need row_ptr -> use a temporary list-of-lists approach

    // Temporary storage: collect all (col,val) per row into a flat buffer
    // Strategy: first do all rows to find nnz, then fill
    int *row_nnz = calloc(m, sizeof(int));

    // Pass 1: symbolic using hash table
    for (int i=0; i<m; i++) {
        int cnt=0;
        for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
            int k=A->col_idx[p];
            for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                int j=AT->col_idx[q];
                // linear probe
                unsigned slot=((unsigned)j*2654435761u)&ht_mask;
                while (ht[slot].gen==i && ht[slot].key!=j)
                    slot=(slot+1)&ht_mask;
                if (ht[slot].gen!=i) {
                    // new entry
                    ht[slot].key=j;
                    ht[slot].gen=i;
                    cols[cnt++]=j;
                }
                // (no value needed in symbolic pass)
            }
        }
        row_nnz[i]=cnt;
    }

    int *rp = malloc((m+1)*sizeof(int));
    rp[0]=0;
    for (int i=0; i<m; i++) rp[i+1]=rp[i]+row_nnz[i];
    int nnz_C=rp[m];
    int *ci = malloc(nnz_C*sizeof(int));
    double *cv = malloc(nnz_C*sizeof(double));

    // Reset generation counter so pass-2 rows appear on a fresh table.
    for (int i=0; i<ht_size; i++) ht[i].gen=-1;

    // Pass 2: numeric
    for (int i=0; i<m; i++) {
        int cnt=0;
        for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
            int k=A->col_idx[p];
            double a_ik=A->values[p];
            for (int q=AT->row_ptr[k]; q<AT->row_ptr[k+1]; q++) {
                int j=AT->col_idx[q];
                double prod=a_ik*AT->values[q];
                unsigned slot=((unsigned)j*2654435761u)&ht_mask;
                while (ht[slot].gen==i && ht[slot].key!=j)
                    slot=(slot+1)&ht_mask;
                if (ht[slot].gen!=i) {
                    ht[slot].key=j;
                    ht[slot].val=prod;
                    ht[slot].gen=i;
                    cols[cnt++]=j;
                } else {
                    ht[slot].val+=prod;
                }
            }
        }

        // sort cols, then write values
        // insertion sort on cols[]
        for (int a=1; a<cnt; a++) {
            int key=cols[a]; int b=a-1;
            while (b>=0 && cols[b]>key) { cols[b+1]=cols[b]; b--; }
            cols[b+1]=key;
        }

        int base=rp[i];
        for (int a=0; a<cnt; a++) {
            int j=cols[a];
            ci[base+a]=j;
            // look up value in hash table
            unsigned slot=((unsigned)j*2654435761u)&ht_mask;
            while (ht[slot].key!=j || ht[slot].gen!=i)
                slot=(slot+1)&ht_mask;
            cv[base+a]=ht[slot].val;
        }
    }

    free(ht); free(cols); free(row_nnz);

    CSR *C=malloc(sizeof(CSR));
    C->m=m; C->n=m; C->nnz=nnz_C;
    C->row_ptr=rp; C->col_idx=ci; C->values=cv;
    return C;
}

//  STRATEGY 3 - HEAP (Row-Merge via Min-Heap)
//
//  How it works:
//    C = A * A^T  means: row i of C = sum over k where A[i,k]!=0
//                        of  a_ik * (row k of A)
//
//    Think of it as MERGING multiple sorted lists (rows of A),
//    each scaled by a_ik.
//
//    Use a min-heap (priority queue) keyed on column index:
//      - Push (col_j, scaled_value, source_row_k, position_in_row_k)
//        for the FIRST element of each contributing row of A^T
//      - Pop minimum col_j -> accumulate value
//      - Push the NEXT element from the same source row
//      - When heap empty: output row i of C is done
//
//    This naturally produces output in sorted column order ->
//    no explicit sort step needed!
//
//  Cost per output row: O(F_i * log(nnz_i)) where F_i = nnz of row i in C
//  Memory: O(nnz_of_row_i) heap - very small per row
//
//  Best for: very sparse output (small F_i), or when sorted output
//            is required and you want to avoid a post-sort step.
//            Also good when m is HUGE (no dense array needed at all).

typedef struct {
    int    col;      // current column index (heap key)
    double val;      // scaled value: a_ik * AT[k, current_pos]
    int    row_k;    // which row of AT we are merging from
    int    pos;      // current position within row_k of AT
    double a_ik;     // scale factor for this row
} HeapEntry;

// Min-heap operations (keyed on col)
static void heap_push(HeapEntry *H, int *sz, HeapEntry e) {
    int i = (*sz)++;
    H[i] = e;
    // sift up
    while (i > 0) {
        int par = (i-1)/2;
        if (H[par].col <= H[i].col) break;
        HeapEntry tmp=H[par]; H[par]=H[i]; H[i]=tmp;
        i=par;
    }
}

static HeapEntry heap_pop(HeapEntry *H, int *sz) {
    HeapEntry top = H[0];
    H[0] = H[--(*sz)];
    // sift down
    int i=0;
    while (1) {
        int l=2*i+1, r=2*i+2, smallest=i;
        if (l<*sz && H[l].col<H[smallest].col) smallest=l;
        if (r<*sz && H[r].col<H[smallest].col) smallest=r;
        if (smallest==i) break;
        HeapEntry tmp=H[i]; H[i]=H[smallest]; H[smallest]=tmp;
        i=smallest;
    }
    return top;
}

static CSR *spgemm_heap(const CSR *A, const CSR *AT) {
    int m = A->m;

    // Upper bound on heap size = max nonzeros in any row of A
    int max_row_nnz = 0;
    for (int i=0; i<m; i++) {
        int rlen = A->row_ptr[i+1]-A->row_ptr[i];
        if (rlen > max_row_nnz) max_row_nnz = rlen;
    }

    HeapEntry *heap = malloc((max_row_nnz+1) * sizeof(HeapEntry));

    // We need a buffer to collect (col,val) pairs because
    // we accumulate equal column indices as we pop.
    // Use dynamic arrays per row.
    int buf_cap = 64;
    int *buf_col  = malloc(buf_cap * sizeof(int));
    double *buf_val = malloc(buf_cap * sizeof(double));

    // Two-pass: symbolic then numeric.
    // Actually with heap we can do single pass - just do numeric directly
    // and store results in a growing flat array, tracking row offsets.

    // Use a resizable flat output array
    int out_cap = A->nnz * 2 + 16;
    int *out_col = malloc(out_cap * sizeof(int));
    double *out_val = malloc(out_cap * sizeof(double));
    int *rp = malloc((m+1)*sizeof(int));
    rp[0]=0;

    for (int i=0; i<m; i++) {
        int hsz=0, cnt=0;

        // push first element of each contributing AT row onto heap
        for (int p=A->row_ptr[i]; p<A->row_ptr[i+1]; p++) {
            int k=A->col_idx[p];
            if (AT->row_ptr[k]==AT->row_ptr[k+1]) continue; // empty AT row
            double a_ik=A->values[p];
            int q=AT->row_ptr[k];
            HeapEntry e;
            e.col   = AT->col_idx[q];
            e.val   = a_ik * AT->values[q];
            e.row_k = k;
            e.pos   = q;
            e.a_ik  = a_ik;
            heap_push(heap, &hsz, e);
        }

        // merge
        while (hsz > 0) {
            HeapEntry e = heap_pop(heap, &hsz);
            int cur_col = e.col;
            double cur_val = e.val;

            // advance this entry to next position in its AT row
            int npos = e.pos + 1;
            if (npos < AT->row_ptr[e.row_k+1]) {
                HeapEntry ne;
                ne.col   = AT->col_idx[npos];
                ne.val   = e.a_ik * AT->values[npos];
                ne.row_k = e.row_k;
                ne.pos   = npos;
                ne.a_ik  = e.a_ik;
                heap_push(heap, &hsz, ne);
            }

            // accumulate all entries with same column (they're next in heap)
            while (hsz>0 && heap[0].col==cur_col) {
                HeapEntry e2 = heap_pop(heap, &hsz);
                cur_val += e2.val;
                int npos2 = e2.pos + 1;
                if (npos2 < AT->row_ptr[e2.row_k+1]) {
                    HeapEntry ne2;
                    ne2.col   = AT->col_idx[npos2];
                    ne2.val   = e2.a_ik * AT->values[npos2];
                    ne2.row_k = e2.row_k;
                    ne2.pos   = npos2;
                    ne2.a_ik  = e2.a_ik;
                    heap_push(heap, &hsz, ne2);
                }
            }

            // store (cur_col, cur_val) in output
            int out_pos = rp[i] + cnt;
            if (out_pos >= out_cap) {
                out_cap *= 2;
                out_col = realloc(out_col, out_cap*sizeof(int));
                out_val = realloc(out_val, out_cap*sizeof(double));
            }
            out_col[out_pos] = cur_col;
            out_val[out_pos] = cur_val;
            cnt++;
        }

        rp[i+1] = rp[i] + cnt;
    }

    free(heap); free(buf_col); free(buf_val);

    int nnz_C = rp[m];
    int *ci = malloc(nnz_C*sizeof(int));
    double *cv = malloc(nnz_C*sizeof(double));
    memcpy(ci, out_col, nnz_C*sizeof(int));
    memcpy(cv, out_val, nnz_C*sizeof(double));
    free(out_col); free(out_val);

    CSR *C = malloc(sizeof(CSR));
    C->m=m; C->n=m; C->nnz=nnz_C;
    C->row_ptr=rp; C->col_idx=ci; C->values=cv;
    return C;
}

//  TIMING HELPER
static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec/1e9;
}

//  MAIN
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix.mtx>\n", argv[0]);
        return 1;
    }

    printf("=== Step 2: SpGEMM Strategy Comparison ===\n");
    printf("Matrix file: %s\n\n", argv[1]);

    // load matrix
    CSR *A = read_mm_csr(argv[1]);
    if (!A) return 1;
    printf("Matrix A:   %d x %d,  nnz = %d\n", A->m, A->n, A->nnz);
    printf("Density:    %.6f\n\n", (double)A->nnz/(A->m*(double)A->n));

    CSR *AT = csr_to_csc(A);  // A^T stored as CSR (rows of A^T = cols of A)

    // Strategy 1: SPA
    printf("--- Strategy 1: SPA (Sparse Accumulator) ---\n");
    printf("  How: dense array acc[m], mark array, reset only touched positions\n");
    double t0 = now_sec();
    CSR *C_spa = spgemm_spa(A, AT);
    double t_spa = now_sec() - t0;
    sort_csr_rows(C_spa);
    printf("  Output nnz:  %d\n", C_spa->nnz);
    printf("  Fill factor: %.2fx\n", (double)C_spa->nnz/A->nnz);
    printf("  Time:        %.6f sec\n\n", t_spa);

    // Strategy 2: HASH
    printf("--- Strategy 2: HASH (Open-Addressing Hash Table) ---\n");
    printf("  How: fixed hash table + generation counter, linear probing\n");
    t0 = now_sec();
    CSR *C_hash = spgemm_hash(A, AT);
    double t_hash = now_sec() - t0;
    sort_csr_rows(C_hash);
    printf("  Output nnz:  %d\n", C_hash->nnz);
    printf("  Fill factor: %.2fx\n", (double)C_hash->nnz/A->nnz);
    printf("  Time:        %.6f sec\n\n", t_hash);

    // Strategy 3: HEAP
    printf("--- Strategy 3: HEAP (Min-Heap Row Merge) ---\n");
    printf("  How: merge sorted AT rows via min-heap, output already sorted\n");
    t0 = now_sec();
    CSR *C_heap = spgemm_heap(A, AT);
    double t_heap = now_sec() - t0;
    printf("  Output nnz:  %d\n", C_heap->nnz);
    printf("  Fill factor: %.2fx\n", (double)C_heap->nnz/A->nnz);
    printf("  Time:        %.6f sec\n\n", t_heap);

    // Correctness Verification
    printf("--- Correctness Verification ---\n");
    int ok_hash = csr_equal(C_spa, C_hash, 1e-10);
    int ok_heap = csr_equal(C_spa, C_heap, 1e-10);
    printf("  SPA  vs HASH: %s\n", ok_hash ? "MATCH" : "MISMATCH");
    printf("  SPA  vs HEAP: %s\n", ok_heap ? "MATCH" : "MISMATCH");

    // Summary Table
    printf("\n=== SUMMARY ===\n");
    printf("%-10s %-12s %-12s %-10s %s\n",
           "Strategy", "Time(sec)", "Speedup", "Output nnz", "Notes");
    printf("%-10s %-12.6f %-12.2f %-10d %s\n",
           "SPA",  t_spa,  1.0, C_spa->nnz,
           "Dense array. Fast for small m.");
    printf("%-10s %-12.6f %-12.2f %-10d %s\n",
           "HASH", t_hash, t_spa/t_hash, C_hash->nnz,
           "Hash table. Good for medium m.");
    printf("%-10s %-12.6f %-12.2f %-10d %s\n",
           "HEAP", t_heap, t_spa/t_heap, C_heap->nnz,
           "Sorted merge. No sort step needed.");

    printf("\n=== When to use which strategy? ===\n");
    printf("  SPA  -> Small m (dense acc fits in cache). Simplest code.\n");
    printf("  HASH -> Large m (hash table << m in memory). Good locality.\n");
    printf("  HEAP -> Very sparse output. Huge m. Needs sorted output.\n");

    // Cleanup
    csr_free(A); csr_free(AT);
    csr_free(C_spa); csr_free(C_hash); csr_free(C_heap);

    printf("\nSUCCESS: Step 2 complete.\n");
    return 0;
}
