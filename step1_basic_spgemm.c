// step1_basic_spgemm.c
// Goal: Read a Matrix Market file, do Gustavson SpGEMM (C = A * A^T), report nnz + timing.
// No optimizations, no strategies. Correctness first.
//
// Compile:  gcc -O2 -Wall -o step1 step1_basic_spgemm.c mmio.c -lm
// Run:      ./step1 test.mtx

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mmio.h"

typedef struct {
    int m, n, nnz;
    int *row_ptr;    // size m+1
    int *col_idx;    // size nnz
    double *values;  // size nnz, NULL if pattern-only
} csr_matrix;

static void free_csr(csr_matrix *M) {
    if (!M) return;
    free(M->row_ptr);
    free(M->col_idx);
    free(M->values);
    free(M);
}

// Read Matrix Market file (coordinate format) into CSR.
// Handles: real/pattern, general/symmetric.
static csr_matrix *read_mm_to_csr(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filename);
        return NULL;
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "ERROR: Not a Matrix Market file: %s\n", filename);
        fclose(f);
        return NULL;
    }

    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
        fprintf(stderr, "ERROR: Must be a sparse coordinate matrix\n");
        fclose(f);
        return NULL;
    }

    int M_rows, N_cols, nz;
    if (mm_read_mtx_crd_size(f, &M_rows, &N_cols, &nz) != 0) {
        fprintf(stderr, "ERROR: Cannot read matrix size\n");
        fclose(f);
        return NULL;
    }

    int symmetric = mm_is_symmetric(matcode);
    // Upper bound on entries if symmetric (off-diagonal entries get mirrored)
    int max_entries = symmetric ? (2 * nz) : nz;

    int *I = malloc(max_entries * sizeof(int));
    int *J = malloc(max_entries * sizeof(int));
    double *val = malloc(max_entries * sizeof(double));
    int count = 0;

    for (int e = 0; e < nz; e++) {
        int r, c;
        double v = 1.0;
        int ok;
        if (mm_is_pattern(matcode)) {
            ok = fscanf(f, "%d %d", &r, &c);
        } else {
            ok = fscanf(f, "%d %d %lg", &r, &c, &v);
        }
        if (ok < 2) {
            fprintf(stderr, "ERROR: Malformed entry at line %d\n", e);
            fclose(f);
            free(I); free(J); free(val);
            return NULL;
        }
        r--; c--; // 1-based -> 0-based

        I[count] = r; J[count] = c; val[count] = v;
        count++;

        if (symmetric && r != c) {
            I[count] = c; J[count] = r; val[count] = v;
            count++;
        }
    }
    fclose(f);
    nz = count;

    // COO -> CSR
    int *row_counts = calloc(M_rows, sizeof(int));
    for (int e = 0; e < nz; e++) row_counts[I[e]]++;

    int *row_ptr = malloc((M_rows + 1) * sizeof(int));
    row_ptr[0] = 0;
    for (int i = 0; i < M_rows; i++) row_ptr[i+1] = row_ptr[i] + row_counts[i];

    int *col_idx = malloc(nz * sizeof(int));
    double *values = malloc(nz * sizeof(double));
    int *cursor = calloc(M_rows, sizeof(int));

    for (int e = 0; e < nz; e++) {
        int r = I[e];
        int pos = row_ptr[r] + cursor[r];
        col_idx[pos] = J[e];
        values[pos] = val[e];
        cursor[r]++;
    }

    free(row_counts); free(cursor);
    free(I); free(J); free(val);

    csr_matrix *M = malloc(sizeof(csr_matrix));
    M->m = M_rows; M->n = N_cols; M->nnz = nz;
    M->row_ptr = row_ptr; M->col_idx = col_idx; M->values = values;
    return M;
}

// CSR -> CSC (general conversion, works for any A)
static csr_matrix *csr_to_csc(const csr_matrix *A) {
    int m = A->m, n = A->n, nnz = A->nnz;

    int *col_counts = calloc(n, sizeof(int));
    for (int e = 0; e < nnz; e++) col_counts[A->col_idx[e]]++;

    int *col_ptr = malloc((n + 1) * sizeof(int));
    col_ptr[0] = 0;
    for (int j = 0; j < n; j++) col_ptr[j+1] = col_ptr[j] + col_counts[j];

    int *row_idx = malloc(nnz * sizeof(int));
    double *values = malloc(nnz * sizeof(double));
    int *cursor = calloc(n, sizeof(int));

    for (int i = 0; i < m; i++) {
        for (int p = A->row_ptr[i]; p < A->row_ptr[i+1]; p++) {
            int j = A->col_idx[p];
            int pos = col_ptr[j] + cursor[j];
            row_idx[pos] = i;
            values[pos] = A->values[p];
            cursor[j]++;
        }
    }
    free(col_counts); free(cursor);

    // Reuse csr_matrix struct: this represents CSC as "rows"=columns of A
    csr_matrix *CSC = malloc(sizeof(csr_matrix));
    CSC->m = n; CSC->n = m; CSC->nnz = nnz;
    CSC->row_ptr = col_ptr; CSC->col_idx = row_idx; CSC->values = values;
    return CSC;
}

// Gustavson: C = A * A^T, A is m x n (CSR). C is m x m.
// AT_csc is the CSC representation of A (columns of A = "rows" of AT_csc).
static csr_matrix *gustavson_square(const csr_matrix *A) {
    int m = A->m;
    csr_matrix *AT_csc = csr_to_csc(A); // AT_csc->row_ptr indexed by A's column dim

    int *mark = malloc(m * sizeof(int));
    double *dense_acc = calloc(m, sizeof(double));
    for (int i = 0; i < m; i++) mark[i] = -1;

    // Pass 1: symbolic (count nnz per output row)
    int *row_nnz = calloc(m, sizeof(int));
    for (int i = 0; i < m; i++) {
        int count = 0;
        for (int p = A->row_ptr[i]; p < A->row_ptr[i+1]; p++) {
            int k = A->col_idx[p];              // inner dim index (column of A)
            for (int q = AT_csc->row_ptr[k]; q < AT_csc->row_ptr[k+1]; q++) {
                int j = AT_csc->col_idx[q];      // row index that also has nonzero at k
                if (mark[j] != i) { mark[j] = i; count++; }
            }
        }
        row_nnz[i] = count;
    }

    int *C_row_ptr = malloc((m + 1) * sizeof(int));
    C_row_ptr[0] = 0;
    for (int i = 0; i < m; i++) C_row_ptr[i+1] = C_row_ptr[i] + row_nnz[i];
    int nnz_C = C_row_ptr[m];

    int *C_col_idx = malloc(nnz_C * sizeof(int));
    double *C_values = malloc(nnz_C * sizeof(double));

    for (int i = 0; i < m; i++) mark[i] = -1;
    int next_idx = 0;

    for (int i = 0; i < m; i++) {
        int row_start = next_idx;
        for (int p = A->row_ptr[i]; p < A->row_ptr[i+1]; p++) {
            int k = A->col_idx[p];
            double a_val = A->values[p];
            for (int q = AT_csc->row_ptr[k]; q < AT_csc->row_ptr[k+1]; q++) {
                int j = AT_csc->col_idx[q];
                double at_val = AT_csc->values[q];
                double prod = a_val * at_val;
                if (mark[j] != i) {
                    mark[j] = i;
                    dense_acc[j] = prod;
                    C_col_idx[next_idx++] = j;
                } else {
                    dense_acc[j] += prod;
                }
            }
        }
        int row_len = next_idx - row_start;
        // insertion sort column indices within this row (small rows -> fine)
        for (int a = 1; a < row_len; a++) {
            int key_idx = C_col_idx[row_start + a];
            double key_val = dense_acc[key_idx];
            int b = a - 1;
            while (b >= 0 && C_col_idx[row_start + b] > key_idx) {
                C_col_idx[row_start + b + 1] = C_col_idx[row_start + b];
                b--;
            }
            C_col_idx[row_start + b + 1] = key_idx;
            (void)key_val; // values extracted separately below
        }
        for (int p = 0; p < row_len; p++) {
            int j = C_col_idx[row_start + p];
            C_values[row_start + p] = dense_acc[j];
            dense_acc[j] = 0.0;
        }
    }

    free(mark); free(dense_acc); free(row_nnz);
    free_csr(AT_csc);

    csr_matrix *C = malloc(sizeof(csr_matrix));
    C->m = m; C->n = m; C->nnz = nnz_C;
    C->row_ptr = C_row_ptr; C->col_idx = C_col_idx; C->values = C_values;
    return C;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix.mtx>\n", argv[0]);
        return 1;
    }

    printf("=== Step 1: Basic SpGEMM (A * A^T) ===\n");
    printf("Matrix file: %s\n", argv[1]);

    csr_matrix *A = read_mm_to_csr(argv[1]);
    if (!A) { fprintf(stderr, "FAILED: Could not read matrix\n"); return 1; }

    printf("Matrix loaded: %d x %d, %d nonzeros\n", A->m, A->n, A->nnz);
    printf("Density: %.8f\n", (double)A->nnz / ((double)A->m * (double)A->n));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    csr_matrix *C = gustavson_square(A);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("Result: %d x %d, %d nonzeros\n", C->m, C->n, C->nnz);
    printf("Fill factor: %.2fx\n", (double)C->nnz / A->nnz);
    printf("SpGEMM time: %.6f seconds\n", elapsed);

    free_csr(A);
    free_csr(C);

    printf("SUCCESS: Basic SpGEMM working\n");
    return 0;
}
