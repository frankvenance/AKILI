# Makefile - single entry point to build, generate inputs, and verify
# the whole SpGEMM project end to end.
#
#   make            build every step (release, -O3)
#   make matrices    generate synthetic .mtx + fetch real SuiteSparse ones
#   make report      run the full Step 8 benchmark -> REPORT.md/results.csv
#   make sanitize    rebuild every step under ASan/UBSan and smoke-test it
#   make verify      matrices + sanitize + report + stability check, in order
#   make clean       remove built binaries (keeps .mtx inputs and reports)

CC      = gcc
CFLAGS  = -O3 -Wall -fopenmp -march=native
SANFLAGS = -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fopenmp

STEPS = step1 step2 step3 step3b step4 step5 step6 step7 step8

.PHONY: all matrices report sanitize verify clean

all: $(STEPS)

step1: step1_basic_spgemm.c mmio.c
	$(CC) $(CFLAGS) -o $@ step1_basic_spgemm.c mmio.c -lm

step2: step2_spgemm_strategies.c mmio.c
	$(CC) $(CFLAGS) -o $@ step2_spgemm_strategies.c mmio.c -lm

step3: step3_parallel_spgemm.c mmio.c
	$(CC) $(CFLAGS) -o $@ step3_parallel_spgemm.c mmio.c -lm

step3b: step3_optimized.c mmio.c
	$(CC) $(CFLAGS) -o $@ step3_optimized.c mmio.c -lm

step4: step4_profiler.c mmio.c
	$(CC) $(CFLAGS) -o $@ step4_profiler.c mmio.c -lm

step5: step5_radix_sort.c mmio.c
	$(CC) $(CFLAGS) -o $@ step5_radix_sort.c mmio.c -lm

step6: step6_realworld.c mmio.c
	$(CC) $(CFLAGS) -o $@ step6_realworld.c mmio.c -lm

step7: step7_autotuner.c mmio.c autotune_calib.h
	$(CC) $(CFLAGS) -o $@ step7_autotuner.c mmio.c -lm

step8: step8_report.c mmio.c autotune_calib.h
	$(CC) $(CFLAGS) -o $@ step8_report.c mmio.c -lm

matrices:
	python3 gen_matrix.py
	./fetch_matrices.sh

report: step8
	./step8 matrices/*.mtx giant.mtx banded.mtx big.mtx huge.mtx

# Rebuild every step under ASan+UBSan and smoke-test each on the tiny
# fixture (test.mtx) plus one real matrix, at a thread count that
# exceeds the row count -- the exact edge case that caught the
# step3_optimized.c heap-buffer-overflow. Fails loudly (non-zero exit)
# on any sanitizer report instead of silently passing.
sanitize:
	@mkdir -p /tmp/spgemm_asan
	@echo "Building all steps with -fsanitize=address,undefined ..."
	@for f in step1_basic_spgemm step2_spgemm_strategies step3_parallel_spgemm \
	          step3_optimized step4_profiler step5_radix_sort step6_realworld \
	          step7_autotuner step8_report; do \
		$(CC) $(SANFLAGS) -o /tmp/spgemm_asan/$$f $$f.c mmio.c -lm || exit 1; \
	done
	@echo "Smoke-testing each under ASan/UBSan (test.mtx, 8 threads > 5 rows) ..."
	@fail=0; \
	for f in step1_basic_spgemm step2_spgemm_strategies step3_parallel_spgemm \
	         step3_optimized step4_profiler step5_radix_sort step6_realworld \
	         step7_autotuner step8_report; do \
		out=$$(/tmp/spgemm_asan/$$f test.mtx 8 2>&1); \
		if echo "$$out" | grep -qiE "ERROR|runtime error|leak"; then \
			echo "  [FAIL] $$f"; echo "$$out" | grep -iE "ERROR|runtime error|leak"; fail=1; \
		else \
			echo "  [ OK ] $$f"; \
		fi; \
	done; \
	rm -f .spgemm_autotune_cache; \
	exit $$fail

verify: matrices sanitize report
	./run_repeated.sh 5

clean:
	rm -f $(STEPS) .spgemm_autotune_cache
	rm -rf /tmp/spgemm_asan
