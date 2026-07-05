# Contributing

Thanks for looking at this project. It started as a personal
performance-engineering exercise, so the bar for contributions is
mostly about keeping the same discipline it was built with: measure
before you claim, and verify after you change something.

## Getting set up

```bash
make matrices   # regenerate synthetic .mtx files, fetch real ones
make all        # build every step
make verify     # full pipeline: matrices -> ASan/UBSan -> report -> stability check
```

`make verify` is the thing to run before opening a PR. It exits
non-zero if anything fails, including sanitizer errors.

## Before submitting a change

- **Build clean.** `gcc -O3 -Wall -fopenmp -march=native` should not
  produce warnings on the files you touched.
- **Run it under ASan/UBSan.** `make sanitize` rebuilds every step with
  `-fsanitize=address,undefined` and smoke-tests each one, including at
  a thread count higher than the matrix's row count. That specific
  edge case (more threads than rows) is what caught a real
  heap-buffer-overflow during development; it's worth testing directly
  rather than trusting that "it looks right."
- **Check correctness against the sequential reference**, not just
  that the program runs. Every step compares its parallel result to a
  known-correct sequential computation before printing OK.
- **Re-test after you change something, not just before.** Two of the
  bugs documented in the README (a heap overflow, and an asymmetric
  benchmark measurement) were both found by testing again after a
  change that looked fine on its own. If a number moves in a way you
  don't expect, chase it before writing it off as noise.
- **If you touch timing or benchmark numbers**, run `./run_repeated.sh`
  a few times rather than trusting one run. This machine (and probably
  yours) has enough scheduling noise that a single measurement isn't a
  fact.

## Adding a new benchmark matrix

Don't commit large `.mtx` files directly. Either:

- add it to `gen_matrix.py` with a fixed seed, so it's reproducible
  from the script, or
- add it to `fetch_matrices.sh` if it's a real matrix from the
  SuiteSparse collection.

`.mtx` files and the `matrices/` directory are gitignored on purpose.

## Style

The C here is intentionally plain: minimal comments, no decorative
ASCII banners, explaining the non-obvious "why" rather than the
"what." Match that rather than adding heavier structure. If a
comment doesn't explain something a reader would otherwise be
surprised by, it's probably not needed.

## Reporting a bug

Open an issue with the matrix (or a script to regenerate it), the
command you ran, and what you expected vs. what happened. If it's a
correctness issue, a diff against the sequential reference is the
fastest way to confirm it's real.
