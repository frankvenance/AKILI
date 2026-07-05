#!/usr/bin/env bash
# fetch_matrices.sh - download a curated set of real SuiteSparse matrices
# for Step 6 benchmarking. Matrices are extracted to matrices/<name>.mtx
#
# Source: https://sparse.tamu.edu  (SuiteSparse Matrix Collection)
# Usage:  ./fetch_matrices.sh
set -euo pipefail

BASE="https://suitesparse-collection-website.herokuapp.com/MM"
OUT="matrices"
mkdir -p "$OUT"

# group/name : chosen to span structures (banded, power-law, symmetric)
MATRICES=(
  "HB/1138_bus"        # 1138x1138  small symmetric (power grid)
  "HB/bcsstk16"        # 4884x4884  structural, banded-ish
  "Williams/cant"      # 62451x62451 FEM, clustered
  "SNAP/ca-GrQc"       # 5242x5242  collaboration graph (power-law)
)

for gm in "${MATRICES[@]}"; do
  name="${gm##*/}"
  target="$OUT/${name}.mtx"
  if [[ -f "$target" ]]; then
    echo "[skip] $name already present"
    continue
  fi
  echo "[fetch] $gm"
  tmp="$(mktemp -d)"
  curl -fsSL "$BASE/${gm}.tar.gz" -o "$tmp/${name}.tar.gz"
  tar -xzf "$tmp/${name}.tar.gz" -C "$tmp"
  mv "$tmp/${name}/${name}.mtx" "$target"
  rm -rf "$tmp"
  echo "        -> $target"
done

echo
echo "Done. Run e.g.:"
for gm in "${MATRICES[@]}"; do
  echo "  ./step6 $OUT/${gm##*/}.mtx"
done
