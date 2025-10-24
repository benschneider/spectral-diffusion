#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

OUT_DIR="${1:-$ROOT_DIR/results/spectral_benchmark}"
CONFIG="${2:-$ROOT_DIR/configs/benchmark_spectral.yaml}"
RUN_PREFIX="${3:-SPECTRAL_BENCH}"
SUMMARY="$OUT_DIR/summary.csv"

mkdir -p "$OUT_DIR"

if [[ -d "$OUT_DIR/runs" ]]; then
  echo "Cleaning previous benchmark runs in $OUT_DIR"
  rm -rf "$OUT_DIR/runs"
fi

echo "[1/4] Running baseline (TinyUNet) training"
python "$ROOT_DIR/train.py" \
  --config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --run-id "${RUN_PREFIX}_baseline"

echo "[2/4] Running spectral UNet training"
python "$ROOT_DIR/train.py" \
  --config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --run-id "${RUN_PREFIX}_spectral" \
  --variant spectral

if [[ -f "$SUMMARY" ]]; then
  echo "[3/4] Summary entries (tail)"
  tail -n 2 "$SUMMARY"
else
  echo "Summary file not found at $SUMMARY" >&2
fi

echo "[4/4] Benchmark comparison (sorted by loss_drop):"
python "$ROOT_DIR/scripts/report_summary.py" \
  --summary "$SUMMARY" \
  --metric loss_drop \
  --top 2

echo "Done. Artifacts stored under $OUT_DIR/runs/${RUN_PREFIX}_*"
