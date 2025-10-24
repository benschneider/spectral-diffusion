#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

OUT_DIR="${1:-$ROOT_DIR/results/taguchi_smoke}"
SUMMARY="$OUT_DIR/summary.csv"
REPORT="$OUT_DIR/taguchi_report.csv"

echo "Cleaning previous taguchi smoke artifacts in $OUT_DIR"
rm -rf "$OUT_DIR"

mkdir -p "$OUT_DIR"

echo "[1/3] Running Taguchi smoke batch"
python -m src.experiments.run_experiment \
  --config "$ROOT_DIR/configs/taguchi_smoke_base.yaml" \
  --array "$ROOT_DIR/configs/taguchi_smoke_array.csv" \
  --output-dir "$OUT_DIR"

if [[ -f "$SUMMARY" ]]; then
  echo "[2/3] Summary CSV:"
  cat "$SUMMARY"
else
  echo "Summary file not found at $SUMMARY"
fi

echo "[3/3] Generating Taguchi S/N report"
python -m src.analysis.taguchi_stats \
  --summary "$SUMMARY" \
  --metric loss_drop_per_second \
  --mode larger \
  --output "$REPORT"

echo "Taguchi report saved to $REPORT"
cat "$REPORT"
