#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

OUT_DIR="${1:-$ROOT_DIR/results/taguchi_minimal}"
SUMMARY="$OUT_DIR/summary.csv"
REPORT="$OUT_DIR/taguchi_report.csv"

if [[ -d "$OUT_DIR" ]]; then
  echo "Cleaning previous minimal run artifacts in $OUT_DIR"
  rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

echo "[1/3] Running Taguchi minimal comparison"
python -m src.experiments.run_experiment \
  --config "$ROOT_DIR/configs/taguchi_minimal_base.yaml" \
  --array "$ROOT_DIR/configs/taguchi_spectral_array.csv" \
  --output-dir "$OUT_DIR"

echo "[2/3] Summary CSV written to $SUMMARY"
if [[ -f "$SUMMARY" ]]; then
  tail -n +1 "$SUMMARY"
else
  echo "Summary file not found; aborting." >&2
  exit 1
fi

echo "[3/3] Generating Taguchi S/N report"
if [[ -f "$REPORT" ]]; then
  echo "Using auto-generated report at $REPORT"
else
  python -m src.analysis.taguchi_stats \
    --summary "$SUMMARY" \
    --metric loss_drop_per_second \
    --mode larger \
    --output "$REPORT"
fi

echo "Report saved to $REPORT"
cat "$REPORT"
