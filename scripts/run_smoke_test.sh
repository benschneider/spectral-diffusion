#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

OUT_DIR="${1:-$ROOT_DIR/results/smoke_run}"
RUN_ID="${2:-SMOKE_RUN}"

echo "[1/3] Running smoke training run (config: configs/smoke.yaml, run_id: $RUN_ID)"
python "$ROOT_DIR/train_model.py" \
  --config "$ROOT_DIR/configs/smoke.yaml" \
  --output-dir "$OUT_DIR" \
  --run-id "$RUN_ID"

echo "[2/3] Latest metrics summary entry:"
tail -n 1 "$ROOT_DIR/results/summary.csv"

SUMMARY_REPORT="$OUT_DIR/taguchi_report.csv"
if [ -f "$ROOT_DIR/results/summary.csv" ]; then
  echo "[3/3] Generating Taguchi report (larger-is-better on loss_drop_per_second)."
  python -m src.analysis.taguchi_stats \
    --summary "$ROOT_DIR/results/summary.csv" \
    --metric loss_drop_per_second \
    --mode larger \
    --output "$SUMMARY_REPORT" || true
  if [ -f "$SUMMARY_REPORT" ]; then
    echo "Generated $SUMMARY_REPORT"
    cat "$SUMMARY_REPORT"
  else
    echo "Taguchi report not created (likely insufficient data)."
  fi
fi
