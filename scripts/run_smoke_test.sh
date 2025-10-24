#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

OUT_DIR="${1:-$ROOT_DIR/results/smoke_run}"
RUN_ID="${2:-SMOKE_RUN}"
SUMMARY="$OUT_DIR/summary.csv"
RUN_DIR="$OUT_DIR/runs/$RUN_ID"
SAMPLE_TAG="smoke_samples"

echo "[1/4] Running smoke training run (config: configs/smoke.yaml, run_id: $RUN_ID)"
python "$ROOT_DIR/train_model.py" \
  --config "$ROOT_DIR/configs/smoke.yaml" \
  --output-dir "$OUT_DIR" \
  --run-id "$RUN_ID"

if [[ -f "$SUMMARY" ]]; then
  echo "[2/4] Latest summary entry:"
  tail -n 1 "$SUMMARY"
else
  echo "Summary file not found at $SUMMARY"
fi

echo "[3/4] Generating samples via sample.py (DDIM, 4 steps)"
python "$ROOT_DIR/sample.py" \
  --run-dir "$RUN_DIR" \
  --tag "$SAMPLE_TAG" \
  --sampler-type "ddim" \
  --num-samples 4 \
  --num-steps 4

SAMPLE_DIR="$RUN_DIR/samples/$SAMPLE_TAG"
METADATA_PATH="$SAMPLE_DIR/metadata.json"
if [[ -f "$METADATA_PATH" ]]; then
  echo "Sample metadata ($METADATA_PATH):"
  cat "$METADATA_PATH"
fi

SUMMARY_REPORT="$OUT_DIR/taguchi_report.csv"
if [ -f "$SUMMARY" ]; then
  echo "[4/4] Generating Taguchi report (larger-is-better on loss_drop_per_second)."
  python -m src.analysis.taguchi_stats \
    --summary "$SUMMARY" \
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
