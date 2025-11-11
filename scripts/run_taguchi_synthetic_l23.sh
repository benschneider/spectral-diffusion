#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

OUT_DIR="${1:-$ROOT_DIR/results/taguchi_l23_synthetic}"
DESIGN="${2:-$ROOT_DIR/configs/taguchi/L23_synthetic.csv}"
BASE_CONFIG="${3:-$ROOT_DIR/configs/taguchi/L23_synthetic.yaml}"
STEPS="${TAGUCHI_L23_STEPS:-120}"
LOG_INTERVAL="${TAGUCHI_L23_LOG_INTERVAL:-10}"

if [[ -d "$OUT_DIR" ]]; then
  echo "Cleaning previous Taguchi L23 artifacts in $OUT_DIR"
  rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

CMD=("$ROOT_DIR/scripts/run_taguchi_synthetic_l23.py"
  --design "$DESIGN"
  --base-config "$BASE_CONFIG"
  --output-dir "$OUT_DIR"
  --steps "$STEPS"
  --log-interval "$LOG_INTERVAL")

if [[ "${TAGUCHI_L23_DRY_RUN:-0}" == "1" ]]; then
  CMD+=("--dry-run")
fi

python "${CMD[@]}"

RESOLVED="$OUT_DIR/L23_synthetic_resolved.csv"
RESULTS="$OUT_DIR/results.csv"

if [[ -f "$RESOLVED" ]]; then
  echo "Resolved CLI table saved to $RESOLVED"
else
  echo "Resolved CLI table missing (check runner output)" >&2
fi

if [[ -f "$RESULTS" ]]; then
  echo "Aggregated metrics saved to $RESULTS"
fi
