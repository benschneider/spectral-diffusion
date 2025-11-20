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

IFS=' ' read -r -a BENCHMARK_SEEDS <<< "${BENCHMARK_SEEDS:-0 1 2}"
echo "Benchmark seeds: ${BENCHMARK_SEEDS[*]}"

run_variant() {
  local variant_label="$1"
  shift || true
  local variant_args=("$@")
  for seed in "${BENCHMARK_SEEDS[@]}"; do
    local run_id="${RUN_PREFIX}_${variant_label}_seed${seed}"
    echo "Running ${variant_label} (seed=${seed})"
    python "$ROOT_DIR/train.py" \
      --config "$CONFIG" \
      --output-dir "$OUT_DIR" \
      --run-id "$run_id" \
      --seed "$seed" \
      "${variant_args[@]}"
  done
}

echo "[1/5] Running baseline (TinyUNet) training across seeds"
run_variant "baseline"

echo "[2/5] Running spectral UNet training across seeds"
run_variant "spectral" --variant spectral

if [[ -f "$SUMMARY" ]]; then
  echo "[3/5] Summary entries (tail)"
  tail -n 2 "$SUMMARY"
else
  echo "Summary file not found at $SUMMARY" >&2
fi

echo "[4/5] Aggregated stats (mean/std/CI per variant)"
python "$ROOT_DIR/scripts/aggregate_benchmark_runs.py" \
  --summary "$SUMMARY" \
  --run-prefix "$RUN_PREFIX"

echo "[5/5] Benchmark comparison (sorted by loss_drop):"
python "$ROOT_DIR/scripts/report_summary.py" \
  --summary "$SUMMARY" \
  --metric loss_drop \
  --top 2

echo "Done. Artifacts stored under $OUT_DIR/runs/${RUN_PREFIX}_*"
