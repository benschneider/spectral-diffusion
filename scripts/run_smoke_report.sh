#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )"/.. && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ $# -ge 1 ]]; then
  BASE_DIR="${1%/}"
  REPORT_ROOT="$BASE_DIR/$TIMESTAMP"
else
  REPORT_ROOT="$ROOT_DIR/results/smoke_report_$TIMESTAMP"
fi
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

SYN_DIR="$REPORT_ROOT/synthetic"
CIFAR_DIR="$REPORT_ROOT/cifar"
TAG_DIR="$REPORT_ROOT/taguchi"
FIG_DIR="$REPORT_ROOT/figures"

mkdir -p "$SYN_DIR" "$CIFAR_DIR" "$TAG_DIR" "$FIG_DIR"

run_synthetic() {
  echo "[1/4] Synthetic smoke benchmark"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_smoke.yaml" \
    --output-dir "$SYN_DIR" \
    --run-id "synthetic_tiny"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_smoke.yaml" \
    --output-dir "$SYN_DIR" \
    --run-id "synthetic_spectral" \
    --variant spectral
}

run_cifar() {
  echo "[2/4] CIFAR-10 smoke benchmark"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar_smoke.yaml" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_tiny"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar_smoke.yaml" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_spectral" \
    --variant spectral
}

run_taguchi() {
  echo "[3/4] Taguchi smoke sweep"
  python -m src.experiments.run_experiment \
    --config "$ROOT_DIR/configs/taguchi_smoke_fast.yaml" \
    --array "$ROOT_DIR/configs/taguchi_spectral_array.csv" \
    --output-dir "$TAG_DIR" \
    --report-metric loss_drop_per_second \
    --report-mode larger
}

generate_report() {
  echo "[4/4] Generating smoke report"
  python "$ROOT_DIR/scripts/figures/generate_figures.py" \
    --report-root "$REPORT_ROOT" \
    --output-dir "$FIG_DIR"
  echo "Smoke report available at $FIG_DIR/summary.md"
  echo "Run artifacts stored in $REPORT_ROOT"
}

run_synthetic
run_cifar
run_taguchi
generate_report
