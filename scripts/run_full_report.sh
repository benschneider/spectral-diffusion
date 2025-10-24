#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [[ $# -ge 1 ]]; then
  BASE_DIR="$1"
else
  BASE_DIR="$ROOT_DIR/results/full_report_$(date +%Y%m%d_%H%M%S)"
fi

SYN_DIR="$BASE_DIR/synthetic"
CIFAR_DIR="$BASE_DIR/cifar"
TAG_DIR="$BASE_DIR/taguchi"
FIG_DIR="$BASE_DIR/figures"

mkdir -p "$SYN_DIR" "$CIFAR_DIR" "$TAG_DIR" "$FIG_DIR"

run_synthetic() {
  echo "[1/4] Synthetic benchmark (TinyUNet vs SpectralUNet)"
  rm -f "$SYN_DIR/summary.csv"
  rm -rf "$SYN_DIR/runs"
  mkdir -p "$SYN_DIR"

  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral.yaml" \
    --output-dir "$SYN_DIR" \
    --run-id "synthetic_tiny"

  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral.yaml" \
    --output-dir "$SYN_DIR" \
    --run-id "synthetic_spectral" \
    --variant spectral
}

run_cifar() {
  echo "[2/4] CIFAR-10 benchmark"
  rm -f "$CIFAR_DIR/summary.csv"
  rm -rf "$CIFAR_DIR/runs"
  mkdir -p "$CIFAR_DIR"

  echo "  • Training TinyUNet baseline"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_baseline"

  echo "  • Training SpectralUNet"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_spectral" \
    --variant spectral
}

run_taguchi() {
  echo "[3/4] Taguchi sweep"
  rm -f "$TAG_DIR/summary.csv" "$TAG_DIR/taguchi_report.csv"
  rm -rf "$TAG_DIR/runs"
  python -m src.experiments.run_experiment \
    --config "$ROOT_DIR/configs/taguchi_smoke_base.yaml" \
    --array "$ROOT_DIR/configs/taguchi_spectral_array.csv" \
    --output-dir "$TAG_DIR" \
    --report-metric loss_drop_per_second \
    --report-mode larger
}

generate_report() {
  echo "[4/4] Generating figures & summary"
  python "$ROOT_DIR/scripts/figures/clean_summaries.py" \
    "$SYN_DIR/summary.csv" \
    "$CIFAR_DIR/summary.csv"
  python "$ROOT_DIR/scripts/figures/generate_figures.py" \
    --synthetic-dir "$SYN_DIR" \
    --cifar-dir "$CIFAR_DIR" \
    --taguchi-dir "$TAG_DIR" \
    --output-dir "$FIG_DIR"
  echo "Report written to $FIG_DIR/summary.md"
}

run_synthetic
run_cifar
run_taguchi
generate_report

echo "Done. Inspect $FIG_DIR for figures and summary."
