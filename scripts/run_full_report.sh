#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

SYN_DIR="$ROOT_DIR/results/spectral_benchmark"
CIFAR_DIR="$ROOT_DIR/results/spectral_benchmark_cifar"
TAG_DIR="$ROOT_DIR/results/taguchi_full"
FIG_DIR="$ROOT_DIR/docs/figures"

run_synthetic() {
  echo "[1/4] Synthetic benchmark (TinyUNet vs SpectralUNet)"
  bash "$ROOT_DIR/scripts/run_spectral_benchmark.sh"
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
