#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [[ $# -ge 1 ]]; then
  BASE_DIR="$1"
else
  BASE_DIR="$ROOT_DIR/results/full_report_1024x_$(date +%Y%m%d_%H%M%S)"
fi

SYN_DIR="$BASE_DIR/synthetic"
CIFAR_DIR="$BASE_DIR/cifar"
TAG_DIR="$BASE_DIR/taguchi"
FIG_DIR="$BASE_DIR/figures"

mkdir -p "$SYN_DIR" "$CIFAR_DIR" "$TAG_DIR" "$FIG_DIR"

echo "Full report (1024x1024) root: $BASE_DIR"

describe_run() {
  local config="$1"
  local run_id="$2"
  local variant_label="$3"
  python - "$config" "$run_id" "$variant_label" <<'PY'
import sys
import yaml

config_path, run_id, variant = sys.argv[1:4]
try:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
except FileNotFoundError:
    cfg = {}
data_family = cfg.get("data", {}).get("family", "default")
data_source = cfg.get("data", {}).get("source", "unknown")
data_size = f"{cfg.get('data', {}).get('height', '?')}x{cfg.get('data', {}).get('width', '?')}"
model = cfg.get("model", {}).get("type", "default")
print(f"  • Run {run_id}: {config_path} (source={data_source}, size={data_size}, family={data_family}, model={model}, variant={variant})")
PY
}

run_synthetic() {
  echo "[1/4] Synthetic benchmarks (1024x1024 images)"
  rm -f "$SYN_DIR/summary.csv"
  rm -rf "$SYN_DIR/runs"
  mkdir -p "$SYN_DIR"

  SYNTHETIC_CONFIGS=(
    "benchmark_synthetic_piecewise.yaml"
    "benchmark_synthetic_texture.yaml"
    "benchmark_synthetic_random_field.yaml"
  )

  for config_file in "${SYNTHETIC_CONFIGS[@]}"; do
    family_name=$(basename "$config_file" .yaml | sed 's/benchmark_synthetic_//')
    learnable_config_file="benchmark_synthetic_${family_name}_learnable.yaml"

    # Run TinyUNet (default)
    describe_run "$ROOT_DIR/configs/$config_file" "${family_name}_1024x1024_tiny" "config-default"
    python "$ROOT_DIR/train.py" \
      --config "$ROOT_DIR/configs/$config_file" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_1024x1024_tiny"

    # Run TinyUNet + Learnable Adapter
    describe_run "$ROOT_DIR/configs/$learnable_config_file" "${family_name}_1024x1024_tiny_learnable" "tiny-learnable"
    python "$ROOT_DIR/train.py" \
      --config "$ROOT_DIR/configs/$learnable_config_file" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_1024x1024_tiny_learnable"

    # Run SpectralUNet
    describe_run "$ROOT_DIR/configs/$config_file" "${family_name}_1024x1024_spectral" "spectral"
    python "$ROOT_DIR/train.py" \
      --config "$ROOT_DIR/configs/$config_file" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_1024x1024_spectral" \
      --variant spectral

    # Run SpectralUNetDeep
    describe_run "$ROOT_DIR/configs/$config_file" "${family_name}_1024x1024_spectral_deep" "spectral_deep"
    python "$ROOT_DIR/train.py" \
      --config "$ROOT_DIR/configs/$config_file" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_1024x1024_spectral_deep" \
      --variant spectral_deep

    # Run Pure SpectralUNet (unet_spectral model type)
    describe_run "$ROOT_DIR/configs/$config_file" "${family_name}_1024x1024_unet_spectral" "unet_spectral"
    python "$ROOT_DIR/train.py" \
      --config "$ROOT_DIR/configs/$config_file" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_1024x1024_unet_spectral" \
      --model-type unet_spectral
  done
}

run_cifar() {
  echo "[2/4] CIFAR-10 benchmark (TinyUNet vs SpectralUNet vs Deep)"
  rm -f "$CIFAR_DIR/summary.csv"
  rm -rf "$CIFAR_DIR/runs"
  mkdir -p "$CIFAR_DIR"

  describe_run "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "cifar_1024x1024_tiny" "config-default"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_1024x1024_tiny"

  describe_run "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "cifar_1024x1024_spectral" "spectral"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_1024x1024_spectral" \
    --variant spectral

  describe_run "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "cifar_32x32_spectral_deep" "spectral_deep"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_32x32_spectral_deep" \
    --variant spectral_deep
}

run_taguchi() {
  echo "[3/4] Taguchi sweep"
  rm -f "$TAG_DIR/summary.csv" "$TAG_DIR/taguchi_report.csv"
  rm -rf "$TAG_DIR/runs"
  describe_run "$ROOT_DIR/configs/taguchi_smoke_base.yaml" "taguchi_32x32_sweep" "array:taguchi_spectral_array.csv"
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