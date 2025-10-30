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

augment_config() {
  local source_cfg="$1"
  local dest_cfg="$2"
  python - "$source_cfg" "$dest_cfg" <<'PY'
import sys
import yaml

src, dst = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

diffusion = cfg.setdefault("diffusion", {})
diffusion["uniform_corruption"] = True

model = cfg.setdefault("model", {})
model["enable_amp_residual"] = True
model["enable_phase_attention"] = True
model.setdefault("amp_hidden_dim", max(int(model.get("base_channels", 32)), 16))
model.setdefault("phase_heads", 1)

sampling = cfg.setdefault("sampling", {})
sampling.setdefault("sampler_type", "masf")
sampling.setdefault("num_steps", 50)
sampling.setdefault("num_samples", 8)

with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
}

sample_with_masf() {
  local run_root="$1"
  local tag="${2:-masf}"
  if [[ ! -d "$run_root" ]]; then
    echo "Skipping MASF sampling; run directory not found: $run_root"
    return
  fi
  python "$ROOT_DIR/sample.py" \
    --run-dir "$run_root" \
    --sampler-type masf \
    --tag "$tag" \
    --num-steps 50 \
    --num-samples 8
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
    tmp_cfg="$(mktemp "$SYN_DIR/${family_name}_1024_spectral_XXXX.yaml")"
    augment_config "$ROOT_DIR/configs/$config_file" "$tmp_cfg"
    describe_run "$tmp_cfg" "${family_name}_1024x1024_spectral" "spectral+uniform"
    python "$ROOT_DIR/train.py" \
      --config "$tmp_cfg" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_1024x1024_spectral" \
      --variant spectral
    sample_with_masf "$SYN_DIR/runs/${family_name}_1024x1024_spectral"

    # Run SpectralUNetDeep
    tmp_cfg_deep="$(mktemp "$SYN_DIR/${family_name}_1024_deep_XXXX.yaml")"
    augment_config "$ROOT_DIR/configs/$config_file" "$tmp_cfg_deep"
    describe_run "$tmp_cfg_deep" "${family_name}_1024x1024_spectral_deep" "spectral_deep+uniform"
    python "$ROOT_DIR/train.py" \
      --config "$tmp_cfg_deep" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_1024x1024_spectral_deep" \
      --variant spectral_deep
    sample_with_masf "$SYN_DIR/runs/${family_name}_1024x1024_spectral_deep" "masf_deep"

    # Run Pure SpectralUNet (unet_spectral model type)
    tmp_cfg_unet="$(mktemp "$SYN_DIR/${family_name}_1024_unet_XXXX.yaml")"
    augment_config "$ROOT_DIR/configs/$config_file" "$tmp_cfg_unet"
    describe_run "$tmp_cfg_unet" "${family_name}_1024x1024_unet_spectral" "unet_spectral+uniform"
    python "$ROOT_DIR/train.py" \
      --config "$tmp_cfg_unet" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_1024x1024_unet_spectral" \
      --variant unet_spectral
    sample_with_masf "$SYN_DIR/runs/${family_name}_1024x1024_unet_spectral" "masf_unet"
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

  tmp_cfg_cifar="$(mktemp "$CIFAR_DIR/cifar_1024_spectral_XXXX.yaml")"
  augment_config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "$tmp_cfg_cifar"
  describe_run "$tmp_cfg_cifar" "cifar_1024x1024_spectral" "spectral+uniform"
  python "$ROOT_DIR/train.py" \
    --config "$tmp_cfg_cifar" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_1024x1024_spectral" \
    --variant spectral
  sample_with_masf "$CIFAR_DIR/runs/cifar_1024x1024_spectral"

  tmp_cfg_cifar_deep="$(mktemp "$CIFAR_DIR/cifar_1024_deep_XXXX.yaml")"
  augment_config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "$tmp_cfg_cifar_deep"
  describe_run "$tmp_cfg_cifar_deep" "cifar_1024x1024_spectral_deep" "spectral_deep+uniform"
  python "$ROOT_DIR/train.py" \
    --config "$tmp_cfg_cifar_deep" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_1024x1024_spectral_deep" \
    --variant spectral_deep
  sample_with_masf "$CIFAR_DIR/runs/cifar_1024x1024_spectral_deep" "masf_deep"
}

run_taguchi() {
  echo "[3/4] Taguchi sweep"
  rm -f "$TAG_DIR/summary.csv" "$TAG_DIR/taguchi_report.csv"
  rm -rf "$TAG_DIR/runs"
  describe_run "$ROOT_DIR/configs/taguchi_smoke_base.yaml" "taguchi_1024x1024_sweep" "array:taguchi_spectral_array.csv"
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
