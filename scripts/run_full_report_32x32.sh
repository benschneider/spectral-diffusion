#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

TAGUCHI_ARRAY_PATH=${TAGUCHI_ARRAY_PATH:-"$ROOT_DIR/configs/taguchi/L27_extended.csv"}
TAGUCHI_RANDOMIZE=${TAGUCHI_RANDOMIZE:-true}
TAGUCHI_MAPPING_SEED=${TAGUCHI_MAPPING_SEED:-$(date +%s)}
TAGUCHI_JOBS=${TAGUCHI_JOBS:-0}
TAGUCHI_REPORT_METRIC=${TAGUCHI_REPORT_METRIC:-loss_drop_per_second}
TAGUCHI_REPORT_MODE=${TAGUCHI_REPORT_MODE:-larger}

if [[ $# -ge 1 ]]; then
  BASE_DIR="$1"
else
  BASE_DIR="$ROOT_DIR/results/full_report_32x32_$(date +%Y%m%d_%H%M%S)"
fi

SYN_DIR="$BASE_DIR/synthetic"
CIFAR_DIR="$BASE_DIR/cifar"
TAG_DIR="$BASE_DIR/taguchi"
FIG_DIR="$BASE_DIR/figures"
ABL_DIR="$BASE_DIR/ablation"

mkdir -p "$SYN_DIR" "$CIFAR_DIR" "$TAG_DIR" "$FIG_DIR" "$ABL_DIR"

echo "Full report (32x32) root: $BASE_DIR"

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

default_parallelism() {
  python - <<'PY'
import os

count = os.cpu_count() or 1
# Aim for a conservative default parallelism that still provides speedup.
print(max(1, min(count, 4)))
PY
}

collect_taguchi_rows() {
  local array_path="$1"
  python - "$array_path" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Design matrix not found: {path}")

with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    priority = ("run", "row", "index", "id")
    key = next((name for name in priority if name in fieldnames), None)
    rows = []
    for idx, entry in enumerate(reader, start=1):
        if key is not None:
            raw_value = entry.get(key)
            if raw_value is not None and str(raw_value).strip():
                try:
                    rows.append(int(float(str(raw_value))))
                    continue
                except ValueError:
                    pass
        rows.append(idx)

if not rows:
    raise SystemExit("No Taguchi rows discovered in the design matrix.")

print("\n".join(str(value) for value in rows))
PY
}

wait_for_taguchi_jobs() {
  local -n __pids_ref=$1
  local -n __rows_ref=$2
  local i pid row
  for i in "${!__pids_ref[@]}"; do
    pid="${__pids_ref[$i]}"
    row="${__rows_ref[$i]}"
    if ! wait "$pid"; then
      echo "Taguchi row $row failed (PID $pid)." >&2
      exit 1
    fi
  done
  __pids_ref=()
  __rows_ref=()
}

finalize_taguchi_outputs() {
  local summary_csv="$1"
  local report_csv="$2"
  local metric="$3"
  local mode="$4"
  if [[ ! -f "$summary_csv" ]]; then
    echo "Taguchi summary not found at $summary_csv; skipping report generation." >&2
    return
  fi
  python - "$summary_csv" "$report_csv" "$metric" "$mode" <<'PY'
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
metric = sys.argv[3]
mode = sys.argv[4]

try:
    from src.analysis.taguchi_stats import generate_taguchi_report
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise SystemExit(f"Unable to generate Taguchi report: {exc}")

report_path.parent.mkdir(parents=True, exist_ok=True)
report = generate_taguchi_report(
    summary_path=summary_path,
    metric=metric,
    mode=mode,
    output_path=report_path,
)
print(f"Generated Taguchi report with {len(report)} rows -> {report_path}")
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

sample_with_ddim() {
  local run_root="$1"
  local tag="${2:-ddim}"
  if [[ ! -d "$run_root" ]]; then
    echo "Skipping DDIM sampling; run directory not found: $run_root"
    return
  fi
  python "$ROOT_DIR/sample.py" \
    --run-dir "$run_root" \
    --sampler-type ddim \
    --tag "$tag" \
    --num-steps 50 \
    --num-samples 8
}

run_synthetic() {
  echo "[1/5] Synthetic benchmarks (32x32 images)"
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
    describe_run "$ROOT_DIR/configs/$config_file" "${family_name}_32x32_tiny" "config-default"
    python "$ROOT_DIR/train.py" \
      --config "$ROOT_DIR/configs/$config_file" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_32x32_tiny"
    sample_with_ddim "$SYN_DIR/runs/${family_name}_32x32_tiny" "ddim_baseline"

    # Run TinyUNet + Learnable Adapter
    describe_run "$ROOT_DIR/configs/$learnable_config_file" "${family_name}_32x32_tiny_learnable" "tiny-learnable"
    python "$ROOT_DIR/train.py" \
      --config "$ROOT_DIR/configs/$learnable_config_file" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_32x32_tiny_learnable"

    # Run SpectralUNet
    tmp_cfg="$(mktemp "$SYN_DIR/${family_name}_32_spectral_XXXX.yaml")"
    augment_config "$ROOT_DIR/configs/$config_file" "$tmp_cfg"
    describe_run "$tmp_cfg" "${family_name}_32x32_spectral" "spectral+uniform"
    python "$ROOT_DIR/train.py" \
      --config "$tmp_cfg" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_32x32_spectral" \
      --variant spectral
    sample_with_masf "$SYN_DIR/runs/${family_name}_32x32_spectral"

    # Run SpectralUNetDeep
    tmp_cfg_deep="$(mktemp "$SYN_DIR/${family_name}_32_deep_XXXX.yaml")"
    augment_config "$ROOT_DIR/configs/$config_file" "$tmp_cfg_deep"
    describe_run "$tmp_cfg_deep" "${family_name}_32x32_spectral_deep" "spectral_deep+uniform"
    python "$ROOT_DIR/train.py" \
      --config "$tmp_cfg_deep" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_32x32_spectral_deep" \
      --variant spectral_deep
    sample_with_masf "$SYN_DIR/runs/${family_name}_32x32_spectral_deep" "masf_deep"

    # Run Pure SpectralUNet (unet_spectral model type)
    tmp_cfg_unet="$(mktemp "$SYN_DIR/${family_name}_32_unet_XXXX.yaml")"
    augment_config "$ROOT_DIR/configs/$config_file" "$tmp_cfg_unet"
    describe_run "$tmp_cfg_unet" "${family_name}_32x32_unet_spectral" "unet_spectral+uniform"
    python "$ROOT_DIR/train.py" \
      --config "$tmp_cfg_unet" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_32x32_unet_spectral" \
      --variant unet_spectral
    sample_with_masf "$SYN_DIR/runs/${family_name}_32x32_unet_spectral" "masf_unet"
  done
}

run_cifar() {
  echo "[2/5] CIFAR-10 benchmark (TinyUNet vs SpectralUNet vs Deep)"
  rm -f "$CIFAR_DIR/summary.csv"
  rm -rf "$CIFAR_DIR/runs"
  mkdir -p "$CIFAR_DIR"

  describe_run "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "cifar_32x32_tiny" "config-default"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_32x32_tiny"
  sample_with_ddim "$CIFAR_DIR/runs/cifar_32x32_tiny" "ddim_baseline"

  tmp_cfg_cifar="$(mktemp "$CIFAR_DIR/cifar_32_spectral_XXXX.yaml")"
  augment_config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "$tmp_cfg_cifar"
  describe_run "$tmp_cfg_cifar" "cifar_32x32_spectral" "spectral+uniform"
  python "$ROOT_DIR/train.py" \
    --config "$tmp_cfg_cifar" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_32x32_spectral" \
    --variant spectral
  sample_with_masf "$CIFAR_DIR/runs/cifar_32x32_spectral"

  tmp_cfg_cifar_deep="$(mktemp "$CIFAR_DIR/cifar_32_deep_XXXX.yaml")"
  augment_config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "$tmp_cfg_cifar_deep"
  describe_run "$tmp_cfg_cifar_deep" "cifar_32x32_spectral_deep" "spectral_deep+uniform"
  python "$ROOT_DIR/train.py" \
    --config "$tmp_cfg_cifar_deep" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_32x32_spectral_deep" \
    --variant spectral_deep
  sample_with_masf "$CIFAR_DIR/runs/cifar_32x32_spectral_deep" "masf_deep"
}

run_feature_ablation() {
  echo "[3/5] Spectral feature toggle ablation"
  rm -f "$ABL_DIR/summary.csv"
  mkdir -p "$ABL_DIR"

  # Baseline spectral without additional toggles
  describe_run "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "cifar_32x32_spectral_plain" "spectral"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --output-dir "$ABL_DIR" \
    --run-id "cifar_32x32_spectral_plain" \
    --variant spectral

  tmp_cfg_ablation="$(mktemp "$ABL_DIR/cifar_32_ablation_XXXX.yaml")"
  augment_config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "$tmp_cfg_ablation"
  describe_run "$tmp_cfg_ablation" "cifar_32x32_spectral_uniform" "spectral+uniform"
  python "$ROOT_DIR/train.py" \
    --config "$tmp_cfg_ablation" \
    --output-dir "$ABL_DIR" \
    --run-id "cifar_32x32_spectral_uniform" \
    --variant spectral
}

run_taguchi() {
  echo "[4/5] Taguchi sweep"
  local array_basename
  array_basename="$(basename "$TAGUCHI_ARRAY_PATH")"
  local array_stem="${array_basename%.csv}"
  rm -f "$TAG_DIR/summary.csv" \
        "$TAG_DIR/taguchi_report.csv" \
        "$TAG_DIR/${array_stem}_summary.csv" \
        "$TAG_DIR"/run_*_metrics.json
  rm -rf "$TAG_DIR/runs"
  describe_run "$ROOT_DIR/configs/taguchi_smoke_base.yaml" "taguchi_32x32_sweep" "array:${array_basename}"

  mapfile -t taguchi_rows < <(collect_taguchi_rows "$TAGUCHI_ARRAY_PATH")
  local total_rows="${#taguchi_rows[@]}"
  if [[ "$total_rows" -eq 0 ]]; then
    echo "No Taguchi rows discovered in $TAGUCHI_ARRAY_PATH" >&2
    exit 1
  fi

  local parallelism="$TAGUCHI_JOBS"
  if ! [[ "$parallelism" =~ ^[0-9]+$ ]]; then
    parallelism=0
  fi
  if (( parallelism <= 0 )); then
    parallelism="$(default_parallelism)"
  fi
  if ! [[ "$parallelism" =~ ^[0-9]+$ ]]; then
    parallelism=1
  fi
  if (( parallelism <= 0 )); then
    parallelism=1
  fi

  echo "  • Executing $total_rows Taguchi rows with up to $parallelism concurrent python run(s)"

  local -a base_cmd=(
    python -m src.experiments.run_experiment
    --config "$ROOT_DIR/configs/taguchi_smoke_base.yaml"
    --array "$TAGUCHI_ARRAY_PATH"
    --output-dir "$TAG_DIR"
    --report-metric "$TAGUCHI_REPORT_METRIC"
    --report-mode "$TAGUCHI_REPORT_MODE"
    --factor-registry "$ROOT_DIR/configs/taguchi/factor_registry.yaml"
  )
  if [[ "${TAGUCHI_RANDOMIZE}" == "true" || "${TAGUCHI_RANDOMIZE}" == "1" ]]; then
    base_cmd+=(--randomize-mapping --seed "$TAGUCHI_MAPPING_SEED")
  fi

  local -a active_pids=()
  local -a active_rows=()
  local idx=0
  local row
  for row in "${taguchi_rows[@]}"; do
    idx=$((idx + 1))
    echo "    - [${idx}/${total_rows}] row ${row}"
    (
      "${base_cmd[@]}" --row "$row"
    ) &
    active_pids+=($!)
    active_rows+=("$row")

    if (( ${#active_pids[@]} >= parallelism )); then
      wait_for_taguchi_jobs active_pids active_rows
    fi
  done

  if (( ${#active_pids[@]} > 0 )); then
    wait_for_taguchi_jobs active_pids active_rows
  fi

  finalize_taguchi_outputs \
    "$TAG_DIR/summary.csv" \
    "$TAG_DIR/taguchi_report.csv" \
    "$TAGUCHI_REPORT_METRIC" \
    "$TAGUCHI_REPORT_MODE"
}

generate_report() {
  echo "[5/5] Generating figures & summary"
  python "$ROOT_DIR/scripts/visualize_uniform_noise.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --t-index 500 \
    --cifar-index 0 \
    --output-dir "$FIG_DIR" \
    --modes gaussian uniform \
    --seed 0
  python "$ROOT_DIR/scripts/figures/clean_summaries.py" \
    "$SYN_DIR/summary.csv" \
    "$CIFAR_DIR/summary.csv"
  python "$ROOT_DIR/scripts/figures/generate_figures.py" \
    --synthetic-dir "$SYN_DIR" \
    --cifar-dir "$CIFAR_DIR" \
    --taguchi-dir "$TAG_DIR" \
    --output-dir "$FIG_DIR" \
    --ablation-dir "$ABL_DIR" \
    --include-taguchi-effects
  echo "Report written to $FIG_DIR/summary.md"
}

run_synthetic
run_cifar
run_feature_ablation
run_taguchi
generate_report

echo "Done. Inspect $FIG_DIR for figures and summary."
