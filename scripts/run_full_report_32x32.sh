#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

detect_rifft_threads() {
  local guess
  guess="$(python - <<'PY' 2>/dev/null || true
import os
count = os.cpu_count() or 1
print(count)
PY
)"
  if [[ -z "$guess" ]]; then
    guess=1
  fi
  echo "$guess"
}

export RIFFT_PREPLAN="${RIFFT_PREPLAN:-auto}"
if [[ -z "${RUSTFFT_THREADS:-}" ]]; then
  export RUSTFFT_THREADS="$(detect_rifft_threads)"
fi

echo "RIFFT env: RIFFT_PREPLAN=${RIFFT_PREPLAN} RUSTFFT_THREADS=${RUSTFFT_THREADS}"

TAGUCHI_ARRAY_PATH=${TAGUCHI_ARRAY_PATH:-"$ROOT_DIR/configs/taguchi/L27_extended.csv"}
TAGUCHI_FACTOR_REGISTRY=${TAGUCHI_FACTOR_REGISTRY:-"$ROOT_DIR/configs/taguchi/factor_registry.yaml"}
TAGUCHI_BASE_CONFIG=${TAGUCHI_BASE_CONFIG:-"$ROOT_DIR/configs/taguchi_smoke_best.yaml"}
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
REPORT_V2_DIR="$BASE_DIR/report_v2"
TAGUCHI_HDF5_ENABLED=${TAGUCHI_HDF5_ENABLED:-0}
TAGUCHI_HDF5_PATH=${TAGUCHI_HDF5_PATH:-"$TAG_DIR/taguchi_runs.h5"}
TAGUCHI_HDF5_PRUNE=${TAGUCHI_HDF5_PRUNE:-0}

mkdir -p "$SYN_DIR" "$CIFAR_DIR" "$TAG_DIR" "$FIG_DIR" "$ABL_DIR" "$REPORT_V2_DIR"

echo "Full report (32x32) root: $BASE_DIR"

describe_run() {
  local config="$1"
  local run_id="$2"
  local variant_label="$3"
  python - "$config" "$run_id" "$variant_label" <<'PY'
import sys
import yaml
import re

def infer_size(cfg, run_id):
    data = (cfg or {}).get("data", {})
    height = data.get("height")
    width = data.get("width")
    if height is None and width is None:
        image_size = data.get("image_size") or (cfg or {}).get("evaluation", {}).get("image_size")
        if isinstance(image_size, int):
            height = height or image_size
            width = width or image_size
    if (height is None or width is None) and run_id:
        match = re.search(r"(\d+)\s*x\s*(\d+)", run_id)
        if match:
            if height is None:
                height = int(match.group(1))
            if width is None:
                width = int(match.group(2))
    return height, width

config_path, run_id, variant = sys.argv[1:4]
try:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
except FileNotFoundError:
    cfg = {}
data_family = cfg.get("data", {}).get("family", "default")
data_source = cfg.get("data", {}).get("source", "unknown")
height, width = infer_size(cfg, run_id)
if height is None:
    height = "?"
if width is None:
    width = "?"
data_size = f"{height}x{width}"
model = cfg.get("model", {}).get("type", "default")
print(f"  • Run {run_id}: {config_path} (source={data_source}, size={data_size}, family={data_family}, model={model}, variant={variant})")
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
  local pids_name="$1"
  local rows_name="$2"
  local -a pids=()
  local -a rows=()
  eval "pids=(\"\${${pids_name}[@]}\")"
  eval "rows=(\"\${${rows_name}[@]}\")"
  local i pid row
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    row="${rows[$i]}"
    if ! wait "$pid"; then
      echo "Taguchi row $row failed (PID $pid)." >&2
      exit 1
    fi
  done
  eval "$pids_name=()"
  eval "$rows_name=()"
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
  echo "[1/4] Synthetic benchmarks (32x32 images)"
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

    # Run TinyUNet (default)
    describe_run "$ROOT_DIR/configs/$config_file" "${family_name}_32x32_tiny" "config-default"
    python "$ROOT_DIR/train.py" \
      --config "$ROOT_DIR/configs/$config_file" \
      --output-dir "$SYN_DIR" \
      --run-id "${family_name}_32x32_tiny"
    sample_with_ddim "$SYN_DIR/runs/${family_name}_32x32_tiny" "ddim_baseline"
  done
}

run_cifar() {
  echo "[2/4] CIFAR-10 benchmark (32x32 TinyUNet)"
  rm -f "$CIFAR_DIR/summary.csv"
  rm -rf "$CIFAR_DIR/runs"
  mkdir -p "$CIFAR_DIR"

  describe_run "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" "cifar_32x32_tiny" "config-default"
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --output-dir "$CIFAR_DIR" \
    --run-id "cifar_32x32_tiny"
  sample_with_ddim "$CIFAR_DIR/runs/cifar_32x32_tiny" "ddim_baseline"
}

run_taguchi() {
  echo "[3/4] Taguchi sweep"
  local array_basename
  array_basename="$(basename "$TAGUCHI_ARRAY_PATH")"
  local array_stem="${array_basename%.csv}"
  rm -f "$TAG_DIR/summary.csv" \
        "$TAG_DIR/taguchi_report.csv" \
        "$TAG_DIR/${array_stem}_summary.csv" \
        "$TAG_DIR"/run_*_metrics.json
  rm -rf "$TAG_DIR/runs"
  describe_run "$ROOT_DIR/configs/taguchi_smoke_base.yaml" "taguchi_32x32_sweep" "array:${array_basename}"

  local -a taguchi_rows=()
  local row_value
  while IFS= read -r row_value; do
    if [[ -n "${row_value// }" ]]; then
      taguchi_rows+=("$row_value")
    fi
  done < <(collect_taguchi_rows "$TAGUCHI_ARRAY_PATH")
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
      --config "$TAGUCHI_BASE_CONFIG"
      --array "$TAGUCHI_ARRAY_PATH"
      --output-dir "$TAG_DIR"
      --report-metric "$TAGUCHI_REPORT_METRIC"
      --report-mode "$TAGUCHI_REPORT_MODE"
      --factor-registry "$TAGUCHI_FACTOR_REGISTRY"
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

  if [[ "${TAGUCHI_HDF5_ENABLED}" == "1" ]]; then
    local prune_flags=()
    if [[ "${TAGUCHI_HDF5_PRUNE}" == "1" ]]; then
      prune_flags=(--prune)
    fi
    echo "Converting Taguchi outputs to ${TAGUCHI_HDF5_PATH}"
    python "$ROOT_DIR/scripts/collate_runs_to_hdf5.py" \
      --runs-root "$TAG_DIR" \
      --taguchi-root "$TAG_DIR" \
      --output "$TAGUCHI_HDF5_PATH" \
      "${prune_flags[@]}"
  fi
}

generate_report() {
  echo "[4/4] Generating report_v2 bundle"
  mkdir -p "$REPORT_V2_DIR/appendix/noise_chains"
  python "$ROOT_DIR/scripts/visualize_uniform_noise.py" \
    --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
    --t-index 500 \
    --cifar-index 0 \
    --output-dir "$REPORT_V2_DIR/appendix/noise_chains" \
    --modes gaussian uniform \
    --seed 0
  python "$ROOT_DIR/scripts/figures/clean_summaries.py" \
    "$SYN_DIR/summary.csv" \
    "$CIFAR_DIR/summary.csv"
  python "$ROOT_DIR/scripts/generate_report_v2.py" \
    --report-root "$BASE_DIR" \
    --output-dir "$REPORT_V2_DIR"
  echo "Report written to $REPORT_V2_DIR/summary.md"
}

run_synthetic
run_cifar
run_taguchi
generate_report

echo "Done. Inspect $REPORT_V2_DIR for figures and summary."
