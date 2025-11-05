#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

TAGUCHI_REPORT_METRIC=${TAGUCHI_REPORT_METRIC:-loss_drop_per_second}
TAGUCHI_REPORT_MODE=${TAGUCHI_REPORT_MODE:-larger}

usage() {
  cat <<'USAGE'
Usage: finalize_full_report_32x32.sh [--sanitize] [REPORT_DIR]

  --sanitize   Run the markdown sanitizer after regenerating figures.
  REPORT_DIR   Optional explicit report root (defaults to the most recent results/full_report_32x32_* directory).
USAGE
}

select_latest_run() {
  local pattern="full_report_32x32_"
  local latest=""
  shopt -s nullglob
  for dir in "$ROOT_DIR"/results/${pattern}*/; do
    if [[ -d "$dir" ]]; then
      if [[ -z "$latest" || "$dir" -nt "$latest" ]]; then
        latest="$dir"
      fi
    fi
  done
  shopt -u nullglob
  if [[ -z "$latest" ]]; then
    echo ""; return
  fi
  printf '%s' "${latest%/}"
}

ensure_pandoc() {
  python - <<'PY'
import sys
try:
    import pypandoc
except ImportError:
    sys.exit(0)
try:
    pypandoc.get_pandoc_path()
except (OSError, RuntimeError):
    try:
        pypandoc.download_pandoc()
        print("Downloaded pandoc via pypandoc.")
    except Exception as exc:  # pragma: no cover - best-effort helper
        print(f"Warning: unable to download pandoc automatically: {exc}", file=sys.stderr)
PY
}

SANITIZE=0
BASE_DIR=""

while (($#)); do
  case "$1" in
    --sanitize)
      SANITIZE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$BASE_DIR" ]]; then
        BASE_DIR="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$BASE_DIR" ]]; then
  BASE_DIR="$(select_latest_run)"
  if [[ -z "$BASE_DIR" ]]; then
    echo "No previous 32x32 full report runs detected under $ROOT_DIR/results." >&2
    exit 1
  fi
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Report root not found: $BASE_DIR" >&2
  exit 1
fi

SYN_DIR="$BASE_DIR/synthetic"
CIFAR_DIR="$BASE_DIR/cifar"
TAG_DIR="$BASE_DIR/taguchi"
ABL_DIR="$BASE_DIR/ablation"
FIG_DIR="$BASE_DIR/figures"

for required in "$SYN_DIR" "$CIFAR_DIR" "$TAG_DIR"; do
  if [[ ! -d "$required" ]]; then
    echo "Required directory missing: $required" >&2
    exit 1
  fi
done

mkdir -p "$FIG_DIR"

echo "Finalising 32x32 full report at $BASE_DIR"

ensure_pandoc

generate_taguchi_report() {
  local summary_csv="$1"
  local report_csv="$2"
  local metric="$3"
  local mode="$4"
  if [[ ! -f "$summary_csv" ]]; then
    echo "Taguchi summary missing at $summary_csv; skipping report generation." >&2
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

generate_taguchi_report \
  "$TAG_DIR/summary.csv" \
  "$TAG_DIR/taguchi_report.csv" \
  "$TAGUCHI_REPORT_METRIC" \
  "$TAGUCHI_REPORT_MODE"

clean_args=()
for summary in \
  "$SYN_DIR/summary.csv" \
  "$CIFAR_DIR/summary.csv" \
  "$TAG_DIR/summary.csv" \
  "$ABL_DIR/summary.csv"; do
  if [[ -f "$summary" ]]; then
    clean_args+=("$summary")
  fi
done

if (( ${#clean_args[@]} )); then
  python "$ROOT_DIR/scripts/figures/clean_summaries.py" "${clean_args[@]}"
fi

if ! python "$ROOT_DIR/scripts/visualize_uniform_noise.py" \
  --config "$ROOT_DIR/configs/benchmark_spectral_cifar.yaml" \
  --t-index 500 \
  --cifar-index 0 \
  --output-dir "$FIG_DIR" \
  --modes gaussian uniform \
  --seed 0; then
  echo "Warning: uniform noise visualisation skipped (likely missing CIFAR-10 dataset)." >&2
fi

figure_args=(
  "$ROOT_DIR/scripts/figures/generate_figures.py"
  --report-root "$BASE_DIR"
  --output-dir "$FIG_DIR"
  --include-taguchi-effects
)
if [[ -d "$ABL_DIR" ]]; then
  figure_args+=(--ablation-dir "$ABL_DIR")
fi

python "${figure_args[@]}"

python - "$FIG_DIR" <<'PY'
import sys
from pathlib import Path

from src.utils.plot_style import is_duplicate

fig_dir = Path(sys.argv[1])
seen: set[str] = set()
suffixes = {".png", ".jpg", ".jpeg", ".svg", ".gif"}
for image in sorted(fig_dir.glob("*")):
    if image.suffix.lower() not in suffixes or not image.is_file():
        continue
    if is_duplicate(image, seen):
        print(f"[SKIP] Duplicate figure {image}")
        image.unlink(missing_ok=True)
PY

if (( SANITIZE )); then
  summary_md="$FIG_DIR/summary.md"
  if [[ -f "$summary_md" ]]; then
    python - "$summary_md" "$FIG_DIR" <<'PY'
import sys
from pathlib import Path
from src.utils.report_sanitizer import sanitize_markdown

md_path = Path(sys.argv[1])
root = Path(sys.argv[2])
sanitize_markdown(md_path, root)
PY
  fi
fi

echo "Report finalised. Inspect $FIG_DIR for generated figures and summary."
