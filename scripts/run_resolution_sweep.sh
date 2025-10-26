#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"

mkdir -p "$ROOT_DIR/results/resolution_sweep"

echo "Running resolution sweep for spectral models..."
for size in 512 768 1024 1536; do
  echo "Running resolution ${size}x${size}"

  # Run baseline model
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/baseline.yaml" \
    --override "data.height=$size" "data.width=$size" "training.num_batches=5" \
    --output-dir "$ROOT_DIR/results/resolution_sweep" \
    --run-id "baseline_${size}"

  # Run spectral model
  python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/baseline.yaml" \
    --override "data.height=$size" "data.width=$size" "model.type=unet_spectral" "training.num_batches=5" \
    --output-dir "$ROOT_DIR/results/resolution_sweep" \
    --run-id "spectral_${size}"

done

echo "Resolution sweep complete. Run analyze_resolution_scaling.py to analyze results."