#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$ROOT_DIR"

mkdir -p "$ROOT_DIR/results/fft_sweep"

echo "Running FFT benchmark sweep across resolutions..."
for size in 256 512 768 1024 1536; do
  echo "Benchmarking FFT at ${size}x${size}"
  python "$ROOT_DIR/benchmarks/benchmark_fft.py" \
    --size "$size" \
    --batch 8 \
    --save "$ROOT_DIR/results/fft_sweep/fft_${size}.json"
done

echo "FFT benchmark sweep complete. Run analyze_fft_scaling.py to analyze results."