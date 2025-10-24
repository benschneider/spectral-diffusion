# Taguchi Spectral Comparison (CIFAR-10, 3 epochs, 10 mini-batches)

## Goal
Baseline vs. spectral variants on CIFAR-10, measuring loss_drop_per_second and runtime over a short run.

## Factors and Levels
- **A**: `spectral.enabled` — 1 = false, 2 = true
- **B**: `spectral.weighting` — 1 = none, 2 = radial
- **C**: `loss.spectral_weighting` — 1 = none, 2 = radial
- **D**: `sampling.sampler_type` — 1 = ddpm (fallback if missing), 2 = ddim (currently triggers fallback warning)

## Array File
`configs/taguchi_spectral_array.csv`

## Usage
```bash
scripts/clean_results.sh --wipe-summary
OMP_NUM_THREADS=1 PYTHONPATH=$(pwd) \
  python -m src.experiments.run_experiment \
    --config configs/taguchi_smoke_base.yaml \
    --array  configs/taguchi_spectral_array.csv \
    --output-dir results/taguchi_spectral_test

python -m src.analysis.taguchi_stats \
  --summary results/taguchi_spectral_test/summary.csv \
  --metric loss_drop_per_second --mode larger \
  --output results/taguchi_spectral_test/taguchi_report.csv

scripts/report_summary.py \
  --summary results/taguchi_spectral_test/summary.csv \
  --metric loss_drop_per_second \
  --top 5 --include-factors
```
