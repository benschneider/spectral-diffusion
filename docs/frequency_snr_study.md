# Frequency-Domain SNR Study (Unified)

This note tracks the live experiment surface after pruning legacy knobs. Every forward pass uses the single spectral operator and `k = 1 / snr_ratio`. The goal is to measure how `spectral_operator_mode` and `snr_ratio` interact with samplers and step counts without extra scaling, clamping, or adaptive governors.

## 1. Building Blocks
- **Noise path**: `src/spectral/operator.py` (unit RMS, centered) and `src/spectral/fft_adapter.py` (applies `k = 1 / snr_ratio`). No uniform_corruption flags or adaptive rescale.
- **Preparation**: `src/training/noise.py` consumes `spectral_operator_mode` and `snr_ratio` from the config and emits `snr_theory`, `snr_emp`, `snr_rel`, and `variance_sum`.
- **Recorder/diagnostics**: `scripts/debug/record_training_steps.py` plus `src/training/diagnostics.py` log the unified metrics only.
- **Design of experiments**: Taguchi factors live in `configs/taguchi/factor_registry.yaml` with OA `configs/taguchi/L27_extended.csv` for six ternary knobs.

## 2. Experiment Matrix
Use four baseline conditions for comparisons:

| ID | operator_mode | snr_ratio | Notes |
|----|---------------|-----------|-------|
| A  | none          | 1.0       | Spatial baseline (no shaping) |
| B  | radial        | 1.0       | Radial shaping, unit scale |
| C  | radial        | 0.7 / 1.4 | SNR sweep with shaping |
| D  | radial_squared| 1.0       | Stronger high-frequency boost |

Pair each condition with samplers (`ddim`, `dpm_solver++`) and step counts (10/20/30) drawn from the Taguchi registry as needed.

## 3. Datasets & Seeds
- **Synthetic 8×8 loop**: fast validation of invariants and monotone `snr_theory`.
- **CIFAR-10 32×32**: main training target; reuse `configs/baseline.yaml` overrides for Taguchi rows.
- **Taguchi sweep**: `scripts/run_taguchi_smoke.sh` or `scripts/run_taguchi_minimal.sh` to execute curated subsets.

## 4. Metrics to Track
- Unified noise stats: `snr_theory`, `snr_emp`, `snr_rel`, `variance_sum`, `noise_channel_std_min/max`.
- Stability: loss/MAE traces, grad norms, `loss_drop_per_second`, throughput.
- Sampler quality: FID/LPIPS via `evaluate.py` (optional).

## 5. Analysis Tips
- Validate invariants via `tests/test_training_noise.py` (RMS=1, variance_sum≈1, snr_rel≈1 for `mode=none`, monotone `snr_theory`).
- Compare Taguchi rows with `taguchi_suite/oa.py` and `src/experiments/run_experiment.py`; mapping integrity is enforced by tests.
- Archive any runs that require legacy knobs under `archive/legacy/` to keep the active pipeline clean.
