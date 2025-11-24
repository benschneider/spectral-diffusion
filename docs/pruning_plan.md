# Unified Forward-Process Pruning Plan

Objective: remove legacy/adaptive/phase code paths and leave only the unified forward process, unified SNR triad, and minimal Taguchi factor set.

## 1) Legacy quarantine
- Move the following to `archive/legacy/` (or delete if unused): `src/utils/adaptive_snr.py`, `src/core/overflow_handler.py`, `src/core/snr_scheduler.py`, any `phase_attention`/PCM code paths, spectral adapters/loss weighting/bandpass/strength hooks, `uniform_corruption`, `snr_scale_min/max`, `adaptive_rescale`.
- Archive legacy Taguchi runners/configs/CSVs that reference removed factors (old L23/L27 manifests, `scripts/run_taguchi_v2_fixed.py`, archive full-report scripts). Leave only the six live factors.

## 2) Config pruning
- Strip legacy keys from active configs (`configs/benchmark_*`, `configs/variants.yaml`, Taguchi YAMLs): remove `uniform_corruption`, `freq_equalized_noise`, `snr_weighting`, `snr_scale_*`, `adaptive_rescale`, phase/adapters, spectral_loss_weighting/bandpass/strength.
- Ensure Taguchi registries/catalogs/suites reference only: `snr_ratio`, `spectral_operator_mode`, `sampler_type`, `sampling_steps`, `train_steps`, `image_resolution`. OA CSVs should carry only level indices for these six.

## 3) Code authority & routing
- Keep a single shaping entrypoint: `src/spectral/operator.py` (per-sample centering, RMS=1).
- Noise path: `src/training/noise.py` (or `src/noise/` if moved) applies only `k=1/snr_ratio`; no extra scaling/clamps; coefficients pulled directly from `build_diffusion`.
- Scheduler: `src/training/scheduler.py` returns raw schedules; no trimming/clamping.
- Training step: `src/training/steps.py` computes loss (MSE/MAE with optional log(snr_rel)) and logs unified metrics; no variance/spectral penalties.

## 4) Step recorder
- `scripts/debug/record_training_steps.py` must only call `NoisePreparer.prepare` and `TrainingStepExecutor.run_step`. Remove any local SNR/variance/FFT math or custom stats; log only the fields emitted by these components.

## 5) Diagnostics & reporting
- Diagnostics/logging should emit only `snr_theory`, `snr_emp`, `snr_rel`, `variance_sum`, `noise_channel_std_min/max`, `grad_norm`, loss/mae histories. Remove `snr_schedule_*`, `snr_effective`, overflow/spectral_pressure/variance_ratio metrics.
- `scripts/generate_report_v2.py` and `scripts/process_stability_metrics.py` should drop legacy fields and surface only the unified SNR triad + variance_sum and performance metrics (loss, loss_drop_per_second, images_per_second).

## 6) Tests & invariants
- Maintain/add tests for:
  - RMS(eps_shaped) ≈ 1 after centering (±1e-4).
  - Var(signal)+Var(noise) ≈ 1 (±1e-3–1e-2 depending on batch size).
  - snr_rel ≈ 1 with shaping off/snr_ratio=1 (±10%).
  - Monotone `snr_theory` across schedule.
  - Spectral operator modes produce distinct frequency responses but preserve RMS.
  - Step recorder uses no custom forward/noise logic.
- Remove tests referencing adaptive/phase/adapter/overflow paths.

## 7) Documentation
- Update README, `docs/config_reference.md`, `docs/training_flow.md`, Taguchi docs (`docs/taguchi_factor_plan.md`, `docs/taguchi_l27_run_notes.md`) to clearly mark legacy sections as archived and highlight only the unified forward process, unified SNR triad, and six Taguchi factors.
- Note the legacy quarantine location and that legacy knobs are unsupported.
