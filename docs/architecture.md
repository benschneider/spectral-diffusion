# Architecture Overview (Unified)

The active pipeline keeps a small, composable surface:

- `src/core/`: baseline conv model + `TinyUNet`, initialisation, losses.
- `src/spectral/`: `spectral_operator` (unit RMS noise shaping) and FFT helpers.
- `src/training/`: dataloaders, diffusion scheduler, noise preparer, samplers, diagnostics, and the training pipeline.
- `src/experiments/`: Taguchi runner wiring factors to configs.
- `scripts/debug/record_training_steps.py`: minimal recorder using the unified noise path.

## Data & Dataloaders
- **Synthetic**: `SyntheticSpectralDataset` (square images) with optional text overlays; controlled via `data.*` and `data.synthetic.*`.
- **CIFAR-10**: standard torchvision loader.

## Diffusion Forward Process
- Coefficients from `src/training/scheduler.py` (`build_diffusion`).
- Noise via `src/training/noise.py`: sample `eps_raw ~ N(0, I)`, apply `spectral_operator(mode)`, scale by `k = 1 / snr_ratio`, then mix with `sqrt_alpha_t` / `sqrt_one_minus_alpha_t`.
- Stats exposed per batch: `snr_theory`, `snr_emp`, `snr_rel`, `variance_sum`, `noise_channel_std_min/max`, `eps_norm`.

## Training Loop
- `TrainingStepExecutor` builds targets (`eps`, `x0`, or `v`), computes loss/MAE, steps the optimiser, and records FFT feedback.
- `TrainingDiagnostics` mirrors loss/grad/noise/coeff/batch stats into `diagnostics/` and factor-specific folders during Taguchi sweeps.
- `TrainingPipeline` orchestrates dataloaders, scheduler, noise preparer, diagnostics, and optional sampling.

## Taguchi System
- Factors: `snr_ratio`, `spectral_operator_mode`, `sampler_type`, `sampling_steps`, `train_steps`, `image_resolution`.
- Registry: `configs/taguchi/factor_registry.yaml` (+ quick variant).
- Designs: `configs/taguchi/L27_extended.csv` (only active OA).
- Runner: `src/experiments/run_experiment.py` + helpers in `taguchi_suite/oa.py`.

## Automation & Reporting
- Smoke/comparison Taguchi scripts live under `scripts/` (`run_taguchi_smoke.sh`, `run_taguchi_minimal.sh`, `run_taguchi_comparison.sh`).
- Reports: `scripts/generate_report_v2.py` collates diagnostics and produces summaries.
- Archived runners (full reports, spectral ablations) reside under `archive/legacy/`.
