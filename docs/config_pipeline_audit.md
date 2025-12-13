# Config & Pipeline Audit (Post-Refactor)

## Inconsistencies Detected
- Training loop expected deprecated keys: `training.num_batches`, loss-aware sampling knobs, and `metrics.loss_threshold`; ignored canonical `training.train_steps`.
- Sampling path consumed `sampling.num_steps` instead of canonical `sampling.sampling_steps`.
- Noise preparer read legacy `spectral.*` fallbacks; redundant scaling path.
- Dataloader sizing tied to `training.num_batches` plus non-canonical synthetic overrides.
- Taguchi factors and reporting mapped to deprecated keys (`sampling.num_steps`, `training.num_batches`, `spectral.*`).
- CLI exposed `diffusion.lambda_var`; model builder applied `initialization.*`, both removed from schema.

## Canonical Schema Enforced
```
model:    type, channels, base_channels, depth
data:     source {synthetic|cifar10}, channels, height, width, family?
diffusion:num_timesteps, beta_schedule {cosine|linear}, prediction_type=eps,
          time_embed_dim, fft_norm=ortho, snr_ratio, spectral_operator_mode
training: batch_size, epochs, train_steps (int|null), log_every
optim:    lr, weight_decay
sampling: enabled, sampler_type, num_samples, sampling_steps
```

## Code Alignment
- Training loop now:
  - Computes `total_steps = train_steps (if set) else epochs * batches_per_epoch`.
  - Removes loss-aware/warmup logic and legacy thresholds.
  - Uses canonical beta_schedule validation and metrics.
- Sampling:
  - Renamed to `sampling_steps`; sampler metadata/logs match.
- Noise:
  - Only reads diffusion fields (`spectral_operator_mode`, `snr_ratio`); no `spectral.*` fallbacks.
- Dataloaders:
  - Fixed-size synthetic/CIFAR loaders; no `num_batches` coupling or non-schema overrides.
- Taguchi:
  - Factors/apply logic constrained to canonical fields; dropped deprecated mappings.
- Model/CLI:
  - Removed `initialization.*` application and `lambda_var` flag.
- Reporting:
  - Factor key map updated to canonical sampling/train fields.

## Files Touched
- Config registry: `configs/taguchi/factor_registry*.yaml`
- CLI: `src/cli/train.py`, `src/cli/sample.py`
- Training: `src/training/pipeline.py`, `src/training/noise.py`, `src/training/builders.py`
- Taguchi/Reporting: `src/experiments/run_experiment.py`, `src/reporting/generate_markdown.py`
- Model: `src/core/model.py`

## Follow-Up (Optional)
- Run smoke: `python train.py --config configs/baseline.yaml --dry-run`
- Verify Taguchi run: `python src/experiments/run_experiment.py --config configs/taguchi_smoke_best.yaml --array configs/taguchi/L27_extended.csv --row 1 --finalize`
