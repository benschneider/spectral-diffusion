# Frequency-Domain SNR Control Study

This note collects the knobs, metrics, and experiment plans needed for the
question:

> **How does explicit control of signal-to-noise ratio (SNR) in the frequency
> domain affect diffusion training stability and sample efficiency compared to
> standard spatial Gaussian noise?**

## 1. Building Blocks

- **Noise preparation (`src/training/noise.py`)** exposes `diffusion` and
  `spectral` config fields for `uniform_corruption`, `uniform_corruption_scale`,
  `corruption_mode`, `fft_norm`, `snr_ratio`, `phase_std`, and
  `adaptive_rescale`. These ultimately call
  `add_uniform_frequency_noise` (`src/spectral/fft_adapter.py`) where
  reciprocal-radius masking, Parseval checks, `snr_ratio`-based scaling, and
  optional adaptive correlation clamps live.
- **Training instrumentation (`src/training/pipeline.py`)** already records the
  injected-noise stats (`noise_batch.stats`), per-step SNR summaries
  (`coeff_stats`), FFT band feedback, gradient norms, and runtime-derived
  figures such as `loss_drop_per_second`, `loss_threshold_steps`, and
  throughput.
- **Adaptive SNR governor (`src/utils/adaptive_snr.py`)** plus the shared
  helpers in `src/training/regulators/adaptive_regulator.py` supply the dynamic
  `snr_target`, overflow handling, micro-reset policy, and variance regulariser
  described in `docs/training_flow.md` / `docs/training_pipeline.md`.
- **Diagnostics + recorder tooling**
  (`src/training/diagnostics.py`, `scripts/debug/record_training_steps.py`) dump
  band means, stability CSVs, and JSONL telemetry with per-step SNR,
  overflow, κ, EMA, and noise RMS.
- **Tests** (`tests/test_training_noise.py`, `tests/test_fft_noise_scaling.py`,
  `tests/test_training_pipeline.py`) already assert that the `snr_ratio`
  parameter survives config parsing, enforces the requested ratio in the
  frequency domain, matches spatial reconstructions, and keeps Parseval energy
  within tolerance. These provide guardrails for refactors.

## 2. Experiment Matrix

| ID | Noise path | SNR control | Notes |
|----|------------|-------------|-------|
| A | **Spatial Gaussian baseline** (`uniform_corruption: false`) | None | Classic DDPM corruption in the spatial domain—control group. |
| B | **Frequency equalised** (`uniform_corruption: true`, `snr_ratio: null`) | None | FFT-domain mask redistributes noise energy without explicit RMS clamp. |
| C | **Frequency + fixed SNR** (`uniform_corruption: true`, `snr_ratio ∈ {0.7, 1.0, 1.3}`) | Static target | Tests whether explicit ratio control stabilises gradients / convergence. |
| D | **Frequency + adaptive governor** (`uniform_corruption: true`, `snr_ratio: null`, `AdaptiveSNRGovernor` enabled) | Dynamic (`snr_target`) | Lets the regulator adjust ratios based on overflow, variance, and κ trends. |

*Optional extensions*:

- Evaluate both TinyUNet (spatial backbone) and SpectralUNet to see whether
  the gains depend on the model operating in frequency space.
- Add `corruption_mode: phase` with small `phase_std` to isolate phase-only
  perturbations.

## 3. Datasets & Config Seeds

1. **Synthetic 8×8 diagnostic loop** – `configs/test_synthetic_spectral.yaml`
   (fast turn-around, ensures frequency corruption behaves). Pair with
   `scripts/debug/record_training_steps.py` for per-step telemetry.
2. **CIFAR-10 32×32 benchmark** – `configs/benchmark_spectral_cifar.yaml`
   (mirrors README workflows). Capture both TinyUNet and SpectralUNet variants.
3. **Taguchi sweep (optional)** – Use `scripts/run_taguchi_synthetic_l23.py`
   with `configs/taguchi/factor_catalog.yaml` to treat `spectral.freq_equalized_noise`
   and `diffusion.snr_ratio` as explicit factors across the DOE array. Enables
   high-level S/N (Taguchi) scoring.

For each dataset, define YAML overlays (or CLI overrides) for the four matrix
rows above. Example CLI snippet:

```bash
python train.py \
  --config configs/benchmark_spectral_cifar.yaml \
  --config-overrides 'diffusion.uniform_corruption=true,
  diffusion.snr_ratio=0.7,
  diffusion.uniform_corruption_scale=0.15,
  diffusion.adaptive_rescale=false'
```

Record the resulting `results/runs/<run_id>/metrics.json` plus the diagnostics
folder per run.

## 4. Stability Metrics to Track

- `diagnostics/stability_metrics.csv` (loss variance, gradient spikes, noise
  RMS trends) via `TrainingDiagnostics`.
- Per-step governor telemetry (`step_metrics.jsonl` when running the recorder),
  focusing on `overflow`, `overflow_ema`, `snr_target`, `variance_ratio`, and
  `micro_reset` events.
- FFT band ratios and spectral pressure penalties (already logged through
  `fft_feedback`). Plot high/low band means vs. training steps to confirm the
  noise path keeps high frequencies energised.
- Gradient norm histories (`grad_norm_history`) and MAE traces for early-step
  oscillations.

## 5. Sample-Efficiency Metrics

- `loss_drop`, `loss_drop_per_second`, and `loss_threshold_steps/time` emitted
  from `TrainingPipeline`.
- Throughput (`steps_per_second`, `images_per_second`) to check whether the
  FFT masking + SNR clamps add measurable overhead.
- Evaluation metrics (`FID`, `LPIPS`, PSNR) via `evaluate.py` or the automated
  `scripts/run_full_report.sh` pipeline once runs finish training.
- Sampler quality: use `TrainingPipeline.generate_samples(...)` and log per-run
  sampler configs alongside sample grids for qualitative inspection.

## 6. Analysis & Figures

1. **Noise-path sanity** – run `scripts/visualize_uniform_noise.py` with each
   config to produce spatial/FFT snapshots verifying SNR equalisation.
2. **Recorder overlays** – `scripts/debug/record_training_steps.py` already
   writes JSONL/PNG outputs; extend plots to overlay `snr_ratio` vs.
   `snr_measured` for conditions B/C/D.
3. **Aggregated comparisons** – feed run folders into
   `src/reporting/generate_markdown.py` to produce tables showing stability and
   efficiency metrics per condition. `src/utils/report_sanitizer.py` can redact
   noisy keys if needed.
4. **Taguchi S/N scores** – when using the DOE flow, combine per-factor CSVs
   (under `results/<suite>/factors/...`) to report delta S/N caused by toggling
   the SNR controls.

## 7. Pending TODOs Before Writing

- [ ] Create YAML snippets (or config templates) for each condition so CI /
  automation can call them directly.
- [ ] Ensure `TrainingDiagnostics` copies `noise_batch.stats["snr_measured"]`
  into `stability_metrics.csv` for easy plotting (currently only visible via
  logged coeff stats).
- [ ] Extend `generate_markdown.py` to highlight `snr_ratio`, `snr_scale_factor`,
  and FFT variance metrics in the report summary table.
- [ ] Add recorder plots comparing `loss_drop_per_second` vs. `snr_ratio`
  across runs (likely a small helper in `scripts/figures/` or a notebook).
- [ ] Double-check that `AdaptiveSNRGovernor` is toggled on/off via CLI flags
  in configs used for rows C vs. D so the comparison isolates noise-path
  effects.

With these pieces in place we can collect paired runs, compare the stability
diagnostics (loss variance, overflow rate, FFT band balance), and report the
sample-efficiency deltas when explicitly targeting frequency-domain SNR versus
standard spatial Gaussian noise.
