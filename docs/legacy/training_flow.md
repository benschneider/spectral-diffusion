# Training Flow Overview (Adaptive Governor v3.0)

> **Legacy note:** The adaptive regulator described here is deprecated. The
> active training path uses fixed DDPM coefficients with optional spectral
> operator + `snr_ratio` scaling only.

This guide explains how the centralised adaptive regulator integrates with the
spectral diffusion training loop. It complements the pipeline notes by showing
where each control signal originates and how it feeds back into the loss.

## Step execution timeline

1. **Scheduler step** – Draws the diffusion timestep, noise level, and
   base log-SNR ratios.
2. **Adaptive regulator update** – `AdaptiveSNRGovernor` consumes κ, ema,
   overflow ratio, the variance ratio `r`, and the hard-step fraction. It
   updates:
   - `alpha_fac` using the responsive gain rule.
   - `overflow_ema` via the accelerated 0.8/0.2 smoother.
   - `snr_target` using the dynamic clamp-aware adjustment plus overflow
     corrections that always move towards harder batches.
   - Band probabilities for hard/medium/easy SNR strata.
3. **Micro-reset gate** – Every 200 steps the governor boosts κ by 20% and
   halves `overflow_ema` to escape over-damped equilibria while tagging the
   event for downstream diagnostics.
4. **Spectral pressure** – After computing reconstruction losses, the trainer
   adds the FFT ratio regulariser *and* the variance stabiliser so
   high-frequency content stays energised without collapsing noise variance.
5. **Safety checks and logging** – The combined loss is checked for NaNs and the
   regulator asserts that `snr_target` remains positive. Metrics are appended to
   `step_metrics.jsonl` alongside the FFT bands.

## Regulator signals

| Signal | Description | Diagnostic output |
|--------|-------------|-------------------|
| κ (`kappa`) | Instantaneous curvature estimate from the governor | Logged each step with micro-reset annotations |
| ema | Running target for κ | Logged to track damping behaviour |
| overflow | Fraction of samples beyond the overflow threshold | Logged raw plus smoothed variants |
| `overflow_ema` | Smoothed overflow ratio driving SNR targeting | Logged with micro-reset events |
| `alpha_fac` | Responsiveness multiplier, clamped to [1.0, 1.3] | Logged per-step |
| `snr_target` | Dynamic target passed back to the noise schedule | Logged per-step |
| `variance_ratio` | EWMA of `std(pred_noise) / std(true_noise)` | Logged per-step with variance penalty |
| `hard_fraction` | EWMA of SNR samples inside [0.4, 0.8] | Logged alongside band probabilities |
| `band_*` | Target sampling probabilities for each band | Logged every window update |
| `micro_reset` | Boolean flag indicating a periodic burst fired | Logged as 1.0 on reset steps |
| FFT bands | Low, mid, and high magnitude means | Logged for spectral pressure analysis |
| `noise_rms` | RMS of injected noise with hysteresis clamp | Logged to confirm bounds |

## Module ownership

All adaptive state lives inside `src/utils/adaptive_snr.py` and is re-exported
through `src/training/regulators`. Recorder scripts, unit tests, and the full
trainer import the same governor so behaviour stays in sync. The recorder simply
serialises the emitted metrics, while the trainer uses them to scale losses,
normalise weights, and enforce safety checks before backpropagation.
