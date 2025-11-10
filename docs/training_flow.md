# Training Flow Overview (Adaptive Regulator v2.0)

This guide explains how the centralised adaptive regulator integrates with the
spectral diffusion training loop. It complements the pipeline notes by showing
where each control signal originates and how it feeds back into the loss.

## Step execution timeline

1. **Scheduler step** – Draws the diffusion timestep, noise level, and
   base log-SNR ratios.
2. **Adaptive regulator update** – `AdaptiveSNRController` receives the latest
   κ estimate, ema target, overflow ratio, and measured `std_ratio`. It updates:
   - `alpha_fac` using the responsive gain rule.
   - `overflow_ema` via the accelerated 0.8/0.2 smoother.
   - `snr_target` using the dynamic clamp-aware adjustment.
3. **Micro-reset gate** – Every 200 steps the controller boosts κ by 20% and
   halves `overflow_ema` to escape over-damped equilibria.
4. **Spectral pressure** – After computing reconstruction losses, the trainer
   adds the FFT ratio regulariser so high-frequency content stays energised.
5. **Safety checks and logging** – The combined loss is checked for NaNs and the
   regulator asserts that `snr_target` remains positive. Metrics are appended to
   `step_metrics.jsonl` alongside the FFT bands.

## Regulator signals

| Signal | Description | Diagnostic output |
|--------|-------------|-------------------|
| κ (`kappa`) | Instantaneous curvature estimate from the controller | Logged each step with micro-reset annotations |
| ema | Running target for κ | Logged to track damping behaviour |
| overflow | Fraction of samples beyond the overflow threshold | Logged raw plus smoothed variants |
| `overflow_ema` | Smoothed overflow ratio used for dynamic targeting | Logged with micro-reset events |
| `alpha_fac` | Responsiveness multiplier, clamped to [1.0, 1.3] | Logged per-step |
| `snr_target` | Dynamic target passed back to the noise schedule | Logged per-step |
| FFT bands | Low, mid, and high magnitude means | Logged for spectral pressure analysis |

## Module ownership

All adaptive state lives inside `src/training/regulators`. Recorder scripts,
unit tests, and the full trainer import the same controllers so behaviour stays
in sync. The recorder simply serialises the emitted metrics, while the trainer
uses them to scale losses and enforce safety checks before backpropagation.
