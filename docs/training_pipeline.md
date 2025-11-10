# Adaptive Regulator System (v2.0)

The v2.0 training stack tightens the feedback loop between diffusion losses
and the adaptive signal-to-noise regulator. All regulator state transitions now
flow through `src/training/regulators/adaptive_regulator.py` and are consumed by
`AdaptiveSNRController` from the same package. Recorder utilities and the main
trainer share these helpers, so changes to proportional gains, overflow
handling, or reset logic propagate consistently between diagnostics runs and
full training jobs.

## Core building blocks

| Module | Responsibility | Notes |
|--------|----------------|-------|
| `src/training/regulators/adaptive_regulator.py` | Defines shared metric containers, α-factor computation, overflow smoothing, and SNR target nudging | Guarantees consistent math across recorders and the trainer |
| `src/training/regulators/adaptive_snr_controller.py` | Owns `AdaptiveSNRController` which applies the regulator math each step | Exposes hooks for micro-resets, diagnostics, and schedule clamps |
| `src/training/steps.py` | Composes diffusion, regulator updates, and spectral regularisation into the batch loss | Adds runtime safety checks before backprop |
| `src/utils/debug_helpers.py` | Provides FFT band means used by the spectral pressure term | Recorder and trainer share the same magnitude bands |

These modules replace the scattered v1.x logic that lived inside scripts and
bespoke helpers. Any feature flags or schedule overrides now enter the system
through a single controller state, ensuring the recorder and trainer observe the
same dynamics.

## Regulator dynamics

The proportional gain reacts to divergence between the instantaneous κ value
and the ema-smoothed target:

```
alpha_fac = clamp(1.05 + 0.4 * abs(kappa - ema), 1.0, 1.3)
```

Overflow smoothing blends live overflow ratios with diagnostic feedback using a
faster 0.8 / 0.2 EMA split. Dynamic SNR targeting multiplies the previous
`snr_target` by `1 + 0.3 * (overflow_ema - 0.02)` and adds an extra ×1.1 boost
whenever the measured `std_ratio` falls below 0.8. The resulting value is
clamped to the schedule's `[snr_min, snr_max]` guard rails to prevent runaway
jumps.

Every 200 steps the controller performs a **micro-reset**:

- κ is multiplied by 1.2 to encourage exploration.
- `overflow_ema` is halved, giving the dynamic target room to rise again.

These nudges keep the regulator from settling into over-damped regimes during
long plateaus.

## Spectral pressure regulariser

After computing the reconstruction losses, the trainer injects a light spectral
pressure term based on FFT band ratios:

```
loss += 0.05 * ((fft_high / (fft_low + 1e-6)) - 1).abs()
```

This discourages high-frequency collapse without overwhelming the diffusion
objective. The FFT band statistics come from `debug_helpers._fft_band_means`, so
recorders and training jobs report compatible diagnostics.

## Diagnostics and safety

Each training step and recorder iteration emits the regulator metrics to
`step_metrics.jsonl`, including:

- κ, ema, overflow, and `overflow_ema`
- `alpha_fac` and the dynamic `snr_target`
- The FFT low/high magnitudes and derived spectral pressure contribution

Short log-interval runs additionally print concise console summaries, while
longer intervals rely on the JSONL feed for post-hoc plotting.

Safety checks guard against silent divergence:

- `assert not torch.isnan(loss)` catches invalid gradients early.
- `assert snr_target > 0` verifies the regulator never drives the target into
  degenerate ranges.

Together these diagnostics make the adaptive regulator transparent and easy to
monitor across recorder experiments, CI tests, and full training jobs.
