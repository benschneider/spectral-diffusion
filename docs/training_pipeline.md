# Adaptive Regulator System (v3.0)

The v3.0 training stack introduces a self-regulating governor that balances
variance, overflow, and sampling pressure over long runs. All regulator state
transitions live inside `src/utils/adaptive_snr.py` and are re-exported through
`src/training/regulators`. Recorder utilities and the main trainer share the
same governor so changes to proportional gains, overflow handling, band
probabilities, or reset logic propagate consistently between diagnostics runs
and full training jobs.

## Core building blocks

| Module | Responsibility | Notes |
|--------|----------------|-------|
| `src/utils/adaptive_snr.py` | Defines metric containers, the band-aware `AdaptiveSNRGovernor`, EWMA trackers, weight normalisation, and epsilon-space conversion helpers | Guarantees consistent math across recorders and the trainer |
| `src/training/regulators/__init__.py` | Re-exports the governor for legacy import paths | Keeps downstream code stable |
| `src/training/steps.py` | Composes diffusion, regulator updates, and spectral/variance regularisation into the batch loss | Adds runtime safety checks before backprop |
| `src/utils/debug_helpers.py` | Provides FFT band means used by the spectral pressure term | Recorder and trainer share the same magnitude bands |

These modules replace the scattered v1.x logic that lived inside scripts and
bespoke helpers. Any feature flags or schedule overrides now enter the system
through a single governor state, ensuring the recorder and trainer observe the
same dynamics.

## Regulator dynamics

The proportional gain reacts to divergence between the instantaneous κ value
and the ema-smoothed target:

```
alpha_fac = clamp(1.05 + 0.4 * abs(kappa - ema), 1.0, 1.3)
```

Overflow smoothing blends live overflow ratios with diagnostic feedback using a
faster 0.8 / 0.2 EMA split. Dynamic SNR targeting multiplies the previous
`snr_target` by `1 + 0.3 * (overflow_ema - 0.02)` and layers on:

- A +10% rescue when the measured `std_ratio` falls below 0.8.
- Overflow pressure that always decreases the next ratio (harder tasks).
- A noise-RMS hysteresis clamp: after three consecutive breaches outside
  `[0.28, 0.35]` the governor nudges the ratio back toward the safe band.

Every 200 steps the governor performs a **micro-reset**:

- κ is multiplied by 1.2 to encourage exploration.
- `overflow_ema` is halved, giving the dynamic target room to rise again.
- The event is logged with an `[SNR-GOV]` prefix for downstream correlation.

Band probabilities for the hard (0.4–0.8), medium (0.8–1.4), and easy
(1.4–2.4) SNR regimes are refreshed every 64 steps. The observed fractions are
compared against targets (0.35 hard, 0.25 easy) and adjusted within guard rails,
with overflow pushing more mass into the hard band.

## Spectral pressure and variance regularisers

After computing the reconstruction losses, the trainer injects two regularisers
based on FFT band ratios and variance stability:

```
loss += 0.05 * ((fft_high / (fft_low + 1e-6)) - 1).abs()
loss += 2e-4 * (variance_ratio - 1.0) ** 2
```

This discourages high-frequency collapse while keeping predicted noise variance
within ±10% of the ground truth. FFT band statistics come from
`debug_helpers._fft_band_means`, and the variance ratio is derived from the
shared epsilon-space converter in `adaptive_snr.py`.

## Diagnostics and safety

Each training step and recorder iteration emits the regulator metrics to
`step_metrics.jsonl`, including:

- κ, ema, overflow, and `overflow_ema`
- `alpha_fac`, the dynamic `snr_target`, and a boolean `micro_reset` flag
- Band probabilities, variance ratio, and the running hard-step fraction
- Noise RMS alongside FFT magnitudes and the spectral pressure contribution

Short log-interval runs additionally print concise console summaries that now
include the latest `snr_target`, `alpha_fac`, and band probabilities, while
longer intervals rely on the JSONL feed for post-hoc plotting.

Safety checks guard against silent divergence:

- `assert not torch.isnan(loss)` catches invalid gradients early.
- `assert snr_target > 0` verifies the governor never drives the target into
  degenerate ranges.

Together these diagnostics make the adaptive regulator transparent and easy to
monitor across recorder experiments, CI tests, and full training jobs.
