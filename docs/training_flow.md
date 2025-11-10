# Training Flow Overview

This document complements the existing training pipeline notes by describing how the adaptive SNR regimes interact with the diffusion update.

### Adaptive SNR Handling (v1.2)

The training loop dynamically adjusts its computation mode based on the current signal-to-noise ratio (SNR):

| Regime | Condition | Update Mode | Loss Type | Behavior |
|:--|:--|:--|:--|:--|
| **Noise-dominant** | SNR < 1 | Stochastic | ε-loss | Normal diffusion noise learning |
| **Balanced** | 1 ≤ SNR ≤ clip | Weighted stochastic | ε-loss | Standard adaptive-SNR weighting |
| **Signal-dominant (overflow)** | SNR > clip | Deterministic (DDIM-like) | x₀-loss | Noise suppressed, output stabilised |
| **Limit (SNR→∞)** | 1−αₜ ≈ 0 | Log-SNR→∞ | Switch to x₀-prediction, freeze adaptive EMA | Fully deterministic reconstruction |

Log entries follow the pattern:

```
[OverflowHandler] mode=deterministic snr=312.7 loss_mode=x0
```

Key parameters:

- `snr_clip` default = 250.0 (configurable via loss config)
- `log_snr_smooth = True` enables soft saturation through the tanh(log-SNR) mapping
- Overflow diagnostics and plots are written alongside the recorder artefacts in `scratch/<run>/diagnostics/overflow_snr.png`

The deterministic branch keeps gradients stable and preserves residual magnitudes when the schedule enters the near-zero-noise regime.

### Module layout

The adaptive flow is implemented across a set of focused helpers:

- `src/core/snr_scheduler.py` derives log-SNR weights, clamped ratios, and measured batch RMS values.
- `src/core/adaptive_weight.py` maintains the EMA-based weighting logic and exposes change-aware diagnostics.
- `src/core/overflow_handler.py` renormalises extreme predictions and emits `[OverflowHandler]` log lines.
- `src/core/diffusion_step.py` hosts regime-selection utilities (`select_regime`, `describe_regime`, `predict_x0`).
- `src/core/fft_feedback.py` centralises FFT residual metrics for both the recorder and the executor.

`scripts/debug/record_training_steps.py` and `src/training/steps.py` compose these modules so the recorder and the main training loop share the same regime bookkeeping.
