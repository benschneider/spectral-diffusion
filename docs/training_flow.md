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
