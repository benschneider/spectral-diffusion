# Adaptive Residual Framework (v1.3)

Version 1.3 extends the unified residual layer with self-tuning SNR weighting
so gradients stay well behaved across resolutions, timesteps, and training
phases without manual retuning.

| Module | Function | Notes |
|--------|----------|-------|
| `compute_residual` | Handles pixel/spectral diffusion targets | Normalised for scale |
| `AdaptiveSNRWeight` | Self-tuning adaptive weighting | fp32 EMA, dynamic gain, SNR clamp |
| `weighted_residual_loss` | Combines adaptive weighting with residual | Quant-safe |
| `DiffusionLoss` | Unified loss entrypoint | Plug-and-play |

This framework prevents SNR blow-ups and maintains smooth gradient scaling even
as the model transitions to better predictions or different image sizes. The
self-tuning SNR weighting normalises the loss by running statistics, adapts its
gain so ``kappa`` remains in a numerically active range, and clamps pathological
SNR spikes before they destabilise updates. Change-aware logging surfaces only
meaningful shifts in the balance term. Future quantised variants share the same
normalisation primitives, so BF16/INT8 runs inherit the improved guarantees.
