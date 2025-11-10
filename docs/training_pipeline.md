# Adaptive Residual Framework (v1.4)

Version 1.4 upgrades the unified residual layer with log-SNR normalisation,
α-residual damping, and overflow safeguards so gradients stay well behaved
across resolutions, timesteps, and training phases without manual retuning.

| Module | Function | Notes |
|--------|----------|-------|
| `compute_residual` | Handles pixel/spectral diffusion targets | Normalised for scale |
| `AdaptiveSNRWeight` | Log-SNR adaptive weighting | fp32 EMA, α-damping, overflow mask |
| `weighted_residual_loss` | Combines adaptive weighting with residual | Auto-freeze, quant-safe |
| `DiffusionLoss` | Unified loss entrypoint | Plug-and-play |

This framework prevents SNR blow-ups and maintains smooth gradient scaling even
as the model transitions to better predictions or different image sizes. The
log-SNR weighting centres per-sample ratios, damps late-stage steps through the
α-residual factor, masks overflowed timesteps, and freezes updates if the
prediction variance drifts more than five times the input. Change-aware logging
surfaces only meaningful shifts in the balance term. Future quantised variants
share the same normalisation primitives, so BF16/INT8 runs inherit the improved
guarantees.
