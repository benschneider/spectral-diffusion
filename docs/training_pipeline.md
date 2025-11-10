# Adaptive Residual Framework (v1.2)

Version 1.2 introduces a unified residual computation layer with adaptive SNR
weighting to keep gradients well behaved across resolutions and timesteps.

| Module | Function | Notes |
|--------|----------|-------|
| `compute_residual` | Handles pixel/spectral diffusion targets | Normalised for scale |
| `AdaptiveSNRWeight` | EMA-based adaptive weighting | Resolution-robust, fp32 EMA |
| `weighted_residual_loss` | Combines adaptive weighting with residual | Quant-safe |
| `DiffusionLoss` | Unified loss entrypoint | Plug-and-play |

This framework prevents SNR blow-ups and maintains smooth gradient scaling even
as the model transitions to better predictions or different image sizes. An
adaptive ``kappa`` floor and fp32 EMA keep the weighting term from collapsing,
while change-aware logging only surfaces material shifts in the balance term.
Future quantised variants share the same normalisation primitives, so BF16/INT8
runs inherit the same numerical guarantees.
