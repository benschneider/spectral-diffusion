# Spectral UNet Research Plan

## Motivation
The current TinyUNet operates entirely in the spatial domain with optional FFT adapters applied pre/post forward passes. To validate the core hypothesis—that direct frequency-space reasoning improves diffusion efficiency—we need a model that natively operates on complex-valued spectra and can be compared fairly against the spatial baseline.

## Research Questions
- Can a frequency-native UNet reduce the number of diffusion steps required for comparable reconstruction quality?
- How do complex convolutions and spectral attention mechanisms affect convergence and sampling stability?
- What spectral weighting or normalization strategies are most effective at each diffusion timestep?
- Are hybrid approaches (e.g., frequency encoder + spatial decoder) more robust than fully spectral pipelines?

## Proposed Architecture
1. **Spectral Input Representation**
   - Treat FFT outputs as stacked real/imag channels or complex tensors.
   - Explore learned magnitude/phase embeddings.
2. **Complex Convolution Blocks**
   - Implement complex-valued convolutions (real/imag weight matrices).
   - Provide fallbacks to dual-channel real convolutions for ablation.
3. **Spectral Attention**
   - Frequency-specific attention (radial, band-limited).
   - Investigate low-rank approximations for efficiency.
4. **Time Conditioning**
   - Extend existing sinusoidal + MLP embedding to operate on spectral features.
5. **Output Projection**
   - Predict spectral ε or x₀ directly; support conversion back to spatial via inverse FFT.

## Experiment Roadmap
| Stage | Goal | Key Tasks | Output |
|-------|------|-----------|--------|
| 1 | Prototype complex conv layers | Implement complex conv/batch-norm modules; unit tests | `src/spectral/layers.py` |
| 2 | Minimal Spectral UNet | Build spectral encoder/decoder skeleton; integrate with TrainingPipeline | `src/core/model_unet_spectral.py` |
| 3 | Training Baseline | Compare spectral vs spatial on synthetic + CIFAR-10 (short runs) | `results/runs/spectral_*` |
| 4 | Advanced Samplers | Evaluate DDIM/DPM-Solver++ interactions with spectral models | Extended summary/plots |
| 5 | Ablations | Weighting schemes, complex vs dual-channel conv, hybrid architectures | Taguchi factor updates |

## Datasets & Configs
- Start with synthetic (8×8) to validate gradients.
- CIFAR-10 (32×32) for core benchmarks.
- Consider frequency-rich datasets (e.g., audio spectrograms) later.

## Metrics & Logging
- Reuse existing metrics (`loss_drop`, `steps_per_second`).
- Add LPIPS/FID comparisons for sampled images.
- Track spectral-specific stats: energy distribution, magnitude/phase histograms.

## Risks & Mitigations
- **Complex gradients**: provide real-valued dual-channel fallback.
- **Performance**: cache FFT kernels, limit depth, leverage existing adapters for warm start.
- **Stability**: enforce normalization, monitor exploding magnitudes.

## Next Steps
1. Implement complex convolution + normalization layers with targeted tests.
2. Define minimal spectral UNet config and ensure compatibility with current TrainingPipeline.
3. Update Taguchi factors to include spectral model toggles for automated sweeps.
