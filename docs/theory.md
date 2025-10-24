# Spectral Diffusion – Theory Notes

This document explains the project in plain language so that anyone can understand why we experiment with frequency space and how it affects diffusion models.

## 1. What is a diffusion model?
Imagine teaching a model to undo noise: we start with a clean picture, add noise step by step until it becomes static, and then train a neural network to reverse that damage one small step at a time. After training, we can start from pure noise and ask the model to run the steps in reverse to paint a realistic image. This is the essence of diffusion models (DDPM, DDIM, etc.).

## 2. Why bring in frequency space?
Pixels show what is happening locally, but many visual patterns live in frequencies (edges = high frequencies, smooth regions = low frequencies). If the model can “see” data in frequency space:
- It may learn which frequency bands matter earlier, speeding up convergence.
- Spectral weighting can emphasise edges or textures that are otherwise washed out.
- Some operations (e.g., global consistency) are simpler after an FFT.

## 3. Spectral adapters vs. SpectralUNet
- **Spectral adapters** are small modules that wrap the existing UNet: FFT → weighting → iFFT. They keep the core model intact but bias it toward certain frequencies.
- **SpectralUNet** is a new architecture that operates directly on complex-valued spectra. It replaces convolutions with complex convolutions so the network “thinks” in frequency space throughout its depth.

## 4. Losses in frequency space
We can also weight the residuals before computing MSE/MAE: for example, a radial weighting boosts high-frequency errors, encouraging sharper details. Frequency-weighted losses were inspired by classic image processing, where matching edge spectra is key to perceived quality.

## 5. Evaluation metrics we track
- **Loss drop / second**: how fast the model is improving the training loss (convergence speed).
- **Images per second**: throughput (how many training examples per second we process).
- **Runtime per epoch**: wall-clock time so we can fairly compare spectral vs spatial.
- **FID / LPIPS**: standard deep learning metrics for generation quality (FID requires torchmetrics).
- **Taguchi S/N ratios**: quantify how much different configuration levels influence the outcome (bigger S/N delta = more impact).

## 6. Putting it together
The project lets you toggle frequency choices (adapters, spectral UNet) and samplers, then measure the trade-off between training speed and reconstruction quality. The figure-generation pipeline translates raw metrics into human-friendly plots and summaries.

Feel free to open issues/discussions if you have questions or ideas for additional spectral analysis!
