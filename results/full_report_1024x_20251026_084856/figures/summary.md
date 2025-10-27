# Results Summary

_Generated 2025-10-26T19:22:06+00:00_
_Source: full_report_1024x_20251026_084856_

## Synthetic Benchmark (generate different type of images, piecewise, parametric textures, random fields)
We compare how quickly the spatial TinyUNet and the spectral version learn to reconstruct tiny synthetic images. 

**Data families tested:**
- **Piecewise**: Structured patterns (checkerboards, stripes, circles) - tests discrete spatial feature learning
- **Texture**: Parametric gratings (oriented, controlled frequency/bandwidth) - tests directional frequency sensitivity
- **Random field**: Power-law spectra (1/f^α falloff) - tests natural image frequency statistics

**FFT Performance Context:**
- torch.fft.fft2 (CPU): 3.9ms per 256×256 image
- numpy.fft.fft2: 10.8ms per 256×256 image

**⚠️ Implementation Caveat:**
Spectral adapters currently rely on Python-level FFT calls, causing host-device sync overhead.
Wall-time results are implementation-limited. Step-based fit-rate metrics (k, t½) are the primary
indicators of convergence efficiency. We benchmark the FFT performance in isolation.

| Run | Loss Drop | Final Loss | Images/s | Runtime (s) | Fit k | Fit R² | t½ | FID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piecewise_1024x1024_tiny | 0.126 | 0.888 | 460.7 | 3.5 | 0.000 | 0.94 | 96963.7 | – |
| piecewise_1024x1024_tiny_learnable | 0.126 | 0.888 | 455.1 | 3.5 | 0.000 | 0.94 | 96963.7 | – |
| piecewise_1024x1024_spectral | 2.567 | 0.584 | 41.8 | 38.2 | 0.104 | 1.00 | 6.7 | – |
| piecewise_1024x1024_spectral_deep | 1.773 | 0.529 | 16.7 | 95.8 | 0.126 | 0.99 | 5.5 | – |
| piecewise_1024x1024_unet_spectral | 2.567 | 0.584 | 41.6 | 38.4 | 0.104 | 1.00 | 6.7 | – |
| texture_1024x1024_tiny | 0.133 | 0.879 | 497.6 | 3.2 | 0.000 | 0.95 | 90937.8 | – |
| texture_1024x1024_tiny_learnable | 0.133 | 0.879 | 496.3 | 3.2 | 0.000 | 0.95 | 90937.8 | – |
| texture_1024x1024_spectral | 2.840 | 0.471 | 42.1 | 38.0 | 0.113 | 1.00 | 6.1 | – |
| texture_1024x1024_spectral_deep | 1.860 | 0.441 | 16.7 | 95.8 | 0.139 | 0.99 | 5.0 | – |
| texture_1024x1024_unet_spectral | 2.840 | 0.471 | 41.6 | 38.4 | 0.113 | 1.00 | 6.1 | – |
| random_field_1024x1024_tiny | 0.107 | 0.893 | 496.7 | 3.2 | 0.000 | 0.96 | 81313.5 | – |
| random_field_1024x1024_tiny_learnable | 0.107 | 0.893 | 496.9 | 3.2 | 0.000 | 0.96 | 81313.5 | – |
| random_field_1024x1024_spectral | 2.959 | 0.414 | 42.2 | 37.9 | 0.114 | 1.00 | 6.1 | – |
| random_field_1024x1024_spectral_deep | 1.824 | 0.416 | 16.7 | 95.6 | 0.140 | 0.99 | 5.0 | – |
| random_field_1024x1024_unet_spectral | 2.959 | 0.414 | 42.5 | 37.6 | 0.114 | 1.00 | 6.1 | – |

**Fit Results Summary:**
- Average convergence rate (k): 0.071
- Average fit quality (R²): 0.98


**Quick takeaways**
- Lowest final loss: random_field_1024x1024_spectral (0.414)
- Fastest throughput: texture_1024x1024_tiny (497.6) images/s
- Trade-off: texture_1024x1024_tiny vs random_field_1024x1024_spectral → 11.8× faster, Δ loss -0.464
- Fastest convergence: random_field_1024x1024_unet_spectral (0.079) loss drop/s

**Convergence Analysis (Exponential Fit):**
- Fastest convergence rate: random_field_1024x1024_spectral_deep (k=0.140, t½=4.957768547190408)
- Highest efficiency (k/runtime): random_field_1024x1024_unet_spectral (0.0030)

![Synthetic loss vs throughput](tradeoff_loss_vs_speed_synthetic.png)

![Synthetic final loss distribution](loss_final_distribution_synthetic.png)

![Synthetic loss curves](loss_curve_synthetic.png)

## CIFAR-10 Reconstruction Benchmark
Same comparison on real CIFAR-10 data to show the accuracy vs. training speed trade-off.

| Run | Loss Drop | Final Loss | Images/s | Runtime (s) | Fit k | Fit R² | t½ | FID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar_1024x1024_tiny | 0.941 | 0.182 | 92.7 | 69.0 | 0.017 | 0.99 | 40.2 | – |
| cifar_1024x1024_spectral | 2.942 | 0.357 | 42.4 | 151.0 | 0.091 | 0.98 | 7.6 | – |
| cifar_1024x1024_spectral_deep | 1.699 | 0.278 | 21.8 | 294.1 | 0.090 | 0.94 | 7.7 | – |

**Fit Results Summary:**
- Average convergence rate (k): 0.066
- Average fit quality (R²): 0.97


**Quick takeaways**
- Lowest final loss: cifar_1024x1024_tiny (0.182)
- Fastest throughput: cifar_1024x1024_tiny (92.7) images/s
- Fastest convergence: cifar_1024x1024_spectral (0.019) loss drop/s

**Convergence Analysis (Exponential Fit):**
- Fastest convergence rate: cifar_1024x1024_spectral (k=0.091, t½=7.611790969344817)
- Highest efficiency (k/runtime): cifar_1024x1024_spectral (0.0006)

![CIFAR-10 loss vs throughput](tradeoff_loss_vs_speed_cifar.png)

![CIFAR-10 final loss distribution](loss_final_distribution_cifar.png)

![CIFAR-10 loss curves](loss_curve_cifar.png)

## Taguchi Factor Sweep
We run a Taguchi orthogonal array to see which frequency-processing settings and sampler choices matter most for convergence speed.

| Rank | Factor | Level | S/N (dB) | Runtime (s) | Images/s | Final Loss |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Spectral adapters enabled | off | -31.11 | 21.184 | 90.71 | 0.501 |
| 1 | Spectral adapters enabled | on | -30.52 | 20.896 | 91.89 | 0.480 |
| 2 | Sampler | DDIM | -30.90 | 21.132 | 90.93 | 0.485 |
| 2 | Sampler | DPM-Solver++ | -30.70 | 20.887 | 91.93 | 0.500 |
| 3 | Spectral attention | off | -30.90 | 21.250 | 90.42 | 0.486 |

![Taguchi S/N main effects](taguchi_snr.png)

![Taguchi loss_drop_per_second distributions](taguchi_loss_drop_per_second.png)

_Higher S/N (less negative) indicates a more robust configuration. Secondary columns show per-level averages for runtime, throughput, and final loss when available._

**Quick takeaways**
- Spectral adapters enabled best at on (-30.52 dB, Δ +0.60 dB vs. off, runtime 20.896s vs 21.184s, images/s 91.89 vs 90.71, final loss 0.480 vs 0.501)
- Sampler best at DPM-Solver++ (-30.70 dB, Δ +0.20 dB vs. DDIM, runtime 20.887s vs 21.132s, images/s 91.93 vs 90.93, final loss 0.500 vs 0.485)
- Spectral attention best at on (-30.75 dB, Δ +0.15 dB vs. off, runtime 20.830s vs 21.250s, images/s 92.18 vs 90.42, final loss 0.496 vs 0.486)

## FFT Benchmark Snapshot
Parameters: batch=4, channels=3, size=256×256, runs=10
- torch.fft.fft2 (CPU): 3.91 ms per call (total 0.039s)
- numpy.fft.fft2: 10.82 ms per call (total 0.108s)
- torch.fft.fft2 (CUDA): not available on this machine
_One-off measurement on local hardware; treat as qualitative guidance._

## FFT Overhead Correction
The benchmark suite measures transform overhead across resolutions.
FFT-corrected runtime t_corrected = t_measured × (1 - fft_fraction).
While not perfect, it allows theoretical extrapolation to GPU-native FFTs.
Scaling exponents p ≈ 1.8–2.0 indicate expected asymptotic flattening for spectral variants.

**⚠️ Implementation Caveat:**
Current spectral adapters use Python-level FFT calls, causing host-device sync overhead.
Wall-time results are implementation-limited. Step-based fit-rate metrics (k, t½) are the primary
indicators of convergence efficiency. We benchmark the FFT performance in isolation.

**📊 Scaling Analysis Methodology:**
- **FFT scaling:** Measured across resolutions 256² to 1536², fitted with power law time ~ N^p
- **Runtime scaling:** Training time vs resolution for baseline vs spectral models
- **Efficiency metrics:** k (convergence rate) and k/runtime (normalized efficiency)
- **FFT correction:** t_corrected = t_measured × (1 - fft_fraction) for fair comparison
