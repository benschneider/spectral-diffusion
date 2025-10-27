# Results Summary

_Generated 2025-10-26T19:22:00+00:00_
_Source: full_report_1024x_20251025_152806_

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
| piecewise_1024x1024_tiny | 0.126 | 0.888 | 470.4 | 3.4 | 0.000 | 0.94 | 96963.7 | – |
| piecewise_1024x1024_tiny_learnable | 0.126 | 0.888 | 492.5 | 3.2 | 0.000 | 0.94 | 96963.7 | – |
| piecewise_1024x1024_spectral | 2.567 | 0.584 | 37.7 | 42.4 | 0.104 | 1.00 | 6.7 | – |
| piecewise_1024x1024_spectral_deep | 1.773 | 0.529 | 16.7 | 95.6 | 0.126 | 0.99 | 5.5 | – |
| piecewise_1024x1024_unet_spectral | 2.567 | 0.584 | 43.1 | 37.2 | 0.104 | 1.00 | 6.7 | – |
| texture_1024x1024_tiny | 0.133 | 0.879 | 503.4 | 3.2 | 0.000 | 0.95 | 90937.8 | – |
| texture_1024x1024_tiny_learnable | 0.133 | 0.879 | 505.1 | 3.2 | 0.000 | 0.95 | 90937.8 | – |
| texture_1024x1024_spectral | 2.840 | 0.471 | 42.8 | 37.4 | 0.113 | 1.00 | 6.1 | – |
| texture_1024x1024_spectral_deep | 1.860 | 0.441 | 16.7 | 95.8 | 0.139 | 0.99 | 5.0 | – |
| texture_1024x1024_unet_spectral | 2.840 | 0.471 | 41.2 | 38.9 | 0.113 | 1.00 | 6.1 | – |
| random_field_1024x1024_tiny | 0.107 | 0.893 | 494.1 | 3.2 | 0.000 | 0.96 | 81313.5 | – |
| random_field_1024x1024_tiny_learnable | 0.107 | 0.893 | 490.9 | 3.3 | 0.000 | 0.96 | 81313.5 | – |
| random_field_1024x1024_spectral | 2.959 | 0.414 | 41.6 | 38.4 | 0.114 | 1.00 | 6.1 | – |
| random_field_1024x1024_spectral_deep | 1.824 | 0.416 | 16.2 | 99.0 | 0.140 | 0.99 | 5.0 | – |
| random_field_1024x1024_unet_spectral | 2.959 | 0.414 | 41.0 | 39.0 | 0.114 | 1.00 | 6.1 | – |

**Fit Results Summary:**
- Average convergence rate (k): 0.071
- Average fit quality (R²): 0.98


**Quick takeaways**
- Lowest final loss: random_field_1024x1024_spectral (0.414)
- Fastest throughput: texture_1024x1024_tiny_learnable (505.1) images/s
- Trade-off: texture_1024x1024_tiny_learnable vs random_field_1024x1024_spectral → 12.1× faster, Δ loss -0.464
- Fastest convergence: random_field_1024x1024_spectral (0.077) loss drop/s

**Convergence Analysis (Exponential Fit):**
- Fastest convergence rate: random_field_1024x1024_spectral_deep (k=0.140, t½=4.957768547190408)
- Highest efficiency (k/runtime): texture_1024x1024_spectral (0.0030)

![Synthetic loss vs throughput](tradeoff_loss_vs_speed_synthetic.png)

![Synthetic final loss distribution](loss_final_distribution_synthetic.png)

![Synthetic loss curves](loss_curve_synthetic.png)

## CIFAR-10 Reconstruction Benchmark
Same comparison on real CIFAR-10 data to show the accuracy vs. training speed trade-off.

| Run | Loss Drop | Final Loss | Images/s | Runtime (s) | Fit k | Fit R² | t½ | FID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar_1024x1024_tiny | 0.941 | 0.182 | 89.9 | 71.2 | 0.017 | 0.99 | 40.2 | – |
| cifar_1024x1024_spectral | 2.942 | 0.357 | 41.0 | 155.9 | 0.091 | 0.98 | 7.6 | – |
| cifar_1024x1024_spectral_deep | 1.699 | 0.278 | 21.8 | 293.7 | 0.090 | 0.94 | 7.7 | – |

**Fit Results Summary:**
- Average convergence rate (k): 0.066
- Average fit quality (R²): 0.97


**Quick takeaways**
- Lowest final loss: cifar_1024x1024_tiny (0.182)
- Fastest throughput: cifar_1024x1024_tiny (89.9) images/s
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
| 1 | Spectral attention | off | -30.33 | 21.282 | 90.22 | 0.483 |
| 1 | Spectral attention | on | -30.73 | 21.217 | 90.49 | 0.529 |
| 2 | Spectral adapters enabled | off | -30.38 | 21.266 | 90.29 | 0.508 |
| 2 | Spectral adapters enabled | on | -30.68 | 21.233 | 90.43 | 0.503 |
| 3 | Cross-domain init | random | -30.58 | 21.253 | 90.34 | 0.505 |

![Taguchi S/N main effects](taguchi_snr.png)

![Taguchi loss_drop_per_second distributions](taguchi_loss_drop_per_second.png)

_Higher S/N (less negative) indicates a more robust configuration. Secondary columns show per-level averages for runtime, throughput, and final loss when available._

**Quick takeaways**
- Spectral attention best at off (-30.33 dB, Δ +0.39 dB vs. on, runtime 21.282s vs 21.217s, images/s 90.22 vs 90.49, final loss 0.483 vs 0.529)
- Spectral adapters enabled best at off (-30.38 dB, Δ +0.30 dB vs. on, runtime 21.266s vs 21.233s, images/s 90.29 vs 90.43, final loss 0.508 vs 0.503)
- Cross-domain init best at GPT-2 (-30.48 dB, Δ +0.10 dB vs. random, runtime 21.246s vs 21.253s, images/s 90.37 vs 90.34, final loss 0.507 vs 0.505)

## FFT Benchmark Snapshot
Parameters: batch=4, channels=3, size=256×256, runs=10
- torch.fft.fft2 (CPU): 3.89 ms per call (total 0.039s)
- numpy.fft.fft2: 10.84 ms per call (total 0.108s)
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
