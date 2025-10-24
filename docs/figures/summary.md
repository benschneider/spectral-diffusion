# Results Summary

## Synthetic Benchmark (8×8 random images)
We compare how quickly the spatial TinyUNet and the spectral version learn to reconstruct tiny synthetic images.

| Run | Loss Drop | Final Loss | Images/s | Runtime (s) | FID |
| --- | --- | --- | --- | --- | --- |
| TinyUNet (Synthetic) | 0.200 | 0.878 | 801.9 | 3.6 | – |
| SpectralUNet (Synthetic) | 1.755 | 0.865 | 305.9 | 9.4 | – |

## CIFAR-10 Reconstruction Benchmark
Same comparison on real CIFAR-10 data to show the accuracy vs. training speed trade-off.

| Run | Loss Drop | Final Loss | Images/s | Runtime (s) | FID |
| --- | --- | --- | --- | --- | --- |
| TinyUNet (CIFAR) | 0.941 | 0.182 | 102.6 | 62.4 | – |
| SpectralUNet (CIFAR) | 2.942 | 0.357 | 48.2 | 132.8 | – |

## Taguchi Factor Sweep
We run a Taguchi orthogonal array to see which frequency-processing settings and sampler choices matter most for convergence speed.

| Rank | Factor | Level | S/N (dB) |
| --- | --- | --- | --- |
| 1 | C | 1 | -28.84 |
| 1 | C | 2 | -27.99 |
| 2 | B | 1 | -28.91 |
| 2 | B | 2 | -28.14 |
| 3 | D | 1 | -28.74 |
