# Spectral Diffusion – Architecture Overview

This document explains how the components fit together so you can navigate the codebase quickly.

## 1. High-level flow
```
configs/ ─► TrainingPipeline ─► results/ ─► visualization/
             (samplers, data)     (metrics, samples)
```

1. **Configs** describe model, data, spectral options, optimisation.
2. **TrainingPipeline** (in `src/training/`) builds models/samplers, runs the diffusion loop, logs metrics, checkpoints.
3. **Automation scripts** (e.g. `scripts/run_full_report.sh`) chain benchmarks + Taguchi sweeps.
4. **Visualization library** (`src/visualization/`) cleans summary CSVs and draws figures/markdown reports.

## 2. Directory map
- `src/core/` – TinyUNet, SpectralUNet, losses, registry.
- `src/spectral/` – FFT adapters, complex spectral layers, utilities.
- `src/training/` – pipeline, schedulers, sampler registry.
- `src/evaluation/` – dataset metrics (MSE/MAE/PSNR, optional FID/LPIPS).
- `src/experiments/` – Taguchi runner (`run_experiment.py`).
- `src/visualization/` – `collect.clean_summary`, `figures.generate_figures`.
- `scripts/` – CLI wrappers for training, sampling, reporting.
- `docs/` – theory notes, architecture overview, generated figures, Taguchi tips.
- `results/` – per-run artefacts (configs, logs, metrics, checkpoints, samples).

## 3. Artefacts per run
```
results/<experiment_root>/YYYYMMDD_HHMMSS/
├── synthetic/
│   └── runs/<run_id>/...
├── cifar/
│   └── runs/<run_id>/...
├── taguchi/
│   ├── summary.csv
│   └── taguchi_report.csv
└── figures/
    ├── loss_metrics_*.png
    ├── runtime_metrics_*.png
    ├── tradeoff_loss_vs_speed_*.png
    ├── *_distribution_*.png
    └── summary.md
```
- Individual training runs still write `config.yaml`, `system.json`, `metrics/<run_id>.json`, checkpoints, and samples inside `runs/<run_id>/`.
- Reports now live under timestamped subdirectories so multiple executions never overwrite each other.
- `figures/summary.md` records the generation timestamp/source and embeds every plot for quick variance checks.
- Taguchi sweeps enrich `taguchi_report.csv` with S/N plus mean runtime/throughput/final-loss for each factor level.

## 4. Figure/report workflow
1. Generate runs (manual CLI or `scripts/run_full_report.sh` / `run_smoke_report.sh`).
2. `src/visualization.collect.clean_summary` deduplicates and labels entries (synthetic, CIFAR, Taguchi).
3. `src/visualization.figures.generate_figures` produces:
   - Bar charts, scatter trade-offs, and box plots for loss/throughput metrics
   - `taguchi_snr.png` (with contextual caption) and optional factor distributions
   - Timestamped `figures/summary.md` inside the chosen report root
4. README Showcase points to the latest generated artefacts.

## 5. How to extend
- **Add a sampler**: subclass `Sampler` (`src/training/sampling.py`), register via `register_sampler("name", Class)`.
- **Add a spectral adapter**: follow the patterns in `src/spectral/adapter.py` or `complex_layers.py`.
- **Add a dataset**: update `configs/` and `src/training/builders.py` for a new loader.
- **Add plots**: extend `src/visualization/figures.py`, then call from `scripts/figures/generate_figures.py`.

With this modular layout you can import the same components in notebooks, CLI scripts, or experiments without rewriting plumbing.

## 6. Roadmap (next wave)
- **Learnable spectral adapters** – replace fixed FFT masks with small MLPs conditioned on timestep embeddings so the model can shift focus across frequency bands automatically.
- **Deep spectral UNet** – extend the current shallow spectral model into a full encoder/decoder with complex down/upsampling and skip connections to test a frequency-first hierarchy.
- **Pretrained/cross-domain initialization sweeps** – new Taguchi factor (`E`) already toggles GPT-2 vs random seeding; future work includes curating additional sources and analysing their impact with the richer runtime/throughput metrics now captured.

## 7. Core Model Architectures

The project's central experiment is a comparison between two different architectural philosophies for diffusion models, embodied by `TinyUNet` and `SpectralUNet`.

*   **`TinyUNet` (The Hybrid Model):** This is a traditional, spatial-domain U-Net. It can be optionally "enhanced" by `SpectralAdapter` modules, which wrap convolutional blocks to perform targeted processing (like band-pass filtering) in the frequency domain before returning to the spatial domain. It tests whether injecting spectral information can improve a conventional architecture.

    ```text
    Input -> [ConvBlock] ----------------------> [ConvBlock] -> Output
                | (Downsample)                        ^ (Upsample)
                v                                     |
             [ConvBlock] --------> [ConvBlock]          |
                |                      ^                |
                v                      | (Skip)         |
             [Bottleneck] -------------+----------------+
    ```

*   **`SpectralUNet` (The Purist Model):** This model operates *entirely* in the frequency domain. It immediately converts the input image with an FFT, processes it using custom complex-valued layers, and only converts it back to the spatial domain with an IFFT at the very end. It tests a more radical, frequency-first approach.

    ```text
    Input -> [FFT] -> [ComplexConv] -> [ComplexBlock] -> [ComplexConv] -> [iFFT] -> Output
    ```

*   **`SpectralUNetDeep` (Frequency Hierarchy):** Extends the purist model into a full encoder/decoder with complex strided downsampling, transposed convolutions for upsampling, and skip connections mirroring TinyUNet—but every operation stays in the frequency domain. This is our “next-wave” architecture for probing whether spectral processing benefits compound across scales.

    ```text
    FFT -> [Encoder (ComplexResidual + Downsample)xL] -> Bottleneck -> [Decoder (Upsample + Skip)xL] -> iFFT
    ```
