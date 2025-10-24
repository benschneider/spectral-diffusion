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
results/runs/<run_id>/
├── config.yaml
├── system.json
├── logs/train.log
├── metrics/<run_id>.json
├── checkpoints/*.pt
└── samples/<tag>/grid.png, metadata.json
```
A global ledger `results/summary.csv` collects headline metrics; Taguchi sweeps append `taguchi_report.csv`.

## 4. Figure/report workflow
1. Generate runs (manual CLI or `scripts/run_full_report.sh`).
2. `src/visualization.collect.clean_summary` deduplicates and labels entries.
3. `src/visualization.figures.generate_figures` produces
   - `loss_metrics_*.png`, `runtime_metrics_*.png`
   - `taguchi_snr.png`
   - `docs/figures/summary.md`
4. README Showcase points to these artefacts.

## 5. How to extend
- **Add a sampler**: subclass `Sampler` (`src/training/sampling.py`), register via `register_sampler("name", Class)`.
- **Add a spectral adapter**: follow the patterns in `src/spectral/adapter.py` or `complex_layers.py`.
- **Add a dataset**: update `configs/` and `src/training/builders.py` for a new loader.
- **Add plots**: extend `src/visualization/figures.py`, then call from `scripts/figures/generate_figures.py`.

With this modular layout you can import the same components in notebooks, CLI scripts, or experiments without rewriting plumbing.
