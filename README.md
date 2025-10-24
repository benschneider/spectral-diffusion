# 🌀 Spectral Diffusion

Exploring how frequency-domain reasoning changes diffusion-model behaviour, efficiency, and optimisation.

---

## Overview
Spectral Diffusion is a research playground for diffusion models with a focus on spectral transforms and performance instrumentation. The repository provides:

- **Architectures**: TinyUNet (spatial), SpectralUNet (frequency-native prototype), spectral adapters.
- **Training pipeline**: DDPM/DDIM-style loop with cosine/linear schedules, ε-prediction, optional SNR weighting.
- **Automation**: Taguchi-style experiment runner with per-run artefacts (config, metrics, logs, checkpoints, system info).
- **Instrumentation**: Throughput (`images_per_second`), convergence speed (`loss_drop_per_second`), FFT timing, structured logging.
- **Evaluation**: Dataset metrics (MSE/MAE/PSNR, optional FID/LPIPS), sampling CLI, figure-generation pipeline for publication-ready plots.

---

## Repository Layout
```
spectral-diffusion/
├── configs/                # Baseline, spectral, Taguchi configs
├── docs/
│   ├── figures/            # Generated plots & summaries
│   ├── spectral_model_research.md
│   └── taguchi_tips.md
├── results/                # Metrics, images, Taguchi outputs (gitignored)
├── scripts/
│   ├── figures/generate_figures.py
│   ├── run_spectral_benchmark.sh
│   ├── run_taguchi_*.sh
│   └── report_summary.py
└── src/
    ├── core/               # Models (TinyUNet, SpectralUNet, losses)
    ├── spectral/           # FFT adapters, complex layers
    ├── training/           # Pipeline, samplers, scheduler
    ├── evaluation/         # Metrics utilities
    └── experiments/        # Taguchi runner
```

---

## Quick Start
```bash
# 1. Clone & install
git clone https://github.com/benschneider/spectral-diffusion.git
cd spectral-diffusion
pip install -r requirements.txt

# 2. (Optional) CIFAR-10 data
mkdir -p data
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o data/cifar-10-python.tar.gz
tar -xzf data/cifar-10-python.tar.gz -C data

# 3. Train + sample (baseline)
RUN_ID=quickstart
python train.py --config configs/baseline.yaml --run-id "$RUN_ID"
python sample.py --run-dir results/runs/"$RUN_ID" --sampler-type dpm_solver++ --num-samples 8 --num-steps 50

# 4. Evaluate
python evaluate.py \
  --generated-dir results/runs/"$RUN_ID"/samples/quickstart_solver \
  --reference-dir path/to/reference \
  --use-lpips

# 5. Automation & analysis
python -m src.experiments.run_experiment \
  --config configs/taguchi_smoke_base.yaml \
  --array configs/taguchi_spectral_array.csv \
  --report-metric loss_drop_per_second
python scripts/report_summary.py --metric loss_drop_per_second --top 5 --include-factors
python scripts/figures/generate_figures.py  # publication-ready plots → docs/figures/

# 6. End-to-end report (synthetic + CIFAR + Taguchi + figures)
scripts/run_full_report.sh
```

**Artefact layout** (per run `results/runs/<run_id>/`):
- `config.yaml`, `system.json`, `logs/train.log`
- `metrics/<run_id>.json`, `checkpoints/*.pt`
- `samples/<tag>/` (images, `metadata.json`)
- Global ledger: `results/summary.csv`

---

## Benchmarks
Generated via `scripts/run_spectral_benchmark.sh` (synthetic) and `configs/benchmark_spectral_cifar*.yaml` (CIFAR-10). Figures are in `docs/figures/`.

| Dataset | TinyUNet loss drop | SpectralUNet loss drop | Images/s (TinyUNet → Spectral) |
|---------|-------------------|------------------------|---------------------------------|
| Synthetic (3 epochs) | ~0.20 | ~1.75 | 800 → 300 |
| CIFAR-10 (1 epoch)   | ~0.94 | ~2.94 | 103 → 48 |

Interpretation: SpectralUNet recovers more loss but costs throughput. Use figures (`docs/figures/loss_metrics_*.png`, `runtime_metrics_*.png`) for publication-ready visuals.

---

## Taguchi Analysis
`src.experiments.run_experiment` with `--report-metric` auto-writes `summary.csv` + `taguchi_report.csv`.
- `docs/taguchi_tips.md` explains interpretation.
- Figures: `docs/figures/taguchi_snr.png`, `taguchi_loss_drop_per_second.png`.

Scripts:
```bash
scripts/run_taguchi_smoke.sh        # tiny smoke sweep
scripts/run_taguchi_minimal.sh      # minimal synthetic sweep
scripts/run_taguchi_comparison.sh   # spectral vs baseline comparison
```

---

## Showcase
- **Benchmark summaries**: `results/spectral_benchmark*/summary.csv`, `summary_metrics.md`
- **Taguchi sweep**: `results/taguchi_spectral_docs/taguchi_report.csv`
- **Figures**: generate via `python scripts/figures/generate_figures.py`
- **Full report pipeline**: `scripts/run_full_report.sh` (synthetic + CIFAR + Taguchi + plots)

---

## Usage Notes
- Toggle spectral adapters: `spectral.enabled: true` with weighting `none | radial | bandpass`.
- Model types: `baseline`, `unet_tiny`, `unet_spectral`.
- Samplers: `ddpm`, `ddim`, `dpm_solver++`, `ancestral`, `dpm_solver2` (extend via `register_sampler`).
- Evaluation CLI supports `--use-fid`, `--use-lpips` when torchmetrics is installed.

---

## Spectral Model Research Plan
Roadmap and experiments described in `docs/spectral_model_research.md`. Current status:
- Complex-valued layers & SpectralUNet prototype implemented.
- Benchmarks (synthetic & CIFAR-10) captured.
- Next steps: tune spectral hyperparameters, explore higher-order samplers, expand spectral diagnostics.

---

## License & Citation
MIT License © 2025 Ben Schneider

```
@software{spectral_diffusion_2025,
  author = {Ben Schneider},
  title  = {Spectral Diffusion: Frequency-Space Diffusion Experiments},
  year   = {2025},
  url    = {https://github.com/benschneider/spectral-diffusion}
}
```
