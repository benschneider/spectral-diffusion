# 🌀 Spectral Diffusion

*How well can a diffusion model learn if it “thinks” in frequency space instead of pixels?*

Spectral Diffusion is a sandbox for answering that question. It lets you run classic diffusion training, swap in spectral adapters or fully spectral models, and compare quality vs. speed with reproducible scripts and reports.

---

## Why frequency space?
Diffusion models usually operate on raw pixels. Yet many signals (images, audio) have structure that is easier to see in the frequency domain. By wrapping or replacing the UNet with FFT-aware components we can:

| Potential gain | What to look for in this repo |
|----------------|--------------------------------|
| Faster convergence | Instrumentation for `loss_drop_per_second`, throughput, and time-to-threshold metrics |
| More efficient sampling | Extended sampler registry (`ddpm`, `ddim`, `dpm_solver++`, `ancestral`, `dpm_solver2`, `masf`) |
| Better high-frequency detail | Uniform frequency corruption in the forward process, amplitude residual + phase correction modules, high-frequency PSNR metrics in reports |
| Controlled ablations | Full-report pipeline now trains spectral variants with and without the new toggles and plots the comparison (`spectral_feature_ablation.png`) |

If you just want the bottom line, jump to **[Results at a Glance](#results-at-a-glance)**.

---

## Getting Started (no jargon required)
```bash
git clone https://github.com/benschneider/spectral-diffusion.git
cd spectral-diffusion
pip install -r requirements.txt

# Optional: download CIFAR-10 once
mkdir -p data
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o data/cifar-10-python.tar.gz
tar -xzf data/cifar-10-python.tar.gz -C data

# Train the baseline TinyUNet and sample a few images
RUN_ID=quickstart
python train.py --config configs/baseline.yaml --run-id "$RUN_ID"
python sample.py --run-dir results/runs/"$RUN_ID" --sampler-type dpm_solver++ --num-samples 8 --num-steps 50

# Evaluate vs. a folder of reference images (optional FID/LPIPS)
python evaluate.py \
  --generated-dir results/runs/"$RUN_ID"/samples/quickstart_solver \
  --reference-dir path/to/reference \
  --use-lpips
```

### Want the full story automatically?
Run one command and grab coffee:
```bash
scripts/run_full_report.sh
```
It will:
1. Benchmark TinyUNet vs. SpectralUNet on synthetic data
2. Train spectral variants with uniform frequency corruption and ARE/PCM modules enabled
3. Repeat on CIFAR-10 (and emit MASF sampler grids for quick inspection)
4. Run a Taguchi sweep over spectral settings and samplers
5. Run a baseline vs toggled spectral ablation (ARE/PCM + uniform corruption + MASF on/off)
6. Generate figures + a report in `docs/figures/`

---

## Results at a Glance
*(Full tables and publication-ready plots live in `docs/figures/`.)*

| Dataset | TinyUNet loss drop | SpectralUNet loss drop | Throughput (images/s) |
|---------|-------------------|------------------------|------------------------|
| Synthetic (3 epochs) | ~0.20 | ~1.75 | 800 → 300 |
| CIFAR-10 (1 epoch)   | ~0.94 | ~2.94 | 103 → 48 |

**Interpretation:** SpectralUNet is more aggressive at reducing loss but costs extra compute today. The Taguchi report (`results/taguchi_spectral_docs/taguchi_report.csv`) shows which spectral settings and samplers matter most.

---

## How things fit together

### Key components
- `src/core/` – TinyUNet (spatial) and SpectralUNet (complex convolutions).
- `src/spectral/` – FFT adapters and complex layers.
- `src/training/` – Training pipeline, scheduler, sampler registry.
- `src/visualization/` – Reusable helpers for figures + summaries.
- `scripts/run_full_report.sh` – End-to-end benchmark + report pipeline.

See **[`docs/architecture.md`](docs/architecture.md)** for diagrams and a more detailed map.

### Theory primer
Curious about the spectral weighting, FFT/iFFT flow, or why we track high-frequency energy? Read the short note in **[`docs/theory.md`](docs/theory.md)** for a gentle walkthrough.

---

## Workflows

| Task | Command(s) |
|------|------------|
| Train & sample baseline | `python train.py ...` + `python sample.py ...` |
| Evaluate generated images | `python evaluate.py --generated-dir ... --use-fid --use-lpips` |
| Synthetic vs Spectral benchmark | `scripts/run_spectral_benchmark.sh` |
| CIFAR-10 benchmark | `python train.py --config configs/benchmark_spectral_cifar.yaml ...` |
| Taguchi sweeps | `scripts/run_taguchi_smoke.sh` / `run_taguchi_minimal.sh` / `run_taguchi_comparison.sh` |
| Smoke report (fast check) | `scripts/run_smoke_report.sh` |
| Make plots & summary | `python scripts/figures/generate_figures.py` |
| Full pipeline (benchmarks + Taguchi + ablation + figures) | `scripts/run_full_report.sh` (includes spectral-feature ablation figure) |

All generated metrics land under `results/…` and are safe to delete/regenerate.

---

## Usage notes for explorers
- **Models:** `baseline`, `unet_tiny`, `unet_spectral` (set `model.type` in YAML). Enable amplitude residuals + phase correction via `model.enable_amp_residual` / `model.enable_phase_attention`.
- **Spectral adapters:** toggle with `spectral.enabled`, choose weighting (`none`, `radial`, `bandpass`), apply to `input/output/per_block`.
- **Diffusion forward noise:** set `diffusion.uniform_corruption: true` to equalize SNR decay across frequencies.
- **Samplers:** `ddpm`, `ddim`, `dpm_solver++`, `ancestral`, `dpm_solver2`, `masf` (extend via `register_sampler`).
- **Ablations:** the full report writes `ablation/summary.csv` and `figures/spectral_feature_ablation.png`, contrasting spectral configs with and without uniform corruption + ARE/PCM + MASF.
- **Metrics:** dataset metrics include FID/LPIPS (torchmetrics-enabled), high-frequency PSNR, convergence stats (`loss_drop_per_second`), throughput, FFT timing.

Looking to extend the project? See **`docs/spectral_model_research.md`** for the ongoing research plan and ideas for new ablations.

---

## Documentation bundle
- **[`docs/theory.md`](docs/theory.md)** – Layperson-friendly explanation of spectral diffusion concepts.
- **[`docs/architecture.md`](docs/architecture.md)** – How configs, pipelines, samplers, and reports connect.
- **[`docs/config_reference.md`](docs/config_reference.md)** – CLI flags, YAML fields, and automation scripts at a glance.
- **[`docs/spectral_model_research.md`](docs/spectral_model_research.md)** – Roadmap, experiments, next hypotheses.
- **[`docs/taguchi_tips.md`](docs/taguchi_tips.md)** – How to read the Taguchi S/N reports.
- **[`docs/figures/summary.md`](docs/figures/summary.md)** – Latest benchmark narrative after running the full report.
- **`docs/notebooks/`** – Slot for interactive demos (planned): start with a notebook that trains TinyUNet vs SpectralUNet for a few epochs and plots loss/time.

---

## License & citation
MIT License © 2025 Ben Schneider

```
@software{spectral_diffusion_2025,
  author = {Ben Schneider},
  title  = {Spectral Diffusion: Frequency-Space Diffusion Experiments},
  year   = {2025},
  url    = {https://github.com/benschneider/spectral-diffusion}
}
```
