# 🌀 Spectral Diffusion

### Exploring frequency-space representations and alternative optimization schemes for diffusion models

---

## 🧭 Overview

**Spectral Diffusion** investigates how frequency-domain processing changes diffusion-model behaviour and efficiency. The codebase now includes:

- A diffusion-ready UNet (timestep conditioning, optional spectral pre/post transforms)
- Unified training pipeline with cosine β schedule, ε-prediction, and optional SNR weighting
- Taguchi batch runner that persists per-run configs/metrics for downstream analysis
- Instrumentation for throughput (`images_per_second`) and convergence speed (`loss_drop_per_second`) across variants

---

## 🎯 Project Goals

1. **Baseline:** Build and benchmark a standard diffusion pipeline (DDPM/DDIM).  
2. **Frequency-space methods:**  
   - Equalized noise injection  
   - Band-specific denoising  
   - Phase-preserving reconstruction  
3. **Optimization experiments:**  
   - Classic gradient descent vs. flow-matching vs. implicit equilibrium updates  
4. **Automation:**  
   - Run structured experiments via YAML + orthogonal arrays  
   - Compute Taguchi signal-to-noise ratios to identify dominant factors  
5. **Evaluation:**  
   - Track FID, LPIPS, runtime, and spectral consistency across runs

---

## ⚙️ Repository Structure
```
spectral-diffusion/
├── src/
│   ├── core/                # Model architectures and losses
│   ├── spectral/            # Frequency transforms and band utilities
│   ├── training/            # Unified training pipeline (baseline + variants)
│   ├── evaluation/          # Metrics, FID, LPIPS, runtime
│   └── experiments/         # Taguchi automation scripts
├── configs/
│   ├── baseline.yaml
│   ├── variants.yaml
│   └── L8_array.csv
├── notebooks/               # Analysis and visualization
├── results/
│   ├── metrics/
│   └── images/
└── README.md
```
---

## 🧪 Experimental Design

Taguchi-style orthogonal arrays drive controlled sweeps. Each CSV row maps to overrides in `TaguchiExperimentRunner._build_config_from_row`, so variants share the unified diffusion pipeline while toggling spectral options or sampler strategies.

---

## 📊 Key Metrics

| Metric | Description |
|--------|--------------|
| **loss_drop_per_second** | Convergence efficiency (higher is better) |
| **images_per_second** | Throughput under current config/hardware |
| **spectral_time_seconds** | Cumulative FFT overhead per run |
| **loss_threshold_time** | Wall time to reach configured target loss (if set) |
| **FID / LPIPS** | (Coming soon) image quality comparators for sampled images |

---

## 🚀 Quick Start

```bash
git clone https://github.com/benschneider/spectral-diffusion.git
cd spectral-diffusion
pip install -r requirements.txt
# Optional: fetch CIFAR-10 locally (or set data.download: true in configs)
mkdir -p data
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o data/cifar-10-python.tar.gz
tar -xzf data/cifar-10-python.tar.gz -C data
python train_model.py --config configs/baseline.yaml --dry-run   # sanity check
python train_model.py --config configs/baseline.yaml              # diffusion training (ε-pred)
# Optional Taguchi batch
python -m src.experiments.run_experiment
# Validation helper (removes dry-run artifacts unless --keep-artifacts is provided)
python validate_setup.py
# Benchmark baseline vs spectral throughput
python benchmarks/benchmark_fft.py --device cpu --batch-size 16
```

Results and metrics will appear in `results/summary.csv`.

---

# Run spectral unit tests
python -m pytest tests/test_spectral_fft.py
python -m pytest tests/test_tiny_unet_spectral.py

# Benchmark baseline vs spectral throughput
python benchmarks/benchmark_fft.py --device cpu --batch-size 16

## 🔄 Flow Overview

### Conceptual Diffusion Cycle

```mermaid
flowchart LR
    A["x₀"] --> B{Spectral enabled?}
    B -->|No| C["xₜ = αₜ x₀ + σₜ ε"]
    B -->|Yes| B1["FFT(x₀)"] --> C
    C --> D["UNet(xₜ, t)"]
    D --> E["Target builder (ε / v / x₀)"]
    E --> F["Diffusion loss + SNR weighting"]
    F --> G["Optimizer step"]
    G --> H{Spectral enabled?}
    H -->|Yes| H1["iFFT(output)"] --> A
    H -->|No| A
```

This highlights the shared diffusion loop. Spectral toggles route data through FFT/iFFT while reusing the same timestep-conditioned UNet.

### System Flow (Single Run)

```mermaid
flowchart LR
    A["Config YAML"] --> B["train_model.py"]
    B --> C["TrainingPipeline"]
    C --> D["Diffusion Loop"]
    D --> E["metrics JSON"]
    D --> F["run.log"]
    D --> G["summary.csv"]
    G --> H["analysis notebooks / Taguchi stats"]
```

### Experiment Flow (Batch Runs)

```mermaid
flowchart TD
    subgraph Inputs
        L8["Taguchi array<br/>configs/L8_array.csv"]
        CFG["Base YAML<br/>configs/baseline.yaml"]
    end
    L8 --> RUNNER["TaguchiExperimentRunner<br/>src/experiments/run_experiment.py"]
    CFG --> RUNNER
    RUNNER --> EXEC["TrainingPipeline per row"]
    EXEC --> METRICS["Per-run artifacts"]
    METRICS --> H["summary.csv"]
    METRICS --> I["metrics JSON"]
    METRICS --> J["log dir"]
```

Each design-row produces its own run ID, config snapshot, metrics JSON, and an entry in `results/summary.csv`. Aggregated Taguchi S/N analysis will build on these artifacts.

### Model Variant Dispatch

```mermaid
flowchart TD
    X["model.type<br/>(YAML)"] -->|baseline / baseline_conv| Y["BaselineConvModel<br/>src/core/model.py"]
    X -->|unet_tiny| Z["TinyUNet<br/>src/core/model_unet_tiny.py"]
    Y --> R["MODEL_REGISTRY<br/>src/core/model.py"]
    Z --> R
    R --> S["build_model()<br/>nn.Module"]
```

- `train_model.py` passes the full config to `build_model()` (`src/core/model.py`).
- `MODEL_REGISTRY` maps type strings to constructors. Register additional architectures here.
- `BaselineConvModel` suits synthetic smoke tests; `TinyUNet` targets 32×32 CIFAR-10 reconstructions.

### Data Source Selection

```mermaid
flowchart TD
    A2["data.source<br/>(YAML)"] -->|synthetic| B2["_make_synthetic_dataloader()<br/>TensorDataset"]
    A2 -->|cifar10| C2["_make_cifar10_dataloader()<br/>torchvision.datasets"]
    C2 --> D2["ReconstructionWrapper<br/>(img, img) pairs"]
    B2 --> E2["torch.utils.data.DataLoader"]
    D2 --> E2
```

- Synthetic mode generates random tensors sized by `data.channels/height/width`, limiting iterations with `training.num_batches`.
- CIFAR-10 mode expects `data.root/cifar-10-batches-py` (or `data.download: true`) and wraps samples so the reconstruction loss applies.
- Spectral toggles live under `spectral.*` in the config (`enabled`, `normalize`, `weighting` ∈ {`none`, `radial`, `bandpass`}); when enabled, the model performs an FFT → optional weighting → inverse FFT round-trip before and after the UNet.

### What Gets Logged & How to Read It

- `results/logs/<run_id>/run.log` – chronological training messages (loss snapshots, runtime).
- `results/logs/<run_id>/config.yaml` – frozen effective configuration.
- `results/metrics/<run_id>.json` – structured metrics:
  - `loss_mean`, `loss_last`, `mae_mean` → reconstruction quality proxies.
  - `runtime_seconds`, `num_steps`, `epochs` → throughput and coverage.
  - `steps_per_second`, `images_per_second`, `runtime_per_epoch` → throughput/efficiency.
  - `loss_initial`, `loss_final`, `loss_drop`, `loss_drop_per_second` → convergence speed indicators.
  - `loss_threshold_*` entries (optional) record when a configured target loss is reached.
  - `spectral_calls`, `spectral_time_seconds` → FFT usage/overhead when spectral mode is enabled.
- `results/summary.csv` – append-only ledger linking run IDs to config/metrics; Taguchi batches now append here automatically.
- `validate_setup.py` removes its dry-run artifacts by default; pass `--keep-artifacts` to inspect them.

> ❗️Still in progress:
> - Spectral transforms (FFT pipeline), diffusion-specific losses/schedulers, and perceptual metrics (FID/LPIPS) remain placeholders.
> - Taguchi experiment runner analysis (factor ranking, S/N ratios) is the next milestone.

### Baseline vs. Spectral Snapshot

| Aspect | Baseline (`baseline_conv`) | Spectral (current/roadmap) |
|--------|----------------------------|----------------------------|
| Domain | Pixel / spatial            | Frequency (FFT round-trip) |
| Noise | Standard Gaussian           | Frequency-equalized noise (future) |
| Attention | None                     | Frequency attention toggle (future) |
| Loss | Reconstruction MSE / MAE     | Band-aware diffusion loss (TODO) |
| Config toggle | `spectral.enabled: false` | `spectral.enabled: true` + `weighting: radial/bandpass` |
| Goal | Stability + quick testing    | Probe spectral benefits, Taguchi optimization |

---

## 🧠 Research Questions
- Does learning in frequency space reduce redundancy in denoising trajectories?
- Can adaptive noise equalization stabilize or accelerate convergence?
- Are hybrid flow/ODE methods preferable to iterative diffusion in high-frequency domains?
- What combination of parameters (Taguchi analysis) yields the best trade-off between speed and fidelity?

---

## 📄 License

MIT License © 2025 Ben Schneider



## 💡 Citation

If you use this repository in academic work, please cite it as:

```
@software{spectral_diffusion_2025,
  author = {Ben Schneider},
  title = {Spectral Diffusion: Frequency-Space Diffusion Experiments},
  year = {2025},
  url = {https://github.com/benschneider/spectral-diffusion}
}
```