# 🌀 Spectral Diffusion

### Exploring frequency-space representations and alternative optimization schemes for diffusion models

---

## 🧭 Overview

**Spectral Diffusion** investigates how operating in **frequency space**—via Fourier or wavelet transforms—affects the behavior, stability, and efficiency of diffusion-based generative models.  
Instead of the usual pixel-domain denoising, we explore **multi-band**, **frequency-aware**, and **adaptive noise** techniques to improve sample quality and reduce computational cost.

This repository provides a flexible framework to:

- Prototype and compare baseline vs. spectral variants of diffusion models  
- Test **non-iterative** or **hybrid flow-matching** approaches  
- Automate experiments with a **Taguchi-inspired design**, minimizing the number of runs while maximizing insight

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

We use a **Taguchi-style factorial design** to minimize test count.

| Factor | Description | Levels |
|--------|--------------|--------|
| A | Frequency-equalized noise | Off / On |
| B | Frequency attention | Off / On |
| C | Sampler type | DDIM / DPM-Solver++ |
| D | Loss weighting | Standard / Frequency-balanced |
| E | Optimizer | Adam / Custom adaptive |

Each run logs its configuration and metrics to allow automatic ranking of factor influence.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|--------------|
| **FID** | Measures realism vs. reference distribution |
| **LPIPS** | Perceptual similarity for visual fidelity |
| **Runtime** | Training and inference speed |
| **Spectral MSE** | Difference between power spectra of generated vs. real samples |

---

## 🚀 Quick Start

```bash
git clone https://github.com/benschneider/spectral-diffusion.git
cd spectral-diffusion
pip install -r requirements.txt
python src/run_experiment.py --config configs/baseline.yaml
```

Results and metrics will appear in results/summary.csv.

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