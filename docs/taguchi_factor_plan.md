# Taguchi Factor Expansion Plan (L16 Two-Level Design)

The goal is to move from the current L8 (5 binary factors) to an L16 experiment so we can examine up to 15 two-level factors in a single batch (16 runs). The table below assigns letter labels following the Taguchi convention and lists the toggles we plan to sweep, using descriptive names instead of acronyms.

| Letter | Factor (two-level) | Level 1 | Level 2 | Notes / Equation | Status / Comments |
|--------|--------------------|---------|---------|------------------|------------------|
| **A** | **Spectral noise equalisation** (`spectral.freq_equalized_noise`) | Off – standard Gaussian | On – uniform spectral mask `m(k,ℓ) = √((r/r_min)^2 + 1)` | Existing factor. Keeps FFT noise energy spread across bands. | ✅ Already in L8 – carries into L16. |
| **B** | **Phase correction module** (`model.enable_phase_attention`) | Disabled | Enabled (multi-head attention on phase) | Formerly PCM. Applies attention to phase before IFFT. | ✅ Existing factor; rename to “phase correction module.” |
| **C** | **Reverse sampler** (`sampling.sampler_type`) | DDIM | DPM-Solver++ | Existing factor C. | ✅ Already hooked up. |
| **D** | **Spectral adapters** (`spectral.enabled`) | TinyUNet (spatial) | SpectralUNet (complex FFT pipeline) | Existing mixed backbone factor. | ✅ Stays as-is. |
| **E** | **Amplitude residual encoder** (`model.enable_amp_residual`) | Off | On | Isolates ARE contribution. | 🆕 To wire into Taguchi runner. |
| **F** | **Frequency band smoothing** (MASF) | Off | On (MASF α=0.9) | Per-band EMA smoothing in sampler. | 🆕 Requires sampler toggle in runner. |
| **G** | **Uniform mask gain** | Full strength (1.0×) | Reduced gain (0.5×) | Partial vs full frequency equalisation. | 🆕 Need to expose gain parameter. |
| **H** | **High-frequency loss weighting** (`loss.spectral_weighting`) | None | Radial weighting | Emphasise high-ω in reconstruction loss. | 🆕 Simple config toggle. |
| **I** | **SNR-based loss scaling** (`diffusion.snr_weighting`) | Disabled | Enabled | Loss scaled by SNR schedule. | 🆕 Already supported; add to runner. |
| **J** | **Phase attention capacity** (`model.phase_heads`) | 1 head | 4 heads | Tests deeper PCM vs minimal. | 🆕 Map Taguchi column to head count. |
| **K** | **Learning-rate schedule** | Constant LR | Cosine decay | Check if spectral stack likes decay. | 🆕 Requires LR scheduler toggle. |
| **L** | **Sampling steps** (`sampling.num_steps`) | 50 | 100 | Spectral variants may need more steps. | 🆕 Runner must override per design row. |
| **M** | **Coarse-resolution warm-up** | None | Short 8×8 pretrain before full run | Implements low-res warm-up. | 🆕 Needs pipeline hook (run tiny warm-up). |
| **N** | **Spectral adapter placement** (`spectral.apply_to`) | Input only | Input + output | Tests adapter placement strategy. | 🆕 Set list on config. |
| **O** | **Uniform mask formula** | Current sqrt mask | Alternative power-law mask | Compare mask definitions. | 🆕 Implement second mask branch. |

> **Optional / Deferred:** FFT backend (CPU vs GPU) can become a future factor once GPU-native FFT is available. Cross-domain weight recycling is left out because the ablations showed little impact compared to the new toggles above.

### Next Steps
1. Generate an L16 orthogonal array with 15 columns (A–O) and add it to `configs/` (e.g., `taguchi_spectral_L16.csv`).
2. Extend `src/experiments/run_experiment.py` to interpret new columns and mutate the config accordingly (e.g., set LR schedule, warm-up flag).
3. Update the reporting pipeline to note the new factors in Taguchi summaries.
4. Run a pilot L16 batch to verify runtime footprint and metric quality.
