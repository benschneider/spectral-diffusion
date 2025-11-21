# Report Cleanup Plan – Generic Spectral Diffusion Study

This document defines a clean, profile‑agnostic reporting structure for all spectral‑diffusion experiments. The objective is to ensure every report answers the same core scientific questions with a minimal, interpretable, and reproducible set of figures.

---

## 1. Core Questions

A cleaned report must answer three questions:

1. **Stability / Convergence**
   - Do training runs converge reliably?
   - How smooth or erratic is optimization under the tested factors?

2. **Sample‑Efficiency / Runtime**
   - How much loss reduction per unit time does each configuration achieve?
   - What is the speed–quality trade‑off?

3. **Qualitative Impact**
   - What visible differences arise (artefacts, texture recovery, sharpness) at comparable quantitative performance?

Only figures contributing to these questions belong in the main report.

---

## 2. Required Figures (Main Report)

### 2.1 Stability & Convergence

- `loss_curve_synthetic.png`  
  Single‑panel plot showing 2–4 representative configurations (e.g., spatial, FFT‑mild, FFT‑equalised).

- `loss_curve_cifar.png`  
  Same structure for CIFAR‑10.

Notes:
- Curves must show initial→final loss, relative smoothness, and highlight SNR‑related settings via legend.
- Plotting code must avoid plotting all configurations; only representative ones.

---

### 2.2 Efficiency vs Speed

- `tradeoff_loss_vs_speed_synthetic.png`
- `tradeoff_loss_vs_speed_cifar.png`

Requirements:
- X-axis: `images_per_second` or inverse runtime.
- Y-axis: `loss_drop_per_second`.
- Highlight the best configurations explicitly.
- Keep plot density minimal and interpretable.

---

### 2.3 Taguchi Factor Effects

- `taguchi_main_effects_primary.png`
- `taguchi_contrib_primary.png`

The set of “primary factors” is selected per profile:

Examples:
- **SNR profile:** `snr_ratio`, `spectral_noise_shaping_strength`, `snr_weighting_mode`, `spectral_adapter_placement`.
- **Sampler profile:** `sampler_type`, `sampling_steps`, `snr_ratio`.
- **Curriculum profile:** `curriculum_mode`, `train_steps`.

Requirements:
- Use `taguchi_report.csv` and `factor_mapping.json` for factor names and levels.
- Labels must be human‑readable.

Optional sanity plot:
- For SNR profiles: measured SNR vs target.

---

### 2.4 Qualitative Examples

Limit to 1–2 grids:

- `samples_profile_comparison_1.png`  
  Compare noise modes, samplers, or architectures at matched final loss.

- `samples_profile_comparison_2.png` (optional)

Requirements:
- Clearly state which factors differ.
- Keep only two grids for the main narrative.

---

## 3. Appendix / Diagnostics (Demoted Figures)

These remain generated but not shown in main summary:

### 3.1 Noising / FFT Visualizations
- All `*_noising_chain.png`
- All `noise_*.png`
- All `corrupted_*.png`

### 3.2 Distributions & Breakdown
- `loss_final_distribution_*`
- `images_per_second_distribution_*`
- `loss_metrics_*`
- `runtime_metrics_*`

### 3.3 Full Taguchi Interactions
- All `taguchi_interaction_*`
- All `taguchi_interactions_*.csv`

These are referenced only as supplementary material.

---

## 4. Taguchi Analysis Requirements

1. Default to `taguchi_report.csv` for factor/metric lookup.  
2. Use `factor_mapping.json` to identify factor columns and rename levels.  
3. Limit main‑effects to profile‑defined primary factors.  
4. Apply clean human‑readable labels to all plots.

---

## 5. Report V2 Output Structure

Each experiment produces:

```
report_v2/
  summary.md
  summary.pdf
  images/
    loss_curve_synthetic.png
    loss_curve_cifar.png
    tradeoff_loss_vs_speed_synthetic.png
    tradeoff_loss_vs_speed_cifar.png
    taguchi_main_effects_primary.png
    taguchi_contrib_primary.png
    samples_profile_comparison_1.png
    samples_profile_comparison_2.png
  appendix/
    noise_chains/
    taguchi_interactions/
    distributions/
```

Only images under `report_v2/images/` appear in the main report.

---

## 6. Summary.md Template (Exact Structure)

1. **Title + metadata**
2. **Stability & Convergence**
   - synthetic → figure + bullets  
   - cifar → figure + bullets
3. **Efficiency vs Runtime**
   - synthetic → figure + bullets  
   - cifar → figure + bullets
4. **Taguchi Factor Effects**
   - main effects → figure + bullets  
   - contributions → figure + bullets  
   - optional SNR sanity → figure + bullets
5. **Qualitative Samples**
   - comparison 1  
   - comparison 2 (optional)
6. **Key Takeaways**
7. **Appendix pointer**

---

## 7. Implementation Notes

- A new script (`scripts/generate_report_v2.py`) should produce all required plots and `summary.md`.
- Existing plotting code moves to `scripts/figures/archive/`.
- The main report must stay concise and directly mapped to the three core research questions.

