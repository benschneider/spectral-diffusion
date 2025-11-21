# Report Cleanup Plan – Frequency-SNR Study

This file is a focused brief for future work on simplifying the auto-generated reports so they directly support **concrete research questions** (SNR, architecture, samplers, curricula, etc.) without overwhelming the reader.

The goal is to reduce the report to a small, interpretable set of figures, with everything else treated as auxiliary diagnostics. The same structure should work for different “report profiles” (e.g., SNR study, sampler study) by changing configuration rather than code.

---

## 1. Storyline & Questions

The cleaned report should answer questions in three broad categories:

1. **Stability / convergence**
   - Do the training runs converge, and how smooth/robust is the optimization under the factors we are studying?
2. **Sample‑efficiency / runtime**
   - For each configuration, how much loss reduction do we get per unit time, and how does that trade off against throughput?
3. **Qualitative impact**
   - What visible changes do the factors cause in samples (e.g., artefacts, sharpness, structure) at comparable quantitative performance?

Everything we keep in the main report should be traceable to at least one of these three categories, regardless of whether the “primary factors” are SNR, samplers, architectures, or curricula.

---

## 2. Figures to Keep (Main Report)

### 2.1 Stability & convergence

Keep exactly one loss curve per dataset family:

- `figures/loss_curve_synthetic.png`
  - Shows convergence/stability on synthetic data.
  - Use a single panel; ensure the legend calls out the key SNR-related settings used (e.g., “spatial baseline”, “FFT mild”, “FFT equalised”).

- `figures/loss_curve_cifar.png`
  - Shows convergence/stability on CIFAR.
  - Even if noisy, the initial→final loss and rough trend are important.

Implementation notes:
- If necessary, adjust the plotting code so these curves aggregate or highlight only a small number of representative runs (e.g., one per SNR mode), not every possible combination.

### 2.2 Efficiency vs speed

Keep the trade‑off plots:

- `figures/tradeoff_loss_vs_speed_synthetic.png`
- `figures/tradeoff_loss_vs_speed_cifar.png`

These should:
- Plot `loss_drop_per_second` vs `images_per_second` (or `runtime_seconds` inverted) for a *small* number of SNR configurations (again, spatial vs FFT vs equalised, and a few `snr_ratio` values).
- Highlight the “best” configurations in the legend or annotations.

### 2.3 Taguchi factor effects

Keep Taguchi factor plots, but constrain them to a *configurable* set of “primary factors” for the current study.

- `figures/taguchi_main_effects.png`
  - Only include factors from a configured list, e.g.:
    - For an **SNR profile**: `snr_ratio`, `spectral_noise_shaping_strength`, `snr_weighting_mode`, `spectral_adapter_placement`.
    - For a **sampler profile**: `sampler_type`, `sampling_steps`, `snr_ratio`.
    - For a **curriculum profile**: `curriculum_mode`, `train_steps`, etc.

- `figures/taguchi_contrib.png`
  - Same factor set as above.
  - Directly answer “which knobs matter most for the primary metric?” (default: `loss_drop_per_second`).

- Optional but useful: a simple metric sanity plot (if not already present), e.g.:
  - For SNR: `snr_measured` versus `snr_ratio` across runs.
  - For other profiles: any diagnostic that checks the factor is doing what it claims.

Implementation notes:
- `scripts/archive/analyze_taguchi_cli.py` currently infers factors heuristically. For this report cleanup, it should preferentially use:
  - `factor_mapping.json` under `taguchi/`
  - or factor columns in `taguchi_report.csv`
  - and explicitly ignore `config_path`, `metrics_path`, `timestamp`, `display_name` etc.

### 2.4 Qualitative examples

Keep only a *small* set of sample grids, structured around the current profile’s questions:

- For an SNR profile:
  - One CIFAR grid comparing spatial vs FFT vs equalised noise modes at matched `snr_ratio` and similar final loss.
  - One CIFAR grid comparing static vs adaptive SNR weighting at a “hard” setting (e.g., high SNR).
- For a sampler profile:
  - One grid comparing DDIM vs DPM‑Solver++ vs MASF at fixed steps.
- For an architecture profile:
  - One grid comparing baseline vs spectral vs deep spectral UNets.

Implementation notes:
- Either:
  - Reuse the existing sample grids but only link two of them in the summary, or
  - Add a script that selects exemplar runs based on Taguchi insights and regenerates just those grids.

---

## 3. Figures to Demote (Appendix / Diagnostics)

These should stay on disk under `figures/`, but not be linked prominently from the main summary:

### 3.1 Noising / FFT visuals

- `synthetic_noising_chain.png`
- `cifar_noising_chain.png`
- `noise_gaussian.png`
- `noise_uniform.png`
- `noise_difference_uniform_minus_gaussian.png`
- `corrupted_gaussian.png`
- `corrupted_uniform.png`
- `corrupted_difference_uniform_minus_gaussian.png`

Rationale:
- These validate the implementation and are useful for debugging, but most readers can’t interpret them deeply. Keep them as an appendix or “Diagnostics” section.

### 3.2 Full distributions / metrics breakdown

- `loss_final_distribution_*.png`
- `images_per_second_distribution_*.png`
- `loss_metrics_*.png`
- `runtime_metrics_*.png`

Rationale:
- High granularity is nice, but the trade‑off plots and a few summary numbers are enough for the main narrative. The distributions can be referenced only if needed.

### 3.3 Full Taguchi interaction grid

- All `taguchi_interaction_*.png`
- All `taguchi_interactions_*.csv`

Rationale:
- These are valuable for detailed analysis but overwhelming in the main report. Keep them as supplementary material; in the main text, describe only the top 1–2 interactions (e.g., J×F or J×B) in prose.

---

## 4. Taguchi Analysis Cleanups Needed

To support the above, the Taguchi analysis should be tightened as follows:

1. **Use the right CSV**
   - Prefer `taguchi_report.csv` (if present) over `summary.csv` as the input for `analyze_taguchi_cli.py`.

2. **Use factor mapping**
   - Read `factor_mapping.json` from the Taguchi run root and use its `mapping` / `factor_levels` to:
     - Identify which columns are factors.
     - Optionally rename them from column letters to factor names (e.g., A→`phase_attention_capacity`).

3. **Filter to profile-defined primary factors**
   - When generating main‑effects and contributions figures, limit factors to a configured list of “primary factors” for the current profile. Examples:
     - SNR profile: `snr_ratio`, `spectral_noise_shaping_strength`, `snr_weighting_mode`, `spectral_adapter_placement`.
     - Sampler profile: `sampler_type`, `sampling_steps`, optionally `snr_ratio`.
     - Curriculum profile: `curriculum_mode`, `train_steps`, etc.

4. **Clean factor labels**
   - Convert raw level labels into short, human‑readable names in the plots (e.g., “0.8 / 1.0 / 1.4” displayed as such, or “off / mild / strong”).

---

## 5. Summary of “Must‑Have” vs “Nice‑to‑Have”

**Must‑have in main report**
- Synthetic loss curve (`loss_curve_synthetic.png`).
- CIFAR loss curve (`loss_curve_cifar.png`).
- Loss vs speed trade‑off plots (`tradeoff_loss_vs_speed_*`).
- Taguchi main effects and contributions for SNR‑related factors (`taguchi_main_effects.png`, `taguchi_contrib.png`).
- 2–3 carefully chosen sample grids illustrating SNR effects qualitatively.

**Nice‑to‑have / appendix**
- All FFT/noising visualizations.
- Detailed distributions (`*_distribution_*.png`, `*_metrics_*.png`).
- Full Taguchi interaction matrices (plots + CSVs).

This plan is intentionally specific so a “fresh mind” (or a future version of you) can implement the cleanup by:
- tightening Taguchi factor detection,
- trimming figure generation to the core set,
- and moving the rest into a clearly labeled diagnostics/appendix area without deleting anything. 

---

## 6. Clean-Slate Report v2 Specification

This section describes a **from-scratch** report layout, assuming we archive the current `scripts/figures/*` machinery and re-implement a simpler reporting stack.

### 6.1 Output structure (profile-agnostic)

Target directory layout for a single report run (root = `results/full_report_32x32_<stamp>`):

- `synthetic/summary.csv` (unchanged: numeric metrics)
- `cifar/summary.csv` (unchanged)
- `taguchi/summary.csv`, `taguchi/taguchi_report.csv`, `taguchi/factor_mapping.json` (unchanged)
- `report_v2/` (new)
  - `report_v2/summary.md` – main human-readable report (source of PDF/HTML).
  - `report_v2/summary.pdf` – rendered PDF (via pandoc/pypandoc).
  - `report_v2/images/` – all plots that the markdown references, with stable names:
    - `loss_curve_synthetic.png`
    - `loss_curve_cifar.png`
    - `tradeoff_loss_vs_speed_synthetic.png`
    - `tradeoff_loss_vs_speed_cifar.png`
    - `taguchi_main_effects_primary.png` (main effects for primary factors)
    - `taguchi_contrib_primary.png` (contributions for primary factors)
    - Optional profile-specific diagnostics, e.g.:
      - `taguchi_snr_measured_vs_target.png` (for SNR profile)
    - `samples_profile_comparison_1.png` (e.g., noise modes, samplers, or architectures)
    - `samples_profile_comparison_2.png` (second comparison if needed)
  - `report_v2/appendix/` – optional diagnostics:
    - `appendix/noise_chains/…` (FFT/noising visuals)
    - `appendix/taguchi_interactions/…` (full interaction plots)
    - `appendix/distributions/…` (loss / runtime distributions)

The idea is: **only images under `report_v2/images/` are linked in the main summary**; everything in `appendix/` is reachable by filename but not referenced prominently.

### 6.2 Summary.md layout (exact sections, generic)

`summary.md` should follow this structure:

1. **Title + experiment metadata**
   - Title line: e.g. “Diffusion Study – 32×32 synthetic + CIFAR (profile: frequency‑SNR)” or another profile label.
   - Bullet list:
     - Dataset(s): synthetic spectral 32×32, CIFAR‑10 32×32
     - Model(s): TinyUNet, SpectralUNet (if used)
     - Key knobs: `snr_ratio`, `spectral_noise_shaping_strength`, `snr_weighting_mode`
     - Report root path

2. **Section 1 – Stability & Convergence**
   - Subsection: Synthetic
     - Embed `images/loss_curve_synthetic.png`
     - 2–3 bullets describing:
       - initial vs final loss range
       - any obvious instability (if present)
       - which SNR settings are plotted (reference legend explicitly)
   - Subsection: CIFAR
     - Embed `images/loss_curve_cifar.png`
     - Same style of bullets; acknowledge noisier curves if applicable.

3. **Section 2 – Sample-Efficiency vs Runtime**
   - Subsection: Synthetic
     - Embed `images/tradeoff_loss_vs_speed_synthetic.png`
     - Bullets answering:
       - Which configuration gives the best `loss_drop_per_second`?
       - Which configuration is the fastest (`images_per_second`) at acceptable loss?
   - Subsection: CIFAR
     - Embed `images/tradeoff_loss_vs_speed_cifar.png`
     - Same style of bullets, highlighting whether the SNR–efficiency story matches synthetic.

4. **Section 3 – Taguchi Factor Effects**
   - Subsection: Main effects
     - Embed `images/taguchi_main_effects_primary.png`
     - Describe:
       - Factors included (by name: `snr_ratio`, `spectral_noise_shaping_strength`, `snr_weighting_mode`, optionally `spectral_adapter_placement`).
       - For each factor: best level and rough effect size on `loss_drop_per_second`.
   - Subsection: Contributions
     - Embed `images/taguchi_contrib_primary.png`
     - Summarise:
       - Relative importance ordering (e.g., “J (snr_ratio) > F (noise mode) > B (weighting mode) > G (adapter placement)”).
   - Optional subsection: profile-specific sanity
     - For SNR: embed `images/taguchi_snr_measured_vs_target.png` (or similar) and summarise whether `snr_measured` tracks `snr_ratio`.
     - For other profiles: use an appropriate diagnostic (e.g., sampler steps vs quality).

5. **Section 4 – Qualitative Samples**
   - Subsection: Profile comparison (1)
     - Embed `images/samples_profile_comparison_1.png`.
     - Explain in 2–3 bullets which factor(s) differ and what visual change they produce.
   - Subsection: Profile comparison (2) – optional
     - Embed `images/samples_profile_comparison_2.png`.
     - Same style of bullets for a second comparison axis (if needed).

6. **Section 5 – Key Takeaways**
   - A short bulleted list (3–6 bullets) explicitly answering:
     - Which SNR settings are best for stability?
     - Which SNR settings are best for sample‑efficiency?
     - Do FFT equalisation and adaptive weighting help, harm, or trade off?
     - Any caveats (e.g. stronger effects on synthetic than CIFAR).
   - If Taguchi suggests follow‑up experiments (e.g. strong J×F interaction), mention them here.

7. **Appendix pointer**
   - One paragraph listing which additional diagnostics are available in `report_v2/appendix/` (noise chains, full Taguchi interactions, distributions), but without inlining those figures.

### 6.3 Archiving current reporting

When implementing Report v2:

- Move or alias:
  - Existing `scripts/figures/generate_figures.py` and related helper functions into `scripts/figures/archive/` (or keep them but stop calling them from the default workflows).
  - Old-style figures (FFT/noise, full interaction grids) should still be generated (optionally via a separate `--include-appendix` flag) but not referenced from `report_v2/summary.md`.
- Add a new entrypoint (e.g. `scripts/generate_report_v2.py`) that:
  - Loads `synthetic/summary.csv`, `cifar/summary.csv`, and `taguchi/taguchi_report.csv`.
  - Produces only the v2 images into `report_v2/images/`.
  - Writes `report_v2/summary.md` according to the structure above.

The aim is to make it possible to **ignore the old reporting stack entirely** when reading or publishing results: new readers should be able to rely on `report_v2/summary.*` + `report_v2/images/` as the canonical view of the experiment. 
