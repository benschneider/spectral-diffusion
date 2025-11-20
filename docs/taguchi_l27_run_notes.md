# Taguchi L27 (32×32 full report) – what the factors actually do

Run root: `results/full_report_32x32_20251117_212900/taguchi`  
Design: `configs/taguchi/L27_extended.csv`  
Mapping: `factor_mapping.json` in this folder (and under each `runs/<run_id>`).

This L27 array maps the Taguchi columns **A, C, D, E, F, G, J, K, L** (we
permanently dropped B, H, and I because they were fully inactive) to nine
hyperparameters. The notes below are tailored to this specific full‑report run
and focus on two things:

- Which knobs really change the model, noise path, or sampler.
- Which knobs are currently metadata / placeholders, so you shouldn’t over‑interpret
  their Taguchi S/N scores.

---

## 1. High‑level impact summary

- **Training‑time knobs that actually change behaviour**
  - **C – spectral_loss_weighting** → changes spectral adapter weighting (`spectral.weighting`) when adapters are enabled.
  - **E – train_steps** → sets `training.num_batches` (effective training steps per run).
  - **F – spectral_noise_shaping_strength** → toggles FFT‑domain noise injection on/off via `diffusion.uniform_corruption`.
  - **G – spectral_adapter_placement** → decides whether TinyUNet is wrapped with FFT adapters at input/output.
  - **J – snr_ratio** → sets the target SNR in `add_uniform_frequency_noise`.
  - **K – image_resolution** → sets `data.height/width` and thus FFT grid size.

- **Sampling‑only knobs**
  - **D – sampler_type** → controls `sampling.sampler_type` (`ddim`, `dpm_solver++`, `masf`).
  - **L – sampling_steps** → controls `sampling.num_steps` (reverse steps in the sampler).

- **Knobs that are effectively no‑ops in this run**
  - **A – phase_attention_capacity**  
    Writes `model.enable_phase_attention` / `model.phase_heads`, but the base model here is `unet_tiny`, which ignores these fields (phase attention only lives in `unet_spectral`).

For Taguchi analysis this means:

- Expect **real main effects** from **C, E, F, G, J, K**.
- Treat **A** (until we run spectral backbones) as *design plumbing*, not meaningful signal.

---

## 2. Factor‑by‑factor details (A–L)

Each factor lists its levels, the intent, and what actually changes in code for
this full‑report run.

### A – phase_attention_capacity (off · tiny · full)

- **Intent**  
  Vary the capacity of the Phase Correction Module (PCM) that operates on FFT phase.

- **Code path**  
  `src/experiments/run_experiment.py::_apply_phase_capacity` maps levels to:
  - `off`  → `model.enable_phase_attention = False`, `model.phase_heads = 0`
  - `tiny` → `model.enable_phase_attention = True`,  `model.phase_heads = 1`
  - `full` → `model.enable_phase_attention = True`,  `model.phase_heads = 4`

- **Reality in this run**  
  The configs under `runs/<run_id>/config.yaml` use `model.type: unet_tiny`.  
  Phase attention is implemented in `model_unet_spectral` (`enable_phase_attention`,
  `phase_heads`, `PhaseCorrectionModule`), not in `unet_tiny`. As a result:
  - The PCM is never instantiated in these runs.
  - A behaves as a **metadata‑only factor**: it appears in configs and reports,
    but it does not change the forward pass or gradients.

### C – spectral_loss_weighting (none · radial_highfreq · aggressive_highfreq)

- **Intent**  
  Control how spectral adapters emphasize different frequency bands when they are used.

- **Code path**
  - Taguchi maps levels to `spectral.weighting` and (for aggressive) sets
    `spectral.bandpass_inner` / `spectral.bandpass_outer`.
  - `src/spectral/fft_utils.configure_spectral_params` pulls these into the model’s
    `spectral_cfg`.
  - `TinyUNet` uses `spectral_cfg.weighting ∈ {none, radial, bandpass}` to build
    `SpectralAdapter` modules that apply a frequency‑domain weight map to features.

- **Reality in this run**
  - This factor shapes the **model’s adapters**, not the loss. The loss still reads
    `loss.spectral_weighting` from the base config, which is held fixed here.
  - When `spectral.enabled` is `false` (G = none), C is effectively inert.  
    When adapters are enabled, C changes which frequencies get more attention in
    the feature pipeline:
    - `none` → adapters are effectively identity (no extra spectral emphasis).
    - `radial_highfreq` → smooth radial weighting with more energy away from DC.
    - `aggressive_highfreq` → band‑pass mask between `bandpass_inner` and `bandpass_outer`.

### D – sampler_type (ddim · dpm_solver_pp · spectral_guided)

- **Intent**  
  Choose the reverse‑time sampler used for generating images *after training*.

- **Code path**
  - Taguchi maps labels to:
    - `ddim`          → `sampling.sampler_type = "ddim"`
    - `dpm_solver_pp` → `sampling.sampler_type = "dpm_solver++"`
    - `spectral_guided` → `sampling.sampler_type = "masf"`
  - `TrainingPipeline.generate_samples` hands this to `src/training/sampling.build_sampler`,
    which instantiates `DDIMSampler`, `DPMSolverPlusPlusSampler`, or `MASFSampler`.

- **Reality in this run**
  - The *training* loop (`TrainingPipeline.run`) never touches `sampling.sampler_type`.
  - D only affects sampling: sample grids and evaluation metrics computed from generated
    images (FID, LPIPS, etc.), not training losses or stability metrics.

### E – train_steps (50 · 100 · 200)

- **Intent**  
  Vary the training budget per run.

- **Code path**
  - Levels map directly to `training.num_batches`.
  - The dataloader uses `num_batches` to size the synthetic dataset (for synthetic
    families) or, for CIFAR, to determine how many batches are drawn per epoch.

- **Reality in this run**
  - E changes both **wall‑clock time** and **amount of optimisation**.
  - Because the Taguchi report focuses on `loss_drop_per_second`, you should read
    E’s effect as “sample efficiency per second” rather than just “final loss”.
  - Interacts with J/F/G: unstable combinations waste steps on noisy gradients
    and show weaker loss_drop per second even with higher step counts.

### F – spectral_noise_shaping_strength (off · mild_equalize · strong_equalize)

- **Intent**  
  Control how aggressively the forward noise is “spectrally equalised”.

- **Code path**
  - Taguchi maps:
    - `off`           → `diffusion.uniform_corruption = False`, `spectral.freq_equalized_noise = False`
    - `mild_equalize` → `diffusion.uniform_corruption = True`,  `spectral.freq_equalized_noise = False`
    - `strong_equalize` → `diffusion.uniform_corruption = True`, `spectral.freq_equalized_noise = True`
  - `NoisePreparer.from_config` consumes `diffusion.uniform_corruption` and
    `diffusion.uniform_corruption_scale` and forwards them to
    `add_uniform_frequency_noise`.

- **Reality in this run**
  - `add_uniform_frequency_noise` uses a reciprocal‑radius mask when
    `uniform_corruption=True`. When `spectral.freq_equalized_noise=True`
    (the “strong” level) the mask is **squared before normalisation**, pushing
    substantially more energy into high-frequency bands while keeping the
    overall RMS comparable.
  - This makes F a **three-level** knob again:
    - `off` → spatial Gaussian baseline.
    - `mild_equalize` → FFT-domain noise with the base radial mask.
    - `strong_equalize` → FFT-domain noise with the squared mask for extra
      high-frequency emphasis.

### G – spectral_adapter_placement (none · input_only · input_and_output)

- **Intent**  
  Decide where TinyUNet is wrapped with FFT adapters that modulate features in
  the frequency domain.

- **Code path**
  - Taguchi maps levels to:
    - `none`            → `spectral.apply_to = []`,        `spectral.enabled = False`
    - `input_only`      → `spectral.apply_to = ["input"]`, `spectral.enabled = True`
    - `input_and_output` → `spectral.apply_to = ["input","output"]`, `spectral.enabled = True`
  - `configure_spectral_params` passes `apply_to` and `enabled` into `TinyUNet`.
  - `TinyUNet` conditionally constructs `SpectralAdapter` modules for the input and/or
    output, applying the weight map controlled by C.

- **Reality in this run**
  - G is a genuine **architectural toggle**: more placements → more spectral context
    and more compute.
  - C (spectral_loss_weighting) only matters when G enables adapters; otherwise
    `spectral.weighting` is ignored.

### J – snr_ratio (0.8 · 1.0 · 1.4)

- **Intent**  
  Set a target signal‑to‑noise ratio for the forward diffusion when FFT corruption
  is enabled, to balance sharpness vs stability.

- **Code path**
  - Levels are written into `diffusion.snr_ratio` and `spectral.snr_ratio`.
  - `NoisePreparer` forwards `snr_ratio` into `add_uniform_frequency_noise`.
  - In the FFT branch, after masking the noise and scaling by `sqrt_one_minus_alpha_t`,
    the function measures per‑sample RMS for signal and noise and rescales the
    noise so that:
    - `RMS(signal) / RMS(noise) ≈ snr_ratio`
  - Diagnostics store `snr_ratio`, `snr_measured`, and `snr_scale_factor` in
    both the per‑step noise stats and the aggregate metrics.

- **Reality in this run**
  - J is a **dominant stability knob**:
    - Lower values (0.8) tend to be more stable but may converge more slowly.
    - Higher values (1.4) tend to sharpen high frequencies but push overflow and
      spectral pressure.
  - Run IDs include the level (e.g. `_snr0p8`, `_snr1p0`, `_snr1p4`), making it
    easy to slice metrics and diagnostics by J.

### K – image_resolution (32 · 64 · 128)

- **Intent**  
  Stress the noise path and adapters at higher spatial resolutions.

- **Code path**
  - Levels map to `data.height` and `data.width`.
  - For CIFAR, the dataloader resizes each image to this target size before
    feeding it to the model.
  - All subsequent FFTs and spectral adapters operate on this grid.

- **Reality in this run**
  - Increasing K increases both **high‑frequency bandwidth** and **compute**.
  - Overflow and `fft_high_mean` typically rise with K, especially for aggressive
    J/F/G combinations.
  - At higher resolutions the same snr_ratio can feel “tighter” because more
    high‑frequency modes share the energy budget.

### L – sampling_steps (30 · 50 · 100)

- **Intent**  
  Trade off sampler speed vs reconstruction quality.

- **Code path**
  - Levels set `sampling.num_steps`, which `TrainingPipeline.generate_samples`
    passes to the chosen sampler (`DDIMSampler`, `DPMSolverPlusPlusSampler`,
    or `MASFSampler`).

- **Reality in this run**
  - L is **sampling‑only**: it does not influence the training loop.
  - Its impact shows up in:
    - Sample sharpness / coherence for a fixed sampler type.
    - Evaluation metrics computed from generated samples, not in
      `loss_drop_per_second` or training stability metrics.

---

## 3. How to read the L27 outputs

Under `results/full_report_32x32_20251117_212900/taguchi` the key artefacts are:

- `L27_extended_summary.csv` and `taguchi_report.csv`  
  Per‑run training metrics plus Taguchi S/N scores (default metric:
  `loss_drop_per_second`).
- `run_<n>_metrics.json`  
  Full training traces, including overflow counts, SNR stats, and FFT feedback.
- `runs/<run_id>/diagnostics/*`  
  Noise statistics, stability CSVs, and plots (including `snr_measured` and
  overflow history).
- `runs/<run_id>/images/*`  
  Sample grids driven by D (sampler_type) and L (sampling_steps).

When interpreting factor main effects:

- Focus on **C, E, F (off vs on), G, J, K** for training‑time behaviour; these
  actually change the forward pass.
- Expect **flat or noisy estimates** for **A, B, H, I** and for the F
  mild/strong split, because these knobs are not fully wired into the training
  or noise code yet.
- Remember that **D** and **L** only affect post‑training sampling, so Taguchi
  scores computed purely from training metrics will not capture their qualitative
  impact on samples.

---

## 4. Grounded hypotheses to check

These hypotheses match the current implementation (not the aspirational factor
descriptions):

- Lower **J (0.8)** with **F = off** should yield the most stable training
  (lowest overflow tail), but may show smaller `loss_drop_per_second` than
  J = 1.0 when adapters are enabled.
- Turning **F on** (mild/strong) at moderate **J (1.0)** and non‑trivial
  **G** (input_only or input_and_output) should:
  - Raise `fft_high_mean` / spectral pressure.
  - Increase `snr_measured`, but also modestly increase overflow statistics.
- Pushing **J to 1.4** without strong spectral support (G = none, C = none)
  should noticeably increase `overflow_count` and the variance of
  `snr_measured`, especially at higher **K**.
- Increasing **K** from 32 → 64 → 128 amplifies all of the above:
  stable configurations remain usable, while marginal ones tip into more frequent
  overflows and slower effective convergence.

Use this note as the ground truth for how the Taguchi factors map to code in
this L27 full‑report run, so that the analysis focuses on the knobs that truly
matter and treats half‑wired controls as design noise rather than signal.
