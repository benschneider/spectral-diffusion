# Diffusion vs Spectral Forward Process Audit (Updated)

## 0. Research Question and Experimental Axis
- Core question: "How does explicit control of signal-to-noise ratio (SNR) in the frequency domain affect the stability and sample-efficiency of diffusion model training compared to standard spatial Gaussian noise?"
- Implementation levers:
  - Baseline: spatial Gaussian noise following DDPM schedule (no shaping).
  - Spectral: FFT-shaped noise with controls: `uniform_corruption`, `freq_equalized_noise`, `corruption_mode`, `uniform_corruption_scale`, `snr_ratio`, `snr_scale_min/max`, phase corruption.
  - Metrics needed: stability (loss curves, grad_norm, SNR_theory vs SNR_emp vs SNR_rel), sample efficiency (loss_drop_per_second, images_per_second), qualitative samples at matched loss.
- All invariants and refactors should make this comparison clean and interpretable.

## Fresh scan – implementation notes (post-unification)
- Forward noise: `x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps_shaped * (1/snr_ratio)`, where `eps_shaped = spectral_operator(eps_raw, mode)` and RMS(eps_shaped)=1. No per-batch clamping of sqrt_alpha_t or sigma; no trimming of the schedule.
- Scheduler: `build_diffusion` returns raw `alpha_bar_t` from beta/logSNR schedules without clamping/trimming.
- Spectral branch: the only shaping entry point is `src/spectral/operator.py` (modes: none, radial, radial_squared). All noise paths call it.
- SNR metrics: unified to `snr_theory = alpha_bar/(1-alpha_bar)`, `snr_emp = Var(signal)/Var(noise)`, `snr_rel = snr_emp/snr_theory`. Diagnostics also log `variance_sum = Var(signal)+Var(noise)` and channel-wise noise std bounds.
- Loss weighting: minimal MSE/MAE with optional `log(snr_rel)` weighting; no adaptive weighting, overflow bridges, or spectral penalties.
- Diagnostics/reporting: stability CSVs and report_v2 summarize only snr_theory/sn_emp/sn_rel, variance_sum, loss, and grad_norm. Taguchi factors reduced to real knobs: snr_ratio, spectral_operator_mode, sampler_type, sampling_steps, train_steps, image_resolution.

## 1. BASELINE: First-Principles Diffusion Math (DDPM)
| Concept | Definition / Notes |
| --- | --- |
| `x_0` | Clean data sample drawn from dataset distribution; assumed zero-mean or centered so Var(`x_0`) matches dataset variance. |
| `eps` | I.I.D. Gaussian noise `eps ~ N(0, I)` that is independent of `x_0`; preserves unit variance in every channel. |
| Forward noise | `x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps`, so signal amplitude follows `sqrt(alpha_bar_t)` and injected noise std is `sqrt(1-alpha_bar_t)`. |
| `SNR_theory(t)` | `alpha_bar_t / (1 - alpha_bar_t)` (ratio of signal variance to noise variance). |
| Variance identities | `signal_var = alpha_bar_t`, `noise_var = 1 - alpha_bar_t`, `signal_var + noise_var = 1`. |
| `alpha_bar_t` schedule | Monotone decreasing cumulative product of alphas; can be defined by beta schedule or log-SNR cosine trajectory so that early steps are almost clean and late steps almost pure noise. |
| Gradient expectations | DDPM objective with `eps`-prediction yields constant-variance gradients across timesteps; weighting by SNR or log-SNR only for variance stabilisation, not to change target distribution. |
| SNR decay | Theoretical SNR is strictly decreasing with `t`; no clipping or re-scaling required for convergence proofs. |

## 2. CURRENT IMPLEMENTATION SCAN (FROM REPO)

### Forward process (x_t construction)
- Baseline: x_t = sqrt_alpha_t*x0 + sqrt_one_minus_alpha_t*noise, but sqrt_alpha_t/sigma are clamped (ALPHA_MIN/ALPHA_MAX/SIGMA_MIN) before use (src/training/noise.py:120-130; src/training/steps.py:97-102).
- Spectral: same signal term; noise shaped via FFT mask (reciprocal radius, optional squared), normalized, scaled by strength, snr_ratio, snr_scale_min/max; optional phase corruption on x0 FFT; adaptive rescale toward target_corr; then multiplied by clamped sqrt_one_minus_alpha_t (src/spectral/fft_adapter.py:125-290).
- Schedule generation trims early steps where sigma < 0.03, altering alpha_bar trajectory (src/training/scheduler.py:86-104).

### alpha/sigma schedule handling
- Uses make_beta_schedule/logsnr_cosine; clamps sigma in schedule; trims leading steps below MIN_SIGMA_THRESHOLD=0.03 (src/training/scheduler.py:12-120).
- Additional per-batch clamps in NoisePreparer/TrainingStepExecutor.

### Spectral shaping operator
- Implemented inline in add_uniform_frequency_noise: FFT mask, optional squared mask, normalization via _normalize_fft_noise, optional phase perturbation; no single S abstraction (src/spectral/fft_adapter.py:178-245).

### Noise scaling and snr_ratio
- Sequential scales: strength multiplier, snr_ratio via base_snr/target ratio, snr_scale_min/max clamp, adaptive_rescale toward target_corr, plus sigma clamp (src/spectral/fft_adapter.py:178-247).

### Variance invariants
- `_normalize_fft_noise` enforces unit RMS pre-strength; subsequent scaling (strength, snr_ratio, snr_scale_min/max, adaptive) and sigma clamp break Var(signal)+Var(noise)=1 (src/spectral/fft_adapter.py:203-247; src/training/steps.py variance_penalty).

### Empirical/theoretical SNR handling
- Theoretical: compute_snr_stats -> snr_raw/clamped/weight/log_snr (src/core/snr_scheduler.py; used in src/training/steps.py and src/core/losses.py).
- Empirical: snr_effective from measure_batch_snr (src/core/snr_scheduler.py) and from fft_adapter stats; snr_measured/base/scale_factor in fft_adapter; snr_schedule trends in debug script. Weighting uses schedule SNR, not empirical.

### Loss weighting
- DiffusionLoss: adaptive weighting via AdaptiveSNRWeight (tanh log-SNR), optional spectral residual weighting, overflow bridge; weighting uses snr_for_weight from schedule (src/core/losses.py, src/core/adaptive_weight.py).
- TrainingStepExecutor: may compute fallback weights via compute_snr_weight; adds variance_penalty and spectral_pressure to loss (src/training/steps.py:240-310).
- Debug recorder mirrors penalties and weighting options (scripts/debug/record_training_steps.py).

### Spectral adapter and penalties
- SpectralAdapter reweights residuals in loss; renormalizes outputs to avoid std drift; spectral_pressure penalty added in TrainingStepExecutor and debug script (src/spectral/adapter.py; src/training/steps.py:260-305; scripts/debug/record_training_steps.py:783-793).

### Gradient paths / grad_norm
- Gradients logged only (src/training/diagnostics.py); no clipping in TrainingStepExecutor. Debug script supports clip_grad_norm/ratio and shock handler.

### Taguchi factor structure
- Factor registry includes snr_ratio, spectral_noise_shaping_strength, snr_weighting_mode, spectral_adapter_placement, etc.; report_v2 primary factors use these legacy knobs (configs/taguchi/factor_registry*.yaml; scripts/generate_report_v2.py PRIMARY_FACTORS_BY_PROFILE).

### Diagnostics and logging
- Stability CSV logs snr_schedule_mean/max/raw_max, snr_effective, overflow stats, variance_ratio/penalty, spectral_pressure (src/training/diagnostics.py).
- fft_adapter stats log snr_effective/snr_measured/snr_base/snr_ratio_target/scale_factor and noisy mean/std (src/spectral/fft_adapter.py).
- Reporting uses snr_schedule_mean and snr_effective in metadata (scripts/generate_report_v2.py).

## Verified invariant violations (per code)
- Forward invariant (x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*eps_shaped, eps_shaped RMS=1): Violated by per-batch clamping (src/training/noise.py:120-130; src/training/steps.py:97-102) and schedule trimming (src/training/scheduler.py:86-104); spectral branch adds non-Gaussian/phase-correlated noise (src/spectral/fft_adapter.py:199-203).
- SNR coherence (snr_theory, snr_emp, snr_rel computed once): Violated—multiple SNRs (snr_schedule, snr_effective, snr_measured, snr_base, snr_scale_factor) computed in different places; weighting uses schedule SNR; empirical SNR not unified (src/training/steps.py coeff_stats, src/core/losses.py, src/spectral/fft_adapter.py stats, scripts/debug/record_training_steps.py).
- Variance preservation (Var(signal)+Var(noise)=1 when shaping off): Violated—clamping and sequential scaling alter variance; invariant only penalized via variance_penalty, not enforced (src/training/steps.py:269-305; src/spectral/fft_adapter.py:203-247).
- Single scaling path (eps_raw -> eps_shaped RMS=1 -> *sqrt(1-alpha)*k): Violated—strength, snr_scale_min/max, adaptive_rescale, phase mode create multiple scales (src/spectral/fft_adapter.py:178-247).
- Gaussianity (noise independent, Gaussian when shaping=none): Violated by phase mode tied to x0 and by mask-colored noise; shaping=none still uses clamped sigma (src/spectral/fft_adapter.py:178-210, 199-203; src/training/noise.py clamps).

## 2. “80% WINS” (tagged, research-aligned)
1) Remove alpha/sigma clamps and schedule trimming [FWD][VAR][SNR]  
   - Files: src/training/scheduler.py, src/training/noise.py, src/training/steps.py.  
   - Rationale: Restore DDPM coefficients to cleanly compare spectral vs spatial SNR.

2) Introduce a single unit-RMS spectral_operator and route all shaping through it [SPEC][VAR][FWD]  
   - Files: new src/spectral/operator.py; replace usage in src/spectral/fft_adapter.py, src/training/noise.py.  
   - Rationale: Isolate frequency shaping so SNR control vs baseline is measurable.

3) Collapse noise scaling to k = 1/snr_ratio (optional simple clamp) [SNR][FWD][VAR]  
   - Files: src/spectral/fft_adapter.py, src/training/noise.py.  
   - Rationale: Removes layered scaling that obscures SNR effects.

4) Unify SNR metrics to {snr_theory, snr_emp, snr_rel} and log them once [SNR][DIAG]  
   - Files: src/training/steps.py, src/training/diagnostics.py, scripts/debug/record_training_steps.py, scripts/generate_report_v2.py.  
   - Rationale: Enables direct measurement of spectral SNR control vs baseline.

5) Simplify loss weighting to uniform or log(SNR_rel) only; remove adaptive/overflow bridge [LOSS][SNR]  
   - Files: src/core/losses.py, src/core/adaptive_weight.py, src/training/steps.py.  
   - Rationale: Aligns gradients with empirical SNR to test stability impact.

6) Remove variance and spectral penalties from loss path [LOSS][VAR][SPEC]  
   - Files: src/training/steps.py, scripts/debug/record_training_steps.py.  
   - Rationale: Avoids confounding the SNR comparison with extra terms.

7) Add invariant checks/tests (Var sum, SNR_rel with shaping off, S RMS=1) [FWD][VAR][SNR][SPEC][DIAG]  
   - Files: tests/*, optional runtime toggles in noise prep.  
   - Rationale: Ensures experiments measure intended SNR manipulations.

8) Prune Taguchi factors to active SNR/spectral knobs [TAG][SPEC][SNR]  
   - Files: configs/taguchi/*, scripts/run_taguchi_suite.py, scripts/generate_report_v2.py factor mappings.  
   - Rationale: Keeps design-of-experiments focused on spectral vs spatial SNR control.

9) Streamline diagnostics/logging/reporting to unified metrics [DIAG][SNR]  
   - Files: src/training/diagnostics.py, scripts/generate_report_v2.py, visualization/figures.  
   - Rationale: Surfaces stability/sample-efficiency vs SNR_rel directly.

10) Optional: add grad_clip_norm config and warnings [LOSS][DIAG]  
    - Files: src/training/steps.py, config reference.  
    - Rationale: Stabilizes training for fair comparison of noise regimes.

## 3. PROJECT TODO LIST (tagged, ordered)
[A] Core correctness fixes  
- Remove alpha/sigma clamps and schedule trimming [FWD][VAR][SNR]  
  Files: src/training/scheduler.py; src/training/noise.py; src/training/steps.py.  
  Acceptance: alpha_t/sigma_t match schedule; no per-batch clamp; Var(sum) uses unclamped coeffs.  
  Breaking: No.  
  Research link: Ensures baseline SNR matches theory for clean spectral vs spatial comparison.

[B] SNR unification + scaling cleanup  
- Add spectral_operator (unit RMS) and replace inline shaping [SPEC][VAR][FWD]  
  Files: src/spectral/operator.py; replace use in src/spectral/fft_adapter.py, src/training/noise.py.  
  Acceptance: RMS(eps_shaped)=1±1e-4; single shaping entrypoint.  
  Breaking: Internal.  
  Research link: Isolates spectral shaping effect on SNR.
- Single k = 1/snr_ratio scaling [SNR][VAR][FWD]  
  Files: src/spectral/fft_adapter.py; src/training/noise.py.  
  Acceptance: Noise std = sqrt(1-alpha)/snr_ratio ±1e-4; no other scales.  
  Breaking: Removes strength/snr_scale_* configs.  
  Research link: Makes SNR control explicit for frequency vs spatial noise.
- Unified SNR logs {snr_theory, snr_emp, snr_rel, variance_sum} [SNR][DIAG]  
  Files: src/training/steps.py; src/training/diagnostics.py; scripts/debug/record_training_steps.py; scripts/generate_report_v2.py.  
  Acceptance: Legacy SNR fields removed; new fields present.  
  Breaking: Log schema.  
  Research link: Directly measures empirical SNR impact of spectral shaping.

[C] Noise pipeline simplification  
- Remove layered scaling (strength, snr_scale_min/max, adaptive_rescale, phase noise) [FWD][SPEC][VAR]  
  Files: src/spectral/fft_adapter.py; configs.  
  Acceptance: Single scale path; no phase-as-noise; shaping optional.  
  Breaking: Config keys removed.  
  Research link: Ensures only intended SNR lever differs between branches.

[D] Loss and weighting consistency  
- Replace adaptive weighting with uniform/log(SNR_rel); remove overflow bridge [LOSS][SNR]  
  Files: src/core/losses.py; src/core/adaptive_weight.py; src/training/steps.py.  
  Acceptance: Weighting mode selectable; no adaptive state.  
  Breaking: Removes adaptive_snr API fields.  
  Research link: Ties weighting to empirical SNR to test stability changes.
- Drop variance/spectral penalties from training loss [LOSS][VAR][SPEC]  
  Files: src/training/steps.py; scripts/debug/record_training_steps.py.  
  Acceptance: Loss = weighted MSE/MAE only.  
  Breaking: Removes penalty-driven terms.  
  Research link: Avoids confounding SNR comparison with extra objectives.

[E] Diagnostics, tests, invariants  
- Add invariant tests (Var sum ≈1; SNR_rel≈1 with shaping off; S RMS=1; monotone SNR schedule) [DIAG][VAR][SNR][SPEC][FWD]  
  Files: tests/*.  
  Acceptance: Tolerances Var 1e-3, SNR_rel 10%.  
  Breaking: No.  
  Research link: Validates that observed effects stem from intended SNR controls.
- Runtime invariant warnings toggle [DIAG][SNR][VAR]  
  Files: src/training/noise.py config hook.  
  Acceptance: Warn if SNR_rel outside [0.3,3].  
  Breaking: No.  
  Research link: Flag runs where spectral SNR deviates from design.

[F] Taguchi and reporting cleanup  
- Prune Taguchi factors to active SNR/spectral knobs [TAG][SNR][SPEC]  
  Files: configs/taguchi/*; scripts/run_taguchi_suite.py; factor_registry*.  
  Acceptance: Factor mapping matches live knobs; stale factors removed.  
  Breaking: Yes (factor set).  
  Research link: Keeps DoE focused on spectral vs spatial SNR.
- Update report generation to new metrics [DIAG][SNR]  
  Files: scripts/generate_report_v2.py; visualization/figures.py.  
  Acceptance: Uses snr_theory/emp/rel; legacy snr_schedule/effective dropped.  
  Breaking: Report schema.  
  Research link: Reports directly answer SNR control question.

[G] Optional stability guard  
- Add grad_clip_norm config/warnings [LOSS][DIAG]  
  Files: src/training/steps.py; config reference.  
  Acceptance: Clip applied when set; norm logs.  
  Breaking: No.  
  Research link: Stabilizes runs to compare noise regimes fairly.

## Reporting gaps
- Current reports/logs surface snr_schedule_mean and snr_effective, not snr_theory/snr_emp/snr_rel.
- Stability CSV lacks snr_emp/rel and variance_sum.
- Taguchi profiles use legacy factors; no exposure of spectral_operator_mode or unified SNR metrics.
- Needed metrics for the research question: snr_theory, snr_emp, snr_rel, variance_sum deviation, loss curves, grad_norm, loss_drop_per_second, images_per_second.

## 4. MINIMAL FORWARD PROCESS SPEC
- Equation: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps_shaped.
- Invariants: RMS(eps_shaped)=1; Var(signal)+Var(noise)=1; eps_shaped independent of x_0.
- Allowed modifications: Spectral operator S that preserves RMS=1; single scalar k (e.g., 1/snr_ratio) applied uniformly after shaping.
- Forbidden modifications: Per-batch clamping of alpha/sigma; multi-stage or data-dependent rescaling; deterministic “noise” tied to x_0; extra penalties altering forward variance.
- Expected metrics per batch: snr_theory, snr_emp, snr_rel, variance_sum deviation, loss, grad_norm (optional dB views).

# Spectral Diffusion Forward-Process Audit (Unified + Restructured)

## 1. Research Axes
This project examines several scientific questions:

1. Correctness of the forward diffusion process and preservation of DDPM invariants.
2. Effects of explicit SNR control in the frequency domain on stability and sample‑efficiency.
3. Coherence between theoretical and empirical SNR across timesteps.
4. Impact of strict variance‑preservation vs. multi-stage variance drift.
5. Role of Gaussianity, frequency masks, and phase perturbations in effective noise.
6. Effectiveness of unified SNR‑based weighting vs. schedule‑based or adaptive weighting.
7. Diagnostics‑driven training stability and invariant‑based failure detection.

These provide the overarching framework; the SNR-in-frequency question is one axis among several.

---

## 2. Research Question (Primary Axis)
**Primary question:**  
How does explicit control of signal‑to‑noise ratio (SNR) in the frequency domain affect the stability and sample‑efficiency of diffusion model training compared to standard spatial Gaussian noise?

Metrics:  
- loss curves, loss_drop_per_second  
- grad_norm trajectories  
- snr_theory, snr_emp, snr_rel  
- variance_sum deviation  
- qualitative samples at matched loss  
- images_per_second

---

## 3. First‑Principles Diffusion Math (DDPM)
- Forward process:  
  `x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*eps`, with `eps ~ N(0, I)` independent of `x_0`.
- Variance identity:  
  `Var(signal) + Var(noise) = 1`.
- Theoretical SNR:  
  `SNR_theory = alpha_bar_t / (1 - alpha_bar_t)`.
- Gaussianity, independence, and monotonic SNR are core invariants.

---

## 4. Current Implementation Scan

### 4.1 Forward process (baseline)
- Clamps `sqrt_alpha_t` and `sqrt_one_minus_alpha_t` inside NoisePreparer.
- Scheduler trims early steps where `sigma < 0.03`.
- Result: forward coefficients no longer match DDPM schedule.

### 4.2 Forward process (spectral)
- Applies frequency mask (reciprocal radius or squared) → normalizes via Parseval → inverse FFT.
- Optional phase corruption alters `x0`’s FFT phases (non-Gaussian, data‑dependent).
- Sequential scaling: strength → snr_ratio → snr_scale_min/max → adaptive_rescale.
- Multiple variance modifications break the DDPM invariant `Var(signal)+Var(noise)=1`.

### 4.3 SNR metrics (fragmented)
- `snr_schedule` (from clamped coefficients; used for weighting).
- `snr_effective` / `snr_measured` (empirical RMS ratio; not used for weighting).
- `snr_ratio` knob (RMS amplitude scaling).
- `snr_scale_factor`, `snr_base`, `snr_ratio_target` (diagnostic).
- Missing unified structure: no `snr_emp` or `snr_rel`.

### 4.4 Variance paths
- `_normalize_fft_noise` guarantees unit RMS before subsequent scaling.
- Post-scaling variance no longer tied to `(1 - alpha_bar_t)`.
- Variance_penalty attempts to compensate but does not enforce invariants.

### 4.5 Loss path and weighting
- AdaptiveSNRWeight uses log-SNR derived from schedule, not the actual injected noise.
- Additional FFT spectral‑pressure penalty and variance penalty modify gradients.

### 4.6 Diagnostics
- Logs snr_schedule_mean, snr_effective, overflow, variance_ratio, spectral_pressure.
- No unified SNR triad.

---

## 5. Comparison Table
(kept concise)

- Gaussianity: violated by spectral masks + phase corruption.  
- Variance preservation: violated by clamps + multi-stage scaling.  
- SNR coherence: schedule vs empirical diverge significantly.  
- Scaling: sequential operations create inconsistent noise budgets.  
- Forward correctness: schedule trimming removes highest-SNR steps.  
- Loss-weighting: mismatched SNR sources.  
- Diagnostics: no unified definition of SNR.

---

## 6. Unified Correct Form (Target Spec)

### 6.1 Forward process
```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps_shaped
```
Where:
- `eps_shaped` has RMS=1 before scaling.
- Frequency shaping allowed only via one spectral operator `S`.
- `k = 1/snr_ratio` is the only scaling factor applied after shaping.
- No per‑batch clamping; schedule defines all coefficients.

### 6.2 Invariants
- RMS(eps_shaped) = 1  
- Var(signal) + Var(noise) = 1  
- snr_emp ≈ snr_theory when shaping is off  
- Monotonic alpha_bar_t and SNR_theory  
- eps_shaped independent of x_0 unless phase mode explicitly documented

### 6.3 Unified SNR definitions
- `snr_theory = alpha_bar_t / (1 - alpha_bar_t)`  
- `snr_emp = (signal_rms^2)/(noise_rms^2)` measured after shaping  
- `snr_rel = snr_emp / snr_theory`  

### 6.4 Diagnostics
Log exactly once per batch:
- snr_theory, snr_emp, snr_rel  
- variance_sum  
- noise_channel_std_min/max  
- grad_norm  
- (optional) snr_dB = 10*log10(snr_emp)

---

## 7. 80% Wins (Refactor Targets)
1. Remove alpha/sigma clamps and schedule trimming.  
2. Introduce `spectral_operator` with unit RMS; replace inline shaping.  
3. Collapse scaling to single factor `k = 1/snr_ratio`.  
4. Unify SNR metrics: snr_theory, snr_emp, snr_rel.  
5. Simplify loss weighting: uniform or log(SNR_rel).  
6. Remove variance and spectral penalties from training loss.  
7. Add forward invariants as tests.  
8. Prune Taguchi factors to active knobs.  
9. Streamline logging and reporting to unified SNR triad.  
10. Optional: add grad_clip_norm.

---

## 8. Project TODO List (Ordered)
[A] Forward correctness  
- Remove clamping and trimming.  
- Use schedule output coefficients directly.

[B] SNR unification + scaling cleanup  
- Insert spectral_operator.  
- Replace multi-stage scaling with single `k`.  
- Implement unified SNR logs + variance_sum.

[C] Noise pipeline simplification  
- Remove strength, snr_scale_min/max, adaptive_rescale, phase-ing for baseline.  
- Make shaping optional.

[D] Loss + weighting  
- Use uniform/log(SNR_rel).  
- Remove penalty terms.

[E] Tests + invariants  
- Add Var-sum, RMS-unit, monotonic-SNR tests.

[F] Taguchi + reporting cleanup  
- Remove legacy knobs and update figures/report.

[G] Optional stability  
- Add grad_clip_norm.

---

## 9. Minimal Forward Spec Summary
- `x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps_shaped`.  
- eps_shaped = S(eps_raw) with RMS=1.  
- noise = sqrt(1-alpha)*eps_shaped*(1/snr_ratio).  
- Invariants enforced via tests and runtime warnings.
