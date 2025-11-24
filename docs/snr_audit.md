# Spectral Diffusion Forward-Process Audit (Unified)

## 0. Research Question and Experimental Axis
- Core question: How does explicit control of SNR in the frequency domain affect stability and sample‑efficiency vs. standard spatial Gaussian noise?
- Implementation levers (live): spectral_operator modes (`none`, `radial`, `radial_squared`) and `snr_ratio`; baseline is unshaped Gaussian noise (`mode=none`, `snr_ratio=1`).
- Metrics: loss curves, grad_norm, `snr_theory`, `snr_emp`, `snr_rel`, `variance_sum`, images_per_second, qualitative samples.

## Current state (post-unification)
- Forward noise: `x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * spectral_operator(eps_raw, mode) * (1/snr_ratio)`, RMS(eps_shaped)=1 with per-sample centering. No per-batch clamping; schedule untrimmed.
- Scheduler: `build_diffusion` returns raw beta/logSNR schedules; no trimming/clamping.
- Spectral shaping: single entrypoint `src/spectral/operator.py`; all noise paths call it.
- SNR metrics: only `snr_theory = alpha_bar/(1-alpha_bar)`, `snr_emp = Var(signal)/Var(noise)`, `snr_rel = snr_emp/snr_theory`, plus `variance_sum` and noise channel std bounds.
- Loss: MSE/MAE with optional `log(snr_rel)` weighting; no adaptive weighting, overflow bridges, or penalties.
- Diagnostics/reporting: stability CSVs/report_v2 use the unified SNR triad + `variance_sum`, loss, grad_norm.
- Taguchi: factors reduced to `snr_ratio`, `spectral_operator_mode`, `sampler_type`, `sampling_steps`, `train_steps`, `image_resolution`.

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

## Current TODOs / gaps
- Quarantine or delete legacy configs/scripts still carrying removed knobs (`uniform_corruption`, `freq_equalized_noise`, `snr_weighting`, `snr_scale_min/max`, phase_*), and fence off archived Taguchi runs (L23/L27 legacy, `scripts/run_taguchi_v2_fixed.py`, archive/*).
- Leave adaptive/overflow utilities (`src/utils/adaptive_snr.py`, `src/core/overflow_handler.py`, `src/core/snr_scheduler.py`) clearly marked legacy; do not let them leak into live configs or tests (e.g., `tests/test_numerical_stability.py`, `tests/test_record_training_steps.py`).
- Phase-attention references in spectral UNet and sanity fixtures (`src/core/model_unet_spectral.py`, `tests/test_spectral_unet_model.py`, `tests/sanity/conftest.py`, README snippets) should be archived or labelled experimental.
- Reporting/doc stragglers: ensure README and archived docs no longer advertise spectral adapters/phase/overflow as active; keep main narrative focused on the SNR triad and minimal Taguchi factors.
- Archive scripts mentioning removed knobs (`scripts/archive/run_full_report*.sh`, other archive tools) should be fenced off or removed.
- Invariants: maintain tests for RMS(eps_shaped)=1, Var(signal)+Var(noise)=1 (shaping off), snr_rel≈1 (shaping off), monotone snr_theory; keep tolerances aligned with small-batch variation.

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
- Primary reporting paths now emit `snr_theory`, `snr_emp`, `snr_rel`, `variance_sum`, loss, grad_norm; ensure any remaining dashboards/scripts (archived notebooks, old Taguchi reports) drop `snr_schedule_*`, `snr_effective`, overflow metrics.
- Taguchi reports should only show factors: `snr_ratio`, `spectral_operator_mode`, `sampler_type`, `sampling_steps`, `train_steps`, `image_resolution`.
- Keep stability CSV/plots on the unified SNR triad + variance_sum; prune overflow/pressure plots from live reports.

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
Legacy details on schedule-driven metrics are archived in
`docs/legacy/snr_weighting_notes.md`.

### 4.4 Variance paths
Legacy variance-penalty behaviour is archived in
`docs/legacy/snr_weighting_notes.md`.

### 4.5 Loss path and weighting
Legacy adaptive weighting and spectral-pressure notes are archived in
`docs/legacy/snr_weighting_notes.md`.

### 4.6 Diagnostics
Legacy SNR logging fields are archived in
`docs/legacy/snr_weighting_notes.md`.

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

## 10. Current Capability Table

| Category         | Allowed in Current Pipeline                                                                                                    | Forbidden / Not Implemented                                                      |
|------------------|------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Forward Process** | unified DDPM coefficients, spectral operator S with centering+unit RMS, single k=1/snr_ratio                               | clamping, trimming, multi-stage scaling, phase-noise                             |
| **Metrics**         | snr_theory, snr_emp, snr_rel, variance_sum                                                                                | snr_schedule*, snr_effective, overflow metrics                                   |
| **Taguchi Factors** | snr_ratio, spectral_operator_mode, sampler_type, sampling_steps, train_steps, image_resolution                             | spectral_adapter_placement, spectral_loss_weighting, adaptive knobs              |
| **Diagnostics**     | grad_norm, variance_sum                                                                                                   | phase demos, spectral_pressure, overflow logs                                    |
| **Loss**            | MSE/MAE with optional log(snr_rel)                                                                                        | adaptive_snr, variance/spectral penalties                                        |

## 11. Consolidated TODO Checklist

- purge remaining legacy config keys
- archive/deprecate adaptive/overflow utilities
- remove phase-attention from live docs and tests
- enforce monotonic snr_theory test
- ensure all runners/scripts use unified SNR triad
- verify Taguchi mapping uses only six factors
- add optional grad_clip_norm to config reference
- remove or archive scripts mentioning deprecated knobs
- validate variance_sum and snr_rel invariants in CI
