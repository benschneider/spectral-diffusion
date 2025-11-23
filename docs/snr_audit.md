# Diffusion vs Spectral Forward Process Audit

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

### 2.1 Forward process in use
- `NoisePreparer` clamps `sqrt_alpha_t` to `[0.01, 0.999]` and `sqrt_one_minus_alpha_t` to `>=1e-4`, before calling `add_uniform_frequency_noise` (`src/training/noise.py:12-166`). This modifies the DDPM schedule whenever the cosine/log-SNR schedule would drive either term outside that window.
- `build_diffusion` trims early timesteps whose `sigma < 0.03` (`src/training/scheduler.py:12-131`), removing the highest-SNR steps outright.
- Baseline branch (`uniform_corruption=False`) still computes `x_t = sqrt_alpha_t*x0 + sqrt_one_minus_alpha_t*noise`, but uses the clamped coefficients and optionally logs stats (`src/spectral/fft_adapter.py:156-177`).
- Spectral branch (`uniform_corruption=True`) runs FFTs, multiplies by a reciprocal-radius mask, optionally squares it for `"freq_equalized"` mode, normalises via Parseval, and inverse FFTs back (`src/spectral/fft_adapter.py:178-210`). This reshapes the noise spectrum and is not Gaussian unless the mask is flat.
- A `phase` mode perturbs the clean signal’s FFT phase (`src/spectral/fft_adapter.py:199-203`), creating deterministic, non-Gaussian “noise” tied to `x0`.
- After colouring, an RMS-based `snr_scale_tensor` rescales the noise to match `snr_ratio` (with optional clamp `snr_scale_min/max`) and an adaptive rescale reduces noise if the correlation to `x0` drops below `target_corr` (`src/spectral/fft_adapter.py:212-247`).
- Stats include FFT energy, RMS ratios, and noise-channel std spread, so users can diagnose the injected distribution (`src/spectral/fft_adapter.py:233-287`).
- Additional renormalisations: `_normalize_fft_noise` enforces unit-RMS noise after iFFT (`src/spectral/fft_adapter.py:205-209`), and `SpectralAdapter` in the loss re-normalises residual FFT weights to clip std drift between 1/3 and 3 (`src/spectral/adapter.py:1-115`).

### 2.2 All forms of SNR
| Name | Location & Formula | Interpretation / Usage |
| --- | --- | --- |
| `snr_schedule` | Calculated as `(sqrt_alpha_t**2)/(sqrt_one_minus_alpha_t**2)` with clamp at `SNR_CLIP` (`src/training/steps.py:217-235`, `scripts/debug/record_training_steps.py:867-885`). | Theoretical schedule derived from clamped coefficients; units are power ratio. Used for loss weighting, overflow logic, and logging even though the actual forward process rescales noise later. |
| `snr_effective` | Measured via `_per_sample_rms(signal)/_per_sample_rms(noise)` inside the FFT adapter and again via `measure_batch_snr` when logging (`src/spectral/fft_adapter.py:233-279`, `src/core/snr_scheduler.py:33-53`). | Empirical RMS amplitude ratio after spectral shaping. Logged for diagnostics; not fed back into weighting. |
| `snr_ratio` | Config knob mirrored into both diffusion and spectral configs (`src/cli/train.py:213-218`, `src/experiments/run_experiment.py:168-177`). Applied as `base_snr / snr_ratio` scaling on the noise component (`src/spectral/fft_adapter.py:214-232`). Units are amplitude ratio (RMS). Used to raise or lower injected noise per batch. |
| `snr_weighting_mode` | Taguchi factor toggling loss weighting (`src/experiments/run_experiment.py:179-192`). Controls DiffusionLoss flags `use_weighting` and `adaptive_snr`. |
| `snr_scale_min` / `snr_scale_max` | Optional clamps for the scaling tensor applied after measuring base SNR (`src/experiments/run_experiment.py:281-301`, `src/spectral/fft_adapter.py:224-231`). Units are amplitude multipliers with defaults given by the Taguchi factor. |
| `snr_scale_factor`, `snr_measured`, `snr_base`, `snr_ratio_target` | Logged inside FFT adapter whenever `snr_ratio` is active to show the base and adjusted ratios (`src/spectral/fft_adapter.py:265-279`). |
| `snr_transform` / `snr_clamped` | Loss configuration controlling the weighting transform (raw SNR, sqrt, or clamped) used when `DiffusionLoss` falls back to manual weights (`src/core/functional/diffusion.py:27-45`, configs set default `"snr"`). |
| `snr_clip` | Maximum SNR allowed before entering “overflow” regime, default 250 (`src/core/losses.py:59-67`, `src/training/steps.py:23-28`). |
| `snr_weighting` diagnostics | `AdaptiveSNRWeight` exposes `mean_weight`, `kappa`, `ema`, etc., derived from smoothed log-SNR and residual magnitudes (`src/core/adaptive_weight.py:1-209`). |
| `snr_headroom`, `snr_high_frac`, `snr_schedule_trend`, `snr_max_trend` | Derived in the debug recorder to track how the running `snr_schedule` relates to the adaptive target (`scripts/debug/record_training_steps.py:922-938`). |
| Not found (`snr_emp`, `snr_ratio_effective`, `corruption_snr_ratio`, `schedule_snr_clamp`) | No variables with those names exist. Empirical SNR is only exposed as `snr_effective`/`snr_measured`; there is no separate “snr_emp” structure. No dedicated clamp keyed to the schedule beyond the global `snr_clip`. |

### 2.3 Variance paths
- `Var(x_0)` is measured on-the-fly either as dataset stats or per-batch `signal_var = Var(x_b - mean channels)` in the recorder (`scripts/debug/record_training_steps.py:642-670`).
- Noise variance after shaping is `Var(noise_term)` and is logged along with std, plus `noise_channel_std_min/max` to quantify band imbalance (`src/spectral/fft_adapter.py:283-287`).
- `_normalize_fft_noise` enforces unit RMS before scaling by `sqrt{1-alpha}` (`src/spectral/fft_adapter.py:203-210`), yet additional operations (mask squaring, adaptive rescale, `snr_ratio` mismatch) alter the variance budget away from `1-alpha_t`.
- FFT mask `mask = sqrt((r/r_min)^2 + 1)` and its squared variant for `freq_equalized` up-weight high frequencies, so band variances are intentionally non-uniform before the `sqrt_one_minus_alpha_t` multiplier (`src/spectral/fft_adapter.py:178-210`).
- Spectral weighting in the loss (`src/core/losses.py:30-44`) and the residual FFT adapter (`src/spectral/adapter.py:1-132`) further redistribute residual energy prior to gradient computation.
- The variance penalty term `lambda_var*(std_pred - std_true)^2` in `TrainingStepExecutor` tries to match model-predicted noise std to the actual `eps`, but this operates on centred RMS amplitudes, not on the schedule’s variance budget (`src/training/steps.py:269-295`).
- `noise_preparer` clamps both `alpha` and `sigma` before constructing `x_t`, so `signal_var + noise_var` no longer equals 1.0 even before spectral reshaping (`src/training/noise.py:118-129`).

### 2.4 Gradient-related computations
- Gradient norms are recorded in `TrainingDiagnostics.record_gradients`, summing squared grad norms and logging history (`src/training/diagnostics.py:229-250`) via helper `grad_norm` (`src/utils/debug_helpers.py:57-76`). There is no in-loop clipping in `TrainingStepExecutor`, so only diagnostics, not safety, depend on these values.
- The debug-only script adds optional global or ratio-based clipping and a “shock handler” that halves LR and clamps gradients temporarily on SNR spikes (`scripts/debug/record_training_steps.py:794-908`). These heuristics are not part of the training pipeline.
- Loss weighting uses `AdaptiveSNRWeight`, which computes per-example weights from tanh(log-SNR), residual magnitudes, and adaptive terms `kappa`, `alpha_fac`, `delta` (`src/core/adaptive_weight.py:130-205`). Weights are normalised with band-aware factors based on SNR thresholds (`src/utils/adaptive_snr.py:207-234`).
- Extra gradient terms include the FFT spectral pressure penalty (0.05 * |high/low-1|) and the variance penalty (`src/training/steps.py:252-305`). Both add bias to gradients, especially at high-SNR steps, and they reference prediction FFTs rather than theoretical signal components.

## 3. COMPARISON TABLE
| Component | First-Principles DDPM | Current Implementation | Match? | Notes / Risks |
| --- | --- | --- | --- | --- |
| Noise distribution | I.I.D. Gaussian `N(0,I)` independent of data. | Can be FFT-colored, phase-perturbed, and adaptively rescaled (`src/spectral/fft_adapter.py:178-247`). | No | Non-Gaussian sampling invalidates DDPM forward assumptions. |
| Scaling factors | `sqrt(alpha_bar_t)` and `sqrt(1-alpha_bar_t)` derived directly from schedule. | Coefficients clamped (`ALPHA_MIN=0.01`, `SIGMA_MIN=1e-4`), plus extra `snr_scale_tensor` and `strength` multipliers. | No | SNR budget per step detaches from schedule. |
| `alpha_bar_t` usage | Monotonic schedule governs both forward process and weighting. | Schedule trimmed for `sigma>=0.03`, then clamped before use; loss uses clamped SNR while noise uses masked/rescaled values. | Partial | Early high-SNR steps removed; weighting references different values from actual noise. |
| Signal component | `sqrt(alpha_bar_t)*x0`. | Same expression but with clamped `sqrt_alpha_t` (`src/training/noise.py:118-125`). | Partial | When schedule wants smaller/larger `sqrt_alpha`, clamping breaks theoretical distribution. |
| Noise component | `sqrt(1-alpha_bar_t)*eps`. | Spectral branch multiplies FFT noise by mask, renormalises, scales by `snr_ratio`, clamps per-band RMS, optionally scales again to hit target correlation (`src/spectral/fft_adapter.py:178-247`). | No | PSD shaping and adaptive rescale change the variance structure. |
| Empirical vs theoretical SNR | Should coincide by construction. | `snr_schedule` logs schedule ratio; `snr_effective` measures actual RMS ratio and diverges widely when `snr_scale_factor` != 1. | No | Weighting is based on schedule SNR, so mismatch is uncorrected. |
| Variance preservation | `Var(signal)+Var(noise)=1`. | Clamping, FFT masks, and `snr_ratio` scaling break the invariant; `variance_ratio` diagnostics show drift and require penalties (`src/training/steps.py:269-305`). | No | Without enforcement, loss sees inconsistent target magnitudes. |
| Spectral shaping effects | None; noise is white. | Optional reciprocal-radius mask and equalisation emphasise HF noise, plus noise channel std spread logged (`src/spectral/fft_adapter.py:178-287`). | Intentional deviation | Needs theory aligning mask weights with DDPM assumptions. |
| Normalization effects | None beyond schedule-defined coefficients. | `_normalize_fft_noise`, spectral adapter re-scaling, overflow renorm push activations back into safe ranges. | Partial | Helps numerics but changes stochastic process. |
| SNR-weighted loss | Typically constant weighting or analytic weighting matched to theory. | Adaptive weighting uses tanh(log-SNR) but based on scheduled SNR rather than measured SNR; freeze logic triggered by prediction std drift. | Partial | Potential mismatch when effective SNR is very different, causing mis-weighted gradients. |
| Gradient norms | Expect roughly stable norms; no extra penalties. | Additional FFT penalty and variance penalty modify gradients; no core clipping except diagnostic script. | No | Gradients can oscillate with spectral pressure excursions. |
| Additions (clamps, adapters) | None. | Multiple safety clamps (alpha, sigma, SNR clip, overflow bridge, spectral adapter). | No | Each clamp should be justified relative to theory. |
| Missing invariants | SNR monotonicity, variance sum, Gaussian noise. | Diagnostics exist but no enforcement; tests absent. | No | Without invariants, training may drift unpredictably. |

## 4. DIAGNOSTIC FLAGS
- Variance budgets are not preserved once FFT masks, `snr_ratio`, and adaptive rescale act (signal+noise ≠ 1); only the variance penalty hints at the drift.
- Schedule SNR is logged and used for weighting, yet effective SNR can be orders of magnitude different because `snr_scale_tensor` clamps noise (`src/spectral/fft_adapter.py:214-279`).
- Noise can be suppressed aggressively (`snr_scale_min` as low as 0.01) leading to `snr_effective` >> schedule SNR; overflow logic still keys off schedule, so high-SNR steps are misdetected.
- Clamping `sqrt_alpha_t`/`sqrt_one_minus_alpha_t` and trimming timesteps remove the highest-SNR steps, so the “forward” chain no longer follows the intended cosine/DDPM path.
- Adaptive SNR weights compute tanh(log-SNR) from schedule values, mixing amplitude ratios with log-power interpretations; but `snr_ratio` is defined on RMS amplitudes, so amplitude vs power units are inconsistent.
- Multiple SNR metrics exist (`snr_schedule`, `snr_effective`, `snr_measured`, `snr_ratio`, `snr_headroom`) without a single source of truth; `snr_emp` is absent even though diagnostics rely on empirical RMS.
- Non-Gaussian “phase” corruption violates the assumptions behind the DDPM loss; when activated it injects structured residuals correlated with `x0`.
- Gradient norms are only monitored; no automatic clipping in the main pipeline means spikes from spectral penalties can destabilize training.
- Overflow handling clamps residuals when `snr_raw > snr_clip`, but because the actual noise variance is lower than expected, overflow warnings may be frequent even when the empirical SNR is moderate.

## 5. CORRECTION SUGGESTIONS
1. **Unify SNR definitions**: pick a canonical empirical metric (`snr_emp = signal_var/noise_var`) measured after all shaping, log it everywhere, and express schedule SNR as `snr_theory`. Keep `snr_ratio` strictly as a desired target ratio.
2. **Rename and document**: rename `snr_effective` to `snr_emp` in stats, reserve `snr_effective` for values after adaptive controllers if needed. Deprecate unused labels (e.g., mention that `snr_emp`/`corruption_snr_ratio` do not exist).
3. **Align formulas**: ensure `snr_scale_tensor` operates on variance (squared RMS) so that the final noise variance equals `(1-alpha) * target`. Apply clamp to squared scale factors, not amplitudes, and document units.
4. **Restore variance consistency**: remove or justify `ALPHA_MIN`, `ALPHA_MAX`, and `SIGMA_MIN` clamps; if stability needs a floor, adjust the schedule instead of clipping per batch. After FFT shaping, rescale the coloured noise so that `mean(noise_component^2) = (1-alpha)` exactly.
5. **Simplify scaling path**: replace sequential operations (`mask -> normalize -> strength -> snr_ratio -> adaptive_rescale`) with a single scaling equation derived from desired `snr_target`. Provide tests that `Var(signal)+Var(noise)=1` within tolerance.
6. **Spectral shaping compatibility**: normalise the mask so that its RMS over frequencies is 1 before applying `sqrt(1-alpha)`; if additional emphasis is needed, incorporate it into the beta/log-SNR schedule rather than ad-hoc multipliers.
7. **Consistent SNR schedule**: if timesteps are trimmed, recompute `alpha_bar` to keep monotonic SNR; avoid clamping inside the training loop.
8. **Loss weighting update**: feed `snr_emp` into `AdaptiveSNRWeight` (perhaps after smoothing) so that weights reflect the actual noise level seen by the model.
9. **Diagnostics/tests**: add automated tests that fail when `snr_emp` deviates from `snr_theory` beyond a tolerance, or when `Var(signal)+Var(noise)` strays from 1.0. Include unit tests for `add_uniform_frequency_noise` verifying Gaussianity when masks are flat.
10. **Logging policy**: log a triad (`snr_theory`, `snr_emp`, `snr_rel = snr_emp/snr_theory`) plus dB versions (`10*log10`). Make this the single source of SNR truth per batch.

## 6. FINAL RECOMMENDED DEFINITIONS
| Quantity | Suggested Definition | Intended Behavior |
| --- | --- | --- |
| Forward process | `x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps_shaped` where `eps_shaped` is whitened noise whose RMS equals 1 before scaling. | Maintains theoretical schedule while allowing optional spectral shaping that preserves total variance. |
| Signal component | `signal = sqrt(alpha_bar_t)*x_0` with no clamping; `alpha_bar_t` derived from an adjusted schedule if trimming is needed. | Monotone decay; matches DDPM theory exactly. |
| Noise component | `noise = sqrt(1-alpha_bar_t)*epsilon_colored`, where `epsilon_colored` has unit variance per sample (post-FFT). | Ensures `Var(noise)=1-alpha_bar_t` even with frequency masks. |
| Variance check | `Var(signal) + Var(noise)` computed empirically per batch. | Should stay within `[1±1e-3]`; failure triggers diagnostics/test. |
| `SNR_theory` | `alpha_bar_t / (1-alpha_bar_t)` saved directly from schedule generation. | Monotonic decrease; same value used for weighting baselines. |
| `SNR_emp` | `(signal_rms^2)/(noise_rms^2)` measured after corruption (`snr_effective` renamed). | Tracks how shaping deviates from theory; equals `SNR_theory` when shaping is neutral. |
| `SNR_rel` | `SNR_emp / SNR_theory` (optionally log-scale). | Equals 1 when implementation matches theory; deviations highlight scaling errors. |
| `grad_norm` | `sqrt(sum(||grad_i||^2))` computed after optional clipping; log both unclipped and clipped values. | Provides stable indicator of training health; enables automated clipping thresholds. |
| Loss | Baseline MSE on `eps` or `v`, optionally plus `lambda_var*(std_pred-std_true)^2` only if variance invariants fail; avoid stacking FFT penalties unless justified. | Keeps objective aligned with DDPM derivation while still penalising rare drift. |
| Diagnostics | Log `snr_theory`, `snr_emp`, `snr_rel`, `variance_ratio`, `noise_channel_std_{min,max}`, `snr_scale_factor`, `snr_dB` alongside gradient stats. Export invariant checks (variance sum, SNR match) as pass/fail counters. | Ensures a single source of truth for SNR while capturing spectral side effects. |

Natural next steps:
1. Implement the unified SNR logging (theory/emp/rel) and wire `snr_emp` into AdaptiveSNRWeight.
2. Add tests for variance invariants in `add_uniform_frequency_noise` and for schedule clamping.
3. Decide whether to remove or formalise the alpha/sigma clamps so the forward process has a clear mathematical description.
