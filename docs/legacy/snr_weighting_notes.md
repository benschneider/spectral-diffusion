# Legacy SNR Weighting Notes (Archived)

These notes capture the pre-unification diagnostics around schedule-driven SNR
weighting, overflow handling, and variance penalties. They are preserved for
historical context and should not influence the current unified forward process.

## Fragmented SNR Metrics

- `snr_schedule` (from clamped coefficients; used for weighting).
- `snr_effective` / `snr_measured` (empirical RMS ratio; not used for weighting).
- `snr_ratio` knob (RMS amplitude scaling).
- `snr_scale_factor`, `snr_base`, `snr_ratio_target` (diagnostic).
- Missing unified structure: no `snr_emp` or `snr_rel`.

## Variance Paths

- `_normalize_fft_noise` guarantees unit RMS before subsequent scaling.
- Post-scaling variance no longer tied to `(1 - alpha_bar_t)`.
- `variance_penalty` attempts to compensate but does not enforce invariants.

## Loss Path and Weighting

- `AdaptiveSNRWeight` uses log-SNR derived from schedule, not the actual injected noise.
- Additional FFT spectral-pressure penalty and variance penalty modify gradients.

## Diagnostics

- Logs `snr_schedule_mean`, `snr_effective`, overflow, `variance_ratio`, `spectral_pressure`.
- No unified SNR triad.
