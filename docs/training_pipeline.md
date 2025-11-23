# Training Pipeline Notes (Unified SNR)

The training stack now relies on the simplified forward process described in `docs/snr_audit.md`:

- Forward noise: `x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps_shaped * (1/snr_ratio)` with `eps_shaped = spectral_operator(eps_raw, mode)` and RMS=1.
- SNR metrics are unified to `snr_theory`, `snr_emp`, `snr_rel`, plus `variance_sum` and grad norms for diagnostics.
- No adaptive SNR governors, overflow bridges, spectral pressure penalties, or per-batch clamping remain in the training path.
- Taguchi and reporting now surface only real knobs: `snr_ratio`, `spectral_operator_mode`, `sampler_type`, `sampling_steps`, `train_steps`, `image_resolution`.

Refer to `src/training/noise.py`, `src/spectral/operator.py`, and `src/training/diagnostics.py` for the current reference implementation.
