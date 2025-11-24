# Spectral Diffusion – Refactor & Pruning TODO List (Unified Forward Process)

This file tracks the concrete steps required to fully prune legacy code, align all modules to the unified DDPM‑correct forward process, and stabilise the repository architecture for future research.

---

## 1. Legacy Removal & Quarantine (Stage 1)

### 1.1 Move legacy components to `archive/legacy/`
Move these modules/scripts with no remaining references in the live pipeline:
- `src/utils/adaptive_snr.py`
- `src/core/overflow_handler.py`
- `src/core/snr_scheduler.py`
- `src/spectral/adapter.py` (legacy residual adapter)
- `src/core/model_unet_spectral.py` (phase attention hooks)
- `scripts/archive/run_full_report*.sh`
- Old Taguchi configs:
  - `configs/taguchi/L23_*`
  - `configs/taguchi/legacy_*`
  - Auto‑generated L27 notes before unification

After moving, purge all imports referencing these archived modules across src/core, src/spectral, tests, and CLI.

### 1.2 Move legacy documentation to `docs/legacy/`
Files to archive:
- `docs/training_flow.md` (adaptive v3.0)
- `docs/taguchi_l27_run_notes.md`
- `docs/taguchi_factor_plan.md`
- Old SNR weighting notes

---

## 2. Config & Schema Purge

### 2.1 Remove deprecated YAML keys from all active configs
Delete keys:
- `uniform_corruption`
- `uniform_corruption_scale`
- `freq_equalized_noise`
- `phase_attention_capacity`
- `spectral_adapter_placement`
- `spectral_loss_weighting`
- `adaptive_snr`
- `snr_scale_min/max`
- `variance_penalty`, `spectral_pressure`

Ensure all active configs only contain:
- `snr_ratio`
- `spectral_operator_mode`
- `sampler_type`
- `sampling_steps`
- `train_steps`
- `image_resolution`

### 2.2 Validate Taguchi configs
- `factor_registry.yaml` must list exactly the six active factors.
- All OA CSV designs must match these factors, with correct level indexing.

### 2.3 CLI and model registry must expose only baseline and unet_tiny variants; remove variant mappings to spectral models.

---

## 3. Step Recorder & Diagnostics (Must Use Unified Paths)

### 3.1 Guarantee recorder does not implement its own noise path
- Remove any local SNR or schedule computations.
- Ensure it calls:
  - `NoisePreparer.prepare(...)`
  - `build_diffusion(...)`
  - `TrainingStepExecutor.step(...)`
- All metrics must come from unified stats:  
  `snr_theory`, `snr_emp`, `snr_rel`, `variance_sum`, `grad_norm`.

Ensure recorder imports no legacy kwargs; remove schedule recomputation and enforce metrics strictly via NoisePreparer and TrainingStepExecutor.

### 3.2 Purge legacy logging fields
Delete references to:
- `snr_schedule_*`
- `snr_effective`
- `overflow_*`
- `spectral_pressure`
- `variance_ratio`

---

## 4. Tests & Invariant Enforcement

### 4.1 Tests required
- RMS(eps_shaped) = 1 ± 1e‑4  
- Var(signal) + Var(noise) = 1 ± tolerance  
- snr_rel ≈ 1 for `spectral_operator_mode=none`  
- Monotonic `snr_theory(t)`  
- Taguchi factor mapping integrity

### 4.2 Remove/relax flaky tests
- Any tests referencing removed knobs must be deleted or archived.
- Delete or archive tests referencing SpectralUNet, spectral adapters, phase attention.

---

## 5. Documentation Update (Active Path Only)

### 5.1 Keep only unified technical documents
Should remain:
- `docs/snr_audit.md`
- `docs/frequency_snr_study.md`
- `docs/config_reference.md`
- `docs/refactor_todolist.md`
- Clean `docs/snr_audit.md` of spectral/adapter/pressure mentions.

### 5.2 Update README
- Remove references to spectral adapters, phase attention, adaptive SNR.
- Describe only the unified forward process and the six Taguchi knobs.

---

## 6. Stage 2: Repository Restructure (after full Stage 1 purge)

Proposed new layout:

```
src/
  diffusion/        # Scheduler, coefficients, forward process
  noise/            # NoisePreparer + spectral_operator
  training/         # Step executor, loss, trainer
  diagnostics/      # Diagnostics + recorder
  taguchi/          # DoE system, registries, OA loading
  models/           # UNet and related architectures
```

If adopted:
- Fix all import paths in one atomic commit.
- Update tests and scripts accordingly.

---

## 8. Report & Taguchi Cleanups

- Remove OA_DESIGN_MAP entries for L23 and L18.
- Ensure only L27 remains.
- Scrub `run_full_report_32x32.sh` of spectral paths.

---

## 7. Completion Criteria

The refactor is complete when:
- No legacy knobs appear in any active YAML config.
- All noise is injected exclusively via `spectral_operator` + single scale `k`.
- All SNR logs use `{snr_theory, snr_emp, snr_rel}`.
- All forward invariants are tested and pass.
- Taguchi experiments operate on the six active factors only.
- Step recorder contains zero independent forward/SNR logic.
- Documentation references no removed mechanisms.
