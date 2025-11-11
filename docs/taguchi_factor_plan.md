# Taguchi DOE Expansion Plan (L16 Binary + L18 Mixed Designs)

Our original goal was to move from the legacy L8 (5 binary factors) to a richer L16 experiment covering up to 15 two-level toggles in a single batch. That roadmap is still valuable for purely binary toggles, but we now also operate a mixed-level **L18 (2¹ × 3⁷)** array in production. The L18 rollout (via `configs/taguchi/factor_registry.yaml` + `configs/taguchi/L18_mixed.csv`) lets us sweep eight high-impact factors with three levels each while keeping the learning-rate schedule as a binary control. The L16 notes below remain as a parking lot for future two-level variants.

## Synthetic L23 diagnostics quickstart

- Baseline YAML: `configs/taguchi/L23_synthetic.yaml` (synthetic-only training loop tuned for early-step diagnostics).
- Python runner: `scripts/run_taguchi_synthetic_l23.py` (maps the 13-factor design to CLI overrides for `record_training_steps.py`).
- Convenience shell entry point: `scripts/run_taguchi_synthetic_l23.sh` (cleans the target directory and forwards optional overrides such as `TAGUCHI_L23_STEPS`).

Run `./scripts/run_taguchi_synthetic_l23.sh` to materialise the full L23 array (23 runs). It produces `L23_synthetic_resolved.csv` with per-factor CLI expansions plus the aggregated metrics table `results.csv` under `results/taguchi_l23_synthetic/` by default.

### Taguchi HDF5 archives

When the full 32×32 report is generated (`scripts/run_full_report_32x32.sh`), set `TAGUCHI_HDF5_ENABLED=1` and (optionally) configure `TAGUCHI_HDF5_PATH` to produce a compressed `taguchi_runs.h5` via `scripts/collate_runs_to_hdf5.py`. The converter ingests every `runs/<run_id>` folder plus the summary/taguchi CSVs, stores them under `/runs` and `/taguchi` groups, and can drop the original JSON/CSV artifacts when `TAGUCHI_HDF5_PRUNE=1`. Downstream tooling can then read this single HDF5 archive instead of scattering through multiple files (install `h5py` if it’s not already available).

### Modular Taguchi suites

Use `scripts/run_taguchi_suite.py` to compose suites from `configs/taguchi/factor_catalog.yaml` and `configs/taguchi/suites.yaml`. The helper generates a curated registry/design/manifest trio under `configs/taguchi/generated/`, estimates the runtime, and (when not running with `--dry-run`/`--estimate-only`) invokes `run_full_report_32x32.sh` with the generated artifacts pinned via `TAGUCHI_FACTOR_REGISTRY`/`TAGUCHI_ARRAY_PATH`. Try `python scripts/run_taguchi_suite.py --suite fast_synthetic --dry-run` to preview a plan or rerun without `--dry-run` to execute the curated configuration.

### Quick-run factor set

The default Taguchi driver now points at `configs/taguchi_smoke_best.yaml` and `configs/taguchi/factor_registry_quick.yaml`, which lock the long-running toggles (`train_steps`, `sampling_steps`, `image_resolution`) to their fastest levels while honoring the L23-derived best-known hyperparameters (high learning rate, adafactor, λ_var=7e-4). Swap `TAGUCHI_BASE_CONFIG`, `TAGUCHI_FACTOR_REGISTRY`, or `TAGUCHI_ARRAY_PATH` if you need a different suite of factors.

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

### Current L18 Mixed-Level Factors (Deployed)

| Column | Factor | Levels |
|--------|--------|--------|
| **A** | Spectral adapter placement (`spectral.apply_to`) | none · input_only · input_and_output |
| **B** | Spectral loss weighting (`spectral.weighting`) | none · radial_highfreq · aggressive_highfreq (band-pass) |
| **C** | Spectral noise shaping strength (`diffusion.uniform_corruption` + `spectral.freq_equalized_noise`) | off · mild_equalize · strong_equalize |
| **D** | Phase attention capacity (`model.enable_phase_attention`, `model.phase_heads`) | off · tiny · full |
| **E** | Sampler type (mapped to registry sampler) | ddim · dpm_solver_pp · spectral_guided (MASF) |
| **F** | Sampling steps (`sampling.num_steps`) | 30 · 50 · 100 |
| **G** | Curriculum mode (`training.curriculum`) | none · lowres_warmup · spectral_first |
| **H** | Learning-rate schedule (`optim.lr_schedule`) | constant · cosine · cosine_warmup |
| **I** | Training steps (`training.num_batches`) | 50 · 100 · 200 |
| **J** | Image resolution (`data.height`/`data.width`) | 32 · 64 · 128 |

These are encoded in `configs/taguchi/factor_registry.yaml` and automatically wired through `src/experiments/run_experiment.py`. The full-report scripts (`run_full_report_32x32.sh`) now execute all 18 combinations and publish `L18_summary.csv` / `taguchi_report.csv` for figure generation.

### Future Research-Driven Candidates

The recent burst of frequency-domain diffusion work (2024–2025) surfaces several promising toggles that could slot into spare L16 columns or replace lower-impact ones. Each entry below keeps the binary Taguchi structure while highlighting why it might be worth piloting.

| Letter | Factor (two-level) | Level 1 | Level 2 | Why It’s Interesting | Status / Comments |
|--------|--------------------|---------|---------|----------------------|-------------------|
| **P** | **Fourier phase diffusion** | Implicit phase handling | Training-free phase-only diffusion branch | Enables zero-shot texture/style transfer via phase swaps (IJCAI 2025 reports large qualitative gains). | 🤔 Prototype in sampler; pairs naturally with factor **B**. |
| **Q** | **Frequency prior filtering** | No frequency refinement | Adaptive magnitude filtering during diffusion | Preserves priors while removing low-frequency artefacts (OpenReview 2025, +15% FID). | 🔬 Needs configurable filter kernel before FFT inverse. |
| **R** | **Spectral autoregression** | Standard denoising | Autoregressive stepping over frequency bins | Reframes diffusion as causal AR in FFT space (Dieleman 2024 discussions). | ⚠️ Medium-high effort; would stress-test sampler registry. |
| **S** | **Frequency-aware token selection** | Full token budget | Prune low-energy frequency tokens | 2–3× faster inference without retraining (AAAI 2025). | 🚀 Low effort once energy ranking is exposed; complements **L**. |
| **T** | **Adaptive spectro-temporal diffusion** | Spatial-only pathway | Joint STFT pathway for audio/time | Extends pipeline to audio while filtering artefacts (EURASIP 2025). | 💤 Defer until audio benchmarks are in place. |
| **U** | **Frequency-aware denoising loss** | Uniform loss weighting | Band-limited weighting (boost mid/high ω) | Improves low-light/RF denoising (PMC 2025) and refines factor **H**. | ✅ Low effort—loss reweighting already partially implemented. |
| **V** | **Spectral motion generator** | Static image FFT | 3D FFT over space-time volumes | Targets video diffusion; high buzz for motion synthesis (LinkedIn 2025 demos). | 🟥 High effort—requires dataset & temporal heads. |

### Next Steps
1. ✅ Generated `configs/taguchi/L18_mixed.csv` (in production) and `configs/taguchi/L27_extended.csv` (adds train_steps + resolution) alongside the older `configs/taguchi_spectral_L16.csv` (binary backlog).
2. ✅ Refactored `src/experiments/run_experiment.py` to consume the factor registry, randomise assignments, and persist per-row metrics (`run_<n>_metrics.json`).
3. ✅ Reports ingest the new Taguchi metadata; full-report scripts include `taguchi_report.csv` automatically.
4. ▶️ Automate aggregation for GitHub Actions by adding a follow-up job that collects all 18 artifacts and re-runs `--finalize`.
5. ▶️ Evaluate future candidates (starting with **P/S/U**) and decide whether they replace existing L18 factors or spin off a dedicated L32 study.
