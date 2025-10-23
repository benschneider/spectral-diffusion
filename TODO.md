# Spectral Diffusion TODOs

| Status | Area | Current Status | Immediate Next Step | Dependency | Notes / Implementation Tip |
| - | - | - | - | - | - |
| 🟡 | Evaluation metrics (`src/evaluation/metrics.py`) | Folder-level MSE/MAE/PSNR implemented; optional FID via torchmetrics | Add LPIPS + integrate sampler outputs for FID/LPIPS | Availability of metric dependencies | Uses PIL + torchvision; raises informative error if torchmetrics missing |
| ⬜ | Logging polish (`train_model.py`, `src/training/pipeline.py`) | Per-run file logging in place | Add optional console log level flag and structured JSON logs | Independent | Hook into CLI via `--log-level` to avoid noisy runs |
| ⬜ | Validation automation | `validate_setup.py` covers dry + fast real run | Integrate script into CI (GitHub Actions workflow) | Metrics + logging updates | Cache CIFAR-10 between CI runs or fall back to synthetic mode |
| ✅ | Taguchi runner outputs (`src/experiments/run_experiment.py`) | Persists per-run configs/metrics and summary entries | Add Taguchi S/N analysis and factor ranking utilities | Depends on metrics completeness | Artifacts now mirror single-run structure for downstream analysis |
| ⬜ | Taguchi S/N analysis (`src/analysis/taguchi_stats.py`) | Not started | Compute S/N ratios & factor ranks; emit `taguchi_report.csv` | Taguchi runner outputs | Use loss_drop_per_second / images_per_second as responses |
| 🟡 | Spectral utilities (`src/spectral/fft_utils.py`) | FFT round-trip + weighting masks available | Expose richer weighting schemes and connect to spectral-aware losses | Independent | Current options: `enabled`, `normalize`, `weighting` (`none`, `radial`, `bandpass`) |
| ⬜ | Spectral adapter module | Pending | Create pluggable FFT adapter (pre/post UNet or per block) | Spectral utilities upgrade | Enables per-block spectral experimentation |
| ⬜ | Spectral-weighted losses | Pending | Add `spectral_weighting` option to loss config (frequency-domain weighting) | Spectral utilities upgrade | Compare data-shaping vs loss-shaping strategies |
| ⬜ | FFT timing instrumentation | Pending | Add CUDA/CPU timing helper & log `spectral_time_seconds` precisely | Spectral utilities upgrade | Use CUDA events when available, perf_counter fallback otherwise |
| ✅ | Real training components | Diffusion training loop active (eps-pred, cosine schedule) | Add v/x0 prediction options, sampling utilities, and image outputs | Spectral utilities upgrade | Maintain baseline `baseline_conv` path for synthetic quick tests |
| ⬜ | Diffusion sampling & image metrics | Not started | Add sampler CLI (DDIM/DDPM) and write images for FID/LPIPS evaluation | Real training components | Store samples under `results/images/<run_id>/` |
| 🟡 | Testing / CI | Unit tests for FFT + TinyUNet spectral toggles added (`tests/`) | Add import/dry-run tests; integrate pytest/CI workflow | Validation script ready | Ensure determinism checks cover seeding and artifact generation |
| ⬜ | Baseline equivalence checks | Pending | Extend `validate_setup.py` with baseline vs spectral-disabled allclose test | Testing / CI | Prevent regressions when toggling spectral path off |
| ⬜ | MODEL_REGISTRY ergonomics | Pending | Improve error messages & allow decorator registration | Testing / CI | Helpful for Taguchi sweeps to catch typos early |
| ⬜ | Structured logging | Pending | Emit JSONL logs + system metadata per run | Logging polish | Capture hardware/pip freeze in `results/logs/<run_id>/system.txt` |
| ⬜ | Dataset handling | CIFAR-10 loader with manual download instructions | Optionally enable auto-download flag and checksum validation | Network availability | Document dataset caching strategy for CI/local users |
| ⬜ | Documentation (`docs/theory.md`, `docs/experiments.md`) | Not started | Split conceptual overview & experiment recipes into docs/ | None | Keep README concise; include flow-matching goals |
| ⬜ | Analysis notebooks | Not started | Plot loss_vs_time, spectral overhead vs efficiency | Metrics availability | Consume `results/summary.csv` and `taguchi_report.csv` |

**Recommended Execution Order**
1. Taguchi S/N analysis tooling
2. Spectral adapter & spectral-weighted losses
3. FFT timing instrumentation
4. Diffusion sampling & metrics (FID/LPIPS)
5. Logging refinements & structured logs
6. Dataset handling polish
7. Testing/CI harness (incl. baseline equivalence)
8. Documentation & analysis notebooks

Legend: ✅ complete · 🟡 in progress · ⬜ pending
