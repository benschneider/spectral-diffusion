# Spectral Diffusion TODOs

| Area | Current Status | Immediate Next Step | Dependency | Notes / Implementation Tip |
| - | - | - | - | - |
| Evaluation metrics (`src/evaluation/metrics.py`) | Basic metrics implemented | Extend with dataset-aware stats (e.g., FID/LPIPS placeholders → real implementations) | Availability of metric dependencies | Use cached features to keep evaluation lightweight once implemented |
| Logging polish (`train_model.py`, `src/training/pipeline.py`) | Per-run file logging in place | Add optional console log level flag and structured JSON logs | Independent | Hook into CLI via `--log-level` to avoid noisy runs |
| Validation automation | `validate_setup.py` covers dry + fast real run | Integrate script into CI (GitHub Actions workflow) | Metrics + logging updates | Cache CIFAR-10 between CI runs or fall back to synthetic mode |
| Taguchi runner outputs (`src/experiments/run_experiment.py`) | Persists per-run configs/metrics and summary entries | Add Taguchi S/N analysis and factor ranking utilities | Depends on metrics completeness | Artifacts now mirror single-run structure for downstream analysis |
| Spectral utilities (`src/spectral/fft_utils.py`) | FFT round-trip + weighting masks available | Expose richer weighting schemes and connect to spectral-aware losses | Independent | Current options: `enabled`, `normalize`, `weighting` (`none`, `radial`, `bandpass`) |
| Real training components | Tiny UNet + reconstruction loss active | Implement diffusion-specific losses & noise schedule; wire spectral toggles into forward pass | Spectral utilities upgrade | Maintain baseline `baseline_conv` path for synthetic quick tests |
| Testing / CI | Unit tests for FFT + TinyUNet spectral toggles added (`tests/`) | Add `tests/test_imports.py` and `tests/test_training_dryrun.py`; integrate pytest or simple runner | Validation script ready | Ensure determinism checks cover seeding and artifact generation |
| Dataset handling | CIFAR-10 loader with manual download instructions | Optionally enable auto-download flag and checksum validation | Network availability | Document dataset caching strategy for CI/local users |

**Recommended Execution Order**
1. Taguchi runner persistence
2. Real diffusion model integration
3. Spectral metrics & weighting extensions
4. Dataset handling polish
5. Metrics enhancements (FID/LPIPS)
6. Logging refinements
7. Testing/CI harness
