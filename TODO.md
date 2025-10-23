# Spectral Diffusion TODOs

| Area | Current Status | Immediate Next Step | Dependency | Notes / Implementation Tip |
| - | - | - | - | - |
| Evaluation metrics (`src/evaluation/metrics.py`) | Basic metrics implemented | Extend with dataset-aware stats (e.g., FID/LPIPS placeholders → real implementations) | Availability of metric dependencies | Use cached features to keep evaluation lightweight once implemented |
| Logging polish (`train_model.py`, `src/training/pipeline.py`) | Per-run file logging in place | Add optional console log level flag and structured JSON logs | Independent | Hook into CLI via `--log-level` to avoid noisy runs |
| Validation automation | `validate_setup.py` covers dry + fast real run | Integrate script into CI (GitHub Actions workflow) | Metrics + logging updates | Cache CIFAR-10 between CI runs or fall back to synthetic mode |
| Taguchi runner outputs (`src/experiments/run_experiment.py`) | Returns metrics in-memory only | Append each run to `results/summary.csv` and persist configs/metrics per run | Depends on metrics completeness | Reuse `append_run_summary()` for consistency across single and batch runs |
| Spectral utilities (`src/spectral/fft_utils.py`) | Identity placeholders | Implement `torch.fft.fft2/ifft2` with optional normalization and derived spectral params | Independent | Gate transforms behind `spectral.enabled` flag so baseline path stays unchanged |
| Real training components | Tiny UNet + reconstruction loss active | Implement diffusion-specific losses & noise schedule; wire spectral toggles into forward pass | Spectral utilities upgrade | Maintain baseline `baseline_conv` path for synthetic quick tests |
| Testing / CI | None | Add `tests/test_imports.py` and `tests/test_training_dryrun.py`; integrate pytest or simple runner | Validation script ready | Ensure determinism checks cover seeding and artifact generation |
| Dataset handling | CIFAR-10 loader with manual download instructions | Optionally enable auto-download flag and checksum validation | Network availability | Document dataset caching strategy for CI/local users |

**Recommended Execution Order**
1. Taguchi runner persistence
2. Spectral utilities upgrade
3. Real diffusion model integration
4. Dataset handling polish
5. Metrics enhancements (FID/LPIPS)
6. Logging refinements
7. Testing/CI harness
