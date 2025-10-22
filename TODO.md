# Spectral Diffusion TODOs

| Area | Current Status | Immediate Next Step | Dependency | Notes / Implementation Tip |
| - | - | - | - | - |
| Evaluation metrics (`src/evaluation/metrics.py`) | Not implemented | Implement `compute_basic_metrics()` to report mean loss, MAE, runtime; leave FID/LPIPS stubs | Metrics logging integration in `TrainingPipeline` | Use `time.perf_counter()` around epochs so Taguchi runs have real runtime stats |
| Logging polish (`train_model.py`, `src/training/pipeline.py`) | Console logs only | Attach `logging.FileHandler` per run (e.g., `results/logs/<run_id>/run.log`) | Independent | Create handler before pipeline instantiation for consistent log capture |
| Validation automation | Manual CLI runs | Add `validate_setup.py` (dry run + short run + artifact presence check) | Metrics + logging updates | Script should reuse `train_from_config()` and assert metrics/log files exist |
| Taguchi runner outputs (`src/experiments/run_experiment.py`) | Returns metrics in-memory only | Append each run to `results/summary.csv` and persist configs/metrics per run | Depends on metrics completeness | Reuse `append_run_summary()` for consistency across single and batch runs |
| Spectral utilities (`src/spectral/fft_utils.py`) | Identity placeholders | Implement `torch.fft.fft2/ifft2` with optional normalization and derived spectral params | Independent | Gate transforms behind `spectral.enabled` flag so baseline path stays unchanged |
| Real training components | Placeholder conv + MSE + synthetic data | Replace with diffusion UNet, noise schedule, and real dataset loader | Depends on spectral toggles (if enabled) | Keep baseline vs spectral variants configurable via shared pipeline |
| Testing / CI | None | Add `tests/test_imports.py` and `tests/test_training_dryrun.py`; integrate pytest or simple runner | Depends on validation script | Ensure determinism checks cover seeding and artifact generation |

**Recommended Execution Order**
1. Metrics + logging enhancements
2. Validation automation script
3. Taguchi runner persistence
4. Spectral utilities upgrade
5. Real diffusion model integration
6. Testing/CI harness
