# Spectral Diffusion TODOs

Legend: ✅ complete · 🟡 in progress · ⬜ pending

| Status | Area | Current Status | Immediate Next Step | Dependency | Notes / Implementation Tip |
| - | - | - | - | - | - |
| 🟡 | Spectral utilities | SpectralAdapter integrated (input/output/per-block) | Expand weighting options & adapter strength mixing | None | Adapter handles FFT/iFFT; timing & loss weighting tracked separately |
| ✅ | Spectral-weighted losses | Residual weighting via SpectralAdapter | Consider mixing strategies & per-frequency strength | Spectral utilities | Works with `loss.spectral_weighting` (none/radial/bandpass) |
| ⬜ | FFT timing instrumentation | Not started | Add CUDA/CPU timers to report precise `spectral_time_seconds` | Spectral Adapter | Use CUDA events + perf_counter fallback |
| ⬜ | Diffusion sampling & image metrics | Not started | Implement sampler CLI (DDIM/DDPM) writing images for FID/LPIPS | Real training components | Store samples under `results/images/<run_id>/` |
| ⬜ | Taguchi S/N analysis | Not started | Compute S/N ratios (larger/smaller-the-better) & output `taguchi_report.csv` | Taguchi runner | Use responses like `loss_drop_per_second`, `images_per_second` |
| 🟡 | Evaluation metrics | Folder-level MSE/MAE/PSNR, opt. FID via torchmetrics | Add LPIPS + integrate sampler outputs for FID/LPIPS | Diffusion sampling | Uses PIL & torchvision; warns if torchmetrics missing |
| ⬜ | Structured logging | Pending | Add JSONL logs & system metadata per run | Logging polish | Capture hardware info in `results/logs/<run_id>/system.txt` |
| 🟡 | Testing / CI | Pytests for FFT, TinyUNet, Taguchi, metrics | Add import/dry-run tests, baseline equivalence check, CI workflow | Validation automation | Ensure deterministic behavior, spectral toggle off == baseline |
| ⬜ | Logging polish | Console logging ready | Add CLI log-level flag & structured logs | Independent | Hook into CLI via `--log-level` |
| ⬜ | Dataset handling | Manual CIFAR download documented | Support auto-download flag + checksum validation | Network availability | Document dataset caching for CI/local |
| ⬜ | Documentation | README updated | Add `docs/theory.md` & `docs/experiments.md` with focused guides | None | Keep README concise, document flow-matching roadmap |
| ⬜ | Analysis notebooks | Not started | Plot loss vs time, FFT overhead vs efficiency, Taguchi summaries | Metrics & S/N tooling | Consume `results/summary.csv`, `taguchi_report.csv` |
| ✅ | Real training components | Diffusion training loop active (ε-pred, cosine schedule) | Next: v/x0 prediction, sampling utilities | Spectral utilities | Baseline-conv path remains for synthetic smoke tests |
| ✅ | Taguchi runner outputs | Per-run configs/metrics persisted | Next: S/N analysis & factor reporting | Metrics availability | Artifacts mirror single-run structure |

**Execution Order**
1. FFT timing instrumentation  
2. Diffusion sampling & image metrics (FID/LPIPS)  
3. Taguchi S/N analysis tooling  
4. Evaluation metrics (LPIPS integration)  
5. Structured logging & log-level CLI flag  
6. Testing/CI harness (import, dry-run, baseline equivalence)  
7. Dataset handling polish  
8. Spectral adapter tuning (mixing strategies)  
9. Documentation & analysis notebooks  
