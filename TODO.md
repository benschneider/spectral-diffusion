# Spectral Diffusion TODOs

Legend: ✅ complete · 🟡 in progress · ⬜ pending

| Status | Area | Current Status | Immediate Next Step | Dependency | Notes / Implementation Tip |
| - | - | - | - | - | - |
| 🟡 | Spectral utilities | SpectralAdapter integrated (input/output/per-block) | Expand weighting options & adapter strength mixing | None | Adapter handles FFT/iFFT; timing & loss weighting tracked separately |
| ✅ | Spectral-weighted losses | Residual weighting via SpectralAdapter | Consider mixing strategies & per-frequency strength | Spectral utilities | Works with `loss.spectral_weighting` (none/radial/bandpass) |
| 🟡 | Diffusion sampling & image metrics | DDPM sampling + evaluation hook (MSE/MAE/PSNR) | Add DDIM/other samplers and compute LPIPS/FID on outputs | Real training components | Images stored in `results/logs/<run_id>/images/`; evaluation block controls scoring |
| 🟡 | Taguchi S/N analysis | CLI `src.analysis.taguchi_stats` available | Integrate into batch workflow & notebooks | Taguchi runner outputs | Generates `taguchi_report.csv` with S/N ratios per factor |
| 🟡 | Evaluation metrics | Folder-level MSE/MAE/PSNR, opt. FID via torchmetrics | Add LPIPS + integrate sampler outputs for FID/LPIPS | Diffusion sampling | Uses PIL & torchvision; warns if torchmetrics missing |
| ⬜ | Sampler support (DDIM/DPM-Solver++) | Fallback to DDPM currently | Implement DDIM solver + add DPM-Solver++ | Diffusion sampling | Necessary for fair spectral comparisons in arrays |
| ⬜ | Structured logging | Pending | Add JSONL logs & system metadata per run | Logging polish | Capture hardware info in `results/logs/<run_id>/system.txt` |
| 🟡 | Testing / CI | Pytests for FFT, TinyUNet, Taguchi, metrics | Add import/dry-run tests, baseline equivalence check, CI workflow | Validation automation | Ensure deterministic behavior, spectral toggle off == baseline |
| ⬜ | Logging polish | Console logging ready | Add CLI log-level flag & structured logs | Independent | Hook into CLI via `--log-level` |
| ⬜ | Dataset handling | Manual CIFAR download documented | Support auto-download flag + checksum validation | Network availability | Document dataset caching for CI/local |
| ⬜ | Documentation | README updated | Add `docs/theory.md` & `docs/experiments.md` with focused guides | None | Keep README concise, document flow-matching roadmap |
| ⬜ | Analysis notebooks | Not started | Plot loss vs time, FFT overhead vs efficiency, Taguchi summaries | Metrics & S/N tooling | Consume `results/summary.csv`, `taguchi_report.csv` |
| ✅ | FFT timing instrumentation | CPU/CUDA timing captured per adapter; metrics recorded | Report sampling/training breakdown in analysis scripts | Spectral utilities | Exposed as `spectral_*_time_seconds` and sampling counterparts |
| ✅ | Real training components | Diffusion training loop active (ε-pred, cosine schedule) | Next: v/x0 prediction, sampling utilities | Spectral utilities | Baseline-conv path remains for synthetic smoke tests |
| ✅ | Taguchi runner outputs | Per-run configs/metrics persisted | Next: S/N analysis & factor reporting | Metrics availability | Artifacts mirror single-run structure |

**Execution Order**
1. Taguchi S/N analysis tooling  
2. Diffusion sampling metrics (LPIPS/FID)  
3. Evaluation metrics (LPIPS integration)  
4. Sampler support (true DDIM/DPM-Solver++)  
5. Structured logging & log-level CLI flag  
6. Testing/CI harness (import, dry-run, baseline equivalence)  
7. Dataset handling polish  
8. Spectral adapter tuning (mixing strategies)  
9. Documentation & analysis notebooks  
