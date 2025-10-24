# Spectral Diffusion TODOs

Legend: ✅ complete · 🟡 in progress · ⬜ pending

| Status | Area | Current Status | Immediate Next Step | Dependency | Notes / Implementation Tip |
| - | - | - | - | - | - |
| ✅ | Pipeline architecture | Training/sampling/evaluation split with run-dir layout documented; scripts & validation updated | Monitor downstream tooling and consolidate legacy aliases when safe | None | `results/runs/<run_id>/...` now hosts configs/logs/metrics/checkpoints/samples; smoke + Taguchi scripts target new paths |
| 🟡 | Spectral utilities | SpectralAdapter integrated (input/output/per-block) | Expand weighting options & adapter strength mixing | None | Adapter handles FFT/iFFT; timing & loss weighting tracked separately |
| ✅ | Spectral-weighted losses | Residual weighting via SpectralAdapter | Consider mixing strategies & per-frequency strength | Spectral utilities | Works with `loss.spectral_weighting` (none/radial/bandpass) |
| 🟡 | Spectral model research | Research plan drafted, complex layers + minimal `SpectralUNet` prototype landed | Benchmark spectral vs spatial baselines on synthetic/CIFAR-10 | Spectral utilities | Manage real/imag parts explicitly; explore complex convolutions & spectral noise targets |
| 🟡 | Diffusion sampling & image metrics | DDPM/DDIM/DPM-Solver++ samplers callable post-training; evaluation CLI writes FID/LPIPS metrics JSON | Integrate FID/LPIPS into reporting dashboards; explore batch inference tooling | Real training components | Samples now live under `results/runs/<run_id>/samples/<tag>/`; evaluation updates metadata when requested |
| ✅ | Sampler framework | Registry live with DDPM, DDIM, DPM-Solver++; tests cover registration/fallbacks | Monitor downstream usage and add higher-order solvers as needed | Pipeline architecture | Legacy `sample_ddpm` shim retained for compatibility until downstream scripts migrate |
| 🟡 | Sampler support (DDIM/DPM-Solver++) | DDPM, DDIM, DPM-Solver++ available | Prototype ancestral/DDPM++ variants & schedule-aware steppers | Sampler framework | Necessary for fair spectral comparisons in arrays |
| 🟡 | Taguchi S/N analysis | CLI `src.analysis.taguchi_stats` available | Integrate into batch workflow & notebooks | Taguchi runner outputs | Generates `taguchi_report.csv` with S/N ratios per factor |
| 🟡 | Evaluation metrics | Folder-level MSE/MAE/PSNR, opt. FID via torchmetrics | Add LPIPS + integrate sampler outputs for FID/LPIPS | Diffusion sampling | Uses PIL & torchvision; warns if torchmetrics missing |
| ⬜ | Structured logging | Pending | Add JSONL logs & system metadata per run | Logging polish | Capture hardware info in `results/runs/<run_id>/system.txt` |
| 🟡 | Testing / CI | Pytests for FFT, TinyUNet, Taguchi, training pipeline, CLI smoke tests, sampler registry | Next: (c) evaluate CLI suite with real metrics, baseline equivalence + CI workflow | Pipeline architecture | Reuse synthetic configs; keep CPU-only path fast |
| ⬜ | Logging polish | Console logging ready | Add CLI log-level flag & structured logs | Independent | Hook into CLI via `--log-level` |
| ⬜ | Dataset handling | Manual CIFAR download documented | Support auto-download flag + checksum validation | Network availability | Document dataset caching for CI/local |
| ⬜ | Documentation | README updated | Add `docs/theory.md` & `docs/experiments.md` with focused guides | None | Keep README concise, document flow-matching roadmap |
| ⬜ | Analysis notebooks | Not started | Plot loss vs time, FFT overhead vs efficiency, Taguchi summaries | Metrics & S/N tooling | Consume `results/summary.csv`, `taguchi_report.csv` |
| ✅ | FFT timing instrumentation | CPU/CUDA timing captured per adapter; metrics recorded | Report sampling/training breakdown in analysis scripts | Spectral utilities | Exposed as `spectral_*_time_seconds` and sampling counterparts |
| ✅ | Real training components | Diffusion training loop active (ε-pred, cosine schedule) | Next: v/x0 prediction, sampling utilities | Spectral utilities | Baseline-conv path remains for synthetic smoke tests |
| ✅ | Taguchi runner outputs | Per-run configs/metrics persisted | Next: S/N analysis & factor reporting | Metrics availability | Artifacts mirror single-run structure |

**Execution Order**
1. Pipeline split + sampler registry scaffolding  
2. Taguchi S/N analysis tooling  
3. Diffusion sampling metrics (LPIPS/FID)  
4. Evaluation metrics (LPIPS integration)  
5. Spectral UNet research prototype  
6. Sampler support (DDIM/DPM-Solver++)  
7. Structured logging & log-level CLI flag  
8. Testing/CI harness (CLI smoke + baseline equivalence)  
9. Dataset handling polish  
10. Spectral adapter tuning (mixing strategies)  
11. Documentation & analysis notebooks  
