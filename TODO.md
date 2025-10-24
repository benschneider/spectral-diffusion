# Spectral Diffusion TODOs

Legend: ✅ complete · 🟡 in progress · ⬜ pending

| Status | Area | Current Status | Immediate Next Step | Dependency | Notes / Implementation Tip |
| - | - | - | - | - | - |
| 🟡 | Spectral model research | Synthetic & CIFAR-10 benchmarks captured (`results/spectral_benchmark*`) | Analyse throughput/quality trade-offs; tune spectral hyperparameters | Spectral utilities | Compare loss/runtime metrics; consider spectral regularisation |
| 🟡 | Sampler support (DDIM/DPM-Solver++) | DDPM, DDIM, DPM-Solver++ available | Prototype ancestral/DDPM++ variants & schedule-aware steppers | Sampler framework | Necessary for fair spectral comparisons in arrays |
| 🟡 | Taguchi S/N analysis | CLI auto-generates reports; scripts consume them | Build notebooks/dashboards to visualise factor rankings | Taguchi runner outputs | Generates `taguchi_report.csv` with S/N ratios per factor |
| 🟡 | Evaluation metrics | Folder-level MSE/MAE/PSNR, opt. FID via torchmetrics | Add LPIPS + integrate sampler outputs for FID/LPIPS | Diffusion sampling | Uses PIL & torchvision; warns if torchmetrics missing |
| 🟡 | Testing / CI | Pytests for FFT, TinyUNet, Taguchi, training pipeline, CLI smoke tests, sampler registry | Next: (c) evaluate CLI suite with real metrics, baseline equivalence + CI workflow | Pipeline architecture | Reuse synthetic configs; keep CPU-only path fast |
| ⬜ | Logging polish | Console logging ready; log-level flags available | Add JSONL logs & richer diagnostics for long runs | Independent | Consider optional `--json-log` flag emitting structured entries |
| ⬜ | Dataset handling | Manual CIFAR download documented | Support auto-download flag + checksum validation | Network availability | Document dataset caching for CI/local |
| ⬜ | Analysis notebooks | Not started | Plot loss vs time, FFT overhead vs efficiency, Taguchi summaries | Metrics & S/N tooling | Consume `results/summary.csv`, `taguchi_report.csv` |
| 🟡 | Diffusion sampling & image metrics | DDPM/DDIM/DPM-Solver++ samplers callable post-training; evaluation CLI writes FID/LPIPS metrics JSON | Integrate FID/LPIPS into reporting dashboards; explore batch inference tooling | Real training components | Samples now live under `results/runs/<run_id>/samples/<tag>/`; evaluation updates metadata when requested |
| ⬜ | Documentation | README updated | Add `docs/theory.md` & `docs/experiments.md` with focused guides | None | Keep README concise, document flow-matching roadmap |
| 🟡 | Spectral utilities | SpectralAdapter integrated (input/output/per-block) | Expand weighting options & adapter strength mixing | None | Adapter handles FFT/iFFT; timing & loss weighting tracked separately |
| ✅ | FFT timing instrumentation | CPU/CUDA timing captured per adapter; metrics recorded | Report sampling/training breakdown in analysis scripts | Spectral utilities | Exposed as `spectral_*_time_seconds` and sampling counterparts |
| ✅ | Pipeline architecture | Training/sampling/evaluation split with run-dir layout documented; scripts & validation updated | Monitor downstream tooling and consolidate legacy aliases when safe | None | `results/runs/<run_id>/...` now hosts configs/logs/metrics/checkpoints/samples; smoke + Taguchi scripts target new paths |
| ✅ | Real training components | Diffusion training loop active (ε-pred, cosine schedule) | Next: v/x0 prediction, sampling utilities | Spectral utilities | Baseline-conv path remains for synthetic smoke tests |
| ✅ | Sampler framework | Registry live with DDPM, DDIM, DPM-Solver++; tests cover registration/fallbacks | Monitor downstream usage and add higher-order solvers as needed | Pipeline architecture | Legacy `sample_ddpm` shim retained for compatibility until downstream scripts migrate |
| ✅ | Spectral-weighted losses | Residual weighting via SpectralAdapter | Consider mixing strategies & per-frequency strength | Spectral utilities | Works with `loss.spectral_weighting` (none/radial/bandpass) |
| ✅ | Structured logging | Per-run `system.json` captures environment; CLI supports log-level flags | Monitor logging volume and extend to JSONL if needed | Logging polish | `--log-level` available on train/sample/evaluate CLIs; metadata stored in `results/runs/<run_id>/system.json` |
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