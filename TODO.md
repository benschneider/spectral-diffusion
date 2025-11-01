# Spectral Diffusion TODOs

Legend: ✅ complete · 🟡 in progress · ⬜ pending

| Status | Area | Current Status | Immediate Next Step | Dependency | Notes / Implementation Tip |
| - | - | - | - | - | - |
| ✅ | Sampler support (DDIM/DPM-Solver++) | Added ancestral & second-order DPM-Solver samplers; registry extensible | Compare new samplers in benchmarks, consider higher-order refinements | Sampler framework | Necessary for fair spectral comparisons in arrays |
| ✅ | Spectral model research | SpectralUNet + SpectralUNetDeep integrated; smoke report trains all variants | Analyse throughput/quality trade-offs; tune spectral hyperparameters | Spectral utilities | Compare loss/runtime metrics; consider spectral regularisation |
| ✅ | Taguchi S/N analysis | CLI auto-generates reports; scripts consume them | Build notebooks/dashboards to visualise factor rankings | Taguchi runner outputs | `taguchi_report.csv` now includes runtime/throughput/final-loss columns |
| ✅ | FFT timing instrumentation | CPU/CUDA timing captured per adapter; metrics recorded | Report sampling/training breakdown in analysis scripts | Spectral utilities | Exposed as `spectral_*_time_seconds` and sampling counterparts |
| ✅ | Pipeline architecture | Training/sampling/evaluation split with timestamped run layout | Monitor downstream tooling and consolidate legacy aliases when safe | None | Smoke + Taguchi scripts target new paths and auto-embed reporting |
| ✅ | Real training components | Diffusion training loop active (ε-pred, cosine schedule) | Next: v/x0 prediction, sampling utilities | Spectral utilities | Baseline-conv path remains for synthetic smoke tests |
| ✅ | Sampler framework | Registry live with DDPM, DDIM, DPM-Solver++; tests cover registration/fallbacks | Monitor downstream usage and add higher-order solvers as needed | Pipeline architecture | Legacy `sample_ddpm` shim retained for compatibility until downstream scripts migrate |
| ✅ | Spectral-weighted losses | Residual weighting via SpectralAdapter | Consider mixing strategies & per-frequency strength | Spectral utilities | Works with `loss.spectral_weighting` (none/radial/bandpass) |
| ✅ | Structured logging | Per-run `system.json` captures environment; CLI supports log-level flags | Monitor logging volume and extend to JSONL if needed | Logging polish | `--log-level` available on train/sample/evaluate CLIs; metadata stored in run dirs |
| ✅ | Taguchi runner outputs | Per-run configs/metrics persisted | Next: S/N analysis & factor reporting | Metrics availability | Artifacts mirror single-run structure |
| ✅ | Spectral toggle ablation | Full-report scripts train spectral plain vs uniform+ARE/PCM+MASF and render comparison figure | Extend ablations to additional datasets / toggle combos | Report pipeline | Produces `ablation/summary.csv` + `spectral_feature_ablation.png` for quick visual checks |
| 🟡 | Testing / CI | Added baseline regression test + broad pytest suite; manual “Run Full Report” workflow available | Integrate into CI workflow (cached datasets, smoke report) | Pipeline architecture | Keep CPU-only path fast; cache CIFAR for CI; manual workflow uploads latest report artifact |
| 🟡 | Diffusion sampling & image metrics | DDPM/DDIM/DPM-Solver++ samplers callable; evaluation CLI writes FID/LPIPS | Surface LPIPS/FID in reports; explore batch inference tooling | Real training components | Samples live under `results/runs/<run_id>/samples/<tag>/` |
| 🟡 | Documentation | README & docs updated (architecture, config reference) | Author walkthrough notebook + CONTRIBUTING guide | None | Highlight learnable adapters + SpectralUNetDeep usage |
| 🟡 | Spectral utilities | Learnable adapters + cross-domain init integrated | Explore adapter strength annealing / learnt masks analytics | None | Adapter stats tracked via `spectral_*` metrics |
| 🟡 | Taguchi extensions | New factor `E` toggles cross-domain init; reports include runtime/throughput | Add per-factor notebook + scenario templates | Taguchi runner outputs | Consider expanding to cover learnable adapter hyperparameters |
| 🟡 | Taguchi L16 expansion | Drafted factor plan covering 15 binary toggles (see docs/taguchi_factor_plan.md) | Implement L16 design (CSV + runner mutations + report update) | Taguchi runner outputs | Add ARE/MASF/warm-up/LR schedule, etc. to the design matrix |
| 🟡 | Logging polish | Console logging ready; optional JSONL logs emitted with `--json-log` | Expand structured payloads (metrics snapshots, step logs) | Independent | JSONL lives at `logs/train.jsonl`; next: richer diagnostics |
| ⬜ | Dataset handling | Manual CIFAR download documented | Support auto-download flag + checksum validation | Network availability | Document dataset caching for CI/local |
| ⬜ | Analysis notebooks | Not started | Plot loss vs time, FFT overhead vs efficiency, Taguchi summaries | Metrics & S/N tooling | Consume `results/summary.csv`, `taguchi_report.csv` |
| ⬜ | Deep spectral evaluation | SpectralUNetDeep in smoke report; needs dedicated benchmark sweeps | Run extended training + compare vs TinyUNet | Spectral deep configs | Track spectral energy per scale |

**Execution Order (new focus)**
1. Finalise CI harness (baseline equivalence, cached datasets)  
2. Surface LPIPS/FID in automated reports  
3. Deep spectral UNet benchmark sweeps + analytics  
4. Adapter analytics (strength annealing, learnt filter inspection)  
5. Documentation walkthrough + CONTRIBUTING draft  
6. Logging polish (structured logs)  
7. Dataset handling automation  
8. Analysis notebooks  
