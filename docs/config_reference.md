# Configuration & CLI Reference

This page explains the core knobs you can tweak when running Spectral Diffusion. Use it alongside the YAML configs in `configs/`.

> **Scope reminder:** The supported knobs in the main pipeline are limited to
> `snr_ratio`, `spectral_operator_mode`, sampler choice/steps, training steps,
> and image resolution. Legacy adaptive/phase/adapter options have been archived.

## 1. Training CLI (`train.py`)
```
python train.py --config configs/baseline.yaml \
                --run-id my_run \
                --output-dir results/my_runs \
                --dry-run
```
| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `configs/baseline.yaml` | YAML file describing model/data/training settings. |
| `--variant {baseline,unet_tiny}` | None | Overrides `model.type` quickly. |
| `--output-dir PATH` | `results/` | Root directory where run artefacts are stored. Run IDs create subfolders. |
| `--run-id NAME` | timestamp | Optional run name. If omitted a timestamp is used. |
| `--dry-run` | False | Skip the training loop (just create log/config scaffolding). |
| `--json-log` | False | Emit `train.jsonl` with structured log entries alongside `train.log`. |
| `--cleanup` | False | Delete artefacts after completion (useful for CI tests). |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, ...). |
| `--snr-ratio` | None | Override `diffusion.snr_ratio`/`spectral.snr_ratio` without editing YAML. |

### Key YAML fields
| Section | Fields | Notes |
|---------|--------|-------|
| `model` | `type` (`baseline`, `unet_tiny`), `base_channels`, `depth` | Baseline conv stack or TinyUNet backbone. |
| `data` | `source` (`synthetic`, `cifar10`), `height/width`, `download`, `family` (for synthetic) | Synthetic families include `piecewise`, `texture`, `random_field`, or `noise`. |
| `training` | `batch_size`, `epochs`, `num_batches`, `log_every` | Set `num_batches` to limit steps for smoke tests. |
| `diffusion` | `num_timesteps`, `beta_schedule`, `schedule_kwargs`, `logsnr`, `prediction_type`, `fft_norm`, `snr_ratio`, `spectral_operator_mode` | Standard DDPM settings plus the two unified spectral knobs consumed by `NoisePreparer`. |
| `sampling` | `enabled`, `sampler_type` (`ddpm`, `ddim`, `dpm_solver++`, `ancestral`, `dpm_solver2`) | Controls optional sampling after training. |
| `evaluation` | `reference_dir`, `use_fid`, `use_lpips` | Provide a folder of real images to compare against. |
| `spectral` | `operator_mode` (`none`, `radial`, `radial_squared`), `snr_ratio`, `operator_mask_params` | Mirrors the diffusion-level knobs if you prefer scoping them under `spectral.*`. |
| `initialization` | `strategy` (`default`, `zeros`), `scale`, `source` (`type: constant/random_normal/file`, plus `values`/`length`/`mean`/`std`/`path`) | Controls optional preset weights. |

## 2. Sampling CLI (`sample.py`)
```
python sample.py --run-dir results/runs/my_run \
                 --tag sample_grid \
                 --sampler-type dpm_solver2 \
                 --num-samples 16 \
                 --num-steps 100
```
| Flag | Default | Description |
|------|---------|-------------|
| `--run-dir PATH` | **required** | Training run directory containing `config.yaml` and `checkpoints/`. |
| `--checkpoint PATH` | latest | Choose a specific checkpoint (otherwise the latest is used). |
| `--tag NAME` | timestamp | Subfolder name under `samples/`. |
| `--sampler-type` | value from `sampling.sampler_type` | Override sampler (`ddpm`, `ddim`, `dpm_solver++`, `ancestral`, `dpm_solver2`). |
| `--num-samples` | YAML default | Override number of generated samples. |
| `--num-steps` | YAML default | Override sampling steps. |
| `--log-level` | `INFO` | Adjust logging verbosity. |

## 3. Evaluation CLI (`evaluate.py`)
```
python evaluate.py --generated-dir results/runs/my_run/samples/sample_grid \
                   --reference-dir data/cifar-10-refs \
                   --use-fid --use-lpips
```
| Flag | Default | Description |
|------|---------|-------------|
| `--generated-dir PATH` | **required** | Folder of generated images (PNG/JPG). |
| `--reference-dir PATH` | **required** | Folder of real/reference images. |
| `--image-size H W` | None | Resize before metric computation. |
| `--use-fid` | False | Compute FID via torchmetrics (requires GPU or patience). |
| `--use-lpips` | False | Compute LPIPS (perceptual similarity). |
| `--strict-filenames` | False | Require filenames to match one-to-one. |
| `--output PATH` | `generated_dir/metrics.json` | Where to write JSON summary. |
| `--update-metadata` | False | Insert metrics into `metadata.json` inside the sample folder. |
| `--log-level` | `INFO` | Verbosity. |

## 4. Automation scripts
| Script | What it does |
|--------|--------------|
| `scripts/run_smoke_report.sh` | Fast end-to-end run (synthetic + CIFAR smoke, Taguchi mini sweep, figures). Takes optional output dir (defaults to timestamped folder). |
| `scripts/run_taguchi_*.sh` | Run the tagged Taguchi scenario (smoke/minimal/comparison). |
| `python scripts/generate_report_v2.py` | Regenerate the cleaned figure set + `summary.md` from existing results. |
| `archive/legacy/scripts/run_full_report_32x32.sh` | Archived full benchmark runner (kept for reference only). |

All scripts respect `PYTHONPATH` and will create timestamped subdirectories when none are provided, keeping your `results/` clean.
