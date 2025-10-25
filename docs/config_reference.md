# Configuration & CLI Reference

This page explains the core knobs you can tweak when running Spectral Diffusion. Use it alongside the YAML configs in `configs/`.

## 1. Training CLI (`train.py`)
```
python train.py --config configs/baseline.yaml \
                --run-id my_run \
                --variant spectral \
                --output-dir results/my_runs \
                --dry-run
```
| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `configs/baseline.yaml` | YAML file describing model/data/training settings. |
| `--variant {baseline,spectral}` | None | Overrides `model.type` quickly (e.g., `spectral` → `unet_spectral`). |
| `--output-dir PATH` | `results/` | Root directory where run artefacts are stored. Run IDs create subfolders. |
| `--run-id NAME` | timestamp | Optional run name. If omitted a timestamp is used. |
| `--dry-run` | False | Skip the training loop (just create log/config scaffolding). |
| `--cleanup` | False | Delete artefacts after completion (useful for CI tests). |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, ...). |

### Key YAML fields
| Section | Fields | Notes |
|---------|--------|-------|
| `model` | `type` (`baseline`, `unet_tiny`, `unet_spectral`, `unet_spectral_deep`), `base_channels`, `depth` | Spectral adapters live under `spectral.*`; the deep spectral model mirrors TinyUNet’s encoder/decoder in the frequency domain. |
| `data` | `source` (`synthetic`, `cifar10`), `height/width`, `download`, `family` (for synthetic) | Synthetic families include `piecewise`, `texture`, `random_field`, or `noise`. |
| `training` | `batch_size`, `epochs`, `num_batches`, `log_every` | Set `num_batches` to limit steps for smoke tests. |
| `diffusion` | `num_timesteps`, `beta_schedule`, `prediction_type` | Standard DDPM settings (cosine/linear schedule). |
| `sampling` | `enabled`, `sampler_type` (`ddpm`, `ddim`, `dpm_solver++`, `ancestral`, `dpm_solver2`) | Controls optional sampling after training. |
| `evaluation` | `reference_dir`, `use_fid`, `use_lpips` | Provide a folder of real images to compare against. |
| `spectral` | `enabled`, `weighting` (`none`, `radial`, `bandpass`), `apply_to` (`input`, `output`, `per_block`), `bandpass_inner`, `bandpass_outer`, `learnable`, `condition` (`time`/`none`), `mlp_hidden_dim`, `learnable_temperature`, `learnable_gain_init` | Toggles spectral adapters and, when `learnable` is true, drives the MLP-conditioned masks used by `SpectralAdapter`. |
| `initialization` | `strategy` (`default`, `zeros`, `cross_domain_flat`), `scale`, `recycle`, `source` (`type: constant/random_normal/file/gpt2`, plus `values`/`length`/`mean`/`std`/`path`), | Controls optional preset weights; `cross_domain_flat` flattens source vectors and tiles them across parameters. |

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
| `scripts/run_full_report.sh` | Longer benchmark (synthetic + CIFAR full, Taguchi) with timestamps. |
| `scripts/run_spectral_benchmark.sh` | Synthetic benchmark only (baseline script). |
| `scripts/run_taguchi_*.sh` | Run the tagged Taguchi scenario (smoke/minimal/comparison). |
| `python scripts/figures/generate_figures.py` | Regenerate figures and markdown from existing results. |

All scripts respect `PYTHONPATH` and will create timestamped subdirectories when none are provided, keeping your `results/` clean.
