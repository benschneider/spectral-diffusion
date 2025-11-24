# Spectral Diffusion (Unified Forward Process)

Spectral Diffusion now ships a single, unified noise path: every forward step shapes Gaussian noise with `spectral_operator(mode)` (centered, unit RMS) and scales it by `k = 1 / snr_ratio`. Models are limited to `baseline` and `unet_tiny`; all legacy adapters, phase hooks, and bespoke schedulers live under `archive/legacy/`.

## Quickstart
```bash
git clone https://github.com/benschneider/spectral-diffusion.git
cd spectral-diffusion
pip install -r requirements.txt

# Train TinyUNet on CIFAR-10 with default knobs
python train.py --config configs/baseline.yaml --run-id quickstart

# Sample a few images from the run
python sample.py --run-dir results/runs/quickstart --sampler-type dpm_solver++ --num-samples 8 --num-steps 50

# Evaluate against a reference folder (FID/LPIPS optional)
python evaluate.py --generated-dir results/runs/quickstart/samples/quickstart_solver --reference-dir path/to/reference --use-lpips
```

## Active Knobs (Taguchi + CLI)
- `snr_ratio` (k = 1 / snr_ratio)
- `spectral_operator_mode` (`none`, `radial`, `radial_squared`)
- `sampler_type` (`ddpm`, `ddim`, `dpm_solver++`, `ancestral`, `dpm_solver2`)
- `sampling_steps`
- `train_steps` (num_batches)
- `image_resolution`

Taguchi factors mirror these knobs exactly. The canonical OA is `configs/taguchi/L27_extended.csv`; `configs/taguchi/factor_registry.yaml` enumerates the levels.

## Workflows
- Train & sample: `python train.py ...` then `python sample.py ...`
- Discover configs: `python -m src.cli.list_configs [--include-csv --filter baseline]`
- Taguchi sweeps: `scripts/run_taguchi_smoke.sh`, `scripts/run_taguchi_minimal.sh`, `scripts/run_taguchi_comparison.sh`
- Diagnostics only: `python scripts/debug/record_training_steps.py --config <yaml> --steps 20`
- Reporting: `python scripts/generate_report_v2.py [--report-root <path>]`
- Archived full pipeline: `archive/legacy/scripts/run_full_report_32x32.sh`

## Forward/Noise Invariants
- `spectral_operator` enforces centered, unit-RMS noise.
- Noise injection uses a single scale `k = 1 / snr_ratio`; no extra clamps or adaptive governors.
- Logged noise stats: `snr_theory`, `snr_emp`, `snr_rel`, `variance_sum`, `noise_channel_std_min/max`, `grad_norm`.
- Tests cover RMS(eps_shaped) = 1, `Var(signal)+Var(noise)=1`, snr_rel≈1 for `mode=none`, monotone `snr_theory`, and Taguchi mapping integrity.

## Documentation
- docs/snr_audit.md
- docs/frequency_snr_study.md
- docs/config_reference.md
- docs/refactor_todolist.md
- Legacy material: docs/legacy/*

## License
MIT License © 2025 Ben Schneider
