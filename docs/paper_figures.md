# Paper Figure Pipeline

This repo includes a single entrypoint script that can generate the paper figures in a fully traceable way (commands, inputs, and source artifacts are recorded).

## One-command pipeline

Run from the repo root:

```bash
python scripts/make_paper_figures.py --out-root ./runs/paper_run_001
```

This produces:

- `./runs/paper_run_001/paper/figures/fig1_method.png` (+ `.pdf`)
- `./runs/paper_run_001/paper/figures/fig2_stability.png` (+ `.pdf`)
- `./runs/paper_run_001/paper/figures/fig3_taguchi.png`
- `./runs/paper_run_001/paper/figures/fig4_samples.png` (+ `.pdf`)
- `./runs/paper_run_001/paper/manifest.json` (machine-readable provenance)
- `./runs/paper_run_001/paper/commands.sh` (exact shell commands, runnable)
- `./runs/paper_run_001/paper/logs/*.log` (stdout/stderr per stage)

## Profiles

- `--profile paper`: larger budget (default).
- `--profile fast`: reduced steps for quick end-to-end validation.

## Reproducibility / provenance

The manifest (`paper/manifest.json`) records:

- output root + timestamp
- git commit hash (if available)
- python/platform info
- all commands executed (or planned in `--dry-run`)
- a figure → source-file mapping for each generated figure

## Safe preview

To generate the manifest/commands without running training:

```bash
python scripts/make_paper_figures.py --out-root ./runs/paper_dry_run --dry-run --force --profile fast
```

