#!/usr/bin/env python3
"""
Reproducible paper figure pipeline.

This script is intended to be the single, traceable entrypoint for generating the
paper figures. It:
  - launches the required training / Taguchi / sampling runs (optional),
  - composes 3–4 paper figures into a dedicated folder,
  - writes a manifest mapping every figure to exact source artifacts + commands.

The output is self-contained under:
  <out_root>/
    runs/...
    taguchi_l27/...
    paper/
      figures/
      logs/
      manifest.json
      commands.sh
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "").replace("+0000", "Z")


def _run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record: Dict[str, Any] = {
        "cmd": list(cmd),
        "cwd": str(cwd),
        "log_path": str(log_path),
        "started_at": started_at,
    }
    if dry_run:
        log_path.write_text("[dry-run] " + " ".join(cmd) + "\n", encoding="utf-8")
        record["returncode"] = None
        record["ended_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return record

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    record["returncode"] = int(proc.returncode)
    record["ended_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)} (see {log_path})")
    return record


def _git_commit(root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
        return out or None
    except Exception:
        return None


def _python_env() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _ensure_clean_dir(path: Path, *, force: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    if not force:
        return
    shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _find_one(glob_root: Path, pattern: str) -> Path:
    hits = sorted(glob_root.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No matches for {pattern} under {glob_root}")
    return hits[0]


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compose_row_images(
    image_paths: Sequence[Path],
    *,
    out_png: Path,
    out_pdf: Path,
    title: str,
    subtitles: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    if len(image_paths) != len(subtitles):
        raise ValueError("image_paths and subtitles must have same length")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=200)
    if n == 1:
        axes = [axes]
    for ax, img, subtitle in zip(axes, images, subtitles):
        ax.imshow(img)
        ax.set_title(subtitle)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_ci(
    series_by_seed: Sequence[pd.Series],
    *,
    label: str,
    ax,
    color: str,
) -> None:
    import numpy as np

    if not series_by_seed:
        return
    df = pd.concat(series_by_seed, axis=1)
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1) if df.shape[1] > 1 else df.std(axis=1, ddof=0)
    n = df.shape[1]
    ci = 1.96 * (std / max(n, 1) ** 0.5)
    x = mean.index.to_numpy()
    ax.plot(x, mean.to_numpy(), label=label, color=color, linewidth=2)
    ax.fill_between(x, (mean - ci).to_numpy(), (mean + ci).to_numpy(), color=color, alpha=0.2, linewidth=0)


def _load_training_history(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "diagnostics" / "training_history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing training history: {csv_path}")
    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError(f"training_history.csv missing 'step' column: {csv_path}")
    return df


def _make_fig2_stability(
    *,
    out_png: Path,
    out_pdf: Path,
    baseline_run_dirs: Sequence[Path],
    best_run_dirs: Sequence[Path],
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    baseline = [_load_training_history(p) for p in baseline_run_dirs]
    best = [_load_training_history(p) for p in best_run_dirs]

    def _series(dfs: Sequence[pd.DataFrame], column: str) -> List[pd.Series]:
        out: List[pd.Series] = []
        for df in dfs:
            if column not in df.columns:
                continue
            s = pd.Series(df[column].to_numpy(), index=df["step"].to_numpy())
            out.append(s)
        return out

    loss_base = _series(baseline, "loss")
    loss_best = _series(best, "loss")
    grad_base = _series(baseline, "grad_norm")
    grad_best = _series(best, "grad_norm")
    snrrel_base = _series(baseline, "snr_rel")
    snrrel_best = _series(best, "snr_rel")

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True, dpi=200)
    _plot_mean_ci(loss_base, label="Baseline (none, snr=1.0)", ax=axes[0], color="tab:blue")
    _plot_mean_ci(loss_best, label="Best (radial, snr=0.8)", ax=axes[0], color="tab:orange")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.2)

    _plot_mean_ci(grad_base, label="Baseline", ax=axes[1], color="tab:blue")
    _plot_mean_ci(grad_best, label="Best", ax=axes[1], color="tab:orange")
    axes[1].set_ylabel("Grad norm")
    axes[1].grid(True, alpha=0.2)

    _plot_mean_ci(snrrel_base, label="Baseline", ax=axes[2], color="tab:blue")
    _plot_mean_ci(snrrel_best, label="Best", ax=axes[2], color="tab:orange")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("snr_rel")
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "baseline_runs": [str(p) for p in baseline_run_dirs],
        "best_runs": [str(p) for p in best_run_dirs],
        "sources": [str(p / "diagnostics" / "training_history.csv") for p in (*baseline_run_dirs, *best_run_dirs)],
    }


def _copy_figure(src: Path, dst_png: Path, dst_pdf: Optional[Path] = None) -> None:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst_png)
    if dst_pdf is not None:
        # Best-effort: only copy a PDF if the source is a PDF. Otherwise omit.
        if src.suffix.lower() == ".pdf":
            shutil.copy(src, dst_pdf)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    cmd: List[str]
    kind: str  # train | taguchi | sample | report


def _default_out_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("paper_%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / stamp


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full paper figure pipeline.")
    parser.add_argument("--out-root", type=Path, default=None, help="Output root (default: ./runs/paper_<ts>).")
    parser.add_argument("--profile", choices=["paper", "fast"], default="paper", help="Compute budget profile.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds for Fig2 stability runs.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing paper/figures outputs.")
    parser.add_argument("--skip-train", action="store_true", help="Skip launching training runs.")
    parser.add_argument("--skip-taguchi", action="store_true", help="Skip Taguchi batch + Taguchi figure generation.")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip sampling for Fig4.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest + commands without executing.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = (args.out_root or _default_out_root()).resolve()
    paper_dir = out_root / "paper"
    figures_dir = paper_dir / "figures"
    logs_dir = paper_dir / "logs"
    status_path = paper_dir / "status.json"
    manifest_path = paper_dir / "manifest.json"

    _ensure_clean_dir(figures_dir, force=args.force)
    _ensure_clean_dir(logs_dir, force=args.force)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("OMP_NUM_THREADS", "1")

    profile = args.profile
    if profile == "fast":
        fig2_steps = 200
        eval_every = 100
        checkpoint_every = 100
        eval_num_samples = 8
        eval_sampling_steps = 50
    else:
        fig2_steps = 2000
        eval_every = 200
        checkpoint_every = 200
        eval_num_samples = 16
        eval_sampling_steps = 50

    # --- Define pipeline commands (all exact, shell reproducible) ---
    fig1_specs: List[RunSpec] = []
    for mode in ("none", "radial", "radial_squared"):
        run_id = f"fig1_{mode}_snr1_seed0"
        fig1_specs.append(
            RunSpec(
                run_id=run_id,
                kind="train",
                cmd=[
                    sys.executable,
                    str(REPO_ROOT / "train.py"),
                    "--config",
                    str(REPO_ROOT / "configs" / "smoke.yaml"),
                    "--output-dir",
                    str(out_root),
                    "--run-id",
                    run_id,
                    "--seed",
                    "0",
                    "--train-steps",
                    "1",
                    "--spectral-operator-mode",
                    mode,
                    "--snr-ratio",
                    "1.0",
                    "--json-log",
                    "--log-level",
                    "INFO",
                ],
            )
        )

    fig2_specs: List[RunSpec] = []
    for seed in args.seeds:
        fig2_specs.append(
            RunSpec(
                run_id=f"fig2_base_seed{seed}",
                kind="train",
                cmd=[
                    sys.executable,
                    str(REPO_ROOT / "train.py"),
                    "--config",
                    str(REPO_ROOT / "configs" / "benchmark_spectral_cifar.yaml"),
                    "--output-dir",
                    str(out_root),
                    "--run-id",
                    f"fig2_base_seed{seed}",
                    "--seed",
                    str(seed),
                    "--train-steps",
                    str(fig2_steps),
                    "--checkpoint-every",
                    str(checkpoint_every),
                    "--eval-every",
                    str(eval_every),
                    "--eval-num-samples",
                    str(eval_num_samples),
                    "--eval-sampling-steps",
                    str(eval_sampling_steps),
                    "--eval-seed",
                    "10000",
                    "--spectral-operator-mode",
                    "none",
                    "--snr-ratio",
                    "1.0",
                    "--json-log",
                    "--log-level",
                    "INFO",
                ],
            )
        )
        fig2_specs.append(
            RunSpec(
                run_id=f"fig2_best_seed{seed}",
                kind="train",
                cmd=[
                    sys.executable,
                    str(REPO_ROOT / "train.py"),
                    "--config",
                    str(REPO_ROOT / "configs" / "benchmark_spectral_cifar.yaml"),
                    "--output-dir",
                    str(out_root),
                    "--run-id",
                    f"fig2_best_seed{seed}",
                    "--seed",
                    str(seed),
                    "--train-steps",
                    str(fig2_steps),
                    "--checkpoint-every",
                    str(checkpoint_every),
                    "--eval-every",
                    str(eval_every),
                    "--eval-num-samples",
                    str(eval_num_samples),
                    "--eval-sampling-steps",
                    str(eval_sampling_steps),
                    "--eval-seed",
                    "10000",
                    "--spectral-operator-mode",
                    "radial",
                    "--snr-ratio",
                    "0.8",
                    "--json-log",
                    "--log-level",
                    "INFO",
                ],
            )
        )

    taguchi_dir = out_root / "taguchi_l27"
    taguchi_specs: List[RunSpec] = [
        RunSpec(
            run_id="taguchi_l27",
            kind="taguchi",
            cmd=[
                sys.executable,
                "-m",
                "src.experiments.run_experiment",
                "--config",
                str(REPO_ROOT / "configs" / "taguchi_smoke_base.yaml"),
                "--array",
                str(REPO_ROOT / "configs" / "taguchi" / "L27_extended.csv"),
                "--factor-registry",
                str(REPO_ROOT / "configs" / "taguchi" / "factor_registry.yaml"),
                "--output-dir",
                str(taguchi_dir),
                "--report-metric",
                "loss_drop_per_second",
                "--report-mode",
                "larger",
                "--log-level",
                "INFO",
            ],
        ),
        RunSpec(
            run_id="taguchi_report_v2",
            kind="report",
            cmd=[
                sys.executable,
                str(REPO_ROOT / "scripts" / "generate_report_v2.py"),
                "--taguchi-dir",
                str(taguchi_dir),
                "--output-dir",
                str(out_root / "taguchi_figures"),
                "--profile",
                "snr",
            ],
        ),
    ]

    sampling_specs: List[RunSpec] = [
        RunSpec(
            run_id="fig4_samples_base",
            kind="sample",
            cmd=[
                sys.executable,
                str(REPO_ROOT / "sample.py"),
                "--run-dir",
                str(out_root / "runs" / "fig2_base_seed0"),
                "--tag",
                "paper_fig4",
                "--sampler-type",
                "ddim",
                "--num-samples",
                "64",
                "--sampling-steps",
                "50",
                "--log-level",
                "INFO",
            ],
        ),
        RunSpec(
            run_id="fig4_samples_best",
            kind="sample",
            cmd=[
                sys.executable,
                str(REPO_ROOT / "sample.py"),
                "--run-dir",
                str(out_root / "runs" / "fig2_best_seed0"),
                "--tag",
                "paper_fig4",
                "--sampler-type",
                "ddim",
                "--num-samples",
                "64",
                "--sampling-steps",
                "50",
                "--log-level",
                "INFO",
            ],
        ),
    ]

    all_specs: List[RunSpec] = []
    all_specs.extend(fig1_specs)
    all_specs.extend(fig2_specs)
    all_specs.extend(taguchi_specs)
    all_specs.extend(sampling_specs)

    commands_sh = paper_dir / "commands.sh"
    commands_sh.parent.mkdir(parents=True, exist_ok=True)
    commands_sh.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"export PYTHONPATH=\"{REPO_ROOT}\"",
                "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
                "",
                *((" ".join(spec.cmd)) for spec in all_specs),
                "",
            ]
        ),
        encoding="utf-8",
    )
    commands_sh.chmod(0o755)

    # --- Execute pipeline ---
    executed: List[Dict[str, Any]] = []

    def _write_status(state: str, *, current: Optional[RunSpec] = None) -> None:
        payload = {
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "current_run_id": current.run_id if current else None,
            "current_kind": current.kind if current else None,
            "current_cmd": current.cmd if current else None,
            "out_root": str(out_root),
        }
        _save_json(status_path, payload)

    figure_map: Dict[str, Dict[str, Any]] = {}

    def _write_manifest() -> None:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo_root": str(REPO_ROOT),
            "out_root": str(out_root),
            "git_commit": _git_commit(REPO_ROOT),
            "env": _python_env(),
            "profile": profile,
            "seeds": list(args.seeds),
            "commands_sh": str(commands_sh),
            "executed": executed,
            "figure_map": figure_map,
            "notes": {
                "controlled_forward_knobs": [
                    "diffusion.snr_ratio",
                    "diffusion.spectral_operator_mode",
                ],
                "sampler_locked_for_fig4": {"sampler_type": "ddim", "sampling_steps": 50, "num_samples": 64},
            },
        }
        _save_json(manifest_path, payload)

    _write_status("starting")
    _write_manifest()

    if not args.skip_train:
        for spec in (*fig1_specs, *fig2_specs):
            log_path = logs_dir / f"{spec.run_id}.log"
            print(f"[paper] {spec.kind}: {spec.run_id}", flush=True)
            _write_status("running", current=spec)
            executed.append(_run_cmd(spec.cmd, cwd=REPO_ROOT, env=env, log_path=log_path, dry_run=args.dry_run))
            _write_manifest()
    if not args.skip_taguchi:
        for spec in taguchi_specs:
            log_path = logs_dir / f"{spec.run_id}.log"
            print(f"[paper] {spec.kind}: {spec.run_id}", flush=True)
            _write_status("running", current=spec)
            executed.append(_run_cmd(spec.cmd, cwd=REPO_ROOT, env=env, log_path=log_path, dry_run=args.dry_run))
            _write_manifest()
    if not args.skip_sampling:
        for spec in sampling_specs:
            log_path = logs_dir / f"{spec.run_id}.log"
            print(f"[paper] {spec.kind}: {spec.run_id}", flush=True)
            _write_status("running", current=spec)
            executed.append(_run_cmd(spec.cmd, cwd=REPO_ROOT, env=env, log_path=log_path, dry_run=args.dry_run))
            _write_manifest()

    # --- Compose figures from artifacts ---
    if not args.dry_run:
        _write_status("composing_figures")
        # Fig 1: noise spectra sanity (eps FFT mags) for operator modes.
        fig1_srcs = []
        fig1_titles = ["none", "radial", "radial_squared"]
        for spec in fig1_specs:
            fig1_srcs.append(_find_one(out_root / "sanity", f"{spec.run_id}_eps_sanity_*_fft_mag.png"))
        fig1_png = figures_dir / "fig1_method.png"
        fig1_pdf = figures_dir / "fig1_method.pdf"
        _compose_row_images(
            fig1_srcs,
            out_png=fig1_png,
            out_pdf=fig1_pdf,
            title="Injected noise spectrum (FFT magnitude)",
            subtitles=fig1_titles,
        )
        figure_map["fig1_method"] = {"png": str(fig1_png), "pdf": str(fig1_pdf), "sources": [str(p) for p in fig1_srcs]}

        # Fig 2: stability + sample-efficiency across seeds.
        baseline_dirs = [out_root / "runs" / f"fig2_base_seed{s}" for s in args.seeds]
        best_dirs = [out_root / "runs" / f"fig2_best_seed{s}" for s in args.seeds]
        fig2_png = figures_dir / "fig2_stability.png"
        fig2_pdf = figures_dir / "fig2_stability.pdf"
        fig2_meta = _make_fig2_stability(
            out_png=fig2_png,
            out_pdf=fig2_pdf,
            baseline_run_dirs=baseline_dirs,
            best_run_dirs=best_dirs,
        )
        figure_map["fig2_stability"] = {"png": str(fig2_png), "pdf": str(fig2_pdf), **fig2_meta}

        # Fig 3: Taguchi main effects (re-use report_v2 output).
        taguchi_img = out_root / "taguchi_figures" / "images" / "taguchi_main_effects_primary.png"
        fig3_png = figures_dir / "fig3_taguchi.png"
        _copy_figure(taguchi_img, fig3_png)
        figure_map["fig3_taguchi"] = {"png": str(fig3_png), "sources": [str(taguchi_img), str(taguchi_dir / "summary.csv"), str(taguchi_dir / "taguchi_report.csv")]}

        # Fig 4: qualitative samples (side-by-side grids).
        base_grid = out_root / "runs" / "fig2_base_seed0" / "samples" / "paper_fig4" / "grid.png"
        best_grid = out_root / "runs" / "fig2_best_seed0" / "samples" / "paper_fig4" / "grid.png"
        fig4_png = figures_dir / "fig4_samples.png"
        fig4_pdf = figures_dir / "fig4_samples.pdf"
        _compose_row_images(
            [base_grid, best_grid],
            out_png=fig4_png,
            out_pdf=fig4_pdf,
            title="Qualitative samples (matched sampler settings)",
            subtitles=["Baseline", "Best"],
        )
        figure_map["fig4_samples"] = {"png": str(fig4_png), "pdf": str(fig4_pdf), "sources": [str(base_grid), str(best_grid)]}

    _write_status("done")
    _write_manifest()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
