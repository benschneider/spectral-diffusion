#!/usr/bin/env python
"""Minimal training step recorder with unified SNR diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config, seed_everything
from src.core import build_model, get_loss_fn
from src.training.builders import build_dataloader, build_optimizer
from src.training.noise import NoisePreparer
from src.training.scheduler import build_diffusion, sample_timesteps
from src.training.steps import TrainingStepExecutor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a few training steps with unified SNR metrics.")
    parser.add_argument("--config", type=Path, required=True, help="Training config path")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to run")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics_run"), help="Directory to store logs")
    parser.add_argument("--device", type=str, default=None, help="Optional device override")
    parser.add_argument("--log-interval", type=int, default=5, help="Print every N steps")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    return parser.parse_args()


def _prepare(config: Dict[str, Any], device: torch.device):
    model = build_model(config.get("model", {})).to(device)
    loss_fn = get_loss_fn(config.get("loss", {}))
    optimizer = build_optimizer(model, config)
    loader = build_dataloader(config)
    diffusion_cfg = config.get("diffusion", {}) or {}
    T = int(diffusion_cfg.get("num_timesteps", 1000))
    schedule = diffusion_cfg.get("beta_schedule", "cosine")
    schedule_kwargs = dict(diffusion_cfg.get("schedule_kwargs", {}) or {})
    if schedule.replace("-", "_").lower() == "logsnr_cosine":
        logsnr_cfg = diffusion_cfg.get("logsnr", {}) or {}
        for key in ("lambda_min", "lambda_max", "delta"):
            if key in logsnr_cfg and key not in schedule_kwargs:
                schedule_kwargs[key] = float(logsnr_cfg[key])
    coeffs = build_diffusion(T, schedule, schedule_kwargs)
    noise_preparer = NoisePreparer.from_config(config)
    step_executor = TrainingStepExecutor(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        prediction_type=str(diffusion_cfg.get("prediction_type", "eps")),
        fft_norm=str(diffusion_cfg.get("fft_norm", "ortho")),
    )
    return model, loss_fn, optimizer, loader, coeffs, noise_preparer, step_executor


def _summarise_snr_spikes(
    *,
    snr_vals: torch.Tensor,
    sqrt_alpha_t: torch.Tensor,
    sqrt_one_minus_t: torch.Tensor,
    timesteps: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    noise: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    threshold: float = 1_000.0,
    top_k: int = 5,
) -> Optional[Dict[str, Any]]:
    """Summarise unusually large SNR values for debugging."""
    snr_flat = snr_vals.view(-1).detach().cpu()
    above = torch.nonzero(snr_flat > float(threshold), as_tuple=False).view(-1)
    if above.numel() == 0:
        return None

    scores = snr_flat[above]
    order = torch.argsort(scores, descending=True)
    selected = above[order][: max(1, int(top_k))]

    entries: list[dict[str, Any]] = []
    for idx in selected.tolist():
        entry = {
            "sample_index": int(idx),
            "timestep": int(timesteps.view(-1)[idx].item()),
            "snr": float(snr_flat[idx].item()),
            "sqrt_alpha": float(sqrt_alpha_t.view(-1)[idx].detach().cpu().item()),
            "sqrt_one_minus": float(sqrt_one_minus_t.view(-1)[idx].detach().cpu().item()),
        }
        entries.append(entry)

    entries.sort(key=lambda e: e["snr"], reverse=True)
    return {
        "threshold": float(threshold),
        "count": int(len(entries)),
        "max_snr": float(entries[0]["snr"]),
        "top_timesteps": [int(e["timestep"]) for e in entries],
        "entries": entries,
    }


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    config["seed"] = args.seed
    seed_everything(config)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        model,
        loss_fn,
        optimizer,
        loader,
        coeffs,
        noise_preparer,
        step_executor,
    ) = _prepare(config, device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "step_metrics.jsonl"

    records = []
    step = 0
    data_iter = iter(loader)
    while step < args.steps:
        try:
            clean, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            clean, _ = next(data_iter)
        clean = clean.to(device)
        B = clean.shape[0]
        timesteps = sample_timesteps(B, coeffs.num_timesteps, device=clean.device)
        noise_batch = noise_preparer.prepare(clean, coeffs, timesteps, base_noise=torch.randn_like(clean))

        outcome = step_executor.run_step(
            clean,
            noise_batch,
            timesteps,
            grad_callback=lambda: None,
        )

        record = {
            "step": step + 1,
            "loss": outcome.loss,
            "mae": outcome.mae,
            "snr_theory": noise_batch.stats.get("snr_theory"),
            "snr_emp": noise_batch.stats.get("snr_emp"),
            "snr_rel": noise_batch.stats.get("snr_rel"),
            "variance_sum": noise_batch.stats.get("variance_sum"),
            "grad_norm": outcome.grad_norm,
        }
        records.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")

        if (step + 1) % args.log_interval == 0:
            print(
                f"step={step+1} loss={outcome.loss:.4f} snr_rel={record['snr_rel']:.3f} variance_sum={record['variance_sum']:.4f}",
                flush=True,
            )
        step += 1

    summary = {
        "steps": len(records),
        "loss_mean": float(sum(r["loss"] for r in records) / max(len(records), 1)),
        "snr_rel_mean": float(sum(r["snr_rel"] for r in records if r["snr_rel"] is not None) / max(len(records), 1)),
        "snr_theory_mean": float(sum(r["snr_theory"] for r in records if r["snr_theory"] is not None) / max(len(records), 1)),
        "snr_emp_mean": float(sum(r["snr_emp"] for r in records if r["snr_emp"] is not None) / max(len(records), 1)),
        "variance_sum_mean": float(sum(r["variance_sum"] for r in records if r["variance_sum"] is not None) / max(len(records), 1)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Step recorder artefacts written", flush=True)


if __name__ == "__main__":
    main()
