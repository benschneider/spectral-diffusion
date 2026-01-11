#!/usr/bin/env python3
"""
Inspect a single training "circuit" at a fixed timestep.

This is a debugging/verification tool to answer:
  - does the model predict eps (or x0/v) correctly on real data?
  - does pred_x0 visually move xt back toward x0?

It loads a run config + checkpoint, constructs xt at a chosen timestep using the
same NoisePreparer used in training, then writes image grids and scalar errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config, seed_everything  # noqa: E402
from src.training.noise import NoisePreparer  # noqa: E402
from src.training.builders import build_dataloader  # noqa: E402
from src.training.scheduler import build_diffusion  # noqa: E402
from src.training.pipeline import TrainingPipeline  # noqa: E402


def _latest_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    candidates = list(ckpt_dir.glob("checkpoint_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    pat = re.compile(r"checkpoint_step_(\d+)\.pt$")
    best = max(candidates, key=lambda p: int(pat.search(p.name).group(1)) if pat.search(p.name) else -1)
    if not pat.search(best.name):
        raise ValueError(f"Unable to infer checkpoint steps under {ckpt_dir}")
    return best


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect a single denoising step (x0 -> xt -> pred_x0) at a fixed timestep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "run_dir",
        type=Path,
        help="Training run directory containing config.yaml and checkpoints/.",
    )
    ckpt_group = p.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint path (overrides --ckpt-step).",
    )
    ckpt_group.add_argument(
        "--ckpt-step",
        type=int,
        default=None,
        help="Checkpoint step to load (e.g. 2000 selects checkpoints/checkpoint_step_2000.pt).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <run_dir>/inspect/ckpt<step>_t<t>/).",
    )
    p.add_argument("-s", "--seed", type=int, default=0, help="Seed for noise generation.")
    p.add_argument("-n", "--num", type=int, default=16, help="Number of batch samples to inspect.")
    p.add_argument(
        "-t",
        "--timestep",
        type=int,
        default=None,
        help="Diffusion timestep index (default: T//2).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu/cuda).",
    )
    return p.parse_args(argv)


def inspect_step(
    *,
    run_dir: Path,
    checkpoint: Optional[Path] = None,
    ckpt_step: Optional[int] = None,
    out_dir: Optional[Path] = None,
    seed: int = 0,
    num: int = 16,
    timestep: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")
    config: Dict[str, Any] = load_config(config_path)
    config["seed"] = int(seed)
    seed_everything(config)

    if checkpoint is not None:
        ckpt = checkpoint.resolve()
    elif ckpt_step is not None:
        ckpt = (run_dir / "checkpoints" / f"checkpoint_step_{int(ckpt_step)}.pt").resolve()
    else:
        ckpt = _latest_checkpoint(run_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    t_override = timestep
    # Determine timestep for default out-dir naming.
    t_name = f"{int(t_override)}" if t_override is not None else "mid"
    step_match = re.search(r"checkpoint_step_(\d+)\.pt$", ckpt.name)
    ckpt_step = step_match.group(1) if step_match else "unknown"
    default_out = run_dir / "inspect" / f"ckpt{ckpt_step}_t{t_name}"
    out_dir = (out_dir or default_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model via TrainingPipeline to reuse the same construction.
    pipeline = TrainingPipeline(config=config, work_dir=out_dir)
    pipeline.load_checkpoint(ckpt)

    device = None
    if device:
        device = torch.device(device)
    else:
        device = pipeline.device

    loader = build_dataloader(config)
    xb, _ = next(iter(loader))
    xb = xb.to(device)
    num = max(1, min(int(num), int(xb.shape[0])))
    xb = xb[:num]

    diffusion_cfg = config.get("diffusion", {}) or {}
    prediction_type = str(diffusion_cfg.get("prediction_type", "eps")).lower()

    T, schedule = pipeline._diffusion_params()  # type: ignore
    coeffs = build_diffusion(T, schedule)
    total_timesteps = coeffs.num_timesteps
    t = int(timestep) if timestep is not None else int(total_timesteps // 2)
    if not 0 <= t < total_timesteps:
        raise ValueError(f"timestep must be within [0,{total_timesteps-1}], got {t}")

    noise_preparer = NoisePreparer.from_config(config)
    t_batch = torch.full((num,), t, device=device, dtype=torch.long)
    noise_batch = noise_preparer.prepare(xb, coeffs, t_batch, base_noise=torch.randn_like(xb))

    with torch.no_grad():
        pred = pipeline.model(noise_batch.noisy, t_batch)

    if prediction_type == "eps":
        pred_eps = pred
        true_eps = noise_batch.eps
        pred_x0 = (noise_batch.noisy - noise_batch.sqrt_one_minus_alpha_t * pred_eps) / noise_batch.sqrt_alpha_t.clamp_min(1e-12)
    elif prediction_type == "x0":
        pred_x0 = pred
        pred_eps = (noise_batch.noisy - noise_batch.sqrt_alpha_t * pred_x0) / noise_batch.sqrt_one_minus_alpha_t.clamp_min(1e-12)
        true_eps = noise_batch.eps
    elif prediction_type == "v":
        v = pred
        pred_x0 = noise_batch.sqrt_alpha_t * noise_batch.noisy - noise_batch.sqrt_one_minus_alpha_t * v
        pred_eps = noise_batch.sqrt_one_minus_alpha_t * noise_batch.noisy + noise_batch.sqrt_alpha_t * v
        true_eps = noise_batch.eps
    else:
        raise ValueError(f"Unsupported prediction_type '{prediction_type}'")

    pred_x0 = pred_x0.clamp(-1.0, 1.0)

    nrow = max(1, int(num ** 0.5))
    grids = {
        "x0": xb.detach().cpu(),
        "xt": noise_batch.noisy.detach().cpu(),
        "pred_x0": pred_x0.detach().cpu(),
        "eps_true": true_eps.detach().cpu(),
        "eps_pred": pred_eps.detach().cpu(),
    }
    for name, tensor in grids.items():
        save_image(
            tensor,
            out_dir / f"{name}.png",
            normalize=True,
            value_range=(-1, 1),
            nrow=nrow,
        )

    mse_eps = float((pred_eps.detach() - true_eps.detach()).pow(2).mean().cpu().item())
    mse_x0 = float((pred_x0.detach() - xb.detach()).pow(2).mean().cpu().item())

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "prediction_type": prediction_type,
        "timestep": int(t),
        "num_samples": int(num),
        "snr_ratio": float(diffusion_cfg.get("snr_ratio", 1.0)),
        "spectral_operator_mode": str(diffusion_cfg.get("spectral_operator_mode", "none")),
        "mse_eps": mse_eps,
        "mse_x0": mse_x0,
        "outputs": {k: str(out_dir / f"{k}.png") for k in grids},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    summary = inspect_step(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        ckpt_step=args.ckpt_step,
        out_dir=args.out,
        seed=args.seed,
        num=args.num,
        timestep=args.timestep,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
