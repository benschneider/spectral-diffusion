#!/usr/bin/env python
"""Quick CIFAR batch statistics before/after spectral noising."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config  # noqa: E402
from src.training.builders import build_dataloader  # noqa: E402
from src.training.noise import NoisePreparer  # noqa: E402
from src.training.scheduler import build_diffusion, sample_timesteps  # noqa: E402


def _default_config() -> dict:
    return {
        "data": {
            "source": "cifar10",
            "root": "data",
            "height": 32,
            "width": 32,
            "channels": 3,
            "download": True,
        },
        "training": {"batch_size": 8, "num_batches": 1},
        "diffusion": {"num_timesteps": 1000, "beta_schedule": "cosine"},
    }


def _load_cfg(path: Path | None) -> dict:
    if path is None:
        return _default_config()
    return load_config(config_path=path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a CIFAR batch before/after spectral noise.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML to reuse (defaults to CIFAR baseline).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda).")
    args = parser.parse_args()

    config = _load_cfg(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    loader = build_dataloader(config)
    xb, _ = next(iter(loader))
    xb = xb.to(device)

    def _stats(t: torch.Tensor) -> dict:
        return {
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
        }

    clean_stats = _stats(xb)
    print("[clean] ", clean_stats)

    diffusion_cfg = config.get("diffusion", {}) or {}
    T = int(diffusion_cfg.get("num_timesteps", 1000))
    schedule = diffusion_cfg.get("beta_schedule", "cosine")
    coeffs = build_diffusion(T, schedule, diffusion_cfg.get("schedule_kwargs"))

    preparer = NoisePreparer.from_config(config)
    timesteps = sample_timesteps(xb.shape[0], coeffs.num_timesteps, xb.device)
    noise_batch = preparer.prepare(xb, coeffs, timesteps)

    noisy_stats = _stats(noise_batch.noisy)
    eps_stats = _stats(noise_batch.eps)
    print("[noisy] ", noisy_stats)
    print("[eps]   ", eps_stats)
    if noise_batch.stats:
        observed = {k: v for k, v in noise_batch.stats.items() if isinstance(v, (int, float))}
        print("[stats] ", observed)


if __name__ == "__main__":
    main()
