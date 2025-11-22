#!/usr/bin/env python
"""Measure expected vs measured SNR for a single CIFAR batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config  # noqa: E402
from src.core.numeric import compute_snr  # noqa: E402
from src.core.snr_scheduler import measure_batch_snr  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Inspect SNR scaling behaviour.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (defaults to CIFAR baseline).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on.")
    args = parser.parse_args()

    config = _load_cfg(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    loader = build_dataloader(config)
    xb, _ = next(iter(loader))
    xb = xb.to(device)

    diffusion_cfg = config.get("diffusion", {}) or {}
    T = int(diffusion_cfg.get("num_timesteps", 1000))
    schedule = diffusion_cfg.get("beta_schedule", "cosine")
    coeffs = build_diffusion(T, schedule, diffusion_cfg.get("schedule_kwargs"))

    preparer = NoisePreparer.from_config(config)
    timesteps = sample_timesteps(xb.shape[0], coeffs.num_timesteps, xb.device)
    batch = preparer.prepare(xb, coeffs, timesteps)

    snr_expected = compute_snr(batch.sqrt_alpha_t, batch.sqrt_one_minus_alpha_t).mean().item()
    measured = measure_batch_snr(xb, batch.noisy, batch.sqrt_alpha_t).snr_measured.mean().item()
    snr_stats = {k: v for k, v in (batch.stats or {}).items() if k.startswith("snr")}

    print(
        f"snr_expected(schedule)={snr_expected:.3f} "
        f"snr_measured(batch)={measured:.3f} timesteps=[{int(timesteps.min()):d},{int(timesteps.max()):d}]"
    )
    if snr_stats:
        print("snr_stats:", snr_stats)


if __name__ == "__main__":
    main()
