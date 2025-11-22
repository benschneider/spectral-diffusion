#!/usr/bin/env python
"""Visualise spectral noising and reconstruction on CIFAR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

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
        "training": {"batch_size": 4, "num_batches": 1},
        "diffusion": {"num_timesteps": 1000, "beta_schedule": "cosine"},
    }


def _load_cfg(path: Path | None) -> dict:
    if path is None:
        return _default_config()
    return load_config(config_path=path)


def main() -> None:
    parser = argparse.ArgumentParser(description="FFT→IFFT reconstruction visualiser for spectral noise.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (defaults to CIFAR baseline).")
    parser.add_argument("--output-dir", type=Path, default=Path("debug_outputs"), help="Directory to save images.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on.")
    args = parser.parse_args()

    config = _load_cfg(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    noisy = batch.noisy.detach()
    recon_fft = torch.fft.ifftn(torch.fft.fftn(noisy, dim=(-2, -1), norm=preparer.fft_norm), dim=(-2, -1), norm=preparer.fft_norm).real

    save_image(xb, args.output_dir / "clean.png", normalize=True)
    save_image(noisy, args.output_dir / "noisy.png", normalize=True)
    save_image(recon_fft, args.output_dir / "recon_fft.png", normalize=True)
    print(f"Saved clean/noisy/recon images to {args.output_dir}")


if __name__ == "__main__":
    main()
