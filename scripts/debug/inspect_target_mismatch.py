#!/usr/bin/env python
"""Compare model predictions vs diffusion targets on a CIFAR batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config  # noqa: E402
from src.core import build_model, get_loss_fn  # noqa: E402
from src.core.functional import compute_target  # noqa: E402
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
        "model": {"type": "unet_tiny", "channels": 3, "base_channels": 32, "depth": 2},
        "diffusion": {"num_timesteps": 1000, "beta_schedule": "cosine", "prediction_type": "eps"},
        "loss": {"reduction": "mean"},
    }


def _load_cfg(path: Path | None) -> dict:
    if path is None:
        return _default_config()
    return load_config(config_path=path)


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.view(x.shape[0], -1)
    y_flat = y.view(y.shape[0], -1)
    x_center = x_flat - x_flat.mean(dim=1, keepdim=True)
    y_center = y_flat - y_flat.mean(dim=1, keepdim=True)
    num = (x_center * y_center).sum(dim=1)
    denom = torch.sqrt((x_center.pow(2).sum(dim=1) * y_center.pow(2).sum(dim=1)) + 1e-8)
    return float((num / denom).mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check target/prediction statistics for CIFAR.")
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
    prediction_type = diffusion_cfg.get("prediction_type", "eps")
    coeffs = build_diffusion(T, schedule, diffusion_cfg.get("schedule_kwargs"))

    preparer = NoisePreparer.from_config(config)
    timesteps = sample_timesteps(xb.shape[0], coeffs.num_timesteps, xb.device)
    noise_batch = preparer.prepare(xb, coeffs, timesteps)

    model = build_model(config.get("model", {}))
    model.to(device)
    loss_fn = get_loss_fn(config.get("loss", {}))

    pred = model(noise_batch.noisy, timesteps)
    target = compute_target(
        prediction_type,
        xb,
        noise_batch.noisy,
        noise_batch.eps,
        noise_batch.sqrt_alpha_t,
        noise_batch.sqrt_one_minus_alpha_t,
    )
    loss = loss_fn(pred, target, noise_batch.sqrt_alpha_t, noise_batch.sqrt_one_minus_alpha_t)
    loss_value = loss[0] if isinstance(loss, tuple) else loss

    print(
        f"pred: mean={pred.mean().item():.4f} std={pred.std().item():.4f} "
        f"target: mean={target.mean().item():.4f} std={target.std().item():.4f}"
    )
    print(f"correlation(pred,target)={_corr(pred.detach(), target.detach()):.4f}")
    print(f"loss={float(loss_value):.5f}")


if __name__ == "__main__":
    main()
