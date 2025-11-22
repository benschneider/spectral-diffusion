#!/usr/bin/env python
"""Run a single CIFAR forward pass to catch exploding/vanishing stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config, seed_everything  # noqa: E402
from src.core import build_model, get_loss_fn  # noqa: E402
from src.core.functional import compute_target  # noqa: E402
from src.training.builders import build_dataloader  # noqa: E402
from src.training.noise import NoisePreparer  # noqa: E402
from src.training.scheduler import build_diffusion, sample_timesteps  # noqa: E402
from src.utils.debug_helpers import grad_norm  # noqa: E402


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
        "model": {"type": "unet_tiny", "channels": 3, "base_channels": 32, "depth": 2},
        "diffusion": {"num_timesteps": 1000, "beta_schedule": "cosine", "prediction_type": "eps"},
        "loss": {"reduction": "mean"},
    }


def _load_cfg(path: Path | None) -> dict:
    if path is None:
        return _default_config()
    return load_config(config_path=path)


def _check_bounds(name: str, tensor: torch.Tensor, threshold: float = 25.0) -> None:
    if not torch.isfinite(tensor).all():
        raise SystemExit(f"[SANITY] {name} contains NaN/Inf.")
    if tensor.abs().max().item() > threshold:
        raise SystemExit(f"[SANITY] {name} magnitude {tensor.abs().max().item():.2f} exceeds {threshold}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR forward+loss sanity check.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (defaults to CIFAR baseline).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for determinism.")
    args = parser.parse_args()

    config = _load_cfg(args.config)
    seed_everything(args.seed)

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
    if isinstance(loss, tuple):
        loss = loss[0]
    loss.backward()
    gnorm = grad_norm(model)

    _check_bounds("batch", xb)
    _check_bounds("noisy", noise_batch.noisy)
    _check_bounds("target", target)
    _check_bounds("prediction", pred)

    if not torch.isfinite(loss.detach()):
        raise SystemExit("[SANITY] Loss is NaN/Inf.")
    if gnorm is None or not math.isfinite(gnorm):
        raise SystemExit("[SANITY] Gradient norm invalid.")
    if gnorm > 1e3:
        raise SystemExit(f"[SANITY] Gradient norm too high ({gnorm:.1f}).")
    if gnorm < 1e-6:
        raise SystemExit(f"[SANITY] Gradient norm vanished ({gnorm:.2e}).")

    print(
        "[SANITY] ok "
        f"clean_mean={xb.mean().item():.3f} noisy_std={noise_batch.noisy.std().item():.3f} "
        f"target_std={target.std().item():.3f} grad_norm={gnorm:.3f}"
    )


if __name__ == "__main__":
    import math  # local import to keep startup fast

    main()
