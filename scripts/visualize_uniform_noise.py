#!/usr/bin/env python
"""Visualise the uniform frequency corruption applied during the forward diffusion step."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spectral.fft_adapter import add_uniform_frequency_noise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render original image, noise component, and corrupted result for uniform frequency corruption."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to an input image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/uniform_noise_preview"),
        help="Directory where visualisations will be saved.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="Noise strength beta (default: 0.02 ~ sqrt_alpha ≈ 0.99).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _load_image(path: Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # 0..1
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
        ]
    )
    img = Image.open(path).convert("RGB")
    tensor = transform(img)
    return tensor.unsqueeze(0)  # add batch dim


def _to_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor in [-1,1] to [0,1] for saving."""
    return tensor.clamp(-1.0, 1.0).add(1.0).div(2.0)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x0 = _load_image(args.input)
    noise = torch.randn_like(x0)

    beta = args.beta
    alpha = 1.0 - beta
    sqrt_alpha = math.sqrt(alpha)
    sqrt_one_minus = math.sqrt(1.0 - alpha)

    sqrt_alpha_t = torch.tensor([sqrt_alpha], dtype=x0.dtype, device=x0.device).view(1, 1, 1, 1)
    sqrt_one_minus_t = torch.tensor([sqrt_one_minus], dtype=x0.dtype, device=x0.device).view(1, 1, 1, 1)

    x_t = add_uniform_frequency_noise(
        x0,
        noise,
        sqrt_alpha_t=sqrt_alpha_t,
        sqrt_one_minus_alpha_t=sqrt_one_minus_t,
        uniform_corruption=True,
    )

    mixed_noise = (x_t - sqrt_alpha * x0) / sqrt_one_minus

    save_image(_to_image(x0), output_dir / "input.png")
    save_image(_to_image(mixed_noise), output_dir / "noise_component.png")
    save_image(_to_image(x_t), output_dir / "corrupted.png")

    print(f"Saved visualisations to {output_dir}")


if __name__ == "__main__":
    main()
