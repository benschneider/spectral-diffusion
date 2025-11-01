#!/usr/bin/env python
"""Visualise the uniform frequency corruption applied during the forward diffusion step."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.spectral.fft_adapter import add_uniform_frequency_noise
from src.training.scheduler import build_diffusion


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render original image, noise components, and corrupted results for different forward-noise modes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to an input image. If omitted, provide --cifar-index to sample from CIFAR-10.",
    )
    parser.add_argument(
        "--cifar-index",
        type=int,
        default=None,
        help="Index of CIFAR-10 training image to visualise (requires dataset downloaded).",
    )
    parser.add_argument(
        "--cifar-root",
        type=Path,
        default=ROOT / "data",
        help="Root directory containing CIFAR-10 data (default: ./data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/uniform_noise_preview"),
        help="Directory where visualisations will be saved.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["gaussian", "uniform"],
        choices=["gaussian", "uniform"],
        help="Noise modes to visualise (default: gaussian uniform).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Noise strength beta used when no --config is supplied.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional training config to match diffusion schedule (uses first spectral config fields).",
    )
    parser.add_argument(
        "--t-index",
        type=int,
        default=None,
        help="Diffusion timestep index when using --config (defaults to T//2).",
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


def _load_cifar(index: int, root: Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    try:
        dataset = datasets.CIFAR10(
            root=str(root),
            train=True,
            download=False,
            transform=transform,
        )
    except RuntimeError:
        dataset = datasets.CIFAR10(
            root=str(root),
            train=True,
            download=True,
            transform=transform,
        )
    if index < 0 or index >= len(dataset):
        raise IndexError(f"CIFAR index {index} out of range (0..{len(dataset)-1})")
    tensor, _ = dataset[index]
    return tensor.unsqueeze(0)


def _to_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor in [-1,1] to [0,1] for saving."""
    return tensor.detach().clamp(-1.0, 1.0).add(1.0).div(2.0)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        x0 = _load_image(args.input)
    elif args.cifar_index is not None:
        x0 = _load_cifar(args.cifar_index, args.cifar_root)
    else:
        raise ValueError("Provide either --input or --cifar-index.")

    noise = torch.randn_like(x0)

    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        diffusion_cfg = cfg.get("diffusion", {})
        num_steps = int(diffusion_cfg.get("num_timesteps", 1000))
        schedule = diffusion_cfg.get("beta_schedule", "cosine")
        coeffs = build_diffusion(num_steps, schedule)
        if args.t_index is not None:
            t = int(args.t_index)
        else:
            t = num_steps // 2
        t = max(0, min(num_steps - 1, t))
        sqrt_alpha = coeffs.sqrt_alphas_cumprod[t].item()
        sqrt_one_minus = coeffs.sqrt_one_minus_alphas_cumprod[t].item()
    else:
        if args.beta is None:
            raise ValueError("Provide --beta when no --config is supplied.")
        beta = args.beta
        alpha = 1.0 - beta
        sqrt_alpha = math.sqrt(alpha)
        sqrt_one_minus = math.sqrt(1.0 - alpha)

    print(
        f"Using sqrt_alpha={sqrt_alpha:.6f}, sqrt_one_minus_alpha={sqrt_one_minus:.6f}"
    )

    sqrt_alpha_t = torch.tensor([sqrt_alpha], dtype=x0.dtype, device=x0.device).view(1, 1, 1, 1)
    sqrt_one_minus_t = torch.tensor([sqrt_one_minus], dtype=x0.dtype, device=x0.device).view(1, 1, 1, 1)

    save_image(_to_image(x0), output_dir / "input.png")

    results = {}

    if "gaussian" in args.modes:
        x_gauss = sqrt_alpha_t * x0 + sqrt_one_minus_t * noise
        noise_gauss = (x_gauss - sqrt_alpha * x0) / sqrt_one_minus
        results["gaussian"] = (x_gauss, noise_gauss)
        save_image(_to_image(noise_gauss), output_dir / "noise_gaussian.png")
        save_image(_to_image(x_gauss), output_dir / "corrupted_gaussian.png")

    if "uniform" in args.modes:
        x_uniform = add_uniform_frequency_noise(
            x0,
            noise,
            sqrt_alpha_t=sqrt_alpha_t,
            sqrt_one_minus_alpha_t=sqrt_one_minus_t,
            uniform_corruption=True,
        )
        noise_uniform = (x_uniform - sqrt_alpha * x0) / sqrt_one_minus
        results["uniform"] = (x_uniform, noise_uniform)
        save_image(_to_image(noise_uniform), output_dir / "noise_uniform.png")
        save_image(_to_image(x_uniform), output_dir / "corrupted_uniform.png")

    if "gaussian" in results and "uniform" in results:
        diff = results["uniform"][1] - results["gaussian"][1]
        save_image(_to_image(diff), output_dir / "noise_difference_uniform_minus_gaussian.png")
        corr_diff = results["uniform"][0] - results["gaussian"][0]
        save_image(_to_image(corr_diff), output_dir / "corrupted_difference_uniform_minus_gaussian.png")

    height, width = x0.shape[-2], x0.shape[-1]
    fy = torch.fft.fftfreq(height, d=1.0)
    fx = torch.fft.fftfreq(width, d=1.0)
    yy = fy[:, None]
    xx = fx[None, :]
    radius = torch.sqrt(xx**2 + yy**2)
    nonzero = radius[radius > 0]
    base_radius = float(nonzero.min().item()) if nonzero.numel() > 0 else 0.0

    md_lines = [
        "## Noise Definitions",
        "",
        f"- Image resolution: {height}×{width}",
        f"- sqrt_alpha = {sqrt_alpha:.6f}",
        f"- sqrt_one_minus_alpha = {sqrt_one_minus:.6f}",
        "",
        "### Gaussian (baseline) noise",
        "- Formulation: $x_t = \\sqrt{\\alpha_t} \\, x_0 + \\sqrt{1-\\alpha_t} \\, \\varepsilon$, "
        "with $\\varepsilon \\sim \\mathcal{N}(0, I)$ sampled i.i.d. per pixel.",
        "",
        "### Uniform spectral noise",
        "- For FFT frequency indices $(k, \\ell)$, define $r(k, \\ell) = \\sqrt{(k/H)^2 + (\\ell/W)^2}$.",
        f"- Let $r_\\text{{min}} = \\min_{{r>0}} r(k, \\ell) = {base_radius:.6e}$. The mask is ",
        "  $m(k, \\ell) = \\sqrt{\\left(\\frac{r(k, \\ell)}{r_\\text{min}}\\right)^2 + 1}$ with the DC bin "
        "  $m(0,0)$ set to $1.0$.",
        "- The noise FFT is scaled by this mask prior to the inverse FFT so that higher frequencies receive "
        "  proportionally more energy while the DC component remains bounded.",
        "",
        "### Saved figures",
        "- `noise_gaussian.png`, `corrupted_gaussian.png`",
        "- `noise_uniform.png`, `corrupted_uniform.png`",
        "- `noise_difference_uniform_minus_gaussian.png`, `corrupted_difference_uniform_minus_gaussian.png`",
    ]
    (output_dir / "noise_definitions.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved visualisations to {output_dir}")


if __name__ == "__main__":
    main()
