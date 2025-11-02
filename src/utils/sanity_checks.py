from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torchvision.utils import save_image


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def check_fft_sanity(
    inputs: torch.Tensor,
    dataset_name: str,
    out_dir: Path,
    prefix: str = "",
) -> Path:
    """
    Record simple statistics about the incoming batch and verify FFT reconstruction.

    Args:
        inputs: Tensor shaped (B, C, H, W)
        dataset_name: Human-readable dataset identifier (e.g., "cifar10").
        out_dir: Directory where artifacts should be written.
        prefix: Optional prefix for filenames, used to avoid collisions.

    Returns:
        Path to the JSON summary file.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    tag = dataset_name.lower().replace(" ", "_")
    if prefix:
        prefix = prefix.rstrip("_") + "_"

    stats = {
        "mean": float(inputs.mean().detach().cpu()),
        "std": float(inputs.std().detach().cpu()),
        "is_complex": bool(inputs.is_complex()),
        "fft_reconstruction_error": None,
    }

    if not inputs.is_complex():
        fft = torch.fft.fft2(inputs, norm="ortho")
        recon = torch.fft.ifft2(fft, norm="ortho").real
        stats["fft_reconstruction_error"] = float(
            (inputs - recon).abs().mean().detach().cpu()
        )

        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.log1p(fft_shift.abs()).mean(dim=1, keepdim=True)
        max_val = magnitude.max().clamp(min=1e-6)
        norm_mag = (magnitude / max_val).clamp(0.0, 1.0)
        save_image(
            norm_mag,
            out_dir / f"{prefix}sanity_{tag}_fft_mag.png",
            normalize=False,
        )

    spatial_slice = inputs[: min(4, inputs.shape[0])].detach().cpu()
    save_image(
        spatial_slice,
        out_dir / f"{prefix}sanity_{tag}_spatial.png",
        normalize=True,
        value_range=(-1, 1),
    )

    stats_path = out_dir / f"{prefix}sanity_{tag}.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats_path


def summarise_fft_stats(json_path: Path) -> Optional[dict]:
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
