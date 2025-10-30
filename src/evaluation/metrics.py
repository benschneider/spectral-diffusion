from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

FID_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    from torchmetrics.image.fid import FrechetInceptionDistance

    FID_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    FrechetInceptionDistance = None  # type: ignore

LPIPS_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    LPIPS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LearnedPerceptualImagePatchSimilarity = None  # type: ignore


def compute_basic_metrics(
    loss_history: Iterable[float],
    mae_history: Iterable[float],
    runtime_seconds: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate basic scalar metrics collected during training."""
    losses = list(loss_history)
    maes = list(mae_history)
    metrics: Dict[str, Any] = {
        "loss_mean": mean(losses) if losses else None,
        "loss_last": losses[-1] if losses else None,
        "mae_mean": mean(maes) if maes else None,
        "runtime_seconds": float(runtime_seconds),
    }
    if extra:
        metrics.update(extra)
    return metrics


def _collect_image_paths(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths = sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )
    if not paths:
        raise ValueError(f"No image files found in {directory}")
    return paths


def _load_image_tensor(path: Path, image_size: Optional[Sequence[int]] = None) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if image_size is not None:
            img = img.resize(tuple(image_size), Image.BILINEAR)
        tensor = TF.to_tensor(img)
    return tensor


@lru_cache(maxsize=32)
def _high_freq_mask_cached(height: int, width: int, cutoff_key: float) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, d=1.0 / float(height))
    fx = torch.fft.fftfreq(width, d=1.0 / float(width))
    yy = fy[:, None]
    xx = fx[None, :]
    radius = torch.sqrt(xx**2 + yy**2)
    mask = (radius >= cutoff_key).to(torch.float32)
    return mask


def _apply_high_pass(tensor: torch.Tensor, cutoff: float = 0.4) -> torch.Tensor:
    height, width = tensor.shape[-2], tensor.shape[-1]
    cutoff_key = round(float(cutoff), 4)
    base_mask = _high_freq_mask_cached(height, width, cutoff_key)
    mask = base_mask.to(device=tensor.device, dtype=tensor.dtype).unsqueeze(0)
    spectrum = torch.fft.fftn(tensor, dim=(-2, -1))
    filtered = spectrum * mask
    return torch.fft.ifftn(filtered, dim=(-2, -1)).real


def _compute_high_freq_psnr(fake: torch.Tensor, real: torch.Tensor) -> float:
    diff = fake - real
    diff_high = _apply_high_pass(diff)
    mse = torch.mean(diff_high**2).item()
    return float("inf") if mse == 0.0 else 10.0 * np.log10(1.0 / mse)


def _compute_pair_metrics(fake: torch.Tensor, real: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((fake - real) ** 2).item()
    mae = torch.mean(torch.abs(fake - real)).item()
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "high_freq_psnr": _compute_high_freq_psnr(fake, real),
    }


def compute_dataset_metrics(
    generated_dir: Path | str,
    reference_dir: Path | str,
    image_size: Optional[Sequence[int]] = None,
    use_fid: bool = False,
    use_lpips: bool = False,
    strict_filenames: bool = True,
) -> Dict[str, Any]:
    """Compute pixel-wise metrics across image folders, optionally FID."""
    generated_dir = Path(generated_dir)
    reference_dir = Path(reference_dir)
    gen_paths = _collect_image_paths(generated_dir)
    ref_paths = _collect_image_paths(reference_dir)

    if strict_filenames:
        if set(p.name for p in gen_paths) != set(p.name for p in ref_paths):
            missing = set(p.name for p in gen_paths) ^ set(p.name for p in ref_paths)
            raise ValueError(
                "Generated and reference directories must contain the same filenames; mismatches: "
                f"{sorted(missing)}"
            )
        ref_lookup = {p.name: p for p in ref_paths}
        paired = [(g, ref_lookup[g.name]) for g in gen_paths]
    else:
        if len(gen_paths) != len(ref_paths):
            raise ValueError("Generated and reference directories must contain the same number of images")
        paired = list(zip(sorted(gen_paths), sorted(ref_paths)))

    pair_metrics: Dict[str, List[float]] = {"mse": [], "mae": [], "psnr": [], "high_freq_psnr": []}
    fid_metric = None
    if use_fid:
        if not FID_AVAILABLE:
            raise RuntimeError("torchmetrics not installed. Install torchmetrics to compute FID.")
        fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False)
    lpips_metric = None
    if use_lpips:
        if not LPIPS_AVAILABLE:
            raise RuntimeError("torchmetrics not installed. Install torchmetrics to compute LPIPS.")
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

    for gen_path, ref_path in paired:
        fake = _load_image_tensor(gen_path, image_size=image_size)
        real = _load_image_tensor(ref_path, image_size=image_size)
        if fake.shape != real.shape:
            raise ValueError(f"Image shape mismatch: {gen_path.name} vs {ref_path.name}")

        metrics = _compute_pair_metrics(fake, real)
        for key, value in metrics.items():
            pair_metrics[key].append(value)

        if fid_metric is not None:
            real_uint8 = (real * 255.0).clamp(0, 255).to(torch.uint8)
            fake_uint8 = (fake * 255.0).clamp(0, 255).to(torch.uint8)
            fid_metric.update(real_uint8.unsqueeze(0), real=True)
            fid_metric.update(fake_uint8.unsqueeze(0), real=False)
        if lpips_metric is not None:
            real_lpips = real.mul(2.0).sub(1.0).unsqueeze(0)
            fake_lpips = fake.mul(2.0).sub(1.0).unsqueeze(0)
            lpips_metric.update(fake_lpips, real_lpips)

    summary = {key: float(np.mean(values)) for key, values in pair_metrics.items()}
    if fid_metric is not None:
        fid_value = fid_metric.compute().item()
        summary["fid"] = fid_value
    if lpips_metric is not None:
        lpips_value = lpips_metric.compute().item()
        summary["lpips"] = lpips_value
    return summary


def compute_fid(generated_path: str, reference_path: str) -> Optional[float]:
    raise NotImplementedError(
        "Use compute_dataset_metrics(..., use_fid=True) with image folders to compute FID."
    )


def compute_lpips(generated_path: str, reference_path: str) -> Optional[float]:
    raise NotImplementedError(
        "Use compute_dataset_metrics(..., use_lpips=True) with image folders to compute LPIPS."
    )


def compute_runtime_stats(runtime_seconds: float, num_steps: int) -> Dict[str, Any]:
    if runtime_seconds <= 0:
        return {"runtime_seconds": 0.0, "steps_per_second": None}
    steps_per_second = num_steps / runtime_seconds if num_steps > 0 else None
    return {
        "runtime_seconds": float(runtime_seconds),
        "steps_per_second": steps_per_second,
    }


def aggregate_metrics(raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
    return dict(raw_metrics)
