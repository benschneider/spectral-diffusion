"""Shared debugging helpers for diagnostic scripts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from torchvision.utils import save_image


def cycle_loader(loader: Iterable) -> Iterator:
    """Yield batches from ``loader`` indefinitely."""

    while True:
        for batch in loader:
            yield batch


def fft_band_means(tensor: torch.Tensor) -> Dict[str, float]:
    if tensor.is_complex():
        spatial = tensor
    else:
        spatial = torch.complex(tensor, torch.zeros_like(tensor))
    fft = torch.fft.fftshift(torch.fft.fft2(spatial, norm="ortho"), dim=(-2, -1))
    magnitude = fft.abs()
    mean_total = float(magnitude.mean().cpu())

    height, width = magnitude.shape[-2:]
    fy = torch.fft.fftfreq(height, d=1.0 / float(height)).to(magnitude.device)
    fx = torch.fft.fftfreq(width, d=1.0 / float(width)).to(magnitude.device)
    yy = fy[:, None]
    xx = fx[None, :]
    radius = torch.sqrt(xx**2 + yy**2)
    mask_high = radius >= 0.25
    if torch.any(mask_high):
        mean_high = float(magnitude[..., mask_high].mean().cpu())
    else:
        mean_high = float("nan")
    return {"fft_mean": mean_total, "fft_high": mean_high}


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total) if total > 0 else 0.0


def parameter_delta(model: torch.nn.Module, previous: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    state = model.state_dict()
    for name, tensor in state.items():
        tensor_cpu = tensor.detach().cpu()
        prev = previous.get(name)
        if prev is None:
            delta = tensor_cpu.float().pow(2).sum().item()
        else:
            delta = (tensor_cpu.float() - prev.float()).pow(2).sum().item()
        total += delta
        previous[name] = tensor_cpu
    return math.sqrt(total)


def structure_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    b = x.shape[0]
    x_flat = x.view(b, -1)
    y_flat = y.view(b, -1)
    x_center = x_flat - x_flat.mean(dim=1, keepdim=True)
    y_center = y_flat - y_flat.mean(dim=1, keepdim=True)
    numerator = (x_center * y_center).sum(dim=1)
    denominator = torch.sqrt((x_center.pow(2).sum(dim=1) * y_center.pow(2).sum(dim=1)) + 1e-8)
    corr = numerator / denominator
    return float(torch.mean(corr).item())


def phase_rms(x: torch.Tensor, y: torch.Tensor, norm: str = "ortho") -> float:
    phi_x = torch.angle(torch.fft.fftn(x, dim=(-2, -1), norm=norm))
    phi_y = torch.angle(torch.fft.fftn(y, dim=(-2, -1), norm=norm))
    dphi = torch.atan2(torch.sin(phi_y - phi_x), torch.cos(phi_y - phi_x))
    return float(dphi.std().item())


def _centered_rms(tensor: torch.Tensor) -> float:
    centered = tensor - tensor.mean()
    return float(centered.pow(2).mean().sqrt().item())


def summarise_snr_spikes(
    *,
    snr_vals: torch.Tensor,
    sqrt_alpha_t: torch.Tensor,
    sqrt_one_minus_t: torch.Tensor,
    timesteps: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    noise: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    threshold: float,
    top_k: int = 3,
) -> Optional[Dict[str, Any]]:
    if snr_vals.numel() == 0:
        return None

    snr_flat = snr_vals.view(-1)
    mask = snr_flat > threshold
    if not torch.any(mask):
        return None

    candidate_indices = torch.nonzero(mask, as_tuple=False).view(-1)
    candidate_snr = snr_flat[candidate_indices]
    order = torch.argsort(candidate_snr, descending=True)
    selected = candidate_indices[order][: top_k if top_k > 0 else None]

    sqrt_alpha_flat = sqrt_alpha_t.view(-1)
    sqrt_one_minus_flat = sqrt_one_minus_t.view(-1)
    timesteps_flat = timesteps.view(-1)

    entries: List[Dict[str, Any]] = []
    for idx in selected.tolist():
        sqrt_alpha = sqrt_alpha_flat[idx]
        sqrt_one_minus = sqrt_one_minus_flat[idx]
        alpha = sqrt_alpha.pow(2)
        one_minus_alpha = sqrt_one_minus.pow(2)

        signal_rms = _centered_rms(clean.view(-1, *clean.shape[1:])[idx])
        noisy_rms = _centered_rms(noisy.view(-1, *noisy.shape[1:])[idx])
        noise_sample = noise.view(-1, *noise.shape[1:])[idx]
        noise_rms = float(noise_sample.detach().pow(2).mean().sqrt().item())
        noise_mean = float(noise_sample.detach().mean().item())
        target_std = float(target.view(-1, *target.shape[1:])[idx].detach().std().item())
        prediction_std = float(
            prediction.view(-1, *prediction.shape[1:])[idx].detach().std().item()
        )

        entries.append(
            {
                "sample_index": int(idx),
                "timestep": int(timesteps_flat[idx].item()),
                "snr": float(snr_flat[idx].item()),
                "sqrt_alpha": float(sqrt_alpha.item()),
                "sqrt_one_minus_alpha": float(sqrt_one_minus.item()),
                "alpha": float(alpha.item()),
                "one_minus_alpha": float(one_minus_alpha.item()),
                "signal_rms": signal_rms,
                "noisy_rms": noisy_rms,
                "noise_rms": noise_rms,
                "noise_mean": noise_mean,
                "target_std": target_std,
                "prediction_std": prediction_std,
            }
        )

    return {
        "threshold": float(threshold),
        "count": int(candidate_indices.numel()),
        "max_snr": float(candidate_snr.max().item()),
        "top_timesteps": [entry["timestep"] for entry in entries],
        "entries": entries,
    }


def log_snr_spike(summary: Dict[str, Any]) -> None:
    header = (
        "[SNRSpike] count={count} threshold={threshold:.1f} max_snr={max_snr:.2f}".format(
            count=summary["count"],
            threshold=summary["threshold"],
            max_snr=summary["max_snr"],
        )
    )
    print(header)
    for entry in summary["entries"]:
        print(
            "[SNRSpike] sample={sample_index} t={timestep} "
            "snr={snr:.2f} sqrt_alpha={sqrt_alpha:.6f} "
            "sqrt_one_minus_alpha={sqrt_one_minus_alpha:.6f} "
            "one_minus_alpha={one_minus_alpha:.8f} signal_rms={signal_rms:.6f} "
            "noisy_rms={noisy_rms:.6f} noise_rms={noise_rms:.6f} noise_mean={noise_mean:+.6f} "
            "target_std={target_std:.6f} prediction_std={prediction_std:.6f}".format(
                **entry
            )
        )


def save_tensor_preview(tensor: torch.Tensor, path: Path, name: str) -> None:
    tensor = tensor.detach().cpu()
    print(
        f"[{name}] mean={tensor.mean():.3f}, std={tensor.std():.3f}, "
        f"min={tensor.min():.3f}, max={tensor.max():.3f}"
    )
    span = tensor.max() - tensor.min()
    scaled = (tensor - tensor.min()) / (span + 1e-8)
    save_image(scaled, path)
