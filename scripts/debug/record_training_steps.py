#!/usr/bin/env python
"""
Lightweight training recorder.

Runs a small number of optimisation steps and logs loss/gradient/FFT statistics
so that we can debug early-training behaviour (e.g., CIFAR spectral collapse)
without waiting for a full experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config, seed_everything
from src.cli.train import apply_variant_override
from src.core import build_model, get_loss_fn
from src.core.functional import compute_snr_weight, compute_target
from src.spectral.fft_adapter import add_uniform_frequency_noise
from src.training.builders import build_dataloader, build_optimizer
from src.training.scheduler import build_diffusion, sample_timesteps
from src.utils.sanity_checks import check_fft_sanity


def _cycle(loader: Iterable) -> Iterator:
    """Infinite iterator over a dataloader."""
    while True:
        for batch in loader:
            yield batch


def _fft_band_means(tensor: torch.Tensor) -> Dict[str, float]:
    """Return overall FFT magnitude and a simple high-frequency average."""
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
    mask_high = radius >= 0.25  # arbitrary cutoff: upper 75% of frequencies
    if torch.any(mask_high):
        mean_high = float(magnitude[..., mask_high].mean().cpu())
    else:
        mean_high = float("nan")
    return {"fft_mean": mean_total, "fft_high": mean_high}


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total) if total > 0 else 0.0


def _parameter_delta(model: torch.nn.Module, previous: Dict[str, torch.Tensor]) -> float:
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


def _structure_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean Pearson correlation between two batched tensors."""
    b = x.shape[0]
    x_flat = x.view(b, -1)
    y_flat = y.view(b, -1)
    x_center = x_flat - x_flat.mean(dim=1, keepdim=True)
    y_center = y_flat - y_flat.mean(dim=1, keepdim=True)
    numerator = (x_center * y_center).sum(dim=1)
    denominator = torch.sqrt((x_center.pow(2).sum(dim=1) * y_center.pow(2).sum(dim=1)) + 1e-8)
    corr = numerator / denominator
    return float(torch.mean(corr).item())


def _phase_rms(x: torch.Tensor, y: torch.Tensor, norm: str = "ortho") -> float:
    phi_x = torch.angle(torch.fft.fftn(x, dim=(-2, -1), norm=norm))
    phi_y = torch.angle(torch.fft.fftn(y, dim=(-2, -1), norm=norm))
    dphi = torch.atan2(
        torch.sin(phi_y - phi_x),
        torch.cos(phi_y - phi_x),
    )
    return float(dphi.std().item())


def run_step_recorder(
    config_path: Path,
    *,
    variant: Optional[str],
    steps: int,
    output_dir: Path,
    device: Optional[str] = None,
    log_interval: int = 10,
    snr_ratio: Optional[float] = None,
    dc_scale_factor: Optional[float] = None,
) -> Path:
    RECORDER_VERSION = "v1.1"
    config = load_config(config_path=config_path)
    apply_variant_override(config, variant)
    seed_everything(config)

    print("[Normalization] Disabled for diagnostic mode")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RecordTrainingSteps] Running version {RECORDER_VERSION}")

    dataset = build_dataloader(config)
    data_iter = _cycle(dataset)

    model = build_model(config.get("model", {}))
    loss_fn = get_loss_fn(config.get("loss", {}))
    optimiser = build_optimizer(model, config)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_obj)

    diffusion_cfg = config.get("diffusion", {}) or {}
    T = int(diffusion_cfg.get("num_timesteps", 1000))
    schedule = diffusion_cfg.get("beta_schedule", "cosine")
    prediction_type = diffusion_cfg.get("prediction_type", "eps")
    snr_weighting = bool(diffusion_cfg.get("snr_weighting", False))
    snr_transform = diffusion_cfg.get("snr_transform", "snr")
    uniform_corruption = bool(diffusion_cfg.get("uniform_corruption", False))
    corruption_scale = float(
        diffusion_cfg.get(
            "uniform_corruption_scale",
            config.get("spectral", {}).get("uniform_corruption_scale", 1.0),
        )
    )
    target_corr = diffusion_cfg.get(
        "similarity_target",
        config.get("spectral", {}).get("similarity_target", 0.7),
    )
    target_corr = float(target_corr) if target_corr is not None else None
    adaptive_rescale = bool(
        diffusion_cfg.get(
            "adaptive_rescale",
            config.get("spectral", {}).get("adaptive_rescale", False),
        )
    )
    fft_norm = diffusion_cfg.get(
        "fft_norm", config.get("spectral", {}).get("fft_norm", "ortho")
    )
    corruption_mode = diffusion_cfg.get(
        "corruption_mode",
        config.get("spectral", {}).get("corruption_mode", "magnitude"),
    )
    phase_std = float(
        diffusion_cfg.get(
            "phase_std", config.get("spectral", {}).get("phase_std", 0.0)
        )
    )
    snr_ratio_cfg = diffusion_cfg.get(
        "snr_ratio",
        config.get("spectral", {}).get("snr_ratio"),
    )
    effective_snr_ratio = (
        float(snr_ratio)
        if snr_ratio is not None
        else (float(snr_ratio_cfg) if snr_ratio_cfg is not None else None)
    )
    spectral_cfg = config.setdefault("spectral", {})
    if effective_snr_ratio is not None:
        diffusion_cfg["snr_ratio"] = effective_snr_ratio
        spectral_cfg["snr_ratio"] = effective_snr_ratio
    dc_scale_cfg = diffusion_cfg.get(
        "dc_scale_factor",
        config.get("spectral", {}).get("dc_scale_factor", 0.1),
    )
    effective_dc_scale = float(dc_scale_factor) if dc_scale_factor is not None else float(dc_scale_cfg)
    diffusion_cfg["dc_scale_factor"] = effective_dc_scale
    spectral_cfg["dc_scale_factor"] = effective_dc_scale

    coeffs = build_diffusion(T, schedule)

    step_records: List[Dict[str, Any]] = []
    previous_state: Dict[str, torch.Tensor] = {}

    for step in range(steps):
        xb, _ = next(data_iter)
        xb = xb.to(device_obj)
        model.train()

        B = xb.shape[0]
        t = sample_timesteps(B, T, xb.device)
        sqrt_alpha_t = coeffs.sqrt_alphas_cumprod[t].view(B, 1, 1, 1).to(device_obj)
        sqrt_one_minus_t = coeffs.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1).to(device_obj)

        noise = torch.randn_like(xb)
        noise_stats: Dict[str, float] = {}
        x_t = add_uniform_frequency_noise(
            xb,
            noise,
            sqrt_alpha_t=sqrt_alpha_t,
            sqrt_one_minus_alpha_t=sqrt_one_minus_t,
            uniform_corruption=uniform_corruption,
            strength=corruption_scale,
            mode=corruption_mode,
            phase_std=phase_std,
            target_corr=target_corr,
            adaptive_rescale=adaptive_rescale,
            stats=noise_stats,
            fft_norm=fft_norm,
            snr_ratio=effective_snr_ratio,
            dc_scale_factor=effective_dc_scale,
        )

        if step == 0 and effective_snr_ratio is not None:
            noisy_mean = noise_stats.get("noisy_mean")
            noisy_std = noise_stats.get("noisy_std")
            mean_ok = noisy_mean is not None and abs(noisy_mean - 0.5) < 0.05
            std_ok = noisy_std is not None and noisy_std > 0
            ratio_str = f"{effective_snr_ratio:g}"
            if mean_ok and std_ok:
                print(f"[Noise] mode={corruption_mode}, snr_ratio={ratio_str}, mean/std check OK")
            else:
                mean_display = "nan" if noisy_mean is None else f"{noisy_mean:.3f}"
                std_display = "nan" if noisy_std is None else f"{noisy_std:.3f}"
                print(
                    f"[Noise] mode={corruption_mode}, snr_ratio={ratio_str}, "
                    f"mean={mean_display} std={std_display}"
                )
            signal_rms = (xb - 0.5).pow(2).mean().sqrt().item()
            noise_rms = (x_t - xb).pow(2).mean().sqrt().item()
            measured_snr = signal_rms / max(noise_rms, 1e-8)
            print(
                f"[FFTNoiseCheck] snr_target={effective_snr_ratio:.3f}, "
                f"measured={measured_snr:.3f}, signal_rms={signal_rms:.3f}, "
                f"noise_rms={noise_rms:.3f}"
            )
            dc_shift = float(x_t.mean().item() - xb.mean().item())
            print(f"[FFTNoiseCheck] mean_shift={dc_shift:+.4f}")

        pred = model(x_t, t)
        target = compute_target(
            prediction_type,
            xb,
            x_t,
            noise,
            sqrt_alpha_t,
            sqrt_one_minus_t,
        )

        residual = pred - target
        weight = compute_snr_weight(sqrt_alpha_t, sqrt_one_minus_t, snr_transform) if snr_weighting else None
        loss = loss_fn(residual, weight)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _grad_norm(model)
        optimiser.step()

        param_delta = _parameter_delta(model, previous_state)
        output_fft = _fft_band_means(pred.detach())
        input_fft = _fft_band_means(xb.detach())
        noisy_fft = _fft_band_means(x_t.detach())
        corr = _structure_correlation(xb.detach(), x_t.detach())

        record: Dict[str, Any] = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "param_delta": param_delta,
            "noise_norm": float(noise.view(B, -1).norm(dim=1).mean().cpu()),
            "output_mean": float(pred.detach().mean().cpu()),
            "output_std": float(pred.detach().std().cpu()),
            "structure_corr": corr,
            "phase_rms": _phase_rms(xb.detach(), x_t.detach(), norm=fft_norm),
        }
        if noise_stats:
            record["structure_corr_pre"] = noise_stats.get("structure_corr_pre")
            record["structure_corr_post"] = noise_stats.get("structure_corr_post")
            record["mse_pre"] = noise_stats.get("mse_pre")
            record["mse_post"] = noise_stats.get("mse_post")
            record["fft_corr"] = noise_stats.get("fft_corr")
            record["signal_energy"] = noise_stats.get("signal_energy")
            record["noise_energy"] = noise_stats.get("noise_energy")
            record["noisy_mean"] = noise_stats.get("noisy_mean")
            record["noisy_std"] = noise_stats.get("noisy_std")
            record["dc_image_mean"] = noise_stats.get("dc_image_mean")
            record["dc_rms_ratio"] = noise_stats.get("dc_rms_ratio")
            record["dc_perturb_mag"] = noise_stats.get("dc_perturb_mag")
            record["dc_scale_factor"] = noise_stats.get("dc_scale_factor")
        record.update({f"output_{k}": v for k, v in output_fft.items()})
        record.update({f"input_{k}": v for k, v in input_fft.items()})
        record.update({f"noisy_{k}": v for k, v in noisy_fft.items()})
        if effective_snr_ratio is not None:
            record["snr_ratio"] = effective_snr_ratio
        step_records.append(record)

        if uniform_corruption and corr < 0.4:
            print(f"⚠️  Step {step}: structure correlation low ({corr:.2f})")

        if step % log_interval == 0 or step == steps - 1:
            save_root = out_dir / f"step_{step:04d}"
            save_root.mkdir(parents=True, exist_ok=True)

            def _save_raw(tensor: torch.Tensor, path: Path, name: str) -> None:
                tensor = tensor.detach().cpu()
                print(
                    f"[{name}] mean={tensor.mean():.3f}, std={tensor.std():.3f}, "
                    f"min={tensor.min():.3f}, max={tensor.max():.3f}"
                )
                span = tensor.max() - tensor.min()
                scaled = (tensor - tensor.min()) / (span + 1e-8)
                save_image(scaled, path)

            _save_raw(xb, save_root / "input.png", "input")
            _save_raw(x_t, save_root / "noisy.png", "noisy")
            _save_raw(pred, save_root / "prediction.png", "prediction")
            check_fft_sanity(
                xb.detach().cpu(),
                dataset_name="cifar10_input",
                out_dir=save_root,
                prefix="",
            )
            check_fft_sanity(
                x_t.detach().cpu(),
                dataset_name="cifar10_noisy",
                out_dir=save_root,
                prefix="",
            )
            check_fft_sanity(
                pred.detach().cpu(),
                dataset_name="model_output",
                out_dir=save_root,
                prefix="",
            )

    metrics_path = out_dir / "step_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for record in step_records:
            handle.write(json.dumps(record))
            handle.write("\n")

    summary_path = out_dir / "summary.json"
    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in step_records if key in r and r[key] is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_path": str(config_path),
                "variant": variant,
                "steps": steps,
                "device": str(device_obj),
                "log_interval": log_interval,
                "final_loss": step_records[-1]["loss"] if step_records else None,
                "mean_structure_corr": _mean("structure_corr"),
                "mean_corr_pre": _mean("structure_corr_pre"),
                "mean_corr_post": _mean("structure_corr_post"),
                "mean_mse_pre": _mean("mse_pre"),
                "mean_mse_post": _mean("mse_post"),
                "mean_fft_corr": _mean("fft_corr"),
                "mean_phase_rms": _mean("phase_rms"),
                "mean_signal_energy": _mean("signal_energy"),
                "mean_noise_energy": _mean("noise_energy"),
                "recorder_version": RECORDER_VERSION,
                "normalization_disabled": True,
                "snr_ratio": effective_snr_ratio,
                "dc_scale_factor": effective_dc_scale,
            },
            handle,
            indent=2,
        )

    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record early training behaviour.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--variant", type=str, default=None, help="Optional variant override (baseline/spectral/...).")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimisation steps to record.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store diagnostics.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (cpu/cuda).")
    parser.add_argument("--log-interval", type=int, default=10, help="Interval for saving images/FFT snapshots.")
    parser.add_argument("--snr-ratio", type=float, default=None, help="Override diffusion.snr_ratio for diagnostics.")
    parser.add_argument("--dc-scale-factor", type=float, default=None, help="Override diffusion.dc_scale_factor.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_path = run_step_recorder(
        config_path=args.config,
        variant=args.variant,
        steps=int(args.steps),
        output_dir=args.output_dir,
        device=args.device,
        log_interval=int(args.log_interval),
        snr_ratio=args.snr_ratio,
        dc_scale_factor=args.dc_scale_factor,
    )
    print(f"Step recorder artefacts written to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
def _phase_rms(x: torch.Tensor, y: torch.Tensor, norm: str = "ortho") -> float:
    phi_x = torch.angle(torch.fft.fftn(x, dim=(-2, -1), norm=norm))
    phi_y = torch.angle(torch.fft.fftn(y, dim=(-2, -1), norm=norm))
    dphi = torch.atan2(
        torch.sin(phi_y - phi_x),
        torch.cos(phi_y - phi_x),
    )
    return float(dphi.std().item())
