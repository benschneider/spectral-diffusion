#!/usr/bin/env python
"""Thin diagnostic wrapper for recording early optimisation behaviour."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config, seed_everything
from src.cli.train import apply_variant_override
from src.core import build_model, get_loss_fn
from src.core.diffusion_step import predict_x0
from src.core.functional import compute_snr_weight, compute_target
from src.core.fft_feedback import compute_fft_feedback
from src.spectral.fft_adapter import add_uniform_frequency_noise
from src.training.builders import build_dataloader, build_optimizer
from src.training.scheduler import (
    build_diffusion,
    loss_aware_timesteps,
    sample_timesteps,
)
from src.training.regulators import AdaptiveSNRController
from src.utils.debug_helpers import (
    cycle_loader,
    fft_band_means,
    grad_norm,
    parameter_delta,
    phase_rms,
    save_tensor_preview,
    structure_correlation,
    summarise_snr_spikes,
)

# Backwards compatibility for downstream tooling/tests relying on the legacy name.
_summarise_snr_spikes = summarise_snr_spikes
from src.utils.sanity_checks import check_fft_sanity

SNR_SPIKE_THRESHOLD = 1_000.0
SNR_SPIKE_TOP_K = 3
SIGMA_MIN = 1e-4
SNR_CLIP = 250.0
PRED_STD_WARN_FACTOR = 5.0
FFT_HIGH_WARN_THRESHOLD = 0.8


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
    adaptive_snr: bool = False,
    log_snr_json: bool = False,
    loader: Optional[DataLoader] = None,
    snr_min: float = 0.5,
    snr_max: float = 2.5,
    snr_inc: float = 0.1,
    snr_dec: float = 0.2,
    snr_kappa_thresh: float = 2.5e-1,
    snr_alpha_fac_high: float = 1.12,
    verbose_logs: bool = False,
) -> Path:
    RECORDER_VERSION = "v2.0"
    config = load_config(config_path=config_path)
    apply_variant_override(config, variant)
    seed_everything(config)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_events: List[Dict[str, Any]] = []

    def _log_event(
        tag: str,
        message: str,
        *,
        step: Optional[int] = None,
        level: str = "info",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {"tag": tag, "message": message, "level": level}
        if step is not None:
            entry["step"] = step
        if extra:
            entry.update(extra)
        diagnostic_events.append(entry)
        if verbose_logs:
            print(message)

    _log_event("Normalization", "[Normalization] Disabled for diagnostic mode")

    _log_event("Recorder", f"[RecordTrainingSteps] Running version {RECORDER_VERSION}")

    def _adaptive_weight_logger(message: str, diag: Dict[str, Any]) -> None:
        step_value = diag.get("step") if isinstance(diag, dict) else None
        try:
            step_idx = int(step_value) if step_value is not None else None
        except (TypeError, ValueError):
            step_idx = None
        extra = {
            key: value
            for key, value in diag.items()
            if isinstance(value, (int, float, bool))
        }
        _log_event("AdaptiveSNRWeight", message, step=step_idx, extra=extra)

    def _attach_adaptive_logger(target: Any) -> None:
        adaptive = getattr(target, "adaptive", None)
        if adaptive is not None and hasattr(adaptive, "set_log_fn"):
            adaptive.set_log_fn(_adaptive_weight_logger)

    data_loader = loader or build_dataloader(config)
    data_iter = cycle_loader(data_loader)

    model = build_model(config.get("model", {}))
    loss_fn = get_loss_fn(config.get("loss", {}))
    _attach_adaptive_logger(loss_fn)
    marker = getattr(loss_fn, "residual_marker", None)
    if callable(marker):
        marker_msg = str(marker())
        _log_event("ResidualMarker", marker_msg)

    diffusion_cfg = config.get("diffusion", {}) or {}
    spectral_cfg = config.setdefault("spectral", {})

    snr_weighting_cfg = diffusion_cfg.get("snr_weighting")
    force_weighting = diffusion_cfg.get("diagnostic_force_weighting", True)
    if hasattr(loss_fn, "set_weighting_enabled"):
        if force_weighting:
            loss_fn.set_weighting_enabled(True)
            snr_weighting = True
            _log_event(
                "AdaptiveWeighting",
                "[RecordTrainingSteps] forcing adaptive weighting on for diagnostics",
            )
        else:
            if snr_weighting_cfg is None:
                enabled = getattr(loss_fn, "use_weighting", True)
            else:
                enabled = bool(snr_weighting_cfg)
            loss_fn.set_weighting_enabled(enabled)
            snr_weighting = enabled
    else:
        if snr_weighting_cfg is None:
            snr_weighting = getattr(loss_fn, "use_weighting", True)
        else:
            snr_weighting = bool(snr_weighting_cfg)

    _attach_adaptive_logger(loss_fn)

    optimiser = build_optimizer(model, config)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_obj)

    diffusion_cfg = dict(diffusion_cfg)
    spectral_cfg = dict(spectral_cfg)

    T = int(diffusion_cfg.get("num_timesteps", 1000))
    schedule = diffusion_cfg.get("beta_schedule", "cosine")
    schedule_kwargs: Dict[str, float] = dict(diffusion_cfg.get("schedule_kwargs", {}) or {})
    schedule_key = schedule.replace("-", "_").lower()
    if schedule_key == "logsnr_cosine":
        logsnr_cfg = diffusion_cfg.get("logsnr", {}) or {}
        for key in ("lambda_min", "lambda_max", "delta"):
            if key in logsnr_cfg and key not in schedule_kwargs:
                schedule_kwargs[key] = float(logsnr_cfg[key])
    prediction_type = diffusion_cfg.get("prediction_type", "eps")
    snr_transform = diffusion_cfg.get("snr_transform", "snr")
    snr_transform = str(snr_transform)
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
    fft_norm = diffusion_cfg.get("fft_norm", config.get("spectral", {}).get("fft_norm", "ortho"))
    corruption_mode = diffusion_cfg.get(
        "corruption_mode",
        config.get("spectral", {}).get("corruption_mode", "magnitude"),
    )
    phase_std = float(
        diffusion_cfg.get("phase_std", config.get("spectral", {}).get("phase_std", 0.0))
    )
    snr_ratio_cfg = diffusion_cfg.get("snr_ratio", config.get("spectral", {}).get("snr_ratio"))
    effective_snr_ratio = (
        float(snr_ratio)
        if snr_ratio is not None
        else (float(snr_ratio_cfg) if snr_ratio_cfg is not None else None)
    )
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

    min_snr_weight = float(diffusion_cfg.get("min_snr_weight", 0.05))
    max_snr_weight = float(diffusion_cfg.get("max_snr_weight", 10.0))
    sar_weight = float(config.get("spectral", {}).get("sar_weight", 0.0))
    loss_aware_enabled = bool(diffusion_cfg.get("loss_aware_sampling", True))
    lais_temperature = float(diffusion_cfg.get("lais_temperature", 1.2))
    lais_decay = float(diffusion_cfg.get("lais_decay", 0.95))

    coeffs = build_diffusion(T, schedule, schedule_kwargs)
    if schedule_key == "logsnr_cosine":
        lam_min = float(schedule_kwargs.get("lambda_min", -13.0))
        lam_max = float(schedule_kwargs.get("lambda_max", 13.0))
        delta = float(schedule_kwargs.get("delta", 0.008))
        _log_event(
            "Schedule",
            f"[Schedule] mode=logsnr_cosine λ∈[{lam_min:.3g},{lam_max:.3g}] δ={delta:.3g}",
            extra={
                "mode": "logsnr_cosine",
                "lambda_min": lam_min,
                "lambda_max": lam_max,
                "delta": delta,
            },
        )
    _log_event(
        "Schedule",
        "[Schedule] trim_offset=%d num_timesteps=%d min_sigma=%.4f"
        % (
            coeffs.trim_offset,
            coeffs.num_timesteps,
            coeffs.min_safe_sigma,
        ),
        extra={
            "trim_offset": coeffs.trim_offset,
            "num_timesteps": coeffs.num_timesteps,
            "min_sigma": coeffs.min_safe_sigma,
        },
    )

    loss_landscape = torch.ones(coeffs.num_timesteps, device=device_obj)

    controller: Optional[AdaptiveSNRController] = None
    if adaptive_snr:
        controller_cfg = diffusion_cfg.get("adaptive_snr_controller", {}) or {}
        controller = AdaptiveSNRController(
            min_snr=float(controller_cfg.get("min_snr", 0.5)),
            max_snr=float(controller_cfg.get("max_snr", 2.5)),
            inc=float(controller_cfg.get("inc", 1.05)),
            dec=float(controller_cfg.get("dec", 0.9)),
            kappa_thresh=float(controller_cfg.get("kappa_thresh", 1e-3)),
            alpha_fac_high=float(controller_cfg.get("alpha_fac_high", 1.8)),
            overflow_high=float(controller_cfg.get("overflow_high", 0.1)),
            initial_ratio=effective_snr_ratio,
        )
        effective_snr_ratio = controller.ratio
        _log_event(
            "AdaptiveSNR",
            (
                f"[AdaptiveSNR] enabled with start ratio={controller.ratio:.3f} "
                f"bounds=[{controller.min_snr:.3f}, {controller.max_snr:.3f}]"
            ),
            extra={
                "start_ratio": controller.ratio,
                "min_snr": controller.min_snr,
                "max_snr": controller.max_snr,
            },
        )

    step_records: List[Dict[str, Any]] = []
    snr_summaries: List[Dict[str, Any]] = []
    overflow_log_count = 0
    overflow_log_limit = 5
    previous_state: Dict[str, torch.Tensor] = {}

    # Adaptive SNR controller state
    current_snr_ratio = effective_snr_ratio

    def _clamp_snr(val: Optional[float]) -> Optional[float]:
        if val is None:
            return None
        return max(snr_min, min(snr_max, float(val)))

    if current_snr_ratio is not None:
        current_snr_ratio = _clamp_snr(current_snr_ratio)

    # --- Adaptive SNR trend state ---
    prev_snr_mean: Optional[float] = None
    prev_snr_max: Optional[float] = None

    # --- Regulator persistent state & helpers ---
    snr_ema_beta = 0.6  # EWMA smoothing for SNR trends
    snr_max_slope_ema: Optional[float] = None
    snr_mean_slope_ema: Optional[float] = None
    inc_armed = False     # hysteresis arm for increases
    dec_armed = False     # hysteresis arm for decreases
    cooldown_steps = 2    # require N quiet steps between changes
    last_change_step = -10

    # returns updated_ema
    def _ewma(prev: Optional[float], value: float, beta: float) -> float:
        return value if prev is None else (beta * value + (1.0 - beta) * prev)

    def _log_diag(
        step: int,
        tag: str,
        message: str,
        *,
        level: str = "info",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        _log_event(tag, message, step=step, level=level, extra=extra)

    def _log_tensor_preview(tensor: torch.Tensor, path: Path, name: str, step: int) -> None:
        tensor_cpu = tensor.detach().cpu()
        stats = {
            "mean": float(tensor_cpu.mean().item()),
            "std": float(tensor_cpu.std().item()),
            "min": float(tensor_cpu.min().item()),
            "max": float(tensor_cpu.max().item()),
        }
        message = (
            f"[{name}] mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"min={stats['min']:.3f}, max={stats['max']:.3f}"
        )
        _log_event(
            "TensorPreview",
            message,
            step=step,
            extra={"tensor": name, **stats},
        )
        save_tensor_preview(tensor, path, name, log_fn=None)

    for step in range(steps):
        xb, _ = next(data_iter)
        xb = xb.to(device_obj)
        model.train()

        B = xb.shape[0]
        if loss_aware_enabled:
            t = loss_aware_timesteps(
                B,
                loss_landscape,
                device=xb.device,
                temperature=lais_temperature,
                min_timestep=0,
            )
        else:
            t = sample_timesteps(B, coeffs.num_timesteps, xb.device)
        sqrt_alpha_t = coeffs.sqrt_alphas_cumprod[t].view(B, 1, 1, 1).to(device_obj)
        sqrt_one_minus_t = coeffs.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1).to(device_obj)

        base_noise = torch.randn_like(xb)
        noise_stats: Dict[str, float] = {}
        x_t, effective_noise = add_uniform_frequency_noise(
            xb,
            base_noise,
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
            snr_ratio=current_snr_ratio,
            dc_scale_factor=effective_dc_scale,
            return_noise=True,
        )

        if step == 0 and effective_snr_ratio is not None:
            noise_term = x_t - xb
            dc_shift = float(noise_term.mean().item())
            noisy_mean = float(x_t.mean().item())
            noisy_std = float(x_t.std().item())
            mean_ok = abs(dc_shift) < 5e-3
            std_ok = noisy_std > 0
            ratio_str = f"{effective_snr_ratio:g}"
            if mean_ok and std_ok:
                _log_diag(step, "Noise", f"[Noise] mode={corruption_mode}, snr_ratio={ratio_str}, mean/std check OK")
            else:
                noise_message = (
                    f"[Noise] mode={corruption_mode}, snr_ratio={ratio_str}, "
                    f"mean={noisy_mean:.3f} std={noisy_std:.3f} mean_shift={dc_shift:+.4f}"
                )
                _log_event(
                    "Noise",
                    noise_message,
                    step=step,
                    level="warning",
                    extra={
                        "mode": corruption_mode,
                        "snr_ratio": effective_snr_ratio,
                        "mean": noisy_mean,
                        "std": noisy_std,
                        "mean_shift": dc_shift,
                    },
                )

        pred = model(x_t, t)
        target = compute_target(
            prediction_type,
            xb,
            x_t,
            effective_noise,
            sqrt_alpha_t,
            sqrt_one_minus_t,
        )

        try:
            denoised = predict_x0(pred, prediction_type, x_t, sqrt_alpha_t, sqrt_one_minus_t)
        except ValueError:
            denoised = None

        residual = pred - target
        adaptive_diag: Optional[Dict[str, float]] = None
        try:
            loss_result = loss_fn(pred, target, sqrt_alpha_t, sqrt_one_minus_t)
        except TypeError:
            weight = (
                compute_snr_weight(
                    sqrt_alpha_t,
                    sqrt_one_minus_t,
                    snr_transform,
                    min_snr=min_snr_weight,
                    max_snr=max_snr_weight,
                )
                if snr_weighting
                else None
            )
            loss = loss_fn(residual, weight)
        else:
            if isinstance(loss_result, tuple):
                loss, adaptive_diag = loss_result
            else:
                loss = loss_result
        mae = F.l1_loss(pred.detach(), target.detach())
        loss_value = float(loss.detach().cpu())
        mae_value = float(mae.detach().cpu())

        fft_feedback = compute_fft_feedback(pred, target, fft_norm=fft_norm, sar_weight=sar_weight)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_value = grad_norm(model)
        optimiser.step()

        param_delta_value = parameter_delta(model, previous_state)
        output_fft = fft_band_means(pred.detach())
        input_fft = fft_band_means(xb.detach())
        noisy_fft = fft_band_means(x_t.detach())

        prediction_std_val = float(pred.detach().std().item())
        input_std_val = float(xb.detach().std().item())
        if input_std_val > 0:
            std_ratio = prediction_std_val / max(input_std_val, 1e-8)
            if std_ratio > PRED_STD_WARN_FACTOR:
                warn_message = (
                    "[WARN] Prediction std drift at step {step}: "
                    "std={std:.3f}, input_std={input_std:.3f}, ratio={ratio:.2f}".format(
                        step=step,
                        std=prediction_std_val,
                        input_std=input_std_val,
                        ratio=std_ratio,
                    )
                )
                _log_event(
                    "PredictionStd",
                    warn_message,
                    step=step,
                    level="warning",
                    extra={
                        "prediction_std": prediction_std_val,
                        "input_std": input_std_val,
                        "std_ratio": std_ratio,
                    },
                )
        else:
            std_ratio = float("inf")
        fft_high_val = output_fft.get("fft_high", float("nan"))
        if not math.isnan(fft_high_val) and fft_high_val > FFT_HIGH_WARN_THRESHOLD:
            _log_event(
                "SpectralWarning",
                f"[WARN] Spectral blowup suspected at step {step}: fft_high={fft_high_val:.3f}",
                step=step,
                level="warning",
                extra={"fft_high": fft_high_val},
            )

        corr = structure_correlation(xb.detach(), x_t.detach())
        phase_rms_val = phase_rms(xb.detach(), x_t.detach(), norm=fft_norm)

        timestep_min = int(t.min().item())
        timestep_max = int(t.max().item())
        timestep_mean = float(t.float().mean().item())
        snr_den = torch.clamp(sqrt_one_minus_t**2, min=SIGMA_MIN**2)
        snr_raw = (sqrt_alpha_t**2) / snr_den
        snr_vals = torch.clamp(snr_raw, max=SNR_CLIP)
        snr_min_val = float(snr_vals.min().item())
        snr_max_val = float(snr_vals.max().item())
        snr_mean_val = float(snr_vals.mean().item())
        snr_raw_max = float(snr_raw.max().item())
        if snr_raw_max > SNR_CLIP:
            overflow = int((snr_raw > SNR_CLIP).sum().item())
            if overflow_log_count < overflow_log_limit:
                clipped_snr = min(snr_raw_max, SNR_CLIP)
                message = (
                    f"[OverflowHandler] step={step} mode=deterministic "
                    f"snr={clipped_snr:.1f} loss_mode=x0 count={overflow}"
                )
                _log_event(
                    "OverflowHandler",
                    message,
                    step=step,
                    level="warning",
                    extra={
                        "snr": clipped_snr,
                        "snr_raw_max": snr_raw_max,
                        "overflow_count": overflow,
                    },
                )
            overflow_log_count += 1

        headroom: Optional[float] = None
        high_snr_fraction: Optional[float] = None
        snr_mean_trend: Optional[float] = None
        snr_max_trend: Optional[float] = None

        if adaptive_snr and current_snr_ratio is not None:
            target_ratio = float(current_snr_ratio)
            headroom = max(target_ratio - snr_max_val, 0.0)
            if snr_vals.numel() > 0:
                high_snr_fraction = float((snr_vals > target_ratio).float().mean().item())

            mean_delta = 0.0 if prev_snr_mean is None else snr_mean_val - prev_snr_mean
            max_delta = 0.0 if prev_snr_max is None else snr_max_val - prev_snr_max
            snr_mean_slope_ema = _ewma(snr_mean_slope_ema, mean_delta, snr_ema_beta)
            snr_max_slope_ema = _ewma(snr_max_slope_ema, max_delta, snr_ema_beta)
            snr_mean_trend = snr_mean_slope_ema
            snr_max_trend = snr_max_slope_ema

        snr_spike_summary = summarise_snr_spikes(
            snr_vals=snr_vals.detach(),
            sqrt_alpha_t=sqrt_alpha_t.detach(),
            sqrt_one_minus_t=sqrt_one_minus_t.detach(),
            timesteps=t.detach(),
            clean=xb.detach(),
            noisy=x_t.detach(),
            noise=effective_noise.detach(),
            target=target.detach(),
            prediction=pred.detach(),
            threshold=SNR_SPIKE_THRESHOLD,
            top_k=SNR_SPIKE_TOP_K,
        )
        if snr_spike_summary:
            header = (
                "[SNRSpike] count={count} threshold={threshold:.1f} max_snr={max_snr:.2f}".format(
                    count=snr_spike_summary["count"],
                    threshold=snr_spike_summary["threshold"],
                    max_snr=snr_spike_summary["max_snr"],
                )
            )
            _log_event(
                "SNRSpike",
                header,
                step=step,
                extra=snr_spike_summary,
            )

        target_mean = float(target.detach().mean().item())
        target_std = float(target.detach().std().item())
        target_abs_max = float(target.detach().abs().max().item())
        residual_mean = float(residual.detach().mean().item())
        residual_std = float(residual.detach().std().item())
        residual_abs_max = float(residual.detach().abs().max().item())

        record: Dict[str, Any] = {
            "step": step,
            "loss": loss_value,
            "mae": mae_value,
            "grad_norm": grad_norm_value,
            "param_delta": param_delta_value,
            "noise_norm": float(effective_noise.view(B, -1).norm(dim=1).mean().cpu()),
            "output_mean": float(pred.detach().mean().cpu()),
            "output_std": prediction_std_val,
            "structure_corr": corr,
            "phase_rms": phase_rms_val,
            "prediction_type": prediction_type,
            "timestep_min": timestep_min,
            "timestep_max": timestep_max,
            "timestep_mean": timestep_mean,
            "snr_min": snr_min_val,
            "snr_max": snr_max_val,
            "snr_mean": snr_mean_val,
            "snr_raw_max": snr_raw_max,
            "snr_headroom": headroom if adaptive_snr and current_snr_ratio is not None else None,
            "snr_high_frac": high_snr_fraction if adaptive_snr and current_snr_ratio is not None else None,
            "snr_mean_trend": snr_mean_trend if adaptive_snr and current_snr_ratio is not None else None,
            "snr_max_trend": snr_max_trend if adaptive_snr and current_snr_ratio is not None else None,
            "target_mean": target_mean,
            "target_std": target_std,
            "target_abs_max": target_abs_max,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "residual_abs_max": residual_abs_max,
            "residual_mse": float(residual.detach().pow(2).mean().item()),
            "input_std": input_std_val,
            "prediction_std_ratio": std_ratio,
            "prediction_std_drift": prediction_std_val,
        }

        if effective_snr_ratio is not None:
            record["snr_ratio"] = effective_snr_ratio
        if snr_spike_summary:
            record["snr_spike_count"] = snr_spike_summary["count"]
            record["snr_spike_max"] = snr_spike_summary["max_snr"]
            record["snr_spike_top_timesteps"] = snr_spike_summary["top_timesteps"]

        if noise_stats:
            for key, value in noise_stats.items():
                record[key] = value
        record.update({f"fft_{key}": float(value) for key, value in fft_feedback.items()})
        record.update({f"output_{k}": v for k, v in output_fft.items()})
        record.update({f"input_{k}": v for k, v in input_fft.items()})
        record.update({f"noisy_{k}": v for k, v in noisy_fft.items()})

        if denoised is not None:
            record["denoised_corr"] = structure_correlation(xb.detach(), denoised.detach())

        weight_stats = None
        if adaptive_diag:
            weight_stats = {
                key: float(value)
                for key, value in adaptive_diag.items()
                if isinstance(value, (int, float))
            }
            record.update(weight_stats)
        elif snr_weighting:
            weight = compute_snr_weight(
                sqrt_alpha_t,
                sqrt_one_minus_t,
                snr_transform,
                min_snr=min_snr_weight,
                max_snr=max_snr_weight,
            )
            weight_stats = {
                "snr_weight_min": float(weight.min().item()),
                "snr_weight_max": float(weight.max().item()),
                "snr_weight_mean": float(weight.mean().item()),
            }
            record.update(weight_stats)

        adaptive_note: Optional[str] = None

        if controller is not None:
            new_ratio, note = controller.update(
                loss=loss_value,
                grad_norm=grad_norm_value,
                fft_feedback=fft_feedback,
                adaptive_diag=adaptive_diag,
                snr_vals=snr_vals,
            )
            effective_snr_ratio = new_ratio
            record.update(controller.latest_metrics)
            metrics = controller.latest_metrics
            headroom = metrics.get("snr_headroom", headroom)
            adaptive_note = note
            if metrics and "snr_ratio" in metrics:
                current_snr_ratio = _clamp_snr(metrics["snr_ratio"])
            elif effective_snr_ratio is not None:
                current_snr_ratio = _clamp_snr(effective_snr_ratio)
            if note:
                _log_event(
                    "AdaptiveSNR",
                    f"[AdaptiveSNR] {note}",
                    step=step,
                    extra={"note": note, "snr_ratio": effective_snr_ratio},
                )

        if adaptive_snr and current_snr_ratio is not None and high_snr_fraction is None and snr_vals.numel() > 0:
            high_snr_fraction = float((snr_vals > float(current_snr_ratio)).float().mean().item())

        if adaptive_snr:
            record["snr_headroom"] = headroom
            record["snr_high_frac"] = high_snr_fraction
            record["snr_mean_trend"] = snr_mean_trend
            record["snr_max_trend"] = snr_max_trend

        prev_snr_mean = snr_mean_val
        prev_snr_max = snr_max_val

        if adaptive_snr and current_snr_ratio is not None and high_snr_fraction is None and snr_vals.numel() > 0:
            high_snr_fraction = float((snr_vals > float(current_snr_ratio)).float().mean().item())

        if adaptive_snr:
            record["snr_headroom"] = headroom
            record["snr_high_frac"] = high_snr_fraction
            record["snr_mean_trend"] = snr_mean_trend
            record["snr_max_trend"] = snr_max_trend

        prev_snr_mean = snr_mean_val
        prev_snr_max = snr_max_val

        if log_snr_json:
            snr_entry = {
                "step": step,
                "snr_min": snr_min_val,
                "snr_max": snr_max_val,
                "snr_mean": snr_mean_val,
                "snr_raw_max": snr_raw_max,
            }
            if adaptive_snr:
                snr_entry.update(
                    {
                        "snr_headroom": headroom,
                        "snr_high_frac": high_snr_fraction,
                        "snr_mean_trend": snr_mean_trend,
                        "snr_max_trend": snr_max_trend,
                    }
                )
            if controller is not None:
                snr_entry.update(controller.latest_metrics)
            snr_summaries.append(snr_entry)

        step_records.append(record)

        if loss_aware_enabled:
            per_example_mse = residual.detach().view(B, -1).pow(2).mean(dim=1)
            buckets = torch.zeros_like(loss_landscape)
            buckets.scatter_add_(0, t, per_example_mse.to(loss_landscape.dtype))
            counts = torch.bincount(t, minlength=coeffs.num_timesteps).to(loss_landscape.device)
            mask = counts > 0
            averages = torch.zeros_like(loss_landscape)
            averages[mask] = buckets[mask] / counts[mask]
            loss_landscape = torch.where(
                mask,
                loss_landscape * lais_decay + averages * (1.0 - lais_decay),
                loss_landscape,
            )

        if uniform_corruption and corr < 0.4:
            _log_diag(step, "StructureCorrelation", f"⚠️  Step {step}: structure correlation low ({corr:.2f})")

        if step % log_interval == 0 or step == steps - 1:
            save_root = out_dir / f"step_{step:04d}"
            save_root.mkdir(parents=True, exist_ok=True)

            _log_event(
                "Loss",
                f"[Loss] step={step} loss={loss_value:.6f} mae={mae_value:.6f}",
                step=step,
                extra={"loss": loss_value, "mae": mae_value},
            )
            fft_message = {
                name: float(fft_feedback[name])
                for name in [
                    "amplitude_mae",
                    "phase_mae",
                    "real_mae",
                    "imag_mae",
                    "complex_mae",
                ]
                if name in fft_feedback
            }
            if fft_message:
                _log_event(
                    "FFTFeedback",
                    "[FFTFeedback] "
                    + ", ".join(f"{name}={fft_message[name]:.6f}" for name in fft_message),
                    step=step,
                    extra=fft_message,
                )
            _log_diag(step, "Timesteps",
                "[Timesteps] min={:d} max={:d} mean={:.1f} "
                "snr_min={:.4f} snr_max={:.4f}".format(
                    timestep_min,
                    timestep_max,
                    timestep_mean,
                    snr_min_val,
                    snr_max_val,
                )
            )
            _log_diag(step, "Targets",
                "[Targets] mean={:.6f} std={:.6f} abs_max={:.6f}".format(
                    target_mean,
                    target_std,
                    target_abs_max,
                )
            )
            _log_diag(step, "Residual",
                "[Residual] mean={:.6f} std={:.6f} abs_max={:.6f}".format(
                    residual_mean,
                    residual_std,
                    residual_abs_max,
                )
            )
            if adaptive_note is not None:
                extra = ""
                if adaptive_snr and current_snr_ratio is not None:
                    extra = f" headroom={headroom:.1f} high_frac={high_snr_fraction:.2f}"
                _log_diag(step, "AdaptiveNoise", f"[AdaptiveNoise] step={step} snr_ratio_next={current_snr_ratio:.3g} action={adaptive_note}{extra}")
            if weight_stats:
                if {"snr_weight_min", "snr_weight_max", "snr_weight_mean"}.issubset(weight_stats):
                    _log_diag(step, "SNRWeight", "[SNRWeight] min={:.6f} max={:.6f} mean={:.6f}".format(
                        weight_stats["snr_weight_min"],
                        weight_stats["snr_weight_max"],
                        weight_stats["snr_weight_mean"],
                    ))
                else:
                    mean_val = weight_stats.get("mean_weight_raw", weight_stats.get("mean_weight", 1.0))
                    max_val = weight_stats.get("max_weight_raw", weight_stats.get("max_weight", 1.0))
                    adaptive_message = (
                        "[AdaptiveSNRWeight] mean={:.6f} max={:.6f} kappa={:.4e} "
                        "alpha_fac={:.2f} overflow={:.3f} overflow_ema={:.3f} "
                        "delta={:.3e}{}".format(
                            mean_val,
                            max_val,
                            weight_stats.get("kappa", 0.0),
                            weight_stats.get("alpha_fac", 1.0),
                            weight_stats.get("overflow", 0.0),
                            weight_stats.get("overflow_ema", 0.0),
                            weight_stats.get("delta", 0.0),
                            " frozen" if weight_stats.get("frozen") else "",
                        )
                    )
                    _log_event(
                        "AdaptiveSNRWeight",
                        adaptive_message,
                        step=step,
                        extra=weight_stats,
                    )

            _log_tensor_preview(xb, save_root / "input.png", "input", step)
            _log_tensor_preview(x_t, save_root / "noisy.png", "noisy", step)
            _log_tensor_preview(pred, save_root / "predicted_noise.png", "predicted_noise", step)
            if denoised is not None:
                _log_tensor_preview(denoised, save_root / "prediction.png", "prediction", step)
            else:
                _log_tensor_preview(pred, save_root / "prediction.png", "prediction", step)

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

            band_line = ", ".join(
                f"{label}={fft_feedback.get(name, float('nan')):.6f}"
                for label, name in (
                    ("amp_low", "amplitude_low_mae"),
                    ("amp_mid", "amplitude_mid_mae"),
                    ("amp_high", "amplitude_high_mae"),
                )
                if name in fft_feedback
            )
            if band_line:
                _log_event(
                    "FFTAmplitudeBands",
                    f"[FFTAmplitudeBands] {band_line}",
                    step=step,
                    extra={
                        "amplitude_low_mae": fft_feedback.get("amplitude_low_mae"),
                        "amplitude_mid_mae": fft_feedback.get("amplitude_mid_mae"),
                        "amplitude_high_mae": fft_feedback.get("amplitude_high_mae"),
                    },
                )
            phase_line = ", ".join(
                f"{label}={fft_feedback.get(name, float('nan')):.6f}"
                for label, name in (
                    ("phase_low", "phase_low_mae"),
                    ("phase_mid", "phase_mid_mae"),
                    ("phase_high", "phase_high_mae"),
                )
                if name in fft_feedback
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
                "final_mae": step_records[-1]["mae"] if step_records else None,
                "mean_structure_corr": _mean("structure_corr"),
                "mean_corr_pre": _mean("structure_corr_pre"),
                "mean_corr_post": _mean("structure_corr_post"),
                "mean_mse_pre": _mean("mse_pre"),
                "mean_mse_post": _mean("mse_post"),
                "mean_fft_corr": _mean("fft_corr"),
                "mean_phase_rms": _mean("phase_rms"),
                "mean_signal_energy": _mean("signal_energy"),
                "mean_noise_energy": _mean("noise_energy"),
                "mean_fft_amplitude_mae": _mean("fft_amplitude_mae"),
                "mean_fft_phase_mae": _mean("fft_phase_mae"),
                "mean_fft_real_mae": _mean("fft_real_mae"),
                "mean_fft_imag_mae": _mean("fft_imag_mae"),
                "mean_fft_complex_mae": _mean("fft_complex_mae"),
                "recorder_version": RECORDER_VERSION,
                "normalization_disabled": True,
                "snr_ratio": effective_snr_ratio,
                "dc_scale_factor": effective_dc_scale,
                "adaptive_snr_enabled": bool(adaptive_snr),
                "final_snr_ratio": current_snr_ratio,
            },
            handle,
            indent=2,
        )

    if diagnostic_events:
        diagnostics_path = out_dir / "diagnostics.jsonl"
        with diagnostics_path.open("w", encoding="utf-8") as handle:
            for entry in diagnostic_events:
                handle.write(json.dumps(entry))
                handle.write("\n")

    if log_snr_json and snr_summaries:
        snr_path = out_dir / "snr_stats.jsonl"
        with snr_path.open("w", encoding="utf-8") as handle:
            for entry in snr_summaries:
                handle.write(json.dumps(entry))
                handle.write("\n")

    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record early training behaviour.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--variant", type=str, default=None, help="Optional variant override (baseline/spectral/...)."
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of optimisation steps to record.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store diagnostics.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (cpu/cuda).")
    parser.add_argument("--log-interval", type=int, default=10, help="Interval for saving images/FFT snapshots.")
    parser.add_argument("--snr-ratio", type=float, default=None, help="Override diffusion.snr_ratio for diagnostics.")
    parser.add_argument(
        "--dc-scale-factor",
        type=float,
        default=None,
        help="Override diffusion.dc_scale_factor.",
    )
    parser.add_argument(
        "--adaptive-snr",
        action="store_true",
        help="Enable progressive SNR control via AdaptiveSNRController.",
    )
    parser.add_argument(
        "--log-snr-json",
        action="store_true",
        help="Write per-step SNR summaries to snr_stats.jsonl.",
    )
    parser.add_argument(
        "--verbose-logs",
        action="store_true",
        help="Also stream diagnostic logs to stdout.",
    )
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
        adaptive_snr=bool(args.adaptive_snr),
        log_snr_json=bool(args.log_snr_json),
        verbose_logs=bool(args.verbose_logs),
    )
    print(f"Step recorder artefacts written to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


def record_training_steps(
    config_path: Path,
    *,
    variant: Optional[str] = None,
    steps: int = 100,
    output_dir: Path,
    device: Optional[str] = None,
    log_interval: int = 10,
    snr_ratio: Optional[float] = None,
    dc_scale_factor: Optional[float] = None,
    loader: Optional[DataLoader] = None,
    adaptive_snr: bool = False,
    snr_min: float = 0.5,
    snr_max: float = 2.5,
    snr_inc: float = 0.1,
    snr_dec: float = 0.2,
    snr_kappa_thresh: float = 2.5e-1,
    snr_alpha_fac_high: float = 1.12,
    snr_overflow_high: float = 0.05,
    verbose_logs: bool = False,
    log_snr_json: bool = False,
) -> Path:
    """Backwards-compatible wrapper exported for utility scripts."""

    return run_step_recorder(
        config_path=config_path,
        variant=variant,
        steps=steps,
        output_dir=output_dir,
        device=device,
        log_interval=log_interval,
        snr_ratio=snr_ratio,
        dc_scale_factor=dc_scale_factor,
        loader=loader,
        adaptive_snr=adaptive_snr,
        snr_min=snr_min,
        snr_max=snr_max,
        snr_inc=snr_inc,
        snr_dec=snr_dec,
        snr_kappa_thresh=snr_kappa_thresh,
        snr_alpha_fac_high=snr_alpha_fac_high,
        snr_overflow_high=snr_overflow_high,
        verbose_logs=verbose_logs,
        log_snr_json=log_snr_json,
    )
