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
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader

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
from src.core.fft_feedback import compute_fft_feedback
from src.utils.sanity_checks import check_fft_sanity
from src.utils.debug_helpers import (
    _fft_band_means,
    _grad_norm,
    _parameter_delta,
    _structure_correlation,
    _phase_rms,
    _predict_x0,
    _centered_rms,
    _summarise_snr_spikes,
    _log_snr_spike,
)


SNR_SPIKE_THRESHOLD = 1_000.0
SNR_SPIKE_TOP_K = 3
SIGMA_MIN = 1e-4
SNR_CLIP = 250.0
PRED_STD_WARN_FACTOR = 5.0
FFT_HIGH_WARN_THRESHOLD = 0.8


def _cycle(loader: Iterable) -> Iterator:
    """Infinite iterator over a dataloader."""
    while True:
        for batch in loader:
            yield batch




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
    RECORDER_VERSION = "v1.1"
    config = load_config(config_path=config_path)
    apply_variant_override(config, variant)
    seed_everything(config)

    print("[Normalization] Disabled for diagnostic mode")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RecordTrainingSteps] Running version {RECORDER_VERSION}")

    data_loader = loader or build_dataloader(config)
    data_iter = _cycle(data_loader)

    model = build_model(config.get("model", {}))
    loss_fn = get_loss_fn(config.get("loss", {}))
    marker = getattr(loss_fn, "residual_marker", None)
    if callable(marker):
        print(marker())
    diffusion_cfg = config.get("diffusion", {}) or {}
    snr_weighting_cfg = diffusion_cfg.get("snr_weighting")
    force_weighting = diffusion_cfg.get("diagnostic_force_weighting", True)
    if hasattr(loss_fn, "set_weighting_enabled"):
        if force_weighting:
            loss_fn.set_weighting_enabled(True)
            snr_weighting = True
            print("[RecordTrainingSteps] forcing adaptive weighting on for diagnostics")
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
    optimiser = build_optimizer(model, config)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_obj)

    T = int(diffusion_cfg.get("num_timesteps", 1000))
    schedule = diffusion_cfg.get("beta_schedule", "cosine")
    schedule_kwargs: Dict[str, float] = dict(
        diffusion_cfg.get("schedule_kwargs", {}) or {}
    )
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

    coeffs = build_diffusion(T, schedule, schedule_kwargs)
    if schedule_key == "logsnr_cosine":
        lam_min = float(schedule_kwargs.get("lambda_min", -13.0))
        lam_max = float(schedule_kwargs.get("lambda_max", 13.0))
        delta = float(schedule_kwargs.get("delta", 0.008))
        print(
            f"[Schedule] mode=logsnr_cosine λ∈[{lam_min:.3g},{lam_max:.3g}] δ={delta:.3g}"
        )
    print(
        "[Schedule] trim_offset=%d num_timesteps=%d min_sigma=%.4f" % (
            coeffs.trim_offset,
            coeffs.num_timesteps,
            coeffs.min_safe_sigma,
        )
    )

    step_records: List[Dict[str, Any]] = []
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

    # Buffer for diagnostic log lines (if not verbose)
    log_buffer = []

    def _log_diag(step, tag, message):
        if tag == "SNR" and log_snr_json:
            log_buffer.append({"step": step, "tag": tag, "message": message})
        elif verbose_logs:
            print(message)
        else:
            log_buffer.append({"step": step, "tag": tag, "message": message})

    for step in range(steps):
        xb, _ = next(data_iter)
        xb = xb.to(device_obj)
        model.train()

        B = xb.shape[0]
        t = sample_timesteps(
            B,
            coeffs.num_timesteps,
            xb.device,
        )
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
                _log_diag(step, "Noise", f"[Noise] mode={corruption_mode}, snr_ratio={ratio_str}, mean={noisy_mean:.3f} std={noisy_std:.3f} mean_shift={dc_shift:+.4f}")
            channel_mean = xb.mean(dim=(2, 3), keepdim=True)
            signal_rms = (xb - channel_mean).pow(2).mean().sqrt().item()
            noise_rms = noise_term.pow(2).mean().sqrt().item()
            measured_snr = signal_rms / max(noise_rms, 1e-8)
            _log_diag(step, "FFTNoiseCheck", f"[FFTNoiseCheck] snr_target={effective_snr_ratio:.3f}, measured={measured_snr:.3f}, signal_rms={signal_rms:.3f}, noise_rms={noise_rms:.3f}")
            _log_diag(step, "FFTNoiseCheck", f"[FFTNoiseCheck] mean_shift={dc_shift:+.4f}")
            channel_noise_mean = noise_term.mean(dim=(0, 2, 3)).cpu().tolist()
            channel_noise_std = noise_term.std(dim=(0, 2, 3), unbiased=False).cpu().tolist()
            _log_diag(step, "FFTNoiseCheck", "[FFTNoiseCheck] channel_noise_mean="
                f"{[round(v, 4) for v in channel_noise_mean]} "
                "channel_noise_std="
                f"{[round(v, 4) for v in channel_noise_std]}")

        pred = model(x_t, t)
        target = compute_target(
            prediction_type,
            xb,
            x_t,
            effective_noise,
            sqrt_alpha_t,
            sqrt_one_minus_t,
        )

        # Reconstruct x0 from the model prediction depending on parameterization
        if prediction_type == "eps":
            denoised = _predict_x0(
                x_t,
                pred,  # model predicts noise (epsilon)
                sqrt_alpha_t,
                sqrt_one_minus_t,
            )
        elif prediction_type == "x0":
            denoised = pred  # model already predicts x0
        elif prediction_type == "v":
            # velocity parameterization: x0 = sqrt_alpha * x_t - sqrt(1-alpha) * v  (consistent with common defs)
            # Here we invert v = sqrt_alpha * eps - sqrt(1-alpha) * x0 => x0 = (sqrt_alpha * x_t - sqrt(1-alpha) * v)
            denoised = x_t * sqrt_alpha_t - pred * sqrt_one_minus_t
        else:
            denoised = None

        residual = pred - target
        adaptive_diag: Optional[Dict[str, float]] = None
        try:
            loss_result = loss_fn(pred, target, sqrt_alpha_t, sqrt_one_minus_t)
        except TypeError:
            weight = (
                compute_snr_weight(sqrt_alpha_t, sqrt_one_minus_t, snr_transform)
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

        # Adaptive SNR controller block moved after SNR and std statistics
        fft_feedback = compute_fft_feedback(pred, target, fft_norm=fft_norm)
        loss_value = float(loss.detach().cpu())
        mae_value = float(mae.detach().cpu())
        weight_stats = None
        if adaptive_diag:
            weight_stats = {
                key: float(value)
                for key, value in adaptive_diag.items()
                if isinstance(value, (int, float))
            }
            if adaptive_diag.get("log_event"):
                msg = (
                    "[AdaptiveSNR] "
                    f"step={step:04d} "
                    f"kappa={adaptive_diag.get('kappa', 0.0):.4e}, "
                    f"ema={adaptive_diag.get('ema', 0.0):.4e}, "
                    f"alpha_fac={adaptive_diag.get('alpha_fac', 1.0):.2f}, "
                    f"overflow={adaptive_diag.get('overflow', 0.0):.3f}, "
                    f"overflow_ema={adaptive_diag.get('overflow_ema', 0.0):.3f}, "
                    f"delta={adaptive_diag.get('delta', 0.0):.3e}, "
                    f"w_mean={adaptive_diag.get('mean_weight', 1.0):.3f}, "
                    f"w_max={adaptive_diag.get('max_weight', 1.0):.3f}"
                    + (" frozen" if adaptive_diag.get("frozen") else "")
                )
                _log_diag(step, "AdaptiveSNR", msg)
        elif snr_weighting:
            weight = compute_snr_weight(sqrt_alpha_t, sqrt_one_minus_t, snr_transform)
            weight_stats = {
                "snr_weight_min": float(weight.min().item()),
                "snr_weight_max": float(weight.max().item()),
                "snr_weight_mean": float(weight.mean().item()),
            }

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
            _log_diag(step, "OverflowHandler",
                f"[OverflowHandler] step={step} mode=deterministic "
                f"snr={min(snr_raw_max, SNR_CLIP):.1f} loss_mode=x0 count={overflow}"
            )

        snr_spike_summary = _summarise_snr_spikes(
            snr_vals=snr_vals.detach(),
            sqrt_alpha_t=sqrt_alpha_t.detach(),
            sqrt_one_minus_t=sqrt_one_minus_t.detach(),
            timesteps=t.detach(),
            clean=xb.detach(),
            noisy=x_t.detach(),
            noise=effective_noise.detach(),
            target=target.detach(),
            prediction=pred.detach(),
        )
        if snr_spike_summary:
            snr_message = (
                f"[SNR] step={step} mean={snr_mean_val:.4f} std={snr_vals.std().item():.4f} "
                f"max={snr_max_val:.4f} count={snr_spike_summary['count']} "
                f"top_timesteps={snr_spike_summary['top_timesteps']}"
            )
            if log_snr_json:
                log_buffer.append({
                    "step": step,
                    "tag": "SNR",
                    "mean": snr_mean_val,
                    "std": float(snr_vals.std().item()),
                    "max": snr_max_val,
                    "count": snr_spike_summary["count"],
                    "top_timesteps": snr_spike_summary["top_timesteps"],
                })
            elif verbose_logs:
                print(snr_message)
        sqrt_alpha_min = float(sqrt_alpha_t.min().item())
        sqrt_alpha_max = float(sqrt_alpha_t.max().item())
        sqrt_one_minus_min = float(sqrt_one_minus_t.min().item())
        sqrt_one_minus_max = float(sqrt_one_minus_t.max().item())

        target_mean = float(target.detach().mean().item())
        target_std = float(target.detach().std().item())
        target_abs_max = float(target.detach().abs().max().item())
        residual_mean = float(residual.detach().mean().item())
        residual_std = float(residual.detach().std().item())
        residual_abs_max = float(residual.detach().abs().max().item())

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _grad_norm(model)
        optimiser.step()

        param_delta = _parameter_delta(model, previous_state)
        output_fft = _fft_band_means(pred.detach())
        prediction_std_val = float(pred.detach().std().item())
        input_std_val = float(xb.detach().std().item())
        if input_std_val > 0:
            std_ratio = prediction_std_val / max(input_std_val, 1e-8)
            if std_ratio > PRED_STD_WARN_FACTOR:
                print(
                    "[WARN] Prediction std drift at step {step}: "
                    "std={std:.3f}, input_std={input_std:.3f}, ratio={ratio:.2f}".format(
                        step=step,
                        std=prediction_std_val,
                        input_std=input_std_val,
                        ratio=std_ratio,
                    )
                )
        else:
            std_ratio = float("inf")
        fft_high_val = output_fft.get("fft_high", float("nan"))
        if not math.isnan(fft_high_val) and fft_high_val > FFT_HIGH_WARN_THRESHOLD:
            print(
                f"[WARN] Spectral blowup suspected at step {step}: fft_high={fft_high_val:.3f}"
            )
        input_fft = _fft_band_means(xb.detach())
        noisy_fft = _fft_band_means(x_t.detach())
        corr = _structure_correlation(xb.detach(), x_t.detach())

        # --- Adaptive SNR controller (acts for the *next* step) ---
        adaptive_note = None
        # These will be filled for record/logging if adaptive_snr and controller is active
        headroom = None
        high_snr_fraction = None
        snr_mean_trend = None
        snr_max_trend = None
        if adaptive_snr and current_snr_ratio is not None:
            overflow = float(adaptive_diag.get("overflow", 0.0)) if adaptive_diag else 0.0
            overflow_ema = float(adaptive_diag.get("overflow_ema", 0.0)) if adaptive_diag else 0.0
            alpha_fac = float(adaptive_diag.get("alpha_fac", 1.0)) if adaptive_diag else 1.0
            kappa_val = float(adaptive_diag.get("kappa", 0.0)) if adaptive_diag else 0.0

            # --- compute trends and predictions ---
            # raw trends (this step - previous step)
            snr_mean_trend = None if prev_snr_mean is None else (snr_mean_val - prev_snr_mean)
            snr_max_trend = None if prev_snr_max is None else (snr_max_val - prev_snr_max)
            # EWMA of trends (smoother slope estimate)
            if snr_mean_trend is not None:
                snr_mean_slope_ema = _ewma(snr_mean_slope_ema, snr_mean_trend, snr_ema_beta)
            if snr_max_trend is not None:
                snr_max_slope_ema = _ewma(snr_max_slope_ema, snr_max_trend, snr_ema_beta)
            # headroom w.r.t. clip with safety margin
            snr_clip_margin = 0.85 * SNR_CLIP
            headroom = snr_clip_margin - snr_raw_max
            predicted_next_max = snr_raw_max + (snr_max_slope_ema if snr_max_slope_ema is not None else 0.0)

            # fraction of very high-SNR samples in the batch
            high_snr_fraction = float((snr_vals > (0.75 * SNR_CLIP)).float().mean().item())

            # --- decision thresholds & guards ---
            in_cooldown = (step - last_change_step) < cooldown_steps
            min_headroom_for_increase = 25.0
            max_trend_for_increase = 2.0
            max_high_frac_for_increase = 0.10
            # predict near-clip state for proactive downshift
            near_clip_predicted = (predicted_next_max >= (0.92 * SNR_CLIP))
            near_clip_current  = (snr_raw_max >= (0.90 * SNR_CLIP))
            std_ratio_ok = (prediction_std_val <= 0 or input_std_val <= 0) or ((prediction_std_val / max(input_std_val, 1e-8)) <= PRED_STD_WARN_FACTOR)

            # --- hysteresis arming logic ---
            # Arm decrease if risky now; require two consecutive risky indications to actually decrease.
            if (overflow > snr_overflow_high) or (overflow_ema > snr_overflow_high) or near_clip_predicted or near_clip_current or (not std_ratio_ok):
                dec_armed = True
            else:
                dec_armed = False

            # Arm increase if safe now; require two consecutive safe indications to actually increase.
            trend_ok = ((snr_mean_slope_ema is None or snr_mean_slope_ema < max_trend_for_increase)
                        and (snr_max_slope_ema is None or snr_max_slope_ema < max_trend_for_increase))
            headroom_ok = (headroom > min_headroom_for_increase) and (high_snr_fraction <= max_high_frac_for_increase)
            weighting_ok = (kappa_val >= snr_kappa_thresh) and (alpha_fac >= snr_alpha_fac_high) and (overflow_ema < snr_overflow_high)
            if trend_ok and headroom_ok and weighting_ok and not near_clip_predicted:
                inc_armed = True
            else:
                inc_armed = False

            # --- apply action with cooldown and proportional step size ---
            # Choose action with priority to stability (decrease wins ties)
            action_taken = None
            if not in_cooldown and dec_armed:
                # proportional down step if predicted to be near clip; else nominal snr_dec
                if near_clip_predicted or near_clip_current:
                    # larger step when dangerously close
                    delta = max(snr_dec, 0.25)
                else:
                    delta = snr_dec
                new_val = _clamp_snr(current_snr_ratio - float(delta))
                if new_val != current_snr_ratio:
                    adaptive_note = f"down:{current_snr_ratio:.3g}->{new_val:.3g} (pred_max={predicted_next_max:.1f}, raw_max={snr_raw_max:.1f}, overflow={overflow:.3f}/{overflow_ema:.3f})"
                    current_snr_ratio = new_val
                    last_change_step = step
                    action_taken = "down"
            elif not in_cooldown and inc_armed:
                # proportional up step based on headroom (smaller than previous design)
                # target headroom ~ 40 → scale to 0..2*snr_inc
                scale = 80.0
                prop = max(0.0, min(2.0 * snr_inc, (headroom - min_headroom_for_increase) / scale))
                delta = max(prop, 0.05)  # ensure a minimum tiny nudge
                new_val = _clamp_snr(current_snr_ratio + float(delta))
                if new_val != current_snr_ratio:
                    adaptive_note = f"up:{current_snr_ratio:.3g}->{new_val:.3g} (kappa={kappa_val:.2e}, alpha_fac={alpha_fac:.2f}, headroom={headroom:.1f}, high_frac={high_snr_fraction:.2f}, pred_max={predicted_next_max:.1f})"
                    current_snr_ratio = new_val
                    last_change_step = step
                    action_taken = "up"

            # Nothing changed: keep note of current assessment for logging clarity
            if adaptive_note is None:
                if dec_armed:
                    adaptive_note = f"hold (decrease-armed; pred_max={predicted_next_max:.1f})"
                elif inc_armed:
                    adaptive_note = f"hold (increase-armed; headroom={headroom:.1f})"

            # Update SNR trend state for next step
            prev_snr_mean = snr_mean_val
            prev_snr_max = snr_max_val

        record: Dict[str, Any] = {
            "step": step,
            "loss": loss_value,
            "mae": mae_value,
            "grad_norm": grad_norm,
            "param_delta": param_delta,
            "noise_norm": float(
                effective_noise.view(B, -1).norm(dim=1).mean().cpu()
            ),
            "output_mean": float(pred.detach().mean().cpu()),
            "output_std": float(pred.detach().std().cpu()),
            "structure_corr": corr,
            "phase_rms": _phase_rms(xb.detach(), x_t.detach(), norm=fft_norm),
            "prediction_type": prediction_type,
            "timestep_min": timestep_min,
            "timestep_max": timestep_max,
            "timestep_mean": timestep_mean,
            "sqrt_alpha_min": sqrt_alpha_min,
            "sqrt_alpha_max": sqrt_alpha_max,
            "sqrt_one_minus_min": sqrt_one_minus_min,
            "sqrt_one_minus_max": sqrt_one_minus_max,
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
            record["noise_channel_std_min"] = noise_stats.get("noise_channel_std_min")
            record["noise_channel_std_max"] = noise_stats.get("noise_channel_std_max")
        record.update({f"output_{k}": v for k, v in output_fft.items()})
        record.update({f"input_{k}": v for k, v in input_fft.items()})
        record.update({f"noisy_{k}": v for k, v in noisy_fft.items()})
        if denoised is not None:
            denoised_corr = _structure_correlation(xb.detach(), denoised.detach())
            record["denoised_corr"] = denoised_corr
            record["denoised_mse"] = float((xb.detach() - denoised.detach()).pow(2).mean().item())

        if effective_snr_ratio is not None:
            record["snr_ratio"] = effective_snr_ratio
        if snr_spike_summary:
            record["snr_spike_count"] = snr_spike_summary["count"]
            record["snr_spike_max"] = snr_spike_summary["max_snr"]
            record["snr_spike_top_timesteps"] = snr_spike_summary["top_timesteps"]

        for key, value in fft_feedback.items():
            record[f"fft_{key}"] = float(value)
        if weight_stats:
            record.update(weight_stats)
        record["snr_ratio_effective"] = current_snr_ratio
        if adaptive_note is not None:
            record["adaptive_noise_action"] = adaptive_note
        step_records.append(record)

        if uniform_corruption and corr < 0.4:
            _log_diag(step, "StructureCorrelation", f"⚠️  Step {step}: structure correlation low ({corr:.2f})")

        if step % log_interval == 0 or step == steps - 1:
            save_root = out_dir / f"step_{step:04d}"
            save_root.mkdir(parents=True, exist_ok=True)

            _log_diag(step, "Loss", f"[Loss] step={step} loss={loss_value:.6f} mae={mae_value:.6f}")
            _log_diag(step, "FFTFeedback",
                "[FFTFeedback] " +
                ", ".join(
                    f"{name}={fft_feedback[name]:.6f}"
                    for name in [
                        "amplitude_mae",
                        "phase_mae",
                        "real_mae",
                        "imag_mae",
                        "complex_mae",
                    ]
                )
            )
            _log_diag(step, "Timesteps",
                "[Timesteps] min={:d} max={:d} mean={:.1f} "
                "sqrt_alpha_min={:.4f} sqrt_alpha_max={:.4f} "
                "snr_min={:.4f} snr_max={:.4f}".format(
                    timestep_min,
                    timestep_max,
                    timestep_mean,
                    sqrt_alpha_min,
                    sqrt_alpha_max,
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
                    _log_diag(step, "AdaptiveSNRWeight",
                        "[AdaptiveSNRWeight] mean={:.6f} max={:.6f} kappa={:.4e} "
                        "alpha_fac={:.2f} overflow={:.3f} overflow_ema={:.3f} "
                        "delta={:.3e}{}".format(
                            weight_stats.get("mean_weight", 1.0),
                            weight_stats.get("max_weight", 1.0),
                            weight_stats.get("kappa", 0.0),
                            weight_stats.get("alpha_fac", 1.0),
                            weight_stats.get("overflow", 0.0),
                            weight_stats.get("overflow_ema", 0.0),
                            weight_stats.get("delta", 0.0),
                            " frozen" if weight_stats.get("frozen") else "",
                        )
                    )

            def _save_and_log(tensor: torch.Tensor, path: Path) -> None:
                tensor = tensor.detach().cpu()
                span = tensor.max() - tensor.min()
                scaled = (tensor - tensor.min()) / (span + 1e-8)
                save_image(scaled, path)

            _save_and_log(xb, save_root / "input.png")
            _save_and_log(x_t, save_root / "noisy.png")
            _save_and_log(pred, save_root / "predicted_noise.png")
            if denoised is not None:
                _save_and_log(denoised, save_root / "prediction.png")
            else:
                _save_and_log(pred, save_root / "prediction.png")
            _log_diag(step, "Save", f"[Save] step={step} images saved (input, noisy, predicted_noise, prediction)")
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
            amp_low = fft_feedback.get("amplitude_low_mae", float("nan"))
            amp_mid = fft_feedback.get("amplitude_mid_mae", float("nan"))
            amp_high = fft_feedback.get("amplitude_high_mae", float("nan"))
            phase_low = fft_feedback.get("phase_low_mae", float("nan"))
            phase_mid = fft_feedback.get("phase_mid_mae", float("nan"))
            phase_high = fft_feedback.get("phase_high_mae", float("nan"))
            _log_diag(step, "FFTBands",
                f"[FFTBands] amp_low={amp_low:.6f}, amp_mid={amp_mid:.6f}, amp_high={amp_high:.6f}, "
                f"phase_low={phase_low:.6f}, phase_mid={phase_mid:.6f}, phase_high={phase_high:.6f}"
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

    # Write diagnostic log buffer to file if not verbose
    if not verbose_logs and log_buffer:
        debug_log_path = out_dir / "debug_log.jsonl"
        with debug_log_path.open("w", encoding="utf-8") as fh:
            for entry in log_buffer:
                fh.write(json.dumps(entry))
                fh.write("\n")

    # Write SNR log if requested
    if log_snr_json:
        snr_entries = [
            entry for entry in log_buffer
            if entry.get("tag") == "SNR" and all(k in entry for k in ("mean", "std", "max", "count", "top_timesteps"))
        ]
        if snr_entries:
            snr_log_path = out_dir / "snr_log.jsonl"
            with snr_log_path.open("w", encoding="utf-8") as fh:
                for entry in snr_entries:
                    fh.write(json.dumps({
                        "step": entry["step"],
                        "mean": entry["mean"],
                        "std": entry["std"],
                        "max": entry["max"],
                        "count": entry["count"],
                        "top_timesteps": entry["top_timesteps"],
                    }))
                    fh.write("\n")

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
    parser.add_argument("--adaptive-snr", action="store_true", help="Dynamically adjust diffusion.snr_ratio during diagnostics using adaptive SNR signals.")
    parser.add_argument("--snr-min", type=float, default=0.5, help="Lower bound for adaptive snr_ratio.")
    parser.add_argument("--snr-max", type=float, default=2.5, help="Upper bound for adaptive snr_ratio.")
    parser.add_argument("--snr-inc", type=float, default=0.1, help="Increment applied to snr_ratio when conditions indicate more signal (cleaner input) is safe.")
    parser.add_argument("--snr-dec", type=float, default=0.2, help="Decrement applied to snr_ratio when conditions indicate instability; lowers SNR (adds noise).")
    parser.add_argument("--snr-kappa-thresh", type=float, default=2.5e-1, help="If kappa exceeds this, we consider increasing SNR.")
    parser.add_argument("--snr-alpha-fac-high", type=float, default=1.12, help="If alpha_fac exceeds this along with kappa, we consider increasing SNR.")
    parser.add_argument("--snr-overflow-high", type=float, default=0.05, help="If overflow (or overflow_ema) exceeds this, we decrease SNR.")
    parser.add_argument("--verbose-logs", action="store_true", help="Print detailed per-step diagnostic logs to stdout. If not set, logs are written to debug_log.jsonl in output-dir.")
    parser.add_argument("--log-snr-json", action="store_true", help="Log per-step SNR statistics to snr_log.jsonl instead of printing.")
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
        adaptive_snr=args.adaptive_snr,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        snr_inc=args.snr_inc,
        snr_dec=args.snr_dec,
        snr_kappa_thresh=args.snr_kappa_thresh,
        snr_alpha_fac_high=args.snr_alpha_fac_high,
        snr_overflow_high=args.snr_overflow_high,
        verbose_logs=getattr(args, "verbose_logs", False),
        log_snr_json=getattr(args, "log_snr_json", False),
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
