#!/usr/bin/env python3
"""Benchmark baseline vs. spectral TinyUNet forward pass throughput."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.model_unet_tiny import TinyUNet  # noqa: E402


# =============================================================
# FFT Benchmark Module — Spectral Diffusion v0.2
# -------------------------------------------------------------
# Measures FFT/iFFT overhead and compares to model runtime.
# Adds support for runtime fraction metrics and file export.
# =============================================================


def build_config(spectral_enabled: bool, weighting: str, shape: Tuple[int, int, int]) -> Dict:
    channels, height, width = shape
    return {
        "channels": channels,
        "base_channels": 32,
        "depth": 2,
        "data": {"channels": channels, "height": height, "width": width},
        "spectral": {
            "enabled": spectral_enabled,
            "normalize": True,
            "weighting": weighting if spectral_enabled else "none",
        },
    }


def benchmark_model(
    model: TinyUNet,
    device: torch.device,
    batch_size: int,
    input_shape: Tuple[int, int, int],
    iterations: int,
    warmup: int,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    model.eval()
    channels, height, width = input_shape
    x = torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)

    timings = []
    for i in range(iterations + warmup):
        start = time.perf_counter()
        with torch.no_grad():
            out = model(x, t=None)
            if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            timings.append(elapsed)
    avg = statistics.mean(timings)
    std = statistics.pstdev(timings) if len(timings) > 1 else 0.0

    stats = getattr(model, "spectral_stats", lambda: {})()
    calls = stats.get("spectral_calls", 0.0)
    total_fft = stats.get("spectral_time_seconds", 0.0)

    return {
        "mean_seconds": avg,
        "std_seconds": std,
        "images_per_second": batch_size / avg if avg > 0 else 0.0,
        "spectral_calls": calls,
        "spectral_time_seconds": total_fft,
    }


def benchmark_fft(size: int = 512, batch: int = 8, device: str = "cuda", dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """Benchmark pure FFT operations."""
    device_obj = torch.device(device)
    x = torch.randn(batch, 3, size, size, device=device_obj, dtype=dtype)

    if device_obj.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    X = torch.fft.fft2(x)
    y = torch.fft.ifft2(X)
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    fft_time = time.perf_counter() - start

    return {
        "size": size,
        "batch": batch,
        "fft_time": fft_time,
        "device": device_obj.type,
        "dtype": str(dtype),
    }


def benchmark_fft_over_model(
    model: TinyUNet,
    size: int = 512,
    batch: int = 8,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Measure total model runtime and compute FFT overhead fraction."""
    device_obj = torch.device(device)
    x = torch.randn(batch, 3, size, size, device=device_obj, dtype=dtype, requires_grad=True)
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = model(x)
    out.mean().backward()
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    total_t = time.time() - t0

    fft_info = benchmark_fft(size=size, batch=batch, device=device, dtype=dtype)
    fft_fraction = fft_info["fft_time"] / total_t if total_t > 0 else 0.0
    fft_info.update({
        "model_time": total_t,
        "fft_fraction_runtime": fft_fraction,
        "device": device_obj.type,
        "dtype": str(dtype),
    })

    print(f"Model total time: {total_t:.6f} seconds")
    print(f"FFT time: {fft_info['fft_time']:.6f} seconds")
    print(f"FFT fraction of runtime: {fft_fraction:.4f}")

    return fft_info


def save_json(result, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TinyUNet spectral overhead.")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for benchmark.")
    parser.add_argument("--channels", type=int, default=3, help="Input channels.")
    parser.add_argument("--height", type=int, default=32, help="Input height.")
    parser.add_argument("--width", type=int, default=32, help="Input width.")
    parser.add_argument("--iterations", type=int, default=20, help="Measured iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument(
        "--spectral-weighting",
        default="radial",
        choices=["none", "radial", "bandpass"],
        help="Spectral weighting to apply when spectral mode is enabled.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "spectral"],
        choices=["baseline", "spectral"],
        help="Which configurations to benchmark.",
    )
    parser.add_argument("--size", type=int, help="Image size for FFT-only benchmark.")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32", "float64"], help="Data type for benchmarking.")
    parser.add_argument("--save", type=str, help="Save FFT benchmark results to JSON file.")
    args = parser.parse_args()

    device = torch.device(args.device)
    shape = (args.channels, args.height, args.width)
    dtype = getattr(torch, args.dtype)

    # Handle FFT-only benchmark
    if args.size is not None:
        result = benchmark_fft(size=args.size, batch=args.batch_size, device=args.device, dtype=dtype)
        print(f"FFT-only benchmark: size={args.size}, batch={args.batch_size}, dtype={args.dtype}, device={args.device}")
        print(f"FFT time: {result['fft_time']:.6f} seconds")

        if args.save:
            save_json(result, args.save)
        return

    # Original model benchmarking
    results = []
    for mode in args.modes:
        spectral_enabled = mode == "spectral"
        config = build_config(spectral_enabled, args.spectral_weighting, shape)
        model = TinyUNet(config).to(device).to(dtype)
        metrics = benchmark_model(
            model=model,
            device=device,
            batch_size=args.batch_size,
            input_shape=shape,
            iterations=args.iterations,
            warmup=args.warmup,
            dtype=dtype,
        )
        results.append((mode, metrics))

    print("Mode\tMean(s)\tStd(s)\tImgs/s\tSpectral Calls\tSpectral Time(s)")
    for mode, metrics in results:
        print(
            f"{mode}\t{metrics['mean_seconds']:.6f}\t{metrics['std_seconds']:.6f}\t"
            f"{metrics['images_per_second']:.2f}\t{metrics['spectral_calls']:.0f}\t"
            f"{metrics['spectral_time_seconds']:.6f}"
        )

    # Additionally benchmark FFT overhead over model for spectral mode if present
    if "spectral" in args.modes:
        spectral_enabled = True
        config = build_config(spectral_enabled, args.spectral_weighting, shape)
        model = TinyUNet(config).to(device).to(dtype)
        fft_over_model_result = benchmark_fft_over_model(
            model=model,
            size=max(shape[1], shape[2]),
            batch=args.batch_size,
            device=args.device,
            dtype=dtype,
        )
        print("\nFFT overhead over model runtime:")
        print(json.dumps(fft_over_model_result, indent=2))

        if args.save:
            base, ext = os.path.splitext(args.save)
            fft_save_path = f"{base}_fft_over_model{ext}"
            save_json(fft_over_model_result, fft_save_path)


if __name__ == "__main__":
    main()
