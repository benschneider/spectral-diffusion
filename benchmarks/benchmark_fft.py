#!/usr/bin/env python3
"""Benchmark baseline vs. spectral TinyUNet forward pass throughput."""

from __future__ import annotations

import argparse
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
) -> Dict[str, float]:
    model.eval()
    channels, height, width = input_shape
    x = torch.randn(batch_size, channels, height, width, device=device)

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
    device = torch.device(device)
    x = torch.randn(batch, 3, size, size, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    X = torch.fft.fft2(x)
    y = torch.fft.ifft2(X)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fft_time = time.perf_counter() - start

    return {
        "size": size,
        "batch": batch,
        "fft_time": fft_time,
        "device": device.type,
        "dtype": str(dtype),
    }


def benchmark_fft_over_model(model: TinyUNet, size: int = 512, batch: int = 8, device: str = "cuda") -> Dict[str, float]:
    """Benchmark FFT overhead within a full model forward pass."""
    device = torch.device(device)
    x = torch.randn(batch, 3, size, size, device=device, requires_grad=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    out = model(x, t=None)
    out.mean().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    # Get spectral stats from model
    stats = getattr(model, "spectral_stats", lambda: {})()
    fft_time = stats.get("spectral_time_seconds", 0.0)
    fft_fraction = fft_time / total_time if total_time > 0 else 0.0

    print(".2%")

    return {
        "size": size,
        "batch": batch,
        "model_time": total_time,
        "fft_time": fft_time,
        "fft_fraction": fft_fraction,
        "device": device.type,
    }


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
    parser.add_argument("--save", type=str, help="Save FFT benchmark results to JSON file.")
    args = parser.parse_args()

    device = torch.device(args.device)
    shape = (args.channels, args.height, args.width)

    # Handle FFT-only benchmark
    if args.size is not None:
        result = benchmark_fft(size=args.size, batch=args.batch_size, device=args.device)
        print(f"FFT-only benchmark: size={args.size}, batch={args.batch_size}")
        print(".6f")
        print(".2f")

        if args.save:
            import json
            with open(args.save, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.save}")
        return

    # Original model benchmarking
    results = []
    for mode in args.modes:
        spectral_enabled = mode == "spectral"
        config = build_config(spectral_enabled, args.spectral_weighting, shape)
        model = TinyUNet(config).to(device)
        metrics = benchmark_model(
            model=model,
            device=device,
            batch_size=args.batch_size,
            input_shape=shape,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        results.append((mode, metrics))

    print("Mode\tMean(s)\tStd(s)\tImgs/s\tSpectral Calls\tSpectral Time(s)")
    for mode, metrics in results:
        print(
            f"{mode}\t{metrics['mean_seconds']:.6f}\t{metrics['std_seconds']:.6f}\t"
            f"{metrics['images_per_second']:.2f}\t{metrics['spectral_calls']:.0f}\t"
            f"{metrics['spectral_time_seconds']:.6f}"
        )


if __name__ == "__main__":
    main()
