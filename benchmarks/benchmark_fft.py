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
    args = parser.parse_args()

    device = torch.device(args.device)
    shape = (args.channels, args.height, args.width)

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
