#!/usr/bin/env python3
"""Micro-benchmark harness for the SpectralBridge."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from spectral.bridge import SpectralBridge

WARMUP_RUNS = 10
BENCH_RUNS = 200


@dataclass
class TimingSummary:
    median_ms: float
    mean_ms: float
    std_ms: float

    @classmethod
    def from_durations(cls, durations: Iterable[float]) -> "TimingSummary":
        samples = list(durations)
        median_ms = statistics.median(samples) * 1000.0
        mean_ms = statistics.fmean(samples) * 1000.0
        std_ms = statistics.pstdev(samples) * 1000.0 if len(samples) > 1 else 0.0
        return cls(median_ms=median_ms, mean_ms=mean_ms, std_ms=std_ms)


def _time_callable(fn, warmup: int = WARMUP_RUNS, runs: int = BENCH_RUNS) -> Tuple[List[float], List[Dict[str, float | int | str]]]:
    for _ in range(warmup):
        fn()
    durations: List[float] = []
    profiles: List[Dict[str, float | int | str]] = []
    for _ in range(runs):
        start = time.perf_counter()
        profile = fn()
        durations.append(time.perf_counter() - start)
        if profile is not None:
            profiles.append(profile)
    return durations, profiles


def benchmark_shape(bridge: SpectralBridge, height: int, width: int) -> Dict[str, object]:
    """Benchmark a single spatial configuration."""

    shape = (height, width)
    print(f"\nBenchmarking FFT2 for input shape {shape}")

    torch_input = torch.randn(shape, dtype=torch.float32)
    numpy_input = torch_input.numpy()

    numpy_durations, _ = _time_callable(lambda: np.fft.fft2(numpy_input))
    numpy_stats = TimingSummary.from_durations(numpy_durations)

    torch_durations, _ = _time_callable(lambda: torch.fft.fft2(torch_input))
    torch_stats = TimingSummary.from_durations(torch_durations)

    def run_bridge() -> Dict[str, float | int | str]:
        tensor = torch_input.clone()
        _, profile = bridge.profile_fft2(tensor)
        return profile.as_dict()

    bridge_durations, bridge_profiles = _time_callable(run_bridge)
    bridge_stats = TimingSummary.from_durations(bridge_durations)

    ffi_overhead_ms = float(
        np.median([p["ffi_s"] for p in bridge_profiles]) * 1000.0
    ) if bridge_profiles else 0.0
    conversion_ms = float(
        np.median([p["conversion_in_s"] + p["conversion_out_s"] for p in bridge_profiles]) * 1000.0
    ) if bridge_profiles else 0.0

    report = {
        "shape": shape,
        "torch_direct": torch_stats.__dict__,
        "numpy": numpy_stats.__dict__,
        "bridge": bridge_stats.__dict__,
        "ffi_overhead_ms": ffi_overhead_ms,
        "conversion_overhead_ms": conversion_ms,
        "profiles": bridge_profiles,
    }
    return report


def main() -> int:
    bridge = SpectralBridge()
    diagnostics = bridge.diagnostics()
    print("Spectral bridge diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")

    shapes = [(256, 256), (512, 512), (1024, 1024)]
    results = [benchmark_shape(bridge, h, w) for h, w in shapes]

    output = {
        "diagnostics": diagnostics,
        "results": results,
    }

    output_path = Path("results/bridge_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved benchmark results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
