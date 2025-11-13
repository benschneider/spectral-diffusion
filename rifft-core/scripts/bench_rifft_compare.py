#!/usr/bin/env python3
"""Compare riff_core FFT timings against torch.fft and numpy.fft."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from riff_core import fft2 as riff_fft2

WARMUP_RUNS = 10
BENCH_RUNS = 1000
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TimingSummary:
    median_ms: float
    mean_ms: float
    std_ms: float

    @classmethod
    def from_samples(cls, samples: Iterable[float]) -> "TimingSummary":
        data = list(samples)
        return cls(
            median_ms=statistics.median(data) * 1000.0,
            mean_ms=statistics.fmean(data) * 1000.0,
            std_ms=statistics.pstdev(data) * 1000.0 if len(data) > 1 else 0.0,
        )


def _time(fn) -> List[float]:
    for _ in range(WARMUP_RUNS):
        fn()
    out: List[float] = []
    for _ in range(BENCH_RUNS):
        start = time.perf_counter()
        fn()
        out.append(time.perf_counter() - start)
    return out


def benchmark_shape(height: int, width: int) -> Dict[str, object]:
    shape = (height, width)
    print(f"\nBenchmarking shape {shape}")
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.complex64)
    numpy_input = torch_input.numpy()

    numpy_stats = TimingSummary.from_samples(_time(lambda: np.fft.fft2(numpy_input)))
    torch_stats = TimingSummary.from_samples(_time(lambda: torch.fft.fft2(torch_input)))

    def run_rifft() -> None:
        tensor = torch_input.clone()
        riff_fft2(tensor)

    riff_stats = TimingSummary.from_samples(_time(run_rifft))

    return {
        "shape": shape,
        "numpy": numpy_stats.__dict__,
        "torch": torch_stats.__dict__,
        "riff_core": riff_stats.__dict__,
    }


def print_report(entries: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 80)
    print("RIFFT Core Benchmark Summary")
    print("=" * 80)
    header = f"{'Shape':<12} {'Impl':<12} {'Median (ms)':>14} {'Mean (ms)':>12} {'Std (ms)':>10}"
    print(header)
    print("-" * len(header))

    def row(stats: Dict[str, float], label: str, shape: Tuple[int, int]) -> None:
        print(
            f"{str(shape):<12} {label:<12} "
            f"{stats['median_ms']:>14.3f} {stats['mean_ms']:>12.3f} {stats['std_ms']:>10.3f}"
        )

    for entry in entries:
        shape = entry["shape"]
        row(entry["torch"], "torch.fft", shape)
        row(entry["numpy"], "numpy", shape)
        row(entry["riff_core"], "riff_core", shape)
        print("-" * len(header))


def main() -> int:
    shapes = [(256, 256), (512, 512), (1024, 1024)]
    results = [benchmark_shape(h, w) for h, w in shapes]
    out_path = ROOT / "results" / "rifft_core_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")
    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
