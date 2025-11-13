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

try:  # Optional RIFFT core bindings built via maturin
    from riff_core import fft2 as riff_fft2
    HAS_RIFF_CORE = True
except Exception:  # pragma: no cover - benchmark helper
    riff_fft2 = None
    HAS_RIFF_CORE = False

WARMUP_RUNS = 10
BENCH_RUNS = 1000


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


def benchmark_shape(bridge: SpectralBridge, height: int, width: int, diagnostics: Dict[str, object] | None = None) -> Dict[str, object]:
    """Benchmark a single spatial configuration."""

    shape = (height, width)
    print(f"\nBenchmarking FFT2 for input shape {shape}")

    # Reset Rust timing stats before benchmark
    bridge.reset_timing_stats()

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

    riff_stats: TimingSummary | None = None
    if HAS_RIFF_CORE:
        def run_riff_core() -> None:
            tensor = torch_input.clone()
            riff_fft2(tensor)

        riff_durations, _ = _time_callable(run_riff_core)
        riff_stats = TimingSummary.from_durations(riff_durations)

    # Get updated diagnostics with timing stats after benchmark
    updated_diagnostics = bridge.diagnostics()

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
        "riff_core": riff_stats.__dict__ if riff_stats else None,
        "ffi_overhead_ms": ffi_overhead_ms,
        "conversion_overhead_ms": conversion_ms,
        "profiles": bridge_profiles,
        "diagnostics": updated_diagnostics,  # Use updated diagnostics with timing data
    }
    return report


def _print_report(results: List[Dict[str, object]]) -> None:
    """Pretty-print the aggregated benchmark results to stdout."""

    def format_stats(stats: Dict[str, float]) -> Tuple[float, float, float]:
        return (
            float(stats["median_ms"]),
            float(stats["mean_ms"]),
            float(stats["std_ms"]),
        )

    print("\n" + "=" * 86)
    print("SpectralBridge Benchmark Summary")
    print("=" * 86)
    header = f"{'Shape':<12} {'Impl':<12} {'Median (ms)':>14} {'Mean (ms)':>12} {'Std (ms)':>10}"
    print(header)
    print("-" * len(header))

    for entry in results:
        shape = entry["shape"]
        impl_rows = [
            (entry["torch_direct"], "torch.fft"),
            (entry["numpy"], "numpy"),
        ]
        riff_entry = entry.get("riff_core")
        if riff_entry:
            impl_rows.append((riff_entry, "riff_core"))
        impl_rows.append((entry["bridge"], "bridge"))

        for impl, label in impl_rows:
            median_ms, mean_ms, std_ms = format_stats(impl)
            print(
                f"{str(shape):<12} {label:<12} "
                f"{median_ms:>14.3f} {mean_ms:>12.3f} {std_ms:>10.3f}"
            )

        bridge_stats = entry["bridge"]
        bridge_median = float(bridge_stats["median_ms"])
        ffi_ms = float(entry["ffi_overhead_ms"])
        conv_ms = float(entry["conversion_overhead_ms"])
        ffi_pct = (ffi_ms / bridge_median * 100.0) if bridge_median else 0.0
        conv_pct = (conv_ms / bridge_median * 100.0) if bridge_median else 0.0

        # Check for detailed Rust timing breakdown
        diagnostics = entry.get("diagnostics", {})
        rust_breakdown = diagnostics.get("rust_timing_breakdown")
        if rust_breakdown:
            fft_compute_ms = rust_breakdown.get("fft_compute", 0) * 1000
            data_transfer_ms = rust_breakdown.get("data_transfer", 0) * 1000
            data_movement_ms = rust_breakdown.get("data_movement", 0) * 1000
            total_rust_ms = fft_compute_ms + data_transfer_ms + data_movement_ms

            if total_rust_ms > 0:
                fft_pct = (fft_compute_ms / total_rust_ms * 100.0)
                transfer_pct = (data_transfer_ms / total_rust_ms * 100.0)
                movement_pct = (data_movement_ms / total_rust_ms * 100.0)

                print(
                    f"{'':<12} {'↳ Rust breakdown':<15} "
                    f"FFT {fft_compute_ms:.3f}ms ({fft_pct:>5.1f}%), "
                    f"Transfer {data_transfer_ms:.3f}ms ({transfer_pct:>5.1f}%), "
                    f"Movement {data_movement_ms:.3f}ms ({movement_pct:>5.1f}%)"
                )
            else:
                print(
                    f"{'':<12} {'↳ breakdown':<12} "
                    f"FFI {ffi_ms:.3f} ms ({ffi_pct:>5.1f}%), "
                    f"DLPack {conv_ms:.3f} ms ({conv_pct:>5.1f}%)"
                )
        else:
            print(
                f"{'':<12} {'↳ breakdown':<12} "
                f"FFI {ffi_ms:.3f} ms ({ffi_pct:>5.1f}%), "
                f"DLPack {conv_ms:.3f} ms ({conv_pct:>5.1f}%)"
            )
        print("-" * len(header))

    print(
        "Note: timings are medians across runs; breakdown uses median FFI/conversion costs.\n"
    )


def main() -> int:
    bridge = SpectralBridge()
    diagnostics = bridge.diagnostics()
    print("Spectral bridge diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")

    shapes = [(256, 256), (512, 512), (1024, 1024)]
    results = [benchmark_shape(bridge, h, w, diagnostics) for h, w in shapes]

    output = {
        "diagnostics": diagnostics,
        "results": results,
    }

    output_path = Path("results/bridge_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved benchmark results to {output_path}")
    _print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
