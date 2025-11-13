"""Torch-friendly wrappers around the RIFFT core."""

from __future__ import annotations

import argparse
import time
from typing import Iterable, Sequence, Tuple

import torch

from . import dlpack_utils
from . import _internal  # type: ignore

_HANDLE = _internal.Handle()


def _to_capsule(tensor: torch.Tensor):
    tensor = dlpack_utils.ensure_fft_ready(tensor)
    return dlpack_utils.to_dlpack(tensor)


def _from_capsule(capsule) -> torch.Tensor:
    return dlpack_utils.from_dlpack(capsule)


def fft2(tensor: torch.Tensor) -> torch.Tensor:
    """Compute a 2-D FFT via RIFFT (returns a new tensor)."""
    capsule = _to_capsule(tensor)
    out_capsule = _HANDLE.fft2(capsule)
    return _from_capsule(out_capsule)


def ifft2(tensor: torch.Tensor) -> torch.Tensor:
    capsule = _to_capsule(tensor)
    out_capsule = _HANDLE.ifft2(capsule)
    return _from_capsule(out_capsule)


def fft_filter_ifft(signal: torch.Tensor, filt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    signal_capsule = _to_capsule(signal)
    filter_capsule = _to_capsule(filt)
    out_capsule, filter_capsule = _HANDLE.fft_filter_ifft(signal_capsule, filter_capsule)
    return _from_capsule(out_capsule), _from_capsule(filter_capsule)


def batched_fft2(tensor: torch.Tensor) -> torch.Tensor:
    return fft2(tensor)


def batched_ifft2(tensor: torch.Tensor) -> torch.Tensor:
    return ifft2(tensor)


def batched_fft_filter_ifft(signal: torch.Tensor, filt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return fft_filter_ifft(signal, filt)


def get_version() -> str:
    return getattr(_internal, "__version__", "0.0.0")


def run_benchmarks(sizes: Sequence[int], iters: int = 50, device: str = "cpu"):
    results = []
    for size in sizes:
        shape = (size, size)
        data = torch.randn(shape, dtype=torch.complex64, device=device)
        start = time.perf_counter()
        tmp = data
        for _ in range(iters):
            tmp = fft2(tmp)
        elapsed = time.perf_counter() - start
        results.append({
            "size": size,
            "ms_per_call": (elapsed / iters) * 1000.0,
        })
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RIFFT Python benchmarks")
    parser.add_argument("--sizes", nargs="*", type=int, default=[256, 512, 1024])
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(list(argv) if argv is not None else None)
    results = run_benchmarks(args.sizes, args.iters, args.device)
    for row in results:
        print(f"{row['size']}^2 : {row['ms_per_call']:.4f} ms")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
