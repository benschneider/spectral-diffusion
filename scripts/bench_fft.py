import argparse
import time

import numpy as np
import torch

if torch.cuda.is_available():
    import torch.backends.cudnn

try:
    import pyfftw
except ImportError:  # pragma: no cover - we report absence
    pyfftw = None


def numpy_fft(batch: int, channels: int, size: int, runs: int) -> float:
    data = np.random.randn(batch, channels, size, size)
    start = time.perf_counter()
    for _ in range(runs):
        np.fft.fft2(data)
    end = time.perf_counter()
    return end - start


def torch_fft(batch: int, channels: int, size: int, runs: int, device: str = "cpu") -> float:
    tensor = torch.randn(batch, channels, size, size, device=device)
    start = time.perf_counter()
    for _ in range(runs):
        torch.fft.fft2(tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start


def main() -> None:
    parser = argparse.ArgumentParser(description="FFT benchmarking utility")
    parser.add_argument("--size", type=int, default=256, help="Spatial dimension (H=W)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--channels", type=int, default=3, help="Channel count")
    parser.add_argument("--runs", type=int, default=20, help="Number of FFT calls")
    parser.add_argument("--cuda", action="store_true", help="Benchmark CUDA, if available")
    parser.add_argument("--pyfftw", action="store_true", help="Benchmark pyFFTW if installed")
    args = parser.parse_args()

    print(f"Benchmarking FFTs for size={args.size}, batch={args.batch}, channels={args.channels}")

    elapsed_torch_cpu = torch_fft(args.batch, args.channels, args.size, args.runs, device="cpu")
    print(f"torch.fft.fft2 (CPU): {elapsed_torch_cpu:.4f}s total, {elapsed_torch_cpu/args.runs:.6f}s per call")

    if args.cuda and torch.cuda.is_available():
        elapsed_torch_cuda = torch_fft(args.batch, args.channels, args.size, args.runs, device="cuda")
        print(
            f"torch.fft.fft2 (CUDA): {elapsed_torch_cuda:.4f}s total, {elapsed_torch_cuda/args.runs:.6f}s per call"
        )
    elif args.cuda:
        print("CUDA requested but not available")

    elapsed_numpy = numpy_fft(args.batch, args.channels, args.size, args.runs)
    print(f"numpy.fft.fft2: {elapsed_numpy:.4f}s total, {elapsed_numpy/args.runs:.6f}s per call")

    if args.pyfftw:
        if pyfftw is None:
            print("pyFFTW not installed")
        else:
            array = pyfftw.empty_aligned((args.batch, args.channels, args.size, args.size), dtype="complex64")
            fft_object = pyfftw.builders.fft2(array, threads=4)
            start = time.perf_counter()
            for _ in range(args.runs):
                fft_object()
            end = time.perf_counter()
            elapsed_pyfftw = end - start
            print(
                f"pyFFTW fft2: {elapsed_pyfftw:.4f}s total, {elapsed_pyfftw/args.runs:.6f}s per call"
            )


if __name__ == "__main__":
    main()
