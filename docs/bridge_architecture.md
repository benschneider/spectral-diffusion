# Spectral Bridge Architecture

This document describes the end-to-end execution path for the **SpectralBridge**
module, spanning the Python interface, the Rust/PyO3 extension, and the FFTW
execution engine. The goals of the redesign are:

- minimise Python ↔ Rust call overhead;
- guarantee zero-copy transfers through DLPack capsules in both directions;
- support plan caching, threading and precision alignment for FFTW;
- expose simple diagnostics and benchmarking hooks.

## Data Flow

```
PyTorch Tensor (torch.Tensor, CPU, contiguous)
          │
          │ torch.utils.dlpack.to_dlpack(x)
          ▼
DLPack capsule (PyCapsule "dltensor")
          │
          │ spectral_core.fft2_dlpack(capsule)
          ▼
Rust BorrowedTensor → FFT plan lookup → FFTW execution (allow_threads)
          │
          │ allocate OutputTensor{32,64} (Vec<complex>, custom deleter)
          ▼
New DLPack capsule (PyCapsule "dltensor")
          │
          │ torch.utils.dlpack.from_dlpack(capsule)
          ▼
PyTorch Tensor (torch.complex64/complex128) – zero copy of result buffer
```

Input tensors remain untouched—the bridge detaches and ensures contiguity before
exporting to DLPack. The returned tensor owns a Rust-managed allocation whose
lifetime is tied to the DLPack capsule via a custom deleter.

## Rust Module

`spectral_core/src/lib.rs` implements three public functions:

- `fft2_dlpack(capsule)` – single forward FFT
- `ifft2_dlpack(capsule)` – inverse FFT with `norm="backward"`
- `fft2_batch_dlpack([capsules])` – batched forward FFT to amortise FFI calls

Key implementation details:

- **DLPack ingestion.** A `BorrowedTensor` wrapper consumes the capsule,
  validates CPU/contiguity constraints, and ensures the original deleter is
  invoked once processing is complete.
- **Output capsules.** `OutputTensor32/64` structures allocate
  `Vec<[f32; 2]>`/`Vec<[f64; 2]>`, populate DLPack metadata and provide a custom
  deleter that frees both the buffer and shape/stride vectors exactly once.
- **Plan caching.** Plans are keyed by `(height, width, direction)` and cached
  in `Lazy<Mutex<HashMap<…>>>`. Plans are `Arc`-wrapped to allow reuse across
  threads. FFTW plans are initialised with `FFTW_MEASURE` and use the global
  thread count.
- **Threading.** `fftw*_init_threads()` is executed once via `OnceLock`. The
  thread count is derived from the `SPECTRAL_BRIDGE_THREADS` environment variable
  (defaulting to `num_cpus::get()`). Module initialisation also calls
  `fftw_plan_with_nthreads` to ensure consistent threading for eager invocations.
- **Precision.** The dtype logic dispatches to complex32/complex64 FFTW plans.
  Real inputs are promoted to complex internally. The inverse transform applies
  the `1/(H·W)` normalisation to match `torch.fft.ifft2` semantics.
- **GIL release.** FFT execution occurs inside `py.allow_threads` to prevent the
  Python interpreter from stalling while FFTW runs.

## Python Bridge Layer

`src/spectral/bridge.py` exposes the high-level `SpectralBridge` class:

- `fft2` / `ifft2` – autograd-aware wrappers using custom `torch.autograd.Function`
  subclasses to preserve gradient propagation while delegating heavy lifting to
  Rust.
- `fft2_batch` – converts an iterable of tensors into DLPack capsules and issues
  a single batched FFI call, dramatically reducing interpreter overhead.
- `fft_filter2` – convenience helper that performs `FFT → mul → iFFT` entirely
  through the bridge.
- `profile_fft2` – returns the FFT result alongside a `CallProfile` dataclass
  containing `conversion`, `ffi`, and `total` timings for benchmarking.
- `diagnostics` – reports backend selection, threading and last call statistics.

Inputs on non-CPU devices automatically fall back to `torch.fft` to maintain
correctness.

## Benchmarking

`scripts/bench_bridge.py` offers a repeatable micro-benchmark harness:

1. Warm-up (10 iterations) to populate FFTW plan caches.
2. Median-of-200 measurement for three implementations: `numpy`, `torch.fft`,
   and the bridge.
3. Collection of FFI/conversion breakdowns via `profile_fft2`.
4. JSON output (`results/bridge_benchmark.json`) capturing diagnostics and per
   shape statistics for downstream analysis.

## Expected Performance

- FFI overhead should remain below ~5% of total runtime for large tiles thanks
  to batching and zero-copy capsules.
- FFTW compute time typically lands within 1.2× of `torch.fft` for identical
  shapes when plan caches are warm.
- Batched invocations reduce Python overhead by ≥3× versus repeated single
  calls, particularly for small tiles.
- Plan caching and thread initialisation eliminate redundant FFTW setup work
  across calls.

## Pointer Ownership & Zero Copy

The zero-copy guarantee is enforced by:

- Borrowing input tensors directly from the DLPack capsule without copying.
- Returning an output capsule whose deleter drops the Rust allocation exactly
  once, letting `torch.utils.dlpack.from_dlpack` adopt the pointer verbatim.
- Tests verify pointer equality between the capsule payload and the resulting
  PyTorch tensor storage to ensure no hidden copies occur.
