# RIFFT Core

RIFFT Core is a self-contained FFT runtime that powers the SpectralBridge pipeline. It combines
RustFFT plans, Rayon parallelism, and optional `std::simd` acceleration to deliver deterministic
2-D FFTs, fused forward/filter/inverse passes, and zero-copy bridges into C and Python runtimes.

```
┌──────────────┐     ┌───────────────┐     ┌───────────────┐
│ DLPack inputs│ ──▶ │ Planner cache │ ──▶ │ Row FFT stage │
└──────────────┘     └───────────────┘     └───────────────┘
        │                                    │
        ▼                                    ▼
┌──────────────┐     ┌───────────────┐     ┌───────────────┐
│ TLS scratch  │ ◀── │ Workspace pool│ ◀── │ Column FFT/T  │
└──────────────┘     └───────────────┘     └───────────────┘
        │                                    │
        ▼                                    ▼
┌──────────────┐     ┌───────────────┐     ┌───────────────┐
│ SIMD fused   │ ◀── │ C / Python FFI│ ◀── │ Torch adapters │
└──────────────┘     └───────────────┘     └───────────────┘
```

## Performance targets

| Size | Target (ms) | Baseline bench (ms) |
| ---- | ----------- | ------------------- |
| 256×256 | ≤ 0.35 | 0.32 on M3 Max |
| 512×512 | ≤ 1.00 | 0.94 on M3 Max |
| 1024×1024 | ≤ 3.00 | 2.8 on M3 Max |

Criterion benches live under `benches/` and report timing, bandwidth, and thread scaling.

## Rust API

```rust
use num_complex::Complex32;
use rifft_core::RifftHandle;

let mut handle = RifftHandle::new();
let mut plane = vec![Complex32::default(); 512 * 512];
handle.fft2d_forward(&mut plane, 512, 512).unwrap();
```

The `RifftHandle` caches plans (height, width, direction, dtype, SIMD flag) and reuses aligned
workspace allocations for every call. Set `RUSTFFT_THREADS` to pin Rayon parallelism and
`RUSTFFT_SMALL_MAX/RUSTFFT_SMALL_CACHE` to tune the small-plan FIFO.

## C ABI

The shared library exports `riff_create_handle`, `riff_fft2d_forward`, `riff_fft2d_inverse`,
`riff_fft2d_fused_filter`, `riff_get_version`, and `riff_get_backend_name`. See
`include/riff_core.h` for the full surface area.

## Python bindings

- Built with Maturin + PyO3 (`python` feature).
- `riff_core.bridge` exposes `fft2`, `ifft2`, `fft_filter_ifft`, plus batched variants.
- CLI: `python -m riff_core.bench --sizes 256 512 1024 --iters 25 --device cpu`.

```python
import torch
from riff_core import fft2

image = torch.randn(256, 256, dtype=torch.complex64)
freq = fft2(image)
```

### Zero-copy DLPack

`riff_core.bridge` converts `torch.Tensor` objects into DLPack capsules via `torch.utils.dlpack`.
The PyO3 shim unwraps the capsule, validates contiguity/alignment, and hands the raw pointer to the
Rust planner without copying. Ownership is transferred back to PyTorch after the transform so the
returned tensor shares memory with the accelerated path.

## Build & test matrix

```
cargo build --release
cargo +nightly build --release --features simd_avx2
maturin develop --features python
pytest python/tests -q
```

Benchmarks:

```
cargo bench --bench bench_fft256
cargo bench --bench bench_fft512
cargo bench --bench bench_fft1024
cargo bench --bench bench_fused
python scripts/bench_rifft_compare.py
```

## Roadmap

- [ ] CUDA + Metal back-ends via opaque `TensorHandle`s.
- [ ] Mixed-precision kernels (bf16/half) with on-the-fly promotion.
- [ ] Autotuned fuse graph builder for multi-filter workloads.
