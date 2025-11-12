# RIFFT-Core Migration Plan

## Goal
Spin the current SpectralBridge Rust FFT core into a standalone high-performance library named **rifft-core** (aka "RIFFT-Core"). The standalone repo must offer a Rust crate, C ABI with DLPack zero-copy bridge, Python bindings (maturin), benchmarks, and documentation that meet or exceed PyTorch CPU FFT performance for ≥512×512 grids.

---
## Task Breakdown

1. **Repository Scaffold**
   - Create top-level layout:
     ```
     rifft-core/
     ├─ Cargo.toml
     ├─ README.md
     ├─ src/
     │  ├─ lib.rs
     │  ├─ fft2d.rs
     │  ├─ fused.rs
     │  ├─ planner.rs
     │  ├─ workspace.rs
     │  ├─ simd.rs
     │  ├─ dlpack.rs
     │  ├─ api_c.rs
     │  └─ types.rs
     ├─ benches/
     ├─ include/riff_core.h
     ├─ python/
     │  ├─ riff_core/
     │  │   ├─ __init__.py
     │  │   ├─ bridge.py
     │  │   └─ dlpack_utils.py
     │  └─ tests/
     ├─ pyproject.toml
     ```
   - Initialize `Cargo.toml` and `pyproject.toml` (maturin) with correct metadata.
   - Stub README with overview + roadmap placeholders.

2. **FFT Engine Port**
   - Copy/adapt existing SpectralBridge FFT modules into new structure:
     - `fft2d.rs`: main in-place FFT pipeline (row FFT → transpose → column FFT or TLS columns).
     - `planner.rs`: global plan cache with small-shape prewarm + env tuning (`RUSTFFT_THREADS`, `RUSTFFT_SMALL_*`).
     - `workspace.rs`: aligned buffers, thread-local scratch, buffer pools.
     - `simd.rs`: std::simd + optional `simd_avx2` feature.
     - `fused.rs`: fused forward/filter/inverse path with reused work buffers.
   - Ensure all operations avoid per-call allocation (aligned work, TLS, preallocated transpose scratch).
   - Integrate Rayon for parallel rows/columns and fuse instrumentation for per-phase timing.

3. **C ABI & DLPack Bridge**
   - `api_c.rs`: expose `riff_handle_t` create/free, `riff_fft2d_forward`, `riff_fft2d_inverse`, `riff_fft2d_fused_filter`, `riff_get_version`, `riff_get_backend_name`.
   - `dlpack.rs`: safe zero-copy tensor import/export with validation (contiguous, float32, 2-D, alignment).
   - `include/riff_core.h`: C declarations matching `api_c.rs`.

4. **Python Bindings (maturin)**
   - `python/riff_core/bridge.py`: DLPack bridge utilities calling into the C ABI via ctypes or PyO3 (prefer maturin extension for zero-copy).
   - Provide `fft2`, `ifft2`, `fft_filter_ifft`, batched variants, benchmarking CLI (`python -m riff_core.bench --sizes ...`).
   - `python/tests`: correctness vs torch.fft for random tensors, batch tests, fused op tests.

5. **Benches & Tests**
   - Criterion benches for FFT256/512/1024 + fused op (collect median runtime, bandwidth, thread scaling, simd vs scalar).
   - Rust unit tests for planner reuse, workspace alignment, DLPack conversions.
   - Python pytest suite (already defined above).

6. **Documentation**
   - README to include: overview, architecture diagram, performance table (target: 256≤0.35ms, 512≤1.0ms, 1024≤3.0ms), Rust/Python examples, zero-copy explanation, roadmap (Metal/CUDA backends).
   - Mention env controls (`RUSTFFT_THREADS`, `RUSTFFT_TRANSPOSE_MIN`, `RUSTFFT_SMALL_*`).

7. **Verification**
   - Ensure `cargo build --release` (and `+nightly --features simd_avx2`) succeeds.
   - Ensure `maturin develop` + Python tests pass.
   - Provide sample benchmark output meeting target numbers.

---
## Notes
- Start by cloning current SpectralBridge FFT engine into the new layout; refactor as needed but keep existing optimizations (Rayon columns, small-plan cache, SIMD helpers).
- Consider splitting workspace management so the C ABI can create/destroy handles with fixed max sizes per use-case.
- Evaluate providing both transpose and TLS column paths, controlled via env, to cover all grid sizes efficiently.
- Keep CI-friendly defaults: SIMD feature optional (nightly), fallback scalar path for stable.

