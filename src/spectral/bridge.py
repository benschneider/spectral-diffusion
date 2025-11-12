"""Rust ↔ PyTorch bridge built on top of DLPack."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils import dlpack

try:  # pragma: no cover - exercised in integration tests
    import ctypes
    import os
    import platform

    # Load the shared library
    lib_name = {
        'Darwin': 'libspectral_core.dylib',
        'Linux': 'libspectral_core.so',
        'Windows': 'spectral_core.dll'
    }.get(platform.system(), 'libspectral_core.so')

    lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'spectral_core', 'target', 'release', lib_name)
    if os.path.exists(lib_path):
        _lib = ctypes.CDLL(lib_path)

        # Define function signatures
        _lib.spectral_init.argtypes = []
        _lib.spectral_init.restype = ctypes.c_int

        _lib.spectral_cleanup.argtypes = []
        _lib.spectral_cleanup.restype = None

        _lib.spectral_version.argtypes = []
        _lib.spectral_version.restype = ctypes.c_char_p

        _lib.spectral_backends.argtypes = []
        _lib.spectral_backends.restype = ctypes.c_char_p

        _lib.spectral_test.argtypes = []
        _lib.spectral_test.restype = ctypes.c_int

        # Initialize the library
        if _lib.spectral_init() == 0:
            _HAS_RUST_BACKEND = True
        else:
            _HAS_RUST_BACKEND = False
    else:
        _lib = None
        _HAS_RUST_BACKEND = False

except (ImportError, OSError):  # pragma: no cover - fallback path
    _lib = None
    _HAS_RUST_BACKEND = False


@dataclass
class CallProfile:
    """Timing information captured for a single bridge invocation."""

    backend: str
    total_s: float
    conversion_in_s: float
    ffi_s: float
    conversion_out_s: float
    thread_count: int
    batch: int = 1
    contiguous_copies: int = 0

    def as_dict(self) -> Dict[str, float | int | str]:
        return {
            "backend": self.backend,
            "total_s": self.total_s,
            "conversion_in_s": self.conversion_in_s,
            "ffi_s": self.ffi_s,
            "conversion_out_s": self.conversion_out_s,
            "thread_count": self.thread_count,
            "batch": self.batch,
            "contiguous_copies": self.contiguous_copies,
        }


def _ensure_complex_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float32, torch.complex64):
        return torch.complex64
    if x.dtype in (torch.float64, torch.complex128):
        return torch.complex128
    raise TypeError(f"Unsupported dtype for spectral bridge: {x.dtype}")


class SpectralBridge:
    """High-level interface that wraps the Rust FFT implementation via DLPack."""

    def __init__(self) -> None:
        self._has_rust = _HAS_RUST_BACKEND
        self.backend = "rust-capi" if self._has_rust else "torch.fft"
        self.dlpack_enabled = self._has_rust
        self.thread_count = 8  # Default thread count
        self._last_profile: CallProfile | None = None

        # Test the Rust backend
        if self._has_rust and _lib:
            test_result = _lib.spectral_test()
            if test_result != 42:
                print(f"Warning: Rust backend test failed (expected 42, got {test_result})")
                self._has_rust = False
                self.backend = "torch.fft"

    # ---------------------------------------------------------------------
    # Public API ----------------------------------------------------------
    # ---------------------------------------------------------------------
    def fft2(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a 2D FFT.

        If a batch dimension is provided (shape ``B × H × W``) the computation
        is dispatched through :meth:`fft2_batch` to amortise FFI overhead.
        """

        if x.ndim == 3:
            results = self.fft2_batch(list(x))
            return torch.stack(results, dim=0)
        return _RustFFT.apply(self, x)

    def ifft2(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a 2D inverse FFT with ``norm="backward"`` semantics."""

        if x.ndim == 3:
            results = [self.ifft2(s) for s in x]
            return torch.stack(results, dim=0)
        result = _RustIFFT.apply(self, x)
        return result

    def fft_filter2(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Apply a frequency-domain filter ``h`` to input ``x``."""

        spectrum = self.fft2(x)
        kernel = self.fft2(h)
        filtered = spectrum * kernel
        response = self.ifft2(filtered)
        if not (x.is_complex() or h.is_complex()):
            return response.real
        return response

    def fft2_batch(self, xs: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        """Run several FFTs via a single FFI call to minimise Python overhead."""

        tensor_list = list(xs)
        if not tensor_list:
            return []

        if not self._has_rust or any(t.device.type != "cpu" for t in tensor_list):
            return [torch.fft.fft2(t) for t in tensor_list]

        staged: List[torch.Tensor] = []
        capsules: List[object] = []
        conversion_in = 0.0
        conversion_out = 0.0
        copies = 0
        start_total = time.perf_counter()

        for tensor in tensor_list:
            candidate = tensor.detach()
            if not candidate.is_contiguous():
                candidate = candidate.contiguous()
                copies += 1
            staged.append(candidate)
            t0 = time.perf_counter()
            capsules.append(dlpack.to_dlpack(candidate))
            conversion_in += time.perf_counter() - t0

        ffi_start = time.perf_counter()
        result_capsules = fft2_batch_dlpack(capsules)  # type: ignore[arg-type]
        ffi_s = time.perf_counter() - ffi_start

        results: List[torch.Tensor] = []
        for capsule, source in zip(result_capsules, tensor_list):
            t0 = time.perf_counter()
            out = dlpack.from_dlpack(capsule)
            conversion_out += time.perf_counter() - t0
            out = out.to(dtype=_ensure_complex_dtype(source), device=source.device)
            if source.requires_grad:
                out.requires_grad_(True)
            results.append(out)

        total = time.perf_counter() - start_total
        self._last_profile = CallProfile(
            backend=self.backend,
            total_s=total,
            conversion_in_s=conversion_in,
            ffi_s=ffi_s,
            conversion_out_s=conversion_out,
            thread_count=self.thread_count,
            batch=len(results),
            contiguous_copies=copies,
        )
        return results

    def profile_fft2(self, x: torch.Tensor) -> Tuple[torch.Tensor, CallProfile]:
        """Run :meth:`fft2` without autograd and return timing information."""

        result, profile = self._fft2_forward(x)
        self._last_profile = profile
        return result, profile

    def diagnostics(self) -> Dict[str, object]:
        """Return runtime information about the active backend."""

        payload: Dict[str, object] = {
            "backend": self.backend,
            "dlpack_enabled": self.dlpack_enabled,
            "thread_count": self.thread_count,
        }
        if self._last_profile is not None:
            payload["last_profile"] = self._last_profile.as_dict()
        return payload

    # ------------------------------------------------------------------
    # Internal helpers -------------------------------------------------
    # ------------------------------------------------------------------
    def _fft2_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, CallProfile]:
        if not self._has_rust or x.device.type != "cpu":
            start = time.perf_counter()
            result = torch.fft.fft2(x)
            end = time.perf_counter()
            profile = CallProfile(
                backend="torch.fft",
                total_s=end - start,
                conversion_in_s=0.0,
                ffi_s=end - start,
                conversion_out_s=0.0,
                thread_count=self.thread_count,
            )
            return result, profile

        tensor = x.detach()
        copies = 0
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
            copies = 1

        start_total = time.perf_counter()
        t0 = time.perf_counter()
        capsule = dlpack.to_dlpack(tensor)
        conversion_in = time.perf_counter() - t0

        ffi_start = time.perf_counter()
        out_capsule = fft2_dlpack(capsule)  # type: ignore[operator]
        ffi_time = time.perf_counter() - ffi_start

        t1 = time.perf_counter()
        result = dlpack.from_dlpack(out_capsule)
        conversion_out = time.perf_counter() - t1

        total = time.perf_counter() - start_total
        result = result.to(dtype=_ensure_complex_dtype(x), device=x.device)
        if x.requires_grad:
            result.requires_grad_(True)

        profile = CallProfile(
            backend=self.backend,
            total_s=total,
            conversion_in_s=conversion_in,
            ffi_s=ffi_time,
            conversion_out_s=conversion_out,
            thread_count=self.thread_count,
            contiguous_copies=copies,
        )
        return result, profile

    def _ifft2_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, CallProfile]:
        if not self._has_rust or x.device.type != "cpu":
            start = time.perf_counter()
            result = torch.fft.ifft2(x)
            end = time.perf_counter()
            profile = CallProfile(
                backend="torch.fft",
                total_s=end - start,
                conversion_in_s=0.0,
                ffi_s=end - start,
                conversion_out_s=0.0,
                thread_count=self.thread_count,
            )
            return result, profile

        tensor = x.detach()
        copies = 0
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
            copies = 1

        start_total = time.perf_counter()
        t0 = time.perf_counter()
        capsule = dlpack.to_dlpack(tensor)
        conversion_in = time.perf_counter() - t0

        ffi_start = time.perf_counter()
        out_capsule = _ifft2_dlpack(capsule)  # type: ignore[operator]
        ffi_time = time.perf_counter() - ffi_start

        t1 = time.perf_counter()
        result = dlpack.from_dlpack(out_capsule)
        conversion_out = time.perf_counter() - t1

        total = time.perf_counter() - start_total
        result = result.to(dtype=_ensure_complex_dtype(x), device=x.device)
        if x.requires_grad:
            result.requires_grad_(True)

        profile = CallProfile(
            backend=self.backend,
            total_s=total,
            conversion_in_s=conversion_in,
            ffi_s=ffi_time,
            conversion_out_s=conversion_out,
            thread_count=self.thread_count,
            contiguous_copies=copies,
        )
        return result, profile

    @property
    def last_profile(self) -> Optional[CallProfile]:
        return self._last_profile


class _RustFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bridge: SpectralBridge, input_tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        result, profile = bridge._fft2_forward(input_tensor)
        bridge._last_profile = profile
        ctx.save_for_backward()
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:  # type: ignore[override]
        grad = torch.fft.ifft2(grad_output)
        return None, grad


class _RustIFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bridge: SpectralBridge, input_tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        result, profile = bridge._ifft2_forward(input_tensor)
        bridge._last_profile = profile
        ctx.save_for_backward()
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:  # type: ignore[override]
        grad = torch.fft.fft2(grad_output)
        return None, grad


_bridge: Optional[SpectralBridge] = None


def get_bridge() -> SpectralBridge:
    global _bridge
    if _bridge is None:
        _bridge = SpectralBridge()
    return _bridge


__all__ = ["SpectralBridge", "get_bridge", "CallProfile"]
