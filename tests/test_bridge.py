"""Unit tests for the SpectralBridge interface."""

from __future__ import annotations

import ctypes

import pytest
import torch
from torch.utils import dlpack

from src.spectral.bridge import SpectralBridge

try:  # pragma: no cover - optional backend
    import spectral_core
except ImportError:  # pragma: no cover - handled by skip markers
    spectral_core = None  # type: ignore


@pytest.fixture(scope="module")
def bridge() -> SpectralBridge:
    return SpectralBridge()


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = torch.linalg.norm(a - b)
    denom = torch.linalg.norm(b)
    if denom == 0:
        return float(diff)
    return float(diff / denom)


def test_fft2_matches_torch(bridge: SpectralBridge) -> None:
    x = torch.randn(128, 128, dtype=torch.float32)
    target = torch.fft.fft2(x)
    estimate = bridge.fft2(x)
    assert estimate.shape == target.shape
    assert estimate.dtype == target.dtype
    assert _relative_error(estimate, target) < 1e-6


def test_ifft2_matches_torch(bridge: SpectralBridge) -> None:
    freq = torch.randn(64, 64, dtype=torch.complex64)
    target = torch.fft.ifft2(freq)
    estimate = bridge.ifft2(freq)
    assert estimate.shape == target.shape
    assert estimate.dtype == target.dtype
    assert _relative_error(estimate, target) < 1e-6


def test_fft2_batch_equivalence(bridge: SpectralBridge) -> None:
    signals = [torch.randn(64, 64, dtype=torch.float32) for _ in range(4)]
    batched = bridge.fft2_batch(signals)
    reference = [bridge.fft2(s) for s in signals]
    assert len(batched) == len(reference)
    for lhs, rhs in zip(batched, reference):
        assert _relative_error(lhs, rhs) < 1e-6


@pytest.mark.skipif(
    spectral_core is None or not hasattr(spectral_core, "fft2_dlpack"),
    reason="Rust backend not available",
)
def test_dlpack_zero_copy_round_trip() -> None:
    tensor = torch.randn(32, 32, dtype=torch.complex64)
    capsule = dlpack.to_dlpack(tensor.detach().clone())
    result_capsule = spectral_core.fft2_dlpack(capsule)

    class DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class DLDataType(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int),
            ("dtype", DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_longlong)),
            ("strides", ctypes.POINTER(ctypes.c_longlong)),
            ("byte_offset", ctypes.c_ulonglong),
        ]

    class DLManagedTensor(ctypes.Structure):
        _fields_ = [
            ("dl_tensor", DLTensor),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
        ]

    getter = ctypes.pythonapi.PyCapsule_GetPointer
    getter.restype = ctypes.c_void_p
    getter.argtypes = [ctypes.py_object, ctypes.c_char_p]

    managed_ptr = getter(result_capsule, b"dltensor")
    managed = ctypes.cast(managed_ptr, ctypes.POINTER(DLManagedTensor)).contents
    buffer_ptr = managed.dl_tensor.data

    result_tensor = dlpack.from_dlpack(result_capsule)
    assert result_tensor.data_ptr() == buffer_ptr
    assert result_tensor.dtype == torch.complex64

    # Accuracy check versus direct torch.fft
    direct = torch.fft.fft2(tensor)
    assert _relative_error(result_tensor, direct) < 1e-6
