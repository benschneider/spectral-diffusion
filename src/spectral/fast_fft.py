"""Backend selector for FFT calls used in spectral tests.

This module opportunistically dispatches large CPU-only FFTs to RIFFT
whenever that runtime is available. Small transforms, GPU tensors, and
autograd-enabled tensors fall back to ``torch.fft`` to preserve existing
behaviour.
"""

from __future__ import annotations

import os
from typing import Sequence

import torch

try:  # pragma: no cover - optional acceleration path
    from rifft import bridge as _rifft_bridge

    _RIFFT_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration path
    _rifft_bridge = None
    _RIFFT_AVAILABLE = False

_DEFAULT_MIN_DIM = 256


def _parse_min_dim(value: str | None) -> int:
    if value is None:
        return _DEFAULT_MIN_DIM
    try:
        parsed = int(value)
    except ValueError:
        return _DEFAULT_MIN_DIM
    return max(1, parsed)


_RIFFT_MIN_DIM = _parse_min_dim(os.environ.get("SPECTRAL_RIFFT_MIN_DIM"))


def rifft_available() -> bool:
    """Expose whether the optional RIFFT runtime can be used."""

    return _RIFFT_AVAILABLE


def _spatial_area(tensor: torch.Tensor) -> float:
    if tensor.ndim < 2:
        raise ValueError("Need at least two spatial dimensions for FFT.")
    return float(tensor.shape[-2] * tensor.shape[-1])


def _dims_match_last_two(tensor: torch.Tensor, dims: Sequence[int] | None) -> bool:
    if dims is None:
        return False
    dims = tuple(dims)
    if len(dims) != 2:
        return False
    ndim = tensor.ndim
    resolved = tuple(d if d >= 0 else ndim + d for d in dims)
    return resolved == (ndim - 2, ndim - 1)


def _can_use_rifft(tensor: torch.Tensor) -> bool:
    if not _RIFFT_AVAILABLE:
        return False
    if tensor.device.type != "cpu":
        return False
    if tensor.requires_grad:
        return False
    if tensor.dtype not in (torch.float32, torch.complex64):
        return False
    if tensor.ndim < 2:
        return False
    h, w = tensor.shape[-2], tensor.shape[-1]
    if max(h, w) < _RIFFT_MIN_DIM:
        return False
    return True


def _forward_scale(norm: str | None, plane: float) -> float:
    if norm is None or norm == "backward":
        return 1.0
    if norm == "forward":
        return 1.0 / plane
    if norm == "ortho":
        return plane ** -0.5
    raise ValueError(f"Unsupported FFT norm: {norm}")


def _inverse_scale(norm: str | None, plane: float) -> float:
    if norm is None or norm == "backward":
        return 1.0
    if norm == "forward":
        return plane
    if norm == "ortho":
        return plane ** 0.5
    raise ValueError(f"Unsupported FFT norm: {norm}")


def _run_rifft_fft(tensor: torch.Tensor) -> torch.Tensor:
    assert _rifft_bridge is not None
    return _rifft_bridge.fft2(tensor, column_major=False, copy_input=True)


def _run_rifft_ifft(tensor: torch.Tensor) -> torch.Tensor:
    assert _rifft_bridge is not None
    return _rifft_bridge.ifft2(tensor, copy_input=True)


def fft2(tensor: torch.Tensor, *, norm: str | None = None) -> torch.Tensor:
    """2-D FFT that prefers RIFFT for large CPU-only tensors."""

    if _can_use_rifft(tensor):
        plane = _spatial_area(tensor)
        scaled = _run_rifft_fft(tensor)
        factor = _forward_scale(norm, plane)
        if factor != 1.0:
            scaled = scaled * factor
        return scaled
    return torch.fft.fft2(tensor, norm=norm)


def ifft2(tensor: torch.Tensor, *, norm: str | None = None) -> torch.Tensor:
    """Inverse 2-D FFT with the same backend selection as :func:`fft2`."""

    if _can_use_rifft(tensor):
        plane = _spatial_area(tensor)
        scaled = _run_rifft_ifft(tensor)
        factor = _inverse_scale(norm, plane)
        if factor != 1.0:
            scaled = scaled * factor
        return scaled
    return torch.fft.ifft2(tensor, norm=norm)


def fftn(
    tensor: torch.Tensor,
    *,
    dim: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    """FFT substitute that only accelerates the last two dimensions."""

    if _dims_match_last_two(tensor, dim) and _can_use_rifft(tensor):
        return fft2(tensor, norm=norm)
    return torch.fft.fftn(tensor, dim=dim, norm=norm)


def ifftn(
    tensor: torch.Tensor,
    *,
    dim: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    """Inverse FFT substitute that mirrors :func:`fftn`."""

    if _dims_match_last_two(tensor, dim) and _can_use_rifft(tensor):
        return ifft2(tensor, norm=norm)
    return torch.fft.ifftn(tensor, dim=dim, norm=norm)


def prefer_rifft(height: int, width: int, *, requires_grad: bool = False) -> bool:
    """Helper used in tests to inspect backend preference."""

    if not _RIFFT_AVAILABLE or requires_grad:
        return False
    if max(height, width) < _RIFFT_MIN_DIM:
        return False
    return True


def configured_min_dim() -> int:
    """Return the current RIFFT activation threshold."""

    return _RIFFT_MIN_DIM
