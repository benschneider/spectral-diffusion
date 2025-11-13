"""Helper utilities for shuttling Torch tensors through DLPack."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils import dlpack as torch_dlpack


def ensure_fft_ready(tensor: torch.Tensor) -> torch.Tensor:
    """Validate dtype/layout and return a contiguous complex64 view."""
    if tensor.dtype not in (torch.complex64, torch.float32):
        raise TypeError("RIFFT expects float32 or complex64 tensors")
    if tensor.dim() < 2:
        raise ValueError("Need at least 2 dimensions (height, width)")
    if tensor.dtype == torch.float32:
        tensor = tensor.to(torch.complex64)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    # Clone so the FFT runs on a detached buffer; RIFFT operates in-place.
    return tensor.clone()


def to_dlpack(tensor: torch.Tensor):
    return torch_dlpack.to_dlpack(tensor)


def from_dlpack(capsule) -> torch.Tensor:
    return torch_dlpack.from_dlpack(capsule)


def spatial_dims(tensor: torch.Tensor) -> Tuple[int, int]:
    if tensor.dim() < 2:
        raise ValueError("Tensor needs height/width dims")
    return tensor.shape[-2], tensor.shape[-1]
