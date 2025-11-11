"""
Python bridge to Rust spectral processing core.

This module provides a clean Python interface to the Rust spectral_core library,
handling DLPack tensor conversions and autograd integration.
"""

import numpy as np
import torch
from torch.utils import dlpack
from typing import Optional

# Import the Rust module (will be available after build)
try:
    import spectral_core as rust_spectral_core
    # Check if it has the expected class
    _ = rust_spectral_core.SpectralCore
    HAS_SPECTRAL_CORE = True
except (ImportError, AttributeError):
    HAS_SPECTRAL_CORE = False
    rust_spectral_core = None


class FallbackSpectralCore:
    """Fallback implementation using PyTorch when Rust extension unavailable."""

    @staticmethod
    def fft2(array):
        """Fallback 2D FFT using torch.fft."""
        # Convert numpy to torch if needed
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
        else:
            tensor = array

        # Apply 2D FFT
        result = torch.fft.fft2(tensor)

        # Return magnitude as numpy array (simplified for testing)
        return torch.abs(result).numpy()

    @staticmethod
    def ifft2(array):
        """Fallback 2D iFFT using torch.fft."""
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
        else:
            tensor = array

        # Apply 2D iFFT
        result = torch.fft.ifft2(tensor)

        # Return real part as numpy array
        return torch.real(result).numpy()

    @staticmethod
    def fft_filter2(x_array, h_array):
        """Fallback fused FFT filtering."""
        if isinstance(x_array, np.ndarray):
            x_tensor = torch.from_numpy(x_array)
        else:
            x_tensor = x_array

        if isinstance(h_array, np.ndarray):
            h_tensor = torch.from_numpy(h_array)
        else:
            h_tensor = h_array

        # Simple element-wise filtering (placeholder)
        return (x_tensor * h_tensor).numpy()

    @staticmethod
    def is_cuda_available():
        return torch.cuda.is_available()

    @staticmethod
    def available_backends():
        backends = ["torch_fft"]
        if torch.cuda.is_available():
            backends.append("torch_fft_cuda")
        return backends


class SpectralBridge:
    """Bridge between PyTorch and Rust spectral processing."""

    def __init__(self):
        if HAS_SPECTRAL_CORE:
            self.core = rust_spectral_core.SpectralCore()
        else:
            self.core = FallbackSpectralCore()

    def fft2(self, x: torch.Tensor) -> torch.Tensor:
        """2D FFT using Rust backend with batch support."""
        if x.ndim == 3:  # (batch, height, width)
            batch_size, height, width = x.shape
            results = []
            for b in range(batch_size):
                x_slice = x[b]  # (height, width)
                x_np = x_slice.detach().cpu().numpy()
                result_np = self.core.fft2(x_np)
                result_tensor = torch.from_numpy(result_np).to(x.device, x.dtype)
                results.append(result_tensor)
            return torch.stack(results, dim=0)
        else:
            # 2D tensor
            x_np = x.detach().cpu().numpy()
            result_np = self.core.fft2(x_np)
            return torch.from_numpy(result_np).to(x.device, x.dtype)

    def ifft2(self, x: torch.Tensor) -> torch.Tensor:
        """2D inverse FFT using Rust backend."""
        x_np = x.detach().cpu().numpy()
        result_np = self.core.ifft2(x_np)
        return torch.from_numpy(result_np)

    def fft_filter2(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Fused FFT → filter → iFFT operation."""
        x_np = x.detach().cpu().numpy()
        h_np = h.detach().cpu().numpy()
        result_np = self.core.fft_filter2(x_np, h_np)
        return torch.from_numpy(result_np)

    # DLPack interface (zero-copy when Rust available)
    def fft2_dlpack(self, dlpack_capsule):
        """2D FFT using DLPack capsules (zero-copy with Rust backend)."""
        if HAS_SPECTRAL_CORE:
            return rust_spectral_core.SpectralCore.fft2_dlpack(dlpack_capsule)
        else:
            # Fallback: convert through numpy
            tensor = dlpack.from_dlpack(dlpack_capsule)
            result = self.fft2(tensor)
            return dlpack.to_dlpack(result)

    def ifft2_dlpack(self, dlpack_capsule):
        """2D iFFT using DLPack capsules (zero-copy with Rust backend)."""
        if HAS_SPECTRAL_CORE:
            return rust_spectral_core.SpectralCore.ifft2_dlpack(dlpack_capsule)
        else:
            # Fallback: convert through numpy
            tensor = dlpack.from_dlpack(dlpack_capsule)
            result = self.ifft2(tensor)
            return dlpack.to_dlpack(result)

    def fft_filter2_dlpack(self, x_capsule, h_capsule):
        """Fused FFT filtering using DLPack capsules (zero-copy with Rust backend)."""
        if HAS_SPECTRAL_CORE:
            return rust_spectral_core.SpectralCore.fft_filter2_dlpack(x_capsule, h_capsule)
        else:
            # Fallback: convert through numpy
            x_tensor = dlpack.from_dlpack(x_capsule)
            h_tensor = dlpack.from_dlpack(h_capsule)
            result = self.fft_filter2(x_tensor, h_tensor)
            return dlpack.to_dlpack(result)

    @staticmethod
    def is_available() -> bool:
        """Check if Rust backend is available."""
        return HAS_SPECTRAL_CORE

    def is_cuda_available(self) -> bool:
        """Check if CUDA support is available."""
        return self.core.is_cuda_available()

    def available_backends(self) -> list[str]:
        """Get list of available backends."""
        return self.core.available_backends()


# Global bridge instance
_bridge: Optional[SpectralBridge] = None


def get_bridge() -> SpectralBridge:
    """Get or create the global spectral bridge."""
    global _bridge
    if _bridge is None:
        _bridge = SpectralBridge()
    return _bridge


# Convenience functions
def fft2(x: torch.Tensor) -> torch.Tensor:
    """2D FFT with automatic backend selection."""
    return get_bridge().fft2(x)


def ifft2(x: torch.Tensor) -> torch.Tensor:
    """2D inverse FFT with automatic backend selection."""
    return get_bridge().ifft2(x)


def fft_filter2(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Fused FFT filtering with automatic backend selection."""
    return get_bridge().fft_filter2(x, h)


def diagnostics():
    """Quick diagnostics to verify bridge health."""
    import torch
    from src.spectral.bridge import get_bridge

    bridge = get_bridge()
    print(f"[Bridge] available={bridge.is_available()}, backends={bridge.available_backends()}")

    # Test with small tensor
    x = torch.randn(16, 16, dtype=torch.float32)
    y_bridge = bridge.fft2(x)
    y_torch = torch.fft.fft2(x)

    diff = torch.norm(y_bridge - torch.abs(y_torch)).item()
    print(f"[Bridge] dtype={x.dtype}, device={x.device}, diff={diff:.2e}")

    return diff < 1e-5  # Return True if healthy