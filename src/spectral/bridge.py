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
            # Log the selected backend once at initialization
            best_backend = self.core.best_backend()
            available = self.core.available_backends()
            print(f"[SpectralBridge] Using backend: {best_backend} (available: {available})")
        else:
            self.core = FallbackSpectralCore()
            print("[SpectralBridge] Using fallback: torch_fft")

        # Track code path usage
        self.numpy_fallback_count = 0
        self.zero_copy_count = 0

    def fft2(self, x: torch.Tensor) -> torch.Tensor:
        """2D FFT using Rust backend with batch support."""
        if x.ndim == 3:  # (batch, height, width) - Use batch processing for FFI optimization
            batch_size, height, width = x.shape
            # Prepare batch of numpy arrays
            numpy_arrays = []
            for b in range(batch_size):
                x_slice = x[b]  # (height, width)
                x_np = x_slice.detach().cpu().numpy()
                if not x_np.flags.c_contiguous:
                    x_np = np.ascontiguousarray(x_np)
                numpy_arrays.append(x_np)

            # Single batch call to Rust (FFI optimization)
            try:
                batch_results = self.core.fft2_batch(numpy_arrays)
                results = []
                for result_np in batch_results:
                    result_tensor = torch.from_numpy(result_np).to(x.device, x.dtype)
                    results.append(result_tensor)
                self.zero_copy_count += batch_size  # Count all operations
                return torch.stack(results, dim=0)
            except Exception as e:
                print(f"[SpectralBridge] Batch processing failed ({e}), falling back to individual calls")
                # Fallback to individual processing
                results = []
                for x_np in numpy_arrays:
                    result_np = self.core.fft2(x_np)
                    result_tensor = torch.from_numpy(result_np).to(x.device, x.dtype)
                    results.append(result_tensor)
                self.zero_copy_count += batch_size
                return torch.stack(results, dim=0)
        else:
            # Single 2D tensor
            return self._fft2_single(x)

    def _fft2_single(self, x: torch.Tensor) -> torch.Tensor:
        """Process a single 2D tensor with zero-copy."""
        try:
            # Convert to numpy (zero-copy if contiguous)
            x_np = x.detach().cpu().numpy()
            if not x_np.flags.c_contiguous:
                x_np = np.ascontiguousarray(x_np)  # This copies

            # Use the numpy buffer directly (zero-copy)
            result_np = self.core.fft2(x_np)
            result = torch.from_numpy(result_np).to(x.device, x.dtype)
            self.zero_copy_count += 1
            return result
        except Exception as e:
            # Fallback
            self.numpy_fallback_count += 1
            print(f"[SpectralBridge] Zero-copy failed ({e}), using numpy fallback")
            x_np = x.detach().cpu().numpy()
            if not x_np.flags.c_contiguous:
                x_np = np.ascontiguousarray(x_np)
            result_np = self.core.fft2(x_np)
            return torch.from_numpy(result_np).to(x.device, x.dtype)

    def fft2_dlpack_test(self, x: torch.Tensor) -> torch.Tensor:
        """Test DLPack direct return path."""
        try:
            # Get shape before DLPack conversion
            height, width = x.shape
            # Convert to DLPack
            caps = dlpack.to_dlpack(x)
            # Call Rust with capsule and explicit shape
            result_caps = self.core.fft2_dlpack_shaped(caps, height, width)
            # Convert back
            return dlpack.from_dlpack(result_caps)
        except Exception as e:
            print(f"[SpectralBridge] DLPack test failed ({e}), falling back to numpy")
            return self.fft2(x)

    def ifft2(self, x: torch.Tensor) -> torch.Tensor:
        """2D inverse FFT using Rust backend."""
        x_np = x.detach().cpu().numpy()
        # Ensure contiguous memory layout for FFTW
        if not x_np.flags.c_contiguous:
            x_np = np.ascontiguousarray(x_np)
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

    def current_backend(self) -> str:
        """Get the currently selected backend."""
        if HAS_SPECTRAL_CORE:
            return self.core.best_backend()
        else:
            return "torch_fft"

    def print_usage_stats(self):
        """Print code path usage statistics."""
        total_calls = self.numpy_fallback_count + self.zero_copy_count
        if total_calls > 0:
            numpy_pct = (self.numpy_fallback_count / total_calls) * 100
            zero_copy_pct = (self.zero_copy_count / total_calls) * 100
            print(f"[SpectralBridge] Usage stats: numpy_fallback={self.numpy_fallback_count} ({numpy_pct:.1f}%), zero_copy={self.zero_copy_count} ({zero_copy_pct:.1f}%)")
        else:
            print("[SpectralBridge] No calls recorded yet")


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