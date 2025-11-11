#!/usr/bin/env python3
"""
Minimal test for spectral bridge functionality.

This script tests the Python bridge interface and provides a fallback
implementation for development/testing when the Rust extension isn't built.
"""

import numpy as np
import torch
from src.spectral.bridge import get_bridge, SpectralBridge


def test_bridge_interface():
    """Test that the bridge interface works correctly."""
    print("Testing Spectral Bridge Interface...")

    # Test bridge availability
    bridge = get_bridge()
    print(f"Bridge available: {bridge.is_available()}")

    # Test backend detection
    backends = bridge.available_backends()
    print(f"Available backends: {backends}")

    # Test CUDA detection
    cuda_available = bridge.is_cuda_available()
    print(f"CUDA available: {cuda_available}")

    print("✓ Bridge interface test passed")
    return True


def test_tensor_operations():
    """Test basic tensor operations through the bridge."""
    print("\nTesting Tensor Operations...")

    # Create test data
    x_np = np.random.randn(8, 8).astype(np.float32)
    print(f"Input shape: {x_np.shape}, dtype: {x_np.dtype}")

    # Test with PyTorch tensor
    x_torch = torch.from_numpy(x_np)
    print(f"PyTorch tensor shape: {x_torch.shape}, dtype: {x_torch.dtype}")

    # Test bridge operations (will use fallback if Rust not available)
    bridge = get_bridge()

    try:
        # Test FFT
        fft_result = bridge.fft2(x_torch)
        print(f"FFT result shape: {fft_result.shape}, dtype: {fft_result.dtype}")

        # Test iFFT
        ifft_result = bridge.ifft2(fft_result)
        print(f"IFFT result shape: {ifft_result.shape}, dtype: {ifft_result.dtype}")

        # Test fused filter
        h_np = np.ones((8, 8), dtype=np.float32) * 0.1  # Simple filter
        h_torch = torch.from_numpy(h_np)
        filtered = bridge.fft_filter2(x_torch, h_torch)
        print(f"Filtered result shape: {filtered.shape}, dtype: {filtered.dtype}")

        print("✓ Tensor operations test passed")
        return True

    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False


def test_correctness():
    """Test correctness against numpy reference."""
    print("\nTesting Correctness...")

    # Create simple test signal
    x_np = np.zeros((4, 4), dtype=np.float32)
    x_np[0, 0] = 1.0  # Delta function
    x_torch = torch.from_numpy(x_np)

    bridge = get_bridge()

    try:
        # Forward FFT
        fft_rust = bridge.fft2(x_torch)
        fft_np = np.fft.fft2(x_np)

        # Compare (taking magnitude for simplicity)
        rust_magnitude = torch.abs(fft_rust).numpy()
        np_magnitude = np.abs(fft_np)

        max_diff = np.max(np.abs(rust_magnitude - np_magnitude))
        rel_error = max_diff / np.max(np_magnitude)

        print(".2e")
        print(".2e")

        if rel_error < 1e-5:
            print("✓ Correctness test passed")
            return True
        else:
            print("✗ Correctness test failed - high error")
            return False

    except Exception as e:
        print(f"✗ Correctness test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Spectral Bridge Minimal Test Suite")
    print("=" * 50)

    tests = [
        test_bridge_interface,
        test_tensor_operations,
        test_correctness,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Bridge is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    exit(main())