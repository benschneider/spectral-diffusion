#!/usr/bin/env python
"""
Quick test script for SyntheticSpectralDataset.

Generates a few samples and saves them to verify the dataset produces
colorful, diverse synthetic textures with balanced spectral distributions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.data.synthetic_dataset import SyntheticSpectralDataset


def test_synthetic_dataset():
    """Test the SyntheticSpectralDataset with various configurations."""
    
    print("=" * 70)
    print("Testing SyntheticSpectralDataset")
    print("=" * 70)
    
    # Test 1: Default configuration
    print("\n[Test 1] Default configuration (freq_mix=0.5, color_mix=0.2)")
    dataset = SyntheticSpectralDataset(
        size=16,
        image_size=32,
        freq_mix=0.5,
        color_mix=0.2,
        use_text=True,
        include_gratings=True,
        include_shapes=True,
        seed=42,
    )
    
    # Generate samples
    samples = []
    for i in range(8):
        img, _ = dataset[i]
        samples.append(img)
    
    batch = torch.stack(samples, dim=0)
    print(f"  Batch shape: {batch.shape}")
    print(f"  Mean: {batch.mean():.3f} (expected: ~0.5)")
    print(f"  Std: {batch.std():.3f} (expected: 0.15-0.35)")
    print(f"  Min: {batch.min():.3f}, Max: {batch.max():.3f}")
    
    # Save grid
    output_dir = ROOT / "scratch" / "synthetic_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(batch, output_dir / "test1_default.png", nrow=4, padding=2)
    print(f"  ✓ Saved to {output_dir / 'test1_default.png'}")
    
    # Test 2: High frequency emphasis
    print("\n[Test 2] High frequency emphasis (freq_mix=0.9)")
    dataset_hf = SyntheticSpectralDataset(
        size=8,
        image_size=32,
        freq_mix=0.9,
        color_mix=0.2,
        seed=43,
    )
    
    samples_hf = torch.stack([dataset_hf[i][0] for i in range(8)], dim=0)
    save_image(samples_hf, output_dir / "test2_high_freq.png", nrow=4, padding=2)
    print(f"  ✓ Saved to {output_dir / 'test2_high_freq.png'}")
    
    # Test 3: Low frequency emphasis
    print("\n[Test 3] Low frequency emphasis (freq_mix=0.1)")
    dataset_lf = SyntheticSpectralDataset(
        size=8,
        image_size=32,
        freq_mix=0.1,
        color_mix=0.2,
        seed=44,
    )
    
    samples_lf = torch.stack([dataset_lf[i][0] for i in range(8)], dim=0)
    save_image(samples_lf, output_dir / "test3_low_freq.png", nrow=4, padding=2)
    print(f"  ✓ Saved to {output_dir / 'test3_low_freq.png'}")
    
    # Test 4: High color correlation (grayscale-like)
    print("\n[Test 4] High color correlation (color_mix=0.9)")
    dataset_gray = SyntheticSpectralDataset(
        size=8,
        image_size=32,
        freq_mix=0.5,
        color_mix=0.9,
        seed=45,
    )
    
    samples_gray = torch.stack([dataset_gray[i][0] for i in range(8)], dim=0)
    save_image(samples_gray, output_dir / "test4_grayscale.png", nrow=4, padding=2)
    print(f"  ✓ Saved to {output_dir / 'test4_grayscale.png'}")
    
    # Test 5: Independent RGB channels
    print("\n[Test 5] Independent RGB channels (color_mix=0.0)")
    dataset_rgb = SyntheticSpectralDataset(
        size=8,
        image_size=32,
        freq_mix=0.5,
        color_mix=0.0,
        seed=46,
    )
    
    samples_rgb = torch.stack([dataset_rgb[i][0] for i in range(8)], dim=0)
    save_image(samples_rgb, output_dir / "test5_independent_rgb.png", nrow=4, padding=2)
    print(f"  ✓ Saved to {output_dir / 'test5_independent_rgb.png'}")
    
    # Test 6: FFT analysis
    print("\n[Test 6] FFT spectral analysis")
    sample = batch[0]  # Use first sample from default config
    
    # Compute FFT
    fft = torch.fft.fftn(sample, dim=(-2, -1))
    magnitude = torch.fft.fftshift(fft.abs(), dim=(-2, -1))
    
    # Average across channels
    mag_avg = magnitude.mean(dim=0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _to_numpy_for_plot(t: torch.Tensor) -> np.ndarray:
        """Detach and move a tensor to CPU NumPy for matplotlib."""

        return t.detach().cpu().numpy()

    # Spatial domain (convert to NumPy to avoid numpy>=2.0 copy warnings)
    spatial_image = _to_numpy_for_plot(sample.permute(1, 2, 0).clamp(0, 1))
    axes[0].imshow(spatial_image)
    axes[0].set_title("Spatial Domain")
    axes[0].axis("off")
    
    # Frequency domain (log scale)
    spectral_image = _to_numpy_for_plot(torch.log1p(mag_avg))
    im = axes[1].imshow(spectral_image, cmap="viridis")
    axes[1].set_title("FFT Magnitude (log scale)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "test6_fft_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved to {output_dir / 'test6_fft_analysis.png'}")
    
    # Test 7: Channel correlation verification
    print("\n[Test 7] Channel correlation verification")
    for color_mix_val in [0.0, 0.5, 1.0]:
        ds = SyntheticSpectralDataset(
            size=64,
            image_size=32,
            color_mix=color_mix_val,
            seed=47,
        )
        
        # Sample multiple images
        batch_test = torch.stack([ds[i][0] for i in range(64)], dim=0)
        
        # Compute channel correlation
        channels_flat = batch_test.permute(1, 0, 2, 3).reshape(3, -1)
        corr_matrix = torch.corrcoef(channels_flat)
        
        # Get off-diagonal correlations
        mask = ~torch.eye(3, dtype=torch.bool)
        off_diag = corr_matrix[mask]
        mean_corr = off_diag.mean().item()
        
        print(f"  color_mix={color_mix_val:.1f} → mean correlation={mean_corr:.3f}")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print(f"✓ Results saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    test_synthetic_dataset()
