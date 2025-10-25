"""Tests for synthetic data generation."""
import math
import pytest
import torch
from src.data.synthetic import _parametric_texture_sample, generate_synthetic_samples


class TestParametricTextureSample:
    """Test parametric texture generation."""

    def test_basic_texture_generation(self):
        """Test basic texture sample generation."""
        params = {
            'frequency_range': [1, 10],
            'bandwidth': 0.25,
            'phase_jitter': math.pi,
            'amplitude': 1.0
        }
        result = _parametric_texture_sample(8, 8, params)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 8, 8)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_texture_with_different_sizes(self):
        """Test texture generation with different image sizes."""
        params = {
            'frequency_range': [1, 5],
            'bandwidth': 0.5,
            'phase_jitter': math.pi/2,
            'amplitude': 2.0
        }

        for height, width in [(4, 4), (16, 16), (32, 8)]:
            result = _parametric_texture_sample(height, width, params)
            assert result.shape == (1, height, width)

    def test_texture_value_range(self):
        """Test that texture values are in reasonable range."""
        params = {
            'frequency_range': [0.5, 2.0],
            'bandwidth': 0.1,
            'phase_jitter': 0.1,
            'amplitude': 0.5
        }
        result = _parametric_texture_sample(16, 16, params)

        # Should be reasonably bounded
        assert result.min() >= -2.0
        assert result.max() <= 2.0
        assert result.std() > 0  # Should have some variation


class TestGenerateSyntheticSamples:
    """Test synthetic sample generation."""

    def test_piecewise_generation(self):
        """Test piecewise constant pattern generation."""
        count, channels, height, width = 2, 3, 8, 8
        data_cfg = {'family': 'piecewise'}

        result = generate_synthetic_samples(count, channels, height, width, data_cfg)

        assert result.shape == (count, channels, height, width)
        assert torch.isfinite(result).all()
        assert (result >= -1.0).all() and (result <= 1.0).all()

    def test_texture_generation(self):
        """Test parametric texture generation."""
        count, channels, height, width = 2, 3, 8, 8
        data_cfg = {'family': 'texture'}

        result = generate_synthetic_samples(count, channels, height, width, data_cfg)

        assert result.shape == (count, channels, height, width)
        assert torch.isfinite(result).all()

    def test_random_field_generation(self):
        """Test random field generation."""
        count, channels, height, width = 2, 3, 8, 8
        data_cfg = {'family': 'random_field'}

        result = generate_synthetic_samples(count, channels, height, width, data_cfg)

        assert result.shape == (count, channels, height, width)
        assert torch.isfinite(result).all()

    def test_noise_generation(self):
        """Test Gaussian noise generation."""
        count, channels, height, width = 2, 3, 8, 8
        data_cfg = {'family': 'noise'}

        result = generate_synthetic_samples(count, channels, height, width, data_cfg)

        assert result.shape == (count, channels, height, width)
        assert torch.isfinite(result).all()
        # Noise should have some variation
        assert result.std() > 0.1

    def test_unknown_family_fallback(self):
        """Test fallback to noise for unknown family."""
        count, channels, height, width = 2, 3, 8, 8
        data_cfg = {'family': 'unknown_family'}

        result = generate_synthetic_samples(count, channels, height, width, data_cfg)

        assert result.shape == (count, channels, height, width)
        assert torch.isfinite(result).all()

    def test_grating_alias(self):
        """Test that 'grating' is treated as 'texture'."""
        count, channels, height, width = 1, 3, 8, 8
        data_cfg = {'family': 'grating'}

        result = generate_synthetic_samples(count, channels, height, width, data_cfg)

        assert result.shape == (count, channels, height, width)
        assert torch.isfinite(result).all()

    def test_powerlaw_alias(self):
        """Test that 'powerlaw' is treated as 'random_field'."""
        count, channels, height, width = 1, 3, 8, 8
        data_cfg = {'family': 'powerlaw'}

        result = generate_synthetic_samples(count, channels, height, width, data_cfg)

        assert result.shape == (count, channels, height, width)
        assert torch.isfinite(result).all()