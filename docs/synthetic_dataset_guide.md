# SyntheticSpectralDataset Guide

## Overview

The `SyntheticSpectralDataset` is a rich, procedural data generator that provides realistic spectral diversity while staying deterministic and fast. It's designed to complement CIFAR-10 training and offers extensive control over visual and spectral properties.

## Features

### 1. Procedural Generation

The dataset combines multiple visual primitives to create diverse synthetic images:

- **Gaussian/Fractal/1/f Noise**: Spectral noise with controllable frequency characteristics
- **Geometric Shapes**: Rectangles, ellipses, lines, and rings
- **Spectral Gratings**: Sinusoidal patterns at random orientations and frequencies
- **Blob Layers**: Gaussian blobs for organic texture
- **Text Snippets**: Optional Lorem Ipsum text using Pillow (if available)

### 2. Spectral & Color Control

Two key parameters control the spectral and color properties:

- **`freq_mix ∈ [0,1]`**: Blends low-frequency and high-frequency emphasis
  - `0.0` = Low-frequency emphasis (smooth, large-scale features)
  - `0.5` = Balanced frequency distribution
  - `1.0` = High-frequency emphasis (fine details, textures)

- **`color_mix ∈ [0,1]`**: Controls RGB channel correlation
  - `0.0` = Independent RGB channels (colorful, chromatic)
  - `0.5` = Partially correlated channels
  - `1.0` = Fully correlated channels (grayscale-like)

### 3. Compositing

Layers are combined using random blend modes:
- **Additive**: Simple addition of layers
- **Multiplicative**: Modulation effects
- **Overlay**: Tanh-based squashing for smooth blending

### 4. Noise Handling (Critical)

The implementation follows strict zero-mean principles:

- All spatial noise has zero mean: `noise = noise - noise.mean(dim=(1, 2), keepdim=True)`
- DC bias is neutralized in FFT operations: `fft[..., 0, 0] = 0.0`
- No static mean offsets are introduced during generation
- Final normalization to [0,1] only happens at the end

This ensures:
- No DC drift during diffusion training
- Stable gradient flow
- Proper spectral energy distribution

## Configuration

### Basic Configuration

```yaml
data:
  source: synthetic
  family: spectral  # Use SyntheticSpectralDataset
  channels: 3
  height: 32
  width: 32
  
  synthetic:
    size: 50000           # Dataset size
    image_size: 32        # Image dimensions (must be square)
    channels: 3           # RGB channels
    freq_mix: 0.5         # Frequency balance (0=low, 1=high)
    color_mix: 0.2        # Color correlation (0=independent, 1=grayscale)
    use_text: true        # Include text snippets
    include_gratings: true    # Include spectral gratings
    include_shapes: true      # Include geometric shapes
    log_fft_energy: false     # Enable FFT diagnostics
    seed: 42              # Random seed for reproducibility
```

### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | int | 50000 | Number of samples in the dataset |
| `image_size` | int | 32 | Image dimensions (square images only) |
| `channels` | int | 3 | Number of color channels (must be 3) |
| `freq_mix` | float | 0.5 | Frequency emphasis (0=low, 1=high) |
| `color_mix` | float | 0.2 | Channel correlation (0=independent, 1=correlated) |
| `use_text` | bool | true | Include text layer (requires Pillow) |
| `include_gratings` | bool | true | Include sinusoidal gratings |
| `include_shapes` | bool | true | Include geometric shapes |
| `log_fft_energy` | bool | false | Log radial FFT energy for diagnostics |
| `seed` | int | 0 | Random seed for deterministic generation |

## Usage Examples

### Example 1: Basic Training

```bash
python scripts/debug/record_training_steps.py \
  --config configs/test_synthetic_spectral.yaml \
  --steps 100 \
  --output-dir scratch/synth_test
```

### Example 2: High-Frequency Textures

```yaml
data:
  synthetic:
    freq_mix: 0.9        # Emphasize high frequencies
    color_mix: 0.1       # Keep colors independent
    include_gratings: true
```

### Example 3: Smooth, Grayscale-like Images

```yaml
data:
  synthetic:
    freq_mix: 0.1        # Emphasize low frequencies
    color_mix: 0.9       # High channel correlation
    include_shapes: true
```

### Example 4: Testing Dataset Directly

```python
from src.training.data.synthetic_dataset import SyntheticSpectralDataset

# Create dataset
dataset = SyntheticSpectralDataset(
    size=1000,
    image_size=32,
    freq_mix=0.5,
    color_mix=0.2,
    seed=42,
)

# Get a sample
image, target = dataset[0]  # Returns (image, image) for reconstruction

# Visualize
dataset.show_sample(0)
```

### Example 5: Quick Testing Script

```bash
python scripts/test_synthetic_dataset.py
```

This generates test images with various configurations and saves them to `scratch/synthetic_test/`.

## Diagnostics

The dataset automatically runs diagnostics on initialization:

### Statistical Checks

- **Mean**: Should be in range [0.45, 0.55]
- **Std**: Should be in range [0.15, 0.35]
- **Parseval Energy**: Spatial and frequency domain energy should match (within 10% tolerance)
- **Channel Correlation**: Should roughly match the `color_mix` parameter (within 0.3)

### FFT Energy Logging

When `log_fft_energy: true`, the dataset logs radial FFT energy profiles for analysis:

```python
dataset = SyntheticSpectralDataset(log_fft_energy=True, ...)
# After generating samples, access energy profiles:
energy_samples = dataset._fft_energy_samples
```

## Integration with Training Pipeline

The dataset is automatically integrated via [`builders.py`](../src/training/builders.py):

```python
from src.training.builders import build_dataloader

# Config with data.source: "synthetic" and data.family: "spectral"
dataloader = build_dataloader(config)
```

The dataloader behaves identically to CIFAR-10 dataloaders, making it a drop-in replacement.

## Performance Characteristics

- **Speed**: Fast generation (~1000 samples/sec on CPU)
- **Memory**: Low memory footprint (generates on-the-fly)
- **Determinism**: Fully deterministic given a seed
- **GPU-Safe**: All operations use PyTorch (no NumPy)

## Expected Results

When training with the synthetic dataset, you should observe:

- ✅ Colorful, diverse synthetic textures (not gray monotone)
- ✅ Balanced spectral distributions across frequencies
- ✅ No DC bias or brightness drift
- ✅ Stable training behavior
- ✅ Mean ≈ 0.5, Std ≈ 0.2-0.3

## Troubleshooting

### Issue: Images are too gray/monotone

**Solution**: Decrease `color_mix` to increase color diversity:
```yaml
synthetic:
  color_mix: 0.0  # Fully independent RGB channels
```

### Issue: Images lack fine details

**Solution**: Increase `freq_mix` to emphasize high frequencies:
```yaml
synthetic:
  freq_mix: 0.8  # More high-frequency content
```

### Issue: Diagnostics warnings

**Solution**: Check that:
1. `image_size` is reasonable (16-128 typically)
2. All feature flags are enabled if you want diversity
3. Seed is set for reproducibility

### Issue: Text not appearing

**Solution**: Ensure Pillow is installed:
```bash
pip install Pillow
```

## Implementation Details

### Layer Generation Pipeline

1. **Spectral Noise**: Frequency-shaped Gaussian noise
2. **Fractal Noise**: 1/f^β noise with random β
3. **Shapes Layer**: Random geometric primitives
4. **Grating Layer**: Sinusoidal patterns (75% probability)
5. **Text Layer**: Lorem Ipsum text (40% probability, if Pillow available)
6. **Blob Layer**: Gaussian blobs for organic texture

### Compositing Process

1. Generate all layers independently
2. Combine using random blend modes (additive/multiplicative/overlay)
3. Apply color mixing to control RGB correlation
4. Remove mean to ensure zero DC bias
5. Normalize to [0,1] range

### Zero-Mean Enforcement

Every layer and intermediate result has its mean removed:

```python
layer = layer - layer.mean(dim=(1, 2), keepdim=True)
```

FFT operations neutralize DC:

```python
fft[..., 0, 0] = 0.0
```

This is critical for stable diffusion training.

## References

- Implementation: [`src/training/data/synthetic_dataset.py`](../src/training/data/synthetic_dataset.py)
- Builder integration: [`src/training/builders.py`](../src/training/builders.py)
- Test script: [`scripts/test_synthetic_dataset.py`](../scripts/test_synthetic_dataset.py)
- Example config: [`configs/test_synthetic_spectral.yaml`](../configs/test_synthetic_spectral.yaml)

## Version History

- **v1.0**: Initial implementation with all core features
- Procedural generation with multiple visual primitives
- Spectral and color control parameters
- Zero-mean noise handling
- Comprehensive diagnostics