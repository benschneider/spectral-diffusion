import torch

from src.spectral.complex_layers import (
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexConvTranspose2d,
    ComplexResidualBlock,
    ComplexSiLU,
)


def _complex_tensor(channels: int, height: int, width: int, batch: int = 2) -> torch.Tensor:
    real = torch.randn(batch, channels, height, width)
    imag = torch.randn(batch, channels, height, width)
    return torch.cat([real, imag], dim=1)


def test_complex_conv2d_shapes_and_grad():
    x = _complex_tensor(channels=3, height=8, width=8, batch=4).requires_grad_(True)
    layer = ComplexConv2d(3, 5, kernel_size=3, padding=1)
    out = layer(x)
    assert out.shape == (4, 10, 8, 8)
    out.norm().backward()
    assert x.grad is not None


def test_complex_batchnorm_runs():
    layer = ComplexBatchNorm2d(4)
    x = _complex_tensor(channels=4, height=6, width=6)
    out = layer(x)
    assert out.shape == x.shape


def test_complex_silu_preserves_shape():
    act = ComplexSiLU()
    x = _complex_tensor(channels=2, height=5, width=5)
    out = act(x)
    assert out.shape == x.shape


def test_complex_residual_block():
    block = ComplexResidualBlock(channels=3)
    x = _complex_tensor(channels=3, height=8, width=8).requires_grad_(True)
    out = block(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_complex_conv_transpose():
    layer = ComplexConvTranspose2d(3, 2, kernel_size=2, stride=2)
    x = _complex_tensor(channels=3, height=8, width=8)
    out = layer(x)
    assert out.shape == (x.shape[0], 4, 16, 16)
