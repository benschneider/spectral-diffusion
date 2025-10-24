from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _split_complex(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if tensor.shape[1] % 2 != 0:
        raise ValueError("Complex tensors must have an even channel dimension (real+imag parts).")
    real, imag = torch.chunk(tensor, 2, dim=1)
    return real, imag


def _merge_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    if real.shape != imag.shape:
        raise ValueError("Real and imaginary parts must have matching shapes.")
    return torch.cat([real, imag], dim=1)


class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution implemented via paired real convolutions.

    Expects input tensors with channel layout [B, 2 * C_in, H, W] where the
    first half of channels represent the real component and the second half the imaginary component.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.real_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.imag_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex(x)
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        return _merge_complex(real_out, imag_out)


class ComplexConvTranspose2d(nn.Module):
    """Complex-valued transposed convolution (upsampling) via paired real convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias: bool = True,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.real_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self.imag_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex(x)
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        return _merge_complex(real_out, imag_out)


class ComplexBatchNorm2d(nn.Module):
    """Applies independent BatchNorm to real and imaginary components."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.real_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.imag_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex(x)
        real = self.real_bn(real)
        imag = self.imag_bn(imag)
        return _merge_complex(real, imag)


class ComplexSiLU(nn.Module):
    """Applies SiLU activation on magnitude while preserving phase."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex(x)
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)
        activated = F.silu(magnitude)
        real_out = activated * torch.cos(phase)
        imag_out = activated * torch.sin(phase)
        return _merge_complex(real_out, imag_out)


class ComplexResidualBlock(nn.Module):
    """Simple complex residual block using ComplexConv2d and activation."""

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = ComplexBatchNorm2d(channels)
        self.act = ComplexSiLU()
        self.conv2 = ComplexConv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = ComplexBatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return residual + out
