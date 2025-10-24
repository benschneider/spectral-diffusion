from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from src.core.time_embed import TimeMLP, sinusoidal_embedding
from src.spectral import (
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexResidualBlock,
    ComplexSiLU,
)


class SpectralUNet(nn.Module):
    """Minimal spectral-domain UNet using complex-valued convolutions."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        in_channels = int(config.get("channels") or data_cfg.get("channels", 3))
        base_channels = int(config.get("base_channels", 32))

        diffusion_cfg = config.get("diffusion", {})
        time_embed_dim = int(config.get("time_embed_dim") or diffusion_cfg.get("time_embed_dim", 128))
        self.time_embed_dim = time_embed_dim
        self.time_mlp = TimeMLP(time_embed_dim, base_channels)

        self.conv_in = ComplexConv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn_in = ComplexBatchNorm2d(base_channels)
        self.act = ComplexSiLU()
        self.block1 = ComplexResidualBlock(base_channels)
        self.block2 = ComplexResidualBlock(base_channels)
        self.conv_out = ComplexConv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def _apply_time(self, x: torch.Tensor, t_feat: Optional[torch.Tensor]) -> torch.Tensor:
        if t_feat is None:
            return x
        real, imag = torch.chunk(x, 2, dim=1)
        real = real + t_feat
        return torch.cat([real, imag], dim=1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        fft = torch.fft.fft2(x, norm="ortho")
        freq = torch.cat([fft.real, fft.imag], dim=1)

        t_feat: Optional[torch.Tensor] = None
        if t is not None:
            emb = sinusoidal_embedding(t, self.time_embed_dim)
            t_feat = self.time_mlp(emb).unsqueeze(-1).unsqueeze(-1)

        freq = self.conv_in(freq)
        freq = self.bn_in(freq)
        freq = self.act(freq)
        freq = self._apply_time(freq, t_feat)

        freq = self.block1(freq)
        freq = self._apply_time(freq, t_feat)
        freq = self.block2(freq)
        freq = self._apply_time(freq, t_feat)

        out_freq = self.conv_out(freq)
        real, imag = torch.chunk(out_freq, 2, dim=1)
        complex_out = torch.complex(real, imag)
        spatial = torch.fft.ifft2(complex_out, norm="ortho").real
        return spatial
