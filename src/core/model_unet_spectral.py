from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from src.core.time_embed import TimeMLP, sinusoidal_embedding
from src.spectral import (
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexConvTranspose2d,
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


class SpectralUNetDeep(nn.Module):
    """Hierarchical spectral UNet with complex down/upsampling and skip connections."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        in_channels = int(config.get("channels") or data_cfg.get("channels", 3))
        base_channels = int(config.get("base_channels", 32))
        depth = max(int(config.get("depth", 3)), 1)
        self.depth = depth

        diffusion_cfg = config.get("diffusion", {})
        time_embed_dim = int(config.get("time_embed_dim") or diffusion_cfg.get("time_embed_dim", 128))
        self.time_embed_dim = time_embed_dim
        self.max_time_channels = base_channels * (2 ** max(depth - 1, 0))
        self.time_mlp = TimeMLP(time_embed_dim, self.max_time_channels)

        self.normalize = True

        self.conv_in = ComplexConv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn_in = ComplexBatchNorm2d(base_channels)
        self.act = ComplexSiLU()

        encoder_blocks: List[nn.Module] = []
        downsamples: List[nn.Module] = []
        channels = base_channels
        for level in range(depth):
            encoder_blocks.append(ComplexResidualBlock(channels))
            if level < depth - 1:
                next_channels = channels * 2
                downsamples.append(
                    ComplexConv2d(channels, next_channels, kernel_size=3, stride=2, padding=1)
                )
                channels = next_channels

        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.downsamples = nn.ModuleList(downsamples)
        self.bottleneck = ComplexResidualBlock(channels)

        upsamples: List[nn.Module] = []
        decoder_blocks: List[nn.Module] = []
        for level in reversed(range(depth - 1)):
            out_channels = base_channels * (2 ** level)
            upsamples.append(
                ComplexConvTranspose2d(channels, out_channels, kernel_size=2, stride=2)
            )
            decoder_blocks.append(
                nn.Sequential(
                    ComplexConv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                    ComplexBatchNorm2d(out_channels),
                    ComplexSiLU(),
                    ComplexResidualBlock(out_channels),
                )
            )
            channels = out_channels

        self.upsamples = nn.ModuleList(upsamples)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.conv_out = ComplexConv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def _apply_time(self, x: torch.Tensor, t_feat: Optional[torch.Tensor]) -> torch.Tensor:
        if t_feat is None:
            return x
        real, imag = torch.chunk(x, 2, dim=1)
        real = real + t_feat[:, : real.shape[1], ...]
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

        skips: List[torch.Tensor] = []
        for level, block in enumerate(self.encoder_blocks):
            freq = block(freq)
            freq = self._apply_time(freq, t_feat)
            if level < self.depth - 1:
                skips.append(freq)
                freq = self.downsamples[level](freq)
                freq = self._apply_time(freq, t_feat)

        freq = self.bottleneck(freq)
        freq = self._apply_time(freq, t_feat)

        for up, block, skip in zip(self.upsamples, self.decoder_blocks, reversed(skips)):
            freq = up(freq)
            freq = self._apply_time(freq, t_feat)
            freq = torch.cat([freq, skip], dim=1)
            freq = block(freq)
            freq = self._apply_time(freq, t_feat)

        out_freq = self.conv_out(freq)
        real, imag = torch.chunk(out_freq, 2, dim=1)
        complex_out = torch.complex(real, imag)
        spatial = torch.fft.ifft2(complex_out, norm="ortho").real
        return spatial
