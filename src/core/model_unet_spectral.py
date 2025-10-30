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


class AmplitudeResidualEncoder(nn.Module):
    """Predict a magnitude-domain residual to refine complex activations."""

    def __init__(self, channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1),
        )

    def forward(self, x_fft: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x_fft)
        return self.net(magnitude)


class PhaseCorrectionModule(nn.Module):
    """Attention-based refiner that predicts phase adjustments per frequency bin."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = max(1, num_heads)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=self.num_heads)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        seq = phase.flatten(2).permute(2, 0, 1)
        refined, _ = self.attn(seq, seq, seq)
        return refined.permute(1, 2, 0).view_as(phase)


def _resolve_attention_heads(channels: int, requested: int) -> int:
    requested = max(1, requested)
    if channels <= 0:
        return 1
    if channels % requested == 0:
        return requested
    for candidate in range(min(channels, requested), 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1


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

        self.enable_amp_residual = bool(config.get("enable_amp_residual", False))
        amp_hidden_dim = max(int(config.get("amp_hidden_dim", base_channels)), 1)
        self.are: Optional[AmplitudeResidualEncoder]
        if self.enable_amp_residual:
            self.are = AmplitudeResidualEncoder(in_channels, amp_hidden_dim)
        else:
            self.are = None

        self.enable_phase_attention = bool(config.get("enable_phase_attention", False))
        phase_heads_cfg = int(config.get("phase_heads", 4))
        resolved_heads = _resolve_attention_heads(in_channels, phase_heads_cfg)
        self.phase_heads: Optional[int] = resolved_heads if self.enable_phase_attention else None
        self.pcm: Optional[PhaseCorrectionModule]
        if self.enable_phase_attention:
            self.pcm = PhaseCorrectionModule(in_channels, num_heads=resolved_heads)
        else:
            self.pcm = None

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
        x_fft = torch.fft.fft2(x, norm="ortho")

        if self.enable_amp_residual and self.are is not None:
            amplitude_residual = self.are(x_fft)
            complex_residual = torch.complex(
                amplitude_residual, torch.zeros_like(amplitude_residual)
            )
            x_fft = x_fft + complex_residual

        if self.enable_phase_attention and self.pcm is not None:
            phase = torch.angle(x_fft)
            phase_correction = self.pcm(phase)
            phase_correction = torch.tanh(phase_correction)
            magnitude = torch.abs(x_fft)
            x_fft = torch.polar(magnitude, phase + phase_correction)

        freq = torch.cat([x_fft.real, x_fft.imag], dim=1)

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
