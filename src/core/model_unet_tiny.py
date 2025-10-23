import time
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.core.time_embed import TimeMLP, sinusoidal_embedding
from src.spectral.fft_utils import (
    apply_weight_map,
    configure_spectral_params,
    fft_transform,
    inverse_fft_transform,
)


class ConvBlock(nn.Module):
    """Basic convolutional block used throughout the tiny UNet."""

    def __init__(self, in_channels: int, out_channels: int, time_width: int = 0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.time_bias = nn.Linear(time_width, out_channels) if time_width > 0 else None

    def forward(self, x: torch.Tensor, t_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(x)
        if self.time_bias is not None and t_feat is not None:
            h = h + self.time_bias(t_feat).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm1(h))
        h = self.conv2(h)
        h = F.silu(self.norm2(h))
        return h


def _build_weight_map(
    spectral_config: Dict[str, Any],
    height: int,
    width: int,
) -> Optional[torch.Tensor]:
    weighting = spectral_config.get("weighting", "none")
    if weighting == "none":
        return None

    fy = torch.fft.fftfreq(height, d=1.0)
    fx = torch.fft.fftfreq(width, d=1.0)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_x**2 + grid_y**2)

    if weighting == "radial":
        weight = 1.0 / (1.0 + radius**2)
    elif weighting == "bandpass":
        inner = spectral_config.get("bandpass_inner", 0.05)
        outer = spectral_config.get("bandpass_outer", 0.25)
        weight = ((radius >= inner) & (radius <= outer)).float()
    else:
        raise ValueError(f"Unknown spectral weighting '{weighting}'")

    return weight.to(torch.float32)


class TinyUNet(nn.Module):
    """Lightweight UNet-style architecture suitable for 32×32 inputs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        in_channels = int(config.get("channels") or data_cfg.get("channels", 3))
        base_channels = int(config.get("base_channels", 32))
        depth = max(int(config.get("depth", 2)), 1)
        time_width = base_channels
        self.time_embed_dim = config.get("time_embed_dim", 128)
        self.time_mlp = TimeMLP(self.time_embed_dim, time_width)

        self.spectral_cfg = configure_spectral_params(config)
        self.weight_map = None
        if self.spectral_cfg.get("enabled", False):
            height = int(data_cfg.get("height", 32))
            width = int(data_cfg.get("width", 32))
            weight = _build_weight_map(config.get("spectral", {}), height, width)
            if weight is not None:
                self.register_buffer("spectral_weight_map", weight)
                self.weight_map = self.spectral_weight_map
        self._spectral_calls: int = 0
        self._spectral_time: float = 0.0

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_channels = in_channels
        for level in range(depth):
            out_channels = base_channels * (2**level)
            self.encoder_blocks.append(ConvBlock(current_channels, out_channels, time_width=time_width))
            if level < depth - 1:
                self.downsamples.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                )
            current_channels = out_channels

        bottleneck_channels = current_channels * 2
        self.bottleneck = ConvBlock(current_channels, bottleneck_channels, time_width=time_width)

        decoder_blocks: List[nn.Module] = []
        upsamplers: List[nn.Module] = []
        decoder_channels = bottleneck_channels
        for level in reversed(range(depth)):
            out_channels = base_channels * (2**level)
            upsamplers.append(
                nn.ConvTranspose2d(decoder_channels, out_channels, kernel_size=2, stride=2)
            )
            decoder_blocks.append(ConvBlock(out_channels * 2, out_channels, time_width=time_width))
            decoder_channels = out_channels

        self.upsamples = nn.ModuleList(upsamplers)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.head = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def _apply_spectral_roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        normalize = self.spectral_cfg.get("normalize", True)
        x_fft = fft_transform(x, normalize=normalize)
        if self.weight_map is not None:
            x_fft = apply_weight_map(x_fft, self.weight_map)
        x = inverse_fft_transform(x_fft, normalize=normalize)
        self._spectral_calls += 1
        self._spectral_time += time.perf_counter() - start
        return x

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.spectral_cfg.get("enabled", False):
            x = self._apply_spectral_roundtrip(x)

        t_feat = None
        if t is not None:
            t_feat = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))

        skips: List[torch.Tensor] = []
        h = x
        for idx, block in enumerate(self.encoder_blocks):
            h = block(h, t_feat)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        h = self.bottleneck(h, t_feat)

        for idx, (upsample, block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            h = upsample(h)
            skip = skips[-(idx + 1)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_feat)

        out = self.head(h)
        if self.spectral_cfg.get("enabled", False):
            # Optional: reproject residual back through inverse FFT to enforce consistency.
            out = self._apply_spectral_roundtrip(out)
        return out

    def spectral_stats(self) -> Dict[str, float]:
        """Return accumulated spectral instrumentation metrics."""
        return {
            "spectral_calls": float(self._spectral_calls),
            "spectral_time_seconds": float(self._spectral_time),
        }
