from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.core.time_embed import TimeMLP, sinusoidal_embedding


class ConvBlock(nn.Module):
    """Basic convolutional block used throughout the tiny UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_width: int = 0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.time_bias = nn.Linear(time_width, out_channels) if time_width > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        t_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.conv1(x)
        if self.time_bias is not None and t_feat is not None:
            h = h + self.time_bias(t_feat).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm1(h))
        h = self.conv2(h)
        h = F.silu(self.norm2(h))
        return h


class TinyUNet(nn.Module):
    """Lightweight UNet-style architecture suitable for 32×32 inputs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        in_channels = int(config.get("channels") or data_cfg.get("channels", 3))
        base_channels = int(config.get("base_channels", 32))
        depth = max(int(config.get("depth", 2)), 1)
        time_width = base_channels
        diffusion_cfg = config.get("diffusion", {})
        time_embed_dim = config.get("time_embed_dim")
        if time_embed_dim is None:
            time_embed_dim = diffusion_cfg.get("time_embed_dim", 128)
        self.time_embed_dim = int(time_embed_dim)
        self.time_mlp = TimeMLP(self.time_embed_dim, time_width)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_channels = in_channels
        for level in range(depth):
            out_channels = base_channels * (2**level)
            self.encoder_blocks.append(
                ConvBlock(current_channels, out_channels, time_width=time_width)
            )
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
            decoder_blocks.append(
                ConvBlock(out_channels * 2, out_channels, time_width=time_width)
            )
            decoder_channels = out_channels

        self.upsamples = nn.ModuleList(upsamplers)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.head = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self._spectral_calls = 0.0

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        return out

    def spectral_stats(self) -> Dict[str, float]:
        return {
            "spectral_calls": float(self._spectral_calls),
            "spectral_time_seconds": 0.0,
            "spectral_cpu_time_seconds": 0.0,
            "spectral_cuda_time_seconds": 0.0,
        }

    def reset_spectral_stats(self) -> None:
        self._spectral_calls = 0.0
