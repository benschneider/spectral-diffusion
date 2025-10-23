from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block used throughout the tiny UNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """Lightweight UNet-style architecture suitable for 32x32 inputs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        in_channels = int(config.get("channels") or data_cfg.get("channels", 3))
        base_channels = int(config.get("base_channels", 32))
        depth = int(config.get("depth", 2))
        depth = max(depth, 1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_channels = in_channels
        for level in range(depth):
            out_channels = base_channels * (2 ** level)
            self.encoder_blocks.append(ConvBlock(current_channels, out_channels))
            if level < depth - 1:
                self.downsamples.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                )
            current_channels = out_channels

        bottleneck_channels = current_channels * 2
        self.bottleneck = ConvBlock(current_channels, bottleneck_channels)

        decoder_blocks: List[nn.Module] = []
        upsamplers: List[nn.Module] = []
        decoder_channels = bottleneck_channels
        for level in reversed(range(depth)):
            out_channels = base_channels * (2 ** level)
            upsamplers.append(
                nn.ConvTranspose2d(decoder_channels, out_channels, kernel_size=2, stride=2)
            )
            decoder_blocks.append(ConvBlock(out_channels * 2, out_channels))
            decoder_channels = out_channels

        self.upsamples = nn.ModuleList(upsamplers)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.head = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for idx, block in enumerate(self.encoder_blocks):
            h = block(h)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        h = self.bottleneck(h)

        for idx, (upsample, block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            h = upsample(h)
            skip = skips[-(idx + 1)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        return self.head(h)
