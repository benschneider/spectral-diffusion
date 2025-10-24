import time
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.core.time_embed import TimeMLP, sinusoidal_embedding
from src.spectral.adapter import SpectralAdapter
from src.spectral.fft_utils import configure_spectral_params


class ConvBlock(nn.Module):
    """Basic convolutional block used throughout the tiny UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_width: int = 0,
        spectral_adapter: Optional[SpectralAdapter] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.time_bias = nn.Linear(time_width, out_channels) if time_width > 0 else None
        self.spectral_adapter = spectral_adapter

    def forward(
        self,
        x: torch.Tensor,
        t_feat: Optional[torch.Tensor] = None,
        block_adapter: Optional[SpectralAdapter] = None,
    ) -> torch.Tensor:
        adapter = block_adapter or self.spectral_adapter
        if adapter is not None:
            x = adapter(x, t_feat)
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
        diffusion_cfg = config.get("diffusion", {})
        time_embed_dim = config.get("time_embed_dim")
        if time_embed_dim is None:
            time_embed_dim = diffusion_cfg.get("time_embed_dim", 128)
        self.time_embed_dim = int(time_embed_dim)
        self.time_mlp = TimeMLP(self.time_embed_dim, time_width)

        self.spectral_cfg = configure_spectral_params(config)
        params = self.spectral_cfg
        enabled = params.get("enabled", False)
        weighting = params.get("weighting", "none")
        normalize = params.get("normalize", True)
        inner = params.get("bandpass_inner", 0.1)
        outer = params.get("bandpass_outer", 0.6)
        apply_to = set(params.get("apply_to", []))
        learnable = params.get("learnable", False)
        condition_mode = params.get("condition", "none") if learnable else "none"
        mlp_hidden = params.get("mlp_hidden_dim", 64)
        learn_temp = params.get("learnable_temperature", 50.0)
        learn_gain_init = params.get("learnable_gain_init", 1.0)
        condition_dim = time_width if (learnable and condition_mode == "time") else 0

        self.spectral_input: Optional[SpectralAdapter] = None
        self.spectral_output: Optional[SpectralAdapter] = None
        self.spectral_block: Optional[SpectralAdapter] = None

        if enabled:
            if "input" in apply_to:
                self.spectral_input = SpectralAdapter(
                    True,
                    weighting,
                    normalize,
                    inner,
                    outer,
                    learnable=False,
                )
            if "output" in apply_to:
                self.spectral_output = SpectralAdapter(
                    True,
                    weighting,
                    normalize,
                    inner,
                    outer,
                    learnable=learnable,
                    condition_dim=condition_dim,
                    mlp_hidden_dim=mlp_hidden,
                    learnable_temperature=learn_temp,
                    learnable_gain_init=learn_gain_init,
                )
            if params.get("per_block", False):
                self.spectral_block = SpectralAdapter(
                    True,
                    weighting,
                    normalize,
                    inner,
                    outer,
                    learnable=learnable,
                    condition_dim=condition_dim,
                    mlp_hidden_dim=mlp_hidden,
                    learnable_temperature=learn_temp,
                    learnable_gain_init=learn_gain_init,
                )

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

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.spectral_input is not None:
            x = self.spectral_input(x, None)

        t_feat = None
        if t is not None:
            t_feat = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))

        skips: List[torch.Tensor] = []
        h = x
        for idx, block in enumerate(self.encoder_blocks):
            h = block(h, t_feat, block_adapter=self.spectral_block)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        h = self.bottleneck(h, t_feat, block_adapter=self.spectral_block)

        for idx, (upsample, block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            h = upsample(h)
            skip = skips[-(idx + 1)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_feat, block_adapter=self.spectral_block)

        out = self.head(h)
        if self.spectral_output is not None:
            out = self.spectral_output(out, t_feat)
        return out

    def spectral_stats(self) -> Dict[str, float]:
        stats = {"spectral_calls": 0.0, "spectral_time_seconds": 0.0}
        stats_cpu = 0.0
        stats_cuda = 0.0
        for adapter in (self.spectral_input, self.spectral_output, self.spectral_block):
            if adapter is not None:
                data = adapter.stats()
                stats["spectral_calls"] += data["spectral_calls"]
                stats["spectral_time_seconds"] += data["spectral_time_seconds"]
                stats_cpu += data.get("spectral_cpu_time_seconds", 0.0)
                stats_cuda += data.get("spectral_cuda_time_seconds", 0.0)
        stats["spectral_cpu_time_seconds"] = stats_cpu
        stats["spectral_cuda_time_seconds"] = stats_cuda
        return stats

    def reset_spectral_stats(self) -> None:
        for adapter in (self.spectral_input, self.spectral_output, self.spectral_block):
            if adapter is not None:
                adapter.reset_stats()
