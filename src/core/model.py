from typing import Any, Dict, Type

import torch
from torch import nn

from .model_unet_tiny import TinyUNet


class BaselineConvModel(nn.Module):
    """
    TEMP baseline: a tiny conv stack that just echoes input shape.
    Replace with real UNet/DiT later.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        channels = int(config.get("data", {}).get("channels", 3))
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass (placeholder)."""
        return self.net(x)


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "baseline": BaselineConvModel,
    "baseline_conv": BaselineConvModel,
    "unet_tiny": TinyUNet,
}


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Factory method to build a diffusion model."""
    model_type = config.get("type", "baseline")
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
    return model_cls(config=config)


# Backwards compatibility alias
BaseDiffusionModel = BaselineConvModel
