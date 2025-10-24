from typing import Any, Dict, Optional, Type

import torch
from torch import nn

from .model_unet_tiny import TinyUNet


class BaselineConvModel(nn.Module):
    """
    TEMP baseline: tiny conv stack that preserves input shape (for reconstruction).
    Replace with real diffusion backbone later.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        data_cfg = config.get("data", {})
        channels = int(config.get("channels") or data_cfg.get("channels", 3))
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x)

    def reset_spectral_stats(self) -> None:  # pragma: no cover - baseline has no spectral adapters
        return


# Backwards compatibility alias
BaseDiffusionModel = BaselineConvModel

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "baseline": BaselineConvModel,
    "baseline_conv": BaselineConvModel,
    "unet_tiny": TinyUNet,
}


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Factory method to build a model from a minimal config."""
    model_type = config.get("type", "baseline")
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return model_cls(config=config)
