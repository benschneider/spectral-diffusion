from typing import Any, Dict

import torch
from torch import nn


class BaseDiffusionModel(nn.Module):
    """
    TEMP baseline: a tiny conv stack that just echoes input shape.
    Replace with real UNet/DiT later.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.get("type", "baseline")
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

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute a single training step."""
        x, y = batch["x"], batch["y"]
        y_hat = self.forward(x)
        return {"pred": y_hat, "target": y}


def build_model(config: Dict[str, Any]) -> BaseDiffusionModel:
    """Factory method to build a diffusion model."""
    return BaseDiffusionModel(config=config)
