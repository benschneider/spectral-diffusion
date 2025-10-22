from typing import Any, Dict

import torch
from torch import nn


class DiffusionLoss(nn.Module):
    """
    TEMP: L2 reconstruction loss as a placeholder.
    Replace with epsilon-prediction / v-prediction losses later.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss value given predictions and targets."""
        return self.mse(predictions, targets)


def get_loss_fn(config: Dict[str, Any]) -> DiffusionLoss:
    """Retrieve the loss function configured for the current experiment."""
    return DiffusionLoss(config=config)
