from __future__ import annotations

import pytest
import torch

from src.core import build_model
from src.training.builders import build_dataloader


@pytest.fixture(scope="session")
def cifar_batch() -> torch.Tensor:
    """
    Small CIFAR batch for sanity checks.
    Skips tests if the CIFAR dataset is not available locally.
    """
    config = {
        "data": {
            "source": "cifar10",
            "root": "data",
            "height": 32,
            "width": 32,
            "channels": 3,
            "download": False,
        },
        "training": {
            "batch_size": 4,
            "num_batches": 1,
        },
    }
    try:
        loader = build_dataloader(config)
    except RuntimeError as exc:  # dataset not present
        pytest.skip(f"CIFAR-10 dataset not available: {exc}")
    batch, _ = next(iter(loader))
    return batch.to(torch.float32)


@pytest.fixture()
def spectral_model() -> torch.nn.Module:
    """Construct a minimal spectral UNet for quick unit checks."""
    model_cfg = {
        "type": "unet_spectral",
        "channels": 3,
        "base_channels": 16,
        "amp_hidden_dim": 16,
        "enable_amp_residual": False,
        "enable_phase_attention": True,
        "phase_heads": 1,
        "diffusion": {"time_embed_dim": 64},
        "data": {"channels": 3},
    }
    model = build_model(model_cfg)
    return model
