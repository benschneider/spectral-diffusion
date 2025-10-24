from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms


def build_dataloader(config: Dict[str, Any]) -> DataLoader:
    """Construct a training dataloader based on configuration."""
    data_cfg = config.get("data", {}) or {}
    training_cfg = config.get("training", {}) or {}
    source = str(data_cfg.get("source", "synthetic")).lower()
    if source == "synthetic":
        return _build_synthetic_dataloader(data_cfg=data_cfg, training_cfg=training_cfg)
    if source == "cifar10":
        return _build_cifar10_dataloader(data_cfg=data_cfg, training_cfg=training_cfg)
    raise ValueError(f"Unsupported data source: {source}")


def _build_synthetic_dataloader(
    data_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
) -> DataLoader:
    bs = int(training_cfg.get("batch_size", 32))
    n_setting = training_cfg.get("num_batches", 50)
    n = int(n_setting) if str(n_setting).isdigit() else 50
    c = int(data_cfg.get("channels", 3))
    h = int(data_cfg.get("height", 32))
    w = int(data_cfg.get("width", 32))
    x = torch.randn(n * bs, c, h, w)
    y = torch.randn_like(x)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)


class _ReconstructionWrapper(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]
        return img, img


def _build_cifar10_dataloader(
    data_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
) -> DataLoader:
    bs = int(training_cfg.get("batch_size", 32))
    target_h = int(data_cfg.get("height", 32))
    target_w = int(data_cfg.get("width", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    download = bool(data_cfg.get("download", False))

    transform = transforms.Compose(
        [
            transforms.Resize((target_h, target_w)),
            transforms.ToTensor(),
        ]
    )
    try:
        base_dataset = datasets.CIFAR10(
            root=data_cfg.get("root", "data"),
            train=True,
            download=download,
            transform=transform,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "CIFAR-10 dataset not found. Download it manually with:\n"
            "  mkdir -p data && curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "
            "-o data/cifar-10-python.tar.gz && tar -xzf data/cifar-10-python.tar.gz -C data\n"
            "Then rerun training or set data.source to 'synthetic'."
        ) from exc

    return DataLoader(
        _ReconstructionWrapper(base_dataset),
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Construct the optimizer for training."""
    optim_cfg = config.get("optim", {}) or {}
    lr = float(optim_cfg.get("lr", 1e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
