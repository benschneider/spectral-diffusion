from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import datasets, transforms

from src.data.synthetic import generate_synthetic_samples
from src.training.data.synthetic_dataset import (
    SyntheticSpectralConfig,
    SyntheticSpectralDataset,
)

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
    num_workers = int(data_cfg.get("num_workers", 0))
    defaults = SyntheticSpectralConfig()

    channels = int(data_cfg.get("channels", defaults.channels))
    height = int(data_cfg.get("height", data_cfg.get("image_size", defaults.image_size)))
    width = int(data_cfg.get("width", height))

    synthetic_overrides = data_cfg.get("synthetic", {}) or {}
    if "image_size" in synthetic_overrides:
        height = width = int(synthetic_overrides["image_size"])

    num_batches = int(training_cfg.get("num_batches", 0))
    dataset_base_size = int(data_cfg.get("size", 0))
    if dataset_base_size <= 0:
        dataset_base_size = defaults.size
    num_samples = bs * num_batches if num_batches > 0 else None

    family = str(data_cfg.get("family", "spectral")).lower()
    if family in {"", "spectral"}:
        if height != width:
            raise ValueError(
                "Synthetic spectral datasets require square images; received height="
                f"{height} and width={width}."
            )
        synth_cfg = dict(vars(defaults))
        synth_cfg.update(synthetic_overrides)
        base_size = max(dataset_base_size, bs)
        synth_cfg.update(
            {
                "channels": channels,
                "image_size": height,
                "size": base_size,
            }
        )
        dataset = SyntheticSpectralDataset(**synth_cfg)
        sampler = None
        if num_samples is not None:
            sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
        )
    else:
        dataset_size = int(data_cfg.get("size", 0))
        if dataset_size <= 0 and num_batches > 0:
            dataset_size = max(bs * num_batches, bs)
        if dataset_size <= 0:
            dataset_size = defaults.size
        dataset = _FamilySyntheticDataset(
            length=dataset_size,
            channels=channels,
            height=height,
            width=width,
            data_cfg={**data_cfg},
        )

    return DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers)


class _FamilySyntheticDataset(Dataset):
    """Dataset wrapper that procedurally samples images from a synthetic family."""

    def __init__(
        self,
        *,
        length: int,
        channels: int,
        height: int,
        width: int,
        data_cfg: Dict[str, Any],
    ) -> None:
        self._length = max(int(length), 1)
        self._channels = int(channels)
        self._height = int(height)
        self._width = int(width)
        self._data_cfg = dict(data_cfg)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):  # pragma: no cover - simple wrapper
        sample = generate_synthetic_samples(
            count=1,
            channels=self._channels,
            height=self._height,
            width=self._width,
            data_cfg=self._data_cfg,
        )[0]
        return sample, sample.clone()


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

    def _parse_bool(value, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return default

    download = _parse_bool(data_cfg.get("download"), default=True)
    root = data_cfg.get("root", "data")
    Path(root).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((target_h, target_w)),
            transforms.ToTensor(),
        ]
    )
    try:
        base_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=transform,
        )
    except RuntimeError as exc:
        msg = (
            "CIFAR-10 dataset not found and automatic download disabled. "
            "Either enable download by setting data.download=true or fetch it manually with:\n"
            "  mkdir -p data && curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "
            "-o data/cifar-10-python.tar.gz && tar -xzf data/cifar-10-python.tar.gz -C data\n"
            "Then rerun training or set data.source to 'synthetic'."
        )
        if download:
            msg = (
                "Failed to download CIFAR-10 automatically. "
                "Check network access or download it manually with:\n"
                "  mkdir -p data && curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "
                "-o data/cifar-10-python.tar.gz && tar -xzf data/cifar-10-python.tar.gz -C data\n"
                "Then rerun training or set data.source to 'synthetic'."
            )
        raise RuntimeError(msg) from exc

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
