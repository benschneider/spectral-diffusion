from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data.synthetic import generate_synthetic_samples
from src.training.optimizers import Adafactor, Lion
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
    defaults = SyntheticSpectralConfig()

    channels = int(data_cfg.get("channels", defaults.channels))
    height = int(data_cfg.get("height", data_cfg.get("image_size", defaults.image_size)))
    width = int(data_cfg.get("width", height))

    family = str(data_cfg.get("family", "spectral")).lower()
    if family in {"", "spectral"}:
        if height != width:
            raise ValueError(
                "Synthetic spectral datasets require square images; received height="
                f"{height} and width={width}."
            )
        dataset = SyntheticSpectralDataset(
            size=defaults.size,
            image_size=height,
            channels=channels,
            freq_mix=defaults.freq_mix,
            color_mix=defaults.color_mix,
            use_text=defaults.use_text,
            include_gratings=defaults.include_gratings,
            include_shapes=defaults.include_shapes,
            log_fft_energy=defaults.log_fft_energy,
            seed=defaults.seed,
        )
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
    else:
        dataset = _FamilySyntheticDataset(
            length=defaults.size,
            channels=channels,
            height=height,
            width=width,
            data_cfg={**data_cfg},
        )

    return DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)


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
    download = bool(data_cfg.get("download", True))
    root = str(data_cfg.get("root", "data"))
    Path(root).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((target_h, target_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
        manual = (
            f"  mkdir -p {root} && curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "
            f"-o {root}/cifar-10-python.tar.gz && tar -xzf {root}/cifar-10-python.tar.gz -C {root}\n"
        )
        if download:
            msg = (
                "Failed to download CIFAR-10 automatically. "
                "Check network access or download it manually with:\n"
                f"{manual}"
                "Then rerun training or set data.source to 'synthetic'."
            )
        else:
            msg = (
                "CIFAR-10 dataset not found and automatic download disabled. "
                "Either enable download by setting data.download=true or fetch it manually with:\n"
                f"{manual}"
                "Then rerun training or set data.source to 'synthetic'."
            )
        raise RuntimeError(msg) from exc

    return DataLoader(
        _ReconstructionWrapper(base_dataset),
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Construct the optimizer for training."""
    optim_cfg = config.get("optim", {}) or {}
    optim_type = str(optim_cfg.get("type", "adamw")).lower()

    def _as_bool(value, default):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return default

    def _as_beta_pair(values, default):
        if values is None:
            return default
        if isinstance(values, (list, tuple)) and len(values) == 2:
            return float(values[0]), float(values[1])
        raise ValueError(f"Expected a pair of betas, received {values!r}")

    lr_value = optim_cfg.get("lr")
    lr = float(lr_value) if lr_value is not None else 1e-4
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if optim_type == "adamw":
        betas = _as_beta_pair(optim_cfg.get("betas"), None)
        if betas is not None:
            return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optim_type == "lion":
        betas = _as_beta_pair(optim_cfg.get("betas", (0.9, 0.99)), (0.9, 0.99))
        return Lion(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    if optim_type == "adafactor":
        relative_step = _as_bool(optim_cfg.get("relative_step"), False)
        scale_parameter = _as_bool(optim_cfg.get("scale_parameter"), False if lr_value is not None else True)
        warmup_init = _as_bool(optim_cfg.get("warmup_init"), False)
        clip_threshold = float(optim_cfg.get("clip_threshold", 1.0))
        decay_rate = float(optim_cfg.get("decay_rate", -0.8))
        beta1 = optim_cfg.get("beta1")
        if beta1 is not None:
            beta1 = float(beta1)
        eps_values = optim_cfg.get("eps", (1e-30, 1e-3))
        if isinstance(eps_values, (list, tuple)) and len(eps_values) == 2:
            eps_tuple = float(eps_values[0]), float(eps_values[1])
        else:
            eps_tuple = (1e-30, 1e-3)
        lr_arg: Optional[float] = None if relative_step else float(lr)
        return Adafactor(
            model.parameters(),
            lr=lr_arg,
            eps=eps_tuple,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )

    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
