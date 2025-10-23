import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

from src.core import build_model, get_loss_fn
from src.evaluation.metrics import compute_basic_metrics


class TrainingPipeline:
    """Unified training pipeline for baseline and spectral diffusion models."""

    def __init__(
        self,
        config: Dict[str, Any],
        work_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.work_dir = work_dir
        self.logger = logger or logging.getLogger(__name__)
        self.model = build_model(config.get("model", {}))
        self.loss_fn = get_loss_fn(config.get("loss", {}))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _make_dataloader(self) -> DataLoader:
        """Build a dataloader based on configuration."""
        data_cfg = self.config.get("data", {})
        source = str(data_cfg.get("source", "synthetic")).lower()
        if source == "synthetic":
            return self._make_synthetic_dataloader(data_cfg=data_cfg)
        if source == "cifar10":
            return self._make_cifar10_dataloader(data_cfg=data_cfg)
        raise ValueError(f"Unsupported data source: {source}")

    def _make_synthetic_dataloader(self, data_cfg: Dict[str, Any]) -> DataLoader:
        bs = int(self.config.get("training", {}).get("batch_size", 32))
        n_setting = self.config.get("training", {}).get("num_batches", 50)
        n = int(n_setting) if str(n_setting).isdigit() else 50
        c = int(data_cfg.get("channels", 3))
        h = int(data_cfg.get("height", 32))
        w = int(data_cfg.get("width", 32))
        x = torch.randn(n * bs, c, h, w)
        y = torch.randn_like(x)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    def _make_cifar10_dataloader(self, data_cfg: Dict[str, Any]) -> DataLoader:
        bs = int(self.config.get("training", {}).get("batch_size", 32))
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

        class ReconstructionWrapper(Dataset):
            def __init__(self, dataset) -> None:
                self.dataset = dataset

            def __len__(self) -> int:
                return len(self.dataset)

            def __getitem__(self, idx: int):
                img, _ = self.dataset[idx]
                return img, img

        return DataLoader(
            ReconstructionWrapper(base_dataset),
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )

    def _make_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.config.get("optim", {}).get("lr", 1e-4))
        wd = float(self.config.get("optim", {}).get("weight_decay", 0.0))
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def setup(self) -> None:
        """Prepare datasets, optimizers, and other resources."""
        self.logger.debug("Setting up training pipeline with config: %s", self.config)
        self.loader = self._make_dataloader()
        self.optimizer = self._make_optimizer()

    def run(self) -> Dict[str, Any]:
        """Execute the (placeholder) training loop and return metrics."""
        self.setup()
        self.model.train()
        epochs = int(self.config.get("training", {}).get("epochs", 1))
        log_every = int(self.config.get("training", {}).get("log_every", 10))
        max_batches = self.config.get("training", {}).get("num_batches")
        batch_limit = int(max_batches) if str(max_batches).isdigit() else None

        step = 0
        loss_history, mae_history = [], []
        wall_start = perf_counter()
        for epoch in range(epochs):
            for batch_idx, (xb, yb) in enumerate(self.loader):
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                mae = F.l1_loss(pred, yb)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                step += 1
                loss_val = float(loss.detach().cpu())
                loss_history.append(loss_val)
                mae_history.append(float(mae.detach().cpu()))
                if step % log_every == 0:
                    self.logger.info("epoch %d step %d loss %.5f", epoch, step, loss_val)
                if batch_limit is not None and (batch_idx + 1) >= batch_limit:
                    break

        runtime_seconds = perf_counter() - wall_start

        metrics = compute_basic_metrics(
            loss_history=loss_history,
            mae_history=mae_history,
            runtime_seconds=runtime_seconds,
            extra={"status": "ok", "num_steps": step, "epochs": epochs},
        )
        self.logger.info("Training metrics: %s", metrics)
        return metrics

    def save_checkpoint(self, step: int) -> Path:
        checkpoint_dir = self.work_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save({"model": self.model.state_dict()}, checkpoint_path)
        return checkpoint_path
