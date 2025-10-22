import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core import build_model, get_loss_fn


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
        """TEMP: synthetic dataloader so the loop runs end-to-end."""
        bs = int(self.config.get("training", {}).get("batch_size", 32))
        n = int(self.config.get("training", {}).get("num_batches", 50))
        c = int(self.config.get("data", {}).get("channels", 3))
        h = int(self.config.get("data", {}).get("height", 32))
        w = int(self.config.get("data", {}).get("width", 32))
        x = torch.randn(n * bs, c, h, w)
        y = torch.randn_like(x)
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

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

        step = 0
        last_loss: Optional[float] = None
        for epoch in range(epochs):
            for xb, yb in self.loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # --- placeholder forward/loss until real code lands ---
                pred = self.model(xb)  # will raise until you implement forward()
                loss = self.loss_fn(pred, yb)  # will raise until you implement loss
                # ------------------------------------------------------

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                step += 1
                last_loss = float(loss.detach().cpu())
                if step % log_every == 0:
                    self.logger.info("epoch %d step %d loss %.5f", epoch, step, last_loss)

        metrics = {
            "loss": last_loss,
            "fid": None,
            "lpips": None,
            "status": "ok",
        }
        return metrics

    def save_checkpoint(self, step: int) -> Path:
        checkpoint_dir = self.work_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save({"model": self.model.state_dict()}, checkpoint_path)
        return checkpoint_path
